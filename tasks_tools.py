"""
tasks_tools.py

Everything this app can do with Google Tasks: create, list, search by
name, update (rename/reschedule), mark complete, and delete tasks. Same
shape as calendar_tools.py — every function goes through
retry.retry_blocking() since Google's API client is synchronous, and
this file has zero awareness of "pending actions" or confirmation; that
orchestration lives in agent.py.

One real difference from calendar_tools.py: a task's due date is just a
DATE, not a specific moment in time — see _task_body()'s comment on why
that means NOT using time_utils.to_google_timestamp() here.
"""

from dataclasses import dataclass
from datetime import date, datetime

from googleapiclient.discovery import build

import google_auth
import text_match
from retry import retry_blocking


@dataclass
class TaskInfo:
    """
    This app's own plain representation of one task — not Google's raw
    API dict.

    Fields:
        id (str): Google's task ID. Needed for update_task()/
        complete_task()/delete_task() later — never spoken aloud.
        title (str): the task's name.
        due_date (date | None): when it's due. None if no due date was
        set — tasks don't need one the way events need a time.
        notes (str): free-text notes. Empty string if none.
        completed (bool): whether it's already been marked done.
    """
    id: str
    title: str
    due_date: date | None = None
    notes: str = ""
    completed: bool = False


# Cached after the first lookup so every task operation doesn't need an
# extra API round trip just to find out which task list to use. Reset to
# None (a fresh lookup happens automatically) on every server restart —
# that's fine, it's a trivial one-time cost per run, not something that
# needs to survive on disk.
_cached_tasklist_id = None


def _get_service():
    """
    Build a fresh Google Tasks API client using this app's saved
    credentials.

    Takes: nothing.

    Returns: a googleapiclient Resource object for the Tasks v1 API.

    Can this fail: yes — whatever google_auth.get_credentials() can
    raise passes straight through. Called fresh inside every function
    below, not cached, so a refreshed token is always picked up
    automatically.
    """
    creds = google_auth.get_credentials()
    return build("tasks", "v1", credentials=creds)


def _get_tasklist_id(service) -> str:
    """
    Get the ID of the default task list, caching it after the first
    successful lookup.

    Takes:
        service: a Google Tasks API client (from _get_service()).

    Returns:
        str: the default task list's ID, or "@default" if the account
        somehow has no task lists at all.

    Can this fail: yes — if the lookup itself fails, the exception
    passes straight through uncaught rather than being caught here.
    Whatever called this is already running inside a
    retry_blocking()-wrapped operation, so the whole operation —
    including this lookup — gets retried together as one unit rather
    than this needing its own separate retry. Because the cache is only
    set AFTER a successful fetch, a failed lookup never leaves a bad
    value cached — the next call just tries a fresh lookup again.
    """
    global _cached_tasklist_id
    if _cached_tasklist_id is None:
        result = service.tasklists().list().execute()
        items = result.get("items", [])
        _cached_tasklist_id = items[0]["id"] if items else "@default"
    return _cached_tasklist_id


def _parse_task(raw: dict) -> TaskInfo:
    """
    Turn one task resource from Google's API into this app's own
    TaskInfo shape.

    Takes:
        raw (dict): one item from Google Tasks' list()/insert()/patch()
        response.

    Returns:
        TaskInfo.

    Can this fail: no — defensively leaves due_date as None if the "due"
    field is missing or in a format that can't be parsed, rather than
    crashing on a task this app didn't create itself.
    """
    due_date = None
    due_raw = raw.get("due")
    if due_raw:
        try:
            # Google returns due dates as full timestamps (always
            # midnight UTC, since Tasks only has date granularity, never
            # a real time of day) — take just the date part.
            due_date = datetime.fromisoformat(due_raw.replace("Z", "+00:00")).date()
        except ValueError:
            due_date = None

    return TaskInfo(
        id=raw["id"],
        title=raw.get("title", "Untitled"),
        due_date=due_date,
        notes=raw.get("notes", ""),
        completed=(raw.get("status") == "completed"),
    )


def _task_body(title: str = None, due_date: date = None, notes: str = None, clear_due_date: bool = False) -> dict:
    """
    Build the request body Google's API expects for creating or updating
    a task, including only the fields that were actually given.

    Takes:
        title, notes: whichever fields should be set. None means "don't
        include this field" — this is what lets update_task() change
        just one field without touching the rest.
        due_date (date | None): a due date to set. Ignored if
        clear_due_date is True.
        clear_due_date (bool): True to explicitly remove an existing due
        date. Needed because Google represents "no due date" as the
        field being entirely absent — clearing one has to be requested
        on purpose, it's not something that just happens by leaving
        due_date as None (None already means "don't touch this field"
        for everything else in this function).

    Returns:
        dict: ready to hand to service.tasks().insert()/patch() as the
        `body` argument.

    Can this fail: no.

    Why this does NOT use time_utils.to_google_timestamp(): that
    function produces a real PKT-offset timestamp (e.g. "...+05:00"),
    which is correct for Calendar events that represent an actual moment
    in time. A task's due date isn't a moment in time — it's just a
    date, and Google's API always expects it written as literal UTC
    midnight ("...T00:00:00.000Z"), the same way the old codebase's
    proven-working version did it. Using a real PKT offset here instead
    of literal "Z" would risk Google normalizing the timestamp and
    reading back a DIFFERENT calendar date than the one intended —
    exactly the kind of timezone bug this whole rebuild exists to avoid,
    just approached from the opposite direction (being "more correct"
    about timezone would actually introduce the bug here, not prevent
    one).
    """
    body = {}
    if title is not None:
        body["title"] = title
    if notes is not None:
        body["notes"] = notes
    if clear_due_date:
        body["due"] = None
    elif due_date is not None:
        body["due"] = f"{due_date.isoformat()}T00:00:00.000Z"
    return body


async def create_task(title: str, due_date: date = None, notes: str = "") -> TaskInfo:
    """
    Create a new task.

    Takes:
        title (str): the task's name.
        due_date (date | None): when it's due. None for no due date.
        notes (str): optional free-text notes.

    Returns:
        TaskInfo: the task as Google actually saved it, including its
        new id.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_create():
        service = _get_service()
        tasklist_id = _get_tasklist_id(service)
        body = _task_body(title=title, due_date=due_date, notes=notes if notes else None)
        created = service.tasks().insert(tasklist=tasklist_id, body=body).execute()
        return _parse_task(created)

    return await retry_blocking(do_create, f"Creating task '{title}'")


async def list_tasks(include_completed: bool = False) -> list[TaskInfo]:
    """
    List tasks from the default task list.

    Takes:
        include_completed (bool): False (default) returns only tasks
        that aren't done yet — the natural default for "what's on my
        to-do list". True also includes tasks already marked complete.

    Returns:
        list[TaskInfo].

    Can this fail: yes — goes through retry_blocking().
    """
    def do_list():
        service = _get_service()
        tasklist_id = _get_tasklist_id(service)
        result = service.tasks().list(
            tasklist=tasklist_id,
            maxResults=100,
            showCompleted=include_completed,
            showHidden=include_completed,
        ).execute()
        return [_parse_task(item) for item in result.get("items", [])]

    return await retry_blocking(do_list, "Listing tasks")


async def search_tasks_by_name(query: str) -> list[TaskInfo]:
    """
    Find tasks whose title plausibly matches a spoken name. Same
    fresh-lookup-fallback role as calendar_tools.search_events_by_name()
    — agent.py checks its own short-term "recently listed" cache first,
    and only calls this when that comes up empty.

    Takes:
        query (str): the name (or part of it) the person said.

    Returns:
        list[TaskInfo]: plausible matches, best match first.

    Searches ALL tasks, including already-completed ones — "delete the
    groceries task" is a reasonable thing to say whether or not it was
    already checked off.

    Can this fail: yes — list_tasks() above can raise.
    """
    candidates = await list_tasks(include_completed=True)
    return text_match.find_matches(query, candidates, get_name=lambda task: task.title)


async def update_task(
    task_id: str,
    title: str = None,
    due_date: date = None,
    notes: str = None,
    clear_due_date: bool = False,
) -> TaskInfo:
    """
    Change one or more fields on an existing task. Only the fields
    actually given get changed — everything else stays exactly as it
    was.

    Takes:
        task_id (str): which task to change.
        title, notes: whichever fields should change. None means leave
        unchanged.
        due_date (date | None): a new due date. Ignored if
        clear_due_date is True.
        clear_due_date (bool): True to remove an existing due date
        entirely, rather than change it to a different one.

    Returns:
        TaskInfo: the task as it looks after the change.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_update():
        service = _get_service()
        tasklist_id = _get_tasklist_id(service)
        body = _task_body(title=title, due_date=due_date, notes=notes, clear_due_date=clear_due_date)
        updated = service.tasks().patch(tasklist=tasklist_id, task=task_id, body=body).execute()
        return _parse_task(updated)

    return await retry_blocking(do_update, f"Updating task {task_id}")


async def complete_task(task_id: str) -> TaskInfo:
    """
    Mark a task as done.

    Takes:
        task_id (str): which task to complete.

    Returns:
        TaskInfo: the task, now marked completed.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_complete():
        service = _get_service()
        tasklist_id = _get_tasklist_id(service)
        updated = service.tasks().patch(
            tasklist=tasklist_id, task=task_id, body={"status": "completed"}
        ).execute()
        return _parse_task(updated)

    return await retry_blocking(do_complete, f"Completing task {task_id}")


async def delete_task(task_id: str) -> None:
    """
    Delete a task.

    Takes:
        task_id (str): which task to delete.

    Returns: nothing.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_delete():
        service = _get_service()
        tasklist_id = _get_tasklist_id(service)
        service.tasks().delete(tasklist=tasklist_id, task=task_id).execute()

    await retry_blocking(do_delete, f"Deleting task {task_id}")