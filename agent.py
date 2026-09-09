"""
agent.py

The orchestrator. This is where the conversation actually happens: text
comes in, this decides whether to just answer, look something up, or
stage a change and ask for confirmation -- and it's the ONLY place in
the whole app that holds conversation state (message history, the one
pending action, the short-term "what did I just mention" cache).

calendar_tools.py and tasks_tools.py know nothing about any of this --
they're pure create/list/update/delete functions. This file is what
decides WHEN to call them, and specifically enforces that nothing ever
gets written to Google without the user's own words confirming it
first. That guarantee (see _handle_confirm_pending_action and the
same-turn guard below) is enforced in code, not left as an instruction
the LLM might or might not follow.

NOTE ON THE SYSTEM PROMPT: _build_system_prompt() below is a
FUNCTIONAL placeholder -- it covers only what's needed for the tools
and the propose/confirm mechanics to work correctly. It does not
attempt the actual JARVIS voice/personality; a separate prompt style
guide is coming later and will be merged into this function, not
replace this file's mechanics.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import re

import calendar_tools
import tasks_tools
import text_match
import time_utils
import llm_client


# How many tool-calling round trips one user turn is allowed before
# giving up rather than looping forever on a confused LLM.
MAX_TOOL_ITERATIONS = 5

# How many items get read out in full when listing events/tasks. Capped
# at the DATA level (here) rather than trusting wording alone to keep
# JARVIS from reading out a long list item by item.
MAX_LISTED_ITEMS = 8


@dataclass
class PendingAction:
    """
    Exactly one of these exists at a time, or none. Built entirely by
    _handle_propose_action, acted on only by
    _handle_confirm_pending_action, cleared by either that or
    _handle_cancel_pending_action.

    Fields follow the same "None means don't touch this field" rule
    that calendar_tools.update_event() / tasks_tools.update_task()
    already use, so confirming an action is just handing these fields
    straight through to the right tool function -- no re-deriving
    anything at confirm time.

    Fields:
        operation (str): "create" | "edit" | "delete" | "complete".
        "complete" only applies to tasks.
        category (str): "event" | "task".
        item_id (str | None): which existing item, for edit/delete/complete.
        title (str | None): NEW title. Required for create. Only set
        for edit if actually renaming.
        start (datetime | None): NEW event start. Events only.
        end (datetime | None): NEW event end. Events only.
        due_date (date | None): NEW task due date. Tasks only.
        clear_due_date (bool): tasks edit only -- True removes the due
        date entirely, distinct from leaving due_date as None (unchanged).
        notes (str | None): NEW notes, if given.
        summary (str): the human-readable confirmation text, already
        rendered once at propose time -- confirm just reuses it.
    """
    operation: str
    category: str
    item_id: str | None = None
    title: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    due_date: date | None = None
    clear_due_date: bool = False
    notes: str | None = None
    summary: str = ""


# --- Conversation-level state -------------------------------------------
# All module-level on purpose: this app is a single ongoing conversation
# with one person, not a multi-user server. See architecture.md.

_conversation_history: list[dict] = []
_pending_action: PendingAction | None = None

# Guards against propose_action and confirm_pending_action firing in the
# SAME LLM turn, which would let something get saved without the user's
# own words confirming it in between. Reset at the start of every
# handle_message() call; set True the moment propose_action runs.
_pending_proposed_this_turn = False

# The short-term "what did JARVIS just mention" cache used to resolve
# spoken references like "the dentist one" without needing a fresh
# Google lookup every time. Replaced wholesale on every list/search, not
# accumulated -- see _update_recent_cache().
_recent_items: list[dict] = []

# --- Proactive reminders (deterministic, code-only -- see CLAUDE.md's
# Rule Zero: never left to the LLM to remember to bring these up) -----

# How close (in minutes) an event's start time needs to be before it
# gets proactively mentioned. 75 is Ali's own starting point for
# testing, not a carefully tuned number -- expect to adjust once this
# has been used for real.
EVENT_REMINDER_WINDOW_MINUTES = 75

# Which task IDs have already been mentioned in a due-today reminder,
# and which date that applies to. Reset automatically the moment a
# check happens on a new date -- see _build_reminder_suffix(). This is
# the one piece of state in this feature that genuinely needs to persist
# across turns (a due-today task should be mentioned once per day, not
# once per turn); the event-proximity check needs no memory at all,
# since re-checking "how far away is it now" fresh every turn already
# gives the right behavior (mentioned every turn while inside the
# window, silent once it's passed) with nothing to track.
_mentioned_tasks_today: set[str] = set()
_mentioned_tasks_date: date | None = None


# --- Tool schemas the LLM sees -------------------------------------------
# These names match what's used throughout architecture.md. Every write
# goes through propose_action -> confirm_pending_action; nothing else in
# this list can save anything to Google.
#
# Split into two groups rather than one flat list: ALWAYS_AVAILABLE_TOOLS
# are offered every turn; PENDING_ACTION_TOOLS are only offered when
# there's actually something staged to confirm or cancel. See
# _get_available_tools() below, which is what actually decides what gets
# sent on a given call -- offering confirm_pending_action/
# cancel_pending_action when nothing is pending wastes tokens describing
# a tool that's guaranteed to fail if called, and it's one less option
# for the LLM to pick wrong.

ALWAYS_AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": (
                "List calendar events between two dates, inclusive. Read-only, runs "
                "immediately -- no confirmation needed. Use the current date given in "
                "the system prompt to resolve relative terms like 'today'/'tomorrow'/"
                "'this week' into actual YYYY-MM-DD dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD -- same as start_date for a single day"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List tasks from the to-do list. Read-only, runs immediately -- no confirmation needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_completed": {
                        "type": "boolean",
                        "description": "True to also include tasks already marked done. Defaults to false.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_action",
            "description": (
                "Stage a create, edit, delete, or complete action on a calendar event or "
                "task. Does NOT save anything -- it only stages the action and returns a "
                "summary to read back to the user for approval. To correct any detail "
                "after the user points something out, call this again with the full "
                "corrected picture -- do not call any other tool for a correction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["create", "edit", "delete", "complete"],
                        "description": "What to do. 'complete' only applies to tasks.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["event", "task"],
                        "description": (
                            "Decide this from context: a specific or clearly implied clock "
                            "time means 'event'. A to-do/deadline with no specific time means "
                            "'task'. Never ask the user which one up front -- guess, then "
                            "state the guess in the confirmation; they'll correct it there if "
                            "it's wrong."
                        ),
                    },
                    "item_reference": {
                        "type": "string",
                        "description": (
                            "Required for edit/delete/complete. A short natural-language "
                            "description of the existing event or task, e.g. 'the dentist "
                            "appointment' or 'buy groceries'. Never an ID. Not used for 'create'."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "The event/task name. Required for 'create'. Optional for 'edit' (only if renaming).",
                    },
                    "date": {
                        "type": "string",
                        "description": "YYYY-MM-DD. Required for creating an event or a task with a due date. Optional for edits.",
                    },
                    "time": {
                        "type": "string",
                        "description": (
                            "HH:MM in 24-hour format. Events only. Required when creating an "
                            "event. Omit entirely for tasks."
                        ),
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Events only. How long the event lasts. Defaults to 60 if not given.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional free-text notes/description.",
                    },
                    "clear_due_date": {
                        "type": "boolean",
                        "description": "Tasks 'edit' only. True to remove an existing due date entirely.",
                    },
                },
                "required": ["operation", "category"],
            },
        },
    },
]

# Only offered when _pending_action is not None -- see _get_available_tools().
PENDING_ACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "confirm_pending_action",
            "description": (
                "Actually save the currently proposed action to Google Calendar/Tasks. "
                "Only call this after the user has clearly said yes/confirmed in their "
                "own words -- never in the same turn as the propose_action call that "
                "created the proposal."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_pending_action",
            "description": "Discard the currently proposed action without saving anything. Call this if the user says no/never mind/cancel.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _get_available_tools() -> list[dict]:
    """
    Which tools the LLM is actually allowed to pick from THIS call --
    not a fixed list. Called fresh on every LLM call within a turn (not
    once before the loop), since state can change mid-turn: if
    propose_action fires partway through a turn, the very next call in
    that same turn should NOT suddenly offer confirm_pending_action --
    that's covered below.

    Takes: nothing (reads module state: _pending_action, _pending_proposed_this_turn).

    Returns:
        list[dict]: ALWAYS_AVAILABLE_TOOLS, plus PENDING_ACTION_TOOLS
        only when there's something genuinely pending FROM A PRIOR TURN.

    Can this fail: no.

    Why "not _pending_proposed_this_turn" matters here, not just
    "_pending_action is not None": this is a second, independent layer
    on top of the same-turn guard already enforced in
    _handle_confirm_pending_action. That guard REFUSES a same-turn
    confirm if the LLM attempts it; this function goes one step further
    and doesn't even OFFER confirm/cancel as options in that situation,
    so there's nothing to attempt in the first place. Belt and
    suspenders on purpose -- one is "don't allow it," this is "don't
    even suggest it's possible."
    """
    if _pending_action is not None and not _pending_proposed_this_turn:
        return ALWAYS_AVAILABLE_TOOLS + PENDING_ACTION_TOOLS
    return ALWAYS_AVAILABLE_TOOLS


# --- The recent-items cache and name resolution ---------------------------

def _event_to_cache_item(event: calendar_tools.CalendarEvent) -> dict:
    """
    Convert a CalendarEvent into the plain dict shape the recent-items
    cache uses. Keeping start/end here (not just id/title) is what lets
    _handle_propose_action later fill in a missing date or time from the
    event's CURRENT schedule when only one of the two is being changed
    -- see the comment in _handle_propose_action for why that matters.
    """
    return {"id": event.id, "title": event.title, "category": "event", "start": event.start, "end": event.end, "due_date": None}


def _task_to_cache_item(task: tasks_tools.TaskInfo) -> dict:
    """Convert a TaskInfo into the same plain dict shape as _event_to_cache_item(), for tasks."""
    return {"id": task.id, "title": task.title, "category": "task", "start": None, "end": None, "due_date": task.due_date}


def _update_recent_cache(items: list[dict]) -> None:
    """
    Replace the short-term memory of "things JARVIS just told the user
    about" with a new set.

    Takes:
        items (list[dict]): the new set, in the shape _event_to_cache_item()/
        _task_to_cache_item() produce.

    Returns: nothing.

    Deliberately REPLACES rather than accumulates -- this represents the
    most recent listing or search result, not a growing history of
    everything ever mentioned in the conversation. Can this fail: no.
    """
    global _recent_items
    _recent_items = items


async def _resolve_item_reference(reference: str, category: str):
    """
    Figure out which existing event or task a spoken reference like "the
    dentist one" refers to.

    Takes:
        reference (str): what the user said, e.g. "the dentist one".
        category (str): "event" or "task" -- only items of this category
        are considered.

    Returns:
        tuple: one of —
            ("resolved", item_dict) — exactly one confident match.
            ("ambiguous", [item_dicts]) — 2+ plausible matches.
            ("not_found", None) — nothing matched at all.
        item_dict is in the same shape as _event_to_cache_item()/
        _task_to_cache_item(), so callers can use its start/end/due_date
        fields directly.

    Checks the short-term cache first (whatever was most recently
    listed or searched) -- only falls back to a fresh Google lookup if
    nothing in the cache matches, per architecture.md's "Editing /
    deleting by name" design. A fresh lookup's results are folded back
    into the cache afterward, so a follow-up correction in the same
    conversation can find the same item via the cache too.

    Can this fail: yes -- the fresh-lookup fallback calls
    calendar_tools.search_events_by_name() / tasks_tools.search_tasks_by_name(),
    either of which can raise (both go through retry_blocking already).
    Not caught here on purpose -- _handle_propose_action's caller
    (_dispatch_tool_call) catches it generically and reports it as a
    normal tool error.
    """
    same_category_cache = [item for item in _recent_items if item["category"] == category]
    cache_matches = text_match.find_matches(reference, same_category_cache, get_name=lambda item: item["title"])
    if len(cache_matches) == 1:
        return "resolved", cache_matches[0]
    if len(cache_matches) > 1:
        return "ambiguous", cache_matches

    if category == "event":
        fresh_events = await calendar_tools.search_events_by_name(reference)
        fresh_items = [_event_to_cache_item(event) for event in fresh_events]
    else:
        fresh_tasks = await tasks_tools.search_tasks_by_name(reference)
        fresh_items = [_task_to_cache_item(task) for task in fresh_tasks]

    if fresh_items:
        _update_recent_cache(fresh_items)

    if len(fresh_items) == 1:
        return "resolved", fresh_items[0]
    if len(fresh_items) > 1:
        return "ambiguous", fresh_items
    return "not_found", None


# --- Tool handlers ---------------------------------------------------------

def _build_confirmation_summary(
    operation: str, category: str, display_title: str, new_title: str | None, start: datetime | None,
    due_date: date | None, notes: str | None, clear_due_date: bool,
) -> str:
    """
    Turn a proposed action into the plain-language text read back to the
    user for approval. Called once, at propose time, and reused as-is
    when the action is later confirmed or reported as saved.

    Takes:
        display_title (str): what to call the item in the sentence --
        always populated, even when nothing about the title is changing
        (e.g. editing just the time of an existing event still needs a
        name to refer to it by).
        new_title (str | None): only set if the title is ACTUALLY being
        changed. Kept separate from display_title so an edit that
        doesn't touch the title never wrongly says "renamed to X" just
        because X happens to be its current name.
        operation, category, start, due_date, notes, clear_due_date: the
        rest of the about-to-be-staged PendingAction's fields.

    Returns: str, ready to speak.

    Can this fail: no.
    """
    if operation == "create":
        if category == "event":
            when = time_utils.format_time_for_speech(start)
            return f"I'll save that as an event -- {display_title}, {when}. Sound good?"
        when = f", due {time_utils.format_date_for_speech(due_date)}" if due_date else ""
        return f"I'll add that as a task -- {display_title}{when}. Sound good?"

    if operation == "edit":
        changes = []
        if new_title:
            changes.append(f"renamed to '{new_title}'")
        if start:
            changes.append(f"moved to {time_utils.format_time_for_speech(start)}")
        if clear_due_date:
            changes.append("due date removed")
        elif due_date:
            changes.append(f"due date changed to {time_utils.format_date_for_speech(due_date)}")
        if notes:
            changes.append("notes updated")
        change_text = "; ".join(changes) if changes else "no changes given"
        return f"I'll update '{display_title}' -- {change_text}. Confirm?"

    if operation == "delete":
        noun = "event" if category == "event" else "task"
        return f"I'll delete the {noun} '{display_title}'. Confirm?"

    return f"I'll mark '{display_title}' as done. Confirm?"  # operation == "complete"


async def _handle_propose_action(args: dict) -> dict:
    """
    Handler for the propose_action tool. Stages a PendingAction and
    returns a summary for the LLM to read back -- never touches Google.

    Takes:
        args (dict): the LLM's tool call arguments, matching
        TOOL_SCHEMAS's propose_action schema above.

    Returns:
        dict: {"status": "proposed", "summary": ...} on success.
        {"status": "not_found"/"ambiguous"/"error", "message": ...}
        otherwise -- these are read by the LLM, not the user directly,
        so it can decide how to phrase the follow-up question.

    Can this fail: no exceptions raised here on purpose -- every
    validation problem returns an error-shaped dict instead. The one
    exception is whatever _resolve_item_reference()'s fresh-lookup
    fallback can raise (a real Google API failure), which is left
    uncaught for _dispatch_tool_call to catch generically.
    """
    global _pending_action, _pending_proposed_this_turn

    operation = args.get("operation")
    category = args.get("category")

    if operation not in ("create", "edit", "delete", "complete"):
        return {"status": "error", "message": f"Unknown operation '{operation}'."}
    if category not in ("event", "task"):
        return {"status": "error", "message": f"Unknown category '{category}'."}
    if operation == "complete" and category != "task":
        return {"status": "error", "message": "'complete' only applies to tasks, not events."}

    item_id = None
    display_title = args.get("title")
    existing = None

    if operation in ("edit", "delete", "complete"):
        reference = args.get("item_reference")
        if not reference:
            return {"status": "error", "message": f"'{operation}' needs to know which {category} -- describe it with item_reference."}
        resolution, result = await _resolve_item_reference(reference, category)
        if resolution == "not_found":
            return {"status": "not_found", "message": f"I couldn't find a {category} matching '{reference}'."}
        if resolution == "ambiguous":
            names = "; ".join(item["title"] for item in result[:5])
            return {"status": "ambiguous", "message": f"More than one {category} matches '{reference}': {names}. Which one?"}
        item_id = result["id"]
        existing = result
        if not display_title:
            display_title = result["title"]

    new_title = args.get("title") if operation in ("create", "edit") else None
    clear_due_date = bool(args.get("clear_due_date", False))
    notes = args.get("notes") if operation in ("create", "edit") else None

    start = end = None
    due_date = None

    if category == "event" and operation in ("create", "edit"):
        date_str = args.get("date")
        time_str = args.get("time")

        if operation == "create" and (not date_str or not time_str):
            return {"status": "error", "message": "Creating an event needs both a date and a time."}

        if operation == "edit" and existing and existing.get("start"):
            # If only one of date/time is being changed, fill in the
            # other from the event's CURRENT start -- otherwise "move it
            # to 4pm" with no date given would silently do nothing at
            # all, since neither date_str nor time_str alone builds a
            # usable start below.
            if not date_str:
                date_str = existing["start"].date().isoformat()
            if not time_str:
                time_str = existing["start"].strftime("%H:%M")

        if date_str and time_str:
            try:
                day = date.fromisoformat(date_str)
                hour_str, minute_str = time_str.split(":")
                start = time_utils.make_datetime(day, int(hour_str), int(minute_str))
            except (ValueError, TypeError):
                return {"status": "error", "message": f"Couldn't understand the date/time '{date_str} {time_str}'."}

            duration = args.get("duration_minutes")
            if duration is None and existing and existing.get("start") and existing.get("end"):
                # Preserve the event's current length if only the time
                # is being moved and no new duration was given.
                duration = int((existing["end"] - existing["start"]).total_seconds() / 60)
            end = start + timedelta(minutes=(duration or 60))

    if category == "task" and operation in ("create", "edit") and not clear_due_date:
        date_str = args.get("date")
        if date_str:
            try:
                due_date = date.fromisoformat(date_str)
            except ValueError:
                return {"status": "error", "message": f"Couldn't understand the date '{date_str}'."}

    if operation == "create" and not display_title:
        return {"status": "error", "message": f"Creating a {category} needs a title."}

    summary = _build_confirmation_summary(operation, category, display_title, new_title, start, due_date, notes, clear_due_date)

    _pending_action = PendingAction(
        operation=operation,
        category=category,
        item_id=item_id,
        title=new_title,
        start=start,
        end=end,
        due_date=due_date,
        clear_due_date=clear_due_date,
        notes=notes,
        summary=summary,
    )
    _pending_proposed_this_turn = True

    return {"status": "proposed", "summary": summary}


async def _handle_confirm_pending_action(args: dict) -> dict:
    """
    Handler for confirm_pending_action. The ONLY function in this whole
    app that actually writes to Google Calendar/Tasks.

    Takes:
        args (dict): unused -- confirm_pending_action takes no
        parameters, it acts on whatever is currently pending.

    Returns:
        dict: {"status": "done", "message": ...} on success.
        {"status": "error", "message": ...} if there's nothing pending,
        if confirmation is being attempted in the same turn it was
        proposed (see _pending_proposed_this_turn), or if the actual
        save failed.

    Can this fail: no exceptions raised here -- a failed save from
    calendar_tools/tasks_tools (a RuntimeError, after their own
    retry_blocking already tried twice) is logged to the console and
    turned into a generic error-shaped result -- the real exception
    text never reaches the LLM, so there's nothing technical for it to
    accidentally relay verbatim in a spoken reply. Deliberately does
    NOT clear _pending_action on a failed save -- the proposal stays
    staged so the user can just ask to try again instead of redoing the
    whole request from scratch.
    """
    global _pending_action

    if _pending_action is None:
        return {"status": "error", "message": "There's nothing pending to confirm."}

    if _pending_proposed_this_turn:
        return {
            "status": "error",
            "message": (
                "This was just proposed in the same turn -- it can only be confirmed in "
                "response to the user's own words in a later message, not automatically."
            ),
        }

    action = _pending_action

    try:
        if action.category == "event":
            if action.operation == "create":
                await calendar_tools.create_event(action.title, action.start, action.end, description=action.notes or "")
            elif action.operation == "edit":
                await calendar_tools.update_event(action.item_id, title=action.title, start=action.start, end=action.end, description=action.notes)
            elif action.operation == "delete":
                await calendar_tools.delete_event(action.item_id)
        else:
            if action.operation == "create":
                await tasks_tools.create_task(action.title, due_date=action.due_date, notes=action.notes or "")
            elif action.operation == "edit":
                await tasks_tools.update_task(action.item_id, title=action.title, due_date=action.due_date, notes=action.notes, clear_due_date=action.clear_due_date)
            elif action.operation == "complete":
                await tasks_tools.complete_task(action.item_id)
            elif action.operation == "delete":
                await tasks_tools.delete_task(action.item_id)
    except RuntimeError as error:
        print(f"[agent] confirm_pending_action failed to save: {error}")
        return {
            "status": "error",
            "message": "That didn't save -- there may be a connection problem. It's still ready if you want to try again.",
        }

    saved_summary = action.summary
    _pending_action = None
    return {"status": "done", "message": f"Saved. {saved_summary}"}


async def _handle_cancel_pending_action(args: dict) -> dict:
    """
    Handler for cancel_pending_action. Discards whatever is currently
    staged without saving anything.

    Takes:
        args (dict): unused.

    Returns:
        dict: {"status": "cancelled"} on success, or
        {"status": "error", "message": ...} if nothing was pending.

    Can this fail: no.
    """
    global _pending_action
    if _pending_action is None:
        return {"status": "error", "message": "There's nothing pending to cancel."}
    _pending_action = None
    return {"status": "cancelled"}


async def _handle_list_events(args: dict) -> dict:
    """
    Handler for list_events. Read-only -- runs immediately, no
    confirmation step. Updates the recent-items cache with whatever it
    finds, so a follow-up "cancel the dentist one" can resolve against
    these results without a fresh lookup.

    Takes:
        args (dict): {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}.

    Returns:
        dict: {"status": "ok", "count": int, "events": [...], "truncated": bool}
        on success. The "events" list is capped at MAX_LISTED_ITEMS --
        "truncated" tells the LLM there were more than that, so it can
        mention the total count without reading out every single one.
        {"status": "error", "message": ...} if the dates couldn't be parsed.

    Can this fail: no exceptions raised here -- calendar_tools.list_events()
    can raise (already goes through retry_blocking there), left uncaught
    for _dispatch_tool_call to catch generically.
    """
    try:
        start_day = date.fromisoformat(args["start_date"])
        end_day = date.fromisoformat(args["end_date"])
    except (KeyError, ValueError, TypeError):
        return {"status": "error", "message": "start_date/end_date must be given as YYYY-MM-DD."}

    events = await calendar_tools.list_events(start_day, end_day)
    _update_recent_cache([_event_to_cache_item(event) for event in events])

    shown = events[:MAX_LISTED_ITEMS]
    return {
        "status": "ok",
        "count": len(events),
        "events": [{"title": event.title, "when": time_utils.format_time_for_speech(event.start)} for event in shown],
        "truncated": len(events) > MAX_LISTED_ITEMS,
    }


async def _handle_list_tasks(args: dict) -> dict:
    """
    Handler for list_tasks. Read-only -- runs immediately, no
    confirmation step. Same cache-updating and truncation behavior as
    _handle_list_events above.

    Takes:
        args (dict): {"include_completed": bool} -- optional, defaults
        to False.

    Returns:
        dict: {"status": "ok", "count": int, "tasks": [...], "truncated": bool}.

    Can this fail: no exceptions raised here -- tasks_tools.list_tasks()
    can raise, left uncaught for _dispatch_tool_call to catch generically.
    """
    include_completed = bool(args.get("include_completed", False))
    tasks = await tasks_tools.list_tasks(include_completed=include_completed)
    _update_recent_cache([_task_to_cache_item(task) for task in tasks])

    shown = tasks[:MAX_LISTED_ITEMS]
    return {
        "status": "ok",
        "count": len(tasks),
        "tasks": [
            {
                "title": task.title,
                "due": time_utils.format_date_for_speech(task.due_date) if task.due_date else None,
                "completed": task.completed,
            }
            for task in shown
        ],
        "truncated": len(tasks) > MAX_LISTED_ITEMS,
    }


_TOOL_HANDLERS = {
    "propose_action": _handle_propose_action,
    "confirm_pending_action": _handle_confirm_pending_action,
    "cancel_pending_action": _handle_cancel_pending_action,
    "list_events": _handle_list_events,
    "list_tasks": _handle_list_tasks,
}


async def _dispatch_tool_call(call: llm_client.ToolCall) -> dict:
    """
    Run whichever tool the LLM asked for and return a JSON-serializable
    result for it to see.

    Takes:
        call (llm_client.ToolCall): the tool call from the LLM (name +
        already-parsed arguments).

    Returns:
        dict: always a dict, even for failures -- so the LLM always has
        something structured to read rather than a raw exception
        message. On an unexpected failure, "message" is a fixed,
        generic sentence, never the exception's own text -- the LLM
        only ever sees "something went wrong," never a raw stack trace
        or JSON error blob it might otherwise relay verbatim to the
        user. The real error is printed to the console instead, where
        it's actually useful for debugging.

    Can this fail: no -- every handler's own EXPECTED failure modes
    (bad input, nothing pending, ambiguous match, etc.) already return
    error-shaped dicts on their own, with messages already written to
    be safe to relay as-is (see e.g. _handle_propose_action's messages).
    Anything else that goes wrong inside a handler (e.g. a real Google
    API failure bubbling up from calendar_tools/tasks_tools) is caught
    here as a last resort, logged, and turned into the same generic
    {"status": "error", ...} shape, so one failed tool call never
    crashes the whole conversation turn OR leaks technical detail into
    a spoken reply.
    """
    handler = _TOOL_HANDLERS.get(call.name)
    if handler is None:
        return {"status": "error", "message": f"Unknown tool '{call.name}'."}

    try:
        return await handler(call.arguments)
    except Exception as error:
        print(f"[agent] Tool '{call.name}' failed: {error}")
        return {
            "status": "error",
            "message": "Something went wrong trying to do that -- there may be a connection problem.",
        }


# --- System prompt: small base + lazy-loaded response-style blocks --------
# PROMPT_BASE is sent on every call -- persona, and the few things the LLM
# genuinely needs before it can make its FIRST tool decision at all. Note
# what's deliberately NOT here: category-guessing, the "don't say an ID
# aloud" rule, and the correction-handling rule all already live directly
# in propose_action's own schema description above -- repeating them here
# would just be paying twice for the same instruction. This file's job is
# base persona/behavior; each tool's own description is the right place
# for that tool's own usage rules, since the LLM sees the schema before it
# ever needs the rule, no separate loading required.
#
# RESPONSE_BLOCKS holds PHRASING/STYLE guidance that only matters once a
# given tool has actually been used -- there's no reason to spend tokens
# on "keep listings brief" on a turn that never lists anything. Mirrors
# the PROMPT_BLOCKS/TOOL_BLOCK_MAP pattern from the previous JARVIS
# version, adapted to this app's smaller, unified tool set (propose_action
# covers what used to be four separate tools, so the block map is shorter
# too -- and it'll grow the same lazy way as future tools/modules get
# added, instead of the base prompt growing with them).

PROMPT_BASE = """You are JARVIS, a personal voice assistant. You speak like JARVIS from Iron Man -- calm, precise, occasionally dry, always composed.

Your replies are read aloud, so write in plain natural speech only -- no markdown, no bullet points, no asterisks or other symbols, nothing that would sound broken spoken aloud. Address the user as sir occasionally, not in every single reply.

Only call a tool when it's actually needed for the request. For greetings and general conversation, just respond directly without calling anything.

Never treat your own proposal as already confirmed -- only call confirm_pending_action after the user's own later words actually confirm it.

If something didn't work, say so plainly -- never claim something succeeded when it didn't."""

RESPONSE_BLOCKS = {
    "listing": (
        "LISTING RULES: keep it brief and natural when speaking results -- say the "
        "count, name one or two items by name, offer to say more instead of reading "
        "through everything."
    ),
    "confirming": (
        "CONFIRMATION RULES: relay a proposal's summary naturally, like you're "
        "actually asking a question, not reading a receipt. Once something is "
        "confirmed and saved, say so plainly and briefly."
    ),
}

TOOL_BLOCK_MAP = {
    "list_events": "listing",
    "list_tasks": "listing",
    "propose_action": "confirming",
    "confirm_pending_action": "confirming",
}


def _time_context() -> str:
    """
    One line giving the LLM the current date/time in PKT, rebuilt fresh
    on every call so a conversation that runs for hours or days never
    goes stale on "today".

    Takes: nothing. Returns: str. Can this fail: no.
    """
    now = time_utils.now()
    current_time = now.strftime("%I:%M %p").lstrip("0")
    return f"Current date and time: {now.strftime('%A, %B %d, %Y')} at {current_time}, Pakistan time."


async def _build_reminder_suffix() -> str | None:
    """
    Check for anything that should be proactively mentioned this turn:
    the single nearest upcoming event today, if it's coming up soon
    enough to matter, and any tasks due today that haven't already been
    mentioned today. Fully deterministic -- this is never left to the
    LLM's judgment or memory (see CLAUDE.md's Rule Zero). The result
    gets appended to whatever JARVIS was already going to say for this
    turn, regardless of what the turn was actually about.

    Takes: nothing.

    Returns:
        str | None: a short, natural-language suffix ready to append to
        the reply, or None if there's nothing worth mentioning right
        now.

    Can this fail: no exceptions raised here on purpose. If either
    Google lookup fails (calendar_tools/tasks_tools raising after their
    own retry_blocking already tried twice), that specific check is
    just skipped -- a missed reminder is a much smaller problem than
    breaking the reply to whatever the person actually asked. The other
    check still runs normally even if one of the two fails.
    """
    global _mentioned_tasks_today, _mentioned_tasks_date

    today = time_utils.today()
    if _mentioned_tasks_date != today:
        _mentioned_tasks_today = set()
        _mentioned_tasks_date = today

    parts: list[str] = []

    # --- the single nearest upcoming event, if close enough ---
    try:
        events_today = await calendar_tools.list_events(today, today)
    except RuntimeError:
        events_today = []

    now = time_utils.now()
    upcoming = [event for event in events_today if event.start > now]
    if upcoming:
        nearest = min(upcoming, key=lambda event: event.start)
        minutes_away = (nearest.start - now).total_seconds() / 60
        if minutes_away <= EVENT_REMINDER_WINDOW_MINUTES:
            parts.append(f"{nearest.title} starts in {round(minutes_away)} minutes -- make sure you're ready.")

    # --- tasks due today, not yet mentioned today ---
    try:
        tasks = await tasks_tools.list_tasks(include_completed=False)
    except RuntimeError:
        tasks = []

    unmentioned_due_today = [
        task for task in tasks
        if task.due_date == today and task.id not in _mentioned_tasks_today
    ]
    if unmentioned_due_today:
        titles = ", ".join(task.title for task in unmentioned_due_today)
        parts.append(f"Also due today: {titles}.")
        for task in unmentioned_due_today:
            _mentioned_tasks_today.add(task.id)

    if not parts:
        return None
    return " ".join(parts)


def _clean_for_speech(text: str) -> str:
    """
    Strip markdown formatting from a reply before it's spoken or shown
    as a caption. PROMPT_BASE tells the LLM not to produce markdown
    (this is read aloud, after all) -- but that's a prompt instruction,
    not a guarantee, and LLMs do sometimes produce bold/bullets/headers
    anyway. This is the code-level check that actually catches it,
    applied at every return path out of handle_message() below, so
    nothing can leave this file without going through it.

    Adapted from the equivalent function in the previous JARVIS build,
    with one real bug fixed: the original header-stripping pattern
    matched a bare '#' ANYWHERE in the text, not just at the start of a
    line -- meaning "the database #1 priority" would have its '#'
    silently deleted, becoming "the database 1 priority", even though
    it was never a markdown header. Fixed by anchoring the pattern to
    the start of a line, matching only genuine header syntax.

    Takes:
        text (str): raw text, usually straight from the LLM.

    Returns:
        str: the same text with markdown syntax removed -- bold/italic
        markers (both `*`/`**` and `_`/`__` styles), headers, inline
        code backticks, bullet/numbered list markers, and markdown
        links (keeping the visible link text, dropping the URL, since
        a spoken URL is useless). Newlines collapse to spaces -- this
        is one flowing spoken reply, not a formatted document.

    Can this fail: no -- regex substitution on a string can't raise for
    any input this function receives. Already-clean plain text (no
    markdown in it at all) passes through completely unchanged.
    """
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


async def handle_message(user_text: str) -> str:
    """
    Process one message from the user and return JARVIS's reply as
    text. This is the single entry point main.py calls for every turn,
    regardless of which device the message came from (screen mic or
    phone mic) -- speech-to-text and text-to-speech both happen outside
    this file, in main.py.

    Takes:
        user_text (str): what the user said, already transcribed.

    Returns:
        str: what JARVIS should say back. Speaking it aloud is main.py's
        job, not this function's. On a normal (non-error) reply, this
        may have a proactive reminder appended -- see
        _build_reminder_suffix() -- checked fresh every turn,
        independent of what the turn was actually about.

    Can this fail: rarely, and only for a real programming bug. Expected
    failure modes are all caught and turned into a normal spoken reply
    so the conversation can keep going -- and none of them leak
    technical detail into what actually gets spoken; the real error is
    always logged to the console instead, never included in the reply
    text:
        - the LLM being unreachable (RuntimeError from llm_client) is
          caught here and returns a generic apology.
        - every tool call's own failures are already caught by
          _dispatch_tool_call below and converted to a generic
          error-result message before the LLM even sees them, so
          there's nothing technical left for the LLM to accidentally
          relay verbatim.
    An unexpected exception (a genuine bug in this file's own logic) is
    deliberately NOT caught here, so it surfaces during testing/
    development instead of being silently hidden behind a generic
    "sorry" message that looks identical to an ordinary external failure.
    """
    global _pending_proposed_this_turn
    _pending_proposed_this_turn = False

    _conversation_history.append({"role": "user", "content": user_text})

    # system_content and used_blocks are both local to this one turn --
    # rebuilt from scratch every call to handle_message(), never carried
    # over from a previous turn. A block that loaded because list_events
    # fired last turn has no bearing on whether this turn needs it too.
    system_content = PROMPT_BASE + "\n\n" + _time_context()
    used_blocks: set[str] = set()

    for _ in range(MAX_TOOL_ITERATIONS):
        messages = [{"role": "system", "content": system_content}] + _conversation_history

        try:
            reply = await llm_client.get_reply(messages, tools=_get_available_tools())
        except RuntimeError as error:
            print(f"[agent] LLM unreachable: {error}")
            return _clean_for_speech("Sorry, I'm having trouble thinking right now -- give me a moment and try again.")

        _conversation_history.append(reply.raw_message)

        if not reply.tool_calls:
            final_text = reply.text or "Sorry, I didn't catch that -- could you say it again?"
            reminder = await _build_reminder_suffix()
            if reminder:
                final_text = f"{final_text} {reminder}"
            return _clean_for_speech(final_text)

        for call in reply.tool_calls:
            result = await _dispatch_tool_call(call)

            # Lazy-load this tool's response-style block, if it has one
            # and it hasn't already been added this turn. Affects only
            # the NEXT LLM call within this same turn -- e.g. after
            # list_events fires, the "listing" block shapes how the
            # results actually get read out.
            block_key = TOOL_BLOCK_MAP.get(call.name)
            if block_key and block_key not in used_blocks:
                used_blocks.add(block_key)
                system_content += "\n\n" + RESPONSE_BLOCKS[block_key]

            _conversation_history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return _clean_for_speech("Sorry, something went wrong processing that -- can you try again?")