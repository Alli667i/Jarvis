"""
calendar_tools.py

Everything this app can do with Google Calendar: create, list, search by
name, update (reschedule), and delete events. Every function here goes
through retry.retry_blocking() — Google's API client library
(googleapiclient) is synchronous, not async-native, so calls run in a
background thread (see retry.py) instead of freezing the whole server
while waiting on the network.

This file only knows how to talk to Google Calendar. It has no idea what
a "pending action" or a "confirmation step" is — that orchestration
(deciding whether something should be an event or a task, guessing
details from context, holding one pending action until it's confirmed)
lives in agent.py, which calls the plain functions here once it already
knows exactly what to create/update/delete.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from googleapiclient.discovery import build

import google_auth
import text_match
import time_utils
from retry import retry_blocking


@dataclass
class CalendarEvent:
    """
    This app's own plain representation of one calendar event — not
    Google's raw API dict. Every function in this file returns and
    accepts this shape, so nothing outside this file needs to know what
    Google's event resource actually looks like.

    Fields:
        id (str): Google's event ID. Needed for update_event()/
        delete_event() later — never spoken aloud to the user.
        title (str): the event's name.
        start (datetime): timezone-aware, PKT.
        end (datetime): timezone-aware, PKT.
        description (str): free-text notes on the event. Empty string if
        none.
    """
    id: str
    title: str
    start: datetime
    end: datetime
    description: str = ""


def _get_service():
    """
    Build a fresh Google Calendar API client using this app's saved
    credentials.

    Takes: nothing.

    Returns: a googleapiclient Resource object for the Calendar v3 API.

    Can this fail: yes — whatever google_auth.get_credentials() can
    raise (see that file) passes straight through. Called fresh inside
    every function below, not cached, so a refreshed or renewed token is
    always picked up automatically without needing a server restart.
    """
    creds = google_auth.get_credentials()
    return build("calendar", "v3", credentials=creds)


def _parse_event(raw: dict) -> CalendarEvent:
    """
    Turn one event resource from Google's API into this app's own
    CalendarEvent shape.

    Takes:
        raw (dict): one item from Google Calendar's list()/insert()/
        patch() response.

    Returns:
        CalendarEvent.

    Can this fail: no for events this app creates (always has a specific
    time). Defensively handles all-day events that might already exist
    on the calendar from somewhere else (Google represents those with a
    "date" field instead of "dateTime") by treating them as spanning the
    full PKT day, so listing/searching never crashes on an event this
    app didn't create itself.
    """
    start_raw = raw.get("start", {})
    end_raw = raw.get("end", {})

    if "dateTime" in start_raw:
        start = datetime.fromisoformat(start_raw["dateTime"]).astimezone(time_utils.PKT)
    else:
        start = time_utils.start_of_day(date.fromisoformat(start_raw["date"]))

    if "dateTime" in end_raw:
        end = datetime.fromisoformat(end_raw["dateTime"]).astimezone(time_utils.PKT)
    else:
        # Google represents all-day events with an EXCLUSIVE end date —
        # a single-day event on Aug 14 has end.date="2026-08-15" (the
        # day after it ends), not "2026-08-14". Subtract a day to get
        # the actual last day the event covers before treating it as a
        # full PKT day.
        end_date_str = end_raw.get("date", start_raw.get("date"))
        end_date = date.fromisoformat(end_date_str) - timedelta(days=1)
        end = time_utils.end_of_day(end_date)

    return CalendarEvent(
        id=raw["id"],
        title=raw.get("summary", "Untitled"),
        start=start,
        end=end,
        description=raw.get("description", ""),
    )


def _event_body(title: str = None, start: datetime = None, end: datetime = None, description: str = None) -> dict:
    """
    Build the request body Google's API expects for creating or updating
    an event, including only the fields that were actually given.

    Takes:
        title, start, end, description: any combination — whichever
        fields should be set. None means "don't include this field,"
        which is what makes update_event() below able to change just one
        field without touching the rest of the event.

    Returns:
        dict: ready to hand to service.events().insert()/patch() as the
        `body` argument.

    Can this fail: yes — time_utils.to_google_timestamp() raises
    ValueError if start/end were given without a timezone attached. Not
    caught here on purpose; that would be a bug in whatever called this,
    not something to paper over silently.
    """
    body = {}
    if title is not None:
        body["summary"] = title
    if description is not None:
        body["description"] = description
    if start is not None:
        body["start"] = {
            "dateTime": time_utils.to_google_timestamp(start),
            "timeZone": time_utils.GOOGLE_TIMEZONE_NAME,
        }
    if end is not None:
        body["end"] = {
            "dateTime": time_utils.to_google_timestamp(end),
            "timeZone": time_utils.GOOGLE_TIMEZONE_NAME,
        }
    return body


async def create_event(title: str, start: datetime, end: datetime, description: str = "") -> CalendarEvent:
    """
    Create a new event on the primary Google Calendar.

    Takes:
        title (str): the event's name.
        start (datetime): timezone-aware — build it with time_utils
        (e.g. time_utils.make_datetime()).
        end (datetime): timezone-aware, same as start.
        description (str): optional free-text notes.

    Returns:
        CalendarEvent: the event as Google actually saved it, including
        its new id.

    Can this fail: yes — goes through retry_blocking() (try once, retry
    once, then raise RuntimeError). See retry.py.
    """
    def do_create():
        service = _get_service()
        body = _event_body(title=title, start=start, end=end, description=description)
        created = service.events().insert(calendarId="primary", body=body).execute()
        return _parse_event(created)

    return await retry_blocking(do_create, f"Creating calendar event '{title}'")


async def list_events(start_day: date, end_day: date) -> list[CalendarEvent]:
    """
    List every event between the start of `start_day` and the end of
    `end_day` (inclusive), in PKT.

    Takes:
        start_day (date): first day to include.
        end_day (date): last day to include — same as start_day for a
        single day's events.

    Returns:
        list[CalendarEvent]: earliest first (Google's own ordering).
        Empty list if nothing's scheduled in that range.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_list():
        service = _get_service()
        result = service.events().list(
            calendarId="primary",
            timeMin=time_utils.to_google_timestamp(time_utils.start_of_day(start_day)),
            timeMax=time_utils.to_google_timestamp(time_utils.end_of_day(end_day)),
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        ).execute()
        return [_parse_event(item) for item in result.get("items", [])]

    return await retry_blocking(do_list, f"Listing calendar events from {start_day} to {end_day}")


async def search_events_by_name(query: str) -> list[CalendarEvent]:
    """
    Find events whose title plausibly matches a spoken name, e.g. "the
    dentist one". Used as a fresh-lookup fallback when nothing in recent
    conversation already matched (see architecture.md's "Editing /
    deleting by name" section) — agent.py checks its own short-term
    cache of recently-listed items first, and only calls this when that
    cache comes up empty.

    Takes:
        query (str): the name (or part of it) the person said.

    Returns:
        list[CalendarEvent]: plausible matches, best match first. Empty
        list if nothing matched well enough.

    Searches from yesterday through 30 days ahead — wide enough to catch
    something scheduled a while back or set up for later, without
    listing the entire calendar history every time.

    Can this fail: yes — list_events() above can raise (already goes
    through retry_blocking() there).
    """
    today = time_utils.today()
    window_start = today - timedelta(days=1)
    window_end = today + timedelta(days=30)
    candidates = await list_events(window_start, window_end)
    return text_match.find_matches(query, candidates, get_name=lambda event: event.title)


async def update_event(
    event_id: str,
    title: str = None,
    start: datetime = None,
    end: datetime = None,
    description: str = None,
) -> CalendarEvent:
    """
    Change one or more fields on an existing event. Only the fields
    actually given get changed — everything else on the event stays
    exactly as it was.

    Takes:
        event_id (str): which event to change (from a previous
        create_event/list_events/search_events_by_name call).
        title, start, end, description: whichever fields should change.
        Leave as None for anything that should stay the same.

    Returns:
        CalendarEvent: the event as it looks after the change.

    Can this fail: yes — goes through retry_blocking(). Also raises
    RuntimeError if event_id doesn't exist (Google returns a 404, which
    surfaces through the same retry-and-report path as any other
    failure).
    """
    def do_update():
        service = _get_service()
        body = _event_body(title=title, start=start, end=end, description=description)
        updated = service.events().patch(calendarId="primary", eventId=event_id, body=body).execute()
        return _parse_event(updated)

    return await retry_blocking(do_update, f"Updating calendar event {event_id}")


async def delete_event(event_id: str) -> None:
    """
    Cancel/delete an event.

    Takes:
        event_id (str): which event to delete.

    Returns: nothing.

    Can this fail: yes — goes through retry_blocking().
    """
    def do_delete():
        service = _get_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()

    await retry_blocking(do_delete, f"Deleting calendar event {event_id}")