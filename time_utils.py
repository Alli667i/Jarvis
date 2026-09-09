"""
time_utils.py

The ONE place in the whole app that knows what time it is and what
timezone we're in. Every other file that needs "now", "today", or needs
to build a specific date/time asks THIS file instead of computing it
themselves.

Why this file exists: in the old JARVIS codebase, two different tool
files each computed "today" using UTC instead of PKT (Pakistan time),
while every other file used PKT. That mismatch caused wrong-day results
for a few hours every night. It happened because the same piece of logic
("what day is it") was written in more than one place, and only some of
those places got updated when the convention was decided. This file
exists so that mistake is structurally harder to make again — there is
now only one place that logic can live.

We use a fixed +5:00 offset instead of a named timezone database entry
(like "Asia/Karachi" via Python's zoneinfo). Two reasons: Pakistan does
not observe daylight saving time, so a fixed offset is not just simpler,
it's exactly as correct as a named timezone would be. And a fixed offset
has zero dependency on the operating system having timezone data
installed, which matters because this app runs on a Termux server on an
old Android phone, not a normal Linux machine.
"""

from datetime import datetime, date, timedelta, timezone


# The one timezone this entire app runs on. Everything below is built on
# top of this single constant. If this project ever needs to support a
# different timezone, this is the only line that should need to change.
PKT = timezone(timedelta(hours=5))

# Separate from the PKT offset above. Google's Calendar API wants a named
# timezone string (not a numeric offset) in the "timeZone" field of an
# event. This is that name. It's kept here, next to PKT, so anyone
# reading this file sees both together instead of finding this string
# hardcoded somewhere else later.
GOOGLE_TIMEZONE_NAME = "Asia/Karachi"


def now() -> datetime:
    """
    What time is it right now, in PKT.

    Takes: nothing.

    Returns: a timezone-aware datetime object set to the current moment
    in PKT. "Timezone-aware" means the object carries its own offset
    (+05:00) rather than being an ambiguous naive datetime that some
    other part of the code could misinterpret as UTC or local system
    time.

    Can this fail: no. Reading the current time cannot error.
    """
    return datetime.now(PKT)


def today() -> date:
    """
    What day is it right now, in PKT — just the date, no time attached.

    Takes: nothing.

    Returns: a date object (year/month/day only) for the current PKT day.
    Useful anywhere code needs to compare "is this due date today?" or
    "is this tomorrow?" without caring about the exact time.

    Can this fail: no.
    """
    return now().date()


def make_datetime(day: date, hour: int, minute: int) -> datetime:
    """
    Build one specific PKT moment out of separate day / hour / minute
    values. This is the one place that actually constructs a
    timezone-aware datetime from parts — used whenever some other part
    of the app has figured out "this happens on this day, at this time"
    (for example: an event proposed for "tomorrow at 3 PM") and needs a
    single datetime object to store, format for speech, or send to
    Google.

    Takes:
        day (date): the calendar day this moment falls on.
        hour (int): 0-23.
        minute (int): 0-59.

    Returns: a timezone-aware datetime in PKT.

    Can this fail: yes — Python's datetime constructor raises ValueError
    on its own if hour/minute are out of range (e.g. hour=25). We don't
    catch that here; it should surface immediately to whoever passed in
    a bad value, rather than being silently swallowed.
    """
    return datetime(day.year, day.month, day.day, hour, minute, 0, tzinfo=PKT)


def start_of_day(day: date) -> datetime:
    """
    The very first moment of a given PKT day — midnight, 00:00:00.

    Takes:
        day (date): which day.

    Returns: a timezone-aware datetime at 00:00:00 PKT on that day.
    Used to build the "from" boundary when asking Google Calendar for
    everything happening on a specific day.

    Can this fail: no — hour/minute are fixed valid values here.
    """
    return make_datetime(day, 0, 0)


def end_of_day(day: date) -> datetime:
    """
    The very last moment of a given PKT day — 23:59:59.

    Takes:
        day (date): which day.

    Returns: a timezone-aware datetime at 23:59:59 PKT on that day.
    Used to build the "to" boundary when asking Google Calendar for
    everything happening on a specific day.

    Can this fail: no.
    """
    moment = make_datetime(day, 23, 59)
    return moment.replace(second=59)


def to_google_timestamp(moment: datetime) -> str:
    """
    Turn a datetime into the exact text format Google's Calendar/Tasks
    APIs expect (RFC3339, e.g. "2026-08-09T15:00:00+05:00").

    Takes:
        moment (datetime): MUST already be timezone-aware (built via
        now(), make_datetime(), start_of_day(), or end_of_day() above —
        never construct a raw datetime by hand elsewhere in this app).

    Returns: a string in RFC3339 format, offset included.

    Can this fail: yes, on purpose. If `moment` has no timezone attached
    (a "naive" datetime), this raises ValueError instead of silently
    guessing what timezone was intended. A naive datetime here is exactly
    the kind of mistake that caused the old UTC/PKT bug — refusing to
    guess is the point.
    """
    if moment.tzinfo is None:
        raise ValueError(
            "to_google_timestamp() got a datetime with no timezone attached. "
            "Build it with now(), make_datetime(), start_of_day(), or "
            "end_of_day() from this file instead of constructing it directly."
        )
    return moment.isoformat()


def _day_phrase(day: date) -> str:
    """
    Internal helper — not meant to be imported elsewhere. Turns a date
    into how a person would naturally say it out loud.

    Both format_time_for_speech() and format_date_for_speech() below need
    this exact same "today / tomorrow / a specific date" logic, so it
    lives here once instead of being written twice.

    Takes:
        day (date): the date to describe.

    Returns: "today", "tomorrow", or a phrase like "Friday, January 9"
    for anything further out.

    Can this fail: no — any valid date produces a valid phrase.
    """
    current = today()
    if day == current:
        return "today"
    if day == current + timedelta(days=1):
        return "tomorrow"
    return f"{day.strftime('%A, %B')} {day.day}"


def format_date_for_speech(day: date) -> str:
    """
    Human-friendly, spoken version of a date with no time attached.
    Meant for Tasks, which have a due date but no specific time of day.

    Takes:
        day (date): the date to describe.

    Returns: e.g. "today", "tomorrow", or "Friday, January 9".

    Can this fail: no.
    """
    return _day_phrase(day)


def format_time_for_speech(moment: datetime) -> str:
    """
    Human-friendly, spoken version of a specific date AND time. Meant for
    Calendar events, which do have a specific time attached.

    Takes:
        moment (datetime): timezone-aware. If it's not already in PKT,
        it gets converted to PKT first so "today"/"tomorrow" and the
        displayed time are correct for this app's one timezone rather
        than whatever timezone the caller happened to pass in.

    Returns: e.g. "today at 3 PM", "tomorrow at 6:30 PM",
    "Friday, January 9 at 3 PM". Minutes are only shown when they're not
    zero, because "3 PM" is how a person actually says it, not "3:00 PM".

    Can this fail: yes — raises ValueError if `moment` has no timezone
    attached, same reasoning as to_google_timestamp() above: refuse to
    guess rather than silently assume a timezone.
    """
    if moment.tzinfo is None:
        raise ValueError(
            "format_time_for_speech() got a datetime with no timezone attached. "
            "Build it with now() or make_datetime() from this file instead of "
            "constructing it directly."
        )

    moment_pkt = moment.astimezone(PKT)
    date_part = _day_phrase(moment_pkt.date())

    hour_12 = moment_pkt.strftime("%I").lstrip("0") or "12"
    am_pm = moment_pkt.strftime("%p")
    if moment_pkt.minute == 0:
        time_part = f"{hour_12} {am_pm}"
    else:
        time_part = f"{hour_12}:{moment_pkt.minute:02d} {am_pm}"

    return f"{date_part} at {time_part}"