"""
retry.py

The ONE place that defines "try again once before giving up" for calls
to things outside this app (the LLM, Google Calendar, Google Tasks, and
whatever gets added later). Every external API call in this app should
go through retry_once() instead of writing its own try/except-and-retry
logic — that's what keeps the retry POLICY (try once more, then report
plainly) consistent everywhere instead of quietly drifting to "3 tries
here, 2 tries there" as different files get written at different times.
"""

from typing import Awaitable, Callable, TypeVar
import asyncio

T = TypeVar("T")


async def retry_once(operation: Callable[[], Awaitable[T]], description: str) -> T:
    """
    Run an async operation. If it fails, try it exactly one more time
    before giving up.

    Takes:
        operation: a no-argument async function that does the actual
        work, e.g. `lambda: some_client.do_the_call(...)`. It takes no
        arguments so this function stays generic — it doesn't need to
        know anything about what's being called.
        description (str): a short, human-readable name for what this
        operation was trying to do, e.g. "Calling the LLM" or "Fetching
        today's calendar events". Used only in the error message if both
        attempts fail, so whoever reads that error knows what broke.

    Returns:
        Whatever `operation()` returns, from whichever attempt succeeded.

    Can this fail: yes, on purpose. If the first attempt fails, this
    tries once more automatically. If the second attempt ALSO fails,
    raises RuntimeError naming the operation and including the second
    attempt's underlying error — by then, retrying silently again isn't
    appropriate, the caller needs to know and decide what to tell the
    user.
    """
    try:
        return await operation()
    except Exception as first_error:
        print(f"[retry] {description} failed once ({first_error}) — retrying...")
        try:
            return await operation()
        except Exception as second_error:
            raise RuntimeError(
                f"{description} failed twice in a row: {second_error}"
            ) from second_error


async def retry_blocking(blocking_operation: Callable[[], T], description: str) -> T:
    """
    Same policy as retry_once() above — try once, retry once, then raise
    a clear RuntimeError — but for an ordinary SYNCHRONOUS function
    instead of an async one. Needed for libraries that don't support
    async/await at all, like Google's API client (used by
    calendar_tools.py / tasks_tools.py).

    Calling a blocking, network-waiting function directly from async
    code would freeze the entire server while it waits — nothing else
    could be handled in the meantime. This runs it in a background
    thread instead (via asyncio.to_thread), so the rest of the app keeps
    responding, then applies the exact same retry_once() policy around
    that.

    Takes:
        blocking_operation: a no-argument, ordinary (non-async) function
        that does the actual work and returns a result.
        description (str): same as retry_once() — a short human-readable
        name for the error message if both attempts fail.

    Returns: whatever blocking_operation() returns.

    Can this fail: yes — same failure behavior as retry_once(): raises
    RuntimeError if both attempts fail.
    """
    return await retry_once(lambda: asyncio.to_thread(blocking_operation), description)