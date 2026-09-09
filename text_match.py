"""
text_match.py

Shared fuzzy name matching — used whenever the app needs to figure out
which calendar event or task a spoken name like "the dentist one" or
"the groceries task" refers to. One shared implementation so
calendar_tools.py and tasks_tools.py don't each write their own
slightly-different matching logic.

Uses only Python's built-in difflib — matching two short strings doesn't
need an extra dependency.
"""

from difflib import SequenceMatcher


def match_score(query: str, candidate: str) -> float:
    """
    How well does `candidate` match what the person said (`query`)?

    Takes:
        query (str): what the person said, e.g. "dentist".
        candidate (str): a real title to compare against, e.g.
        "Dentist Appointment - Dr. Khan".

    Returns:
        float: 0.0 (no relation at all) to 1.0 (exact match). If `query`
        appears as a clean substring of `candidate` (case-insensitive),
        this returns 0.9 — treated as a near-certain match, since
        referring to something by a short piece of its real name is
        exactly how a person actually talks. Anything else falls back to
        a general similarity score from difflib, which tolerates small
        differences in wording, typos, or word order.

    Can this fail: no — any two strings produce a score.
    """
    query_clean = query.strip().lower()
    candidate_clean = candidate.strip().lower()

    if not query_clean or not candidate_clean:
        return 0.0

    if query_clean == candidate_clean:
        return 1.0

    if query_clean in candidate_clean:
        return 0.9

    return SequenceMatcher(None, query_clean, candidate_clean).ratio()


def find_matches(query: str, candidates: list, get_name, threshold: float = 0.6) -> list:
    """
    Given a list of candidate items (calendar events, tasks, or anything
    with a name), find which ones plausibly match what the person said,
    best match first.

    Takes:
        query (str): what the person said, e.g. "the dentist one".
        candidates (list): the items to search through, e.g. a list of
        CalendarEvent objects.
        get_name: a function that takes one candidate and returns its
        display name as a string. Passed in rather than assuming every
        candidate has a `.title` attribute, so this works for any item
        shape without this file needing to know what a CalendarEvent or
        a task even is.
        threshold (float): minimum score (0.0-1.0) to count as a match
        at all. Default 0.6 is lenient enough to catch small wording
        differences without matching things that aren't really related.

    Returns:
        list: the candidates that scored at or above `threshold`, sorted
        best match first. Empty list if nothing matched well enough —
        that's the caller's signal to fall back to a fresh lookup, or to
        tell the person nothing was found.

    Can this fail: no.
    """
    scored = [(candidate, match_score(query, get_name(candidate))) for candidate in candidates]
    matching = [(candidate, score) for candidate, score in scored if score >= threshold]
    matching.sort(key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in matching]