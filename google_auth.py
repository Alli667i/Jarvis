"""
google_auth.py

The ONE place that handles Google OAuth credentials — loading the saved
token and refreshing it when it's expired. calendar_tools.py and
tasks_tools.py both use get_credentials() from here instead of each
reading token.json and refreshing credentials themselves — same DRY
reasoning as every other shared file in this app.

This file does NOT handle the one-time browser consent step — that
can't run on this headless Termux server at all (see architecture.md).
It's done once on a normal computer with a real browser, and the
resulting token.json gets copied onto this server afterward. This file
only ever expects token.json to already exist here.
"""

import os

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Kept as one combined scope list even though Module 1 only uses Calendar
# and Tasks — Gmail support is a later module, and requesting its scope
# now means the one-time browser login won't need to be redone when that
# module gets built. If this list ever changes, the one-time login DOES
# need to be redone, since a saved token only carries the scopes it was
# originally granted with.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/tasks",
]

TOKEN_PATH = "token.json"


def get_credentials() -> Credentials:
    """
    Load this app's saved Google credentials, refreshing the access
    token first if it's expired.

    Takes: nothing.

    Returns: valid Credentials, ready to hand to
    googleapiclient.discovery.build().

    Can this fail: yes, on purpose — refuses to guess or silently do
    nothing when auth isn't set up correctly.
        - Raises RuntimeError if token.json doesn't exist at all. Means
          the one-time browser login (see architecture.md) hasn't been
          done yet, or token.json wasn't copied onto this server.
        - Raises RuntimeError if the token has expired and there's no
          refresh token to renew it with. Means the one-time browser
          login needs to be redone from scratch.
    A successful refresh is saved back to token.json immediately, so a
    renewed token survives a server restart too.
    """
    if not os.path.exists(TOKEN_PATH):
        raise RuntimeError(
            f"No {TOKEN_PATH} found. Google Calendar/Tasks need a one-time "
            f"login on a normal computer first — see architecture.md for the "
            f"setup steps — then copy the resulting {TOKEN_PATH} onto this "
            f"server."
        )

    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    except ValueError as parse_error:
        # Google's library requires token.json to already contain a
        # refresh_token to even parse it successfully — so a malformed or
        # hand-edited file shows up here as a ValueError, not later as a
        # missing-refresh-token problem. Converted into the same clear
        # guidance as the other auth failure modes below.
        raise RuntimeError(
            f"{TOKEN_PATH} exists but isn't in the format Google's library "
            f"expects ({parse_error}). It may be corrupted or hand-edited. "
            f"The one-time browser login needs to be redone — see "
            f"architecture.md."
        ) from parse_error

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PATH, "w") as token_file:
                token_file.write(creds.to_json())
        else:
            # In practice this is hard to reach through a file that parsed
            # successfully above (Google's library already requires a
            # refresh_token to parse at all) — kept as a defensive fallback
            # rather than assumed impossible.
            raise RuntimeError(
                f"{TOKEN_PATH} exists but isn't usable and can't be "
                f"refreshed automatically. The one-time browser login needs "
                f"to be redone — see architecture.md."
            )

    return creds