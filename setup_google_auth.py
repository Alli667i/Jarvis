"""
setup_google_auth.py

ONE-TIME SETUP SCRIPT. Run this on a normal computer with a real
browser -- NOT on the Samsung A40/Termux server. The A40 is headless
and can't open a browser to complete Google's login screen (see
architecture.md's "Google OAuth Setup" section for the full context).

What it does: uses credentials.json (downloaded from Google Cloud
Console) to open a browser, let you log into your Google account, and
approve the permissions this app needs. Saves the result as token.json.

Imports SCOPES and TOKEN_PATH directly from google_auth.py rather than
redefining them here -- this guarantees the token this script produces
always asks for exactly the permissions the rest of the app actually
uses, with no way for the two to quietly drift out of sync.

After running this successfully:
  1. Copy the token.json this creates onto the Samsung A40, into the
     same project folder as the rest of the app.
  2. This script itself never needs to run on the A40 -- only its
     OUTPUT (token.json) goes there.
  3. If google_auth.py's SCOPES list ever changes (e.g. a future
     module needs a new permission), this whole process needs to be
     redone -- a saved token only carries the scopes it was originally
     granted with.

Usage:
    python setup_google_auth.py
"""

import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

from google_auth import SCOPES, TOKEN_PATH

CREDENTIALS_PATH = "credentials.json"


def main() -> None:
    """
    Run the one-time Google OAuth login flow and save the resulting
    token.

    Takes: nothing. Reads CREDENTIALS_PATH from disk and SCOPES from
    google_auth.py.

    Returns: nothing -- writes TOKEN_PATH ("token.json") on success.

    Can this fail: yes, and this is meant to be run interactively by a
    person watching it happen, so failures print a clear message and
    stop (sys.exit(1)) rather than being caught and retried silently --
    there's nothing meaningful to retry automatically here; a failed
    login needs a person to notice and act.
        - Exits if credentials.json isn't present, with instructions
          for exactly where to get it.
        - If the browser login flow itself fails or is cancelled,
          whatever google_auth_oauthlib raises is left to surface
          directly -- its own error message is already clear, wrapping
          it wouldn't add anything.
    """
    if not os.path.exists(CREDENTIALS_PATH):
        print(
            f"No {CREDENTIALS_PATH} found in this folder.\n\n"
            f"Get it from Google Cloud Console:\n"
            f"  1. https://console.cloud.google.com/ -- create or select a project\n"
            f"  2. Enable these three APIs: Gmail API, Google Calendar API, Google Tasks API\n"
            f"  3. Credentials -> Create Credentials -> OAuth client ID -> Desktop app\n"
            f"  4. Download the JSON, save it as '{CREDENTIALS_PATH}' right here, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.exists(TOKEN_PATH):
        print(f"Note: {TOKEN_PATH} already exists here and will be overwritten.\n")

    print("Requesting these permissions:")
    for scope in SCOPES:
        print(f"  - {scope}")
    print("\nOpening a browser for you to log in and approve access...\n")

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    credentials = flow.run_local_server(port=0)

    with open(TOKEN_PATH, "w") as token_file:
        token_file.write(credentials.to_json())

    print(f"\nDone -- {TOKEN_PATH} created.")
    print(f"Next step: copy {TOKEN_PATH} onto the Samsung A40, into the same project folder.")


if __name__ == "__main__":
    main()