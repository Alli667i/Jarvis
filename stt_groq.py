"""
stt_groq.py

Speech-to-text using Groq's hosted Whisper API. This is the provider
that runs when STT_ENGINE=groq in .env (see config.py). Nothing else in
the app should import this file directly — everything goes through
config.transcribe(), which is what decides that "groq" means this file.

To add a different STT provider later — a different hosted API, or a
local model running on-device — write a new file with a transcribe()
function shaped the same way as this one (takes audio bytes, returns
text, raises RuntimeError on failure), then wire it into config.py.
"""

import httpx

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# The "turbo" variant trades a little accuracy for speed — the right
# trade for a voice assistant that needs to reply quickly, not transcribe
# for a legal record.
GROQ_MODEL = "whisper-large-v3-turbo"


async def transcribe(audio_bytes: bytes, api_key: str) -> str:
    """
    Send a recorded audio clip to Groq and get back the words that were
    said.

    Takes:
        audio_bytes (bytes): raw contents of a recorded audio clip, e.g.
        webm/opus straight out of a browser's MediaRecorder. Sent as-is
        — nothing gets saved to disk first.
        api_key (str): Groq API key (config.GROQ_API_KEY).

    Returns:
        str: the transcribed text, whitespace trimmed. Empty string if
        Groq heard nothing (e.g. silence was recorded).

    Can this fail: yes.
        - Raises RuntimeError if Groq responds with anything other than
          success — bad key, rate limit, bad audio, server error on
          Groq's end. The error message includes Groq's own response
          text so the actual cause is visible, not hidden.
        - Raises RuntimeError if the request can't reach Groq at all
          (network down, timeout, DNS failure). These are normally a
          different exception type from the httpx library — caught here
          and re-raised as RuntimeError so callers only ever need to
          handle one error type from this function, not several
          different network-library exceptions.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.webm", audio_bytes, "audio/webm")},
                data={"model": GROQ_MODEL, "language": "en"},
            )
    except httpx.RequestError as network_error:
        raise RuntimeError(
            f"Could not reach Groq for transcription: {network_error}"
        ) from network_error

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq transcription failed ({response.status_code}): {response.text}"
        )

    return response.json().get("text", "").strip()