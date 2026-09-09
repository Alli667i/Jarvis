"""
tts_fish.py

Text-to-speech using Fish Audio's cloud API. This is the provider that
runs when TTS_ENGINE=fish in .env (see config.py). Nothing else in the
app should import this file directly — everything goes through
config.speak(), which is what decides "fish" means this file.

To add a different TTS provider later (ElevenLabs, a local model, etc.),
write a new file with a speak() function shaped the same way as this
one (takes text + settings, returns a file path, raises RuntimeError on
failure), then wire it into config.py.
"""

import uuid
from pathlib import Path

from fish_audio_sdk import Session, TTSRequest
from fish_audio_sdk.exceptions import HttpCodeErr
import aiofiles

# Fish Audio's synthesis backend. Kept as one named constant here rather
# than buried inline, so it's easy to find if Fish ever renames or
# deprecates it.
FISH_BACKEND = "s2.1-pro-free"

# Used only if FISH_REFERENCE_ID isn't set in .env. This is the voice
# already proven to work in the previous version of this app.
DEFAULT_REFERENCE_ID = "c231dcd3116a4c0984e3bced753c1274"


async def speak(text: str, api_key: str, reference_id: str | None, output_dir: Path) -> str:
    """
    Turn text into a spoken mp3 file using Fish Audio.

    Takes:
        text (str): what to say out loud.
        api_key (str): Fish Audio API key (config.FISH_API_KEY).
        reference_id (str | None): which Fish voice to use. If None
        (not set in .env), falls back to DEFAULT_REFERENCE_ID above.
        output_dir (Path): folder to save the generated mp3 into
        (config.AUDIO_OUTPUT_DIR — shared with any other TTS provider,
        not something this file decides on its own).

    Returns:
        str: path to the generated mp3 file on disk.

    Can this fail: yes.
        - Raises RuntimeError if Fish Audio rejects the request — bad
          key, bad reference_id, rate limit. The error message includes
          the status code and Fish's own message.
        - Raises RuntimeError for anything else that goes wrong during
          synthesis — network issue, connection drop mid-stream, etc.
        - Either way, any partially-written audio file is deleted before
          the error is raised, so a failed request never leaves a
          broken mp3 behind.
        - This function does NOT fall back to a different TTS engine on
          failure. If a fallback engine is ever wanted, that decision
          belongs in config.py (the one place that knows about more
          than one provider), not here.
    """
    voice = reference_id or DEFAULT_REFERENCE_ID
    output_path = output_dir / f"{uuid.uuid4().hex}.mp3"

    session = Session(api_key)
    request = TTSRequest(text=text, reference_id=voice, format="mp3")

    try:
        async with aiofiles.open(output_path, "wb") as audio_file:
            async for chunk in session.tts.awaitable(request, backend=FISH_BACKEND):
                await audio_file.write(chunk)
    except HttpCodeErr as api_error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Fish Audio rejected the request ({api_error.status}): {api_error.message}"
        ) from api_error
    except Exception as other_error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Fish Audio synthesis failed: {other_error}") from other_error

    return str(output_path)