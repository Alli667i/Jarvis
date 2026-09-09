"""
tts_edge.py

Text-to-speech using Microsoft Edge's TTS engine (via the edge-tts
library). Needs no API key or account — that's what makes it a solid
fallback: it can't fail for "your key expired" or "your account ran out
of credit" reasons the way a paid provider can.

This file works two ways:
  1. As a normal, selectable engine (TTS_ENGINE=edge in .env) — nothing
     fallback-specific lives in this file itself, it's a provider file
     shaped just like tts_fish.py.
  2. As the automatic fallback config.py reaches for if the configured
     primary engine fails. That fallback behavior lives in config.py,
     not here — this file doesn't know or care that it might be used
     as a fallback.
"""

import uuid
from pathlib import Path

import edge_tts

# The voice, speed, and pitch JARVIS speaks with here. Matches what the
# old codebase used — a calm, slightly slower, slightly lower voice.
VOICE = "en-GB-RyanNeural"
RATE = "-3%"
PITCH = "-8Hz"


async def speak(text: str, output_dir: Path) -> str:
    """
    Turn text into a spoken mp3 file using edge-tts.

    Takes:
        text (str): what to say out loud.
        output_dir (Path): folder to save the generated mp3 into
        (config.AUDIO_OUTPUT_DIR — shared with every other TTS provider).

    Returns:
        str: path to the generated mp3 file on disk.

    Can this fail: yes — raises RuntimeError if edge-tts can't reach
    Microsoft's service, or synthesis fails for any other reason. Any
    partially-written file is deleted before the error is raised. This
    function doesn't fall back to anything else on its own — if it's
    being used AS the fallback and it also fails, there's nothing left
    to fall back to, and that's the caller's problem to handle.
    """
    output_path = output_dir / f"{uuid.uuid4().hex}.mp3"

    try:
        communicate = edge_tts.Communicate(text=text, voice=VOICE, rate=RATE, pitch=PITCH)
        await communicate.save(str(output_path))
    except Exception as edge_error:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"edge-tts synthesis failed: {edge_error}") from edge_error

    return str(output_path)