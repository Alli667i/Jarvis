"""
config.py

The ONE place that reads settings from .env and decides which provider
is actually running for each swappable piece (speech-to-text, text-to-
speech). Nothing else in this app should read os.environ directly, and
nothing else should import stt_groq.py or tts_fish.py directly either —
everything goes through transcribe() and speak() below.

Why it works this way: to swap text-to-speech from Fish Audio to
something else later, you write a new file (say tts_elevenlabs.py) with
a speak() function shaped the same way as tts_fish.py's, add one line
below telling speak() to use it when TTS_ENGINE=elevenlabs, and flip
that one setting in .env. Nothing that currently calls config.speak()
needs to change at all. Same idea for transcription.

This file fails loudly and immediately if something required is missing
— at server startup, not in the middle of a conversation. A clear error
now is much better than a confusing failure later.

Expected .env keys:
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL      -- always required
    STT_ENGINE                                 -- optional, defaults to "groq"
    GROQ_API_KEY                               -- required if STT_ENGINE=groq
    TTS_ENGINE                                 -- optional, defaults to "fish"
    FISH_API_KEY                               -- required if TTS_ENGINE=fish
    FISH_REFERENCE_ID                          -- optional, which Fish voice to use

Note on TTS fallback: if the configured TTS_ENGINE fails for any reason,
speak() automatically retries once using edge-tts (no API key needed) so
a single provider outage never leaves a reply with no audio at all. This
happens regardless of TTS_ENGINE, and is logged to the console so it's
never a silent swap — see speak() below for exactly how.
"""

import os
from pathlib import Path


def _load_env_file(path: str = ".env") -> None:
    """
    Read a .env file and load its KEY=VALUE lines into the process's
    environment, so os.getenv() can see them afterward.

    Written by hand instead of using the python-dotenv library — a .env
    file is just lines of KEY=VALUE text, and reading that doesn't need
    a dependency to do it.

    Takes:
        path (str): where the .env file lives. Defaults to ".env" in
        whatever folder the server is started from.

    Returns: nothing.

    Can this fail: no. If the file doesn't exist, this does nothing and
    moves on quietly — someone might be setting real environment
    variables directly instead of using a .env file, which is still
    valid. Missing individual settings get caught later by _require()
    below, with a clear error naming exactly which one is missing.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()

            # Skip blank lines and comment lines.
            if not line or line.startswith("#"):
                continue

            # A valid line looks like KEY=VALUE. Skip anything that
            # doesn't have an "=" instead of crashing on a malformed line.
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Don't overwrite a variable that's already set in the real
            # environment. This lets a real "export SOME_KEY=..." take
            # priority over the .env file, useful for temporarily
            # overriding one value without editing the file.
            if key not in os.environ:
                os.environ[key] = value


def _require(name: str) -> str:
    """
    Read a required setting. Stops the server immediately with a clear
    message if it's missing, instead of letting some far-away piece of
    code fail confusingly later because a value it expected was empty.

    Takes:
        name (str): the exact key to look for, e.g. "GROQ_API_KEY".

    Returns:
        str: the value.

    Can this fail: yes, on purpose. Raises RuntimeError if the setting
    is missing or blank.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required setting '{name}'. Add it to your .env file "
            f"(or export it as a real environment variable) before "
            f"starting the server."
        )
    return value


def _optional(name: str, default: str) -> str:
    """
    Read a setting that's allowed to be missing, falling back to a
    default value if it is.

    Takes:
        name (str): the exact key to look for.
        default (str): what to use if it's not set.

    Returns:
        str: the value from .env, or `default` if it wasn't set.

    Can this fail: no.
    """
    return os.getenv(name, default)


# Load .env into the environment before anything below tries to read it.
_load_env_file()


# --- LLM settings -----------------------------------------------------
# No dispatch function needed here, unlike STT/TTS below. Swapping LLM
# providers (DeepSeek, Groq, OpenAI, whatever) doesn't change the shape
# of the call — they're all OpenAI-compatible endpoints, so it's just a
# different key/URL/model, not different code. llm_client.py reads these
# three values directly.
LLM_API_KEY = _require("LLM_API_KEY")
LLM_BASE_URL = _require("LLM_BASE_URL")
LLM_MODEL = _require("LLM_MODEL")


# --- Speech-to-text settings -------------------------------------------
STT_ENGINE = _optional("STT_ENGINE", "groq")

if STT_ENGINE == "groq":
    GROQ_API_KEY = _require("GROQ_API_KEY")
else:
    raise RuntimeError(
        f"STT_ENGINE is set to '{STT_ENGINE}' in .env, but config.py doesn't "
        f"have that engine wired up yet. Supported right now: groq. To add "
        f"a new one: write a file shaped like stt_groq.py, then add a branch "
        f"for it below in both this settings block and the transcribe() "
        f"function."
    )


# --- Text-to-speech settings --------------------------------------------
TTS_ENGINE = _optional("TTS_ENGINE", "fish")

# Where generated spoken-reply audio files get saved. Shared across every
# TTS provider so they all write to the same place instead of each
# picking their own — same DRY reasoning as time_utils.py's one shared
# PKT source.
AUDIO_OUTPUT_DIR = Path("static")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

if TTS_ENGINE == "fish":
    FISH_API_KEY = _require("FISH_API_KEY")
    # Optional — if not set, tts_fish.py decides what voice to fall back to.
    FISH_REFERENCE_ID = os.getenv("FISH_REFERENCE_ID")
elif TTS_ENGINE == "edge":
    pass  # edge-tts needs no API key or other settings
else:
    raise RuntimeError(
        f"TTS_ENGINE is set to '{TTS_ENGINE}' in .env, but config.py doesn't "
        f"have that engine wired up yet. Supported right now: fish. To add "
        f"a new one, e.g. elevenlabs: write tts_elevenlabs.py with a "
        f"speak(text, ...) function shaped like tts_fish.py's, then add a "
        f"branch for it below in both this settings block and the speak() "
        f"function."
    )


async def transcribe(audio_bytes: bytes) -> str:
    """
    Turn a recorded audio clip into text. This is the one function the
    rest of the app calls for speech-to-text — it doesn't need to know
    or care which actual provider is doing the work behind it.

    Takes:
        audio_bytes (bytes): the raw contents of a recorded audio clip
        (e.g. what a browser's MediaRecorder produces). No file needs to
        be saved to disk first — this goes straight to the provider.

    Returns:
        str: the transcribed text.

    Can this fail: yes. Whichever provider function actually runs (e.g.
    stt_groq.transcribe) can raise its own RuntimeError if the API call
    fails — bad key, network issue, bad audio, Groq's servers erroring.
    This function doesn't catch or retry that itself; it's up to
    whatever called transcribe() (the agent, or the voice endpoint in
    main.py) to decide what to say to the user when that happens.
    """
    if STT_ENGINE == "groq":
        from stt_groq import transcribe as groq_transcribe
        return await groq_transcribe(audio_bytes, GROQ_API_KEY)

    # Should never actually reach here — STT_ENGINE was already checked
    # above when this file loaded. Kept as a safety net rather than
    # assuming nothing can change it after the fact.
    raise ValueError(f"Unknown STT_ENGINE '{STT_ENGINE}'.")


async def speak(text: str) -> str:
    """
    Turn text into spoken audio. This is the one function the rest of
    the app calls for text-to-speech — it doesn't need to know or care
    which actual provider is doing the work behind it.

    Takes:
        text (str): what JARVIS should say out loud.

    Returns:
        str: file path to the generated audio file.

    Can this fail: yes, but there's a safety net. If the configured
    TTS_ENGINE fails for any reason, this automatically retries once
    using edge-tts (no API key needed, so it's very unlikely to fail for
    the same reason the primary engine just did). A warning is printed
    to the server console whenever this happens, so the fallback is
    never silent even though you won't hear a difference in the reply
    itself. If edge-tts ALSO fails — or if TTS_ENGINE was already set to
    "edge", meaning it just failed as the primary and there's nothing
    left to fall back to — that final error is what gets raised.
    """
    try:
        return await _speak_with_configured_engine(text)
    except Exception as primary_error:
        if TTS_ENGINE == "edge":
            raise  # edge-tts already ran and failed — no fallback left.

        print(
            f"[TTS] '{TTS_ENGINE}' failed ({primary_error}) — "
            f"falling back to edge-tts for this reply."
        )
        from tts_edge import speak as edge_speak
        return await edge_speak(text, AUDIO_OUTPUT_DIR)


async def _speak_with_configured_engine(text: str) -> str:
    """
    Internal — runs whichever provider TTS_ENGINE currently points to.
    Not meant to be called from outside this file; speak() above is the
    public entry point, and it's the one that adds fallback handling
    around this.

    Takes:
        text (str): what to say out loud.

    Returns:
        str: file path to the generated audio file.

    Can this fail: yes — whatever the underlying provider raises (see
    tts_fish.py / tts_edge.py) passes straight through, uncaught. speak()
    above is what catches it and decides whether to fall back.
    """
    if TTS_ENGINE == "fish":
        from tts_fish import speak as fish_speak
        return await fish_speak(text, FISH_API_KEY, FISH_REFERENCE_ID, AUDIO_OUTPUT_DIR)
    if TTS_ENGINE == "edge":
        from tts_edge import speak as edge_speak
        return await edge_speak(text, AUDIO_OUTPUT_DIR)

    # Should never actually reach here — TTS_ENGINE was already checked
    # when this file loaded. Kept as a safety net rather than assuming
    # nothing can change it after the fact.
    raise ValueError(f"Unknown TTS_ENGINE '{TTS_ENGINE}'.")