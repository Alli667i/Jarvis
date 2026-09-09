"""
main.py

The FastAPI server tying everything together: receives voice (or, for
testing, typed text), transcribes it, hands it to agent.handle_message(),
speaks the reply, and serves the generated audio back to the browser.

SCOPE OF THIS VERSION: single device only. One browser tab captures its
own mic AND plays the reply, in one direct request/response cycle. No
WebSocket, no broadcast-to-multiple-devices mechanism yet. That's a
deliberate staged decision, not an oversight -- see architecture.md's
"Output broadcasting" section and progress.md's Next Up. Proving the
whole voice + calendar/tasks loop works end to end on the simplest
possible setup comes before adding a second device into the mix.

main.py itself holds no conversation state -- every request just calls
into agent.py, which owns all of that (see agent.py's module docstring).
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

import config
import agent


app = FastAPI()

# Serves everything in config.AUDIO_OUTPUT_DIR (e.g. static/abc123.mp3)
# at /static/abc123.mp3, so the browser can play generated replies.
app.mount("/static", StaticFiles(directory=str(config.AUDIO_OUTPUT_DIR)), name="static")

# How many recent reply audio files to keep on disk. Older ones get
# deleted every time a new one is generated -- without this, the static
# folder grows by one mp3 per reply forever, which matters on a
# storage-constrained phone server.
MAX_KEPT_AUDIO_FILES = 20

# The two frontend files (see index.html / app.js). Expected right next
# to main.py, at the project root -- not in a templates/ or static/
# subfolder.
INDEX_HTML_PATH = Path("index.html")
APP_JS_PATH = Path("app.js")


class TextMessage(BaseModel):
    """Request body for POST /chat/text."""
    text: str


class ChatReply(BaseModel):
    """
    Response body for both /chat/text and /chat/voice.

    Fields:
        reply_text (str): what JARVIS said, as text.
        audio_url (str | None): where the browser can play the spoken
        version from. None if speech synthesis itself failed -- the
        text reply still comes through either way; a voice assistant
        that goes completely silent on a TTS hiccup is worse than one
        that occasionally falls back to a written reply.
    """
    reply_text: str
    audio_url: str | None = None


def _cleanup_old_audio() -> None:
    """
    Delete old generated reply audio files, keeping only the
    MAX_KEPT_AUDIO_FILES most recently created.

    Takes: nothing. Returns: nothing.

    Can this fail: no -- uses unlink(missing_ok=True), so a file that's
    already gone (e.g. deleted by hand) is silently skipped rather than
    raising.
    """
    audio_files = sorted(config.AUDIO_OUTPUT_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
    for old_file in audio_files[:-MAX_KEPT_AUDIO_FILES]:
        old_file.unlink(missing_ok=True)


async def _speak_reply(text: str) -> str | None:
    """
    Turn a reply into spoken audio and return the browser-playable URL.

    Takes:
        text (str): what JARVIS is saying.

    Returns:
        str | None: "/static/<filename>.mp3", or None if speech
        synthesis failed entirely.

    Can this fail: no exception raised here -- config.speak() already
    retries via its own Fish-then-edge-tts fallback (see config.py); if
    BOTH engines fail, that RuntimeError is caught here and turned into
    a None audio_url rather than breaking the whole reply. The text
    reply always gets through regardless of whether speech synthesis
    worked.
    """
    try:
        audio_path = await config.speak(text)
    except RuntimeError as error:
        print(f"[main] Speech synthesis failed, replying with text only: {error}")
        return None

    _cleanup_old_audio()
    filename = Path(audio_path).name
    return f"/static/{filename}"


@app.post("/chat/text", response_model=ChatReply)
async def chat_text(message: TextMessage) -> ChatReply:
    """
    Debugging/testing convenience: send typed text, get back the same
    reply shape a voice turn would produce (text + a spoken audio URL).
    Not JARVIS's normal interface -- a way to exercise the whole
    pipeline, or just agent.py, without needing a working microphone.

    Takes (request body):
        text (str): what to say to JARVIS, typed instead of spoken.

    Returns:
        ChatReply.

    Can this fail: an unhandled exception here means a real bug in
    agent.py's own logic (see agent.py's docstring on handle_message) --
    deliberately not caught, so it surfaces as a clear error during
    development rather than being hidden.
    """
    reply_text = await agent.handle_message(message.text)
    audio_url = await _speak_reply(reply_text)
    return ChatReply(reply_text=reply_text, audio_url=audio_url)


@app.post("/chat/voice", response_model=ChatReply)
async def chat_voice(audio: UploadFile = File(...)) -> ChatReply:
    """
    The main voice endpoint. Upload a recorded audio clip, get back
    JARVIS's reply as text plus a URL to the spoken version.

    Takes (multipart upload):
        audio: the recorded audio file (e.g. webm/opus from a browser's
        MediaRecorder).

    Returns:
        ChatReply. If nothing intelligible was heard (silence, or
        transcription failed), reply_text explains that directly rather
        than sending an empty message into the agent for no reason.

    Can this fail: transcription failing (bad audio, Groq unreachable)
    is caught here and turned into a generic spoken reply -- the real
    error is logged to the console for debugging, never spoken aloud
    or included in reply_text; reading a raw exception message out loud
    would be a worse experience than a plain "try again". An unhandled
    exception past that point means a real bug in agent.py, same as
    chat_text() above -- not caught here on purpose.
    """
    audio_bytes = await audio.read()

    try:
        user_text = await config.transcribe(audio_bytes)
    except RuntimeError as error:
        print(f"[main] Transcription failed: {error}")
        reply_text = "Sorry, I couldn't understand that -- could you try again?"
        audio_url = await _speak_reply(reply_text)
        return ChatReply(reply_text=reply_text, audio_url=audio_url)

    if not user_text.strip():
        reply_text = "I didn't catch that -- could you say it again?"
        audio_url = await _speak_reply(reply_text)
        return ChatReply(reply_text=reply_text, audio_url=audio_url)

    reply_text = await agent.handle_message(user_text)
    audio_url = await _speak_reply(reply_text)
    return ChatReply(reply_text=reply_text, audio_url=audio_url)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """
    Serves the real JARVIS console page (index.html).

    Takes: nothing.

    Returns:
        str: index.html's contents, or a short fallback message if the
        file isn't where it's expected -- e.g. main.py started from the
        wrong working directory. A clear message here beats a raw
        FileNotFoundError traceback for something this easy to get
        wrong when running the server by hand.

    Can this fail: no unhandled failure -- the missing-file case is
    handled explicitly rather than left to raise.
    """
    if not INDEX_HTML_PATH.exists():
        return "<h1>JARVIS backend running -- index.html not found next to main.py.</h1>"
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/app.js")
async def app_js() -> FileResponse:
    """
    Serves app.js with the correct JavaScript content type -- index.html
    loads it via <script src="app.js">, which resolves to this route
    since index.html is served from "/".

    Takes: nothing.

    Returns:
        FileResponse. media_type is set explicitly rather than left to
        be guessed, so this is never accidentally served as plain text
        (which would make the browser refuse to execute it).

    Can this fail: yes -- raises a clean HTTPException(404) if
    APP_JS_PATH doesn't exist. This is checked explicitly rather than
    relying on FileResponse itself to handle a missing file gracefully
    — tested directly, and it does not: an absent file makes
    FileResponse raise a raw, unhandled RuntimeError instead of a
    proper 404, which would surface to the browser as a bare server
    crash rather than a clear "not found."
    """
    if not APP_JS_PATH.exists():
        raise HTTPException(status_code=404, detail="app.js not found next to main.py.")
    return FileResponse(APP_JS_PATH, media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    print("JARVIS is online. Navigate to http://localhost:8001")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)