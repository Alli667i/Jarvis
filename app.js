/*
  app.js

  Everything the JARVIS console page actually does: tap the mic button
  to start/stop recording, send the recording to the backend, play back
  the reply, and drive the signal trace from REAL audio data (the live
  mic while listening, the actual playing reply while speaking) rather
  than a canned animation.

  Kept as one plain script, no build step, no framework -- this runs
  straight off disk on a Termux server with no bundler available, and
  the whole page is small enough that a framework would add more
  complexity than it removes. See architecture.md's "Frontend design
  system" section for the design decisions this file implements; this
  file is the implementation, that section is the reference if it ever
  needs to be rebuilt.

  Testing note: this file is loaded directly by a real browser normally,
  but its logic is also exercised by test/app.test.js under Node with
  jsdom + hand-written stubs for MediaRecorder/AudioContext/fetch (none
  of which jsdom implements) -- see that file for what's actually
  verified without a real browser.
*/

const STATE = {
  STANDBY: "STANDBY",
  LISTENING: "LISTENING",
  PROCESSING: "PROCESSING",
  SPEAKING: "SPEAKING",
};

// How many bars the trace renders. Matches AnalyserNode.fftSize below
// (fftSize / 2 = frequency bin count) -- deliberately kept equal so
// every bin maps to exactly one bar, no re-binning needed.
const BAR_COUNT = 32;
const FFT_SIZE = BAR_COUNT * 2;

// Tap-to-start/tap-to-stop has no natural upper bound on recording
// length the way push-to-talk does (releasing a held button). This is
// a safety net against leaving the mic open indefinitely if the user
// gets distracted mid-recording -- auto-stops and sends whatever was
// recorded so far.
const MAX_RECORDING_MS = 60000;

// --- Module-level state -----------------------------------------------
// A single page, a single ongoing interaction -- no need for anything
// fancier than plain variables here, matching the same "this app is
// one conversation, not a multi-session server" reasoning used
// throughout the backend (see agent.py).

let currentState = STATE.STANDBY;

let mediaRecorder = null;
let recordedChunks = [];
let micStream = null;
let micAudioContext = null;
let micAnalyser = null;
let recordingTimeoutId = null;

let playbackAudioContext = null;
let playbackAnalyser = null;
let currentAudioElement = null;

let traceAnimationFrameId = null;
let traceAnimationTick = 0;

// --- DOM references -----------------------------------------------------
// Looked up once at load time. Exposed on `window` (not just module-local
// consts) specifically so test/app.test.js can reach them without this
// file needing any test-only export machinery.

const dom = {
  connectionDot: document.getElementById("connection-dot"),
  stateCode: document.getElementById("state-code"),
  statusLabel: document.getElementById("status-label"),
  caption: document.getElementById("caption"),
  micButton: document.getElementById("mic-button"),
  trace: document.getElementById("trace"),
};

/**
 * Build the BAR_COUNT bar elements inside the trace container once, at
 * page load. Returns the list of bar elements so the render loop can
 * update their heights directly without re-querying the DOM every
 * frame.
 *
 * Takes: nothing.
 * Returns: HTMLElement[] -- the bar divs, in left-to-right order.
 * Can this fail: no.
 */
function buildTraceBars() {
  const bars = [];
  for (let i = 0; i < BAR_COUNT; i++) {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = "2px";
    dom.trace.appendChild(bar);
    bars.push(bar);
  }
  return bars;
}

const traceBars = buildTraceBars();

/**
 * Move the page into a new state: updates every piece of UI that
 * depends on state (labels, mic button enabled/disabled) and starts
 * the matching trace animation.
 *
 * Takes:
 *   newState (string): one of the STATE constants above.
 * Returns: nothing.
 * Can this fail: no.
 */
function setState(newState) {
  currentState = newState;
  dom.stateCode.textContent = newState;
  dom.statusLabel.textContent = newState;
  dom.micButton.disabled = newState === STATE.PROCESSING || newState === STATE.SPEAKING;
  dom.micButton.setAttribute(
    "aria-label",
    newState === STATE.LISTENING ? "Stop and send" : "Start listening"
  );
}

/**
 * Render one frame of the trace for the CURRENT state. Called on every
 * animation frame by startTraceAnimation() below -- never call this
 * directly from anywhere else, it always acts on whatever
 * `currentState` is right now.
 *
 * Takes: nothing (reads module state).
 * Returns: nothing.
 * Can this fail: no -- every branch has a fallback synthetic pattern,
 * so even if an analyser isn't ready yet for some reason, the trace
 * still renders something reasonable instead of throwing.
 */
function renderTraceFrame() {
  traceAnimationTick++;

  if (currentState === STATE.LISTENING && micAnalyser) {
    renderFromAnalyser(micAnalyser);
  } else if (currentState === STATE.SPEAKING && playbackAnalyser) {
    renderFromAnalyser(playbackAnalyser);
  } else if (currentState === STATE.PROCESSING) {
    renderSearchingPattern();
  } else {
    renderIdlePattern();
  }

  traceAnimationFrameId = requestAnimationFrame(renderTraceFrame);
}

/**
 * Draw the trace from a real AnalyserNode's current frequency data --
 * used for both LISTENING (live mic) and SPEAKING (playing reply
 * audio). Same function for both; the only difference is which
 * analyser is passed in.
 *
 * Takes:
 *   analyser (AnalyserNode): must have fftSize already set to
 *   FFT_SIZE, so its frequencyBinCount matches BAR_COUNT exactly.
 * Returns: nothing.
 * Can this fail: no.
 */
function renderFromAnalyser(analyser) {
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  for (let i = 0; i < BAR_COUNT; i++) {
    const amplitude = data[i] / 255; // 0..1
    traceBars[i].style.height = `${4 + amplitude * 40}px`;
  }
}

/**
 * Synthetic trace pattern for STANDBY -- a near-flat line with a
 * barely-there flicker, since there's no real signal to show when
 * nothing is happening.
 *
 * Takes: nothing. Returns: nothing. Can this fail: no.
 */
function renderIdlePattern() {
  for (let i = 0; i < BAR_COUNT; i++) {
    const h = 2 + Math.sin(traceAnimationTick * 0.02 + i * 0.3) * 1;
    traceBars[i].style.height = `${h}px`;
  }
}

/**
 * Synthetic trace pattern for PROCESSING -- a searching, irregular
 * motion, deliberately different in character from the idle flicker
 * (not just a color change) since something IS actively happening
 * even though there's no real signal to visualize (the LLM and any
 * tool calls are in flight on the server).
 *
 * Takes: nothing. Returns: nothing. Can this fail: no.
 */
function renderSearchingPattern() {
  for (let i = 0; i < BAR_COUNT; i++) {
    const phase = (traceAnimationTick * 0.08 + i * 0.4) % (Math.PI * 2);
    const h = 4 + Math.abs(Math.sin(phase)) * 18;
    traceBars[i].style.height = `${h}px`;
  }
}

/**
 * Start the trace's render loop. Safe to call more than once -- won't
 * stack multiple loops, since renderTraceFrame() only ever schedules
 * its own next frame.
 *
 * Takes: nothing. Returns: nothing. Can this fail: no.
 */
function startTraceAnimation() {
  if (traceAnimationFrameId !== null) {
    return; // already running
  }
  renderTraceFrame();
}

/**
 * Show a connection/error problem on the small status dot, and put an
 * explanation in the caption. Used for things the user needs to know
 * about but that aren't a normal spoken reply -- e.g. the mic
 * permission being denied, or the server being unreachable.
 *
 * Takes:
 *   message (string): shown in the caption area.
 * Returns: nothing.
 * Can this fail: no.
 */
function showError(message) {
  dom.connectionDot.classList.add("error");
  dom.caption.textContent = message;
  setState(STATE.STANDBY);
}

/**
 * Clear any error indication set by showError() above -- called at the
 * start of a new attempt, so an old error doesn't linger visually once
 * things are working again.
 *
 * Takes: nothing. Returns: nothing. Can this fail: no.
 */
function clearError() {
  dom.connectionDot.classList.remove("error");
}

/**
 * Begin recording from the microphone: requests mic access, sets up a
 * live AnalyserNode so the trace can react to real input, and starts
 * a MediaRecorder capturing the audio. Also arms the MAX_RECORDING_MS
 * safety timeout.
 *
 * Takes: nothing.
 * Returns: Promise<void>.
 * Can this fail: yes -- if getUserMedia is denied or unavailable, this
 * catches that and shows a clear error via showError() rather than
 * leaving the page stuck or throwing an unhandled rejection. State
 * stays STANDBY on failure.
 */
async function startListening() {
  clearError();

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    showError("Couldn't access the microphone -- check permissions and try again.");
    return;
  }

  micAudioContext = new AudioContext();
  const source = micAudioContext.createMediaStreamSource(micStream);
  micAnalyser = micAudioContext.createAnalyser();
  micAnalyser.fftSize = FFT_SIZE;
  source.connect(micAnalyser);

  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";
  mediaRecorder = new MediaRecorder(micStream, { mimeType });
  recordedChunks = [];
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  mediaRecorder.start();

  recordingTimeoutId = setTimeout(() => {
    if (currentState === STATE.LISTENING) {
      stopListeningAndSend();
    }
  }, MAX_RECORDING_MS);

  dom.caption.textContent = "";
  setState(STATE.LISTENING);
}

/**
 * Stop the current recording, tear down the mic stream/analyser
 * (microphones should never stay open longer than needed), and send
 * the recorded audio to the backend.
 *
 * Takes: nothing.
 * Returns: Promise<void>.
 * Can this fail: yes -- network/server failures are caught and shown
 * via showError(), returning the page to STANDBY rather than leaving
 * it stuck in PROCESSING forever.
 */
async function stopListeningAndSend() {
  if (recordingTimeoutId !== null) {
    clearTimeout(recordingTimeoutId);
    recordingTimeoutId = null;
  }

  const recordingStopped = new Promise((resolve) => {
    mediaRecorder.onstop = resolve;
  });
  mediaRecorder.stop();
  await recordingStopped;

  micStream.getTracks().forEach((track) => track.stop());
  micAudioContext.close();
  micStream = null;
  micAudioContext = null;
  micAnalyser = null;

  setState(STATE.PROCESSING);

  const audioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  let data;
  try {
    const response = await fetch("/chat/voice", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}`);
    }
    data = await response.json();
  } catch (error) {
    showError("Couldn't reach JARVIS -- check the server and try again.");
    return;
  }

  await speakReply(data.reply_text, data.audio_url);
}

/**
 * Show the reply's text and, if audio synthesis succeeded server-side,
 * play it back with the trace reacting to the real playing audio. If
 * synthesis failed (audio_url is null -- see main.py's _speak_reply()),
 * the text still gets shown, just without spoken audio.
 *
 * Takes:
 *   text (string): what JARVIS said, shown in the caption the instant
 *   playback starts (see architecture.md's "Audio/text sync" note --
 *   this is the simple, backend-compatible version, not
 *   sentence-by-sentence reveal).
 *   audioUrl (string | null): where to play the reply from, or null if
 *   speech synthesis failed entirely.
 * Returns: Promise<void>.
 * Can this fail: no unhandled failure -- if playback itself throws
 * (e.g. a browser autoplay restriction), that's caught and the page
 * still returns to STANDBY with the text already shown, rather than
 * getting stuck in SPEAKING with silent audio.
 */
async function speakReply(text, audioUrl) {
  if (!audioUrl) {
    dom.caption.textContent = text;
    setState(STATE.STANDBY);
    return;
  }

  currentAudioElement = new Audio(audioUrl);
  playbackAudioContext = new AudioContext();
  const source = playbackAudioContext.createMediaElementSource(currentAudioElement);
  playbackAnalyser = playbackAudioContext.createAnalyser();
  playbackAnalyser.fftSize = FFT_SIZE;
  source.connect(playbackAnalyser);
  playbackAnalyser.connect(playbackAudioContext.destination);

  const playbackFinished = new Promise((resolve) => {
    currentAudioElement.onended = resolve;
    currentAudioElement.onerror = resolve;
  });

  dom.caption.textContent = text;
  setState(STATE.SPEAKING);

  try {
    await currentAudioElement.play();
  } catch (error) {
    // Autoplay was blocked, or playback failed for some other reason.
    // The text is already shown -- just fall back to STANDBY instead
    // of hanging in SPEAKING with no audio actually playing.
    cleanUpPlayback();
    setState(STATE.STANDBY);
    return;
  }

  await playbackFinished;
  cleanUpPlayback();
  setState(STATE.STANDBY);
}

/**
 * Tear down the playback AudioContext/analyser after a reply finishes
 * (or fails) playing, so resources don't accumulate turn after turn.
 *
 * Takes: nothing. Returns: nothing. Can this fail: no.
 */
function cleanUpPlayback() {
  if (playbackAudioContext) {
    playbackAudioContext.close();
  }
  playbackAudioContext = null;
  playbackAnalyser = null;
  currentAudioElement = null;
}

/**
 * The mic button's click handler -- routes to start or stop depending
 * on the current state. A no-op if clicked while PROCESSING/SPEAKING,
 * though the button is disabled then anyway (see setState()), so this
 * is a defensive second guard, not the only thing preventing it.
 *
 * Takes: click event (unused). Returns: nothing.
 * Can this fail: no -- delegates to startListening()/
 * stopListeningAndSend(), which handle their own failures.
 */
function handleMicButtonClick() {
  if (currentState === STATE.STANDBY) {
    startListening();
  } else if (currentState === STATE.LISTENING) {
    stopListeningAndSend();
  }
}

dom.micButton.addEventListener("click", handleMicButtonClick);

setState(STATE.STANDBY);
startTraceAnimation();