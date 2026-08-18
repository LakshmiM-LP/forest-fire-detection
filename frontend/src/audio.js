// Generates the same alert beep patterns as the video pipeline's
// baked-in audio (src/inference/video_temporal.py), but live in the
// browser via the Web Audio API for the real-time camera feed.

let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  // Browsers suspend AudioContext until a user gesture unlocks it --
  // "Start Camera" counts as that gesture, so resume defensively here.
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

// Call this once, directly inside the click handler that starts the
// camera, so the browser's autoplay policy doesn't block sound later.
export function unlockAudio() {
  getAudioContext();
}

function beep(ctx, startTime, duration, frequency, volume = 0.3) {
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();

  oscillator.type = "sine";
  oscillator.frequency.value = frequency;

  // Small fade in/out avoids the audible "click" of a hard on/off edge
  const fade = 0.01;
  gain.gain.setValueAtTime(0, startTime);
  gain.gain.linearRampToValueAtTime(volume, startTime + fade);
  gain.gain.setValueAtTime(volume, Math.max(startTime + fade, startTime + duration - fade));
  gain.gain.linearRampToValueAtTime(0, startTime + duration);

  oscillator.connect(gain);
  gain.connect(ctx.destination);

  oscillator.start(startTime);
  oscillator.stop(startTime + duration);
}

/**
 * Plays the alert pattern for a given alert type, mirroring the
 * beep counts/timing used in the exported video's audio track:
 *   initial  -> 3 medium beeps   (new fire/smoke event confirmed)
 *   resumed  -> 1 short chirp    (same event picked back up after a
 *                                 brief gap -- deliberately quieter
 *                                 than "initial" to avoid alarm
 *                                 fatigue for what is physically the
 *                                 same ongoing fire)
 *   fast     -> 4 quick beeps    (event still active at 5s)
 *   slow     -> 2 low beeps      (event still active at 15s, escalated)
 *   closed   -> silence (intentional -- no sound on close)
 */
export function playAlert(type) {
  const ctx = getAudioContext();
  const now = ctx.currentTime;

  if (type === "initial") {
    for (let i = 0; i < 3; i++) {
      beep(ctx, now + i * 0.45, 0.25, 1000, 0.35);
    }
  } else if (type === "resumed") {
    beep(ctx, now, 0.18, 1000, 0.25);
  } else if (type === "fast") {
    for (let i = 0; i < 4; i++) {
      beep(ctx, now + i * 0.22, 0.12, 1000, 0.3);
    }
  } else if (type === "slow") {
    for (let i = 0; i < 2; i++) {
      beep(ctx, now + i * 0.9, 0.45, 800, 0.4);
    }
  }
  // "closed" intentionally plays nothing
}
