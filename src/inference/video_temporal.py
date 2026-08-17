import cv2
from ultralytics import YOLO
import os
import sys
import subprocess
import wave
import math
import struct
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/fire_smoke_yolo11n_best.pt"

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = "data/videos/42034-431422873.mp4"

OUTPUT_DIR = "runs"

TEMP_VIDEO = os.path.join(OUTPUT_DIR, "temporal_detection_temp.mp4")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "temporal_detection_h264.mp4")
AUDIO_FILE = os.path.join(OUTPUT_DIR, "alarm_audio.wav")


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

# Only classes we care about
TARGET_CLASSES = ["fire", "smoke"]

# A box covering nearly the whole frame is usually a real, close-up
# fire/smoke event (not a glitch) UNLESS confidence is also weak.
# Only treat it as a likely false positive when BOTH conditions hold.
FULL_FRAME_AREA_RATIO = 0.98
FULL_FRAME_MIN_CONFIDENCE = 0.55


# ============================================================
# TEMPORAL DEBOUNCE SETTINGS (TIME-BASED, NOT FRAME-COUNT-BASED)
# ============================================================

# How many recent seconds we look back over when deciding
# whether to CONFIRM a new event.
CONFIRM_WINDOW_SECONDS = 1.5

# Fraction of frames within that window that must be positive
# to confirm an event. Using a ratio (not "every single frame")
# means a couple of flickered/missed frames won't reset progress.
CONFIRM_RATIO = 0.6

# How many recent seconds we look back over when deciding
# whether to CLOSE an active event.
CLOSE_WINDOW_SECONDS = 5.0

# Fraction of frames within that window that must be clear
# (no detection) to close an event. Deliberately stricter/slower
# than the confirm side: for a safety system, closing an event
# too early is worse than closing it a little late.
CLOSE_RATIO = 0.85

# If a new event would start within this many seconds of the
# previous event closing, treat it as a continuation of the same
# event instead of a brand-new one (avoids event-count inflation
# and repeated beep sequences for what is physically one fire).
MERGE_GAP_SECONDS = 4.0


# ============================================================
# PERSISTENCE ALERT SETTINGS
# ============================================================

PERSISTENCE_ALERT_1_SECONDS = 5
PERSISTENCE_ALERT_2_SECONDS = 15


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 44100
BEEP_FREQUENCY = 1000

INITIAL_BEEP_DURATION = 0.25
INITIAL_BEEP_GAP = 0.20

FAST_BEEP_DURATION = 0.12
FAST_BEEP_GAP = 0.10

SLOW_BEEP_DURATION = 0.45
SLOW_BEEP_GAP = 0.45


# ============================================================
# PREPARATION
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("=" * 65)
print("FIRE / SMOKE TEMPORAL DETECTION")
print("=" * 65)
print(f"Input video: {VIDEO_PATH}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Confirm window: {CONFIRM_WINDOW_SECONDS}s @ ratio {CONFIRM_RATIO}")
print(f"Close window: {CLOSE_WINDOW_SECONDS}s @ ratio {CLOSE_RATIO}")
print(f"Merge gap: {MERGE_GAP_SECONDS}s")
print("=" * 65)


# ============================================================
# CHECK FFMPEG
# ============================================================

try:
    check_ffmpeg = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if check_ffmpeg.returncode != 0:
        raise RuntimeError
except Exception:
    print()
    print("FFmpeg was not found.")
    print("Make sure 'ffmpeg -version' works in your terminal.")
    sys.exit(1)


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Fire & Smoke YOLO model loaded successfully!")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Could not open video: {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = total_frames / fps

print()
print(f"FPS        : {fps:.2f}")
print(f"Resolution : {width}x{height}")
print(f"Frames     : {total_frames}")
print(f"Duration   : {duration_seconds:.2f}s")
print()


# ============================================================
# CONVERT TIME WINDOWS -> FRAME COUNTS FOR THIS VIDEO'S FPS
# ============================================================

confirm_window_frames = max(1, int(round(CONFIRM_WINDOW_SECONDS * fps)))
close_window_frames = max(1, int(round(CLOSE_WINDOW_SECONDS * fps)))

# Single rolling window big enough to serve both checks
history_maxlen = max(confirm_window_frames, close_window_frames)
detection_history = deque(maxlen=history_maxlen)


# ============================================================
# DOWNSCALE FOR PROCESSING
# ============================================================
# YOLO resizes every frame down to imgsz=640 internally anyway, so
# feeding it a full 4K frame wastes memory and CPU without adding
# detection accuracy. Downscaling once here (decode -> inference ->
# drawing -> writing) cuts memory use at every stage of the loop,
# not just the final FFmpeg export -- important on machines with
# limited RAM.

MAX_PROCESS_HEIGHT = 1080

orig_width, orig_height = width, height

if height > MAX_PROCESS_HEIGHT:
    scale = MAX_PROCESS_HEIGHT / height
    width = int(round(orig_width * scale / 2) * 2)   # keep even (codec requirement)
    height = MAX_PROCESS_HEIGHT
    print(f"Downscaling frames: {orig_width}x{orig_height} -> {width}x{height}")
else:
    print(f"No downscaling needed: {orig_width}x{orig_height}")

print()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (width, height))

if not out.isOpened():
    print("Could not create temporary video.")
    cap.release()
    sys.exit(1)


# ============================================================
# STATE
# ============================================================

frame_number = 0

event_active = False
event_start_time = None
event_number = 0
last_event_end_time = None

initial_alert_triggered = False
persistence_alert_1_triggered = False
persistence_alert_2_triggered = False

# Each item: {"time": seconds, "type": "initial" / "fast" / "slow"}
audio_events = []


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("=" * 65)
print("Starting video processing...")
print("=" * 65)
print()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    if (orig_width, orig_height) != (width, height):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    frame_number += 1
    current_time = frame_number / fps

    detected = False
    detected_class = None
    detected_confidence = 0.0

    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            class_name_lower = class_name.lower()

            if class_name_lower not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clip box to frame bounds (defensive against
            # out-of-range coordinates near edges)
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            frame_area = width * height

            # ------------------------------------------------
            # Only discard a near-full-frame box when confidence
            # is ALSO weak -- a strong, confident detection that
            # happens to cover most of the frame is very likely a
            # real close-up fire/smoke event, not a glitch.
            # ------------------------------------------------
            is_suspect_full_frame = (
                box_area > FULL_FRAME_AREA_RATIO * frame_area
                and confidence < FULL_FRAME_MIN_CONFIDENCE
            )

            if is_suspect_full_frame:
                continue

            detected = True

            if confidence > detected_confidence:
                detected_confidence = confidence
                detected_class = class_name

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"{class_name.upper()} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    # ========================================================
    # ROLLING DETECTION HISTORY
    # ========================================================

    detection_history.append(detected)

    def ratio_over_last(n, want_true):
        n = min(n, len(detection_history))
        if n == 0:
            return 0.0
        recent = list(detection_history)[-n:]
        matches = sum(1 for v in recent if v == want_true)
        return matches / n

    confirm_ratio_now = ratio_over_last(confirm_window_frames, True)
    close_ratio_now = ratio_over_last(close_window_frames, False)

    # Only meaningful once we've actually seen enough frames to
    # fill the relevant window -- avoids confirming/closing on
    # thin data right at the very start of the clip.
    enough_history_for_confirm = len(detection_history) >= confirm_window_frames
    enough_history_for_close = len(detection_history) >= close_window_frames

    # ========================================================
    # NEW / CONTINUING FIRE EVENT
    # ========================================================

    if (
        not event_active
        and enough_history_for_confirm
        and confirm_ratio_now >= CONFIRM_RATIO
    ):

        event_active = True

        is_continuation = (
            last_event_end_time is not None
            and (current_time - last_event_end_time) <= MERGE_GAP_SECONDS
        )

        if is_continuation:
            # Resume the same event: keep the original event number
            # and (approximate) original start time so duration keeps
            # accumulating, and don't re-fire the initial 3-beep --
            # that would be alarm fatigue for a fire that never
            # actually stopped.
            print()
            print("=" * 65)
            print(f"FIRE/SMOKE EVENT #{event_number} RESUMED")
            print(f"Time: {current_time:.2f}s")
            print("(within merge gap of previous event -- no new alert)")
            print("=" * 65)
            print()

        else:
            event_number += 1
            event_start_time = current_time - CONFIRM_WINDOW_SECONDS

            initial_alert_triggered = True
            persistence_alert_1_triggered = False
            persistence_alert_2_triggered = False

            audio_events.append({"time": current_time, "type": "initial"})

            print()
            print("=" * 65)
            print(f"FIRE/SMOKE EVENT #{event_number} CONFIRMED")
            print(f"Time: {current_time:.2f}s")
            print(f"Class: {detected_class}")
            print(f"Confidence: {detected_confidence:.2f}")
            print("INITIAL ALERT: 3 MEDIUM BEEPS")
            print("=" * 65)
            print()

    # ========================================================
    # PERSISTENCE ALERTS
    # ========================================================

    if event_active:

        event_duration = current_time - event_start_time

        if (
            event_duration >= PERSISTENCE_ALERT_1_SECONDS
            and not persistence_alert_1_triggered
        ):
            persistence_alert_1_triggered = True
            audio_events.append({"time": current_time, "type": "fast"})

            print()
            print(f"EVENT #{event_number} STILL ACTIVE")
            print(f"Duration: {event_duration:.2f}s")
            print("PERSISTENCE ALERT: FAST BEEPS")
            print()

        if (
            event_duration >= PERSISTENCE_ALERT_2_SECONDS
            and not persistence_alert_2_triggered
        ):
            persistence_alert_2_triggered = True
            audio_events.append({"time": current_time, "type": "slow"})

            print()
            print(f"EVENT #{event_number} STILL PERSISTING")
            print(f"Duration: {event_duration:.2f}s")
            print("ESCALATED ALERT: SLOW BEEPS")
            print()

    # ========================================================
    # CLOSE EVENT
    # ========================================================

    if (
        event_active
        and enough_history_for_close
        and close_ratio_now >= CLOSE_RATIO
    ):

        event_active = False
        last_event_end_time = current_time
        event_duration = current_time - event_start_time

        print()
        print(f"FIRE EVENT #{event_number} CLOSED")
        print(f"Event duration: {event_duration:.2f}s")
        print(
            f"Clear in {CLOSE_RATIO * 100:.0f}%+ of the last "
            f"{CLOSE_WINDOW_SECONDS:.1f}s."
        )
        print()

        event_start_time = None
        initial_alert_triggered = False
        persistence_alert_1_triggered = False
        persistence_alert_2_triggered = False

    # ========================================================
    # VISUAL STATUS
    # ========================================================

    if event_active:

        event_duration = current_time - event_start_time

        if event_duration >= PERSISTENCE_ALERT_2_SECONDS:
            status_text = "FIRE PERSISTING"
            status_color = (0, 0, 255)
        elif event_duration >= PERSISTENCE_ALERT_1_SECONDS:
            status_text = "FIRE STILL ACTIVE"
            status_color = (0, 165, 255)
        else:
            status_text = "FIRE/SMOKE CONFIRMED"
            status_color = (0, 0, 255)

    elif confirm_ratio_now > 0:

        pct = int(confirm_ratio_now * 100)
        status_text = f"Checking... {pct}%"
        status_color = (0, 255, 255)

    else:
        status_text = "Monitoring..."
        status_color = (0, 255, 0)

    # ========================================================
    # STATUS BANNER
    # ========================================================

    cv2.rectangle(frame, (10, 10), (850, 75), (0, 0, 0), -1)

    cv2.putText(
        frame,
        status_text,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        status_color,
        3
    )

    if event_active:
        event_duration = current_time - event_start_time
        info_text = f"Event #{event_number} | Duration: {event_duration:.1f}s"
    else:
        info_text = f"Events detected: {event_number}"

    cv2.putText(
        frame,
        info_text,
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    out.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print()
print("=" * 65)
print("VIDEO PROCESSING COMPLETED")
print("=" * 65)
print(f"Confirmed fire/smoke events: {event_number}")
print(f"Total audio alerts: {len(audio_events)}")


# ============================================================
# AUDIO GENERATION
# ============================================================

def add_beep(audio, start_time, duration, frequency, volume=16000):

    start_sample = int(start_time * SAMPLE_RATE)
    sample_count = int(duration * SAMPLE_RATE)

    for i in range(sample_count):

        index = start_sample + i
        if index < 0 or index >= len(audio):
            continue

        fade_length = int(0.01 * SAMPLE_RATE)

        if i < fade_length:
            envelope = i / fade_length
        elif i > sample_count - fade_length:
            envelope = (sample_count - i) / fade_length
        else:
            envelope = 1.0

        sample = int(
            volume * envelope
            * math.sin(2 * math.pi * frequency * i / SAMPLE_RATE)
        )

        audio[index] = max(-32768, min(32767, sample))


def create_audio():

    print()
    print("Creating alarm audio...")

    total_duration = duration_seconds + 1
    total_samples = int(total_duration * SAMPLE_RATE)
    audio = [0] * total_samples

    for index, event in enumerate(audio_events, start=1):

        start = event["time"]
        alert_type = event["type"]

        if alert_type == "initial":

            print(f"Alert {index}: INITIAL 3-beep at {start:.2f}s")

            for beep_number in range(3):
                beep_start = start + beep_number * (
                    INITIAL_BEEP_DURATION + INITIAL_BEEP_GAP
                )
                add_beep(audio, beep_start, INITIAL_BEEP_DURATION, BEEP_FREQUENCY)

        elif alert_type == "fast":

            print(f"Alert {index}: FAST persistence at {start:.2f}s")

            for beep_number in range(4):
                beep_start = start + beep_number * (
                    FAST_BEEP_DURATION + FAST_BEEP_GAP
                )
                add_beep(audio, beep_start, FAST_BEEP_DURATION, BEEP_FREQUENCY, 14000)

        elif alert_type == "slow":

            print(f"Alert {index}: SLOW persistence at {start:.2f}s")

            for beep_number in range(2):
                beep_start = start + beep_number * (
                    SLOW_BEEP_DURATION + SLOW_BEEP_GAP
                )
                add_beep(audio, beep_start, SLOW_BEEP_DURATION, 800, 18000)

    with wave.open(AUDIO_FILE, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        packed_audio = b"".join(struct.pack("<h", sample) for sample in audio)
        wav_file.writeframes(packed_audio)

    print(f"Audio created: {AUDIO_FILE}")


create_audio()


# ============================================================
# CREATE FINAL H264 VIDEO
# ============================================================

print()
print("Creating browser-compatible H264 video...")

if os.path.exists(OUTPUT_VIDEO):
    try:
        os.remove(OUTPUT_VIDEO)
    except Exception:
        pass

# Frames were already downscaled (MAX_PROCESS_HEIGHT) before being
# written to TEMP_VIDEO, so no scale filter is needed here -- this
# step is now just a codec conversion (mp4v -> h264/aac), which is
# far lighter than encoding a full 4K stream.
ffmpeg_command = [
    "ffmpeg", "-y",
    "-i", TEMP_VIDEO,
    "-i", AUDIO_FILE,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "25",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "128k",
    "-movflags", "+faststart",
    "-shortest",
    OUTPUT_VIDEO
]

result = subprocess.run(ffmpeg_command, capture_output=True, text=True)

if result.returncode != 0:
    print()
    print("=" * 65)
    print("FFMPEG FAILED (see stderr below)")
    print("=" * 65)
    print(result.stderr[-3000:])  # tail, in case it's long
    print("=" * 65)
    sys.exit(1)

# Sanity-check the output is actually a complete, playable file
# before trusting it -- catches silent truncation/OOM kills that
# ffmpeg's own return code sometimes doesn't flag clearly.
if not os.path.exists(OUTPUT_VIDEO) or os.path.getsize(OUTPUT_VIDEO) < 1024:
    print()
    print("FFmpeg exited 0 but the output file looks wrong (missing or tiny).")
    print("Check container memory limits -- this usually means it was killed mid-encode.")
    sys.exit(1)

if os.path.exists(TEMP_VIDEO):
    os.remove(TEMP_VIDEO)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 65)
print("TEMPORAL VERIFICATION COMPLETED!")
print("=" * 65)
print(f"Final video: {OUTPUT_VIDEO}")
print(f"Confirmed events: {event_number}")
print(f"Total audio alerts: {len(audio_events)}")
print()
print("Alarm behavior:")
print(f"  Confirm  -> {CONFIRM_RATIO*100:.0f}%+ positive in last {CONFIRM_WINDOW_SECONDS}s -> 3 medium beeps")
print(f"  5s active  -> fast persistence alert")
print(f"  15s active -> slower persistence alert")
print(f"  Close    -> {CLOSE_RATIO*100:.0f}%+ clear in last {CLOSE_WINDOW_SECONDS}s -> event closed")
print(f"  Re-detection within {MERGE_GAP_SECONDS}s of close -> same event resumed, no repeat alert")
print("=" * 65)