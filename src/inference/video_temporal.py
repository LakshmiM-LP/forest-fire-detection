import cv2
from ultralytics import YOLO
import os
import sys
import subprocess
import wave
import math
import struct


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/fire_smoke_yolo11n_best.pt"

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = "data/videos/42034-431422873.mp4"

OUTPUT_DIR = "runs"

TEMP_VIDEO = os.path.join(
    OUTPUT_DIR,
    "temporal_detection_temp.mp4"
)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "temporal_detection_h264.mp4"
)

AUDIO_FILE = os.path.join(
    OUTPUT_DIR,
    "alarm_audio.wav"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

# Fire/smoke must be detected in 5 consecutive frames
REQUIRED_DETECTION_FRAMES = 5

# Event closes after 10 consecutive frames
# with no fire/smoke
REQUIRED_CLEAR_FRAMES = 10


# ============================================================
# PERSISTENCE ALERT SETTINGS
# ============================================================

# First persistence alert after fire remains active
# for approximately 5 seconds.
PERSISTENCE_ALERT_1_SECONDS = 5

# Second persistence alert after fire remains active
# for approximately 15 seconds.
PERSISTENCE_ALERT_2_SECONDS = 15


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 44100

BEEP_FREQUENCY = 1000

# Initial alert:
# 3 medium beeps
INITIAL_BEEP_DURATION = 0.25
INITIAL_BEEP_GAP = 0.20

# 5-second persistence alert:
# faster beeps
FAST_BEEP_DURATION = 0.12
FAST_BEEP_GAP = 0.10

# 15-second persistence alert:
# slower/stronger beeps
SLOW_BEEP_DURATION = 0.45
SLOW_BEEP_GAP = 0.45


# ============================================================
# PREPARATION
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print()
print("=" * 65)
print("🔥 TEMPORAL FIRE / SMOKE DETECTION")
print("=" * 65)

print(
    f"Input video: {VIDEO_PATH}"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)

print(
    f"Required detection frames: "
    f"{REQUIRED_DETECTION_FRAMES}"
)

print(
    f"Required clear frames: "
    f"{REQUIRED_CLEAR_FRAMES}"
)

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
    print("❌ FFmpeg was not found.")
    print(
        "Make sure 'ffmpeg -version' works "
        "in your terminal."
    )

    sys.exit(1)


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print(
    "✅ Fire & Smoke YOLO model loaded successfully!"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        f"❌ Could not open video: "
        f"{VIDEO_PATH}"
    )

    sys.exit(1)


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

duration_seconds = (
    total_frames / fps
)


print()
print(
    f"FPS        : {fps:.2f}"
)

print(
    f"Resolution : {width}x{height}"
)

print(
    f"Frames     : {total_frames}"
)

print(
    f"Duration   : {duration_seconds:.2f}s"
)

print()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    TEMP_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():

    print(
        "❌ Could not create temporary video."
    )

    cap.release()

    sys.exit(1)


# ============================================================
# TEMPORAL STATE
# ============================================================

frame_number = 0

consecutive_detection_frames = 0

consecutive_clear_frames = 0

event_active = False

event_start_time = None

event_number = 0


# ============================================================
# ALERT STATE
# ============================================================

initial_alert_triggered = False

persistence_alert_1_triggered = False

persistence_alert_2_triggered = False


# ============================================================
# AUDIO EVENTS
# ============================================================

# Each item:
#
# {
#     "time": seconds,
#     "type": "initial" / "fast" / "slow"
# }

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


    frame_number += 1


    current_time = (
        frame_number / fps
    )


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

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]

            class_name_lower = (
                class_name.lower()
            )


            # ------------------------------------------------
            # Only Fire / Smoke
            # ------------------------------------------------

            if class_name_lower not in [
                "fire",
                "smoke"
            ]:

                continue


            detected = True


            # Keep highest-confidence detection
            if confidence > detected_confidence:

                detected_confidence = (
                    confidence
                )

                detected_class = (
                    class_name
                )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # Ignore almost-full-frame boxes
            # ------------------------------------------------

            box_area = (
                max(0, x2 - x1)
                *
                max(0, y2 - y1)
            )

            frame_area = (
                width * height
            )


            if (
                box_area >
                0.95 * frame_area
            ):

                detected = False

                continue


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )


            label = (
                f"{class_name.upper()} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


    # ========================================================
    # TEMPORAL COUNTERS
    # ========================================================

    if detected:

        consecutive_detection_frames += 1

        consecutive_clear_frames = 0

    else:

        consecutive_detection_frames = 0

        consecutive_clear_frames += 1


    # ========================================================
    # NEW FIRE EVENT
    # ========================================================

    if (
        not event_active
        and
        consecutive_detection_frames
        >= REQUIRED_DETECTION_FRAMES
    ):

        event_active = True

        event_number += 1

        event_start_time = (
            current_time
            -
            (
                REQUIRED_DETECTION_FRAMES
                / fps
            )
        )


        # Reset alert states
        initial_alert_triggered = True

        persistence_alert_1_triggered = False

        persistence_alert_2_triggered = False


        # ----------------------------------------------------
        # Initial alert
        # ----------------------------------------------------

        audio_events.append(
            {
                "time": current_time,
                "type": "initial"
            }
        )


        print()
        print("=" * 65)
        print(
            f"🚨 FIRE/SMOKE EVENT "
            f"#{event_number} CONFIRMED"
        )

        print(
            f"Time: "
            f"{current_time:.2f}s"
        )

        print(
            f"Class: "
            f"{detected_class}"
        )

        print(
            f"Confidence: "
            f"{detected_confidence:.2f}"
        )

        print(
            "🔊 INITIAL ALERT: "
            "3 MEDIUM BEEPS"
        )

        print("=" * 65)
        print()


    # ========================================================
    # PERSISTENCE ALERTS
    # ========================================================

    if event_active:

        event_duration = (
            current_time
            -
            event_start_time
        )


        # ----------------------------------------------------
        # 5-second persistence alert
        # ----------------------------------------------------

        if (
            event_duration
            >= PERSISTENCE_ALERT_1_SECONDS
            and
            not persistence_alert_1_triggered
        ):

            persistence_alert_1_triggered = True


            audio_events.append(
                {
                    "time": current_time,
                    "type": "fast"
                }
            )


            print()
            print(
                f"⚠️ EVENT #{event_number} "
                f"STILL ACTIVE"
            )

            print(
                f"Duration: "
                f"{event_duration:.2f}s"
            )

            print(
                "🔊 PERSISTENCE ALERT: "
                "FAST BEEPS"
            )

            print()


        # ----------------------------------------------------
        # 15-second persistence alert
        # ----------------------------------------------------

        if (
            event_duration
            >= PERSISTENCE_ALERT_2_SECONDS
            and
            not persistence_alert_2_triggered
        ):

            persistence_alert_2_triggered = True


            audio_events.append(
                {
                    "time": current_time,
                    "type": "slow"
                }
            )


            print()
            print(
                f"🔴 EVENT #{event_number} "
                f"STILL PERSISTING"
            )

            print(
                f"Duration: "
                f"{event_duration:.2f}s"
            )

            print(
                "🔊 ESCALATED ALERT: "
                "SLOW BEEPS"
            )

            print()


    # ========================================================
    # CLOSE EVENT
    # ========================================================

    if (
        event_active
        and
        consecutive_clear_frames
        >= REQUIRED_CLEAR_FRAMES
    ):

        event_active = False

        event_duration = (
            current_time
            -
            event_start_time
        )


        print()
        print(
            f"✅ FIRE EVENT "
            f"#{event_number} CLOSED"
        )

        print(
            f"Event duration: "
            f"{event_duration:.2f}s"
        )

        print(
            f"No fire/smoke for "
            f"{REQUIRED_CLEAR_FRAMES} "
            f"consecutive frames."
        )

        print()


        # Reset counters
        event_start_time = None

        initial_alert_triggered = False

        persistence_alert_1_triggered = False

        persistence_alert_2_triggered = False


    # ========================================================
    # VISUAL STATUS
    # ========================================================

    if event_active:

        event_duration = (
            current_time
            -
            event_start_time
        )


        if (
            event_duration
            >= PERSISTENCE_ALERT_2_SECONDS
        ):

            status_text = (
                "🔴 FIRE PERSISTING"
            )

            status_color = (
                0,
                0,
                255
            )

        elif (
            event_duration
            >= PERSISTENCE_ALERT_1_SECONDS
        ):

            status_text = (
                "⚠️ FIRE STILL ACTIVE"
            )

            status_color = (
                0,
                165,
                255
            )

        else:

            status_text = (
                "🚨 FIRE/SMOKE CONFIRMED"
            )

            status_color = (
                0,
                0,
                255
            )


    elif (
        consecutive_detection_frames
        > 0
    ):

        status_text = (
            "Checking "
            f"{consecutive_detection_frames}/"
            f"{REQUIRED_DETECTION_FRAMES}"
        )

        status_color = (
            0,
            255,
            255
        )


    else:

        status_text = (
            "Monitoring..."
        )

        status_color = (
            0,
            255,
            0
        )


    # ========================================================
    # STATUS BANNER
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (850, 75),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        status_text,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        status_color,
        3
    )


    # ========================================================
    # EVENT INFORMATION
    # ========================================================

    if event_active:

        event_duration = (
            current_time
            -
            event_start_time
        )


        info_text = (
            f"Event #{event_number} | "
            f"Duration: "
            f"{event_duration:.1f}s"
        )

    else:

        info_text = (
            f"Events detected: "
            f"{event_number}"
        )


    cv2.putText(
        frame,
        info_text,
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

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

print(
    f"Confirmed fire/smoke events: "
    f"{event_number}"
)

print(
    f"Total audio alerts: "
    f"{len(audio_events)}"
)


# ============================================================
# AUDIO GENERATION
# ============================================================

def add_beep(
    audio,
    start_time,
    duration,
    frequency,
    volume=16000
):

    start_sample = int(
        start_time * SAMPLE_RATE
    )

    sample_count = int(
        duration * SAMPLE_RATE
    )


    for i in range(sample_count):

        index = (
            start_sample + i
        )


        if (
            index < 0
            or
            index >= len(audio)
        ):

            continue


        # Small fade-in / fade-out
        fade_length = int(
            0.01 * SAMPLE_RATE
        )


        if i < fade_length:

            envelope = (
                i / fade_length
            )

        elif (
            i >
            sample_count - fade_length
        ):

            envelope = (
                sample_count - i
            ) / fade_length

        else:

            envelope = 1.0


        sample = int(
            volume
            *
            envelope
            *
            math.sin(
                2
                *
                math.pi
                *
                frequency
                *
                i
                /
                SAMPLE_RATE
            )
        )


        audio[index] = max(
            -32768,
            min(32767, sample)
        )


def add_gap(
    audio,
    start_time,
    duration
):
    # Silence is already represented by 0,
    # so nothing needs to be added.
    pass


# ============================================================
# CREATE AUDIO
# ============================================================

def create_audio():

    print()
    print(
        "Creating alarm audio..."
    )


    # Add one second so the final beep
    # has enough room.
    total_duration = (
        duration_seconds + 1
    )


    total_samples = int(
        total_duration *
        SAMPLE_RATE
    )


    audio = [
        0
    ] * total_samples


    # --------------------------------------------------------
    # Process every alert
    # --------------------------------------------------------

    for index, event in enumerate(
        audio_events,
        start=1
    ):

        start = event["time"]

        alert_type = event["type"]


        # ====================================================
        # INITIAL ALERT
        # ====================================================

        if alert_type == "initial":

            print(
                f"Alert {index}: "
                f"INITIAL 3-beep "
                f"at {start:.2f}s"
            )


            for beep_number in range(3):

                beep_start = (
                    start
                    +
                    beep_number
                    *
                    (
                        INITIAL_BEEP_DURATION
                        +
                        INITIAL_BEEP_GAP
                    )
                )


                add_beep(
                    audio,
                    beep_start,
                    INITIAL_BEEP_DURATION,
                    BEEP_FREQUENCY
                )


        # ====================================================
        # FAST PERSISTENCE ALERT
        # ====================================================

        elif alert_type == "fast":

            print(
                f"Alert {index}: "
                f"FAST persistence "
                f"at {start:.2f}s"
            )


            for beep_number in range(4):

                beep_start = (
                    start
                    +
                    beep_number
                    *
                    (
                        FAST_BEEP_DURATION
                        +
                        FAST_BEEP_GAP
                    )
                )


                add_beep(
                    audio,
                    beep_start,
                    FAST_BEEP_DURATION,
                    BEEP_FREQUENCY,
                    14000
                )


        # ====================================================
        # SLOW PERSISTENCE ALERT
        # ====================================================

        elif alert_type == "slow":

            print(
                f"Alert {index}: "
                f"SLOW persistence "
                f"at {start:.2f}s"
            )


            for beep_number in range(2):

                beep_start = (
                    start
                    +
                    beep_number
                    *
                    (
                        SLOW_BEEP_DURATION
                        +
                        SLOW_BEEP_GAP
                    )
                )


                add_beep(
                    audio,
                    beep_start,
                    SLOW_BEEP_DURATION,
                    800,
                    18000
                )


    # --------------------------------------------------------
    # Write WAV
    # --------------------------------------------------------

    with wave.open(
        AUDIO_FILE,
        "wb"
    ) as wav_file:

        wav_file.setnchannels(1)

        wav_file.setsampwidth(2)

        wav_file.setframerate(
            SAMPLE_RATE
        )


        packed_audio = b"".join(
            struct.pack(
                "<h",
                sample
            )
            for sample in audio
        )


        wav_file.writeframes(
            packed_audio
        )


    print(
        f"✅ Audio created: "
        f"{AUDIO_FILE}"
    )


# ============================================================
# GENERATE AUDIO
# ============================================================

create_audio()


# ============================================================
# CREATE FINAL H264 VIDEO
# ============================================================

print()
print(
    "Creating browser-compatible H264 video..."
)


if os.path.exists(
    OUTPUT_VIDEO
):

    try:

        os.remove(
            OUTPUT_VIDEO
        )

    except Exception:

        pass


ffmpeg_command = [

    "ffmpeg",

    "-y",

    "-i",
    TEMP_VIDEO,

    "-i",
    AUDIO_FILE,

    "-map",
    "0:v:0",

    "-map",
    "1:a:0",

    "-c:v",
    "libx264",

    "-preset",
    "fast",

    "-crf",
    "23",

    "-pix_fmt",
    "yuv420p",

    "-c:a",
    "aac",

    "-b:a",
    "128k",

    "-movflags",
    "+faststart",

    "-shortest",

    OUTPUT_VIDEO
]


result = subprocess.run(
    ffmpeg_command,
    capture_output=True,
    text=True
)


if result.returncode != 0:

    print()
    print(
        "❌ FFmpeg failed."
    )

    print(
        result.stderr
    )

    sys.exit(1)


# ============================================================
# CLEAN TEMPORARY VIDEO
# ============================================================

if os.path.exists(
    TEMP_VIDEO
):

    os.remove(
        TEMP_VIDEO
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 65)
print("✅ TEMPORAL VERIFICATION COMPLETED!")
print("=" * 65)

print(
    f"Final video: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Confirmed events: "
    f"{event_number}"
)

print(
    f"Total audio alerts: "
    f"{len(audio_events)}"
)

print()
print(
    "Alarm behavior:"
)

print(
    "  Initial event  → 3 medium beeps"
)

print(
    "  5 sec active   → fast persistence alert"
)

print(
    "  15 sec active  → slower persistence alert"
)

print(
    "  10 clear frames → event closed"
)

print("=" * 65)