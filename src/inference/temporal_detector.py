"""
Shared fire/smoke temporal-verification state machine.

This is the same confirm/close/merge debounce logic used by
src/inference/video_temporal.py, extracted into a reusable class so
it can also drive real-time, per-frame detection (e.g. a live camera
WebSocket) without re-encoding a video file.

video_temporal.py is intentionally left as its own tested,
subprocess-invoked script for batch video files -- this class is
used specifically for the live-frame path.
"""

from collections import deque


class TemporalFireDetector:

    def __init__(
        self,
        model,
        target_classes=("fire", "smoke"),
        confidence_threshold=0.50,
        confirm_window_seconds=1.5,
        confirm_ratio=0.6,
        close_window_seconds=5.0,
        close_ratio=0.85,
        merge_gap_seconds=4.0,
        persistence_alert_1_seconds=5,
        persistence_alert_2_seconds=15,
        full_frame_area_ratio=0.98,
        full_frame_min_confidence=0.55,
        assumed_fps=10.0,
    ):
        self.model = model
        self.target_classes = set(c.lower() for c in target_classes)
        self.confidence_threshold = confidence_threshold

        self.confirm_window_seconds = confirm_window_seconds
        self.confirm_ratio = confirm_ratio
        self.close_window_seconds = close_window_seconds
        self.close_ratio = close_ratio
        self.merge_gap_seconds = merge_gap_seconds

        self.persistence_alert_1_seconds = persistence_alert_1_seconds
        self.persistence_alert_2_seconds = persistence_alert_2_seconds

        self.full_frame_area_ratio = full_frame_area_ratio
        self.full_frame_min_confidence = full_frame_min_confidence

        # Live streams don't have a fixed fps, so history length is
        # sized off an assumed rate and refreshed as real timestamps
        # come in -- the ratio checks below use elapsed wall-clock
        # time, not frame counts, so this only affects deque sizing.
        confirm_frames = max(1, int(round(confirm_window_seconds * assumed_fps)))
        close_frames = max(1, int(round(close_window_seconds * assumed_fps)))
        self._history = deque(maxlen=max(confirm_frames, close_frames) * 3)

        self.event_active = False
        self.event_start_time = None
        self.event_number = 0
        self.last_event_end_time = None

        self.persistence_alert_1_triggered = False
        self.persistence_alert_2_triggered = False

    def reset(self):
        self.__init__(
            self.model,
            target_classes=self.target_classes,
            confidence_threshold=self.confidence_threshold,
            confirm_window_seconds=self.confirm_window_seconds,
            confirm_ratio=self.confirm_ratio,
            close_window_seconds=self.close_window_seconds,
            close_ratio=self.close_ratio,
            merge_gap_seconds=self.merge_gap_seconds,
            persistence_alert_1_seconds=self.persistence_alert_1_seconds,
            persistence_alert_2_seconds=self.persistence_alert_2_seconds,
            full_frame_area_ratio=self.full_frame_area_ratio,
            full_frame_min_confidence=self.full_frame_min_confidence,
        )

    def _ratio_within(self, seconds, want_true, current_time):
        cutoff = current_time - seconds
        window = [v for (t, v) in self._history if t >= cutoff]
        if not window:
            return 0.0, 0
        matches = sum(1 for v in window if v == want_true)
        return matches / len(window), len(window)

    def process_frame(self, frame, current_time, width, height):
        """
        Runs detection on a single frame and advances the state
        machine. Returns (detections, status) where status is a
        dict describing what the UI/caller should show or play.
        """

        detected = False
        detections = []

        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence_threshold,
            verbose=False,
        )

        frame_area = width * height

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]

                if class_name.lower() not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                box_area = max(0, x2 - x1) * max(0, y2 - y1)

                is_suspect_full_frame = (
                    box_area > self.full_frame_area_ratio * frame_area
                    and confidence < self.full_frame_min_confidence
                )
                if is_suspect_full_frame:
                    continue

                detected = True
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "box": [x1, y1, x2, y2],
                })

        self._history.append((current_time, detected))

        confirm_ratio_now, confirm_n = self._ratio_within(
            self.confirm_window_seconds, True, current_time
        )
        close_ratio_now, close_n = self._ratio_within(
            self.close_window_seconds, False, current_time
        )

        alerts = []

        # ---- confirm / merge ----
        if (
            not self.event_active
            and confirm_n > 0
            and confirm_ratio_now >= self.confirm_ratio
        ):
            self.event_active = True

            is_continuation = (
                self.last_event_end_time is not None
                and (current_time - self.last_event_end_time) <= self.merge_gap_seconds
            )

            if is_continuation:
                alerts.append("resumed")
            else:
                self.event_number += 1
                self.event_start_time = current_time - self.confirm_window_seconds
                self.persistence_alert_1_triggered = False
                self.persistence_alert_2_triggered = False
                alerts.append("initial")

        # ---- persistence ----
        if self.event_active:
            event_duration = current_time - self.event_start_time

            if (
                event_duration >= self.persistence_alert_1_seconds
                and not self.persistence_alert_1_triggered
            ):
                self.persistence_alert_1_triggered = True
                alerts.append("fast")

            if (
                event_duration >= self.persistence_alert_2_seconds
                and not self.persistence_alert_2_triggered
            ):
                self.persistence_alert_2_triggered = True
                alerts.append("slow")

        # ---- close ----
        if (
            self.event_active
            and close_n > 0
            and close_ratio_now >= self.close_ratio
        ):
            self.event_active = False
            self.last_event_end_time = current_time
            alerts.append("closed")
            self.event_start_time = None
            self.persistence_alert_1_triggered = False
            self.persistence_alert_2_triggered = False

        event_duration = (
            (current_time - self.event_start_time)
            if (self.event_active and self.event_start_time is not None)
            else 0.0
        )

        status = {
            "detected": detected,
            "event_active": self.event_active,
            "event_number": self.event_number,
            "event_duration": round(event_duration, 2),
            "confirm_progress": round(confirm_ratio_now, 2) if not self.event_active else None,
            "alerts": alerts,
        }

        return detections, status