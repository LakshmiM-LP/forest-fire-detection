import { useEffect, useRef, useState } from "react";
import { WS_URL } from "../api.js";
import { unlockAudio, playAlert } from "../audio.js";

const CAPTURE_INTERVAL_MS = 400; // ~2.5 fps to the server

export default function LivePanel({ onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // hidden, for capturing frames to send
  const overlayRef = useRef(null); // visible, for drawing boxes
  const wsRef = useRef(null);
  const intervalRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [telemetry, setTelemetry] = useState(null);
  const [error, setError] = useState(null);

  async function start() {
    setError(null);
    unlockAudio(); // must happen inside a user-gesture handler, or the

    // browser will silently block sound later
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const ws = new WebSocket(`${WS_URL}/ws/live`);
      wsRef.current = ws;

      ws.onopen = () => {
        setRunning(true);
        onStatusChange({ state: "idle", label: "Live camera monitoring" });

        intervalRef.current = setInterval(() => {
          captureAndSend();
        }, CAPTURE_INTERVAL_MS);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) return;

        setTelemetry(data);
        drawOverlay(data.detections || []);

        const alerts = data.status.alerts || [];
        alerts.forEach((alertType) => playAlert(alertType));

        const s = data.status;
        if (s.event_active) {
          onStatusChange({
            state: "alert",
            label: `Event #${s.event_number} active — ${s.event_duration}s`,
          });
        } else if (s.confirm_progress > 0) {
          onStatusChange({
            state: "checking",
            label: `Confirming… ${(s.confirm_progress * 100).toFixed(0)}%`,
          });
        } else {
          onStatusChange({ state: "idle", label: "Live camera monitoring" });
        }
      };

      ws.onerror = () => {
        setError("Connection to detection server failed.");
      };

      ws.onclose = () => {
        setRunning(false);
      };
    } catch (err) {
      setError(err.message || "Could not access camera");
    }
  }

  function stop() {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();

    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    setRunning(false);
    setTelemetry(null);
    onStatusChange({ state: "idle", label: "Standby" });

    const ctx = overlayRef.current?.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
  }

  function captureAndSend() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (video.videoWidth === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result.split(",")[1];
          if (ws.readyState === WebSocket.OPEN) ws.send(base64);
        };
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.7
    );
  }

  function drawOverlay(detections) {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay || video.videoWidth === 0) return;

    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;

    const scaleX = video.clientWidth / video.videoWidth;
    const scaleY = video.clientHeight / video.videoHeight;

    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    detections.forEach((d) => {
      if (!Array.isArray(d.box) || d.box.length !== 4) return;

      const [x1, y1, x2, y2] = d.box;
      const rx = x1 * scaleX;
      const ry = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      ctx.strokeStyle = "#ff7a33";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(rx, ry, rw, rh);

      const label = `${d.class.toUpperCase()} ${d.confidence.toFixed(2)}`;
      ctx.font = "600 13px 'JetBrains Mono', monospace";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "#ff7a33";
      ctx.fillRect(rx, ry - 20, textWidth + 12, 20);
      ctx.fillStyle = "#0c1210";
      ctx.fillText(label, rx + 6, ry - 6);
    });
  }

  useEffect(() => {
    return () => stop(); // cleanup on unmount
  }, []);

  return (
    <div className="panels">
      <div className="panel">
        <h3 className="panel-title">Live Camera</h3>

        <div className="media-preview">
          <video ref={videoRef} muted playsInline style={{ display: running ? "block" : "none" }} />
          <canvas ref={overlayRef} />
          {!running && (
            <div style={{ padding: "60px 20px", textAlign: "center", color: "var(--text-dim)" }}>
              Camera is off
            </div>
          )}
        </div>
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {!running ? (
          <button className="btn" onClick={start}>
            Start Camera
          </button>
        ) : (
          <button className="btn danger" onClick={stop}>
            Stop Camera
          </button>
        )}

        {error && (
          <p style={{ color: "var(--ember)", fontSize: 13, marginTop: 12 }}>{error}</p>
        )}
      </div>

      <div className="panel">
        <h3 className="panel-title">Live Telemetry</h3>

        {!telemetry && <p className="empty-state">Start the camera to see live readings.</p>}

        {telemetry && (
          <>
            <div className="telemetry-row">
              <span className="telemetry-label">Event active</span>
              <span className={"telemetry-value " + (telemetry.status.event_active ? "ember" : "sage")}>
                {telemetry.status.event_active ? "YES" : "NO"}
              </span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-label">Events so far</span>
              <span className="telemetry-value">{telemetry.status.event_number}</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-label">Event duration</span>
              <span className="telemetry-value">{telemetry.status.event_duration}s</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-label">Detections this frame</span>
              <span className="telemetry-value">{telemetry.detections.length}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
