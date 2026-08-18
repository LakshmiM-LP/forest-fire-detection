import { useRef, useState } from "react";
import { API_URL } from "../api.js";

export default function ImagePanel({ onStatusChange }) {
  const [imageUrl, setImageUrl] = useState(null);
  const [file, setFile] = useState(null);
  const [detections, setDetections] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setImageUrl(URL.createObjectURL(f));
    setDetections(null);
    setError(null);
    onStatusChange({ state: "idle", label: "Image loaded — ready to scan" });
  }

  function drawBoxes(dets, naturalWidth, naturalHeight) {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    dets.forEach((d) => {
      if (!Array.isArray(d.box) || d.box.length !== 4) return; // skip malformed entries safely

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

  async function runDetection() {
    if (!file) return;
    setLoading(true);
    setError(null);
    onStatusChange({ state: "checking", label: "Scanning image…" });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const data = await res.json();
      setDetections(data.detections || []);

      const img = imgRef.current;
      if (img) {
        drawBoxes(data.detections || [], img.naturalWidth, img.naturalHeight);
      }

      if (data.detected) {
        onStatusChange({
          state: "alert",
          label: `${data.detections.length} detection(s) found`,
        });
      } else {
        onStatusChange({ state: "idle", label: "No fire or smoke detected" });
      }
    } catch (err) {
      setError(err.message || "Detection failed");
      onStatusChange({ state: "idle", label: "Standby" });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panels">
      <div className="panel">
        <h3 className="panel-title">Image</h3>

        {!imageUrl && (
          <label className="dropzone">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            Click to choose an image, or drop one here
          </label>
        )}

        {imageUrl && (
          <div className="media-preview">
            <img ref={imgRef} src={imageUrl} onLoad={() => detections && drawBoxes(detections, imgRef.current.naturalWidth, imgRef.current.naturalHeight)} />
            <canvas ref={canvasRef} />
          </div>
        )}

        {imageUrl && (
          <div style={{ display: "flex", gap: 10 }}>
            <button className="btn" onClick={runDetection} disabled={loading}>
              {loading ? "Scanning…" : "Run Detection"}
            </button>
            <button
              className="btn danger"
              onClick={() => {
                setImageUrl(null);
                setFile(null);
                setDetections(null);
                onStatusChange({ state: "idle", label: "Standby" });
              }}
            >
              Clear
            </button>
          </div>
        )}

        {error && (
          <p style={{ color: "var(--ember)", fontSize: 13, marginTop: 12 }}>{error}</p>
        )}
      </div>

      <div className="panel">
        <h3 className="panel-title">Detections</h3>

        {detections === null && (
          <p className="empty-state">Run detection to see results here.</p>
        )}

        {detections && detections.length === 0 && (
          <p className="empty-state">No fire or smoke detected in this image.</p>
        )}

        {detections && detections.length > 0 && (
          <div>
            {detections.map((d, i) => (
              <span className="detection-chip" key={i}>
                {d.class.toUpperCase()} · {(d.confidence * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
