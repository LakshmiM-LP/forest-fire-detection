import { useEffect, useRef, useState } from "react";
import { API_URL } from "../api.js";

export default function VideoPanel({ onStatusChange }) {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // queued | processing | done | error
  const [log, setLog] = useState([]);
  const [eventCount, setEventCount] = useState(0);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setJobId(null);
    setStatus(null);
    setLog([]);
    setEventCount(0);
    setError(null);
    onStatusChange({ state: "idle", label: "Video loaded — ready to process" });
  }

  async function uploadAndProcess() {
    if (!file) return;
    setStatus("uploading");
    onStatusChange({ state: "checking", label: "Uploading video…" });

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/video/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed (${res.status})`);

      const data = await res.json();
      setJobId(data.job_id);
      setStatus("queued");
      onStatusChange({ state: "checking", label: "Processing video…" });
    } catch (err) {
      setError(err.message);
      setStatus("error");
      onStatusChange({ state: "idle", label: "Standby" });
    }
  }

  useEffect(() => {
    if (!jobId) return;

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/video/status/${jobId}`);
        const data = await res.json();
        setStatus(data.status);
        setLog(data.log || []);
        setEventCount(data.event_count || 0);

        if (data.status === "done") {
          clearInterval(pollRef.current);
          const hasAlert = (data.event_count || 0) > 0;
          onStatusChange({
            state: hasAlert ? "alert" : "idle",
            label: hasAlert
              ? `${data.event_count} fire/smoke event(s) found in video`
              : "No events detected",
          });
        }

        if (data.status === "error") {
          clearInterval(pollRef.current);
          setError(data.error || "Processing failed");
          onStatusChange({ state: "idle", label: "Standby" });
        }
      } catch (err) {
        // transient network hiccup while polling -- keep trying
      }
    }, 2000);

    return () => clearInterval(pollRef.current);
  }, [jobId]);

  const resultUrl = jobId && status === "done" ? `${API_URL}/video/result/${jobId}` : null;

  return (
    <div className="panels">
      <div className="panel">
        <h3 className="panel-title">Video</h3>

        {!file && (
          <label className="dropzone">
            <input
              type="file"
              accept="video/*"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            Click to choose a video, or drop one here
          </label>
        )}

        {file && !resultUrl && (
          <div className="media-preview">
            <video src={URL.createObjectURL(file)} controls />
          </div>
        )}

        {resultUrl && (
          <div className="media-preview">
            <video src={resultUrl} controls autoPlay />
          </div>
        )}

        {file && !jobId && (
          <button className="btn" onClick={uploadAndProcess}>
            Run Temporal Verification
          </button>
        )}

        {resultUrl && (
          <a href={resultUrl} download className="btn" style={{ display: "inline-block", textDecoration: "none" }}>
            Download Result
          </a>
        )}

        {error && (
          <p style={{ color: "var(--ember)", fontSize: 13, marginTop: 12 }}>{error}</p>
        )}
      </div>

      <div className="panel">
        <h3 className="panel-title">Processing Status</h3>

        <div className="telemetry-row">
          <span className="telemetry-label">Job status</span>
          <span
            className={
              "telemetry-value " +
              (status === "done" ? "sage" : status === "error" ? "ember" : status ? "amber" : "")
            }
          >
            {status ? status.toUpperCase() : "—"}
          </span>
        </div>

        <div className="telemetry-row">
          <span className="telemetry-label">Confirmed events</span>
          <span className={"telemetry-value " + (eventCount > 0 ? "ember" : "sage")}>
            {eventCount}
          </span>
        </div>

        <div className="log">
          {log.length === 0 && "Waiting for job to start…"}
          {log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
