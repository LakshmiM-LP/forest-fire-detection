import { useState } from "react";
import ScanRing from "./components/ScanRing.jsx";
import ImagePanel from "./components/ImagePanel.jsx";
import VideoPanel from "./components/VideoPanel.jsx";
import LivePanel from "./components/LivePanel.jsx";

const TABS = [
  { id: "image", label: "Image" },
  { id: "video", label: "Video" },
  { id: "live", label: "Live" },
];

export default function App() {
  const [tab, setTab] = useState("image");
  const [status, setStatus] = useState({ state: "idle", label: "Standby" });

  function handleTabChange(id) {
    setTab(id);
    setStatus({ state: "idle", label: "Standby" });
  }

  const statusText = {
    idle: "MONITORING",
    checking: "CONFIRMING",
    alert: "FIRE / SMOKE DETECTED",
  }[status.state];

  return (
    <div className="shell">
      <p className="eyebrow">Real-Time Detection System</p>
      <h1 className="title">Forest Fire &amp; Smoke Detection</h1>
      <p className="subtitle">
        YOLOv11n-based detection with temporal verification across image,
        video, and live camera input — built to reduce false alarms while
        catching real events fast.
      </p>

      <div className="hero">
        <ScanRing state={status.state} />
        <div className="hero-text">
          <p className={`hero-status ${status.state}`}>{statusText}</p>
          <p className="hero-meta">{status.label}</p>
        </div>
      </div>

      <div className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => handleTabChange(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "image" && <ImagePanel onStatusChange={setStatus} />}
      {tab === "video" && <VideoPanel onStatusChange={setStatus} />}
      {tab === "live" && <LivePanel onStatusChange={setStatus} />}

      <p className="footer-note">
        Fire &amp; Smoke Detection · YOLOv11n · FastAPI · React
      </p>
    </div>
  );
}
