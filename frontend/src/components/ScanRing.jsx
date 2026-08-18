export default function ScanRing({ state }) {
  // state: "idle" | "checking" | "alert"
  return (
    <div className={`scan-ring ${state}`}>
      <div className="scan-ring-dot" />
    </div>
  );
}
