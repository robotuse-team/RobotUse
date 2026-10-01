import { useEffect, useState } from "react";
import { startReplay3d } from "../api/client";

export function SceneView({ trialPath }: { trialPath?: string }) {
  const [url, setUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    startReplay3d(trialPath)
      .then(({ url }) => setUrl(url))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [trialPath]);

  if (loading) {
    return (
      <div className="empty-state" style={{ height: "100%", display: "grid", placeItems: "center" }}>
        Starting 3D viewer...
      </div>
    );
  }

  if (error || !url) {
    return (
      <div
        className="empty-state"
        style={{ height: "100%", display: "grid", placeItems: "center" }}
      >
        <div style={{ maxWidth: 520, textAlign: "center", display: "grid", gap: 8 }}>
          <div style={{ fontWeight: 600 }}>3D replay unavailable for this run.</div>
          <div style={{ opacity: 0.8 }}>
            {error
              ? `Reason: ${error.replace(/^Error:\s*/, "")}`
              : "No replay URL was returned by the server."}
          </div>
        </div>
      </div>
    );
  }

  return <iframe src={url} className="scene-iframe" title="3D Scene Replay" />;
}
