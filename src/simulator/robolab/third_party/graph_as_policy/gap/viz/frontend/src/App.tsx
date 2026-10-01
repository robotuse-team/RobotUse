import { useEffect, useState } from "react";
import { getTrials, getVizTrial } from "./api/client";
import { ExecutionView } from "./components/ExecutionView";
import { InspectorPanel } from "./components/InspectorPanel";
import { SceneView } from "./components/SceneView";
import { StateMachineView } from "./components/StateMachineView";
import type { Selection, VizTrial } from "./types/viz";

type ViewMode = "state-machine" | "execution" | "3d-scene";

export default function App() {
  const [trials, setTrials] = useState<string[]>([]);
  const [activeTrial, setActiveTrial] = useState<string | undefined>();
  const [viz, setViz] = useState<VizTrial | null>(null);
  const [view, setView] = useState<ViewMode>("state-machine");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Selection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function init() {
      setLoading(true);
      setError(null);

      let discoveredTrials: string[] = [];
      try {
        discoveredTrials = await getTrials();
        if (cancelled) return;
        setTrials(discoveredTrials);
      } catch {
        setTrials([]);
      }

      const firstTrial = discoveredTrials[0];
      if (!cancelled) setActiveTrial(firstTrial);

      try {
        const nextViz = await getVizTrial(firstTrial);
        if (cancelled) return;
        setViz(nextViz);
        setSelected(initialSelection(nextViz));
        setView(initialView(nextViz));
      } catch (nextError) {
        if (!cancelled) setError(String(nextError));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void init();
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep the trial rail fresh: new runs land on disk while the server is
  // up, so poll the (re-scanning) /api/trials endpoint.
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const next = await getTrials();
        setTrials((prev) =>
          prev.length === next.length && prev.every((p, i) => p === next[i])
            ? prev
            : next,
        );
      } catch {
        // transient — keep the current list
      }
    }, 10_000);
    return () => clearInterval(interval);
  }, []);

  async function handleSelectTrial(path: string) {
    setActiveTrial(path);
    setLoading(true);
    setError(null);
    try {
      const nextViz = await getVizTrial(path);
      setViz(nextViz);
      setSelected(initialSelection(nextViz));
      setHovered(null);
      setView(initialView(nextViz));
    } catch (nextError) {
      setError(String(nextError));
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="screen-state screen-state--loading">
        <div className="screen-state__orb" />
        <div>Loading workflow explorer…</div>
      </div>
    );
  }

  if (!viz || error) {
    return (
      <div className="screen-state screen-state--error">
        <div>{error || "No visualization data available."}</div>
      </div>
    );
  }

  return (
    <div className="viz-app">
      {trials.length > 1 && (
        <aside className="trial-rail">
          <div className="trial-rail__title">Trials</div>
          <div className="trial-rail__list">
            {trials.map((trialPath) => (
              <button
                key={trialPath}
                type="button"
                className={`trial-rail__item ${trialPath === activeTrial ? "is-active" : ""}`}
                onClick={() => { void handleSelectTrial(trialPath); }}
              >
                {trialPath}
              </button>
            ))}
          </div>
        </aside>
      )}

      <main className="workspace-shell">
        <div className="workspace-toolbar">
          <div className="view-tabs">
            {(["state-machine", "execution", "3d-scene"] as ViewMode[]).map((mode) => {
              const disabled = mode === "3d-scene" && !viz.meta.has_scene_log;
              return (
                <button
                  key={mode}
                  type="button"
                  className={`view-tabs__tab ${view === mode ? "is-active" : ""}`}
                  disabled={disabled}
                  title={
                    disabled
                      ? "3D replay unavailable for this run: no scene_log/ was recorded (scene logging is opt-in via gap.viz.TrialLogger)"
                      : undefined
                  }
                  onClick={() => setView(mode)}
                >
                  {labelForView(mode)}
                </button>
              );
            })}
          </div>
        </div>

        <div className={`workspace-body ${view !== "state-machine" ? "workspace-body--full" : ""}`}>
          <section className="workspace-stage">
            {view === "state-machine" && (
              <StateMachineView
                workflow={viz.workflow}
                selected={selected}
                onSelect={setSelected}
              />
            )}
            {view === "execution" && (
              <ExecutionView
                viz={viz}
                selected={selected}
                hovered={hovered}
                onSelect={setSelected}
                onHover={setHovered}
              />
            )}
            {view === "3d-scene" && (
              <SceneView trialPath={activeTrial} />
            )}
          </section>

          {view === "state-machine" && (
            <InspectorPanel
              viz={viz}
              selected={selected}
              trialPath={activeTrial}
              onSelect={setSelected}
            />
          )}
        </div>

        {view === "execution" && selected && (
          <div className="inspector-modal-backdrop" onClick={() => setSelected(null)}>
            <div className="inspector-modal" onClick={(e) => e.stopPropagation()}>
              <button
                type="button"
                className="inspector-modal__close"
                onClick={() => setSelected(null)}
              >
                &times;
              </button>
              <InspectorPanel
                viz={viz}
                selected={selected}
                trialPath={activeTrial}
                onSelect={setSelected}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function initialSelection(_viz: VizTrial): Selection | null {
  return null;
}

function initialView(viz: VizTrial): ViewMode {
  return viz.execution.steps.length > 0 ? "execution" : "state-machine";
}

function labelForView(view: ViewMode): string {
  switch (view) {
    case "state-machine":
      return "State Machine";
    case "execution":
      return "Execution";
    case "3d-scene":
      return "3D Scene";
    default:
      return view;
  }
}

