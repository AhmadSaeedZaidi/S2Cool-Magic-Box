// -------------------------------------------------------------------------
// ModelComparisonChart.jsx — Model metrics split by prediction target.
//
// Fetches /v1/ml/models and renders separate sections for GHI and Temp
// models with clean tables and small visual bars for relative comparison.
// -------------------------------------------------------------------------
import { useState, useEffect, useMemo } from "react";

/** Inline metric bar scaled to the max in its group. */
function MetricBar({ value, max, color }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="text-[12px] font-mono text-s2-text w-14 text-right">{value.toFixed(2)}</span>
      <div className="flex-1 h-2 rounded bg-[#1e1e22]">
        <div className="h-2 rounded" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function ModelGroup({ title, models, isChampion }) {
  const maxMae = Math.max(...models.map((m) => m.mae));
  const maxRmse = Math.max(...models.map((m) => m.rmse));

  return (
    <section className="rounded-lg bg-s2-card border border-s2-border p-3">
      <h3 className="text-xs uppercase tracking-widest text-s2-muted mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] font-mono">
          <thead>
            <tr className="text-s2-muted text-left border-b border-s2-border">
              <th className="px-3 py-2 w-[180px]">Model</th>
              <th className="px-3 py-2">MAE</th>
              <th className="px-3 py-2">RMSE</th>
              <th className="px-3 py-2 w-16">R\u00b2</th>
              <th className="px-3 py-2 w-16">Status</th>
            </tr>
          </thead>
          <tbody className="text-s2-text">
            {models.map((m, i) => {
              const champion = i === 0;
              return (
                <tr key={m.model_name} className="border-b border-s2-border">
                  <td className={`px-3 py-2 ${champion ? "text-s2-blue" : "text-s2-muted"}`}>
                    {m.model_name}
                  </td>
                  <td className="px-3 py-2">
                    <MetricBar value={m.mae} max={maxMae} color={champion ? "#3b82f6" : "#71717a"} />
                  </td>
                  <td className="px-3 py-2">
                    <MetricBar value={m.rmse} max={maxRmse} color={champion ? "#3b82f6" : "#71717a"} />
                  </td>
                  <td className="px-3 py-2 text-center">{m.r2.toFixed(3)}</td>
                  <td className="px-3 py-2 text-center">
                    {champion ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-s2-blue/20 text-s2-blue">Champion</span>
                    ) : (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#27272a] text-s2-muted">Challenger</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-s2-muted mt-2 px-1">
        Lower MAE/RMSE is better. R\u00b2 closer to 1.0 indicates stronger fit. Champion is selected by lowest MAE.
      </p>
    </section>
  );
}

export default function ModelComparisonChart() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch("/v1/ml/models")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const { ghiModels, tempModels } = useMemo(() => {
    if (!data) return { ghiModels: [], tempModels: [] };
    const ghi = data.models.filter((m) => m.model_name.includes("GHI")).sort((a, b) => a.mae - b.mae);
    const temp = data.models.filter((m) => m.model_name.includes("Temp")).sort((a, b) => a.mae - b.mae);
    return { ghiModels: ghi, tempModels: temp };
  }, [data]);

  if (loading || !data) {
    return (
      <div className="rounded-lg bg-s2-card border border-s2-border p-6 text-center text-s2-muted text-sm">
        {loading ? "Loading model metrics..." : "No data"}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <ModelGroup title="GHI Prediction Models (W/m\u00b2)" models={ghiModels} />
      <ModelGroup title="Temperature Prediction Models (\u00b0C)" models={tempModels} />
    </div>
  );
}
