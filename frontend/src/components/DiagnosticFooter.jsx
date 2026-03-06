// -------------------------------------------------------------------------
// DiagnosticFooter.jsx — AI model metadata and live error metrics.
// -------------------------------------------------------------------------

/** Single diagnostic key/value pair. */
function DiagItem({ label, value }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] uppercase tracking-wider text-s2-muted">
        {label}
      </span>
      <span className="text-[13px] font-mono text-s2-text">{value}</span>
    </div>
  );
}

export default function DiagnosticFooter({
  loading,
  error,
  sourceMode,
  prodMetrics,
}) {
  const modelName = prodMetrics?.model_name || "XGBoost_GHI_v1 & XGBoost_Temp_v1";
  const rmse = prodMetrics?.rmse != null ? Number(prodMetrics.rmse).toFixed(3) : "--";
  const mae = prodMetrics?.mae != null ? Number(prodMetrics.mae).toFixed(3) : "--";
  const inertia =
    prodMetrics?.thermal_inertia_summary ||
    "Peak load expected 2h after solar peak";

  // Data source label
  let statusLabel = sourceMode === "PREDICTION" ? "AI Forecast" : "Historical Record";
  if (loading) statusLabel = "Loading...";
  if (error) statusLabel = `Error: ${error}`;

  return (
    <footer className="flex flex-wrap items-start justify-between gap-4 rounded-lg bg-s2-card border border-s2-border px-4 py-3">
      <div className="flex flex-wrap items-center gap-5">
        <DiagItem label="Active Models" value={modelName} />
        <DiagItem label="RMSE" value={rmse} />
        <DiagItem label="MAE" value={mae} />
        <DiagItem label="Data Source" value={statusLabel} />
      </div>
      <div className="flex flex-col gap-1 items-end text-right">
        <span className="text-xs text-s2-muted">
          Thermal Inertia: {inertia}
        </span>
      </div>
    </footer>
  );
}
