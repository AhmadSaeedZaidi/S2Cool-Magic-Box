// -------------------------------------------------------------------------
// BacktestChart.jsx — Actual vs Predicted GHI backtest with day filter.
//
// Fetches /v1/ml/backtest?city=X for 7 days and lets user select which
// days to display. Includes data table.
// -------------------------------------------------------------------------
import { useState, useEffect, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import DataTable from "./DataTable";

function LegendChip({ color, label, dashed }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-s2-muted">
      <span
        className="inline-block w-4 h-0.5"
        style={{
          backgroundColor: dashed ? "transparent" : color,
          borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
        }}
      />
      {label}
    </span>
  );
}

export default function BacktestChart({ city }) {
  const [raw, setRaw] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDays, setSelectedDays] = useState(7); // show last N days

  useEffect(() => {
    setLoading(true);
    fetch(`/v1/ml/backtest?city=${encodeURIComponent(city)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { setRaw(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [city]);

  // Get unique dates + filter
  const { chartData, availableDates, dailyStats, tableColumns } = useMemo(() => {
    if (!raw) return { chartData: [], availableDates: [], dailyStats: [], tableColumns: [] };

    const dates = [...new Set(raw.points.map((p) => p.date_utc))];
    const showDates = dates.slice(-selectedDays);

    const filtered = raw.points.filter((p) => showDates.includes(p.date_utc));
    const cd = filtered.map((p, i) => ({
      idx: i,
      label: `${p.date_utc.slice(5)} ${String(p.hour).padStart(2, "0")}h`,
      date: p.date_utc,
      hour: p.hour,
      actual: p.actual_ghi,
      predicted: p.predicted_ghi,
      error: Math.abs(p.actual_ghi - p.predicted_ghi),
    }));

    // Per-day stats
    const stats = showDates.map((d) => {
      const dayPts = cd.filter((p) => p.date === d);
      const errors = dayPts.map((p) => p.error);
      const mae = errors.reduce((a, v) => a + v, 0) / errors.length;
      const rmse = Math.sqrt(errors.map((e) => e * e).reduce((a, v) => a + v, 0) / errors.length);
      return { date: d, mae, rmse, points: dayPts.length };
    });

    const tCols = [
      { key: "label", label: "Timestamp" },
      { key: "actual", label: "Actual GHI", decimals: 1 },
      { key: "predicted", label: "Predicted GHI", decimals: 1 },
      { key: "error", label: "|Error|", decimals: 1 },
    ];

    return { chartData: cd, availableDates: dates, dailyStats: stats, tableColumns: tCols };
  }, [raw, selectedDays]);

  if (loading || !raw) {
    return (
      <div className="rounded-lg bg-s2-card border border-s2-border p-6 text-center text-s2-muted text-sm">
        {loading ? "Loading backtest..." : "No data"}
      </div>
    );
  }

  return (
    <section className="rounded-lg bg-s2-card border border-s2-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <h3 className="text-xs uppercase tracking-widest text-s2-muted">
          Backtest: Actual vs Predicted GHI — {city}
        </h3>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-[11px] uppercase tracking-wider text-s2-muted">Show last</label>
            <select
              value={selectedDays}
              onChange={(e) => setSelectedDays(Number(e.target.value))}
              className="h-[28px] rounded border border-s2-border bg-[#0f0f12] text-s2-text px-2 text-xs font-mono"
            >
              {[1, 2, 3, 5, 7].map((n) => (
                <option key={n} value={n}>{n} day{n > 1 ? "s" : ""}</option>
              ))}
            </select>
          </div>
          <LegendChip color="#22d3ee" label="Actual GHI" />
          <LegendChip color="#fbbf24" label="Predicted GHI" dashed />
        </div>
      </div>

      <div style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis
              dataKey="label"
              stroke="#a1a1aa"
              tick={{ fill: "#a1a1aa", fontSize: 9, fontFamily: "monospace" }}
              interval={23}
            />
            <YAxis
              stroke="#a1a1aa"
              tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "monospace" }}
              label={{ value: "W/m\u00b2", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#f4f4f5", fontFamily: "monospace" }}
            />
            <Line type="monotone" dataKey="actual" stroke="#22d3ee" strokeWidth={1.5} dot={false} name="Actual GHI" />
            <Line type="monotone" dataKey="predicted" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Predicted GHI" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Per-day error summary */}
      <div className="mt-3">
        <h4 className="text-[11px] uppercase tracking-widest text-s2-muted mb-2">Daily Error Summary</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-s2-muted text-left border-b border-s2-border">
                <th className="px-2.5 py-1.5">Date</th>
                <th className="px-2.5 py-1.5">MAE (W/m\u00b2)</th>
                <th className="px-2.5 py-1.5">RMSE (W/m\u00b2)</th>
                <th className="px-2.5 py-1.5">Points</th>
              </tr>
            </thead>
            <tbody className="text-s2-text">
              {dailyStats.map((s) => (
                <tr key={s.date} className="border-b border-s2-border">
                  <td className="px-2.5 py-1">{s.date}</td>
                  <td className="px-2.5 py-1">{s.mae.toFixed(2)}</td>
                  <td className="px-2.5 py-1">{s.rmse.toFixed(2)}</td>
                  <td className="px-2.5 py-1">{s.points}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="mt-2">
        <DataTable title="Hourly Backtest Data" columns={tableColumns} rows={chartData} />
      </div>
    </section>
  );
}
