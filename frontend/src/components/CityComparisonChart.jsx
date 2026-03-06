// -------------------------------------------------------------------------
// CityComparisonChart.jsx — 4-city GHI overlay + summary table.
//
// Fetches /v1/compare/cities on date change. Includes hour-range filter
// so user can zoom into a specific time window.
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

const CITY_COLORS = {
  Islamabad: "#3b82f6",
  Lahore: "#fbbf24",
  Karachi: "#22d3ee",
  Peshawar: "#a78bfa",
};

function LegendChip({ color, label }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-s2-muted">
      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

export default function CityComparisonChart({ date }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hourRange, setHourRange] = useState([0, 23]);

  useEffect(() => {
    setLoading(true);
    fetch("/v1/compare/cities", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date_utc: date }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [date]);

  // Build merged 24-hour array filtered by hour range
  const { merged, summaryRows, tableRows, tableColumns } = useMemo(() => {
    if (!data) return { merged: [], summaryRows: [], tableRows: [], tableColumns: [] };

    const cities = Object.keys(CITY_COLORS);
    const allMerged = Array.from({ length: 24 }, (_, h) => {
      const row = { hour: h, hourLabel: `${String(h).padStart(2, "0")}:00` };
      data.cities.forEach((c) => {
        row[c.city] = c.hours[h]?.predicted_ghi_wm2 ?? 0;
      });
      return row;
    });

    const filtered = allMerged.filter((r) => r.hour >= hourRange[0] && r.hour <= hourRange[1]);

    // Summary: per-city peak GHI, average GHI, operating %
    const sRows = data.cities.map((c) => {
      const hrs = filtered.map((r) => r[c.city]);
      const peak = Math.max(...hrs);
      const avg = hrs.reduce((a, v) => a + v, 0) / hrs.length;
      return {
        city: c.city,
        operatingPct: c.operating_pct,
        peakGhi: peak,
        avgGhi: avg,
        psh: hrs.reduce((a, v) => a + v, 0) / 1000,
      };
    });

    // Table columns and rows for raw hourly data
    const tCols = [
      { key: "hourLabel", label: "Hour" },
      ...cities.map((c) => ({ key: c, label: `${c} (W/m\u00b2)`, decimals: 1 })),
    ];

    return { merged: filtered, summaryRows: sRows, tableRows: filtered, tableColumns: tCols };
  }, [data, hourRange]);

  if (loading || !data) {
    return (
      <div className="rounded-lg bg-s2-card border border-s2-border p-6 text-center text-s2-muted text-sm">
        {loading ? "Loading city comparison..." : "No data"}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* GHI overlay chart */}
      <section className="rounded-lg bg-s2-card border border-s2-border p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <h3 className="text-xs uppercase tracking-widest text-s2-muted">
            4-City GHI Comparison — {date}
          </h3>
          <div className="flex flex-wrap gap-3">
            {Object.entries(CITY_COLORS).map(([city, color]) => (
              <LegendChip key={city} color={color} label={city} />
            ))}
          </div>
        </div>

        {/* Hour range filter */}
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <label className="text-[11px] uppercase tracking-wider text-s2-muted">Hour Range:</label>
          <div className="flex items-center gap-2">
            <input
              type="number" min="0" max="23"
              value={hourRange[0]}
              onChange={(e) => setHourRange([Math.min(Number(e.target.value), hourRange[1]), hourRange[1]])}
              className="w-14 h-[30px] rounded border border-s2-border bg-[#0f0f12] text-s2-text px-2 text-xs font-mono"
            />
            <span className="text-s2-muted text-xs">to</span>
            <input
              type="number" min="0" max="23"
              value={hourRange[1]}
              onChange={(e) => setHourRange([hourRange[0], Math.max(Number(e.target.value), hourRange[0])])}
              className="w-14 h-[30px] rounded border border-s2-border bg-[#0f0f12] text-s2-text px-2 text-xs font-mono"
            />
          </div>
        </div>

        <div style={{ height: 340 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis
                dataKey="hourLabel"
                stroke="#a1a1aa"
                tick={{ fill: "#a1a1aa", fontSize: 11, fontFamily: "monospace" }}
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
              {Object.entries(CITY_COLORS).map(([city, color]) => (
                <Line key={city} type="monotone" dataKey={city} stroke={color} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <DataTable title="Hourly GHI by City" columns={tableColumns} rows={tableRows} />
      </section>

      {/* City summary table (replaces the ugly bar chart) */}
      <section className="rounded-lg bg-s2-card border border-s2-border p-3">
        <h3 className="text-xs uppercase tracking-widest text-s2-muted mb-3">
          City Performance Summary
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px] font-mono">
            <thead>
              <tr className="text-s2-muted text-left border-b border-s2-border">
                <th className="px-3 py-2">City</th>
                <th className="px-3 py-2">Operating %</th>
                <th className="px-3 py-2">Peak GHI (W/m\u00b2)</th>
                <th className="px-3 py-2">Avg GHI (W/m\u00b2)</th>
                <th className="px-3 py-2">PSH (h)</th>
              </tr>
            </thead>
            <tbody className="text-s2-text">
              {summaryRows.map((r) => (
                <tr key={r.city} className="border-b border-s2-border">
                  <td className="px-3 py-2" style={{ color: CITY_COLORS[r.city] }}>{r.city}</td>
                  <td className="px-3 py-2">{r.operatingPct.toFixed(1)}%</td>
                  <td className="px-3 py-2">{r.peakGhi.toFixed(1)}</td>
                  <td className="px-3 py-2">{r.avgGhi.toFixed(1)}</td>
                  <td className="px-3 py-2">{r.psh.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
