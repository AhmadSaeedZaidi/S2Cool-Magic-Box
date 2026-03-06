// -------------------------------------------------------------------------
// SeasonalChart.jsx — 4 seasonal average GHI curves for a given city.
//
// Fetches /v1/compare/seasonal?city=X and draws Summer / Spring /
// Winter / Autumn. Includes temp overlay and summary table.
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

const SEASON_COLORS = {
  Summer: "#fbbf24",
  Autumn: "#f97316",
  Winter: "#3b82f6",
  Spring: "#22c55e",
};

function LegendChip({ color, label }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-s2-muted">
      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

export default function SeasonalChart({ city }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`/v1/compare/seasonal?city=${encodeURIComponent(city)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, [city]);

  const { merged, summaryRows, tableColumns } = useMemo(() => {
    if (!data) return { merged: [], summaryRows: [], tableColumns: [] };

    const seasons = Object.keys(SEASON_COLORS);
    const m = Array.from({ length: 24 }, (_, h) => {
      const row = { hour: h, hourLabel: `${String(h).padStart(2, "0")}:00` };
      data.curves.forEach((curve) => {
        const pt = curve.hours.find((p) => p.hour === h);
        row[curve.season] = pt ? pt.avg_ghi_wm2 : 0;
        row[`${curve.season}_temp`] = pt ? pt.avg_temp_c : 0;
      });
      return row;
    });

    // Summary per season
    const sRows = data.curves.map((curve) => {
      const ghiVals = curve.hours.map((p) => p.avg_ghi_wm2);
      const tempVals = curve.hours.map((p) => p.avg_temp_c);
      return {
        season: curve.season,
        peakGhi: Math.max(...ghiVals),
        avgGhi: ghiVals.reduce((a, v) => a + v, 0) / ghiVals.length,
        psh: ghiVals.reduce((a, v) => a + v, 0) / 1000,
        avgTemp: tempVals.reduce((a, v) => a + v, 0) / tempVals.length,
        peakTemp: Math.max(...tempVals),
        sunriseHour: ghiVals.findIndex((v) => v > 10),
        sunsetHour: 23 - [...ghiVals].reverse().findIndex((v) => v > 10),
      };
    });

    const tCols = [
      { key: "hourLabel", label: "Hour" },
      ...seasons.map((s) => ({ key: s, label: `${s} GHI`, decimals: 1 })),
    ];

    return { merged: m, summaryRows: sRows, tableColumns: tCols };
  }, [data]);

  if (loading || !data) {
    return (
      <div className="rounded-lg bg-s2-card border border-s2-border p-6 text-center text-s2-muted text-sm">
        {loading ? "Loading seasonal data..." : "No data"}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <section className="rounded-lg bg-s2-card border border-s2-border p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <h3 className="text-xs uppercase tracking-widest text-s2-muted">
            Seasonal GHI Variation — {city}
          </h3>
          <div className="flex flex-wrap gap-3">
            {Object.entries(SEASON_COLORS).map(([s, color]) => (
              <LegendChip key={s} color={color} label={s} />
            ))}
          </div>
        </div>
        <div style={{ height: 320 }}>
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
              {Object.entries(SEASON_COLORS).map(([season, color]) => (
                <Line key={season} type="monotone" dataKey={season} stroke={color} strokeWidth={2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <DataTable title="Seasonal Hourly GHI Data" columns={tableColumns} rows={merged} />
      </section>

      {/* Seasonal summary table */}
      <section className="rounded-lg bg-s2-card border border-s2-border p-3">
        <h3 className="text-xs uppercase tracking-widest text-s2-muted mb-3">
          Seasonal Performance Summary — {city}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px] font-mono">
            <thead>
              <tr className="text-s2-muted text-left border-b border-s2-border">
                <th className="px-3 py-2">Season</th>
                <th className="px-3 py-2">Peak GHI</th>
                <th className="px-3 py-2">Avg GHI</th>
                <th className="px-3 py-2">PSH (h)</th>
                <th className="px-3 py-2">Avg Temp</th>
                <th className="px-3 py-2">Peak Temp</th>
                <th className="px-3 py-2">Sunrise</th>
                <th className="px-3 py-2">Sunset</th>
              </tr>
            </thead>
            <tbody className="text-s2-text">
              {summaryRows.map((r) => (
                <tr key={r.season} className="border-b border-s2-border">
                  <td className="px-3 py-2" style={{ color: SEASON_COLORS[r.season] }}>{r.season}</td>
                  <td className="px-3 py-2">{r.peakGhi.toFixed(1)} W/m\u00b2</td>
                  <td className="px-3 py-2">{r.avgGhi.toFixed(1)} W/m\u00b2</td>
                  <td className="px-3 py-2">{r.psh.toFixed(2)}</td>
                  <td className="px-3 py-2">{r.avgTemp.toFixed(1)}\u00b0C</td>
                  <td className="px-3 py-2">{r.peakTemp.toFixed(1)}\u00b0C</td>
                  <td className="px-3 py-2">{r.sunriseHour >= 0 ? `${String(r.sunriseHour).padStart(2, "0")}:00` : "--"}</td>
                  <td className="px-3 py-2">{r.sunsetHour <= 23 ? `${String(r.sunsetHour).padStart(2, "0")}:00` : "--"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
