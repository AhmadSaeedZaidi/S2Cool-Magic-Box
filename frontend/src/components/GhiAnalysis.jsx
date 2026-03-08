// -------------------------------------------------------------------------
// GhiAnalysis.jsx — Dedicated GHI deep-dive section.
//
// Sub-sections:
//   1. Statistics cards (peak, avg, total irradiance, PSH, zero-hours, daylight)
//   2. Daily GHI profile (area chart + annotations)
//   3. GHI vs Temperature scatter plot
//   4. 7-day trend (multi-line overlay + heatmap grid)
//   5. 4-city GHI comparison (line chart)
//   6. Seasonal GHI variation (line chart)
//   7. GHI distribution histogram
//   All sections include collapsible data tables.
// -------------------------------------------------------------------------
import { useState, useEffect, useMemo } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from "recharts";
import DataTable from "./DataTable";

const CITY_COLORS = {
  Islamabad: "#3b82f6",
  Lahore: "#fbbf24",
  Karachi: "#22d3ee",
  Peshawar: "#a78bfa",
};
const SEASON_COLORS = {
  Summer: "#fbbf24",
  Autumn: "#f97316",
  Winter: "#3b82f6",
  Spring: "#22c55e",
};
const DAY_COLORS = [
  "#fbbf24",
  "#3b82f6",
  "#22d3ee",
  "#a78bfa",
  "#f97316",
  "#ef4444",
  "#22c55e",
];

// ---- Reusable sub-components ----

function StatCard({ label, value, unit, sub }) {
  return (
    <div className="rounded-lg bg-[#101014] border border-s2-border p-3 text-center">
      <p className="text-[10px] uppercase tracking-widest text-s2-muted mb-1">
        {label}
      </p>
      <p className="text-xl font-bold text-s2-text font-mono">
        {value}
        {unit && (
          <span className="text-xs font-normal text-s2-muted ml-1">
            {unit}
          </span>
        )}
      </p>
      {sub && <p className="text-[10px] text-s2-muted mt-0.5">{sub}</p>}
    </div>
  );
}

function SectionCard({ title, info, children }) {
  return (
    <section className="rounded-lg bg-s2-card border border-s2-border p-4">
      <div className="flex items-baseline gap-2 mb-3">
        <h3 className="text-xs uppercase tracking-widest text-s2-muted">
          {title}
        </h3>
        {info && (
          <span className="text-[10px] text-s2-muted italic">{info}</span>
        )}
      </div>
      {children}
    </section>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg bg-[#101014] border border-s2-border p-2 text-xs font-mono shadow-lg">
      <p className="text-s2-muted mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || p.stroke || "#fbbf24" }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  );
}

// ---- Main component ----

export default function GhiAnalysis({ city, date }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Fetch from /v1/ghi/analysis
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    fetch("/v1/ghi/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ city, date_utc: date }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`API ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [city, date]);

  if (loading)
    return (
      <div className="text-center py-20 text-s2-muted text-sm">
        Loading GHI analysis…
      </div>
    );
  if (error)
    return (
      <div className="text-center py-20 text-s2-red text-sm">
        Error: {error}
      </div>
    );
  if (!data) return null;

  const { statistics: stats, weekly_trend, city_comparison, seasonal } = data;

  return (
    <div className="flex flex-col gap-3 flex-1">
      {/* ====== 1. KPI Cards ====== */}
      <StatsRow stats={stats} />

      {/* ====== 2. Daily GHI Profile ====== */}
      <DailyProfileChart stats={stats} />

      {/* ====== 3. GHI vs Temperature ====== */}
      <GhiTempCorrelation stats={stats} />

      {/* ====== 4. 7-Day Trend ====== */}
      <WeeklyTrend weeklyTrend={weekly_trend} />

      {/* ====== 5. City Comparison ====== */}
      <CityGhiComparison cityComparison={city_comparison} />

      {/* ====== 6. Seasonal Variation ====== */}
      <SeasonalGhi seasonal={seasonal} city={city} />

      {/* ====== 7. GHI Distribution ====== */}
      <GhiDistribution stats={stats} />
    </div>
  );
}

// =====================================================================
// Sub-section components
// =====================================================================

// 1. Statistics cards
function StatsRow({ stats }) {
  const daylightHours = 24 - stats.zero_hours;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
      <StatCard
        label="Peak GHI"
        value={stats.peak_ghi.toFixed(1)}
        unit="W/m²"
        sub="Maximum hourly irradiance"
      />
      <StatCard
        label="Average GHI"
        value={stats.avg_ghi.toFixed(1)}
        unit="W/m²"
        sub="24-hour mean"
      />
      <StatCard
        label="Total Irradiance"
        value={(stats.total_irradiance_whm2 / 1000).toFixed(2)}
        unit="kWh/m²"
        sub="Daily solar energy density"
      />
      <StatCard
        label="Peak Sun Hours"
        value={stats.psh.toFixed(2)}
        unit="hours"
        sub="Equivalent full-sun hours"
      />
      <StatCard
        label="Daylight Hours"
        value={daylightHours}
        unit="hrs"
        sub={
          stats.sunrise_hour != null
            ? `${String(stats.sunrise_hour).padStart(2, "0")}:00 – ${String(stats.sunset_hour).padStart(2, "0")}:00`
            : "—"
        }
      />
      <StatCard
        label="Zero-GHI Hours"
        value={stats.zero_hours}
        unit="/ 24"
        sub="No solar generation possible"
      />
    </div>
  );
}

// 2. Daily GHI Profile
function DailyProfileChart({ stats }) {
  const chartData = stats.hours.map((h) => ({
    ...h,
    hourLabel: `${String(h.hour).padStart(2, "0")}:00`,
  }));

  const peakHour = stats.hours.reduce(
    (best, h) => (h.ghi > best.ghi ? h : best),
    stats.hours[0]
  );

  return (
    <SectionCard
      title="Daily GHI Profile"
      info="Hourly Global Horizontal Irradiance with daylight boundaries"
    >
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={chartData}>
          <defs>
            <linearGradient id="ghiGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="hourLabel" tick={{ fill: "#71717a", fontSize: 10 }} />
          <YAxis
            yAxisId="ghi"
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI (W/m²)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <YAxis
            yAxisId="temp"
            orientation="right"
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "Temp (°C)",
              angle: 90,
              position: "insideRight",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
            iconSize={10}
          />
          {stats.sunrise_hour != null && (
            <ReferenceLine
              yAxisId="ghi"
              x={`${String(stats.sunrise_hour).padStart(2, "0")}:00`}
              stroke="#22c55e"
              strokeDasharray="4 4"
              label={{
                value: "Sunrise",
                fill: "#22c55e",
                fontSize: 10,
                position: "top",
              }}
            />
          )}
          {stats.sunset_hour != null && (
            <ReferenceLine
              yAxisId="ghi"
              x={`${String(stats.sunset_hour).padStart(2, "0")}:00`}
              stroke="#f97316"
              strokeDasharray="4 4"
              label={{
                value: "Sunset",
                fill: "#f97316",
                fontSize: 10,
                position: "top",
              }}
            />
          )}
          <ReferenceLine
            yAxisId="ghi"
            x={`${String(peakHour.hour).padStart(2, "0")}:00`}
            stroke="#ef4444"
            strokeDasharray="2 2"
            label={{
              value: `Peak ${peakHour.ghi.toFixed(0)}`,
              fill: "#ef4444",
              fontSize: 10,
              position: "top",
            }}
          />
          <Area
            yAxisId="ghi"
            type="monotone"
            dataKey="ghi"
            name="GHI (W/m²)"
            stroke="#fbbf24"
            fill="url(#ghiGrad)"
            strokeWidth={2}
          />
          <Line
            yAxisId="temp"
            type="monotone"
            dataKey="temp"
            name="Temp (°C)"
            stroke="#22d3ee"
            strokeWidth={1.5}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Callout */}
      <p className="text-[11px] text-s2-muted mt-2 px-1">
        <span className="text-s2-gold font-semibold">GHI = 0</span> during
        nighttime hours ({stats.zero_hours} of 24 hours).
      </p>

      <DataTable
        title="Hourly GHI Data"
        columns={[
          { key: "hourLabel", label: "Hour" },
          { key: "ghi", label: "GHI (W/m²)", decimals: 2 },
          { key: "temp", label: "Temp (°C)", decimals: 2 },
        ]}
        rows={chartData}
      />
    </SectionCard>
  );
}

// 3. GHI vs Temperature scatter plot
function GhiTempCorrelation({ stats }) {
  // Only include hours with GHI > 0 for meaningful correlation
  const scatterData = stats.hours
    .filter((h) => h.ghi > 0)
    .map((h) => ({
      ghi: h.ghi,
      temp: h.temp,
      hour: h.hour,
      hourLabel: `${String(h.hour).padStart(2, "0")}:00`,
    }));

  // Simple linear regression for trend line
  const regression = useMemo(() => {
    if (scatterData.length < 2) return null;
    const n = scatterData.length;
    const sumX = scatterData.reduce((s, d) => s + d.ghi, 0);
    const sumY = scatterData.reduce((s, d) => s + d.temp, 0);
    const sumXY = scatterData.reduce((s, d) => s + d.ghi * d.temp, 0);
    const sumXX = scatterData.reduce((s, d) => s + d.ghi * d.ghi, 0);
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // Correlation coefficient
    const meanX = sumX / n;
    const meanY = sumY / n;
    const ssX = scatterData.reduce((s, d) => s + (d.ghi - meanX) ** 2, 0);
    const ssY = scatterData.reduce((s, d) => s + (d.temp - meanY) ** 2, 0);
    const ssXY = scatterData.reduce(
      (s, d) => s + (d.ghi - meanX) * (d.temp - meanY),
      0
    );
    const r = ssX > 0 && ssY > 0 ? ssXY / Math.sqrt(ssX * ssY) : 0;

    // Two endpoints for the trend line
    const minGhi = Math.min(...scatterData.map((d) => d.ghi));
    const maxGhi = Math.max(...scatterData.map((d) => d.ghi));
    return {
      slope,
      intercept,
      r: r.toFixed(3),
      line: [
        { ghi: minGhi, temp: slope * minGhi + intercept },
        { ghi: maxGhi, temp: slope * maxGhi + intercept },
      ],
    };
  }, [scatterData]);

  return (
    <SectionCard
      title="GHI vs Temperature Correlation"
      info={
        regression
          ? `Pearson r = ${regression.r} (daylight hours only)`
          : "Insufficient data"
      }
    >
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            type="number"
            dataKey="ghi"
            name="GHI"
            unit=" W/m²"
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI (W/m²)",
              position: "insideBottom",
              offset: -5,
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <YAxis
            type="number"
            dataKey="temp"
            name="Temp"
            unit=" °C"
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "Temp (°C)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="rounded-lg bg-[#101014] border border-s2-border p-2 text-xs font-mono shadow-lg">
                  <p className="text-s2-muted">{d.hourLabel}</p>
                  <p className="text-s2-gold">
                    GHI: {d.ghi.toFixed(1)} W/m²
                  </p>
                  <p className="text-s2-cyan">
                    Temp: {d.temp.toFixed(1)} °C
                  </p>
                </div>
              );
            }}
          />
          <Scatter
            name="Daylight Hours"
            data={scatterData}
            fill="#fbbf24"
            fillOpacity={0.7}
            r={5}
          />
          {regression && (
            <Scatter
              name="Trend Line"
              data={regression.line}
              fill="none"
              line={{ stroke: "#ef4444", strokeWidth: 2, strokeDasharray: "6 3" }}
              legendType="line"
              r={0}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>

      <DataTable
        title="Correlation Data (Daylight Hours)"
        columns={[
          { key: "hourLabel", label: "Hour" },
          { key: "ghi", label: "GHI (W/m²)", decimals: 1 },
          { key: "temp", label: "Temp (°C)", decimals: 1 },
        ]}
        rows={scatterData}
      />
    </SectionCard>
  );
}

// 4. Weekly trend
function WeeklyTrend({ weeklyTrend }) {
  // Multi-line overlay: each day is a separate series on 0-23 hour axis
  const overlayData = useMemo(() => {
    const hours = Array.from({ length: 24 }, (_, i) => {
      const row = { hour: i, hourLabel: `${String(i).padStart(2, "0")}:00` };
      weeklyTrend.forEach((day) => {
        const pt = day.hours.find((h) => h.hour === i);
        row[day.date_utc] = pt ? pt.ghi : 0;
      });
      return row;
    });
    return hours;
  }, [weeklyTrend]);

  // Heatmap grid data
  const heatmapData = useMemo(() => {
    return weeklyTrend.map((day) => ({
      date: day.date_utc,
      peak: day.peak_ghi,
      avg: day.avg_ghi,
      psh: day.psh,
      zero: day.zero_hours,
      hours: day.hours,
    }));
  }, [weeklyTrend]);

  // Summary table rows
  const summaryRows = weeklyTrend.map((d) => ({
    date: d.date_utc,
    peak: d.peak_ghi,
    avg: d.avg_ghi,
    totalKwh: (d.total_irradiance_whm2 / 1000).toFixed(2),
    psh: d.psh,
    zero: d.zero_hours,
    sunrise: d.sunrise_hour != null ? `${String(d.sunrise_hour).padStart(2, "0")}:00` : "—",
    sunset: d.sunset_hour != null ? `${String(d.sunset_hour).padStart(2, "0")}:00` : "—",
  }));

  return (
    <SectionCard
      title="7-Day GHI Trend"
      info="Daily GHI profiles overlaid + summary statistics"
    >
      {/* Multi-line overlay */}
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={overlayData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="hourLabel"
            tick={{ fill: "#71717a", fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI (W/m²)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 10, color: "#a1a1aa" }}
            iconSize={8}
          />
          {weeklyTrend.map((day, i) => (
            <Line
              key={day.date_utc}
              type="monotone"
              dataKey={day.date_utc}
              name={day.date_utc}
              stroke={DAY_COLORS[i % DAY_COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              strokeOpacity={i === weeklyTrend.length - 1 ? 1 : 0.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Heatmap grid */}
      <div className="mt-4">
        <h4 className="text-[11px] uppercase tracking-widest text-s2-muted mb-2">
          GHI Heatmap (W/m²)
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono border-collapse">
            <thead>
              <tr>
                <th className="text-left text-s2-muted p-1 sticky left-0 bg-s2-card">
                  Date
                </th>
                {Array.from({ length: 24 }, (_, i) => (
                  <th key={i} className="text-center text-s2-muted p-1 min-w-[28px]">
                    {String(i).padStart(2, "0")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmapData.map((day) => (
                <tr key={day.date}>
                  <td className="text-s2-text p-1 whitespace-nowrap sticky left-0 bg-s2-card">
                    {day.date}
                  </td>
                  {day.hours.map((h) => {
                    const intensity = Math.min(h.ghi / 1000, 1);
                    return (
                      <td
                        key={h.hour}
                        className="p-1 text-center"
                        style={{
                          backgroundColor: `rgba(251, 191, 36, ${intensity * 0.8})`,
                          color: intensity > 0.5 ? "#09090b" : "#a1a1aa",
                        }}
                        title={`${day.date} ${String(h.hour).padStart(2, "0")}:00 — ${h.ghi.toFixed(0)} W/m²`}
                      >
                        {h.ghi > 0 ? h.ghi.toFixed(0) : "·"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary table */}
      <div className="mt-4">
        <h4 className="text-[11px] uppercase tracking-widest text-s2-muted mb-2">
          Daily Summary
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-s2-muted text-left border-b border-s2-border">
                <th className="pb-1 pr-3">Date</th>
                <th className="pb-1 pr-3">Peak (W/m²)</th>
                <th className="pb-1 pr-3">Avg (W/m²)</th>
                <th className="pb-1 pr-3">Total (kWh/m²)</th>
                <th className="pb-1 pr-3">Peak Sun Hours</th>
                <th className="pb-1 pr-3">Zero Hrs</th>
                <th className="pb-1 pr-3">Sunrise</th>
                <th className="pb-1">Sunset</th>
              </tr>
            </thead>
            <tbody className="text-s2-text">
              {summaryRows.map((r) => (
                <tr key={r.date} className="border-b border-s2-border/50">
                  <td className="py-1 pr-3">{r.date}</td>
                  <td className="py-1 pr-3 text-s2-gold">{r.peak.toFixed(1)}</td>
                  <td className="py-1 pr-3">{r.avg.toFixed(1)}</td>
                  <td className="py-1 pr-3">{r.totalKwh}</td>
                  <td className="py-1 pr-3">{r.psh.toFixed(2)}</td>
                  <td className="py-1 pr-3">{r.zero}</td>
                  <td className="py-1 pr-3">{r.sunrise}</td>
                  <td className="py-1">{r.sunset}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </SectionCard>
  );
}

// 5. City GHI comparison
function CityGhiComparison({ cityComparison }) {
  const chartData = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => {
      const row = { hour: i, hourLabel: `${String(i).padStart(2, "0")}:00` };
      cityComparison.forEach((c) => {
        const pt = c.hours.find((h) => h.hour === i);
        row[c.city] = pt ? pt.ghi : 0;
      });
      return row;
    });
  }, [cityComparison]);

  const summaryRows = cityComparison.map((c) => ({
    city: c.city,
    peak: c.peak_ghi,
    avg: c.avg_ghi,
    psh: c.psh,
    totalKwh: (c.hours.reduce((s, h) => s + h.ghi, 0) / 1000).toFixed(2),
  }));

  return (
    <SectionCard
      title="City GHI Comparison"
      info="Same date across all 4 cities"
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="hourLabel"
            tick={{ fill: "#71717a", fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI (W/m²)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
            iconSize={10}
          />
          {cityComparison.map((c) => (
            <Line
              key={c.city}
              type="monotone"
              dataKey={c.city}
              stroke={CITY_COLORS[c.city] || "#71717a"}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {/* Summary table */}
      <div className="mt-3">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-s2-muted text-left border-b border-s2-border">
              <th className="pb-1 pr-3">City</th>
              <th className="pb-1 pr-3">Peak (W/m²)</th>
              <th className="pb-1 pr-3">Avg (W/m²)</th>
              <th className="pb-1 pr-3">Total (kWh/m²)</th>
              <th className="pb-1">Peak Sun Hours</th>
            </tr>
          </thead>
          <tbody className="text-s2-text">
            {summaryRows.map((r) => (
              <tr key={r.city} className="border-b border-s2-border/50">
                <td
                  className="py-1 pr-3 font-semibold"
                  style={{ color: CITY_COLORS[r.city] || "#a1a1aa" }}
                >
                  {r.city}
                </td>
                <td className="py-1 pr-3 text-s2-gold">{r.peak.toFixed(1)}</td>
                <td className="py-1 pr-3">{r.avg.toFixed(1)}</td>
                <td className="py-1 pr-3">{r.totalKwh}</td>
                <td className="py-1">{r.psh.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <DataTable
        title="Hourly GHI by City"
        columns={[
          { key: "hourLabel", label: "Hour" },
          ...cityComparison.map((c) => ({
            key: c.city,
            label: `${c.city} (W/m²)`,
            decimals: 1,
          })),
        ]}
        rows={chartData}
      />
    </SectionCard>
  );
}

// 6. Seasonal GHI variation
function SeasonalGhi({ seasonal, city }) {
  const chartData = useMemo(() => {
    return Array.from({ length: 24 }, (_, i) => {
      const row = { hour: i, hourLabel: `${String(i).padStart(2, "0")}:00` };
      seasonal.forEach((s) => {
        const pt = s.hours.find((h) => h.hour === i);
        row[s.season] = pt ? pt.avg_ghi_wm2 : 0;
        row[`${s.season}_temp`] = pt ? pt.avg_temp_c : 0;
      });
      return row;
    });
  }, [seasonal]);

  const summaryRows = seasonal.map((s) => {
    const ghiVals = s.hours.map((h) => h.avg_ghi_wm2);
    const tempVals = s.hours.map((h) => h.avg_temp_c);
    const sunrise = s.hours.findIndex((h) => h.avg_ghi_wm2 >= 1);
    const sunsetIdx = [...s.hours].reverse().findIndex((h) => h.avg_ghi_wm2 >= 1);
    const sunset = sunsetIdx >= 0 ? 23 - sunsetIdx : null;
    return {
      season: s.season,
      peakGhi: Math.max(...ghiVals).toFixed(1),
      avgGhi: (ghiVals.reduce((a, b) => a + b, 0) / 24).toFixed(1),
      psh: (ghiVals.reduce((a, b) => a + b, 0) / 1000).toFixed(2),
      avgTemp: (tempVals.reduce((a, b) => a + b, 0) / 24).toFixed(1),
      sunrise: sunrise >= 0 ? `${String(sunrise).padStart(2, "0")}:00` : "—",
      sunset: sunset != null ? `${String(sunset).padStart(2, "0")}:00` : "—",
    };
  });

  return (
    <SectionCard
      title={`Seasonal GHI Variation — ${city}`}
      info="Representative day per season"
    >
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData}>
          <defs>
            {seasonal.map((s) => (
              <linearGradient
                key={s.season}
                id={`grad_${s.season}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="0%"
                  stopColor={SEASON_COLORS[s.season]}
                  stopOpacity={0.3}
                />
                <stop
                  offset="100%"
                  stopColor={SEASON_COLORS[s.season]}
                  stopOpacity={0.02}
                />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="hourLabel"
            tick={{ fill: "#71717a", fontSize: 10 }}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI (W/m²)",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "#a1a1aa" }}
            iconSize={10}
          />
          {seasonal.map((s) => (
            <Area
              key={s.season}
              type="monotone"
              dataKey={s.season}
              name={s.season}
              stroke={SEASON_COLORS[s.season]}
              fill={`url(#grad_${s.season})`}
              strokeWidth={2}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>

      {/* Seasonal summary */}
      <div className="mt-3">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-s2-muted text-left border-b border-s2-border">
              <th className="pb-1 pr-3">Season</th>
              <th className="pb-1 pr-3">Peak GHI</th>
              <th className="pb-1 pr-3">Avg GHI</th>
              <th className="pb-1 pr-3">Peak Sun Hours</th>
              <th className="pb-1 pr-3">Avg Temp</th>
              <th className="pb-1 pr-3">Sunrise</th>
              <th className="pb-1">Sunset</th>
            </tr>
          </thead>
          <tbody className="text-s2-text">
            {summaryRows.map((r) => (
              <tr key={r.season} className="border-b border-s2-border/50">
                <td
                  className="py-1 pr-3 font-semibold"
                  style={{ color: SEASON_COLORS[r.season] }}
                >
                  {r.season}
                </td>
                <td className="py-1 pr-3 text-s2-gold">{r.peakGhi}</td>
                <td className="py-1 pr-3">{r.avgGhi}</td>
                <td className="py-1 pr-3">{r.psh}</td>
                <td className="py-1 pr-3 text-s2-cyan">{r.avgTemp} °C</td>
                <td className="py-1 pr-3">{r.sunrise}</td>
                <td className="py-1">{r.sunset}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <DataTable
        title="Hourly Seasonal GHI"
        columns={[
          { key: "hourLabel", label: "Hour" },
          ...seasonal.map((s) => ({
            key: s.season,
            label: `${s.season} (W/m²)`,
            decimals: 1,
          })),
        ]}
        rows={chartData}
      />
    </SectionCard>
  );
}

// 7. GHI Distribution histogram
function GhiDistribution({ stats }) {
  // Build bins of 100 W/m² width
  const bins = useMemo(() => {
    const buckets = {};
    const binWidth = 100;
    stats.hours.forEach((h) => {
      const binStart = Math.floor(h.ghi / binWidth) * binWidth;
      const label = `${binStart}–${binStart + binWidth}`;
      if (!buckets[label])
        buckets[label] = { range: label, binStart, count: 0, hours: [] };
      buckets[label].count += 1;
      buckets[label].hours.push(
        `${String(h.hour).padStart(2, "0")}:00`
      );
    });
    return Object.values(buckets).sort((a, b) => a.binStart - b.binStart);
  }, [stats.hours]);

  return (
    <SectionCard
      title="GHI Distribution"
      info="Frequency of GHI values across 24 hours"
    >
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={bins}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="range"
            tick={{ fill: "#71717a", fontSize: 10 }}
            label={{
              value: "GHI Range (W/m²)",
              position: "insideBottom",
              offset: -5,
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            allowDecimals={false}
            label={{
              value: "Hours",
              angle: -90,
              position: "insideLeft",
              style: { fill: "#71717a", fontSize: 10 },
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="rounded-lg bg-[#101014] border border-s2-border p-2 text-xs font-mono shadow-lg">
                  <p className="text-s2-muted">{d.range} W/m²</p>
                  <p className="text-s2-gold">{d.count} hour(s)</p>
                  <p className="text-s2-muted text-[10px] mt-1">
                    {d.hours.join(", ")}
                  </p>
                </div>
              );
            }}
          />
          <Bar
            dataKey="count"
            name="Hours"
            fill="#fbbf24"
            fillOpacity={0.8}
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>

      <DataTable
        title="Distribution Table"
        columns={[
          { key: "range", label: "GHI Range (W/m²)" },
          { key: "count", label: "Hour Count" },
          { key: "hoursList", label: "Hours" },
        ]}
        rows={bins.map((b) => ({ ...b, hoursList: b.hours.join(", ") }))}
      />
    </SectionCard>
  );
}
