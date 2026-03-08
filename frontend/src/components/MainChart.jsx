// -------------------------------------------------------------------------
// MainChart.jsx — Four focused mini-charts for the Live Controller tab.
//
// 1. Solar Generation (kW) — gold area
// 2. Electrical Load (kW) — red dashed line
// 3. Grid Deficit (kW) — grey bar
// 4. Ambient Temperature (°C) — cyan line
//
// Each chart is independently readable with its own Y-axis and tooltip.
// -------------------------------------------------------------------------
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import DataTable from "./DataTable";

/** Dark-themed tooltip shared by all mini-charts. */
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg bg-s2-card border border-s2-border px-3 py-2 text-xs shadow-lg">
      <p className="font-mono text-s2-text mb-1">{label}</p>
      {payload.map((entry) => (
        <p key={entry.name} style={{ color: entry.color }} className="font-mono">
          {entry.name}: {Number(entry.value).toFixed(2)}
        </p>
      ))}
    </div>
  );
}

/** Shared X-axis props. */
const XAXIS_PROPS = {
  dataKey: "hourLabel",
  stroke: "#a1a1aa",
  tick: { fill: "#a1a1aa", fontSize: 10, fontFamily: "monospace" },
  interval: 1,
};

/** Shared Y-axis tick style. */
const yTick = { fill: "#a1a1aa", fontSize: 10, fontFamily: "monospace" };

/** Mini-chart wrapper with title and accent color. */
function MiniCard({ title, color, children }) {
  return (
    <div className="rounded-lg bg-s2-card border border-s2-border p-3 flex flex-col">
      <div className="flex items-center gap-2 mb-2">
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <h4 className="text-[11px] uppercase tracking-widest text-s2-muted">
          {title}
        </h4>
      </div>
      <div style={{ minHeight: 180 }} className="flex-1">
        {children}
      </div>
    </div>
  );
}

export default function MainChart({ chartData }) {
  const tableColumns = [
    { key: "hourLabel", label: "Hour" },
    { key: "ghi", label: "GHI (W/m\u00b2)" },
    { key: "solarKw", label: "Solar (kW)", decimals: 3 },
    { key: "loadKw", label: "Load (kW)", decimals: 3 },
    { key: "gridDeficit", label: "Grid (kW)", decimals: 3 },
    { key: "ambientTemp", label: "Temp (\u00b0C)" },
  ];

  return (
    <section className="flex flex-col gap-3">
      {/* 2×2 grid of mini-charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* 1. Solar Generation */}
        <MiniCard title="Solar Generation (kW)" color="#fbbf24">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <defs>
                <linearGradient id="solarGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis {...XAXIS_PROPS} />
              <YAxis
                tick={yTick}
                label={{ value: "kW", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 10 }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Area
                type="monotone"
                dataKey="solarKw"
                stroke="#fbbf24"
                fill="url(#solarGrad)"
                strokeWidth={2}
                name="Solar Gen (kW)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </MiniCard>

        {/* 2. Electrical Load */}
        <MiniCard title="Electrical Load (kW)" color="#ef4444">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis {...XAXIS_PROPS} />
              <YAxis
                tick={yTick}
                label={{ value: "kW", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 10 }}
              />
              <Tooltip content={<ChartTooltip />} />
              <ReferenceLine y={0} stroke="#3f3f46" strokeWidth={1} />
              <Line
                type="stepAfter"
                dataKey="loadKw"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Electrical Load (kW)"
              />
            </LineChart>
          </ResponsiveContainer>
        </MiniCard>

        {/* 3. Grid Deficit */}
        <MiniCard title="Grid Deficit (kW)" color="#71717a">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis {...XAXIS_PROPS} />
              <YAxis
                tick={yTick}
                label={{ value: "kW", angle: -90, position: "insideLeft", fill: "#a1a1aa", fontSize: 10 }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Bar
                dataKey="gridDeficit"
                fill="#71717a"
                fillOpacity={0.8}
                name="Grid Deficit (kW)"
                radius={[3, 3, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </MiniCard>

        {/* 4. Ambient Temperature */}
        <MiniCard title="Ambient Temperature (°C)" color="#22d3ee">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 12, left: 0, bottom: 4 }}>
              <defs>
                <linearGradient id="tempGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
              <XAxis {...XAXIS_PROPS} />
              <YAxis
                tick={{ fill: "#22d3ee", fontSize: 10, fontFamily: "monospace" }}
                label={{ value: "°C", angle: -90, position: "insideLeft", fill: "#22d3ee", fontSize: 10 }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Area
                type="monotone"
                dataKey="ambientTemp"
                stroke="#22d3ee"
                fill="url(#tempGrad)"
                strokeWidth={2}
                name="Ambient Temp (°C)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </MiniCard>
      </div>

      {/* GHI night-time callout */}
      <div className="flex items-center gap-2 px-1">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-s2-gold" />
        <span className="text-[10px] text-s2-muted">
          Solar generation is zero when GHI = 0 (start and end of day). No solar power is possible during those hours.
        </span>
      </div>

      {/* Data table */}
      <DataTable
        title="Hourly Breakdown"
        columns={tableColumns}
        rows={chartData}
      />
    </section>
  );
}
