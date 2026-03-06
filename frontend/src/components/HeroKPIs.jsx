// -------------------------------------------------------------------------
// HeroKPIs.jsx — Top row of metric cards.
//
// All values are derived from chartData (the 24-hour array) so they
// update instantly when the user drags a slider.
// -------------------------------------------------------------------------
import { useMemo } from "react";

/** Individual KPI card. */
function KpiCard({ label, value, unit, sub }) {
  return (
    <div className="flex flex-col gap-1.5 flex-1 min-w-[155px] rounded-lg bg-[#101014] border border-s2-border p-3">
      <span className="text-[11px] uppercase tracking-wider text-s2-muted">
        {label}
      </span>
      <span className="text-xl font-bold font-mono text-s2-text">
        {value}
        {unit && (
          <span className="text-xs font-normal text-s2-muted ml-1.5">
            {unit}
          </span>
        )}
      </span>
      {sub && <span className="text-[10px] text-s2-muted">{sub}</span>}
    </div>
  );
}

export default function HeroKPIs({ chartData }) {
  const metrics = useMemo(() => {
    if (!chartData || chartData.length === 0) {
      return {
        selfSuffPct: "0.0",
        peakGhi: "0",
        psh: "0.00",
        totalDeficit: "0.00",
        solarProduced: "0.00",
        zeroGhiHours: 24,
      };
    }

    // Solar Self-Sufficiency %: hours solar covers load / hours cooling needed
    const coolingHours = chartData.filter((r) => r.loadKw > 0);
    const solarCoveredHours = coolingHours.filter((r) => r.solarKw >= r.loadKw).length;
    const selfSuffPct = coolingHours.length > 0
      ? ((solarCoveredHours / coolingHours.length) * 100).toFixed(1)
      : "100.0";

    // Peak GHI
    const peakGhi = Math.max(...chartData.map((r) => r.ghi)).toFixed(0);

    // PSH
    const psh = (chartData.reduce((acc, r) => acc + r.ghi, 0) / 1000).toFixed(2);

    // Total Energy Deficit
    const totalDeficit = chartData
      .reduce((acc, r) => acc + r.gridDeficit, 0)
      .toFixed(2);

    // Total Solar Energy Produced (kWh) — sum of solarKw (1h buckets)
    const solarProduced = chartData
      .reduce((acc, r) => acc + r.solarKw, 0)
      .toFixed(2);

    // Hours with zero GHI (nighttime)
    const zeroGhiHours = chartData.filter((r) => r.ghi < 1).length;

    return { selfSuffPct, peakGhi, psh, totalDeficit, solarProduced, zeroGhiHours };
  }, [chartData]);

  return (
    <section className="flex flex-col gap-2.5 rounded-lg bg-s2-card border border-s2-border p-3">
      <div className="flex flex-wrap gap-2.5">
        <KpiCard
          label="Solar Self-Sufficiency"
          value={`${metrics.selfSuffPct}%`}
          sub="Hours solar covers cooling load"
        />
        <KpiCard label="Peak GHI" value={metrics.peakGhi} unit="W/m²" />
        <KpiCard label="Peak Sun Hours" value={metrics.psh} unit="h" />
        <KpiCard
          label="Solar Energy Produced"
          value={metrics.solarProduced}
          unit="kWh"
        />
        <KpiCard
          label="Grid Energy Needed"
          value={metrics.totalDeficit}
          unit="kWh"
        />
      </div>
      {/* GHI zero-hour callout */}
      <div className="flex items-center gap-2 px-1">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-s2-muted" />
        <span className="text-[11px] text-s2-muted">
          {metrics.zeroGhiHours} of 24 hours have zero GHI (nighttime) — no solar generation possible during those hours
        </span>
      </div>
    </section>
  );
}
