// -------------------------------------------------------------------------
// PipelineDetails.jsx — Clean infographic explaining the ML pipeline.
//
// Covers ingestion, preprocessing, cyclical encoding, rolling windows,
// train/test split, model selection, and auto-retraining.
// -------------------------------------------------------------------------

function InfoCard({ title, children }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg bg-[#101014] border border-s2-border p-4">
      <h4 className="text-xs uppercase tracking-widest text-s2-blue font-semibold">
        {title}
      </h4>
      <div className="text-[13px] leading-relaxed text-s2-muted">{children}</div>
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <pre className="rounded bg-s2-bg border border-s2-border p-2 text-[12px] font-mono text-s2-cyan overflow-x-auto">
      {children}
    </pre>
  );
}

export default function PipelineDetails() {
  return (
    <section className="rounded-lg bg-s2-card border border-s2-border p-4">
      <h3 className="text-xs uppercase tracking-widest text-s2-muted mb-4">
        ML Pipeline Architecture
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {/* Step 1: Data Ingestion */}
        <InfoCard title="1. Data Ingestion">
          <p>Raw weather data is ingested from Open-Meteo API and stored in PostgreSQL via the
          automated ingest pipeline.</p>
          <CodeBlock>{`# Scheduled via cron / GitHub Actions
python -m ingest.main \\
  --cities islamabad,lahore,karachi,peshawar \\
  --start 2023-01-01 \\
  --end today`}</CodeBlock>
          <p className="mt-1 text-[11px] text-s2-muted">
            Source fields: GHI, temperature, humidity, wind speed, cloud cover, pressure (hourly resolution).
          </p>
        </InfoCard>

        {/* Step 2: Preprocessing */}
        <InfoCard title="2. Preprocessing & Cleaning">
          <p>Raw data is validated, missing values interpolated, and outliers flagged before
          feature engineering.</p>
          <CodeBlock>{`# Null handling
df = df.interpolate(method='time')

# Outlier cap (IQR method)
df['ghi'] = df['ghi'].clip(0, 1400)
df['temp'] = df['temp'].clip(-10, 55)`}</CodeBlock>
          <p className="mt-1 text-[11px] text-s2-muted">
            Records with &gt;30% missing fields in a day are dropped entirely.
          </p>
        </InfoCard>

        {/* Step 3: Cyclical encoding */}
        <InfoCard title="3. Cyclical Time Encoding">
          <p>Hour-of-day and day-of-year are encoded as sin/cos pairs to preserve
          the circular nature of time features.</p>
          <CodeBlock>{`hour_sin = sin(2\u03c0 \u00d7 hour / 24)
hour_cos = cos(2\u03c0 \u00d7 hour / 24)
doy_sin  = sin(2\u03c0 \u00d7 day_of_year / 365)
doy_cos  = cos(2\u03c0 \u00d7 day_of_year / 365)`}</CodeBlock>
          <p className="mt-1 text-[11px] text-s2-muted">
            This ensures hour 23 is close to hour 0 in feature space, unlike raw integer encoding.
          </p>
        </InfoCard>

        {/* Step 4: Rolling windows */}
        <InfoCard title="4. Rolling Window Features">
          <p>Lag features and rolling means capture short-term weather momentum.</p>
          <CodeBlock>{`GHI_lag_1      = GHI(t-1)
GHI_rolling_3h = mean(GHI[t-3 : t])
GHI_rolling_6h = mean(GHI[t-6 : t])
temp_rolling_3h = mean(Temp[t-3 : t])`}</CodeBlock>
          <p className="mt-1 text-[11px] text-s2-muted">
            Window sizes (3h, 6h) were selected via cross-validation to minimize GHI prediction RMSE.
          </p>
        </InfoCard>

        {/* Step 5: Train–Test split */}
        <InfoCard title="5. Temporal Train\u2013Test Split">
          <p>Data is split temporally (not randomly) to prevent future data leakage.</p>
          <div className="flex items-center gap-1 mt-1">
            <span className="flex-1 h-3 rounded-l bg-s2-blue" />
            <span className="flex-1 h-3 bg-s2-gold" />
            <span className="w-1/4 h-3 rounded-r bg-s2-red" />
          </div>
          <div className="flex justify-between text-[10px] font-mono text-s2-muted mt-0.5">
            <span>Train (70%)</span>
            <span>Val (15%)</span>
            <span>Test (15%)</span>
          </div>
          <p className="mt-1 text-[11px] text-s2-muted">
            Validation and test sets are consecutive future windows.
          </p>
        </InfoCard>

        {/* Step 6: Model selection */}
        <InfoCard title="6. Champion / Challenger Selection">
          <p>XGBoost was selected as champion over LSTM based on lower MAE and faster inference.</p>
          <table className="w-full mt-1 text-[11px] font-mono">
            <thead>
              <tr className="text-s2-muted text-left border-b border-s2-border">
                <th className="pb-1 pr-3">Model</th>
                <th className="pb-1 pr-3">MAE</th>
                <th className="pb-1 pr-3">RMSE</th>
                <th className="pb-1">Inference</th>
              </tr>
            </thead>
            <tbody className="text-s2-text">
              <tr className="border-b border-s2-border">
                <td className="py-1 pr-3 text-s2-blue">XGBoost</td>
                <td className="py-1 pr-3">42.3</td>
                <td className="py-1 pr-3">68.7</td>
                <td className="py-1">~2 ms</td>
              </tr>
              <tr>
                <td className="py-1 pr-3 text-s2-muted">LSTM</td>
                <td className="py-1 pr-3">55.1</td>
                <td className="py-1 pr-3">81.4</td>
                <td className="py-1">~18 ms</td>
              </tr>
            </tbody>
          </table>
        </InfoCard>
      </div>

      {/* Auto-retraining pipeline (full width) */}
      <div className="mt-3">
        <InfoCard title="7. Automated Retraining Pipeline">
          <p>The model is <strong className="text-s2-text">retrained daily</strong> via a scheduled pipeline that ingests
          new data and validates the new model against the current champion before promotion.</p>

          {/* Pipeline flow diagram */}
          <div className="flex flex-wrap items-center gap-2 mt-2 text-[11px] font-mono">
            <span className="px-2 py-1 rounded bg-s2-blue/20 text-s2-blue border border-s2-blue/30">Ingest New Data</span>
            <span className="text-s2-muted">\u2192</span>
            <span className="px-2 py-1 rounded bg-s2-gold/20 text-s2-gold border border-s2-gold/30">Preprocess + Features</span>
            <span className="text-s2-muted">\u2192</span>
            <span className="px-2 py-1 rounded bg-s2-cyan/20 text-s2-cyan border border-s2-cyan/30">Train Challenger</span>
            <span className="text-s2-muted">\u2192</span>
            <span className="px-2 py-1 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Validate vs Champion</span>
            <span className="text-s2-muted">\u2192</span>
            <span className="px-2 py-1 rounded bg-s2-red/20 text-s2-red border border-s2-red/30">Promote if Better</span>
          </div>

          <div className="mt-2 text-[11px] text-s2-muted space-y-1">
            <p><strong className="text-s2-text">Schedule:</strong> Daily at 02:00 UTC via GitHub Actions cron trigger.</p>
            <p><strong className="text-s2-text">Validation gate:</strong> Challenger must beat champion MAE by at least 2% on the latest 7-day window.</p>
            <p><strong className="text-s2-text">Rollback:</strong> If challenger fails validation, current champion stays active. Alert sent to monitoring.</p>
            <p><strong className="text-s2-text">Artifacts:</strong> Model weights, metrics JSON, and training metadata are versioned in the repository.</p>
          </div>
        </InfoCard>
      </div>
    </section>
  );
}
