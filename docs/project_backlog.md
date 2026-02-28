# MagicBox Project Backlog (Execution-Ready)

## P0 — Model Ops & Daily Learning Loop

### 1) Daily Data Ingestion Reliability
- **Goal**: New daily data lands in canonical tables with schema validation and quality checks.
- **Deliverables**:
  - Ingestion job scheduled daily.
  - Schema validator with hard-fail on breaking changes.
  - Data quality report (missingness, duplicates, outliers) persisted per run.
- **Acceptance Criteria**:
  - 7 consecutive scheduled runs complete successfully.
  - Any schema break blocks downstream training/scoring.
  - Quality report is queryable for each run date.

### 2) Champion/Challenger XGBoost Pipeline
- **Goal**: Keep production model stable while continuously evaluating daily challenger retrains.
- **Deliverables**:
  - Daily challenger training on rolling window (default 120 days).
  - Frozen champion model registry entry with artifact hash.
  - Automated comparison report against champion on holdout + recent slice.
- **Acceptance Criteria**:
  - Challenger trained daily with versioned artifact.
  - Promotion only when challenger beats champion on defined gate metrics.
  - One-command rollback to previous champion.

### 3) Retrain Trigger Policy (Scheduled + Event-Driven)
- **Goal**: Retrain when needed, not just on a timer.
- **Deliverables**:
  - Drift monitors (PSI + feature null-rate drift + target shift proxy).
  - Performance monitors (RMSE/MAE/AUC as applicable).
  - Trigger rules and escalation policy.
- **Acceptance Criteria**:
  - Trigger thresholds configured and tested in staging.
  - Alert generated within 5 minutes of threshold breach.
  - Retrain/rollback recommendation logged per alert.

### 4) Model Registry + Reproducibility Contract
- **Goal**: Every prediction can be traced to exact model/data/features.
- **Deliverables**:
  - Registry metadata: model version, feature set version, training cutoff, metrics.
  - Immutable artifact storage.
  - Prediction logs include model version + feature hash.
- **Acceptance Criteria**:
  - 100% of production predictions are traceable.
  - Re-running training with same snapshot reproduces metrics within tolerance.

## P0 — Decision Dashboard (Frontend + API Integration)

### 5) Dashboard V1 (Control + KPI + Chart + Diagnostics)
- **Goal**: Single-pane decision UI for site/date scenario analysis and AI forecast mode.
- **Deliverables**:
  - Global header (city/date/heartbeat).
  - Parameter sidebar (COP/threshold/PV config/operating hours).
  - KPI row (Solar Coverage, Savings, Peak GHI, PSH, CO2 Offset).
  - Main chart (solar area, load dashed line, grid deficit bars, temp secondary axis).
  - AI diagnostic footer (active model, cutoff date, RMSE/MAE, inertia summary).
- **Acceptance Criteria**:
  - City/date changes call `/v1/simulate/day`.
  - Control changes update chart instantly in local state.
  - Visuals follow strict style constraints from product guidelines.

### 6) API Contract Hardening for Dashboard
- **Goal**: Ensure frontend can safely consume simulation and diagnostics data.
- **Deliverables**:
  - Stable response contract for daily simulation.
  - Metrics endpoint for production RMSE/MAE + metadata.
  - Graceful error response model.
- **Acceptance Criteria**:
  - Dashboard renders complete state from API without manual data edits.
  - Any API failure degrades gracefully with visible status and no crash.

## P1 — Prediction Quality & Explainability

### 7) Forecast Feature Enrichment
- **Goal**: Improve next-day accuracy for ambient/load/solar behavior.
- **Deliverables**:
  - Time features, weather lag features, site seasonal flags.
  - Feature drift scoring per site.
- **Acceptance Criteria**:
  - Offline benchmark beats current baseline by agreed margin.

### 8) Explainability Layer
- **Goal**: Explain why decisions changed day-to-day.
- **Deliverables**:
  - SHAP summary per daily run.
  - Top feature movement panel in dashboard diagnostics.
- **Acceptance Criteria**:
  - Top 3 drivers shown for each prediction run and stored for audit.

## P1 — Platformization

### 9) Orchestration & Observability
- **Goal**: Production-grade visibility for jobs and model lifecycle.
- **Deliverables**:
  - Scheduled workflows for ingest/train/eval/promote.
  - Centralized logs and alerts.
- **Acceptance Criteria**:
  - End-to-end run status visible for every daily cycle.

### 10) CI/CD Guardrails
- **Goal**: Prevent broken deploys and lockfile/runtime mismatches.
- **Deliverables**:
  - Python version pin checks.
  - Dependency lock validation in CI.
  - Smoke test for `/v1/simulate/day` and dashboard route.
- **Acceptance Criteria**:
  - CI blocks merge on lock mismatch or API contract break.

## Execution Order (Recommended)
1. P0-1 Ingestion reliability
2. P0-2 Champion/challenger training
3. P0-6 API hardening
4. P0-5 Dashboard V1
5. P0-3 Trigger policy
6. P0-4 Registry traceability
7. P1 items
