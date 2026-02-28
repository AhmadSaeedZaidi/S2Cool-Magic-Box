"""Promote latest trained model to production if it improves RMSE."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LATEST_METRICS_PATH = Path("artifacts/metrics_latest.json")
PRODUCTION_DIR = Path("artifacts/production")
PRODUCTION_METRICS_PATH = PRODUCTION_DIR / "production_metrics.json"
PRODUCTION_MODEL_PATH = PRODUCTION_DIR / "ghi_model"


@dataclass
class CandidateModel:
    """Candidate model details for promotion decisions."""

    name: str
    rmse: float
    mae: float
    source_path: Path


def configure_logging() -> None:
    """Configure console logging for promotion workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON payload from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def best_candidate_from_latest(latest: dict[str, Any]) -> CandidateModel:
    """Select best model (lowest RMSE) from latest training run metrics."""
    candidates = []
    for model_key in ("xgboost", "lstm"):
        metrics = latest.get(model_key)
        if not isinstance(metrics, dict):
            continue
        source_path = Path(str(metrics.get("model_path", "")))
        if not source_path.exists():
            continue
        candidates.append(
            CandidateModel(
                name=str(metrics.get("model_name", model_key)),
                rmse=float(metrics["rmse"]),
                mae=float(metrics["mae"]),
                source_path=source_path,
            )
        )

    if not candidates:
        raise RuntimeError("No valid candidate models found in latest metrics.")

    return min(candidates, key=lambda candidate: candidate.rmse)


def read_current_production_rmse() -> float | None:
    """Read production RMSE if a production metrics file exists."""
    if not PRODUCTION_METRICS_PATH.exists():
        return None
    production_payload = load_json(PRODUCTION_METRICS_PATH)
    model_info = production_payload.get("model", {})
    rmse = model_info.get("rmse")
    return float(rmse) if rmse is not None else None


def promote(candidate: CandidateModel, latest_payload: dict[str, Any]) -> None:
    """Promote candidate model to production and persist metadata."""
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

    suffix = candidate.source_path.suffix or ".bin"
    destination_model = PRODUCTION_MODEL_PATH.with_suffix(suffix)
    shutil.copy2(candidate.source_path, destination_model)

    production_payload = {
        "model": {
            "name": candidate.name,
            "rmse": candidate.rmse,
            "mae": candidate.mae,
            "path": str(destination_model),
        },
        "source_training_run": latest_payload.get("trained_at_utc"),
    }
    PRODUCTION_METRICS_PATH.write_text(json.dumps(production_payload, indent=2), encoding="utf-8")

    logger.info("Promoted model: %s", candidate.name)
    logger.info("Production model path: %s", destination_model)
    logger.info("Production metrics path: %s", PRODUCTION_METRICS_PATH)


def main() -> None:
    """Promote latest best model only when RMSE improves over production."""
    configure_logging()

    latest_payload = load_json(LATEST_METRICS_PATH)
    candidate = best_candidate_from_latest(latest_payload)

    current_prod_rmse = read_current_production_rmse()
    logger.info("Latest candidate: %s RMSE=%.4f MAE=%.4f", candidate.name, candidate.rmse, candidate.mae)

    if current_prod_rmse is None:
        logger.info("No production baseline found; promoting latest candidate.")
        promote(candidate, latest_payload)
        return

    if candidate.rmse < current_prod_rmse:
        logger.info("Candidate RMSE improved (%.4f < %.4f); promoting.", candidate.rmse, current_prod_rmse)
        promote(candidate, latest_payload)
    else:
        logger.info(
            "Promotion skipped; candidate RMSE did not improve (%.4f >= %.4f).",
            candidate.rmse,
            current_prod_rmse,
        )


if __name__ == "__main__":
    main()
