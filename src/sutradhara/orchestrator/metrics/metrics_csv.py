"""Utilities for exporting metrics to CSV format."""

import csv
from pathlib import Path
from typing import Any, Dict, List

from sutradhara.orchestrator.common.request_classes import RequestReplayOutcome
from sutradhara.orchestrator.logger.logger import setup_logging

logger = setup_logging()


def _write_csv_rows(filepath: Path, rows: List[dict]) -> None:
    """Helper to write CSV rows to a file."""
    if not rows:
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_csv(experiment_dir: Path, outcomes: List[RequestReplayOutcome]) -> None:
    """Save metrics as CSV files in experiment directory structure.

    Creates:
        {experiment_dir}/
            ├── request_metrics.csv
            ├── iteration_metrics.csv
            ├── batch_metrics.csv
            └── overlap_metrics.csv
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)

    request_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    for outcome in outcomes:
        payload = outcome.metrics.to_csv_payload()
        if payload.get("request"):
            request_rows.append(payload["request"])
        iteration_rows.extend(payload.get("iterations", []))
        batch_rows.extend(payload.get("batches", []))
        overlap_rows.extend(payload.get("overlaps", []))

    _write_csv_rows(experiment_dir / "request_metrics.csv", request_rows)
    _write_csv_rows(experiment_dir / "iteration_metrics.csv", iteration_rows)
    _write_csv_rows(experiment_dir / "batch_metrics.csv", batch_rows)
    _write_csv_rows(experiment_dir / "overlap_metrics.csv", overlap_rows)

    logger.info("Saved %d request metrics to %s", len(request_rows), experiment_dir)
