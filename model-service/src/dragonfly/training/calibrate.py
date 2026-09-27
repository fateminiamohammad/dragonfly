"""Fit a checkpoint's temperature on held-out data, save it, and re-evaluate. No retraining.

  dragonfly-calibrate --checkpoint runs/dragonfly-s --calibration data/calibration.jsonl --test data/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..calibration import fit_temperature
from ..data import load_records
from ..engine import Engine
from .evaluate import collect_logits, evaluate, question_rows
from .train import fitting

log = logging.getLogger("dragonfly.calibrate")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test")
    ap.add_argument("--device")
    a = ap.parse_args()

    engine = Engine.load(a.checkpoint, a.device)
    rows, states = fitting(engine, *question_rows(load_records(a.calibration)))
    engine.config.temperature = fit_temperature(collect_logits(engine, rows, states), [q["label"] for q in rows])
    log.info("fitted temperature %.3f on %d questions", engine.config.temperature, len(rows))
    root = Path(a.checkpoint)
    (root / "config.json").write_text(json.dumps(asdict(engine.config), indent=2))

    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["temperature"] = engine.config.temperature
    if a.test:
        metrics["test"] = evaluate(engine, load_records(a.test))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
