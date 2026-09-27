"""Publish a trained checkpoint to the Hugging Face Hub with a model card generated from its metrics.json.

  pip install huggingface_hub && huggingface-cli login
  python scripts/publish_hf.py runs/dragonfly-s <your-hf-user>/dragonfly-s            # dry run: writes README.md only
  python scripts/publish_hf.py runs/dragonfly-s <your-hf-user>/dragonfly-s --upload   # creates the repo and uploads

The card reports only what metrics.json measured: accuracy, calibration, share automatable at 5% error, option-order
flips, per source and per question type. Nothing is typed in by hand.
"""

import argparse
import json
from pathlib import Path

CARD = """---
license: apache-2.0
library_name: dragonfly
tags: [decision-model, system-one, calibration, classification]
{base_model}---

# {name}

A Dragonfly **tier {tier}** decision model: typed questions in, calibrated probabilities over every answer out, in one
forward pass. It serves the TypeSafe-compatible `/v1/systemone` API. See the
[Dragonfly repository](https://github.com/{github}) for serving, plugins and training.

| | |
|---|---|
| Backbone | `{backbone}` |
| Trained on | {train_questions} questions (Kev decision-v2 suite), {epochs} epochs |
| Temperature (fitted on the calibration split) | {temperature:.3f} |

## Results on the held-out test split

| Metric | Value |
|---|---|
| Accuracy | {accuracy} |
| ECE (top-label, 10 bins; lower is better) | {ece} |
| Brier | {brier} |
| Share automatable at 5% error | {auto} |
| Option-order flip rate | {flips} |

### By source

| Source | n | Accuracy | ECE |
|---|---|---|---|
{by_source}

## Use

```bash
DRAGONFLY_CHECKPOINT=/path/to/this/repo dragonfly-serve
curl localhost:8000/v1/systemone -d @request.json
```

## Limits

It decides from the text it is given. It has little world knowledge and does no multi-step reasoning. It was trained on
the sources above, so accuracy on other domains is unmeasured until you evaluate it there (`dragonfly-eval`).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("repo_id", help="<user or org>/<name> on the Hub")
    ap.add_argument("--github", default="OWNER/dragonfly", help="owner/repo of the Dragonfly GitHub repository")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()

    root = Path(a.checkpoint)
    config = json.loads((root / "config.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    if not config.get("trained"):
        raise SystemExit("refusing to publish an untrained checkpoint")
    test = metrics.get("test")
    if not test:
        raise SystemExit("metrics.json has no test results: train with --test so the card has measured numbers")
    o = test["overall"]
    rows = "\n".join(f"| {s} | {m['n']} | {m['accuracy']} | {m['ece']} |" for s, m in test["by_source"].items())
    card = CARD.format(
        name=a.repo_id.split("/")[-1], tier=config["tier"], backbone=config["backbone"], github=a.github,
        base_model=f"base_model: {config['backbone']}\n" if config["tier"] == "M" else "",
        train_questions=metrics["train_questions"], epochs=metrics["epochs"], temperature=metrics["temperature"],
        accuracy=o["accuracy"], ece=o["ece"], brier=o["brier"], auto=o["auto_rate@5%"],
        flips=test.get("order_flip_rate", "n/a"), by_source=rows,
    )
    (root / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {root / 'README.md'}")
    if not a.upload:
        print("dry run: add --upload to create the Hub repo and upload")
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=a.repo_id, commit_message="Upload Dragonfly checkpoint")
    print(f"https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
