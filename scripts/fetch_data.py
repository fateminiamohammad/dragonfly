"""Download Kev's frozen decision-v2 suite (Apache-2.0, github.com/jaredpalmer/kev) into ./data/decision-v2, pinned
to one commit so results are comparable with Kev's published numbers. The underlying datasets (Banking77, BoolQ,
AG News, MNLI, SST-5, Yelp, TREC, DBpedia, Amazon, IMDB) keep their own licenses; see the manifest.

  python scripts/fetch_data.py
"""

import sys
import urllib.request
from pathlib import Path

KEV_COMMIT = "5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109"
BASE = f"https://raw.githubusercontent.com/jaredpalmer/kev/{KEV_COMMIT}/evals/decision-v2"
FILES = ["manifest.json", "train.jsonl", "calibration.jsonl", "development.jsonl", "test.jsonl"]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "decision-v2"
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = out / name
        if dest.exists():
            print(f"exists  {dest}")
            continue
        print(f"fetch   {name}")
        try:
            urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        except Exception as e:
            dest.unlink(missing_ok=True)
            sys.exit(f"failed to download {name}: {e}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
