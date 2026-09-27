# Contributing

Thanks for helping build Dragonfly.

## Setup

```bash
cd model-service
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or cu126 for an NVIDIA GPU
pip install -e ".[train,dev]"
pytest && ruff check src tests
```

The tests use a tiny random model and need no downloads and no GPU.

## Rules

- **Numbers need evidence.** A change that claims better accuracy, calibration or speed includes before/after output
  from `dragonfly-eval` or the benchmark, and names the hardware.
- **Keep the API compatible.** `/v1/systemone` must stay compatible with TypeSafe and Kev clients.
- **Keep the plugin API stable.** Changes to `dragonfly.plugins.api` follow semver. Breaking changes need a major
  version bump and a note in the changelog.
- **Keep the hot path fast.** Nothing blocking goes into the request path: no network calls, no disk I/O.
- **Never commit secrets or data.** No `.env` files, API keys, datasets or checkpoints. `.gitignore` covers the usual
  places.

## Pull requests

- Branch from `main`.
- Keep each PR focused on one change.
- Fill in the PR template.
- CI must pass: ruff, pytest, and the image build.
