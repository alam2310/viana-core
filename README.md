# ViAna — Indian Traffic Video Analytics

Offline traffic video analytics: detect, classify, track, and count vehicles with CSV reports and optional annotated video.

## Overview

ViAna v2 is a monorepo with a GPU CV engine (`src/viana/`), FastAPI job orchestrator (`src/orchestrator/`), and a planned Next.js UI (`apps/web/`). Shared API contracts live in `packages/contracts/`.

**AI agents:** read [`AGENTS.md`](AGENTS.md) first, then [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

## Installation

```bash
# GPU container (recommended)
docker compose build
docker compose up -d
docker compose exec dev bash

# Inside container or local venv
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

## Usage

```bash
# Health check (orchestrator stub)
make api-dev
curl http://localhost:8000/health

# Engine CLI (stubs until Phase 3+)
python -m viana --help
python -m viana prescan --source /data/video.mp4 --project-id nh48
python -m viana run --config /data/job.json

# Run tests
make test
```

OpenAPI spec: [`openapi.yaml`](openapi.yaml). Human API reference: [`docs/api_contracts.md`](docs/api_contracts.md).

## Development

```bash
make install          # pip install -e ".[dev]"
make test             # pytest tests/viana/
make lint             # ruff + bandit + import boundaries
make typecheck        # mypy on active packages
make format           # ruff format

# Single-file verification (fast feedback for agents)
ruff check src/viana/cli.py
mypy src/viana/cli.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), and [`docs/design/README.md`](docs/design/README.md).

## Repository layout

```
ViAna/
├── src/viana/           # CV engine (active)
├── src/orchestrator/    # FastAPI job API (active)
├── apps/web/            # Next.js UI (Phase 7+)
├── packages/contracts/  # Shared schemas & types
├── configs/             # classes.yaml, engine_defaults.yaml
├── models/              # v1 + pretrained weights
├── docs/                # Plans, specs, governance, UI guides
├── legacy/              # Discardable — old code & historical docs
└── tests/viana/         # Engine tests
```

## Legacy / parity

Pre-v2 code lives under **`legacy/`**. Parity procedure: [`legacy/PARITY.md`](legacy/PARITY.md).

## Security

[`SECURITY.md`](SECURITY.md) · [`THREAT_MODEL.md`](THREAT_MODEL.md)
