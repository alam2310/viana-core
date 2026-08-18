# ViAna — Indian Traffic Video Analytics

Offline traffic video analytics: detect, classify, track, and count vehicles with CSV reports and optional annotated video.

## AI agents

Read **[`AGENTS.md`](AGENTS.md)** first.

| Doc | Purpose |
|-----|---------|
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | What exists today |
| [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) | Implementation plan |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker & dev setup |

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

## Quick start

```bash
docker compose up -d
docker compose exec dev bash
pip install -e ".[dev]"
make api-dev          # http://localhost:8000/health
python -m viana --help
```

## Legacy / parity

Pre-v2 code and historical docs live under **`legacy/`** (including `legacy/blueprint.md` and `legacy/inference/inference_engine.py`). See [`legacy/PARITY.md`](legacy/PARITY.md).
