# Deployment & Local Development

**First-time setup:** [`ops/ENVIRONMENT_SETUP.md`](ops/ENVIRONMENT_SETUP.md) (host GPU, Docker build, verification).

## Docker (CV engine + API)

```bash
docker compose build
docker compose up -d
docker compose exec dev bash

# Inside container (live mount or fresh image)
pip install -e ".[dev]"
pip install -q "numpy>=1.26.0,<2"
pip install -q "trackers==2.6.0" --no-deps
make api-dev            # FastAPI on :8000 (compose already runs uvicorn)
python -m viana --help
```

### Local repo hygiene

Gitignored artifact folders at the **repo root** (`debug_pretrain/`, `runs/`) are safe to delete:

```bash
./scripts/cleanup-local-artifacts.sh
```

Environment variables (`docker-compose.yml`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTHONPATH` | `/app/ViAna/src` | Import viana/orchestrator |
| `VIANA_OUTPUT_PARENT` | `/data/viana-outputs` | Artifact output root |
| `VIANA_DATA_ROOT` | `./data` (host) | Mount host data → `/data` |

## UI (host)

```bash
cd apps/web
cp .env.example .env.local
npm install && npm run dev
```

Container config for UI docker manager: `docker/orchestrator_config.yaml.example`

## Tests

```bash
pytest tests/viana/
pytest training/uvh/tests/
```

## UVH retraining (optional)

See [`../training/README.md`](../training/README.md).
