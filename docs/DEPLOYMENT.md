# Deployment & Local Development

**First-time setup:** [`ops/ENVIRONMENT_SETUP.md`](ops/ENVIRONMENT_SETUP.md) (host GPU, Docker build, verification).

## Docker (CV engine + API)

```bash
docker compose build
docker compose up -d
docker compose exec dev bash

# Inside container (live mount). Image already has numpy<2 + trackers --no-deps.
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
| `VIANA_HOST_REPO_ROOT` | compose project dir | Host repo path for intake rewrite |
| `VIANA_HOST_DATA_ROOT` | `{repo}/data` | Host data path for intake rewrite |
| `VIANA_INTAKE_ROOTS` | `/data:/app/ViAna` | Container prefixes `POST /jobs/intake` may read |
| `VIANA_PATH_MAPS` | (empty) | Extra `host->container` pairs, `;`-separated |
| `VIANA_EXTRA_INTAKE_ROOT` | (unset) | Appended to `VIANA_INTAKE_ROOTS` for an extra volume |

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
