# Deployment & Local Development

## Docker (CV engine + API)

```bash
# From repo root
docker compose build    # required after Dockerfile changes
docker compose up -d
docker compose exec dev bash

# Inside container (if image predates pyproject install, or for live mount updates)
pip install -e ".[dev]"
make api-dev            # FastAPI on :8000
python -m viana --help
```

### Local repo hygiene

Gitignored artifact folders at the **repo root** (`debug_pretrain/`, `runs/`) are duplicates of `legacy/artifacts/`. Safe to delete:

```bash
rm -rf debug_pretrain/ runs/    # use sudo if Docker created files as nobody
# or:
./scripts/cleanup-local-artifacts.sh
```

See `legacy/artifacts/README.md` and `docs/PHASE_0_SIGNOFF.md`.

Environment variables (`docker-compose.yml`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYTHONPATH` | `/app/ViAna/src` | Import viana/orchestrator |
| `VIANA_OUTPUT_PARENT` | `/data/viana-outputs` | Artifact output root |
| `VIANA_DATA_ROOT` | `./data` (host) | Mount host data → `/data` |

## UI (host, Phase 7+)

```bash
cd apps/web
cp .env.example .env.local
npm install && npm run dev
```

Container config for UI docker manager: `docker/orchestrator_config.yaml.example`

## Tests

```bash
pytest tests/viana/       # active engine
pytest legacy/tests/      # legacy taxonomy only
```

## Legacy training / Docker setup (historical)

See [`../legacy/docs/ITVA_Environment_Setup_Guide.md`](../legacy/docs/ITVA_Environment_Setup_Guide.md).
