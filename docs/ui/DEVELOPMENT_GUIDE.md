# UI Development Guide

## Monorepo layout

| Path | Role |
|------|------|
| `apps/web/` | Next.js 15 UI (Phase 7) |
| `packages/contracts/` | Shared JSON schemas + TypeScript types |
| `src/orchestrator/` | FastAPI inside container |
| `src/viana/` | CV engine CLI |

## Local development

```bash
# 1. Container (GPU)
docker compose up -d
docker compose exec dev bash

# 2. Install Python package (inside container)
pip install -e ".[dev]"

# 3. API (inside container)
make api-dev
# → http://localhost:8000/health

# 4. UI (on host, when scaffolded)
cd apps/web && npm run dev
```

## Environment variables

| Variable | Where | Purpose |
|----------|-------|---------|
| `VIANA_DATA_ROOT` | host `docker-compose` | Mount host data to `/data` |
| `VIANA_OUTPUT_PARENT` | container | Default output parent dir |
| `NEXT_PUBLIC_API_URL` | `apps/web` | FastAPI base URL |

## Contracts

- Types: `packages/contracts/typescript/index.ts`
- Schemas: `packages/contracts/schemas/*.json`
- Mock data: `packages/contracts/fixtures/*.json`

Use fixtures to build UI before GPU engine is complete.

## Related docs

- `USER_FLOWS.md` — screen flows
- `CALIBRATION_CANVAS.md` — line drawing spec
- `API_INTEGRATION.md` — HTTP + WebSocket
- `STATE_MACHINE.md` — job states
- `COMPONENT_MAP.md` — feature → component mapping
- `OUTPUT_PATHS.md` — artifact locations
