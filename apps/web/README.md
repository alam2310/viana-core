# ViAna Web UI (`apps/web`)

Next.js 15 application — **not scaffolded yet** (Phase 7).

## Purpose

- Container lifecycle on the **host** (docker ps/start via Next.js API routes)
- Job queue UI talking to **container FastAPI** at `http://localhost:8000`
- Pre-scan review, calibration canvas, telemetry WebSocket

## Setup (when implemented)

```bash
cd apps/web
npm install
cp .env.example .env.local
npm run dev
```

## Shared types

Import from `@viana/contracts` (path alias → `packages/contracts/typescript`).

See `docs/ui/DEVELOPMENT_GUIDE.md` for full workflow.
