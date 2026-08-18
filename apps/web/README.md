# ViAna Web UI (`apps/web`)

Next.js 15 host application: container lifecycle on the host, job UI against FastAPI in Docker (`http://localhost:8000`).

Until endpoints are marked ✅ in `docs/PROJECT_STATUS.md`, keep `NEXT_PUBLIC_USE_MOCKS=true` (fixtures in `packages/contracts/fixtures/`).

## Setup

```bash
cd apps/web
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000

## Shared types

Import from `@viana/contracts` (path alias → `packages/contracts/typescript`).

See `docs/ui/DEVELOPMENT_GUIDE.md`.
