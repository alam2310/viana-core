---
name: viana-api-route
description: Add a new FastAPI orchestrator route. Use when implementing HTTP endpoints in src/orchestrator/.
---

# New API route

Follow the pattern in `src/orchestrator/routes/health.py`.

1. Add router module under `src/orchestrator/routes/`
2. Register router in `src/orchestrator/app.py`
3. Update `packages/contracts/schemas/` and `openapi.yaml` before handler logic
4. Mark endpoint in `docs/PROJECT_STATUS.md` when implemented
