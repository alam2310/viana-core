# UI Documentation Index

**UI agent entry:** `apps/web/AGENTS.md`  
**Status / API readiness:** `docs/PROJECT_STATUS.md` § API matrix  
**Types:** `packages/contracts/typescript/index.ts`  
**Mocks:** `packages/contracts/fixtures/`

---

## Read order for UI development

1. `apps/web/AGENTS.md`
2. `docs/specs/ui_specifications.md` (product requirements)
3. `docs/ui/USER_FLOWS.md`
4. `docs/ui/CALIBRATION_CANVAS.md`
5. `docs/ui/OVERLAY_COLORS.md`
6. `docs/ui/API_INTEGRATION.md`
7. `docs/ui/STATE_MACHINE.md`
8. `docs/api_contracts.md`

---

## Doc index

| File | Contents |
|------|----------|
| `DEVELOPMENT_GUIDE.md` | Local setup, env vars, monorepo map |
| `USER_FLOWS.md` | Step-by-step screens |
| `CALIBRATION_CANVAS.md` | Line drawing, coords, clamping |
| `OVERLAY_COLORS.md` | Processed-video line + per-class box colors |
| `API_INTEGRATION.md` | HTTP + WebSocket, errors |
| `STATE_MACHINE.md` | Job states, resume/fresh |
| `COMPONENT_MAP.md` | Planned component layout |
| `OUTPUT_PATHS.md` | Where result files live |

---

## Mock vs live

Phase 6 endpoints are ✅. Default E2E env is `NEXT_PUBLIC_USE_MOCKS=false`. Fixtures remain for local UI-only work.

```typescript
import prescanFixture from '../../../packages/contracts/fixtures/prescan_response.json';
```

Set `NEXT_PUBLIC_USE_MOCKS=true` (see `apps/web/.env.example`).

---

## UI source location

```
apps/web/
├── AGENTS.md           ← start here
├── package.json
├── src/
│   ├── app/            ← Next.js App Router + /api/container/*
│   ├── components/
│   ├── features/       ← container, dashboard, queue, prescan, calibration, telemetry
│   └── lib/            ← api-client, container-manager
└── .env.example
```

Phase 8 workflows talk to the live API when `NEXT_PUBLIC_USE_MOCKS=false`.
