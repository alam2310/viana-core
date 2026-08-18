# UI Agent — apps/web

**Read first:** `/AGENTS.md` (repo root) → `docs/PROJECT_STATUS.md` → this file.

You are building the **Next.js 15** host application. The CV engine runs **inside Docker**; you talk to it via HTTP/WebSocket on port `8000`.

---

## 1. Your owned paths

```
apps/web/                 ← all UI source code
docs/ui/                  ← UI specifications (read-only unless fixing docs)
packages/contracts/       ← types & fixtures (coordinate via schema PRs)
```

**Do not edit:** `src/viana/`, `src/orchestrator/`, `legacy/`

---

## 2. Architecture reminder

| Layer | Where | Your code? |
|-------|-------|------------|
| Container lifecycle | Host `apps/web/src/app/api/container/` | ✅ Yes |
| Job API | Container `:8000` | ❌ Consume only |
| CV processing | `python -m viana` in container | ❌ Never |

---

## 3. Hard rules

1. Import types from `@viana/contracts` → `packages/contracts/typescript`
2. **Never** send `job_id` or `gpu_device` in `POST /jobs` body
3. Job queue state: sync from `GET /jobs`; localStorage = UI prefs only
4. Canvas coords = pixel space of `video_meta.width` × `video_meta.height`; clamp to frame
5. Check `docs/PROJECT_STATUS.md` API matrix — use mocks when endpoint is ❌
6. Container start/stop uses `docker/orchestrator_config.yaml` on host

---

## 4. Mock mode (required until API Phase 6)

Copy `apps/web/.env.example` → `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_USE_MOCKS=true
```

When `USE_MOCKS=true`, `lib/api-client.ts` should return fixture data from `packages/contracts/fixtures/`.

---

## 5. Planned features (see docs/ui/)

| Feature | Spec |
|---------|------|
| Container health + auto-start | `USER_FLOWS.md` §1 |
| Video queue (50+) | `docs/specs/ui_specifications.md` |
| Prescan + OCR review | `USER_FLOWS.md` §2 |
| Calibration canvas | `CALIBRATION_CANVAS.md` |
| Progress WebSocket | `API_INTEGRATION.md` |
| Paused job resume/fresh | `STATE_MACHINE.md` |
| Profile save/apply batch | `USER_FLOWS.md` §2 |

---

## 6. Tech stack (locked)

- Next.js 15 App Router
- Tailwind v4
- Shadcn/UI
- TypeScript strict

---

## 7. Phase 7 scaffold checklist

When scaffolding `apps/web/` (stub `package.json` + `tsconfig.json` already present):

- [x] `package.json` with scripts `dev`, `build`, `lint`
- [x] Path alias `@viana/contracts` → `../../packages/contracts/typescript`
- [x] `src/lib/api-client.ts` with mock/real switch
- [x] `src/lib/container-manager.ts` (host docker)
- [x] `src/app/api/container/status/route.ts`
- [x] `src/app/api/container/start/route.ts`
- [x] Placeholder dashboard `src/app/page.tsx`

## 7b. Phase 8 workflow checklist

- [x] Prescan modal + OCR review + frame offset
- [x] Calibration canvas (pixel space, clamp, drag endpoints)
- [x] Pending path queue (localStorage) + GET /jobs sync
- [x] Paused job resume / start-fresh UX
- [x] Telemetry_detail toggle (prefs) + mock WS panel

Keep `NEXT_PUBLIC_USE_MOCKS=true` until API endpoints are ✅.

---

## 8. Verification

```bash
cd apps/web
npm install
npm run dev
# → http://localhost:3000
```

Container must be running for real API mode: `docker compose up -d`

---

## 9. Questions?

If a spec is missing, add to `docs/ui/` and `docs/api_contracts.md` **before** implementing. Do not guess API shapes.
