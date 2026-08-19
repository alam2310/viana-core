# Step 4 — UI implementation

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | New — **UI v2** |
| **Blocks** | Step 5 |
| **Blocked by** | Steps 1–3 ✅ |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 4. Update tracker after **each** sub-step.

---

## Objective

Implement Step 1 `REDESIGN.md` against Step 2–3 APIs: intake browser, queue table, prescan review, live monitor, artifacts.

---

## Prerequisites

- [ ] `docs/ui/REDESIGN.md` approved
- [ ] Steps 2–3 complete
- [ ] `NEXT_PUBLIC_USE_MOCKS=false`, API on `:8000`

---

## Build order

| Sub-step | Work | Primary paths |
|----------|------|---------------|
| **4.1** | Filesystem intake + queue table (prescan statuses) | `apps/web/src/features/`, `api/fs/` |
| **4.2** | Prescan review modal — propose → confirm/edit + summary | `features/prescan/` |
| **4.3** | Live monitor sidebar + structured telemetry | `features/telemetry/`, queue |
| **4.4** | Completed artifacts + empty 15-min copy | dashboard |
| **4.5** | Polish + `COMPONENT_MAP.md` sync | `apps/web/`, `docs/ui/` |

**UI-only gaps from discovery:** G18 host browser, G22 telemetry formatters.

---

## Contract changes during Step 4

Stop → file gap in Step 2 → schema sync → resume. See `CONTRACT_SYNC.md`.

---

## Exit criteria

- [x] Sub-steps 4.1–4.5 ✅ in `TRACKER.md`
- [x] Full intake → review → READY → process flow works manually
- [x] `AGENT_PROGRESS.md` Step 4 checklist done

Full CSV proof is **Step 5** — only after [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) has no open blockers.

---

## Stabilization

While fixing prescan/queue issues after initial Step 4 build:

- Follow [`STABILIZATION.md`](STABILIZATION.md) and **Execution path** S01–S09 in [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md)
- **Lane A (this chat):** S03 → S04 → S05 (source proxy, video-seek scrub, docs)
- **Lane B/C:** S02, S06–S09 — Step 3 patch chat (coordinator provides prompt)
- Step 5 blocked until **S07** (F001) cleared

| Seq | Lane A work |
|-----|-------------|
| S03 | `apps/web/src/app/api/proxy/source/route.ts` |
| S04 | `prescan-review-modal.tsx`, `calibration-canvas.tsx` — slider seeks `<video>`, not `prescan/preview` |
| S05 | Re-scan OCR only; sync `COMPONENT_MAP.md` |

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Stabilization path S01–S09 assigned; UI lane A picks up at S03 after S02 |
| 2026-08-19 | Renumbered from Step 3 (six-step plan) |
