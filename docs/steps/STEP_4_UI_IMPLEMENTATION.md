# Step 4 — UI implementation

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
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

- [ ] Sub-steps 4.1–4.5 ✅ in `TRACKER.md`
- [ ] Full intake → review → READY → process flow works manually
- [ ] `AGENT_PROGRESS.md` Step 4 checklist done

Full CSV proof is **Step 5**.

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Renumbered from Step 3 (six-step plan) |
