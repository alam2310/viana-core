# Step 3 — UI implementation

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | New — **UI implementation v2** |
| **Blocks** | Step 4 |
| **Blocked by** | Step 1; Step 2 if backend gaps listed |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 3. Update tracker after **each** sub-step.

---

## Objective

Implement the Step 1 UX spec in `apps/web/`, with live API and job `metadata` on submit for wall-clock aggregation.

---

## Prerequisites

- [ ] `docs/ui/REDESIGN.md` approved (from Step 1)
- [ ] Step 2 complete or skipped (prescan backend aligned)
- [ ] API running (`NEXT_PUBLIC_USE_MOCKS=false`)

---

## Build order

| Sub-step | Work | Primary paths |
|----------|------|---------------|
| **3.1** | Prescan modal — propose → confirm/edit per REDESIGN | `apps/web/src/features/prescan/` |
| **3.2** | Dashboard metadata on submit | `apps/web/src/features/dashboard/` |
| **3.3** | Completed job artifacts + aggregate | dashboard, `api-client.ts` |
| **3.4** | Polish (toasts, empty states) | `apps/web/` |
| **3.5** | Doc sync | `docs/ui/COMPONENT_MAP.md` |

---

## Contract changes during Step 3

Stop implementation → file Step 2 proposal → complete schema sync → resume. See `docs/governance/CONTRACT_SYNC.md`.

---

## Exit criteria

- [ ] Sub-steps 3.1–3.5 ✅ in `TRACKER.md`
- [ ] Prescan → submit → aggregate works manually
- [ ] `AGENT_PROGRESS.md` Step 3 checklist done

Full CSV proof is **Step 4**.

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step created (numbered Steps 1–5) |
