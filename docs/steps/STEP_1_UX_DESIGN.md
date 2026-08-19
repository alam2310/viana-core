# Step 1 — UX design

| Field | Value |
|-------|-------|
| **Status** | ⬜ Not started — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | New — **UX design** (design-only) |
| **Blocks** | Step 3 |
| **Blocked by** | — |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 1.

---

## Objective

Produce a UX specification so prescan, calibration, queue, and job outcomes are clear for operators — especially **OCR-proposed vs user-edited wall-clock metadata** needed for `{stem}_15min.csv`.

---

## Scope

**In scope:** prescan modal, dashboard/queue, aggregate UX, copy and layout (no React code).

**Out of scope:** `apps/web/` implementation; engine/API code; new API fields (list in Step 2).

---

## Read order

1. `docs/specs/ui_specifications.md` §3
2. `docs/ui/USER_FLOWS.md`, `COMPONENT_MAP.md`, `CALIBRATION_CANVAS.md`
3. `docs/PROJECT_STATUS.md`
4. `apps/web/src/features/prescan/prescan-modal.tsx` (reference only)

---

## Deliverables

| # | Deliverable | Path |
|---|-------------|------|
| 1.1 | Master redesign doc | `docs/ui/REDESIGN.md` |
| 1.2 | Updated flows (if needed) | `docs/ui/USER_FLOWS.md` |
| 1.3 | Updated component map | `docs/ui/COMPONENT_MAP.md` |
| 1.4 | Contract proposals (if any) | `STEP_2_CONTRACT_SYNC.md` § Proposals |

---

## Exit criteria

- [ ] Developer can implement Step 3 without open UX questions
- [ ] Contracts satisfied OR proposals filed for Step 2
- [ ] `TRACKER.md` Step 1 deliverables checked off
- [ ] `AGENT_PROGRESS.md` Step 1 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step created (numbered Steps 1–5) |
