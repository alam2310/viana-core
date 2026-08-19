# Step tracker (living)

**Last updated:** 2026-08-19  
**Current Step:** **1** — UX design  
**Canonical plan:** [`PLAN.md`](PLAN.md)  
**Agent checklist:** [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) — **read when starting or finishing any Step**

> Update this file when a Step changes status. Agents: do not rely on chat memory.

---

## Status overview

| Step | Name | Status | Owner chat | Started | Completed |
|------|------|--------|------------|---------|-----------|
| **1** | UX design | ⬜ Not started | New — UX design | — | — |
| **2** | Contract sync | ⏸ Skipped (until needed) | New — Contract | — | — |
| **3** | UI implementation | ⬜ Not started | New — UI v2 | — | — |
| **4** | E2E verification | ⬜ Not started | UI v2 or QA | — | — |
| **5** | Hardening backlog | ⬜ Not started | Per item | — | — |

**Status legend:** ⬜ Not started · 🔄 In progress · ✅ Complete · ⏸ Blocked / skipped · ❌ Cancelled

---

## Step 1 — UX design

| Deliverable | Status | Path |
|-------------|--------|------|
| Redesign spec (flows + screens) | ⬜ | `docs/ui/REDESIGN.md` (create) |
| Prescan modal spec | ⬜ | `docs/ui/REDESIGN.md` or `COMPONENT_MAP.md` |
| Dashboard / queue spec | ⬜ | `docs/ui/REDESIGN.md` |
| Aggregate / artifact UX | ⬜ | `docs/ui/REDESIGN.md` |
| Contract proposals (if any) | ⬜ | `STEP_2_CONTRACT_SYNC.md` § Proposals |

---

## Step 2 — Contract sync

| Gate | Status |
|------|--------|
| Required? | **TBD after Step 1** — likely no change |
| Schemas updated | — |
| Fixtures + TS synced | — |
| `api_contracts.md` updated | — |

---

## Step 3 — UI implementation

| Sub-step | Status | Surface |
|----------|--------|---------|
| 3.1 Prescan modal redesign | ⬜ | `apps/web/src/features/prescan/` |
| 3.2 Dashboard metadata on submit | ⬜ | `apps/web/src/features/dashboard/` |
| 3.3 Completed job artifacts + aggregate | ⬜ | dashboard / job cards |
| 3.4 Polish (toasts, empty states) | ⬜ | `apps/web/` |
| 3.5 `COMPONENT_MAP.md` synced | ⬜ | `docs/ui/` |

---

## Step 4 — E2E verification

| Check | Status |
|-------|--------|
| Test clip with known `user_start_time` | ⬜ |
| `{stem}_15min.csv` non-empty, correct windows | ⬜ |
| `time_map.json` present on completed job | ⬜ |
| Evidence in `verification/4_15min_results.md` | ⬜ |
| `PROJECT_STATUS.md` updated | ⬜ |

---

## Step 5 — Hardening (ordered backlog)

| Item | Work | Status | Chat |
|------|------|--------|------|
| 5.1 | Bake `trackers` + `numpy<2` in Docker image | ⬜ | API / DevOps |
| 5.2 | Pause / resume / PAUSED UX | ⬜ | UI (+ API) |
| 5.3 | Faster DELETE → CANCELLED | ⬜ | API |
| 5.4 | Browser / Playwright click-through | ⬜ | UI / QA |
| 5.5 | Extra camera clip validation | ⬜ | Engine / QA |
| 5.6 | GPU tests in CI | ⬜ | DevOps |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Renumbered Steps 1–5 (was A–E); added `AGENT_PROGRESS.md` |
| 2026-08-19 | Created Step tracker; Phases 0–9 complete; Step 1 is current focus |
