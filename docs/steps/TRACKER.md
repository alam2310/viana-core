# Step tracker (living)

**Last updated:** 2026-08-19  
**Current Step:** **1** — UX discovery & design (Phase 1.1 in progress)  
**Canonical plan:** [`PLAN.md`](PLAN.md)  
**Agent checklist:** [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md)

> Update this file when a Step changes status. Agents: do not rely on chat memory.

---

## Status overview

| Step | Name | Status | Owner chat | Started | Completed |
|------|------|--------|------------|---------|-----------|
| **1** | UX discovery & design | 🔄 In progress | UX discovery chat | 2026-08-19 | — |
| **2** | Backend alignment | ⏸ Skipped until needed | New — Backend | — | — |
| **3** | UI implementation | ⬜ Not started | New — UI v2 | — | — |
| **4** | E2E verification | ⬜ Not started | UI v2 or QA | — | — |
| **5** | Hardening backlog | ⬜ Not started | Per item | — | — |

**Status legend:** ⬜ Not started · 🔄 In progress · ✅ Complete · ⏸ Skipped · ❌ Cancelled

---

## Step 1 — UX discovery & design

| Phase / deliverable | Status | Path |
|-------------------|--------|------|
| 1.1 Discovery Q&A + sign-off | 🔄 | `docs/ui/DISCOVERY.md` |
| 1.2 Task-type prescan matrix | ⬜ | `DISCOVERY.md` §3 |
| 1.3 Redesign spec | ⬜ | `docs/ui/REDESIGN.md` |
| 1.4 Flows / component map | ⬜ | `USER_FLOWS.md`, `COMPONENT_MAP.md` |
| 1.5 Backend gap list | ⬜ | `STEP_2_BACKEND_ALIGNMENT.md` § Work items |

---

## Step 2 — Backend alignment

| Gate | Status |
|------|--------|
| Required? | **TBD after Step 1** |
| Contract / schema | — |
| Prescan engine | — |
| Prescan API route | — |
| Tests updated | — |

---

## Step 3 — UI implementation

| Sub-step | Status | Surface |
|----------|--------|---------|
| 3.1 Prescan modal (propose → confirm/edit) | ⬜ | `apps/web/src/features/prescan/` |
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

---

## Step 5 — Hardening

| Item | Work | Status | Chat |
|------|------|--------|------|
| 5.1 | Docker image bake | ⬜ | API / DevOps |
| 5.2 | Pause / resume UX | ⬜ | UI (+ API) |
| 5.3 | Faster DELETE → CANCELLED | ⬜ | API |
| 5.4 | Playwright UI pass | ⬜ | UI / QA |
| 5.5 | Extra camera clip | ⬜ | Engine / QA |
| 5.6 | GPU tests in CI | ⬜ | DevOps |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-19 | Step 1 = discovery + design; Step 2 = backend alignment incl. prescan |
| 2026-08-19 | Steps 1–5 numbered; `AGENT_PROGRESS.md` added |
| 2026-08-19 | Tracker created; Phases 0–9 complete |
