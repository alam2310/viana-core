# Agent progress checklist (Steps 1–5)

**Read this when starting or finishing any post-v0.1 Step.** Do not rely on chat memory.

| Doc | When |
|-----|------|
| [`TRACKER.md`](TRACKER.md) | Every session — current Step and deliverable status |
| This file | When a Step **starts**, **completes**, or **skips** |
| [`STEP_N_*.md`](.) | Step-specific scope and exit criteria |

**Naming:** **Phase 0–9** = original build (complete). **Step 1–5** = post-v0.1 work (active).

---

## On starting any Step

1. Read [`TRACKER.md`](TRACKER.md) — confirm current Step and prerequisites.
2. Read the matching `STEP_N_*.md` and kickoff in [`KICKOFF_PROMPTS.md`](KICKOFF_PROMPTS.md).
3. In **`TRACKER.md`**:
   - Set Step status to **🔄 In progress**
   - Fill **Started** date
   - Set **Current Step** at top of `TRACKER.md`
4. In **`docs/PROJECT_STATUS.md`**:
   - Update **Current focus** line to this Step
5. Append a row to the Step file **Log** section (e.g. `STEP_1_UX_DESIGN.md`).

---

## On completing Step 1 — UX discovery & design

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 1 ✅; set **Completed** date | `TRACKER.md` |
| 2 | Check off phases 1.1–1.5 in tracker | `TRACKER.md` § Step 1 |
| 3 | Confirm `DISCOVERY.md` §6 signed off | `docs/ui/DISCOVERY.md` |
| 4 | Set **Current Step** → **2** (or **3** if Step 2 skipped) | `TRACKER.md` |
| 5 | Update **Current focus** | `PROJECT_STATUS.md` |
| 6 | Changelog entry | `PROJECT_STATUS.md` § Changelog |
| 7 | Log completion | `STEP_1_UX_DESIGN.md` § Log |
| 8 | Ensure `REDESIGN.md` exists | `docs/ui/REDESIGN.md` |
| 9 | Update flows/components if changed | `docs/ui/USER_FLOWS.md`, `COMPONENT_MAP.md` |
| 10 | File Step 2 work items OR mark Step 2 ⏸ Skipped | `STEP_2_BACKEND_ALIGNMENT.md`, `TRACKER.md` |

**Do not edit** `apps/web/`, `src/viana/`, or `src/orchestrator/` in Step 1.

---

## On completing Step 2 — Backend alignment

Skip if Step 1 filed no work items and UX fits current prescan/API.

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 2 ✅ or ⏸ Skipped with reason | `TRACKER.md` |
| 2 | Check contract + prescan gates | `TRACKER.md` § Step 2 |
| 3 | All work items ✅ in `STEP_2_BACKEND_ALIGNMENT.md` | § Work items |
| 4 | Set **Current Step** → **3** | `TRACKER.md` |
| 5 | Update **Current focus** | `PROJECT_STATUS.md` |
| 6 | Changelog if HTTP, prescan, or schema changed | `PROJECT_STATUS.md` |
| 7 | Log completion | `STEP_2_BACKEND_ALIGNMENT.md` § Log |
| 8 | Tests pass for touched paths | `tests/viana/test_prescan.py`, orchestrator tests |

**Typical edits:** `packages/contracts/*`, `src/viana/stages/prescan.py`, `src/orchestrator/routes/`, `docs/api_contracts.md`.

---

## On completing Step 3 — UI implementation

Update **`TRACKER.md` § Step 3** after each sub-step (3.1–3.5), not only at the end.

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark sub-steps 3.1–3.5 ✅ | `TRACKER.md` § Step 3 |
| 2 | Mark Step 3 ✅; **Completed** date | `TRACKER.md` |
| 3 | Set **Current Step** → **4** | `TRACKER.md` |
| 4 | Update **Current focus** | `PROJECT_STATUS.md` |
| 5 | Changelog entry | `PROJECT_STATUS.md` |
| 6 | Log completion | `STEP_3_UI_IMPLEMENTATION.md` § Log |
| 7 | Sync component map with shipped UI | `docs/ui/COMPONENT_MAP.md` |

**Code surface:** `apps/web/` only (unless Step 2 contract edits).

---

## On completing Step 4 — E2E verification

| # | Action | File(s) |
|---|--------|---------|
| 1 | Check all D1–D3 rows in tracker | `TRACKER.md` § Step 4 |
| 2 | Mark Step 4 ✅ | `TRACKER.md` |
| 3 | Set **Current Step** → **5** (or done if skipping hardening) | `TRACKER.md` |
| 4 | Move § Next goals to done / update focus | `PROJECT_STATUS.md` |
| 5 | Changelog — note 15-min CSV verified or gap | `PROJECT_STATUS.md` |
| 6 | Log results | `STEP_4_E2E_VERIFICATION.md` § Log |
| 7 | Add evidence file | `docs/steps/verification/4_15min_results.md` |

**If engine bug found:** fix in Engine chat; update tests in `tests/viana/`; note in Step 4 log.

---

## On completing Step 5 — Hardening (per item)

Complete checklist **per backlog item** (5.1–5.6), not necessarily the whole Step at once.

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark item ✅ in tracker | `TRACKER.md` § Step 5 |
| 2 | Log item in Step 5 file | `STEP_5_HARDENING.md` § Log |
| 3 | Update ops/deployment docs if needed | `docs/ops/ENVIRONMENT_SETUP.md`, `DEPLOYMENT.md` |
| 4 | Changelog if user-visible | `PROJECT_STATUS.md` |
| 5 | Optional evidence | `docs/steps/verification/5_N_short_name.md` |

When **all** items done or explicitly deferred: mark Step 5 ✅ in `TRACKER.md`.

---

## Quick reference — files agents touch

| File | Update when |
|------|-------------|
| `docs/steps/TRACKER.md` | Every Step start / sub-step / complete |
| `docs/PROJECT_STATUS.md` | Step complete; focus change; changelog |
| `docs/steps/STEP_N_*.md` | Log notes; proposals (Step 2) |
| `docs/steps/AGENT_PROGRESS.md` | Only if process changes (rare) |
| `docs/ui/DISCOVERY.md` | Step 1 discovery Q&A (before REDESIGN) |
| `docs/ui/REDESIGN.md` | Step 1 final spec |
| `docs/ui/COMPONENT_MAP.md` | Step 1 spec; Step 3 sync |
| `packages/contracts/*` | Step 2 only |
| `src/viana/stages/prescan.py` | Step 2 if prescan engine changes |
| `src/orchestrator/` prescan routes | Step 2 if API changes |
| `apps/web/*` | Step 3 only |
| `docs/steps/verification/*` | Step 4+ evidence |

---

## Commit message examples

```
Complete Step 1: UX discovery and REDESIGN spec
Complete Step 2: prescan proposes separate OCR vs user fields
Step 2: engine prescan per-task proposal behavior
Step 3.1: prescan modal OCR review layout
Complete Step 4: verify 15min CSV on hiv000001_inframe
Step 5.1: bake trackers into Docker image
```
