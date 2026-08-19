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

## On completing Step 1 — UX design

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 1 ✅; set **Completed** date | `TRACKER.md` |
| 2 | Check off all Step 1 deliverables | `TRACKER.md` § Step 1 |
| 3 | Set **Current Step** → **2** (or **3** if Step 2 skipped) | `TRACKER.md` |
| 4 | Update **Current focus** | `PROJECT_STATUS.md` |
| 5 | Changelog entry | `PROJECT_STATUS.md` § Changelog |
| 6 | Log completion note | `STEP_1_UX_DESIGN.md` § Log |
| 7 | Ensure primary output exists | `docs/ui/REDESIGN.md` |
| 8 | Update flows/components if changed | `docs/ui/USER_FLOWS.md`, `COMPONENT_MAP.md` |
| 9 | File contract proposals (or mark none) | `STEP_2_CONTRACT_SYNC.md` § Proposals |
| 10 | If no contract changes: mark Step 2 ⏸ Skipped | `TRACKER.md` |

**Do not edit** `apps/web/` in Step 1 (design-only).

---

## On completing Step 2 — Contract sync

Skip this section if Step 2 was not required.

| # | Action | File(s) |
|---|--------|---------|
| 1 | Mark Step 2 ✅ or ⏸ Skipped with reason | `TRACKER.md` |
| 2 | Check schema / fixture / TS gates | `TRACKER.md` § Step 2 |
| 3 | Set **Current Step** → **3** | `TRACKER.md` |
| 4 | Update **Current focus** | `PROJECT_STATUS.md` |
| 5 | Changelog if HTTP or schema changed | `PROJECT_STATUS.md` |
| 6 | Log completion | `STEP_2_CONTRACT_SYNC.md` § Log |
| 7 | Confirm contract order per | `docs/governance/CONTRACT_SYNC.md` |

**Typical edits:** `packages/contracts/schemas/`, `typescript/index.ts`, `fixtures/`, `src/viana/config/job.py`, `docs/api_contracts.md`, `openapi.yaml`.

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
| `docs/ui/REDESIGN.md` | Step 1 deliverable |
| `docs/ui/COMPONENT_MAP.md` | Step 1 spec; Step 3 sync |
| `packages/contracts/*` | Step 2 only |
| `apps/web/*` | Step 3 only |
| `docs/steps/verification/*` | Step 4+ evidence |

---

## Commit message examples

```
Complete Step 1: prescan UX redesign spec
Complete Step 2: add proposed_metadata to prescan schema
Step 3.1: prescan modal OCR review layout
Complete Step 4: verify 15min CSV on hiv000001_inframe
Step 5.1: bake trackers into Docker image
```
