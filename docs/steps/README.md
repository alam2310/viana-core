# Post-v0.1 Steps (index)

Work **after** Phases 0–9 is tracked as **Steps 1–6** (not Phases).

| Doc | Purpose |
|-----|---------|
| [`TRACKER.md`](TRACKER.md) | **Living status** |
| [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) | What to update per Step |
| [`PLAN.md`](PLAN.md) | Narrative plan + chat map |
| [`KICKOFF_PROMPTS.md`](KICKOFF_PROMPTS.md) | Copy-paste for new chats |
| [`STEP_1_UX_DESIGN.md`](STEP_1_UX_DESIGN.md) | UX discovery & design |
| [`STEP_2_CONTRACTS_AND_API.md`](STEP_2_CONTRACTS_AND_API.md) | Schemas + intake/confirm APIs |
| [`STEP_3_ENGINE_AND_ORCHESTRATOR.md`](STEP_3_ENGINE_AND_ORCHESTRATOR.md) | Prescan engine + workers |
| [`STEP_4_UI_IMPLEMENTATION.md`](STEP_4_UI_IMPLEMENTATION.md) | UI build |
| [`STEP_5_E2E_VERIFICATION.md`](STEP_5_E2E_VERIFICATION.md) | 15-min CSV E2E |
| [`STEP_6_HARDENING.md`](STEP_6_HARDENING.md) | Post-ship backlog |
| [`STABILIZATION.md`](STABILIZATION.md) | Step 4→5 gate rules; agents log defects here |
| [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) | Living defect list + **execution path S01–S09** |

**Agent rule:** New chat per Step (2–4). During Step 4 acceptance, log defects in [`STABILIZATION_BACKLOG.md`](STABILIZATION_BACKLOG.md) per [`STABILIZATION.md`](STABILIZATION.md).

---

## Naming

| Term | Meaning |
|------|---------|
| **Phase 0–9** | Original platform build — **complete** |
| **Step 1–6** | Post-v0.1 — **active** |
| **Sub-step** | e.g. 4.1, 4.2 inside Step 4 |

---

## Backend split (Steps 2 vs 3)

| Step | Owns |
|------|------|
| **2** | Contracts, `JobStatus`, intake/confirm HTTP, validation, job storage shape |
| **3** | Prescan engine, worker queue, auto-aggregate, partial MP4, dark-frame skip |
