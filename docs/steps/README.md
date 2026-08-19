# Post-v0.1 Steps (index)

Work **after** implementation Phases 0–9 is tracked here as **Steps 1–5** (not Phases) to avoid confusion with `docs/PROJECT_PLAN.md`.

| Doc | Purpose |
|-----|---------|
| [`TRACKER.md`](TRACKER.md) | **Living status** — update when a Step starts or completes |
| [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) | **What to update** when progressing through Steps 1–5 |
| [`PLAN.md`](PLAN.md) | Full narrative plan, chat strategy, timeline |
| [`KICKOFF_PROMPTS.md`](KICKOFF_PROMPTS.md) | Copy-paste prompts for new agent chats |
| [`STEP_1_UX_DESIGN.md`](STEP_1_UX_DESIGN.md) | UX redesign (design-only) |
| [`STEP_2_CONTRACT_SYNC.md`](STEP_2_CONTRACT_SYNC.md) | Contract changes (conditional) |
| [`STEP_3_UI_IMPLEMENTATION.md`](STEP_3_UI_IMPLEMENTATION.md) | Build UI from redesign |
| [`STEP_4_E2E_VERIFICATION.md`](STEP_4_E2E_VERIFICATION.md) | Verify `{stem}_15min.csv` end-to-end |
| [`STEP_5_HARDENING.md`](STEP_5_HARDENING.md) | Post-ship hardening backlog |

**Implementation phases (historical):** `docs/PROJECT_PLAN.md`, `docs/PROJECT_STATUS.md` § Overall progress.

**Agent rule:** Start a **new chat** per Step. Follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) on start and complete. Repo docs are the source of truth.

---

## Naming convention

| Term | Meaning |
|------|---------|
| **Phase 0–9** | Original v2 platform build (engine, API, UI, parity) — **complete** |
| **Step 1–5** | Post-v0.1 product polish and hardening — **active** |
| **Sub-step** | Ordered work inside a Step (e.g. 3.1, 3.2 in Step 3) |

---

## How to update progress

1. **Starting a Step** — see [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On starting any Step.
2. **Finishing a Step** — complete the checklist for that Step in `AGENT_PROGRESS.md`.
3. Always edit [`TRACKER.md`](TRACKER.md) and the Step file **Log**.
4. Commit with message referencing the Step (e.g. `Complete Step 1: prescan UX spec`).
