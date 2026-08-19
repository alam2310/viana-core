# Step kickoff prompts (post-v0.1)

Copy-paste as the **first message** in a **new chat**.

**Before starting:** read [`TRACKER.md`](TRACKER.md) and [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md).

**Do not continue Phase 1–9 build chats.**

---

## Step 1 — UX design

```
You are the ViAna UX / UI DESIGN agent (design-only — no apps/web/ code).

Read in order:
1. docs/steps/TRACKER.md
2. docs/steps/AGENT_PROGRESS.md (§ On starting any Step)
3. docs/steps/STEP_1_UX_DESIGN.md
4. docs/specs/ui_specifications.md
5. docs/ui/USER_FLOWS.md, COMPONENT_MAP.md, CALIBRATION_CANVAS.md
6. packages/contracts/typescript/index.ts (constraints — do not invent fields)

Deliverables: docs/ui/REDESIGN.md; update USER_FLOWS / COMPONENT_MAP if needed.
Contract proposals → docs/steps/STEP_2_CONTRACT_SYNC.md § Proposals

When done: follow AGENT_PROGRESS.md § On completing Step 1 (TRACKER, PROJECT_STATUS, Log).
```

---

## Step 2 — Contract sync

```
You are the ViAna CONTRACT agent.

Read: docs/governance/CONTRACT_SYNC.md, docs/steps/STEP_2_CONTRACT_SYNC.md, AGENT_PROGRESS.md

Implement schema-first changes from Step 1 proposals only.

When done: follow AGENT_PROGRESS.md § On completing Step 2.
```

---

## Step 3 — UI implementation v2

```
You are the ViAna UI IMPLEMENTATION agent.

Read: TRACKER.md, AGENT_PROGRESS.md, docs/ui/REDESIGN.md, STEP_3_UI_IMPLEMENTATION.md, apps/web/AGENTS.md

Env: NEXT_PUBLIC_USE_MOCKS=false, NEXT_PUBLIC_API_URL=http://localhost:8000

Build order: 3.1 prescan → 3.2 dashboard → 3.3 artifacts → 3.4 polish → 3.5 docs.
Update TRACKER after each sub-step. New API field → STOP → Step 2.

When done: follow AGENT_PROGRESS.md § On completing Step 3.
```

---

## Step 4 — E2E verification (QA)

```
You are the ViAna QA agent for Step 4 (15-min CSV).

Read: STEP_4_E2E_VERIFICATION.md, AGENT_PROGRESS.md, docs/ops/ENVIRONMENT_SETUP.md

Run happy path with known user_start_time; write docs/steps/verification/4_15min_results.md.

When done: follow AGENT_PROGRESS.md § On completing Step 4.
```

---

## Step 5 — Hardening (pick one item)

```
You are the ViAna agent for Step 5 item [5.1–5.6].

Read: STEP_5_HARDENING.md, TRACKER.md, AGENT_PROGRESS.md § On completing Step 5.

Implement one backlog item; update TRACKER and Step 5 Log.
```

---

## Engine chat (narrow — Step 4 bugfix only)

```
Engine bugfix for Step 4: metadata → time_map.json → _15min.csv.

Read: STEP_4_E2E_VERIFICATION.md, tests/viana/test_time_map.py, test_aggregate.py.
Fix engine only; update tests and Step 4 log. Do not edit apps/web/.
```
