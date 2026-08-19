# Step kickoff prompts (post-v0.1)

Copy-paste as the **first message** in a **new chat** (unless noted).

**Before starting:** read [`TRACKER.md`](TRACKER.md) and [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md).

---

## When to start NEW vs continue

| Step | Chat |
|------|------|
| **1** | **One new chat** for discovery + design. Resume later → **new chat** + read `docs/ui/DISCOVERY.md` |
| **2** | **Always new** — Backend (never Phase 1–9 build chats) |
| **3** | **Always new** — UI v2 |
| **4** | New QA chat, or continue Step 3 |
| **5** | New per backlog item |
| Planning / this thread | **Not** for implementation |

---

## Step 1 — UX discovery & design (start here)

```
You are the ViAna UX DISCOVERY & DESIGN agent.

Your job has TWO phases in THIS chat:
  Phase 1.1 — Discovery: understand the big picture; ask me questions; record answers.
  Phase 1.2 — Design: write the redesign spec after discovery sign-off.

Read first:
1. docs/steps/TRACKER.md + AGENT_PROGRESS.md
2. docs/steps/STEP_1_UX_DESIGN.md
3. docs/project_context.md (ViAna_Moving, ViAnaNP, ViAnaJunction)
4. docs/specs/ui_specifications.md §3 (per-task prescan/calibration)
5. docs/ui/DISCOVERY.md — maintain this file as we talk
6. packages/contracts/typescript/index.ts — PrescanResponse, JobMetadata
7. apps/web/src/features/prescan/prescan-modal.tsx + src/viana/stages/prescan.py (current behavior)

v0.1 scope: ViAna_Moving only. Prescan must PROPOSE time, date, location, horizon line, counting line.
User must CONFIRM or EDIT each before submit. Future tasks differ (NP = metadata only; Junction = polygon + gates) — capture in task matrix even if not shipped.

START with Phase 1.1:
- Summarize your understanding of the product in 5–8 bullets.
- Ask me structured questions (batch 5–8 at a time) about operators, prescan UX, queue, artifacts, and task-type differences.
- Record Q&A in docs/ui/DISCOVERY.md §2.
- Fill §3 task-type matrix and §5 backend gaps as we go.
- Do NOT write REDESIGN.md until I confirm discovery sign-off (DISCOVERY.md §6).

After sign-off, Phase 1.2:
- Write docs/ui/REDESIGN.md (screens, ViAna_Moving propose→confirm flow, extensibility).
- Update USER_FLOWS.md / COMPONENT_MAP.md if needed.
- Copy backend gaps to docs/steps/STEP_2_BACKEND_ALIGNMENT.md § Work items.

No apps/web/ code. When done: AGENT_PROGRESS.md § On completing Step 1.
```

---

## Step 2 — Backend alignment (conditional)

```
You are the ViAna BACKEND alignment agent (Step 2).

Read:
1. docs/steps/STEP_2_BACKEND_ALIGNMENT.md § Work items (from Step 1)
2. docs/governance/CONTRACT_SYNC.md
3. docs/ui/REDESIGN.md + DISCOVERY.md §5
4. src/viana/stages/prescan.py, orchestrator prescan route, packages/contracts/schemas/

Implement only listed work items: contracts, prescan engine, and/or API route.
Schema-first for any payload changes. Add tests.

Surfaces: packages/contracts/, src/viana/stages/prescan.py, src/orchestrator/, tests/

Do NOT edit apps/web/ (Step 3). When done: AGENT_PROGRESS.md § On completing Step 2.
```

### Step 2 — Engine-only slice

```
ViAna ENGINE agent — Step 2 prescan slice only.

Read STEP_2_BACKEND_ALIGNMENT.md work items assigned to engine.
Implement src/viana/stages/prescan.py + tests/viana/test_prescan.py.
Schema changes → coordinate Contract step first.
```

### Step 2 — API-only slice

```
ViAna API agent — Step 2 prescan route slice only.

Read STEP_2_BACKEND_ALIGNMENT.md + api_contracts.md.
Map prescan HTTP to engine; update fixtures if response shape changed.
```

---

## Step 3 — UI implementation v2

```
You are the ViAna UI IMPLEMENTATION agent (Step 3).

Read:
1. docs/ui/REDESIGN.md (required — from Step 1)
2. docs/steps/STEP_3_UI_IMPLEMENTATION.md + TRACKER.md
3. apps/web/AGENTS.md

Env: NEXT_PUBLIC_USE_MOCKS=false, NEXT_PUBLIC_API_URL=http://localhost:8000

Build: 3.1 prescan propose/confirm/edit → 3.2 dashboard metadata → 3.3 artifacts → 3.4 polish → 3.5 docs.

New API field needed? STOP → Step 2.

When done: AGENT_PROGRESS.md § On completing Step 3.
```

---

## Step 4 — E2E verification

```
ViAna QA agent — Step 4 (15-min CSV).

Read: STEP_4_E2E_VERIFICATION.md, AGENT_PROGRESS.md, docs/ops/ENVIRONMENT_SETUP.md

Run ViAna_Moving happy path with known user_start_time.
Write docs/steps/verification/4_15min_results.md.
```

---

## Step 5 — Hardening (one item)

```
ViAna agent — Step 5 item [5.1–5.6] only.

Read: STEP_5_HARDENING.md, TRACKER.md, AGENT_PROGRESS.md.
```

---

## Engine — Step 4 bugfix only

```
Engine bugfix: metadata → time_map.json → _15min.csv (Step 4).

Read STEP_4_E2E_VERIFICATION.md, test_time_map.py, test_aggregate.py.
No apps/web/ edits.
```
