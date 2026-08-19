# Step kickoff prompts (post-v0.1)

Copy-paste as the **first message** in a **new chat**.

**Read:** [`TRACKER.md`](TRACKER.md) + [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md).

---

## When to start NEW vs continue

| Step | Chat |
|------|------|
| 1 | One chat for discovery + design; resume later → **new** + `DISCOVERY.md` |
| 2 | **Always new** |
| 3 | **Always new** (after Step 2 merges) |
| 4 | **Always new** (after Step 3) |
| 5 | New QA or continue Step 4 |
| 6 | New per item |

---

## Step 1 — UX discovery & design ✅ (reference)

Step 1 is complete. See `docs/ui/DISCOVERY.md` and `docs/ui/REDESIGN.md`. Do not restart unless redesign changes.

---

## Step 2 — Contracts & API foundation (start here)

```
You are the ViAna CONTRACT + API agent (Step 2).

Read:
1. docs/steps/TRACKER.md — Step 2 current; Step 1 complete
2. docs/steps/AGENT_PROGRESS.md § On completing Step 2
3. docs/steps/STEP_2_CONTRACTS_AND_API.md
4. docs/governance/CONTRACT_SYNC.md
5. docs/ui/REDESIGN.md § job lifecycle + docs/ui/DISCOVERY.md §7
6. packages/contracts/schemas/job_status.schema.json, job_submit.schema.json, telemetry.schema.json

Scope (Step 2 ONLY — not engine workers):
- P1–P6: schemas → TS → fixtures → job.py → api_contracts.md → openapi.yaml
- G1, G14, G15: proposed_* + confirmed fields; JobStatus enum
- G16: POST /jobs/intake
- G17: PATCH /jobs/{id}/prescan → READY
- G4: validate metadata HH:MM:SS + DD-MM-YYYY, all three mandatory
- G20: output_dir on job config

Do NOT implement: prescan worker queue, dark-frame skip, auto-aggregate, partial MP4 (Step 3).
Do NOT edit apps/web/ (Step 4).

Tests: tests/orchestrator/. When done: AGENT_PROGRESS.md § Step 2.
```

---

## Step 3 — Engine prescan & orchestrator workers

```
You are the ViAna ENGINE + ORCHESTRATOR agent (Step 3).

Prerequisite: Step 2 complete (contracts + intake/confirm routes exist).

Read:
1. docs/steps/STEP_3_ENGINE_AND_ORCHESTRATOR.md
2. docs/steps/AGENT_PROGRESS.md § On completing Step 3
3. src/viana/stages/prescan.py, src/orchestrator/workers/pool.py

Implement:
- G13 prescan worker queue (bulk intake)
- G7 dark-frame auto-skip in sampler
- G8 frame preview endpoint or prescan-at-offset
- G12 auto-aggregate on COMPLETED
- G19 partial _processed.mp4 range serving
- G9 ETA + crossing count on status/telemetry
- GPU workers only pick READY jobs; PRESCAN_FAILED retry

Do NOT edit apps/web/. Schema gap → stop and note for Step 2.

When done: AGENT_PROGRESS.md § Step 3.
```

---

## Step 4 — UI implementation v2

```
You are the ViAna UI agent (Step 4).

Prerequisite: Steps 2–3 complete.

Read: docs/ui/REDESIGN.md, STEP_4_UI_IMPLEMENTATION.md, apps/web/AGENTS.md
Env: NEXT_PUBLIC_USE_MOCKS=false, NEXT_PUBLIC_API_URL=http://localhost:8000

Build: 4.1 intake+queue → 4.2 prescan review → 4.3 live monitor → 4.4 artifacts → 4.5 polish.

When done: AGENT_PROGRESS.md § Step 4.
```

---

## Step 5 — E2E verification

```
ViAna QA — Step 5. Read STEP_5_E2E_VERIFICATION.md.
Run intake → prescan → confirm → COMPLETED → verify _15min.csv.
Write docs/steps/verification/5_15min_results.md.
```

---

## Step 6 — Hardening (one item)

```
ViAna agent — Step 6 item [6.1–6.7]. Read STEP_6_HARDENING.md, TRACKER.md.
```

---

## Engine — Step 5 bugfix only

```
Engine bugfix for Step 5: time_map → _15min.csv. Read STEP_5_E2E_VERIFICATION.md. No apps/web/.
```
