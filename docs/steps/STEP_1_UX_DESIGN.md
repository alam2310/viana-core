# Step 1 — UX discovery & design

| Field | Value |
|-------|-------|
| **Status** | ✅ Complete — see [`TRACKER.md`](TRACKER.md) |
| **Chat** | **New** — UX discovery & design (one chat for whole Step 1) |
| **Blocks** | Step 2 (if backend gaps), Step 3 |
| **Blocked by** | — |

**On complete:** follow [`AGENT_PROGRESS.md`](AGENT_PROGRESS.md) § On completing Step 1.

---

## Objective

Understand the **full product UX** (big picture + per-task prescan differences), capture stakeholder input via Q&A, then finalize a redesign spec implementable in Step 3.

**v0.1 focus:** `ViAna_Moving` — prescan proposes **time, date, location, horizon line, counting line**; user **confirms or edits** each before submit.

**Future tasks** (`ViAnaNP`, `ViAnaJunction`): document prescan/calibration differences in the task matrix even if not shipped in v0.1.

---

## Two phases (same chat)

| Phase | Work | Output |
|-------|------|--------|
| **1.1 Discovery** | Read docs; ask user questions; understand goals | [`docs/ui/DISCOVERY.md`](../ui/DISCOVERY.md) |
| **1.2 Design finalize** | Wireframes/flows/copy; task-type matrix; prescan review UX | [`docs/ui/REDESIGN.md`](../ui/REDESIGN.md) |

Do **not** skip 1.1. Do **not** write `REDESIGN.md` until discovery sign-off in `DISCOVERY.md` §6.

---

## Scope

**In scope**

- Product-wide screen map (dashboard, queue, prescan, telemetry, artifacts, container)
- Per-`task_type` prescan & calibration requirements
- `ViAna_Moving`: proposed vs user-edited OCR + lines; wall-clock impact on `_15min.csv`
- Extensibility notes for NP / Junction
- Backend gap list for Step 2 (if prescan API/engine must change)

**Out of scope**

- `apps/web/` code; engine/API implementation (Step 2)
- Inventing API fields without filing Step 2 work items

---

## Read order (before Q&A)

1. `docs/project_context.md` — three product lines
2. `docs/specs/ui_specifications.md` §3 — task-specific calibration
3. `docs/ui/USER_FLOWS.md`, `COMPONENT_MAP.md`, `CALIBRATION_CANVAS.md`
4. `packages/contracts/typescript/index.ts` — `PrescanResponse`, `JobMetadata`, `task_type`
5. `docs/api_contracts.md` — `POST /utils/prescan`
6. `apps/web/src/features/prescan/prescan-modal.tsx` (current UI)
7. `src/viana/stages/prescan.py` (what engine proposes today)

---

## Discovery topics (ask the user)

Use chat Q&A; record in `DISCOVERY.md` §2. Suggested themes:

1. **Operators** — who uses the UI, typical batch size, error tolerance
2. **ViAna_Moving prescan** — which fields are mandatory before submit; OCR low-confidence UX; date/time format
3. **Line proposal** — trust auto-lines vs always edit; “apply to all pending” behavior
4. **Queue & monitor** — what job card must show (metadata, progress, artifacts)
5. **Completed job** — download paths, aggregate trigger, empty `_15min.csv` messaging
6. **Future tasks** — how should task picker affect prescan UI (even if NP/Junction not in v0.1)
7. **Visual / density** — modal vs wizard; sidebar layout priorities

---

## Deliverables

| # | Deliverable | Path |
|---|-------------|------|
| 1.1 | Discovery log + Q&A + sign-off | `docs/ui/DISCOVERY.md` |
| 1.2 | Task-type prescan matrix | `DISCOVERY.md` §3 + `REDESIGN.md` |
| 1.3 | Master redesign spec | `docs/ui/REDESIGN.md` |
| 1.4 | Updated flows / components | `USER_FLOWS.md`, `COMPONENT_MAP.md` |
| 1.5 | Backend gap list (if any) | `docs/steps/STEP_2_BACKEND_ALIGNMENT.md` § Work items |

---

## Exit criteria

- [ ] `DISCOVERY.md` §6 signed off (user confirmed in chat)
- [ ] `REDESIGN.md` covers all screens + `ViAna_Moving` prescan confirm/edit flow
- [ ] Step 2 work items filed OR Step 2 marked skip in `TRACKER.md`
- [ ] `AGENT_PROGRESS.md` Step 1 checklist done

---

## Log

| Date | Note |
|------|------|
| 2026-08-19 | Step created (numbered Steps 1–5) |
| 2026-08-19 | Split into discovery + design; added task-type prescan matrix |
| 2026-08-19 | Phase 1.1: telemetry UX plan (§9), PRESCAN_FAILED status, container path backlog 5.7 |
| 2026-08-19 | Step 1 complete — REDESIGN.md, flows, component map, Step 2 work items |
