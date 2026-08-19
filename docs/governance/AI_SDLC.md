# AI SDLC Governance

How AI agents should work on this repo to minimize context drift, hallucination, and integration errors.

---

## 1. Principles

| Principle | Practice |
|-----------|----------|
| **Repo over memory** | Read `AGENTS.md` and `docs/PROJECT_STATUS.md` every session |
| **Schema first** | Change `packages/contracts/schemas/` before TypeScript, Pydantic, or routes |
| **Status honesty** | If an endpoint is not ✅ in `PROJECT_STATUS.md`, do not implement UI as if it exists without mocks |
| **Single writer per layer** | Engine logic in `src/viana/` only; no CV code in `apps/web/` |
| **ADRs for pivots** | New architecture decision → `docs/adr/NNN-title.md` |
| **Parity record** | Phase 9 signed off — see `tests/viana/fixtures/PARITY_NOTES.md` |

---

## 2. Session checklist (every agent)

1. Read `AGENTS.md`
2. Read `docs/PROJECT_STATUS.md` — confirm current phase or Step
3. If doing post-v0.1 work: read `docs/steps/TRACKER.md` and `docs/steps/AGENT_PROGRESS.md`
4. Read task-specific `AGENTS.md` (`apps/web/`, etc.)
5. Read relevant schema in `packages/contracts/schemas/`
6. Implement
7. Update `PROJECT_STATUS.md` and follow `AGENT_PROGRESS.md` if you completed a Step
8. Run tests listed in `AGENTS.md` §7 (`make test`, `make lint`, `make typecheck`)

---

## 3. Anti-hallucination rules

**Do not:**

- Invent API endpoints not in `docs/api_contracts.md`
- Invent CSV columns not in `events_*.schema.json`
- Assume parity reference is `tests/viana/fixtures/PARITY_NOTES.md` (legacy tree removed)
- Send `job_id` / `gpu_device` from UI submit payloads
- Put 15-minute binning inside the GPU frame loop

**Do:**

- Cite file paths when referencing behavior
- Use fixtures when API is ❌ in status matrix
- Add failing tests before fixing engine bugs
- Cross-check `configs/classes.yaml` for class names

---

## 4. Contract change workflow

Full cross-track rules: **`docs/governance/CONTRACT_SYNC.md`**

```
1. Edit packages/contracts/schemas/*.json
2. Edit packages/contracts/typescript/index.ts
3. Add/update packages/contracts/fixtures/*.json (if UI mocks the shape)
4. Edit src/viana/config/job.py (Pydantic)
5. Edit docs/api_contracts.md + openapi.yaml (if HTTP)
6. Update docs/PROJECT_STATUS.md if endpoint newly available
7. Implement handler / UI consumer (other tracks wait for steps 1–6)
```

---

## 5. Branch / agent parallelism

| Agent | Owned paths | Coordination |
|-------|-------------|--------------|
| Engine | `src/viana/`, `tests/viana/`, `configs/` | Exposes CLI; no HTTP in engine |
| API | `src/orchestrator/` | Spawns CLI; owns job state |
| UI | `apps/web/`, `docs/ui/` | Consumes contracts only |

**Conflict resolution:** `packages/contracts/` wins over any implementation. See `CONTRACT_SYNC.md`.

**Parallel development:** `PARALLEL_AGENTS.md`, kickoff prompts in `KICKOFF_PROMPTS.md`.

**Post-v0.1 Steps 1–5:** `docs/steps/TRACKER.md`, `docs/steps/AGENT_PROGRESS.md`, kickoffs in `docs/steps/KICKOFF_PROMPTS.md`.

---

## 6. Definition of done (per phase or Step)

- Code matches schema
- Tests pass (`pytest tests/viana/` or UI lint/build)
- `PROJECT_STATUS.md` updated
- **Steps 1–5:** `TRACKER.md` + checklist in `docs/steps/AGENT_PROGRESS.md` completed
- No edits to `training/` unless extending the retrain workflow
- ADR added if behavior diverges from plan

---

## 7. Related docs

- `docs/steps/AGENT_PROGRESS.md` — what to update when progressing Steps 1–5
- `docs/governance/SOURCE_OF_TRUTH.md` — which file owns which fact
- `docs/governance/CONTEXT_MAP.md` — quick lookup index
- `docs/adr/` — decision log
