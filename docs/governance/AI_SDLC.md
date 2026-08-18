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
| **Parity before delete** | Do not remove `legacy/` until Phase 9 gate in `PROJECT_STATUS.md` |

---

## 2. Session checklist (every agent)

1. Read `AGENTS.md`
2. Read `docs/PROJECT_STATUS.md` — confirm current phase
3. Read task-specific `AGENTS.md` (`apps/web/`, etc.)
4. Read relevant schema in `packages/contracts/schemas/`
5. Implement
6. Update `PROJECT_STATUS.md` if you completed a milestone
7. Run tests listed in `AGENTS.md` §7 (`make test`, `make lint`, `make typecheck`)

---

## 3. Anti-hallucination rules

**Do not:**

- Invent API endpoints not in `docs/api_contracts.md`
- Invent CSV columns not in `events_*.schema.json`
- Assume `inference_engine.py` lives under `src/` (it is `legacy/inference/`)
- Send `job_id` / `gpu_device` from UI submit payloads
- Put 15-minute binning inside the GPU frame loop

**Do:**

- Cite file paths when referencing behavior
- Use fixtures when API is ❌ in status matrix
- Add failing tests before fixing engine bugs
- Cross-check `configs/classes.yaml` for class names

---

## 4. Contract change workflow

```
1. Edit packages/contracts/schemas/*.json
2. Edit packages/contracts/typescript/index.ts
3. Edit src/viana/config/job.py (Pydantic)
4. Edit `docs/api_contracts.md` (human summary)
5. Update docs/PROJECT_STATUS.md if endpoint newly available
6. Implement handler / UI consumer
```

---

## 5. Branch / agent parallelism

| Agent | Owned paths | Coordination |
|-------|-------------|--------------|
| Engine | `src/viana/`, `tests/viana/`, `configs/` | Exposes CLI; no HTTP in engine |
| API | `src/orchestrator/` | Spawns CLI; owns job state |
| UI | `apps/web/`, `docs/ui/` | Consumes contracts only |

**Conflict resolution:** `packages/contracts/` wins over any implementation.

---

## 6. Definition of done (per phase)

- Code matches schema
- Tests pass (`pytest tests/viana/` or UI lint/build)
- `PROJECT_STATUS.md` updated
- No edits to `legacy/` except documented path fixes
- ADR added if behavior diverges from plan

---

## 7. Related docs

- `docs/governance/SOURCE_OF_TRUTH.md` — which file owns which fact
- `docs/governance/CONTEXT_MAP.md` — quick lookup index
- `docs/adr/` — decision log
