# AGENTS.md — AI Agent Onboarding (ViAna Monorepo)

**Read this file first.** Do not rely on chat history. The repository is the source of truth.

## 1. What this project is

Offline Indian traffic video analytics: detect, classify, track, count vehicles, export CSV reports and optional annotated video. A Next.js UI orchestrates a Dockerized FastAPI backend that runs the `viana` CV engine.

## 2. Read order (by task)

| Your task | Read in order |
|-----------|----------------|
| **Any task** | This file → `docs/PROJECT_STATUS.md` → `docs/governance/SOURCE_OF_TRUTH.md` |
| **Engine (`src/viana/`)** | `.cursor/rules/viana.mdc` → `docs/PROJECT_PLAN.md` § Engine → `tests/viana/fixtures/PARITY_NOTES.md` |
| **API (`src/orchestrator/`)** | `docs/api_contracts.md` → `packages/contracts/schemas/` → `docs/PROJECT_PLAN.md` § Orchestrator |
| **UI (`apps/web/`)** | `apps/web/AGENTS.md` → `docs/ui/README.md` → `packages/contracts/typescript/` |
| **Contracts / types** | `packages/contracts/README.md` → update schema **before** code |
| **Post-v0.1 work (Steps 1–5)** | `docs/steps/TRACKER.md` → `AGENT_PROGRESS.md` → `STEP_N_*.md` |

## 3. Repository map

```
src/viana/          → CV engine CLI (ACTIVE — implement here)
src/orchestrator/   → FastAPI job API (ACTIVE — implement here)
apps/web/           → Next.js 15 UI (ACTIVE — implement here)
packages/contracts/ → JSON schemas + TS types (SOURCE OF TRUTH for API/data)
configs/            → classes.yaml, engine_defaults.yaml
models/             → weights (v1 production + pretrained)
training/           → UVH retrain toolkit (optional)
docs/archive/       → pre-v2 research log (historical)
tests/viana/        → new engine tests
docs/               → plans, status, UI specs, governance, Steps, ADRs
```

## 4. Hard rules (all agents)

1. **Never invent API fields** — check `packages/contracts/schemas/` and `docs/api_contracts.md`.
2. **Backend owns jobs** — UI must not send `job_id` or `gpu_device` on submit.
3. **Schemas before code** — if the contract changes, update `packages/contracts/` first; see `docs/governance/CONTRACT_SYNC.md` for parallel UI/API/engine rules.
4. **Update `docs/PROJECT_STATUS.md`** when completing a phase, Step, or API endpoint — follow `docs/steps/AGENT_PROGRESS.md` for Steps 1–5.
5. **Record decisions** in `docs/adr/` when architecture changes.
6. **No inline 15-min aggregation in the GPU loop** — events CSV first, aggregate separately (ADR 001).
7. **Geometry** — lines must be within frame bounds; mandatory on every run.

## 5. What is implemented today

See **`docs/PROJECT_STATUS.md`** for the live matrix. **Phases 0–9 are complete** (parity signed off 2026-08-19).

| Component | Status |
|-----------|--------|
| Phases 0–9 (engine, API, UI, parity) | ✅ Complete |
| **Post-v0.1 Steps 1–5** | ⬜ Active — see `docs/steps/TRACKER.md`; follow `docs/steps/AGENT_PROGRESS.md` on start/complete |

## 6. Parallel development

- **Engine agent:** `src/viana/`, `tests/viana/`, `configs/`
- **UI agent:** `apps/web/`, `docs/ui/`, consume `packages/contracts/`
- **API agent:** `src/orchestrator/`, `docs/api_contracts.md`

Sync via **contracts only**. If the UI needs a new field, add it to the schema first and note it in `PROJECT_STATUS.md`.

**Parallel chats (Phases 0–9):** `docs/governance/PARALLEL_AGENTS.md`, `docs/governance/KICKOFF_PROMPTS.md`  
**Post-v0.1 Steps:** `docs/steps/KICKOFF_PROMPTS.md` — start **new** chats; do not continue Phase build threads.

## 7. Verification commands

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
pytest tests/viana/
python -m viana --help
make api-dev    # :8000/health
make lint && make typecheck
```

### Single-file verification (agents)

Fast feedback without a full build — target one file or test module:

```bash
ruff check src/viana/cli.py
ruff format --check src/viana/cli.py
mypy src/viana/cli.py
bandit -r src/viana/cli.py -c pyproject.toml
pytest tests/viana/test_phase0.py -q
```

TypeScript (after Phase 7 scaffold):

```bash
npx eslint apps/web/src/path/to/file.ts
npx tsc --noEmit -p apps/web
```

## Pattern References

- New API contract field: follow the pattern in `packages/contracts/schemas/job_submit.schema.json`, then sync `packages/contracts/typescript/index.ts` and `src/viana/config/job.py`
- New FastAPI route: follow the pattern in `src/orchestrator/routes/health.py`
- Structured logging: see `src/orchestrator/logging_config.py` for reference implementation
- Job validation: follow the pattern in `src/viana/config/job.py`
- Artifact paths: follow the pattern in `src/viana/io/paths.py`
- CLI command: follow the pattern in `src/viana/cli.py`

## 8. Historical docs

- `docs/archive/ITVA_RESEARCH_LOG.md` — pre-v2 research and environment pivots.
- `docs/ops/ENVIRONMENT_SETUP.md` — from-scratch Docker/GPU setup.
