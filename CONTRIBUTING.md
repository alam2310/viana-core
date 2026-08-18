# Contributing to ViAna

## Before you start

1. Read [`AGENTS.md`](AGENTS.md) and [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)
2. Read task-specific `AGENTS.md` under `src/viana/`, `src/orchestrator/`, or `apps/web/`
3. **Schema first** — update `packages/contracts/schemas/` before implementation

## Development workflow

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg

make test
make lint
make typecheck
```

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(engine): add events CSV writer
fix(api): return 409 when checkpoint exists
docs: update PROJECT_STATUS API matrix
```

Enforced via `commitizen` pre-commit hook (`.pre-commit-config.yaml`).

## Pull requests

Use the PR template. Update `docs/PROJECT_STATUS.md` when completing milestones.

## Design documentation

When changing module boundaries or data flows, update the relevant file in `docs/design/`.

## Do not

- Extend `legacy/` (parity reference only)
- Invent API fields outside `packages/contracts/schemas/`
- Put 15-minute aggregation inside the GPU frame loop
