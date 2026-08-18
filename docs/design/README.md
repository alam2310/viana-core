# Design intent documentation

Module-level preconditions, invariants, and rationale for the v2 platform. Update when changing boundaries or data flows (see `AGENTS.md`).

| Module | Document |
|--------|----------|
| CV engine | [`engine.md`](engine.md) |
| Job API | [`orchestrator.md`](orchestrator.md) |
| Contracts | `packages/contracts/README.md` |

## Cross-cutting invariants

1. **Event-sourced analytics** — GPU loop writes crossing events only; 15-min bins are a separate stage (ADR 001).
2. **Backend-owned jobs** — `job_id` and `gpu_device` are never client-supplied (ADR 002).
3. **Schema first** — `packages/contracts/schemas/` wins over implementation.
4. **Mandatory geometry** — counting and horizon lines must be within frame bounds.

## Preconditions (platform)

- Videos are accessible inside the container at absolute paths under `/data`.
- At most two concurrent GPU jobs (`cuda:0`, `cuda:1`).
- Output artifacts land under `{output.parent_dir}/{project_id}/`.
