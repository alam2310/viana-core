# Engine Agent — src/viana

**Read first:** `/AGENTS.md` → `docs/PROJECT_STATUS.md` → `docs/PROJECT_PLAN.md`

## Owned paths

- `src/viana/` — all engine code
- `tests/viana/` — engine tests
- `configs/classes.yaml`, `configs/engine_defaults.yaml` — with care (update docs if changing)

## Reference (read-only)

- `legacy/inference/inference_engine.py` — parity behavior
- `legacy/PARITY.md` — comparison procedure

## Rules

See `.cursor/rules/viana.mdc`

## CLI entry

```bash
python -m viana prescan|run|resume|aggregate
```

## Do not

- Import FastAPI or HTTP handlers here
- Modify `legacy/` except path fixes
- Inline 15-min aggregation in the frame loop

## Update on completion

Mark phases in `docs/PROJECT_STATUS.md` CLI matrix.
