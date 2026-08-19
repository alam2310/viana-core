# Phase 0 Sign-Off

**Closed:** 2026-08-18  
**Next phase:** Phase 1 — Contracts & config (`docs/PROJECT_PLAN.md`)

---

## Exit criteria (all met)

- [x] Monorepo layout: `src/viana`, `src/orchestrator`, `apps/web`, `packages/contracts`, `docs/`, `training/`
- [x] Engine CLI stubs (`prescan`, `run`, `resume`, `aggregate`)
- [x] FastAPI `GET /health` stub
- [x] JSON schemas + TypeScript types + UI mock fixtures
- [x] Governance: `AGENTS.md`, `docs/governance/*`, `.cursor/rules/*`
- [x] Legacy quarantined then removed Phase 9 (parity in `tests/viana/fixtures/PARITY_NOTES.md`)
- [x] UI agent context (`apps/web/AGENTS.md`, `docs/ui/*`)
- [x] Configs: `configs/classes.yaml`, `configs/engine_defaults.yaml`
- [x] Phase 0 hygiene pass (this sign-off)

---

## Delivered in hygiene pass (2026-08-18)

| Item | Location |
|------|----------|
| `checkpoint.schema.json` | `packages/contracts/schemas/` |
| `job_status.schema.json` | `packages/contracts/schemas/` |
| `run_result.schema.json` | `packages/contracts/schemas/` |
| `job_submit_response.json` fixture | `packages/contracts/fixtures/` |
| `checkpoint_resume.json` fixture | `packages/contracts/fixtures/` |
| Dockerfile `pip install -e ".[dev]"` | `Dockerfile` |
| Models documentation | `models/README.md` |
| tmux ops doc relocated | `docs/ops/TMUX_README.md` |
| UI package stub | `apps/web/package.json`, `tsconfig.json` |
| UVH training toolkit | `training/README.md` |

---

## Known deferred (Phase 1+)

| Item | Target phase |
|------|--------------|
| Full Pydantic ↔ schema validation | Phase 1 |
| `checkpoint.py` read/write implementation | Phase 2 |
| OpenAPI export from FastAPI | Phase 6 |
| Next.js full scaffold (`next`, `tailwind`, pages) | Phase 7 |
| CI workflow (`.github/workflows/`) | Phase 1 or 9 |
| Delete `legacy/` | ✅ Phase 9 (2026-08-19) — see `tests/viana/fixtures/PARITY_NOTES.md` |

---

## Local cleanup (developer machines)

If present at repo root (gitignored training/debug artifacts):

```bash
./scripts/cleanup-local-artifacts.sh
```

Files owned by Docker (`nobody`) may require `sudo` — the script retries with `sudo` automatically.

---

## Verification

```bash
pip install -e ".[dev]"
pytest tests/viana/
python -m viana --help
make api-dev   # GET http://localhost:8000/health
```

After Dockerfile change, rebuild the image:

```bash
docker compose build
```
