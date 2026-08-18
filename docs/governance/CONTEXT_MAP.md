# Context Map — Where to find things

Quick index for AI agents. **Start at `AGENTS.md`**, not here.

---

## I want to…

| Goal | Go to |
|------|--------|
| Know what to build next | `docs/PROJECT_STATUS.md` |
| See full phase plan | `docs/PROJECT_PLAN.md` |
| Understand system design | `docs/ARCHITECTURE.md` |
| Build CV engine | `src/viana/AGENTS.md`, `legacy/inference/inference_engine.py` |
| Build FastAPI API | `src/orchestrator/AGENTS.md`, `docs/api_contracts.md` |
| Build Next.js UI | `apps/web/AGENTS.md`, `docs/ui/README.md` |
| Add API field | `packages/contracts/schemas/` first — `docs/governance/CONTRACT_SYNC.md` |
| Run parallel agent chats | `docs/governance/KICKOFF_PROMPTS.md` |
| Mock API in UI | `packages/contracts/fixtures/` |
| Compare old vs new counts | `legacy/PARITY.md` |
| Understand vehicle classes | `configs/classes.yaml`, `legacy/docs/VEHICLE_CLASSIFICATION.md` |
| Run Docker | `docker-compose.yml`, `README.md` |
| AI workflow rules | `docs/governance/AI_SDLC.md` |
| Resolve doc conflict | `docs/governance/SOURCE_OF_TRUTH.md` |

---

## Directory purposes

| Path | Purpose |
|------|---------|
| `src/viana/` | New CV engine |
| `src/orchestrator/` | Job API |
| `apps/web/` | Next.js UI (host) |
| `packages/contracts/` | Shared schemas/types/fixtures |
| `configs/` | Runtime YAML config |
| `models/` | Neural network weights |
| `legacy/` | **Discard later** — old code |
| `tests/viana/` | New engine tests |
| `docs/ui/` | UI specifications |
| `docs/governance/` | AI SDLC rules |
| `docs/adr/` | Architecture decisions |
| `.cursor/rules/` | Cursor agent rules |

---

## UI development (parallel agent)

All UI context lives under:

1. `apps/web/AGENTS.md` — entry point
2. `docs/ui/README.md` — doc index
3. `docs/ui/*.md` — detailed specs
4. `packages/contracts/` — types + mocks
5. `docs/specs/ui_specifications.md` — product requirements

**UI source code** will live in `apps/web/` once Phase 7 scaffolds Next.js. Until then, use fixtures and spec docs — do not assume components exist.
