# Context Map — Where to find things

Quick index for AI agents. **Start at `AGENTS.md`**, not here.

---

## I want to…

| Goal | Go to |
|------|--------|
| Know what to build next | `docs/PROJECT_STATUS.md` |
| See full phase plan | `docs/PROJECT_PLAN.md` |
| Understand system design | `docs/ARCHITECTURE.md` |
| Build CV engine | `src/viana/AGENTS.md` |
| Build FastAPI API | `src/orchestrator/AGENTS.md`, `docs/api_contracts.md` |
| Build Next.js UI | `apps/web/AGENTS.md`, `docs/ui/README.md` |
| Add API field | `packages/contracts/schemas/` first — `docs/governance/CONTRACT_SYNC.md` |
| Run parallel agent chats | `docs/governance/KICKOFF_PROMPTS.md` (Phases) · `docs/steps/KICKOFF_PROMPTS.md` (Steps) |
| Post-v0.1 work (Steps 1–5) | `docs/steps/TRACKER.md`, `docs/steps/AGENT_PROGRESS.md` |
| Mock API in UI | `packages/contracts/fixtures/` |
| Parity / legacy counts | `tests/viana/fixtures/PARITY_NOTES.md` |
| Understand vehicle classes | `configs/classes.yaml`, `training/uvh/taxonomy/TAXONOMY.md` |
| Set up Docker/GPU from scratch | `docs/ops/ENVIRONMENT_SETUP.md` |
| Retrain UVH model | `training/README.md` |
| Run Docker (daily) | `docs/DEPLOYMENT.md`, `docker-compose.yml` |
| AI workflow rules | `docs/governance/AI_SDLC.md` |
| Resolve doc conflict | `docs/governance/SOURCE_OF_TRUTH.md` |

---

## Directory purposes

| Path | Purpose |
|------|---------|
| `src/viana/` | CV engine |
| `src/orchestrator/` | Job API |
| `apps/web/` | Next.js UI (host) |
| `packages/contracts/` | Shared schemas/types/fixtures |
| `configs/` | Runtime YAML config |
| `models/` | Neural network weights |
| `training/` | UVH retrain toolkit (optional) |
| `tests/viana/` | Engine tests |
| `docs/ui/` | UI specifications |
| `docs/ops/` | Environment setup |
| `docs/archive/` | Historical research |
| `docs/governance/` | AI SDLC rules |
| `docs/adr/` | Architecture decisions |
| `.cursor/rules/` | Cursor agent rules |
