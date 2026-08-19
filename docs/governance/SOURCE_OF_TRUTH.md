# Source of Truth Map

When two documents disagree, **this table wins**.

| Topic | Authoritative file | Notes |
|-------|-------------------|--------|
| **Current phase / what exists** | `docs/PROJECT_STATUS.md` | Update on every milestone |
| **Implementation plan** | `docs/PROJECT_PLAN.md` | |
| **API request/response shape** | `packages/contracts/schemas/*.json` | |
| **TypeScript types** | `packages/contracts/typescript/index.ts` | Must match schemas |
| **Python job models** | `src/viana/config/job.py` | Must match schemas |
| **Engine CLI JobConfig** | `packages/contracts/schemas/job_config.schema.json` | Includes backend `job_id` / `gpu_device` / `output_dir` |
| **Time map artifact** | `packages/contracts/schemas/time_map.schema.json` | `{stem}.time_map.json` |
| **Human API summary** | `docs/api_contracts.md` | Must match schemas |
| **OpenAPI spec** | `openapi.yaml` | Must match `api_contracts.md` |
| **Threat model** | `THREAT_MODEL.md` | Security scope |
| **Class names & aggregation flags** | `configs/classes.yaml` | Inference time |
| **UVH training label map** | `training/uvh/taxonomy/vehicle_taxonomy.json` | Retrain only |
| **Engine default thresholds** | `configs/engine_defaults.yaml` | Overridable per job |
| **System topology** | `docs/ARCHITECTURE.md` | |
| **UI flows & canvas** | `docs/ui/*.md` | |
| **UI mock data** | `packages/contracts/fixtures/*.json` | Until API ✅ |
| **Parity record** | `tests/viana/fixtures/PARITY_NOTES.md` | Phase 9 signed off |
| **Environment setup** | `docs/ops/ENVIRONMENT_SETUP.md` | From-scratch Docker/GPU |
| **Historical research** | `docs/archive/ITVA_RESEARCH_LOG.md` | **Not** current status |
| **Architecture decisions** | `docs/adr/*.md` | |
| **Production model weights** | `models/v1/itva_medium_1088p.pt` | |
| **Pedestrian weights** | `models/pretrained/yolo11l.pt` | |
| **Model directory guide** | `models/README.md` | UVH `public/` vs production `v1/` |
| **UVH retrain workflow** | `training/README.md` | Optional |

## Stale documents (read for context only)

- `README.md` § Installation — see `docs/ops/ENVIRONMENT_SETUP.md` for full stack
- `docs/DEPLOYMENT.md` — quick commands; setup detail in `docs/ops/`
