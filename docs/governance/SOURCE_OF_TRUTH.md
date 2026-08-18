# Source of Truth Map

When two documents disagree, **this table wins**.

| Topic | Authoritative file | Notes |
|-------|-------------------|--------|
| **Current phase / what exists** | `docs/PROJECT_STATUS.md` | Update on every milestone |
| **Implementation plan** | `docs/PROJECT_PLAN.md` | |
| **API request/response shape** | `packages/contracts/schemas/*.json` | |
| **TypeScript types** | `packages/contracts/typescript/index.ts` | Must match schemas |
| **Python job models** | `src/viana/config/job.py` | Must match schemas |
| **Human API summary** | `docs/api_contracts.md` | Must match schemas |
| **Class names & aggregation flags** | `configs/classes.yaml` | Inference time |
| **UVH training label map** | `legacy/configs/vehicle_taxonomy.json` | Training / legacy only |
| **Engine default thresholds** | `configs/engine_defaults.yaml` | Overridable per job |
| **System topology** | `docs/ARCHITECTURE.md` | |
| **UI flows & canvas** | `docs/ui/*.md` | |
| **UI mock data** | `packages/contracts/fixtures/*.json` | Until API ✅ |
| **Parity procedure** | `legacy/PARITY.md` | |
| **Legacy behavior reference** | `legacy/inference/inference_engine.py` | Do not extend |
| **Historical research** | `legacy/blueprint.md` | **Not** current status |
| **Architecture decisions** | `docs/adr/*.md` | |
| **Production model weights** | `models/v1/itva_medium_1088p.pt` | |
| **Pedestrian weights** | `models/pretrained/yolo11l.pt` | |

## Stale documents (read for context only)

- `README.md` § Golden Master — Docker setup still valid; directory tree see `AGENTS.md`
- `folderstructure.txt` — moved to `legacy/artifacts/`
- `legacy/docs/ITVA_Environment_Setup_Guide.md` — historical Docker + training setup
- `legacy/docs/VEHICLE_CLASSIFICATION.md` — UVH taxonomy background
