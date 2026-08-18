---
name: viana-contract-change
description: Change shared API or CSV contracts. Use when adding fields to job submit, telemetry, or artifact schemas.
---

# Contract change

Follow the pattern in `packages/contracts/schemas/job_submit.schema.json`.

1. Edit JSON schema in `packages/contracts/schemas/`
2. Sync `packages/contracts/typescript/index.ts`
3. Sync `src/viana/config/job.py` (Pydantic)
4. Update `docs/api_contracts.md` and fixtures under `packages/contracts/fixtures/`
