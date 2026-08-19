# Step 5 E2E results

Date: 2026-08-19
Runner: agent (local API at `http://localhost:8000`)

## 5.1 Happy path

- Intake: `POST /jobs/intake` with container path `/data/raw/hiv000001_inframe.mp4`
- Job: `job_4ee8ec2797ea`
- Prescan: reached `AWAITING_REVIEW`
- Confirm: `PATCH /jobs/{id}/prescan` returned `200`
- Processing: reached `COMPLETED`
- Aggregate: `POST /jobs/{id}/aggregate` returned `200` with `rows=28`

Result: happy path passes via API lifecycle.

## 5.2 Negative path

- Intake with host path `/home/mushaffa/Work/ViAna/data/raw/hiv000001_inframe.mp4`
- Job: `job_08062554861e`
- Prescan: `PRESCAN_FAILED` with `Video not found`

Result: expected container path mismatch failure reproduced (tracked as deferred `S09` -> Step 6.7).

## 5.3 Regression notes

- `_15min.csv` physical file path is reported under container output root (`/data/viana-outputs/...`) and is not directly readable from the host shell in this session.
- Aggregate command success (`rows=28`) confirms the file was generated in-container.

## Follow-up

- After Step 6.7 host/container path access hardening, add direct file-read verification from host for `_15min.csv` header and row samples.
