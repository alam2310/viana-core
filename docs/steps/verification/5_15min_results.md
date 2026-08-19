# Step 5 E2E results

Date: 2026-08-19  
Runner: QA agent (local API at `http://localhost:8000`)

## 5.1 Happy path (intake -> prescan -> confirm -> READY -> PROCESSING -> COMPLETED -> aggregate)

### Run A (full lifecycle + aggregate)

- Endpoint shape used (current contract): `POST /jobs/intake` with `source_video_paths` array.
- Intake payload: `{"source_video_paths":["/data/raw/hiv000001_inframe.mp4"],"project_id":"step5-e2e-qa"}`
- Intake response: `201 Created` with `job_id=job_3a11b45c4756`
- Observed status sequence from polling:
  - `AWAITING_REVIEW`
  - `PROCESSING`
  - `COMPLETED`
- Confirm call: `PATCH /jobs/job_3a11b45c4756/prescan` -> `200 OK`
- Aggregate call: `POST /jobs/job_3a11b45c4756/aggregate` -> `200 OK`
  - `rows=28`
  - `aggregate_15min=/data/viana-outputs/step5-e2e-qa/hiv000001_inframe_15min.csv`

### Run B (explicit READY proof)

- Intake payload: `{"source_video_paths":["/data/raw/hiv000001_inframe.mp4"],"project_id":"step5-e2e-ready-proof"}`
- Intake response: `201 Created` with `job_id=job_bd503984f992`
- Confirm call: `PATCH /jobs/job_bd503984f992/prescan` -> `200 OK`
- Confirm response body includes `status: "READY"` (explicit gate evidence before worker processing).
- Subsequent observed states: `PROCESSING` then terminal `COMPLETED`.

Result: happy-path lifecycle passes, including READY gate and successful aggregate output.

## 5.1 `_15min.csv` outcome verification

CSV verified in-container at:

- `/data/viana-outputs/step5-e2e-qa/hiv000001_inframe_15min.csv`

Evidence from container read:

- Row count including header: `29` (1 header + 28 data rows)
- Header:
  - `window_start,window_end,date,location,class_name,category,class_type,sub_class,direction,count,partial`
- First data row:
  - `02:15,02:30,18-10-2024,LITO-RARARANKI,Car,Passenger,Light Fast,Car,in,12,false`
- Last data row:
  - `02:15,02:30,18-10-2024,LITO-RARARANKI,Taxi,Passenger,Light Fast,Taxi,out,0,false`

Result: `_15min.csv` exists, is non-empty, and contains expected aggregation rows.

## 5.2 Negative path observations

### Path mapping constraint repro (host path submitted to containerized API)

- Intake payload:
  - `{"source_video_paths":["/home/mushaffa/Work/ViAna/data/raw/hiv000001_inframe.mp4"],"project_id":"step5-e2e-qa-negative"}`
- Intake response: `201 Created` with `job_id=job_5d35e0f5bddf`
- Terminal status: `PRESCAN_FAILED`
- Error message:
  - `Video not found: /home/mushaffa/Work/ViAna/data/raw/hiv000001_inframe.mp4`

Interpretation:

- This is the expected host-vs-container filesystem mapping failure.
- Still aligns with deferred stabilization item `S09` -> Step 6.7 (path validation/hardening lane).

## 5.3 Regression notes

- Contract drift observed vs older Step 5 notes:
  - `POST /jobs/intake` now expects `source_video_paths` (array).
  - `source_video_path` and `frame_offset_sec` are rejected as extra/missing fields (422).
- Confirm contract drift observed:
  - `PATCH /jobs/{id}/prescan` rejects `start_fresh` as extra field (422).
- These are documentation/test payload shape updates, not processing regressions.

## Follow-up

- Keep Step 6.7 open for intake path validation + host/container mapping UX.
- Optional additional QA pass after 6.7: verify same negative-path request is rejected earlier with clearer operator guidance.
