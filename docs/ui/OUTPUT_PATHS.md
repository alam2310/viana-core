# Output Paths

## Configuration

`output.parent_dir` in `configs/engine_defaults.yaml` or `docker/orchestrator_config.yaml.example`.

Default: `/data/viana-outputs`

## Per-project directory

```
/data/viana-outputs/{project_id}/
```

Layout and retention: **ADR 003** (`docs/adr/003-output-artifact-layout.md`).

## Operator deliverables (project root)

`stem` = source filename without extension. These stay flat so “Open output” and job-details links remain stable.

| File | Description |
|------|-------------|
| `{stem}_events.csv` | Raw crossing events (**full debug** — all engine fields) |
| `{stem}_events_report.csv` | Operator event report (derived at COMPLETED) |
| `{stem}_15min.csv` | Clock 15-min aggregation (vehicles + pedestrians, zero-filled) |
| `{stem}_processed.mp4` | Annotated video when `render_video` (colors: `docs/ui/OVERLAY_COLORS.md`) |

## Profiles (project-shared)

`/data/viana-outputs/{project_id}/profiles/{profile_id}.json`

## Work / meta (not operator reports)

```
{project}/_meta/{stem}/
  checkpoint.json    # resume-required while incomplete (6.2 PAUSED); kept after COMPLETED
  time_map.json      # audit / clock anchors
  run_result.json    # engine terminal outcome
  manifest.json      # reserved
{project}/_meta/jobs/
  {job_id}.job.json  # orchestrator spawn config
```

Legacy flat sidecars (`{stem}.checkpoint.json`, etc.) are still **read** if present so existing PAUSED jobs can resume without migration.

## Ephemeral

`{project}/prescan/{prescan_id}_preview.jpg` — review JPEG (S01 disk fallback while `AWAITING_REVIEW`). **Deleted on job COMPLETED**; not needed for resume.

## Retention rules

| Class | On COMPLETED |
|-------|----------------|
| Operator deliverables | keep |
| Incomplete checkpoint | never delete (resume) |
| Complete checkpoint / time_map / run_result / job JSON | keep under `_meta/` |
| Prescan preview JPEG for that job | delete |
| Profiles | keep |

`start_fresh` / overwrite still wipes that stem’s CSVs, MP4, and sidecars (canonical + legacy).

## UI linking

Use `output_dir` from `JobSubmitResponse` / `GET /jobs/{id}` — do not derive paths client-side except for display of the three deliverable filenames.
