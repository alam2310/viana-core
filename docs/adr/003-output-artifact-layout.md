---
status: Accepted
applies_to: viana-core, orchestrator
---

# ADR 003: Output Artifact Layout & Retention (S29 / F024)

## Status

Accepted (2026-08-22)

## Context

After a successful `COMPLETED` run, project `output_dir` mixed operator deliverables with JSON sidecars, job configs, checkpoints, and prescan JPEGs. Operators opening the folder could not tell reports from internals. Step **6.2** still needs incomplete checkpoints for explicit PAUSED resume — mass-delete without a layout is unsafe.

## Inventory (intake → prescan → run → aggregate)

| Path (pre-S29 flat) | Writer | Stage | Class |
|---------------------|--------|-------|-------|
| `{stem}_events.csv` | engine `EventsWriter` | run | **operator deliverable** |
| `{stem}_15min.csv` | engine `aggregate` (auto or CLI) | aggregate | **operator deliverable** |
| `{stem}_processed.mp4` | engine `FfmpegRenderer` (if `render_video`) | run | **operator deliverable** |
| `{stem}.checkpoint.json` | engine process loop | run | **resume-required** while incomplete (PAUSED/FAILED/PROCESSING); keep after COMPLETED for complete-run detection + aggregate guard |
| `{stem}.time_map.json` | engine on success | run | **ephemeral / audit** (events already carry wall times; not needed to resume) |
| `{stem}.run_result.json` | engine terminal | run | **orchestrator-only** (stdout also carries RunResult) |
| `{stem}.manifest.json` | (reserved; not written today) | — | **ephemeral / audit** |
| `{job_id}.job.json` | orchestrator `_write_job_config` | spawn | **orchestrator-only** |
| `prescan/{prescan_id}_preview.jpg` | engine prescan | prescan | **ephemeral** (review UI; S01 disk fallback while `AWAITING_REVIEW`) |
| `profiles/{profile_id}.json` | API / engine profiles | any | **operator deliverable** (project-shared calibration) |

Source video under intake mounts is **not** an output artifact.

## Decision

1. **Keep layout** (not delete-on-COMPLETED for sidecars): non-deliverables live under `{project}/_meta/`.
2. **Deliverables stay flat** at `{project}/` so UI “Open output” / job-details paths remain stable.
3. **Never delete incomplete checkpoints** — required for Step 6.2 explicit resume.
4. **On COMPLETED only:** delete that job’s **prescan preview JPEG** (true ephemeral). Do not delete checkpoints, CSVs, MP4, time_map, run_result, or profiles.
5. **Legacy flat sidecars** (`{stem}.checkpoint.json`, etc.) remain readable so existing PAUSED jobs resume without migration.

### Canonical tree

```
{parent_dir}/{project_id}/
  {stem}_events.csv
  {stem}_15min.csv
  {stem}_processed.mp4          # optional
  profiles/{profile_id}.json
  prescan/{prescan_id}_preview.jpg   # ephemeral; removed on COMPLETED
  _meta/{stem}/
    checkpoint.json
    time_map.json
    run_result.json
    manifest.json               # reserved
  _meta/jobs/
    {job_id}.job.json
```

Rejected alternatives for this ADR: delete all sidecars on COMPLETED; `{stem}/.work/` nested under a deliverable-looking folder (operators confuse with reports).

## Consequences

- `viana.io.paths.artifact_paths` points meta keys under `_meta/{stem}/`.
- Readers resolve legacy flat paths when the canonical file is missing.
- `start_fresh` wipes both canonical and legacy run sidecars + deliverables for that stem.
- Docs: `docs/ui/OUTPUT_PATHS.md`. No S32 CSV column changes.

## Out of scope

- CSV schema trim (S32 / F027)
- Pause/resume UX (Step 6.2)
- Migrating historical COMPLETED trees on disk (compat read is enough)
