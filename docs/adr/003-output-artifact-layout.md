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

| Path | Writer | Stage | Class |
|------|--------|-------|-------|
| *(source video)* | intake mount | intake | **not an output artifact** |
| `prescan/{prescan_id}_preview.jpg` | engine `run_prescan` | prescan | **ephemeral** — review UI + S01 disk fallback while `AWAITING_REVIEW`; **deleted on COMPLETED** |
| `{job_id}.job.json` | orchestrator `_write_job_config` | spawn (run/resume) | **orchestrator-only** — CLI spawn config |
| `{stem}_events.csv` | engine `EventsCsvWriter` | run | **operator deliverable (debug)** — full engine fields (S32) |
| `{stem}_events_report.csv` | engine `write_events_report_csv` | run (on COMPLETED) | **operator deliverable** — trimmed report derived from events |
| `{stem}_processed.mp4` | engine `FfmpegRenderer` (if `render_video`) | run | **operator deliverable** |
| `checkpoint.json` under `_meta/{stem}/` | engine process loop | run | **resume-required** while incomplete (Step 6.2 PAUSED); **kept** after COMPLETED for complete-run detection + aggregate guard |
| `time_map.json` under `_meta/{stem}/` | engine on success | run | **ephemeral / audit** — wall-clock anchors; not needed to resume |
| `run_result.json` under `_meta/{stem}/` | engine terminal | run | **orchestrator-only** — stdout also carries `RunResult` |
| `manifest.json` under `_meta/{stem}/` | (reserved) | — | **ephemeral / audit** |
| `{stem}_15min.csv` | engine `aggregate` (auto or CLI) | aggregate | **operator deliverable** |
| `profiles/{profile_id}.json` | API / engine profiles | any | **operator deliverable** (project-shared calibration) |

**Pre-S29 flat sidecars** (`{stem}.checkpoint.json`, `{stem}.time_map.json`, `{stem}.run_result.json`, `{job_id}.job.json` at project root) are **read-only compat** — never mass-deleted; new writes use `_meta/` (see below).

### Rejected layouts (documented before any wipe)

| Alternative | Why rejected |
|-------------|--------------|
| Delete all sidecars on COMPLETED | Breaks complete-checkpoint detection (S36), audit, future S34 job persistence |
| `{stem}/.work/` nested under stem | Looks like an operator report folder; breaks flat “Open output” |
| Mass delete historical COMPLETED trees | Unsafe; legacy flat checkpoints must remain readable for PAUSED resume |

## Decision

1. **Keep layout** (not delete-on-COMPLETED for sidecars): non-deliverables live under `{project}/_meta/`.
2. **Deliverables stay flat** at `{project}/` so UI “Open output” / job-details paths remain stable.
3. **Never delete incomplete checkpoints** — required for Step 6.2 explicit resume.
4. **On COMPLETED only:** delete that job’s **prescan preview JPEG** (true ephemeral). Do not delete checkpoints, CSVs, MP4, time_map, run_result, or profiles.
5. **Legacy flat sidecars** (`{stem}.checkpoint.json`, etc.) remain readable so existing PAUSED jobs resume without migration.

### Canonical tree

```
{parent_dir}/{project_id}/
  {stem}_events.csv              # debug (full fields)
  {stem}_events_report.csv       # operator report (S32)
  {stem}_15min.csv
  {stem}_processed.mp4           # optional
  profiles/{profile_id}.json
  prescan/{prescan_id}_preview.jpg   # ephemeral; removed on COMPLETED
  _meta/{stem}/
    checkpoint.json
    time_map.json
    run_result.json
    manifest.json                # reserved
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
