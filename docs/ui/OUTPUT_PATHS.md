# Output Paths

## Configuration

`output.parent_dir` in `configs/engine_defaults.yaml` or `docker/orchestrator_config.yaml.example`.

Default: `/data/viana-outputs`

## Per-project directory

```
/data/viana-outputs/{project_id}/
```

## Per-video artifacts (`stem` = filename without extension)

| File | Description |
|------|-------------|
| `{stem}_events.csv` | Raw crossing events |
| `{stem}_15min.csv` | Clock 15-min aggregation (vehicle classes, zero-filled) |
| `{stem}_processed.mp4` | Annotated video |
| `{stem}.manifest.json` | Job snapshot, `partial` flag |
| `{stem}.time_map.json` | OCR time anchors |
| `{stem}.checkpoint.json` | Resume state (PAUSED/FAILED only) |
| `{stem}.run_result.json` | Final status + paths |

## Profiles

`/data/viana-outputs/{project_id}/profiles/{profile_id}.json`

## Prescan previews

`/data/viana-outputs/{project_id}/prescan/{prescan_id}_preview.jpg`

## UI linking

Use `output_dir` from `JobSubmitResponse` / `GET /jobs/{id}` — do not derive paths client-side except for display.
