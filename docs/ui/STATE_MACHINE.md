# Job State Machine

```mermaid
stateDiagram-v2
    [*] --> PENDING: POST /jobs
    PENDING --> PROCESSING: worker picks job
    PROCESSING --> COMPLETED: success
    PROCESSING --> PAUSED: crash or cancel mid-run
    PROCESSING --> FAILED: unrecoverable error
    PAUSED --> PROCESSING: POST resume
    PAUSED --> PROCESSING: POST start-fresh
    COMPLETED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

## UI actions by state

| State | UI shows | Actions |
|-------|----------|---------|
| PENDING | Queue position | Cancel |
| PROCESSING | Progress bar, optional telemetry | Cancel |
| PAUSED | **Flash / highlight** video | Resume, Start fresh |
| COMPLETED | Output file links | Re-aggregate |
| FAILED | Error message | Start fresh (if checkpoint) |
| CANCELLED | — | Remove from queue view |

## Checkpoint

- File: `{output_dir}/{stem}.checkpoint.json`
- `checkpoint_exists: true` only for PAUSED/FAILED with checkpoint on disk
- Engine never auto-resumes on plain `POST /jobs` if checkpoint exists → API returns **409**
