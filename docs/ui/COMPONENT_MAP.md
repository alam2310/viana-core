# Component Map (planned)

| Feature module | Path (planned) | Responsibility |
|----------------|----------------|----------------|
| Container | `features/container/` | docker health, start |
| Projects | `features/projects/` | project_id, output parent display |
| Queue | `features/queue/` | job list, localStorage cache of project_id |
| Prescan | `features/prescan/` | modal, scrubber, OCR review |
| Calibration | `features/calibration/` | HTML5 canvas, line drag, profiles |
| Telemetry | `features/telemetry/` | WS hook, detail toggle |
| Dashboard | `app/page.tsx` | layout, active job viewport |

## Shadcn components (from `docs/specs/ui_specifications.md`)

- Modal: prescan review
- Toast: errors
- Dropdown: task type (ViAna only for v0.1)
- Progress: job bars
- Table: telemetry (virtualized list for performance)
