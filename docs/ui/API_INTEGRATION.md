# API Integration (UI ↔ Container)

## Split responsibility

| Layer | Runs on | Calls |
|-------|---------|-------|
| Container manager | Host (Next.js `/api/container/*`) | `docker ps`, `docker start` |
| Analytics API | Container port 8000 | FastAPI routes |

## Base URL

`NEXT_PUBLIC_API_URL` default: `http://localhost:8000`

## Key calls

| When | Method | Endpoint |
|------|--------|----------|
| Prescan button | POST | `/utils/prescan` |
| Preview image | GET | `preview_url` from prescan response |
| Submit job | POST | `/jobs` |
| Refresh queue | GET | `/jobs?project_id=` |
| Paused job detail | GET | `/jobs/{id}` |
| Resume | POST | `/jobs/{id}/resume` |
| Start fresh | POST | `/jobs/{id}/start-fresh` |
| Live updates | WS | `/ws/jobs` |

## Error handling

| HTTP | Meaning | UI action |
|------|---------|-----------|
| 400 | Invalid geometry / config | Toast + highlight field |
| 409 | Checkpoint exists without resume/fresh | Show resume modal |
| 503 | GPU workers busy | Show queue position |
| 500 | Engine crash | Show PAUSED if checkpoint exists |

## Do not send

- `job_id` on submit
- `gpu_device` on submit

Backend assigns both.
