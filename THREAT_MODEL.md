# Threat model — ViAna (v0.1)

Offline traffic video analytics with a host Next.js UI, containerized FastAPI orchestrator, and GPU CV engine.

## 1. System context

ViAna processes **local video files** on operator-controlled hardware. The UI manages Docker lifecycle on the host and calls the orchestrator on `localhost:8000`. There is no multi-tenant cloud deployment in v0.1.

**Assumptions:** trusted operators on the host; videos may contain PII (plates, faces); container has GPU and filesystem access to mounted `/data`.

## 2. Assets

| Asset | Description | Sensitivity |
|-------|-------------|-------------|
| Source videos | Traffic camera footage on disk | High (PII) |
| CSV outputs | Counts, timestamps, locations | Medium |
| Model weights | `models/v1/*.pt` | Medium (IP) |
| Job configs | Calibration lines, metadata | Low |
| Container socket (host UI) | Docker API from Next.js routes | High |

## 3. Entry points & trust boundaries

| Entry point | Trust boundary | Reachable assets |
|-------------|----------------|------------------|
| Host UI (`apps/web`) | Local operator | Docker socket, API URL |
| FastAPI `:8000` | Local network / localhost | Job queue, `/data` paths |
| `POST /jobs` body | UI → orchestrator | Spawns engine on video paths |
| WebSocket `/ws/jobs` | UI → orchestrator | Telemetry stream |
| `python -m viana` CLI | Container shell | Full `/data`, GPU |

## 4. Threats

| ID | Threat | Actor | Impact | Status |
|----|--------|-------|--------|--------|
| T1 | Path traversal via crafted `source_video_path` | malicious UI / user | Read/write arbitrary files | open |
| T2 | Docker socket abuse from compromised UI | local attacker | Host takeover | open |
| T3 | Resource exhaustion (GPU / disk) | operator mistake | DoS | partially_mitigated (2-GPU cap planned) |
| T4 | Sensitive data in logs | misconfiguration | PII leakage | partially_mitigated (structured logging policy) |
| T5 | Vulnerable dependencies | supply chain | RCE / data breach | partially_mitigated (Dependabot, CodeQL) |

## 5. Deprioritized

| Threat | Reason |
|--------|--------|
| Remote unauthenticated API exploitation | v0.1 binds API to localhost / trusted LAN only |
| Multi-tenant data isolation | Single-operator offline deployment |

## 6. Open questions

- Will the orchestrator validate video paths stay under `/data`?
- Is TLS required when API is exposed beyond localhost?

## 7. Provenance

- mode: bootstrap
- date: 2026-08-18
- tool: agentready remediation

## 8. Recommended mitigations

| Mitigation | Threat IDs | Effort |
|------------|------------|--------|
| Path allowlist under `/data` for job submit | T1 | S |
| Restrict Docker socket permissions; document host hardening | T2 | M |
| Enforce max queue depth and disk quotas | T3 | M |
| Redact PII fields in structlog processors | T4 | S |
| Keep `requirements.txt` + weekly Dependabot | T5 | S |

See also [`SECURITY.md`](SECURITY.md).
