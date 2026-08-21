## 2026-08-21 - Rejected Next.js Security Headers CSP
**Vulnerability:** Missing security headers on Next.js frontend
**Learning:** A strict default-src-only CSP breaks this application because the UI talks to the orchestrator on another origin/port (HTTP + WebSocket) and plays/serves artifact MP4s. HSTS on local HTTP is also unsafe for our Docker/dev setup.
**Prevention:** Do not add a strict default-src-only CSP. Any future CSP must explicitly allow the API, WS, and media/blob URLs.
## 2026-08-21 - PII Redaction in Structured Logs
**Vulnerability:** Sensitive PII metadata logged to standard output
**Learning:** By default, structlog logs all key-value arguments, including potentially sensitive JobMetadata fields like locations and dates, which causes PII leaks in monitoring infrastructure.
**Prevention:** Always implement and include a recursive redaction processor in the structlog pipeline to scrub sensitive keys before formatting.
