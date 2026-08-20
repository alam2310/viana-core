## 2026-08-20 - Next.js Security Headers
**Vulnerability:** Missing security headers on Next.js frontend
**Learning:** Next.js config didn't have X-Frame-Options, HSTS, or CSP configured, exposing the UI to clickjacking and MIME sniffing risks.
**Prevention:** Always implement `async headers()` in `next.config.ts` to enforce baseline frontend protections (CSP, X-Frame-Options, X-Content-Type-Options).
