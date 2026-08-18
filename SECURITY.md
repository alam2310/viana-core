# Security

ViAna v0.1 is an **offline, operator-controlled** system. Report security concerns privately to the repository maintainer.

## Scope

- Host Next.js UI and Docker lifecycle
- FastAPI orchestrator (`src/orchestrator/`)
- CV engine CLI (`src/viana/`)
- Artifacts under configured output directories

## Threat model

Structured analysis: **[`THREAT_MODEL.md`](THREAT_MODEL.md)**

## Secure development

- Dependency updates: Dependabot (`.github/dependabot.yml`)
- SAST: CodeQL (`.github/workflows/ci.yml`)
- Pre-commit: ruff, bandit, commit message checks (`.pre-commit-config.yaml`)
- Do not commit secrets, `.env`, or model weights not intended for distribution

## Reporting

If you discover a vulnerability, contact the maintainer directly rather than opening a public issue for sensitive details.
