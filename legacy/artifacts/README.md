# Legacy artifact snapshots

**Do not use for navigation.** This folder holds discardable training/debug outputs from the pre-v2 era.

| Path | Contents |
|------|----------|
| `debug_pretrain/` | Debug visualization PNGs from pretrain audits |
| `runs/` | Ultralytics validation run outputs |
| `folderstructure.txt` | **Stale** repo tree snapshot (~2026-01, pre-v2 layout) |

Current repository layout: see `/AGENTS.md` and `docs/PROJECT_STATUS.md`.

Safe to delete this entire folder after v2 parity sign-off (Phase 9), unless you need historical audit artifacts.

To remove duplicate root copies (`debug_pretrain/`, `runs/`): `./scripts/cleanup-local-artifacts.sh` from repo root.
