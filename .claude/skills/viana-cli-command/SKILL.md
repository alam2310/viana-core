---
name: viana-cli-command
description: Add or extend a viana CLI command. Use when working on src/viana/cli.py or engine stages.
---

# CLI command

Follow the pattern in `src/viana/cli.py` (`prescan` stub).

1. Add Typer command with typed options and `-> None` return
2. Keep CV logic out of orchestrator — implement in `src/viana/stages/`
3. Document artifact paths via `src/viana/io/paths.py`
4. Update CLI matrix in `docs/PROJECT_STATUS.md`
