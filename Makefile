.PHONY: help install test test-training engine-cli api-dev ui-dev lint typecheck format boundaries check-status-sync

help:
	@echo "ViAna monorepo targets:"
	@echo "  make install      - pip install -e '.[dev]' (inside container)"
	@echo "  make test         - run engine tests (tests/viana/)"
	@echo "  make test-training - run UVH taxonomy tests (training/uvh/tests/)"
	@echo "  make lint         - ruff + bandit"
	@echo "  make typecheck    - mypy on viana + orchestrator"
	@echo "  make format       - ruff format"
	@echo "  make boundaries   - import-linter contracts"
	@echo "  make check-status-sync - verify TRACKER/backlog/PROJECT_STATUS alignment"
	@echo "  make engine-cli   - show viana CLI help"
	@echo "  make api-dev      - run FastAPI orchestrator (dev)"
	@echo "  make ui-dev       - run Next.js UI (requires npm in apps/web)"

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-training:
	pytest training/uvh/tests/

lint:
	ruff check src tests
	bandit -r src/viana src/orchestrator -c pyproject.toml

typecheck:
	mypy src/viana src/orchestrator

format:
	ruff format src tests

boundaries:
	lint-imports

check-status-sync:
	python3 scripts/check_status_sync.py

engine-cli:
	python -m viana --help

api-dev:
	uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000 --reload --app-dir src

ui-dev:
	cd apps/web && npm run dev
