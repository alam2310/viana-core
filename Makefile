.PHONY: help install test test-legacy engine-cli api-dev ui-dev

help:
	@echo "ViAna monorepo targets:"
	@echo "  make install      - pip install -e '.[dev]' (inside container)"
	@echo "  make test         - run new engine tests (tests/viana/)"
	@echo "  make test-legacy  - run legacy taxonomy tests"
	@echo "  make engine-cli   - show viana CLI help"
	@echo "  make api-dev      - run FastAPI orchestrator (dev)"
	@echo "  make ui-dev       - run Next.js UI (requires npm in apps/web)"

install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-legacy:
	pytest legacy/tests/

engine-cli:
	python -m viana --help

api-dev:
	uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000 --reload --app-dir src

ui-dev:
	cd apps/web && npm run dev
