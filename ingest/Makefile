.PHONY: install lint format test run-current run-historic help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies via Poetry
	poetry install

lint: ## Check formatting (black) and linting (ruff)
	poetry run black --check .
	poetry run ruff check .

format: ## Auto-format code (black + ruff --fix)
	poetry run black .
	poetry run ruff check --fix .

test: ## Run the test suite
	poetry run pytest

run-current: ## Ingest current forecast data for all cities
	poetry run python -m ingest.main --mode current

run-historic: ## Ingest historic archive data for all cities
	poetry run python -m ingest.main --mode historic
