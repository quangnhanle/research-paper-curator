.PHONY: help start stop restart status logs health setup format lint test test-cov clean \
        prod-start prod-stop prod-restart prod-status prod-logs prod-clean

COMPOSE_LOCAL  = docker compose -f compose.yml
COMPOSE_PROD   = docker compose -f compose.yml -f compose.prod.yml

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local ────────────────────────────────────────────────────────────────────
start: ## [local] Start all services
	$(COMPOSE_LOCAL) up --build -d

stop: ## [local] Stop all services
	$(COMPOSE_LOCAL) down

restart: ## [local] Restart all services
	$(COMPOSE_LOCAL) restart

status: ## [local] Show service status
	$(COMPOSE_LOCAL) ps

logs: ## [local] Tail service logs
	$(COMPOSE_LOCAL) logs -f

clean: ## [local] Stop and remove volumes
	$(COMPOSE_LOCAL) down -v
	docker system prune -f

# ── Production ───────────────────────────────────────────────────────────────
prod-start: ## [prod] Build and start with production overrides
	$(COMPOSE_PROD) up --build -d

prod-stop: ## [prod] Stop production services
	$(COMPOSE_PROD) down

prod-restart: ## [prod] Restart production services
	$(COMPOSE_PROD) restart

prod-status: ## [prod] Show production service status
	$(COMPOSE_PROD) ps

prod-logs: ## [prod] Tail production service logs
	$(COMPOSE_PROD) logs -f

prod-clean: ## [prod] Stop production and remove volumes (DESTRUCTIVE)
	$(COMPOSE_PROD) down -v
	docker system prune -f

# ── Health checks ─────────────────────────────────────────────────────────────
health: ## Check all services health
	@echo "Checking service health..."
	@curl -s http://localhost:8000/api/v1/health | jq . || echo "API not responding"
	@curl -s http://localhost:9200/_cluster/health | jq . || echo "OpenSearch not responding"
	@curl -s http://localhost:8080/api/v2/monitor/health || echo "Airflow not responding"

# ── Development ───────────────────────────────────────────────────────────────
setup: ## Install Python dependencies
	uv sync

format: ## Format code
	uv run ruff format

lint: ## Lint and type check
	uv run ruff check --fix
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html