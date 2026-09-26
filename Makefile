UV ?= uv
VENV ?= .venv
SERVICE_IMAGE ?= ubuntu:24.04
SERVICE_HOST_PORT ?= 8000
SERVICE_CONTAINER_PORT ?= 8000

.PHONY: help install install-browser test test-integration run serve serve-container stop-container

help:
	@echo "Dependency order:"
	@echo "  install"
	@echo "    └── install-browser"
	@echo ""
	@echo "  test               requires install"
	@echo "  test-integration   requires install-browser"
	@echo "  run                requires install-browser"
	@echo "  serve              requires install-browser"
	@echo "  serve-container     Run service in Ubuntu container with code volume mount"
	@echo "  stop-container      Stop and remove the service container"
	@echo ""
	@echo "Targets:"
	@echo "  make install            Create .venv and install aqe"
	@echo "  make install-browser    Download Playwright Chromium"
	@echo "  make test               Run the unit tests"
	@echo "  make test-integration   Run the Playwright test"
	@echo "  make run                Run SPEC_PATH from .env"
	@echo "  make serve              Serve using HOST and PORT from .env"
	@echo "  make serve-container     Serve from Ubuntu container with code volume mount"
	@echo "  make stop-container      Stop and remove the service container"

install: $(VENV)/.install

$(VENV)/.install: pyproject.toml
	$(UV) venv $(VENV)
	$(UV) pip install -e .
	@touch $@

install-browser: $(VENV)/.install-browser

$(VENV)/.install-browser: $(VENV)/.install
	$(UV) run playwright install chromium
	$(UV) run playwright install-deps
	@touch $@

test: $(VENV)/.install
	$(UV) run pytest

test-integration: $(VENV)/.install-browser
	$(UV) run pytest -m integration

run: $(VENV)/.install-browser
	$(UV) run aqe run

serve: $(VENV)/.install-browser
	$(UV) run aqe serve

serve-container:
	# Stop and remove existing container if it exists
	@docker stop aqe-service 2>/dev/null || true
	@docker rm aqe-service 2>/dev/null || true
	# Start service in Ubuntu container as root user for full privileges
	# This allows automatic tool installation (curl, wget, jq, Python, etc.) when needed
	# Using --network host to allow container to access host services via localhost
	docker run -d --name aqe-service \
		--network host \
		-v $(PWD):/app \
		-v $(PWD)/runs:/app/runs \
		-e HOST=0.0.0.0 \
		-e PORT=$(SERVICE_CONTAINER_PORT) \
		-e DEBIAN_FRONTEND=noninteractive \
		-e PYTHONUNBUFFERED=1 \
		$(SERVICE_IMAGE) \
		bash -c "apt-get update && \
			apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl ca-certificates openssl git && \
			pip3 install --break-system-packages uv && \
			cd /app && \
			uv venv --clear && \
			. .venv/bin/activate && \
			uv pip install -e . && \
			uv run aqe serve"
	@echo "Service running on http://localhost:$(SERVICE_HOST_PORT)"
	@echo "Container runs as root user for full privileges"
	@echo "Container uses host networking (can access host services via localhost)"
	@echo "Playwright Chromium will be installed on-demand when browser is needed"
	@echo "To stop: make stop-container"
	@echo "To view logs: docker logs aqe-service -f"

stop-container:
	@echo "Stopping and removing aqe-service container..."
	@docker stop aqe-service 2>/dev/null || echo "Container not running"
	@docker rm aqe-service 2>/dev/null || echo "Container already removed"
	@echo "Container stopped and removed successfully"
