UV ?= uv
VENV ?= .venv
SERVICE_IMAGE ?= ubuntu:24.04
SERVICE_HOST_PORT ?= 8000
SERVICE_CONTAINER_PORT ?= 8000

# System Chrome configuration
USE_SYSTEM_CHROME ?= false
CHROME_EXECUTABLE_PATH ?=

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
	@echo "System Chrome Configuration (alternative to downloading Playwright Chrome):"
	@echo "  USE_SYSTEM_CHROME         Use system Chrome instead of downloading (default: false)"
	@echo "  CHROME_EXECUTABLE_PATH   Path to Chrome executable (auto-detected if not set)"
	@echo ""
	@echo "Examples:"
	@echo "  make install-browser USE_SYSTEM_CHROME=true"
	@echo "  make serve-container USE_SYSTEM_CHROME=true"
	@echo "  USE_SYSTEM_CHROME=true CHROME_EXECUTABLE_PATH=/usr/bin/google-chrome make serve-container"
	@echo ""
	@echo "Targets:"
	@echo "  make install            Create .venv and install aqe"
	@echo "  make install-browser    Download Playwright Chromium or use system Chrome"
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
	@if [ -f .env ]; then \
		. ./.env && \
		if [ "$$USE_SYSTEM_CHROME" = "true" ]; then \
			echo "Using system Chrome instead of downloading Playwright Chrome"; \
		else \
			NODE_TLS_REJECT_UNAUTHORIZED=0 $(UV) run playwright install chromium; \
			$(UV) run playwright install-deps; \
		fi; \
	else \
		NODE_TLS_REJECT_UNAUTHORIZED=0 $(UV) run playwright install chromium; \
		$(UV) run playwright install-deps; \
	fi
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
		-e USE_SYSTEM_CHROME=$(USE_SYSTEM_CHROME) \
		-e CHROME_EXECUTABLE_PATH=$(CHROME_EXECUTABLE_PATH) \
		$(SERVICE_IMAGE) \
		bash -c "apt-get update && \
			apt-get install -y --no-install-recommends python3 python3-pip python3-venv curl ca-certificates openssl git \
			libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libdbus-1-3 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 && \
			if [ '$$USE_SYSTEM_CHROME' = 'true' ]; then \
				echo 'Installing system Chrome...'; \
				wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
				sh -c 'echo \"deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main\" >> /etc/apt/sources.list.d/google.list' && \
				apt-get update && \
				apt-get install -y google-chrome-stable; \
			fi && \
			pip3 install --break-system-packages uv && \
			cd /app && \
			uv venv --clear && \
			. .venv/bin/activate && \
			uv pip install -e . && \
			if [ '$$USE_SYSTEM_CHROME' != 'true' ]; then \
				NODE_TLS_REJECT_UNAUTHORIZED=0 uv run playwright install chromium && \
				uv run playwright install-deps; \
			fi && \
			uv run aqe serve"
	@echo "Service running on http://localhost:$(SERVICE_HOST_PORT)"
	@echo "Container runs as root user for full privileges"
	@echo "Container uses host networking (can access host services via localhost"
	@echo "Playwright Chromium has been installed"
	@echo "To stop: make stop-container"
	@echo "To view logs: docker logs aqe-service -f"

stop-container:
	@echo "Stopping and removing aqe-service container..."
	@docker stop aqe-service 2>/dev/null || echo "Container not running"
	@docker rm aqe-service 2>/dev/null || echo "Container already removed"
	@echo "Container stopped and removed successfully"
