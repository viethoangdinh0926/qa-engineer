UV ?= uv
VENV ?= .venv
SANDBOX_IMAGE ?= aqe-sandbox:local

.PHONY: help install install-browser sandbox-image test test-integration run serve

help:
	@echo "Dependency order:"
	@echo "  install"
	@echo "    └── install-browser"
	@echo "  sandbox-image"
	@echo ""
	@echo "  test               requires install"
	@echo "  test-integration   requires install-browser, sandbox-image"
	@echo "  run                requires install-browser, sandbox-image"
	@echo "  serve              requires install-browser, sandbox-image"
	@echo ""
	@echo "Targets:"
	@echo "  make install            Create .venv and install aqe"
	@echo "  make install-browser    Download Playwright Chromium"
	@echo "  make sandbox-image      Build $(SANDBOX_IMAGE)"
	@echo "  make test               Run the unit tests"
	@echo "  make test-integration   Run the Playwright and Docker test"
	@echo "  make run                Run SPEC_PATH from .env"
	@echo "  make serve              Serve using HOST and PORT from .env"

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

sandbox-image: $(VENV)/.sandbox-image

$(VENV)/.sandbox-image: docker/sandbox/Dockerfile
	docker build -t $(SANDBOX_IMAGE) docker/sandbox
	@mkdir -p $(VENV)
	@touch $@

test: $(VENV)/.install
	$(UV) run pytest

test-integration: $(VENV)/.install-browser $(VENV)/.sandbox-image
	$(UV) run pytest -m integration

run: $(VENV)/.install-browser $(VENV)/.sandbox-image
	$(UV) run aqe run

serve: $(VENV)/.install-browser $(VENV)/.sandbox-image
	$(UV) run aqe serve
