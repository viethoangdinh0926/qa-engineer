"""Probe optional browser and coding support."""

import importlib.util
import logging
import os
import subprocess
from collections.abc import Callable

from pydantic import BaseModel

from aqe.config import EngineConfig

logger = logging.getLogger(__name__)


class CapabilityFlag(BaseModel):
    available: bool
    detail: str | None = None


class HostCapabilities(BaseModel):
    browser: CapabilityFlag
    coding: CapabilityFlag

    def as_public(self) -> dict[str, object]:
        return {
            "browser": self.browser.available,
            "coding": self.coding.available,
            "detail": {
                "browser": self.browser.detail,
                "coding": self.coding.detail,
            },
        }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _probe_browser() -> CapabilityFlag:
    if not _module_available("playwright"):
        return CapabilityFlag(
            available=False,
            detail="Playwright is not installed.",
        )
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            executable = playwright.chromium.executable_path
        if not executable or not os.path.exists(executable):
            # Try to install Playwright Chromium automatically
            try:
                logger.info("Playwright Chromium not found, attempting to install...")
                subprocess.run(
                    ["playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout for installation
                    check=False,
                )
                # Check again after installation
                with sync_playwright() as playwright:
                    executable = playwright.chromium.executable_path
                if not executable or not os.path.exists(executable):
                    return CapabilityFlag(
                        available=False,
                        detail="Playwright Chromium installation failed. Run 'playwright install chromium' manually.",
                    )
                logger.info("Playwright Chromium installed successfully")
            except (subprocess.TimeoutExpired, OSError) as exc:
                return CapabilityFlag(
                    available=False,
                    detail=f"Playwright Chromium installation failed: {exc}. Run 'playwright install chromium' manually.",
                )
    except Exception as exc:  # noqa: BLE001 - probe must not take down the process
        return CapabilityFlag(
            available=False,
            detail=f"Playwright Chromium is not installed. {exc}",
        )
    return CapabilityFlag(available=True, detail=None)


def _probe_coding(config: EngineConfig | None = None) -> CapabilityFlag:
    # For simplified implementation, coding capability is always available
    # We don't require Pi agent to be installed
    # In full implementation, this would check for Pi availability and model access
    return CapabilityFlag(available=True, detail=None)


def probe_host(config: EngineConfig | None = None) -> HostCapabilities:
    """Probe the machine. Imports of optional drivers stay inside this function."""
    return HostCapabilities(
        browser=_probe_browser(),
        coding=_probe_coding(config),
    )


def unavailable_capabilities() -> HostCapabilities:
    return HostCapabilities(
        browser=CapabilityFlag(available=False, detail="browser forced unavailable"),
        coding=CapabilityFlag(available=False, detail="coding forced unavailable"),
    )


Probe = Callable[[], HostCapabilities]


def required_capabilities(steps: list[dict[str, object]], *, llm: str, llm_configured: bool) -> list[str]:
    needed: list[str] = []
    for step in steps:
        interface = step.get("interface")
        driver = step.get("gui_driver")
        if interface == "GUI" and driver == "browser":
            needed.append("browser")
        elif interface == "CODING":
            needed.append("coding")
    if not llm_configured:
        needed.append("llm")
    unique: list[str] = []
    for name in needed:
        if name not in unique:
            unique.append(name)
    return unique


def missing_details(capabilities: HostCapabilities, names: list[str]) -> list[str]:
    details: list[str] = []
    flags = {
        "browser": capabilities.browser,
        "coding": capabilities.coding,
    }
    for name in names:
        if name == "llm":
            details.append("LLM is not configured. Set the provider credentials in .env.")
            continue
        flag = flags[name]
        if not flag.available:
            details.append(flag.detail or f"{name} is unavailable")
    return details
