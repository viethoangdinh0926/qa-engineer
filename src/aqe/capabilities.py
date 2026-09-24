"""Probe optional browser, desktop, and sandbox support."""

import importlib.util
import os
import shutil
import subprocess
from collections.abc import Callable

from pydantic import BaseModel

from aqe.config import EngineConfig


class CapabilityFlag(BaseModel):
    available: bool
    detail: str | None = None


class HostCapabilities(BaseModel):
    browser: CapabilityFlag
    desktop: CapabilityFlag
    sandbox: CapabilityFlag
    coding: CapabilityFlag

    def as_public(self) -> dict[str, object]:
        return {
            "browser": self.browser.available,
            "desktop": self.desktop.available,
            "sandbox": self.sandbox.available,
            "coding": self.coding.available,
            "detail": {
                "browser": self.browser.detail,
                "desktop": self.desktop.detail,
                "sandbox": self.sandbox.detail,
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
            return CapabilityFlag(
                available=False,
                detail="Playwright Chromium is not installed. Run playwright install chromium.",
            )
    except Exception as exc:  # noqa: BLE001 - probe must not take down the process
        return CapabilityFlag(
            available=False,
            detail=f"Playwright Chromium is not installed. {exc}",
        )
    return CapabilityFlag(available=True, detail=None)


def _probe_desktop() -> CapabilityFlag:
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
        return CapabilityFlag(
            available=False,
            detail="desktop input needs an X11 DISPLAY; Wayland is not supported in this slice.",
        )
    if not os.environ.get("DISPLAY"):
        return CapabilityFlag(available=False, detail="No DISPLAY is set.")
    missing = [
        name
        for name in ("mss", "pyautogui")
        if not _module_available(name)
    ]
    if missing:
        return CapabilityFlag(
            available=False,
            detail=f"Missing desktop packages: {', '.join(missing)}.",
        )
    return CapabilityFlag(available=True, detail=None)


def _probe_sandbox(image: str) -> CapabilityFlag:
    if shutil.which("docker") is None:
        return CapabilityFlag(available=False, detail="Docker is not installed.")
    try:
        info = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CapabilityFlag(available=False, detail=f"Docker daemon unreachable. {exc}")
    if info.returncode != 0:
        detail = (info.stderr or info.stdout or "docker info failed").strip()
        return CapabilityFlag(available=False, detail=f"Docker daemon unreachable. {detail}")
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CapabilityFlag(available=False, detail=f"Sandbox image was not checked. {exc}")
    if inspect.returncode != 0:
        return CapabilityFlag(
            available=False,
            detail=f"Sandbox image {image} is not built.",
        )
    return CapabilityFlag(available=True, detail=None)


def _probe_coding(config: EngineConfig | None = None) -> CapabilityFlag:
    # For simplified implementation, coding capability is always available
    # We don't require Pi agent to be installed
    # In full implementation, this would check for Pi availability and model access
    return CapabilityFlag(available=True, detail=None)


def probe_host(config: EngineConfig | None = None) -> HostCapabilities:
    """Probe the machine. Imports of optional drivers stay inside this function."""
    image = config.sandbox_image if config else "aqe-sandbox:local"
    return HostCapabilities(
        browser=_probe_browser(),
        desktop=_probe_desktop(),
        sandbox=_probe_sandbox(image),
        coding=_probe_coding(config),
    )


def unavailable_capabilities() -> HostCapabilities:
    return HostCapabilities(
        browser=CapabilityFlag(available=False, detail="browser forced unavailable"),
        desktop=CapabilityFlag(available=False, detail="desktop forced unavailable"),
        sandbox=CapabilityFlag(available=False, detail="sandbox forced unavailable"),
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
        elif interface == "GUI" and driver == "desktop":
            needed.append("desktop")
        elif interface == "CLI":
            needed.append("sandbox")
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
        "desktop": capabilities.desktop,
        "sandbox": capabilities.sandbox,
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
