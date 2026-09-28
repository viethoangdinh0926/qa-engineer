"""Headless Chromium driver. It does not need a desktop session."""

from aqe.errors import HarnessError
from aqe.state import GUIAction

_CONTROL_SCRIPT = """els => els.map(el => {
  const label = el.labels && el.labels.length ? el.labels[0].innerText : "";
  return (el.getAttribute("aria-label") || el.innerText || label || el.getAttribute("value") || el.getAttribute("placeholder") || "").trim();
})"""


def match_control(requested: str, names: list[str]) -> str | None:
    """Pick the control on the page that the planned name refers to."""
    present = [name.strip() for name in names if name and name.strip()]
    if not present:
        return None
    folded = {name.casefold(): name for name in present}
    key = requested.strip().casefold()
    if key in folded:
        return folded[key]
    if key:
        for name in present:
            folded_name = name.casefold()
            if key in folded_name or folded_name in key:
                return name
    if len(present) == 1:
        return present[0]
    return None


class PlaywrightDriver:
    """Launch Chromium with headless=True and drive the viewport."""

    def __init__(self) -> None:
        self.actions: list[GUIAction] = []
        self._playwright = None
        self._browser = None
        self._page = None

    def page_text(self) -> str:
        page = self._ensure_page()
        try:
            return page.locator("body").inner_text(timeout=5000)
        except Exception as exc:
            raise HarnessError("browser_launch_failed", f"Chromium page text failed: {exc}") from exc

    def page_source(self) -> str:
        page = self._ensure_page()
        try:
            return page.content()
        except Exception as exc:
            raise HarnessError("browser_launch_failed", f"Chromium page source failed: {exc}") from exc

    def capture(self) -> bytes:
        page = self._ensure_page()
        try:
            return page.screenshot(type="png")
        except Exception as exc:
            raise HarnessError("browser_launch_failed", f"Chromium screenshot failed: {exc}") from exc

    def act(self, action: GUIAction) -> None:
        page = self._ensure_page()
        self.actions.append(action)
        try:
            self._perform(page, action)
        except HarnessError:
            raise
        except Exception as exc:
            raise HarnessError("browser_launch_failed", f"Chromium action failed: {exc}") from exc

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    def _ensure_page(self):
        if self._page is not None:
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise HarnessError("browser_launch_failed", "Playwright is not installed.") from exc
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
            self._page = self._browser.new_page()
        except Exception as exc:
            self.close()
            raise HarnessError("browser_launch_failed", f"Chromium failed to launch: {exc}") from exc
        return self._page

    def _perform(self, page, action: GUIAction) -> None:
        if action.action == "goto":
            if not action.text:
                raise HarnessError("browser_launch_failed", "goto is missing a URL")
            page.goto(action.text, wait_until="domcontentloaded", timeout=15000)
            return
        if action.action == "press":
            page.keyboard.press(action.text or "Enter")
            return
        action = self._resolve(page, action)
        locator = self._locator(page, action)
        if action.action == "click":
            if locator is None and action.coordinate:
                page.mouse.click(action.coordinate[0], action.coordinate[1])
                return
            if locator is None:
                raise HarnessError("browser_launch_failed", "click has no selector")
            self._act_on(page, action, locator.click, timeout=10000)
            return
        if action.action == "type":
            if locator is None:
                raise HarnessError("browser_launch_failed", "type has no selector")
            self._act_on(page, action, locator.fill, action.text or "", timeout=10000)

    def _resolve(self, page, action: GUIAction) -> GUIAction:
        if action.action not in {"click", "type"} or not action.selector:
            return action
        role = action.selector.get("role") or ("button" if action.action == "click" else "textbox")
        requested = action.selector.get("name") or ""
        names = self._control_names(page, role)
        chosen = match_control(requested, names)
        if not chosen or chosen == requested:
            return action
        selector = dict(action.selector)
        selector["role"] = role
        selector["name"] = chosen
        return action.model_copy(update={"selector": selector})

    def _control_names(self, page, role: str) -> list[str]:
        if role == "button":
            selector = "button, input[type=submit], input[type=button], [role=button]"
        elif role == "textbox":
            selector = "input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, [role=textbox]"
        else:
            selector = f"[role={role}]"
        try:
            names = page.locator(selector).evaluate_all(_CONTROL_SCRIPT)
        except Exception:  # noqa: BLE001 - a page with no controls still uses the planned name
            return []
        return [str(name) for name in names if str(name).strip()]

    def _act_on(self, page, action: GUIAction, method, *args, **kwargs) -> None:
        try:
            method(*args, **kwargs)
        except Exception as exc:
            role = (action.selector or {}).get("role", action.action)
            names = self._control_names(page, str(role))
            available = ", ".join(names) if names else "none"
            raise HarnessError(
                "browser_launch_failed",
                f"Chromium action failed: {exc} The page's {role} controls are: {available}.",
            ) from exc

    def _locator(self, page, action: GUIAction):
        if not action.selector:
            return None
        role = action.selector.get("role", "button")
        name = action.selector.get("name")
        if name:
            return page.get_by_role(role, name=name)
        return page.get_by_role(role)
