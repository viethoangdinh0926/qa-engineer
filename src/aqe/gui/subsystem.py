"""Capture, ground, then act."""

from pathlib import Path

from aqe.config import EngineConfig
from aqe.gui.desktop_driver import DesktopDriver
from aqe.gui.grounding import CoordinateGrounder, Grounder, SelectorGrounder
from aqe.gui.playwright_driver import PlaywrightDriver
from aqe.state import ActionResult, TestStep


class GUISubsystem:
    """Visual loop shared by the browser and desktop drivers."""

    def __init__(
        self,
        browser_driver: PlaywrightDriver,
        desktop_driver: DesktopDriver,
        browser_grounder: Grounder | None = None,
        desktop_grounder: Grounder | None = None,
    ) -> None:
        self.browser_driver = browser_driver
        self.desktop_driver = desktop_driver
        self.browser_grounder = browser_grounder or SelectorGrounder()
        self.desktop_grounder = desktop_grounder or CoordinateGrounder()

    def execute_visual_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        driver = self.desktop_driver if step.gui_driver == "desktop" else self.browser_driver
        grounder = self.desktop_grounder if step.gui_driver == "desktop" else self.browser_grounder
        image = driver.capture()
        operations = step.operations or [grounder.ground(image, step)]
        for operation in operations:
            grounded = grounder.ground(image, step, operation)
            driver.act(grounded)
        image = driver.capture()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        screenshot = evidence_dir / f"step-{step.step}.png"
        screenshot.write_bytes(image)
        reader = getattr(driver, "page_text", None)
        page_text = reader() if callable(reader) else ""
        summary = page_text.strip() or step.action
        evidence: dict[str, str] = {"summary": summary, "screenshot": str(screenshot)}
        source_reader = getattr(driver, "page_source", None)
        if callable(source_reader):
            page_source = source_reader().strip()
            if page_source:
                evidence["page_source"] = page_source
        return ActionResult(ok=True, summary=summary, evidence=evidence)

    def close(self) -> None:
        closer = getattr(self.browser_driver, "close", None)
        if closer:
            closer()


def build_gui(config: EngineConfig) -> GUISubsystem:
    del config
    return GUISubsystem(PlaywrightDriver(), DesktopDriver())
