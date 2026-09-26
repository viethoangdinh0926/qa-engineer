"""Capture, ground, then act in the browser."""

from pathlib import Path

from aqe.config import EngineConfig
from aqe.gui.grounding import Grounder, SelectorGrounder
from aqe.gui.playwright_driver import PlaywrightDriver
from aqe.state import ActionResult, TestStep


class GUISubsystem:
    """Browser loop: capture the page, ground each operation, then act."""

    def __init__(
        self,
        browser_driver: PlaywrightDriver,
        browser_grounder: Grounder | None = None,
    ) -> None:
        self.browser_driver = browser_driver
        self.browser_grounder = browser_grounder or SelectorGrounder()

    def execute_visual_action(self, step: TestStep, evidence_dir: Path) -> ActionResult:
        driver = self.browser_driver
        grounder = self.browser_grounder
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
    return GUISubsystem(PlaywrightDriver())
