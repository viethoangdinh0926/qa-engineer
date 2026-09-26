"""Desktop input via mss and PyAutoGUI. Live use requires an X11 DISPLAY."""

import os

from aqe.errors import HarnessError
from aqe.state import GUIAction


class DesktopDriver:
    """Capture the screen and send OS input. Sends nothing without DISPLAY."""

    def __init__(self) -> None:
        self.actions: list[GUIAction] = []

    def capture(self) -> bytes:
        self._require_display()
        try:
            import mss
        except ImportError as exc:
            raise HarnessError("desktop_input_failed", "mss is not installed.") from exc
        try:
            with mss.mss() as grabber:
                monitor = grabber.monitors[0]
                shot = grabber.grab(monitor)
                return mss.tools.to_png(shot.rgb, shot.size)
        except Exception as exc:
            raise HarnessError("desktop_input_failed", f"Screen capture failed: {exc}") from exc

    def act(self, action: GUIAction) -> None:
        self._require_display()
        self.actions.append(action)
        try:
            import pyautogui
        except ImportError as exc:
            raise HarnessError("desktop_input_failed", "pyautogui is not installed.") from exc
        try:
            if action.action == "click" and action.coordinate:
                pyautogui.click(x=action.coordinate[0], y=action.coordinate[1])
            elif action.action == "type" and action.text:
                pyautogui.write(action.text)
            elif action.action == "press" and action.text:
                pyautogui.press(action.text)
        except Exception as exc:
            raise HarnessError("desktop_input_failed", f"Desktop input failed: {exc}") from exc

    def _require_display(self) -> None:
        if not os.environ.get("DISPLAY"):
            raise HarnessError(
                "desktop_input_failed",
                "Desktop input needs an X11 DISPLAY. No input was sent.",
            )
