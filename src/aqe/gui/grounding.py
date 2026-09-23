"""Turn a screenshot and a step into a concrete GUI action."""

from typing import Protocol

from aqe.state import GUIAction, TestStep


class Grounder(Protocol):
    def ground(self, image: bytes, step: TestStep, operation: GUIAction | None = None) -> GUIAction:
        """Return the action to perform. Image bytes are the current capture."""


class SelectorGrounder:
    """Use the planner's role and name. This is the default browser grounder."""

    def ground(self, image: bytes, step: TestStep, operation: GUIAction | None = None) -> GUIAction:
        del image
        if operation is not None:
            return operation
        if step.operations:
            return step.operations[0]
        return GUIAction(action="click", selector={"role": "button", "name": step.action})


class CoordinateGrounder:
    """Return a point. Tests pass coordinates; otherwise a fixed point is used."""

    def __init__(self, coordinate: list[int] | None = None) -> None:
        self.coordinate = coordinate or [1, 1]

    def ground(self, image: bytes, step: TestStep, operation: GUIAction | None = None) -> GUIAction:
        del image
        if operation is not None and operation.coordinate:
            return operation
        if operation is not None:
            return operation.model_copy(update={"coordinate": list(self.coordinate)})
        return GUIAction(action="click", coordinate=list(self.coordinate), text=step.action)


class VlmGrounder:
    """Extension point for a pixel-grounding model. No model is bundled."""

    def ground(self, image: bytes, step: TestStep, operation: GUIAction | None = None) -> GUIAction:
        del image
        if operation is not None:
            return operation
        raise NotImplementedError(
            "VlmGrounder is the extension point for a ShowUI-style model. "
            "No vision model is bundled in this slice."
        )
