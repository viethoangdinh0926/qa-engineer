"""Harness failures, distinct from assertion results."""


class HarnessError(Exception):
    """The execution harness broke before an assertion could be judged."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SpecValidationError(Exception):
    """The request body cannot become a run."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
