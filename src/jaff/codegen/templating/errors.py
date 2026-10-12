# ABOUTME: The one internal exception for template problems found after parsing
# ABOUTME: Points at a template line or a body line; render.py turns it into a ParserError

from typing import Optional


class TemplateError(ValueError):
    """A template problem at a known place in its block.

    Attributes:
        nline: Absolute template line, or ``None`` for the directive's header line.
        body_index: 0-based index of the offending body line, when the problem is
            in the body; it takes precedence over *nline*.
    """

    def __init__(
        self, message: str, nline: Optional[int] = None, body_index: Optional[int] = None
    ) -> None:
        super().__init__(message)
        self.nline = nline
        self.body_index = body_index
