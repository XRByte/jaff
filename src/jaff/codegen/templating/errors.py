# ABOUTME: The one internal exception for template problems found after parsing
# ABOUTME: Carries an optional template line; render.py turns it into a ParserError

from typing import Optional


class TemplateError(ValueError):
    """A template problem at a known line; ``nline`` None means the header line."""

    def __init__(self, message: str, nline: Optional[int] = None) -> None:
        super().__init__(message)
        self.nline = nline
