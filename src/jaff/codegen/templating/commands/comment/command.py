# ABOUTME: COMMENT command: the comment token used when directives are echoed
# ABOUTME: A file setting: no body, no output; defaults to the language's comment

from dataclasses import dataclass

from ..base import FileSetting


@dataclass(frozen=True)
class Comment(FileSetting):
    """``COMMENT "token"``, e.g. ``COMMENT "#"``."""

    token: str
