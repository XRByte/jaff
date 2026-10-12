# ABOUTME: DEFINE command: a named option set that later blocks pull in with USE
# ABOUTME: Has no body; it renders nothing and only registers its options

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List

from ..base import Command

if TYPE_CHECKING:
    from ...context import RenderContext
    from ...resolve import Resolved


@dataclass(frozen=True)
class Define(Command):
    """``DEFINE name { ... }``."""

    has_body: ClassVar[bool] = False
    name: str

    def render(self, body: List[str], resolved: Resolved, ctx: RenderContext) -> str:
        """A DEFINE produces no output."""
        return ""
