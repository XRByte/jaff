# ABOUTME: END command: closes the body of the directive above it
# ABOUTME: Never forms a block of its own; the engine consumes it while reading a body

from dataclasses import dataclass
from typing import ClassVar

from ..base import Command


@dataclass(frozen=True)
class End(Command):
    """``END``."""

    has_body: ClassVar[bool] = False
