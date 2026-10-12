# ABOUTME: REPEAT command: expand the body once per item of a network property
# ABOUTME: Holds the header's shape; validation and rendering arrive with the renderer

from dataclasses import dataclass
from typing import Tuple

from .base import Command


@dataclass(frozen=True)
class Repeat(Command):
    """``REPEAT vars IN prop``."""

    vars: Tuple[str, ...]
    prop: str
