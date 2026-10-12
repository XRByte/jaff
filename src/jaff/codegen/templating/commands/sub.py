# ABOUTME: SUB command: substitute network-wide tokens such as $nspec$ in the body
# ABOUTME: Holds the header's shape; validation and rendering arrive with the renderer

from dataclasses import dataclass
from typing import Tuple

from .base import Command


@dataclass(frozen=True)
class Sub(Command):
    """``SUB tokens``."""

    tokens: Tuple[str, ...]
