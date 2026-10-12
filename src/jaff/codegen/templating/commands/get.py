# ABOUTME: GET command: substitute properties of one species, reaction or element
# ABOUTME: Holds the header's shape; validation and rendering arrive with the renderer

from dataclasses import dataclass
from typing import Tuple

from .base import Command


@dataclass(frozen=True)
class Get(Command):
    """``GET props IN entity``."""

    props: Tuple[str, ...]
    entity: str
