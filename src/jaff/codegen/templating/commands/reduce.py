# ABOUTME: REDUCE command: expand $( ... )$ into a sum over network properties
# ABOUTME: Holds the header's shape; validation and rendering arrive with the renderer

from dataclasses import dataclass
from typing import Tuple

from .base import Command


@dataclass(frozen=True)
class Reduce(Command):
    """``REDUCE vars IN props``."""

    vars: Tuple[str, ...]
    props: Tuple[str, ...]
