# ABOUTME: HAS command: substitute 1 or 0 for whether an entity is in the network
# ABOUTME: Holds the header's shape; validation and rendering arrive with the renderer

from dataclasses import dataclass

from .base import Command


@dataclass(frozen=True)
class Has(Command):
    """``HAS kind entity``."""

    kind: str
    entity: str
