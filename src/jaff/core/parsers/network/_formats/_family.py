from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import NetworkFormat
    from ._context import ParseContext
    from ._record import ParsedRecord, Record

_FAMILY_REGISTRY: dict[str, type["FormatFamily"]] = {}


def register_family(cls: type["FormatFamily"]) -> type["FormatFamily"]:
    """Register a custom FormatFamily subclass, keyed by its ``name``."""
    if not cls.name:
        raise ValueError(
            f"{cls.__name__} must set a non-empty 'name' to be registered as a "
            f"format family"
        )
    if cls.name in _FAMILY_REGISTRY:
        raise ValueError(f"A format family named {cls.name!r} is already registered")
    _FAMILY_REGISTRY[cls.name] = cls
    return cls


class FormatFamily:
    """Owns the ordered bucket of one subpackage's line-types and parses it.

    The default (added in a later task) walks the bucket in file order and
    dispatches each record to the member whose ``_global_re`` matches. Subclasses
    (e.g. UCLCHEM) override ``process``.
    """

    name: str = ""
    priority: int = 0

    def __init__(self, members: list["NetworkFormat"]):
        self.members = sorted(members, key=lambda m: m.priority)

    def process(
        self, records: list["Record"], ctx: "ParseContext"
    ) -> list["ParsedRecord"]:
        raise NotImplementedError
