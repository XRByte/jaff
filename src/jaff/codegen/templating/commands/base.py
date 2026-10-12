# ABOUTME: Base class for $$JAFF commands, built from a parsed directive header
# ABOUTME: Subclasses are frozen dataclasses whose fields mirror the header's parts

from dataclasses import dataclass
from typing import Any, ClassVar, List


@dataclass(frozen=True)
class Command:
    """A parsed ``$$JAFF`` command; subclasses add the header's fields."""

    has_body: ClassVar[bool] = True

    @classmethod
    def from_items(cls, items: List[Any]) -> "Command":
        """Build the command from its header parts, in grammar order.

        Args:
            items: Header parts as plain strings or tuples of names.

        Returns:
            The command instance.
        """
        return cls(*items)
