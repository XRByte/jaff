# ABOUTME: Options attached to one $$JAFF directive block: REPLACE pairs, USE names, groups
# ABOUTME: Shared by the syntax layer and the command classes; holds data only

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Options:
    """Options of one directive block, in the order they were written."""

    replace: List[Tuple[str, str]] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    groups: Dict[str, Dict[str, str]] = field(default_factory=dict)
