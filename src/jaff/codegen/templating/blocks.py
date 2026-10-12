# ABOUTME: Block types produced by the template engine's single pass over a template
# ABOUTME: TextBlock holds plain lines; CommandBlock holds one parsed directive and its body

from dataclasses import dataclass
from typing import List, Union

from .syntax import Directive


@dataclass
class TextBlock:
    """Template lines copied to the output unchanged."""

    lines: List[str]
    line: int


@dataclass
class CommandBlock:
    """One ``$$JAFF`` directive: its parsed header, raw header lines and body lines."""

    directive: Directive
    header_lines: List[str]
    body_lines: List[str]
    line: int


Block = Union[TextBlock, CommandBlock]
