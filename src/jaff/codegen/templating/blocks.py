# ABOUTME: Block types produced by the template engine's single pass over a template
# ABOUTME: TextBlock holds plain lines; CommandBlock holds one command, its options and body

from dataclasses import dataclass
from typing import List, Optional, Union

from .commands import Command
from .options import Options


@dataclass
class TextBlock:
    """Template lines copied to the output unchanged."""

    lines: List[str]
    line: int


@dataclass
class CommandBlock:
    """One ``$$JAFF`` directive: command, options, raw header lines and body lines.

    ``body_lines`` is empty for commands without a body (``DEFINE``).
    """

    command: Command
    options: Optional[Options]
    header_lines: List[str]
    body_lines: List[str]
    line: int


Block = Union[TextBlock, CommandBlock]
