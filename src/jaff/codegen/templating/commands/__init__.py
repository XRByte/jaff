# ABOUTME: Registry of $$JAFF commands keyed by the grammar's command aliases
# ABOUTME: The syntax layer builds commands through it; one module per command

from typing import Dict, Type

from .base import Command
from .define import Define
from .end import End
from .get import Get
from .has import Has
from .reduce import Reduce
from .repeat import Repeat
from .sub import Sub

COMMANDS: Dict[str, Type[Command]] = {
    "repeat": Repeat,
    "sub": Sub,
    "get": Get,
    "has": Has,
    "reduce": Reduce,
    "define": Define,
    "end": End,
}

__all__ = [
    "COMMANDS",
    "Command",
    "Define",
    "End",
    "Get",
    "Has",
    "Reduce",
    "Repeat",
    "Sub",
]
