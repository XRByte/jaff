# ABOUTME: DEFINE command: a named option set that later blocks pull in with USE
# ABOUTME: Has no body; it renders nothing and only registers its options

from dataclasses import dataclass
from typing import ClassVar

from .base import Command


@dataclass(frozen=True)
class Define(Command):
    """``DEFINE name { … }``."""

    has_body: ClassVar[bool] = False
    name: str
