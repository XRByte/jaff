# ABOUTME: LANG command: names the template's target language, overriding its extension
# ABOUTME: A file setting: no body, no output; read by TemplateParser before rendering

from dataclasses import dataclass

from ..base import FileSetting


@dataclass(frozen=True)
class Lang(FileSetting):
    """``LANG name``, e.g. ``LANG fortran`` or ``LANG c++``."""

    name: str
