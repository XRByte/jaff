"""Shared argument helpers for the JAFF command-line entry points."""

from __future__ import annotations

from enum import Enum


class DuplicatePolicy(str, Enum):
    """Valid ``--duplicate-policy`` values shared by the JAFF CLI tools.

    Selects how two rate coefficients sharing a reaction, mechanism, and
    temperature range are resolved during network construction (see
    :class:`~jaff.core.network.NetworkSpec`).
    """

    preserve_first = "preserve-first"
    preserve_last = "preserve-last"
    error = "error"


def funcfile_arg(value: str) -> bool | str:
    """
    Parse the ``--funcfile`` flag, mapping ``true``/``false`` onto booleans.

    A command line cannot express a boolean literal the way a ``jaffgen.toml``
    can, so ``--funcfile false`` (case-insensitive) is the CLI spelling for
    ``funcfile=False`` — skip auxiliary-function loading entirely — and
    ``--funcfile true`` for ``funcfile=True``, which scans the network
    directory.  Passing ``true`` is the only way to override a ``funcfile``
    path set in a config file back to a directory scan.

    Parameters
    ----------
    value : str
        Raw value supplied on the command line.

    Returns
    -------
    bool | str
        ``False`` to skip, ``True`` to scan, otherwise *value* unchanged.
    """
    lowered = value.lower()
    if lowered == "false":
        return False
    if lowered == "true":
        return True

    return value
