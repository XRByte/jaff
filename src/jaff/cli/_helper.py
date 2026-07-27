"""Shared argument helpers for the JAFF command-line entry points."""

from __future__ import annotations


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
