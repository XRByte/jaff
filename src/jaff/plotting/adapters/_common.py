# ABOUTME: Shared helpers for the plotting adapters
# ABOUTME: Label extraction from reaction-like objects

"""Helpers shared by the rate and cross-section adapters."""

from __future__ import annotations

from typing import Any


def label_of(item: Any) -> str:
    """LaTeX equation of a reaction-like object, else ``str``."""
    return item.get_latex() if hasattr(item, "get_latex") else str(item)
