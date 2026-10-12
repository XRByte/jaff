# ABOUTME: jaff.plotting public API: spec-driven matplotlib figures
# ABOUTME: plot / plot_rates / plot_xsecs builders plus theme registries

"""Spec-driven, publication-style plotting for JAFF.

See :func:`plot` for the figure-spec format, and :func:`plot_rates` /
:func:`plot_xsecs` for reaction plots.
"""

from .api import plot, plot_rates, plot_xsecs
from .render import render
from .spec import AxesSpec, FigureSpec, SeriesSpec
from .themes import PALETTES, STYLES, register_palette, register_style

__all__ = [
    "plot",
    "plot_rates",
    "plot_xsecs",
    "render",
    "FigureSpec",
    "AxesSpec",
    "SeriesSpec",
    "STYLES",
    "PALETTES",
    "register_style",
    "register_palette",
]
