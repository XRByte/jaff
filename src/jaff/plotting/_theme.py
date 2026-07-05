"""House plotting theme for JAFF.

The theme is built from seaborn's own style machinery
(:func:`seaborn.axes_style` + :func:`seaborn.plotting_context`) so figures get
the genuine seaborn look -- soft grid, scaled typography, muted spines --
rather than the boxy matplotlib default.  The default style is ``"darkgrid"``
(see :data:`DEFAULT_STYLE`) with the JAFF brand palette; a thin layer of JAFF
overrides (palette, figure size, save DPI) sits on top.

By default the theme is applied *scoped* (only while a JAFF plot is being
drawn, via :func:`theme_context`), so importing or instantiating the plotter
never mutates global matplotlib state.  Call :func:`apply_global_theme` to opt
into a sticky, session-wide theme instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from cycler import cycler

DEFAULT_STYLE: str = "darkgrid"
DEFAULT_CONTEXT: str = "notebook"

#: seaborn "muted" palette -- the default JAFF colour cycle (soft, polished).
MUTED_PALETTE: list[str] = [
    "#4878d0",
    "#ee854a",
    "#6acc64",
    "#d65f5f",
    "#956cb4",
    "#8c613c",
    "#dc7ec0",
    "#797979",
    "#d5bb67",
    "#82c6e2",
]

#: seaborn "deep" palette -- a more saturated alternative colour cycle.
DEEP_PALETTE: list[str] = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]

#: JAFF logo palette -- warm, brand-matching alternative colour cycle.
LOGO_PALETTE: list[str] = [
    "#8b6cff",
    "#e05fb0",
    "#ff6a5a",
    "#ffc24b",
]


def theme_rc(
    palette: list[str] | None = None,
    style: str = DEFAULT_STYLE,
    context: str = DEFAULT_CONTEXT,
    font_scale: float = 1.05,
    **overrides: Any,
) -> dict[str, Any]:
    """Return the JAFF ``rcParams`` dictionary, built from a seaborn theme.

    Parameters
    ----------
    palette : list[str] or None, optional
        Colour cycle to use.  Defaults to :data:`LOGO_PALETTE` (the JAFF brand
        colours).
    style : str, optional
        seaborn axes style (``"whitegrid"``, ``"darkgrid"``, ``"white"``,
        ``"ticks"``).  Defaults to :data:`DEFAULT_STYLE` (``"darkgrid"``).
    context : str, optional
        seaborn plotting context (``"paper"``, ``"notebook"``, ``"talk"``,
        ``"poster"``).  Scales fonts and line widths.  Defaults to
        :data:`DEFAULT_CONTEXT`.
    font_scale : float, optional
        Extra font scaling on top of *context*, by default ``1.05``.
    **overrides
        Individual ``rcParams`` entries to override.

    Returns
    -------
    dict
        A dictionary suitable for :meth:`matplotlib.rcParams.update`,
        :func:`matplotlib.pyplot.rc_context`, or the seaborn objects
        ``Plot.theme`` / ``Plot.config.theme``.
    """
    colors = palette if palette is not None else LOGO_PALETTE
    rc: dict[str, Any] = {}
    # Seaborn's style (grid/spines/background) and context (font/line scale).
    rc.update(sns.axes_style(style))
    rc.update(sns.plotting_context(context, font_scale=font_scale))
    # JAFF overrides: palette + publication figure/output defaults.
    rc.update(
        {
            "axes.prop_cycle": cycler(color=colors),
            "figure.figsize": (6.4, 4.0),
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "mathtext.fontset": "dejavusans",
            "legend.frameon": False,
        }
    )
    rc.update(overrides)
    return rc


def despine(ax: plt.Axes) -> None:
    """Remove the top and right spines for the trimmed seaborn finish."""
    sns.despine(ax=ax)


@contextmanager
def theme_context(rc: dict[str, Any]) -> Iterator[None]:
    """Apply *rc* only for the duration of the ``with`` block.

    Wraps :func:`matplotlib.pyplot.rc_context` so figures, axes, and raw
    matplotlib artists (bars, fills) created inside the block pick up the
    theme without any global state being mutated.

    The seaborn objects color cycle is not read from ``rcParams``, so this is
    paired with an explicit ``Plot.theme(rc)`` call in the plotter.
    """
    with plt.rc_context(rc):
        yield


def apply_global_theme(
    palette: list[str] | None = None,
    style: str = DEFAULT_STYLE,
    context: str = DEFAULT_CONTEXT,
    font_scale: float = 1.05,
    **overrides: Any,
) -> None:
    """Apply the JAFF seaborn theme globally and persistently for the session.

    Mutates the global ``matplotlib.rcParams`` *and* the seaborn objects
    ``Plot.config.theme`` so every subsequent plot -- JAFF or otherwise --
    inherits the house style.  This is the opt-in counterpart to the default
    scoped behaviour.

    Parameters
    ----------
    palette : list[str] or None, optional
        Colour cycle; defaults to :data:`LOGO_PALETTE`.
    style, context, font_scale
        Forwarded to :func:`theme_rc`.
    **overrides
        Individual ``rcParams`` entries to override.
    """
    rc = theme_rc(palette, style, context, font_scale, **overrides)
    mpl.rcParams.update(rc)
    so.Plot.config.theme.update(rc)
