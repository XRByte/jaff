# ABOUTME: Hollow marker placement at even on-screen arc-length spacing
# ABOUTME: Runs after the figure is drawn so axis limits and scales are final

"""Marker series: hollow markers spaced evenly along the drawn curve.

Markers are placed by on-screen arc length, so flat stretches get few and
steep ones as many as they need.  Placement needs final limits, so the
renderer queues :class:`PendingMarks` and calls :func:`place` after a draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .themes import Style


@dataclass(frozen=True, eq=False)
class PendingMarks:
    """A marker series waiting for final axis limits.

    ``gap_pt`` overrides the style's marker spacing (the sparser spacing of a
    line+marker series).
    """

    ax: Axes
    x: np.ndarray
    y: np.ndarray
    color: str
    marker: str
    lw: float
    zorder: float | None
    gap_pt: float | None = None


def mark_indices(disp: np.ndarray, gap_px: float) -> np.ndarray:
    """Indices of display points spaced every *gap_px* of arc length.

    Parameters
    ----------
    disp : numpy.ndarray
        ``(n, 2)`` display coordinates of the visible curve.
    gap_px : float
        Target spacing in pixels.

    Returns
    -------
    numpy.ndarray
        Sorted unique indices into *disp* (always includes ``0``).
    """
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(disp, axis=0).T))])
    targets = np.arange(0.0, arc[-1] + 1e-9, gap_px)
    return np.unique(np.clip(np.searchsorted(arc, targets), 0, arc.size - 1))


def place(fig: Figure, pending: list[PendingMarks], style: Style) -> None:
    """Draw every queued marker series onto its axes."""
    if not pending:
        return
    fig.canvas.draw()
    for p in pending:
        gap_px = (p.gap_pt or style.marker["gap_pt"]) * fig.dpi / 72.0
        disp = p.ax.transData.transform(np.column_stack([p.x, p.y]))
        bb = p.ax.bbox
        vis = (
            (disp[:, 0] >= bb.x0 - 0.5)
            & (disp[:, 0] <= bb.x1 + 0.5)
            & (disp[:, 1] >= bb.y0 - 0.5)
            & (disp[:, 1] <= bb.y1 + 0.5)
        )
        if not vis.any():
            continue
        idx = mark_indices(disp[vis], gap_px)
        p.ax.plot(
            p.x[vis][idx],
            p.y[vis][idx],
            linestyle="",
            marker=p.marker,
            markersize=style.marker["size"],
            markerfacecolor=style.surface,
            markeredgecolor=p.color,
            markeredgewidth=p.lw,
            color=p.color,
            zorder=p.zorder,
        )
