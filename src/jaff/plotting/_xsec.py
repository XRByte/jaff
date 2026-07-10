"""Domain helpers for cross-section axis scaling.

Cross-section grids often pad the high-energy tail with zeros, which would
stretch a plot far past the meaningful data.  These pure functions decide the
energy range to show and whether a log x-axis is warranted, independent of any
matplotlib state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def positive_span(
    df: pd.DataFrame, x_col: str = "energy", y_col: str = "xsec"
) -> tuple[float | None, float | None]:
    """Range of *x_col* over which any curve has finite, positive *y_col*.

    Parameters
    ----------
    df : pandas.DataFrame
        Long frame with the given x/y columns.
    x_col, y_col : str, optional
        Column names, by default ``"energy"`` / ``"xsec"``.

    Returns
    -------
    tuple[float | None, float | None]
        ``(lo, hi)`` x bounds, or ``(None, None)`` if no point qualifies.
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    mask = np.isfinite(y) & (y > 0) & np.isfinite(x)
    if not mask.any():
        return None, None
    xm = x[mask]
    return float(xm.min()), float(xm.max())


def finite_span(
    df: pd.DataFrame, x_col: str = "energy"
) -> tuple[float | None, float | None]:
    """Full finite range of *x_col* (fallback when nothing is positive)."""
    x = df[x_col].to_numpy()
    finite = x[np.isfinite(x)]
    if not finite.size:
        return None, None
    return float(finite.min()), float(finite.max())


def use_log_x(energy_log: bool, lo: float | None, hi: float | None) -> bool:
    """Decide whether the x-axis should actually be log-scaled.

    A log axis is dropped to linear when the data spans less than one decade,
    which keeps narrow ranges readable.

    Parameters
    ----------
    energy_log : bool
        Caller's requested preference.
    lo, hi : float or None
        Energy span (e.g. from :func:`positive_span`).

    Returns
    -------
    bool
        ``True`` to use a log x-axis.
    """
    if not energy_log:
        return False
    if lo is not None and hi is not None and lo > 0 and np.log10(hi / lo) < 1.0:
        return False
    return True


def padded_limits(
    lo: float, hi: float, log: bool, pad: float = 0.03
) -> tuple[float, float]:
    """Pad an energy range so data does not sit flush against the spines.

    Pads multiplicatively on a log axis, additively on a linear one.

    Parameters
    ----------
    lo, hi : float
        Energy bounds (``hi > lo``).
    log : bool
        Whether the axis is log-scaled.
    pad : float, optional
        Fractional padding, by default ``0.03``.

    Returns
    -------
    tuple[float, float]
        Padded ``(lo, hi)``.
    """
    if log:
        factor = (hi / lo) ** pad
        return lo / factor, hi * factor
    margin = pad * (hi - lo)
    return lo - margin, hi + margin
