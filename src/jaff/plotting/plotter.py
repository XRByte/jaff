"""Publication-quality plotting for JAFF, built on the seaborn objects API.

:class:`Plotter` draws line data (rate coefficients) and photo cross sections.
Curves are rendered with the seaborn objects interface (``seaborn.objects``)
onto caller-supplied or freshly created matplotlib axes via ``Plot.on(ax)``;
band bars and shaded fills are drawn with matplotlib directly (the objects
``Area`` mark does not render reliably onto an existing axes).

By default the house theme is applied *scoped* -- only while a figure is being
drawn -- so instantiating :class:`Plotter` never mutates global matplotlib
state.  Pass ``global_theme=True`` (or call
:func:`jaff.plotting.apply_global_theme`) to opt into a sticky session theme.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn.objects as so

from . import _frames, _units, _xsec
from ._theme import (
    MUTED_PALETTE,
    apply_global_theme,
    despine,
    theme_context,
    theme_rc,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..physics.photo_reactions._photochemistry import XsecsProps


class Plotter:
    """Publication-quality plotter using the seaborn objects interface.

    Parameters
    ----------
    palette : list[str] or None, optional
        Colour cycle for curves.  Defaults to the seaborn "muted" palette
        (:data:`jaff.plotting._theme.MUTED_PALETTE`).  Pass
        :data:`jaff.plotting._theme.DEEP_PALETTE` for a more saturated cycle,
        or :data:`jaff.plotting._theme.LOGO_PALETTE` for the brand palette.
    global_theme : bool, optional
        If ``True``, apply the house theme globally and persistently on
        construction (mutating ``matplotlib.rcParams``).  If ``False``
        (default), the theme is applied only for the duration of each plot
        call, leaving global state untouched.
    **rc_overrides
        Individual ``rcParams`` entries to override in the theme.
    """

    #: Display labels for the cross-section processes.
    _PROC_LABELS: dict[str, str] = {
        "photo_absorption": "Photoabsorption",
        "photodecay": "Photodecay",
    }

    _RASTER: frozenset[str] = frozenset({"png", "jpg", "jpeg", "tif", "tiff"})

    def __init__(
        self,
        palette: list[str] | None = None,
        global_theme: bool = False,
        **rc_overrides: Any,
    ) -> None:
        self._palette = palette if palette is not None else MUTED_PALETTE
        self._rc = theme_rc(self._palette, **rc_overrides)
        self._global = global_theme
        if global_theme:
            apply_global_theme(self._palette, **rc_overrides)

    # -- theming -----------------------------------------------------------

    def _theme_scope(self):
        """Context manager that scopes the theme unless it is applied globally."""
        if self._global:
            # Already applied globally; no scoping needed.
            from contextlib import nullcontext

            return nullcontext()
        return theme_context(self._rc)

    def _apply_plot_theme(self, plot: so.Plot) -> so.Plot:
        """Attach the house rc theme to a seaborn ``Plot`` (objects color cycle
        is independent of rcParams, so it is set separately via ``.scale``)."""
        return plot.theme(self._rc)

    # -- output ------------------------------------------------------------

    def __finish(
        self,
        fig: plt.Figure,
        show: bool,
        save: bool,
        filename: str,
        dpi: int = 300,
    ) -> None:
        """Lay out, optionally save (format from extension), optionally show."""
        fig.tight_layout()

        # Save before show: plt.show() may close/clear the figure.
        if save:
            ext = Path(filename).suffix.lower().lstrip(".")
            kw: dict[str, Any] = {"bbox_inches": "tight"}
            if ext in self._RASTER:
                kw["dpi"] = dpi
            fig.savefig(filename, **kw)

        if show:
            plt.show()

    # -- generic line plot -------------------------------------------------

    def plot(
        self,
        x: list | float | np.ndarray,
        y: list | float | np.ndarray,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        xlabel: str = "",
        ylabel: str = "",
        xscale: str = "linear",
        yscale: str = "linear",
        title: str = "",
        label: str = "",
        grid: bool = True,
        show: bool = True,
        save: bool = False,
        filename: str = "plot.png",
        **line_kw: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Generic line plot.

        Parameters
        ----------
        x, y
            Data to plot.
        fig, ax
            Existing figure/axes to draw onto.  Created if ``None``.
        label
            Legend entry; a legend is drawn when non-empty.
        save
            Write to ``filename``.  Output format is inferred from the
            extension (``.png``, ``.pdf``, ``.svg``, ``.jpg`` ...).
        **line_kw
            Forwarded to :class:`seaborn.objects.Line` (e.g. ``linewidth``,
            ``linestyle``, ``marker``).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        df = _frames.line_frame(x, y)

        with self._theme_scope():
            if fig is None or ax is None:
                fig, ax = plt.subplots()

            plot = so.Plot(df, x="x", y="y").add(
                so.Line(color=self._palette[0], **line_kw)
            )
            plot = self.__scale_axes(plot, xscale, yscale)
            plot = plot.label(x=xlabel, y=ylabel, title=title)
            self._apply_plot_theme(plot).on(ax).plot()

            ax.grid(grid)
            despine(ax)
            if label:
                # Objects marks carry no legend label for a single group; attach
                # the entry to the drawn line directly.
                if ax.lines:
                    ax.lines[-1].set_label(label)
                    ax.legend()

            self.__finish(fig, show, save, filename)

        return fig, ax

    @staticmethod
    def __scale_axes(plot: so.Plot, xscale: str, yscale: str) -> so.Plot:
        """Apply log scales to a ``Plot`` where requested (linear is the default)."""
        scales: dict[str, str] = {}
        if xscale == "log":
            scales["x"] = "log"
        if yscale == "log":
            scales["y"] = "log"
        return plot.scale(**scales) if scales else plot

    # -- cross-section plot ------------------------------------------------

    def plot_xsec(
        self,
        xsecs: XsecsProps,
        processes: list[str] | None = None,
        layout: str = "overlay",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        energy_unit: str = "eV",
        xsec_unit: str = "cm^2",
        energy_log: bool = True,
        xsec_log: bool = True,
        trim: bool = True,
        shade: bool | float = False,
        show_bands: bool = False,
        bands: pd.DataFrame | None = None,
        title: str = "",
        grid: bool = True,
        show: bool = True,
        save: bool = False,
        filename: str = "xsec.png",
    ) -> tuple[plt.Figure, Any]:
        """Plot photo cross sections sigma(E) on log-log axes.

        Single home for cross-section plotting: handles unit conversion, axis
        scaling and labelling.  ``Reaction.plot_xsecs`` delegates here.

        Parameters
        ----------
        xsecs
            Mapping as returned by :func:`jaff.physics.get_xsec` -- carries
            ``photon_energy`` (eV) plus any of ``photo_absorption``,
            ``photodecay`` (all in cm^2).
        processes
            Subset of the process keys to draw.  Default: every process
            present (non-``None``) in ``xsecs``.
        layout
            ``"overlay"`` (default) draws every process on one axes;
            ``"subplots"`` draws one stacked panel per process sharing the
            energy axis.
        energy_unit
            Horizontal-axis unit: ``"eV"``, ``"erg"``, ``"nm"``, ``"um"``.
        xsec_unit
            Cross-section unit: ``"cm^2"``, ``"Mb"``, ``"barn"``.
        energy_log, xsec_log
            Log-scale the respective axis (default ``True``).
        trim
            Tighten the energy axis to the range where the cross section is
            positive (default ``True``).
        shade
            Shade the region under each curve.  ``True`` uses a default alpha;
            a float sets the alpha explicitly.  On a log y-axis the fill runs
            down to the bottom of the axis.
        show_bands
            Overlay the band-averaged cross section as bars (requires
            *bands*).  See :attr:`jaff.core.reaction.Reaction.band_xsecs`.
        bands
            Band-averaged cross-section table (``lower``/``upper``/``eavg`` in
            eV, ``xsec`` in cm²) used when *show_bands* is ``True``.
        title
            Axes/figure title.
        grid, show, save, filename
            Standard rendering controls.

        Returns
        -------
        tuple[Figure, Axes | numpy.ndarray]
            For ``layout="overlay"`` the second item is the single axes; for
            ``layout="subplots"`` it is the array of per-process axes.
        """
        if layout not in ("overlay", "subplots"):
            raise ValueError(f"layout must be 'overlay' or 'subplots', got {layout!r}")

        energy = xsecs["photon_energy"]
        if energy is None:
            raise ValueError("xsecs has no 'photon_energy' data to plot.")

        if processes is None:
            processes = [k for k in self._PROC_LABELS if xsecs.get(k) is not None]
        # Keep only requested processes that actually carry data.
        processes = [p for p in processes if xsecs.get(p) is not None]
        if not processes:
            raise ValueError("xsecs has no cross-section data to plot.")

        # (label, sigma_cm2) pairs in the requested process order.
        series = [(self._PROC_LABELS.get(k, k), np.asarray(xsecs[k])) for k in processes]
        df = _frames.xsec_frame(np.asarray(energy), series, energy_unit, xsec_unit)

        band_df = (
            _frames.band_frame(bands, energy_unit, xsec_unit)
            if show_bands and bands is not None
            else None
        )

        with self._theme_scope():
            if layout == "subplots":
                return self.__plot_subplots(
                    df,
                    energy_unit=energy_unit,
                    xsec_unit=xsec_unit,
                    energy_log=energy_log,
                    xsec_log=xsec_log,
                    trim=trim,
                    shade=shade,
                    grid=grid,
                    band_df=band_df,
                    title=title,
                    show=show,
                    save=save,
                    filename=filename,
                )
            return self.__plot_overlay(
                df,
                fig=fig,
                ax=ax,
                energy_unit=energy_unit,
                xsec_unit=xsec_unit,
                energy_log=energy_log,
                xsec_log=xsec_log,
                trim=trim,
                shade=shade,
                grid=grid,
                band_df=band_df,
                title=title,
                show=show,
                save=save,
                filename=filename,
            )

    # -- cross-section rendering helpers -----------------------------------

    def __draw_xsec(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        *,
        energy_unit: str,
        xsec_unit: str,
        energy_log: bool,
        xsec_log: bool,
        trim: bool,
        shade: bool | float,
        grid: bool,
        set_xlabel: bool,
        title: str,
    ) -> bool:
        """Draw the cross-section curves in *df* onto a single axes.

        Returns the effective log-x decision so a shared subplot x-axis can be
        scaled consistently.
        """
        labels = list(dict.fromkeys(df["process"]))
        multi = len(labels) > 1

        # Trim / dynamic-scale decisions from the positive data span.
        lo, hi = _xsec.positive_span(df)
        if lo is None:
            lo, hi = _xsec.finite_span(df)
        log_x = _xsec.use_log_x(energy_log, lo, hi)

        plot = so.Plot(df, x="energy", y="xsec")
        if multi:
            plot = plot.add(so.Line(), color="process").scale(
                color=so.Nominal(self._palette[: len(labels)], order=labels)
            )
        else:
            plot = plot.add(so.Line(color=self._palette[0]))

        scales: dict[str, str] = {}
        if log_x:
            scales["x"] = "log"
        if xsec_log:
            scales["y"] = "log"
        if scales:
            plot = plot.scale(**scales)

        plot = plot.label(
            x=_units.energy_label(energy_unit) if set_xlabel else "",
            y=_units.xsec_label(xsec_unit),
            title=title,
        )
        self._apply_plot_theme(plot).on(ax).plot()

        ax.grid(grid)
        despine(ax)
        if trim and lo is not None and hi is not None and hi > lo:
            ax.set_xlim(*_xsec.padded_limits(lo, hi, log_x))

        if shade:
            self.__shade(ax, df, labels, multi, alpha=shade)
        if not set_xlabel:
            ax.set_xlabel("")
        return log_x

    def __shade(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        labels: list[str],
        multi: bool,
        *,
        alpha: bool | float,
    ) -> None:
        """Fill the area under each process curve down to the axis bottom."""
        a = 0.18 if alpha is True else float(alpha)
        base = ax.get_ylim()[0]
        for i, label in enumerate(labels):
            sub = df[df["process"] == label]
            color = self._palette[i % len(self._palette)] if multi else self._palette[0]
            ax.fill_between(
                sub["energy"].to_numpy(),
                base,
                sub["xsec"].to_numpy(),
                color=color,
                alpha=a,
                linewidth=0,
            )

    def __draw_bands(self, ax: plt.Axes, band_df: pd.DataFrame) -> None:
        """Overlay band-averaged cross sections as bars, clipping open bands."""
        if band_df is None or band_df.empty:
            return
        xmin, xmax = ax.get_xlim()
        ybottom = ax.get_ylim()[0]
        lower = band_df["lower"].to_numpy(dtype=float)
        upper = band_df["upper"].to_numpy(dtype=float)
        # Clip an open (inf) or over-wide top edge to the visible axis range.
        upper = np.where(np.isfinite(upper), upper, xmax)
        upper = np.minimum(upper, xmax)
        width = np.clip(upper - lower, a_min=0.0, a_max=None)
        ax.bar(
            lower,
            band_df["xsec"].to_numpy(dtype=float),
            width=width,
            bottom=ybottom,
            align="edge",
            facecolor="none",
            edgecolor="#555555",
            linewidth=1.1,
            alpha=0.9,
            zorder=1.5,
            label="Band average",
        )

    def __plot_overlay(
        self,
        df: pd.DataFrame,
        *,
        fig: plt.Figure | None,
        ax: plt.Axes | None,
        band_df: pd.DataFrame | None,
        title: str,
        show: bool,
        save: bool,
        filename: str,
        **draw_kw: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Overlay layout: all processes on one axes."""
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        self.__draw_xsec(ax, df, set_xlabel=True, title=title, **draw_kw)
        self.__draw_bands(ax, band_df)
        self.__finish(fig, show, save, filename)
        return fig, ax

    def __plot_subplots(
        self,
        df: pd.DataFrame,
        *,
        band_df: pd.DataFrame | None,
        title: str,
        show: bool,
        save: bool,
        filename: str,
        **draw_kw: Any,
    ) -> tuple[plt.Figure, Any]:
        """Subplots layout: one stacked panel per process, shared energy axis."""
        labels = list(dict.fromkeys(df["process"]))
        n = len(labels)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(6.4, 2.6 * n), squeeze=False)
        axes = axes[:, 0]
        for i, (a, label) in enumerate(zip(axes, labels)):
            sub = df[df["process"] == label].reset_index(drop=True)
            self.__draw_xsec(
                a,
                sub,
                set_xlabel=(i == n - 1),  # only the bottom panel
                title=label,
                **draw_kw,
            )
            self.__draw_bands(a, band_df)
        if title:
            fig.suptitle(title)
        self.__finish(fig, show, save, filename)
        return fig, axes
