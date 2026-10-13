---
tags:
    - Api
    - Plotting
---

# plot

`#!python plot(spec, *, save=None, show=True, close=True)`

Render a figure specification. The spec is a plain nested dict (or a
`FigureSpec`); it is validated before anything is drawn and a `ValueError`
names the offending path, e.g. `axes[1].series[0]: x and y lengths differ
(200 vs 199)`.

**Parameters**

**spec** : _dict or FigureSpec_
: The figure (schema below).

**save** : _str, pathlib.Path or None, optional_
: Output path; the format follows the suffix. Raster formats use the style's
  save dpi.

**show** : _bool, optional_
: Call `matplotlib.pyplot.show()`. Default `True`.

**close** : _bool, optional_
: After saving / showing, remove the figure from pyplot (`plt.close(fig)`) so
  repeated calls do not pile up open figures. The returned figure can still be
  edited and saved with `fig.savefig(...)`. Default `True`; pass `False` to
  keep it open in pyplot (e.g. for inline notebook display, or when you need
  the canvas itself, such as `fig.canvas.get_renderer()`). Ignored after an
  interactive, non-blocking show (`plt.isinteractive()`, e.g. IPython with a
  Qt/Tk or ipympl backend), so the window stays open.

**Returns**

_tuple\[Figure, numpy.ndarray\] or None_
: `(fig, axes)` with `axes` a 2-D array shaped like `grid`, or `None` if the
  spec holds no series.

## Schema

```python
{
  "grid": (2, 3),            # rows, cols (default (1, 1)); axes fill row-major
  "style": "paper",          # name or partial dict (see Themes)
  "palette": "okabe-ito",    # name, colour list, or {label: colour}
  "figsize": None,           # default: grid x style panel size, grown for side legends
  "sharex": False, "sharey": False,
  "legend": "auto",          # auto | axes | figure | none
  "legend_title": None,      # bold heading of the shared legend / the legend groups
  "title": None,             # figure title
  "axes": [
    {
      "series": [...],
      "xscale": "log", "yscale": "log",    # "linear" (default) | "log"
      "xlim": None, "ylim": None,
      "xlabel": "", "ylabel": "",
      "title": "",           # panel name, drawn inside (upper right)
      "tag": None,           # extra inside label, under the title
      "legend": None,        # None: figure policy; "axes" (own group) | "none"
    },
  ],
}
```

Series:

```python
{
  "x": array, "y": array,    # 1-D, same length
  "label": "",
  "color": None,             # None: palette (label map, then cycle)
  "style": None,             # linestyle ("-", (0, (6, 3))), marker ("o", "X", "D") or both ("-o")
  "role": "data",            # "reference": thick dark-grey line beneath the rest
  "lw": None, "zorder": None,
  "kind": "line",            # "bars": horizontal bars, needs "x_lo" and "x_hi"
}
```

Non-finite points are dropped, as are non-positive values on log axes; a series
left with fewer than two points (bars: none) is skipped with a warning.

**Layout notes**

- Tight grid: the gap between panels is `style.gap` inches in both
  directions; margins are sized to the outer tick and axis labels. Rows and
  columns of the grid with no used panel take no space (8 panels in a 4×4
  grid are laid out as 2×4); the returned `axes` keep the requested shape,
  with those cells hidden. Hidden cells inside used rows stay (legend host). Without
  `figsize` each panel box is `style.panel_size`; with `figsize` the figure
  is kept as given and the panels take what is left.
- X tick labels by place in the column (hidden cells don't count): the
  topmost panel puts its x tick labels (and x label) on the top spine; the
  bottommost panel (also one above a hidden cell, or the only one) at the
  bottom; a middle panel shows none when its x range equals the bottommost
  panel's, else draws them inside above the bottom spine, with its y range
  widened downward so no curve runs under them (with an explicit `ylim` they
  stay outside instead). Ticks stay on all four sides.
- Axis labels shown by every panel are drawn once: a common y label over two
  or more rows becomes `fig.supylabel`, a common x label over two or more
  columns becomes `fig.supxlabel`. Otherwise a middle panel drops an x label
  its column's bottom panel repeats, and a y label is dropped where the panel
  to the left repeats it.
- `title` is drawn inside the panel (upper right) with headroom above the
  data, or above the panel when wider than ~85 % of it; `tag` sits inside too.
- Legends: one shared legend (`"auto"`), or per-panel groups in the right
  margin (`"axes"`); see [the overview](index.md#legends). A legend with more
  entries than the style's `legend_max` (default 12) is not drawn; a warning
  names the axes (or `figure`).
- With exponent ticks (`"paper"`), log axes get at most ~4 labelled decades
  (every 1, 2, 3, 4, 5, 10, … decades) and unlabelled minor ticks on decades
  (every decade for steps up to 5, thinned to at most 4 between labels for
  larger steps, e.g. every 2 for a step of 10); an axis spanning less than 1.5 decades gets ticks at fractional
  exponents (`1.05`, `1.1`, ...) and no minor ticks. These are fixed
  positions chosen from the final limits: changing the limits of the returned
  axes does not re-tick them. An axis too narrow for distinct labels keeps its
  decade ticks.
- With inside tick labels (`"paper"`), the x range is widened on the left by
  the widest y label (plus the tick length); a panel with an explicit `xlim`
  keeps its y tick labels outside instead.
- Headroom for inside titles is applied once per shared y range (`sharey`),
  using the largest data span of the panels sharing it.
- A shared legend over `legend_max` entries that keys several panels falls
  back to per-panel groups (`"axes"`), with one warning.
- Without `figsize`, the figure is made tall enough for the right-margin
  legends.

**Validation notes**

- With `sharex` / `sharey`, all axes must use the same `xscale` / `yscale`;
  otherwise a `ValueError` is raised.
- `color` must be a string; `style` must be a string or a tuple.
- Text fields (`label`, `title`, `xlabel`, ...) given as `None` count as empty.
- Style and palette names are checked when the spec is built; errors are
  prefixed `figure.style` / `figure.palette`.

## Example

```python
import numpy as np
from jaff.plotting import plot

nh = np.logspace(17, 22, 300)
panel = lambda name, ref, sim, rt_off: {
    "title": name, "xscale": "log", "yscale": "log",
    "xlabel": r"$N_{\rm H}$ (cm$^{-2}$)", "ylabel": r"$n_i/n_{\rm H}$",
    "series": [
        {"x": nh, "y": ref, "label": "Gong+2017", "role": "reference"},
        {"x": nh, "y": sim, "label": "simulation"},
        {"x": nh, "y": rt_off, "label": "RT off", "style": "o", "color": "k"},
    ],
}
# 2 x 2 grid, 3 panels: the shared legend fills the free bottom-right cell
plot({"grid": (2, 2), "sharex": True,
      "axes": [panel("CO", *co), panel("C", *c), panel("O", *o)]})
```
