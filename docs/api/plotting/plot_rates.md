---
tags:
    - Api
    - Plotting
---

# plot_rates

`#!python plot_rates(rates, *, grid=None, tmin=None, tmax=None, var="tgas", npoints=200, xscale="log", yscale="log", xlabel=r"$T$ (K)", ylabel=r"$\kappa$", labels=None, style=None, palette=None, sharex=False, sharey=False, legend="auto", title=None, save=None, show=True, close=True)`

Plot rate coefficients against gas temperature.

**Parameters**

**rates** : _reaction-like, sympy.Basic, (x, y) tuple, or list of these_
: Each leaf is a reaction (anything with `rate_coefficient`, optionally `tmin` / `tmax` /
  `get_latex`), a SymPy expression (needs `tmin` and `tmax`), or an `(x, y)`
  tuple used verbatim. With `grid`, each top-level item is one panel and a
  nested list is overlaid on one panel; without `grid` everything is overlaid.

**grid** : _tuple\[int, int\] or None, optional_
: `(rows, cols)`; panels fill row-major. Without `grid`, more curves than the
  style's `legend_max` (default 12) are split in order into panels of at most
  `legend_max` curves on an automatic grid: up to three panels in one row,
  else `cols = ceil(sqrt(n))`, `rows = ceil(n / cols)`. An explicit grid is
  never split.

**tmin, tmax** : _float or None, optional_
: Temperature range (K). Reactions fall back to their own bounds, then to
  `2.73` / `1e6`. The range must satisfy `0 < tmin < tmax`; otherwise a
  `ValueError` is raised.

**var** : _str, optional_
: Temperature symbol substituted in reaction rates. Default `"tgas"`.

**npoints** : _int, optional_
: Log-spaced samples per curve. Default `200`.

**xscale, yscale** : _str or list\[str\], optional_
: `"log"` / `"linear"` for every panel, or one entry per panel (a list
  needs an explicit `grid` once the curves are split automatically).

**xlabel, ylabel** : _str, optional_
: Axis labels; the rate axis defaults to `$\kappa$` (no units), read as
  "log₁₀ κ" with the `"paper"` exponent ticks. A label shared by all panels is drawn once for the figure
  (`supxlabel` over two or more columns, `supylabel` over two or more rows);
  otherwise it is kept only on the outer panels of the grid.

**labels** : _list\[str\] or None, optional_
: Legend labels, one per leaf in input order. Default: LaTeX equations.

**style, palette, sharex, sharey, legend, title** : _optional_
: Figure options; see [`plot`](plot.md) and [Themes](themes.md).

**save, show, close** : _optional_
: As in [`plot`](plot.md).

**Returns**

_tuple\[Figure, numpy.ndarray\] or None_
: `None` when no rate could be evaluated.

**Notes**

Series with fewer than two plottable points on the chosen scales (e.g. a zero
rate on a log axis) are dropped before panels are assembled, with one summary
warning, so the automatic split never produces blank panels. With the split,
the legend switches to per-panel groups in the right margin
(`legend="axes"`); an explicit `legend="figure"` does too, with a warning.
Otherwise one shared legend keys every panel.

A panel holding a single curve shows its equation as the panel title (inside
the panel, or above it when too long). Rates
that cannot be evaluated against temperature alone (photo, cosmic-ray,
interpolated-function rates) are skipped with one summary warning (the
per-reaction reason is logged at debug level); a panel left empty is
dropped.

**Example**

```python
from jaff.plotting import plot_rates

r = list(net.reactions)
plot_rates(r[:5], tmin=10, tmax=1e4)                    # overlay
plot_rates(r[:30])                                      # 3 panels of 12/12/6
plot_rates(r[:20], style={"legend_max": 20})            # one panel, 20 curves
plot_rates([r[0], r[1:4]], grid=(1, 2), yscale=["log", "linear"])
```
