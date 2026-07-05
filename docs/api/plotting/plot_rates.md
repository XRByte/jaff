---
tags:
    - Api
    - Plotting
---

# plot_rates

`#!python plot_rates(rates, *, tmin=None, tmax=None, var="tgas", npoints=100, labels=None, palette=None, xlabel="Temperature (K)", ylabel=r"Rate coefficient $k$", xscale="log", yscale="log", shade=False, title="", fig=None, ax=None, grid=True, show=True, save=False, filename="rates.png")`

Plots one or more rate coefficients on shared axes with a legend. Each curve is
coerced to `(x, y, label)` and drawn with a distinct line width (thinner curves
in front). Inputs are duck-typed and dispatched per item, so a single call may
mix reactions, expressions, and arrays.

**Parameters**

**rates** : _reaction-like, sympy.Basic, (x, y) tuple, or list of these_
: What to plot. Each item is one of:

    - a **reaction-like** object (anything exposing `rate`, and usually `tmin` / `tmax` / `get_latex`): its rate is evaluated over its temperature range on a log grid;
    - a **SymPy expression**: evaluated over `[tmin, tmax]` (both required);
    - an **`(x, y)`** pair of arrays: used verbatim.

**tmin, tmax** : _float or None, optional_
: Temperature range (K). For reactions, falls back to each reaction's own bounds, then to `2.73` / `1e6`. Required for bare SymPy expressions.

**var** : _str, optional_
: Symbol name to substitute when evaluating reaction rates. Default `"tgas"`.

**npoints** : _int, optional_
: Number of log-spaced sample points. Default `100`.

**labels** : _list\[str\] or None, optional_
: Legend labels aligned to `rates`. Defaults to each reaction's LaTeX equation (or `str`).

**palette** : _list\[str\] or None, optional_
: Colour-cycle override (e.g. `MUTED_PALETTE` for many curves).

**xlabel, ylabel** : _str, optional_
: Axis labels. Default `"Temperature (K)"` and `Rate coefficient $k$`.

**xscale, yscale** : _str, optional_
: `"log"` (default) or `"linear"`.

**shade** : _bool or float, optional_
: Shade the area under each curve. Default `False`.

**title** : _str, optional_
: Plot title. Default `""`.

**fig, ax** : _matplotlib.figure.Figure / matplotlib.axes.Axes or None, optional_
: Existing figure/axes to draw on. Created if `None`.

**grid, show, save, filename** : _optional_
: Standard rendering controls. `filename` default `"rates.png"`.

**Returns**

_tuple\[matplotlib.figure.Figure, matplotlib.axes.Axes\] or None_
: The figure and axes, or `None` when no input could be evaluated.

!!! note
    Photo-reaction rates carry a symbolic radiation-density variable and cannot
    be evaluated as a function of temperature; such inputs are skipped with a
    warning.

**Examples**

```python
from jaff.plotting import plot_rates
import sympy as sp

plot_rates(net.reactions[0])                     # single reaction
plot_rates(list(net.reactions))                  # overlay all, legend by equation
plot_rates([r1, r2], tmax=1e4, shade=True)       # shared bounds + shading

t = sp.Symbol("tgas")
plot_rates([1e-10 * (t / 300) ** 0.5], tmin=10, tmax=1e4, labels=["my rate"])
```
