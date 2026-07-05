---
tags:
    - Api
    - Plotting
---

# Plotter

`jaff.plotting.Plotter`

Low-level renderer behind [`plot_rates`](plot_rates.md) and
[`plot_xsecs`](plot_xsecs.md). Most users should call those free functions;
`Plotter` is the styling + rendering engine they delegate to.

## Constructor

`#!python Plotter(palette=None, global_theme=False, **rc_overrides)`

**palette** : _list\[str\] or None, optional_
: Colour cycle for curves. Defaults to `LOGO_PALETTE` (the JAFF brand palette). Pass `MUTED_PALETTE` / `DEEP_PALETTE` for the seaborn cycles.

**global_theme** : _bool, optional_
: If `True`, apply the house theme globally and persistently on construction (mutating `matplotlib.rcParams`). If `False` (default), the theme is scoped to each plot call, leaving global state untouched.

**\*\*rc_overrides**
: Individual theme entries to override, forwarded to `theme_rc` (e.g. `style="whitegrid"`, `context="talk"`, or any `rcParams` key).

## Methods

### render_series

`#!python render_series(df, *, x_col, y_col, group_col, xlabel, ylabel, log_x, log_y, dynamic_x=False, trim=False, shade=False, bands_df=None, layout="overlay", title="", fig=None, ax=None, grid=True, show=True, save=False, filename="plot.png")`

The single home for multi-curve rendering. Takes a tidy long `DataFrame` (one
group per curve) and handles log scales, dynamic-log downgrade (`dynamic_x`),
axis trimming (`trim`), shaded fills (`shade`), band bars (`bands_df`),
per-curve line-width variation, the legend, and the house theme. Both free
functions build a frame and call this.

Returns `#!python (fig, ax)` for `layout="overlay"`, or `#!python (fig, axes)`
(an array of axes) for `layout="subplots"`.

### plot

`#!python plot(x, y, fig=None, ax=None, xlabel="", ylabel="", xscale="linear", yscale="linear", title="", label="", grid=True, show=True, save=False, filename="plot.png", **line_kw)`

Generic single-line plot. Retained for backward compatibility;
[`plot_rates`](plot_rates.md) is preferred for one or many curves.

### plot_xsec

`#!python plot_xsec(xsecs, processes=None, layout="overlay", fig=None, ax=None, energy_unit="eV", xsec_unit="cm^2", energy_log=True, xsec_log=True, trim=True, shade=False, show_bands=False, bands=None, title="", grid=True, show=True, save=False, filename="xsec.png")`

Plots a single `XsecsProps` mapping. Retained for backward compatibility;
[`plot_xsecs`](plot_xsecs.md) is preferred (it accepts one or many reactions).

## Example

```python
from jaff.plotting import Plotter, MUTED_PALETTE

# Bespoke styling; then compose onto the returned axes.
p = Plotter(palette=MUTED_PALETTE, style="whitegrid", context="talk")
fig, ax = p.plot([1, 2, 3], [4, 5, 6], xscale="log", yscale="log", show=False)
ax.set_title("custom")
```
