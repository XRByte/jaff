---
tags:
    - Api
icon: phosphor/chart-line
---

# jaff.plotting

Publication-style plotting in pure matplotlib. A figure is described as
**data**: a `grid` of axes, each holding a list of series. Themes supply the
look; every series can override it.

## Functions

| Function                          | Description                                                                 |
| --------------------------------- | --------------------------------------------------------------------------- |
| [`plot`](plot.md)                 | Render a figure-spec dict (any data: rates, cross sections, statistics ...) |
| [`plot_rates`](plot_rates.md)     | Rate coefficients vs temperature, one panel per item or overlaid            |
| [`plot_xsecs`](plot_xsecs.md)     | Photo cross sections vs photon energy, with optional band-average bars      |
| `render`                          | `FigureSpec` → `(fig, axes)` without saving or showing                      |
| `register_style`, `register_palette` | Add named themes (see [Themes](themes.md))                               |

## Quick start

```python
from jaff import Network
from jaff.plotting import plot, plot_rates, plot_xsecs

net = Network("networks/uclchem_small_gas/uclchem_small_gas_network.jet")
r = list(net.reactions)

plot_rates(r[:3])                                 # one axes, three curves
plot_rates([r[0], [r[1], r[2]]], grid=(1, 2))      # nested list = overlay
plot_rates(r[:4], grid=(2, 2), style="talk", palette="sunset")

photo = [p for p in net.reactions.photo_reactions() if p.xsecs_dict]
plot_xsecs(photo[:4], grid=(2, 2), bands=True, save="xsecs.pdf")
```

Nesting rule: each top-level item is one panel; a nested list is overlaid on
one axes. Without `grid` everything shares a single axes, unless there are
more curves than the style's `legend_max` (default 12): then the curves are
split in order over an automatic grid. Panels fill the grid row by row;
leftover cells are hidden (the bottom-right one hosts the shared legend). An
axis label common to every panel is drawn once for the whole figure
(`supxlabel` / `supylabel`) when the panels span two or more columns / rows.

Figures are closed in pyplot after saving / showing (`close=True`), so loops
don't accumulate open figures; the returned `(fig, axes)` still work. Pass
`close=False` to keep the figure open. After an interactive, non-blocking show
(`plt.isinteractive()`) the figure is left open.

The `Reaction.plot_rate_coefficient`, `Reaction.plot_xsecs`,
`Reactions.plot_rates` and `Reactions.plot_xsecs` methods forward their keyword
arguments to these functions.

## Look

White surface, black spines on all four sides, inward major and minor ticks on
all sides, DejaVu Sans with STIX maths. Panels sit on a tight grid: the same
gap (`style.gap`, inches) between columns and rows, margins sized to the
outer labels. In each column the top panel carries its x tick labels on the
top spine and the bottom panel at the bottom; a middle panel shows none when
it shares the bottom panel's x range, else small labels inside above its
bottom spine. Rows and columns without a used panel take no space. Each
panel box is `style.panel_size`, unless `figsize` fixes the figure.

In the default `"paper"` style:

- y tick labels are drawn **inside** the panel, right of the left spine (labels
  at the top/bottom edge are blanked, and the x range gets an empty strip on
  the left so no curve runs under a label);
- panel **titles** sit inside, upper right, with headroom above the data (a
  title too wide for the panel goes above it);
- log axes are labelled by exponent (`−3` for 10⁻³) with at most ~4 labelled
  decades, minor ticks on decades (at most 4 between labels); axes under 1.5 decades get fractional
  exponents (`1.05`); the axis label gains a `log₁₀` prefix.

Marker series are hollow markers spaced evenly along the curve on screen; a
line+marker style (`"-o"`, `"--D"`) adds sparse markers to a line. A series
with `"role": "reference"` is a thick dark-grey line drawn beneath the rest.
See [Themes](themes.md).

## Legends

The default `legend="auto"` (same as `"figure"`) draws **one shared legend**
for the whole figure, de-duplicated by label, in the bottom-right hidden grid
cell when the grid is unfinished, else in the right margin. A label keeps one
colour and style across every panel, so the shared legend keys them all.

`legend="axes"` (opt-in, for panels with different labels) draws one legend
**group per panel**, stacked in the right margin and headed by the panel's
title, tag or letter `(a)`, `(b)`, … (an untitled panel shows its letter
inside). `"none"` draws no legend; an axes-level `legend` (`"axes"` /
`"none"`) overrides the figure policy for that panel.

A legend with more than the style's `legend_max` (default 12) entries is not
drawn (with a warning); a shared legend over the cap that keys several panels
falls back to per-panel groups. `legend_title="Source"` adds a bold heading to
the shared legend (or above the stack of groups). Panel names are explicit: use the axes `title` or
`tag`; a lone series is not turned into a tag.
