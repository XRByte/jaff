---
tags:
    - Api
icon: phosphor/chart-line
---

# jaff.plotting

Publication-quality plotting for reaction rates and photo cross sections, built
on the [seaborn objects](https://seaborn.pydata.org/tutorial/objects_interface.html)
interface. The preferred entry points are the free functions `plot_rates` and
`plot_xsecs`, which accept a single item **or a list** and overlay them on
shared axes.

## Functions

| Function                            | Description                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| [`plot_rates`](plot_rates.md)       | Plot one or more rate coefficients (reactions, SymPy expressions, or `(x, y)` arrays)         |
| [`plot_xsecs`](plot_xsecs.md)       | Plot photo cross sections for one or more reactions, with optional shading and band bars       |
| `apply_global_theme`                | Apply the house theme globally and persistently for the session                                |

## Classes

| Class                     | Description                                                            |
| ------------------------- | --------------------------------------------------------------------- |
| [`Plotter`](plotter.md)   | Low-level renderer; `render_series` is the shared multi-curve backend |

## Quick start

```python
from jaff import Network
from jaff.plotting import plot_rates, plot_xsecs

net = Network("networks/h_photoionization/h_photo.jet", rad_bands=[1, 13.6, 100, "inf"])

# One reaction, or many on shared axes with a legend.
plot_rates(net.reactions[0])
plot_rates(list(net.reactions))              # overlay all rates
net.reactions.plot_rates()                   # equivalent, via the catalogue

# Cross sections, with shading and band-averaged bars.
photo = net.reactions.photo_reactions()[0]
plot_xsecs(photo, shade=True, show_bands=True)
plot_xsecs(net.reactions.photo_reactions())  # overlay several reactions
```

The `Reaction.plot_rate_coefficient` / `Reaction.plot_xsecs` methods and the
`Reactions.plot_rates` / `Reactions.plot_xsecs` catalogue methods are thin
wrappers over these functions.

## Theming

The house theme is built from seaborn's own style machinery
(`seaborn.axes_style` + `seaborn.plotting_context`). By default it is applied
*scoped* — only while a figure is being drawn — so importing or using the
plotter never mutates global matplotlib state. Opt into a sticky, session-wide
theme with `apply_global_theme()`.

The default style is `"darkgrid"` with the JAFF brand palette. Three palettes
are exported:

| Palette         | Description                                            |
| --------------- | ------------------------------------------------------ |
| `LOGO_PALETTE`  | JAFF brand colours (default): purple, magenta, coral, amber |
| `MUTED_PALETTE` | seaborn "muted" (10 colours; use for many curves)      |
| `DEEP_PALETTE`  | seaborn "deep" (10 colours)                            |

```python
from jaff.plotting import Plotter, MUTED_PALETTE, apply_global_theme

Plotter(palette=MUTED_PALETTE, style="whitegrid")   # per-plotter override
apply_global_theme(palette=MUTED_PALETTE)            # sticky, whole session
```

## Design

Both free functions build a tidy long `DataFrame` and delegate to
[`Plotter.render_series`](plotter.md), the single home for multi-curve
rendering (log scales, axis trimming, shaded fills, band bars, per-curve
line-width variation, and legends). The plotting package imports only
`numpy` / `pandas` / `sympy` / `seaborn` — never `jaff.core` — so it stays a
leaf dependency and can also plot bare SymPy expressions and arrays.
