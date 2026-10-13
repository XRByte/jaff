---
tags:
    - Api
    - Reaction
---

# Reaction.plot_rate_coefficient

`#!python plot_rate_coefficient(**kwargs)`

Plots the rate coefficient against gas temperature. Forwards to
[`jaff.plotting.plot_rates`](../../plotting/plot_rates.md) with this reaction
as the only item, so every `plot_rates` keyword (`tmin`, `tmax`, `xscale`,
`style`, `palette`, `save`, `show`, `close`, ...) is accepted. The panel is titled with
the reaction equation (inside the panel in the `"paper"` style, or above it
when too long). The temperature axis spans \[`tmin`, `tmax`\],
defaulting to the reaction's bounds and then 2.73 K / 1e6 K.

**Returns**

_tuple\[matplotlib.figure.Figure, numpy.ndarray\] or None_
: `(fig, axes)`, or `None` if the rate cannot be evaluated numerically (e.g.
  a photo reaction, whose rate carries the symbolic radiation-density
  variable).

**Example**

```python
fig, axes = net.reactions[0].plot_rate_coefficient(tmin=10, tmax=1e4,
                                                   save="rate.pdf", show=False)
```
