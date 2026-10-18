---
tags:
    - Api
    - Reaction
---

# Reaction.plot_xsecs

`#!python plot_xsecs(**kwargs)`

Plots photo cross sections against photon energy or wavelength. Forwards to
[`jaff.plotting.plot_xsecs`](../../plotting/plot_xsecs.md) with this reaction
as the only item, so every `plot_xsecs` keyword (`processes`, `energy_unit`,
`xsec_unit`, `xscale`, `yscale`, `bands`, `style`, `palette`, `save`, `show`, `close`,
...) is accepted. The panel is titled with the reaction equation (inside, or
above when too long) and the curves are labelled by process in one shared
legend in the right margin.

**Returns**

_tuple\[matplotlib.figure.Figure, numpy.ndarray\] or None_
: `(fig, axes)`, or `None` (with a log message) when `photo` (or its `xsec`) is `None`
  or no requested process has data.

**Example**

```python
photo = net.reactions.photo_reactions()[0]
photo.plot_xsecs()                                   # all processes, eV vs Mb
photo.plot_xsecs(processes="photodecay", yscale="linear")
photo.plot_xsecs(energy_unit="nm", xsec_unit="cm^2", bands=True)
```
