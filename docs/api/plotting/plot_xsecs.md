---
tags:
    - Api
    - Plotting
---

# plot_xsecs

`#!python plot_xsecs(reactions, *, processes="all", layout="overlay", energy_unit="eV", xsec_unit="Mb", energy_log=True, xsecs_log=True, trim=True, shade=False, show_bands=False, palette=None, title=None, fig=None, ax=None, grid=True, show=True, save=False, filename="")`

Plots photo cross sections for one or more reactions on shared axes. Handles
unit conversion, log scaling, axis trimming, optional shading, and optional
band-averaged bars. Reactions without cross-section data are skipped with a log
message; returns `None` if nothing can be drawn.

**Parameters**

**reactions** : _reaction-like or list of reaction-like_
: A single reaction (anything exposing `xsecs_dict` and `band_xsecs`) or a list of them. Their cross sections are overlaid.

**processes** : _str or list\[str\] or None, optional_
: Which processes to draw. `"all"` (default) or `None` plots every process with data; a single key or list selects a subset. Valid keys: `"photo_absorption"`, `"photodecay"`. An invalid key raises `KeyError`.

**layout** : _str, optional_
: `"overlay"` (default) draws all curves on one axes; `"subplots"` gives each curve its own stacked panel.

**energy_unit** : _str, optional_
: Horizontal-axis unit: `"eV"` (default), `"erg"`, `"nm"`, or `"um"`.

**xsec_unit** : _str, optional_
: Cross-section unit: `"Mb"` (default), `"cm^2"`, or `"barn"`.

**energy_log, xsecs_log** : _bool, optional_
: Log-scale the energy / cross-section axis. Default `True`.

**trim** : _bool, optional_
: Tighten the energy axis to the positive data span. Default `True`.

**shade** : _bool or float, optional_
: Shade the area under each curve. `True` uses a default alpha; a float sets the alpha explicitly. Default `False`.

**show_bands** : _bool, optional_
: Overlay the band-averaged cross section as bars for every reaction that has them (see [`Reaction.band_xsecs`](../core/reaction/band_xsecs.md)). Default `False`.

**palette** : _list\[str\] or None, optional_
: Colour-cycle override.

**title** : _str or None, optional_
: Plot title. Defaults to the LaTeX equation for a single reaction, else empty.

**fig, ax** : _matplotlib.figure.Figure / matplotlib.axes.Axes or None, optional_
: Existing figure/axes to draw on (overlay only). Created if `None`.

**grid, show, save, filename** : _optional_
: Standard rendering controls.

**Returns**

_tuple\[matplotlib.figure.Figure, matplotlib.axes.Axes\] or None_
: The figure and axes (overlay) or array of axes (subplots); `None` when no reaction has cross-section data.

!!! note
    With a single reaction the legend uses the process names; with several it
    prefixes each with the reaction so curves stay distinguishable.

**Examples**

```python
from jaff.plotting import plot_xsecs

photo = net.reactions.photo_reactions()[0]
plot_xsecs(photo)                                   # one reaction, all processes
plot_xsecs(photo, shade=True, show_bands=True)      # shading + band bars
plot_xsecs(photo, energy_unit="nm", xsec_unit="cm^2")
plot_xsecs(net.reactions.photo_reactions())         # overlay several reactions
```
