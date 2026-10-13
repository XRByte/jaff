---
tags:
    - Api
    - Plotting
---

# plot_xsecs

`#!python plot_xsecs(reactions, *, grid=None, processes="all", energy_unit="eV", xsec_unit="Mb", xscale="log", yscale="log", bands=False, style=None, palette=None, sharex=False, sharey=False, legend="auto", title=None, save=None, show=True, close=True)`

Plot photo cross sections against photon energy (or wavelength).

**Parameters**

**reactions** : _reaction-like or list_
: Anything exposing `xsecs_dict` (and optionally `band_xsecs` / `get_latex`).
  Nesting and `grid` work as in [`plot_rates`](plot_rates.md), including the
  automatic split past `legend_max` curves; the split keeps all curves of one
  reaction on the same panel.

**processes** : _str, list\[str\] or None, optional_
: `"all"` (default), `"photo_absorption"`, `"photodecay"`, or a list. An
  unknown key raises `KeyError`.

**energy_unit** : _str, optional_
: `"eV"` (default), `"erg"`, `"nm"` or `"um"`.

**xsec_unit** : _str, optional_
: `"Mb"` (default), `"barn"` or `"cm^2"`.

**xscale, yscale** : _str or list\[str\], optional_
: Per figure or per panel.

**bands** : _bool, optional_
: Overlay band-average cross sections (from `Reaction.band_xsecs`) as
  horizontal bars; reactions without bands add nothing. An open band stays
  open in any unit: in wavelength units the open ionising band extends to the
  short-wavelength edge of the axes.

**style, palette, sharex, sharey, legend, title, save, show, close** : _optional_
: As in [`plot_rates`](plot_rates.md).

**Returns**

_tuple\[Figure, numpy.ndarray\] or None_
: `None` when no reaction has cross-section data.

**Notes**

A panel holding one reaction is titled with its equation (inside the panel,
or above it when too long) and its curves are labelled by process. Several
reactions on one panel are labelled `equation — process`. One shared legend
keys all panels, with one colour per process across the figure.

**Example**

```python
from jaff.plotting import plot_xsecs

photo = [p for p in net.reactions.photo_reactions() if p.xsecs_dict]
plot_xsecs(photo[0], bands=True)
plot_xsecs(photo[:4], grid=(2, 2), processes="photodecay", energy_unit="nm",
           xscale="linear")
```
