---
tags:
    - Api
    - Reaction
---

# Reaction.band_xsecs

`#!python band_xsecs` _(property)_

Band-averaged cross sections for this reaction, one row per radiation band, as a
tidy `pandas.DataFrame`. Assembled from the `rad_groups` back-references, which
are populated when a radiation field is configured on the network (via the
`rad_bands` argument to `Network`). Intended as the data source for band bar
plots (see the `show_bands` option of [`plot_xsecs`](plot_xsecs.md)).

**Returns**

_pandas.DataFrame_
: One row per band this reaction contributes to, with columns:

| Column      | Unit | Description                                                                                     |
| ----------- | ---- | ----------------------------------------------------------------------------------------------- |
| `lower`     | eV   | Lower band edge                                                                                 |
| `upper`     | eV   | Upper band edge (`inf` for an open top band)                                                     |
| `eavg`      | eV   | Photon-number-weighted band-average energy                                                        |
| `xsec`      | cm²  | Photon-number-weighted band-average cross section (`NaN` for custom-rate reactions)              |
| `xsec_frac` | —    | Fraction of the total cross section (or `dRad`) attributed to the band                            |

: The frame is empty (with the columns above) when the reaction contributes to
no band, e.g. no radiation field is configured. Rows are ordered by ascending
band index.

**Example**

```python
net = Network("networks/h_photoionization/h_photo.jet", rad_bands=[1, 13.6, 100, "inf"])
rxn = net.reactions.photo_reactions()[0]
rxn.band_xsecs          # DataFrame: lower / upper / eavg / xsec / xsec_frac
rxn.plot_xsecs(show_bands=True)   # overlay the band-averaged bars on σ(E)
```
