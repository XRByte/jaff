---
tags:
    - Api
    - Species
---

# normalized_names

`#!python normalized_names(pos='j', neg='k')`

Returns a normalized identifier string for every species in the collection, in catalogue order. Each name is lowercased and has `"+"` replaced by `pos` and `"-"` replaced by `neg`, producing strings that are valid variable names in C, Fortran, and Python. With the defaults, `"HCO+"` becomes `"hcoj"` and `"e-"` becomes `"ek"`. The `j`/`k` choice matches the convention used by `get_fidx` and avoids the charge-marker-vs-element-letter collision (e.g. `Sn` tin vs `S-`).

**Parameters**

_pos_ : `str`, optional
: Replacement for `"+"`, by default `"j"`.

_neg_ : `str`, optional
: Replacement for `"-"`, by default `"k"`.

**Returns**

_Vector\[str\]_
: Normalized, code-safe identifier string for each species, in catalogue order.
