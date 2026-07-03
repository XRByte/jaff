---
tags:
    - Api
    - Species
---

# normalized_names

`#!python normalized_names(pos='p', neg='n')`

Returns a normalized identifier string for every species in the collection, in catalogue order. Each name is lowercased and has `"+"` replaced by `pos` and `"-"` replaced by `neg`, producing strings that are valid variable names in C, Fortran, and Python. With the defaults, `"HCO+"` becomes `"hcop"` and `"e-"` becomes `"en"`.

**Parameters**

_pos_ : `str`, optional
: Replacement for `"+"`, by default `"p"`.

_neg_ : `str`, optional
: Replacement for `"-"`, by default `"n"`.

**Returns**

_Vector\[str\]_
: Normalized, code-safe identifier string for each species, in catalogue order.
