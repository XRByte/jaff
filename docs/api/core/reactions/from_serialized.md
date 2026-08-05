---
tags:
    - Api
    - Reaction
---

# from_serialized

`#!python from_serialized(serialized)`

Return **all** reactions sharing a name-level serialized form, always as a list.
Several reactions can share one serialized form when they differ only by
mechanism/`type` (e.g. thermal vs cosmic-ray desorption). For scalar-or-list
ergonomics, index the catalogue directly (`reactions[serialized]`); for a
non-raising `Vector`, use [`all`](all.md); to pick one, index with
`(serialized, type)`.

**Parameters**

**serialized** : _str_
: Canonical form `"<sorted_reactants>__<sorted_products>"`, e.g. `"H.H2O+__H2.OH+"`.

**Returns**

_list\[Reaction\]_
: Every reaction with that serialized form.

**Raises**

_KeyError_
: If no reaction has that serialized form.
