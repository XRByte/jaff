---
tags:
    - Api
    - Reaction
---

# from_serialized

`#!python from_serialized(serialized)`

Return **all** reactions sharing a name-level serialized form. Several reactions
can share one serialized form when they differ only by mechanism/`type` (e.g.
thermal vs cosmic-ray desorption), so a list is always returned. Use
[`all`](all.md) for the non-raising variant, or index with `(serialized, type)`
to pick one.

**Parameters**

**serialized** : _str_
: Canonical form `"<sorted_reactants>__<sorted_products>"`, e.g. `"H.H2O+__H2.OH+"`.

**Returns**

_list\[Reaction\]_
: Every reaction with that serialized form.

**Raises**

_KeyError_
: If no reaction has that serialized form.
