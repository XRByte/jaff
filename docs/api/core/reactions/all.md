---
tags:
    - Api
    - Reaction
---

# all

`#!python all(serialized)`

Return all reactions sharing a name-level serialized form. Unlike
[`from_serialized`](from_serialized.md), a missing key yields an empty `Vector`
rather than raising.

**Parameters**

**serialized** : _str_
: Canonical form `"<sorted_reactants>__<sorted_products>"`, e.g. `"H.H2O+__H2.OH+"`.

**Returns**

_Vector\[Reaction\]_
: Every reaction with that serialized form, or an empty `Vector` if none.
