---
tags:
    - Api
    - Reaction
---

# with_type

`#!python with_type(type)`

Filters the catalogue and returns every reaction whose type label matches *type*, preserving their relative catalogue order.

**Parameters**

**type** : _str_
: Reaction-type label to match, e.g. `"photo"`, `"cosmic_ray"`. Must be one of the type strings stored on `Reaction.type`.

**Returns**

_Vector\[Reaction\]_
: All reactions of the specified type.
