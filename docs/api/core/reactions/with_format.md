---
tags:
    - Api
    - Reaction
---

# with_format

`#!python with_format(format)`

Filters the catalogue and returns every reaction read by the parser named *format*, preserving their relative catalogue order. The result is cached per *format* and recomputed after the catalogue changes.

Reactions loaded from a `.jaff` file written before the format was stored have no format and never match.

**Parameters**

**format** : _str_
: Parser name to match, e.g. `"kida"`, `"krome"`, `"prizmo"`, `"udfa"`, `"uclchem"`. Compared against the reaction's private `_format` field.

**Returns**

_Vector\[Reaction\]_
: All reactions produced by that parser.
