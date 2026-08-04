---
tags:
    - Api
    - Reaction
---

# add

`#!python add(reaction)`

Appends a `Reaction` to the end of the catalogue, updating the internal name-index and serialized-key index so that the new reaction is immediately accessible by verbatim string or serialized key. Duplicate checking is not performed here; duplicates over the same temperature range are resolved during network construction by the network's `duplicate_policy`.

**Parameters**

**reaction** : _Reaction_
: The `Reaction` instance to append to the catalogue.

**Raises**

_ValueError_
: If *reaction* is not a `Reaction` instance.
