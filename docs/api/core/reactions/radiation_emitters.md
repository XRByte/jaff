---
tags:
    - Api
    - Reaction
---

# radiation_emitters

`#!python radiation_emitters()`

Returns every reaction that emits radiation, i.e. has a `.jfunc`
[`spectralYield<N>`](../../../user-guide/designing-networks/auxiliary-functions.md#spectral-yield-spectralyieldn)
(`reaction.rad.spectral_yield` is set), preserving their relative catalogue order. Any
reaction type can emit, not only photo reactions. The result is cached and recomputed
after the catalogue changes.

**Returns**

_Vector\[Reaction\]_
: All reactions with a spectral yield.
