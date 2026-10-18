---
tags:
    - Api
    - Reaction
---

# rate_coefficients

`#!python rate_coefficients()`

Returns the symbolic rate-coefficient expression (`Reaction.rate_coefficient`) for every reaction in the catalogue, in catalogue order. Units depend on reaction order (e.g. cm³ s⁻¹ for two-body reactions). For the full rates including the reactant densities use [`rates()`](rates.md).

**Returns**

_Vector\[sympy.Basic\]_
: SymPy rate-coefficient expression for each reaction, in the same order as the catalogue.
