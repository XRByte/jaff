---
tags:
    - Api
    - Reaction
---

# rates

`#!python rates()`

Returns the symbolic reaction rate per unit volume (`Reaction.rate`: the rate coefficient times each core reactant's number density) for every reaction in the catalogue, in catalogue order. For the coefficients alone use [`rate_coefficients()`](rate_coefficients.md).

**Returns**

_Vector\[sympy.Basic\]_
: SymPy reaction-rate expression for each reaction, in the same order as the catalogue.
