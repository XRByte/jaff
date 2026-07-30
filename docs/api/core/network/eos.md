---
tags:
    - Api
    - Network
---

# eos

`#!python eos(gamma=1.6666666666667)`

Returns the symbolic ideal-gas specific internal energy of the network, built
from the network's total number density (`ntot`) and mass density (`rho`):

<!-- prettier-ignore -->
$$ e = \dfrac{n_\mathrm{tot}\, k_B\, T_\mathrm{gas}}{\rho\,(\gamma - 1)} \quad [\mathrm{erg\,g^{-1}}] $$

where $k_B$ is the Boltzmann constant in CGS units, $T_\mathrm{gas}$ is the
`tgas` symbol, and $\gamma$ is the adiabatic index. The code generator uses
this expression to form the temperature column of the Jacobian via the chain
rule $\partial \dot{x} / \partial e = (\partial \dot{x} / \partial T) / (\partial e / \partial T)$.

**Parameters**

**gamma** : _float, optional_
: Adiabatic index. Default `5/3 ≈ 1.6̄` (monoatomic ideal gas).

**Returns**

_sympy.Expr_
: Symbolic specific internal energy in CGS units (erg/g).
