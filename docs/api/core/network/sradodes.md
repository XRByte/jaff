---
tags:
    - Api
    - Network
---

# sradodes

`#!python sradodes(order=0)`

Returns symbolic radiation ODE expressions in a given pattern

**Parameters**

**order** : _int, optional_
: Radiation list pattern which is ordered as:

| Order | Radiation list pattern                   |
| ----- | ---------------------------------------- |
| 0     | \[$rd_0$, $f_0$, $rd_1$, $f_1$, ...\]      |
| 1     | \[$f_0$, $rd_0$, $f_1$, $rd_1$, ...\]      |
| 2     | \[$rd_0$, $rd_1$, ..., $f_0$, $f_1$, ...\] |
| 3     | \[$f_0$, $f_1$, ..., $rd_0$, $rd_1$, ...\] |

Where $rd_i$ is the radiation energy density / photon density and $f_i$ and is the radiation flux.
Each density equation removes what photo reactions absorb and adds what reactions with a
[`spectralYield<N>`](../../../user-guide/designing-networks/auxiliary-functions.md#spectral-yield-spectralyieldn)
emit, plus the network
[`globalEmissivity`](../../../user-guide/designing-networks/auxiliary-functions.md#global-emissivity-globalemissivity),
and removes what the network
[`globalAbsorptionCoefficient`](../../../user-guide/designing-networks/auxiliary-functions.md#global-absorption-coefficient-globalabsorptioncoefficient)
absorbs. The flux equations carry the photo-reaction absorption and, unless
`absorption_attenuates_flux` is off, the `globalAbsorptionCoefficient` absorption too.

Default is `0`.

**Returns**

_list of sympy.Expr_
: One expression per radiation band.
