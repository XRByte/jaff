---
tags:
    - Api
    - Network
---

# eos

`#!python eos(props=None)`

Returns the symbolic internal energy of the network for the equation of state
selected by an `EosProps`. The configuration is taken from `props` or, when
`props` is `None`, from the `eos_props` passed to the `Network` constructor
(an ideal gas with `gamma = 1.6666666666667` if none was given).

The code generator uses this expression to form the temperature column of the
Jacobian via the chain rule
$\partial \dot{x} / \partial e = (\partial \dot{x} / \partial T) / (\partial e / \partial T)$.

```python
from jaff import Network
from jaff.physics import EosProps

net = Network("COthin", eos_props=EosProps("ideal", gamma=5.0 / 3.0))
e = net.eos()
e.specific  # erg/g
```

**Parameters**

**props** : _EosProps or None, optional_
: EOS configuration. `None` (default) uses the constructor's `eos_props`.

**Returns**

_Eos_
: Symbolic internal energy exposing the forms below (CGS units).

| Property       | Expression                    | Units      |
| -------------- | ----------------------------- | ---------- |
| `volumetric`   | $E$                           | erg cm⁻³   |
| `specific`     | $E / \rho$                    | erg g⁻¹    |
| `per_particle` | $E / n_\mathrm{tot}$          | erg        |
| `molar`        | $N_A\, E / n_\mathrm{tot}$    | erg mol⁻¹  |

Each form is the volumetric energy divided by `Eos.normaliser(form)`
(`1`, $\rho$, $n_\mathrm{tot}$ or $n_\mathrm{tot}/N_A$), which the code
generator also uses to normalise `dE/dt`.

**Raises**

_ValueError_
: If the configuration is not an `EosProps`.

## EosProps

`#!python EosProps(type, **kwargs)`

Omitted parameters take the type's default (`ideal`: `gamma = 1.6666666666667`).
Validated on construction: an unknown `type`, a missing or unexpected
parameter, or an adiabatic index $\le 1$ raises `ValueError`; a non-numeric
index or a non-`dict` `gamma_map` raises `TypeError`.

| `type`                          | Parameters                                   | Volumetric energy                                                    |
| ------------------------------- | -------------------------------------------- | -------------------------------------------------------------------- |
| `ideal`                         | `gamma` (float > 1, default 1.6666666666667) | $E = \dfrac{n_\mathrm{tot}\, k_B\, T_\mathrm{gas}}{\gamma - 1}$      |
| `multi_gamma`                   | `default_gamma` (float > 1), `gamma_map` (dict[str, float > 1]) | per-species sum; `gamma_map` keyed by species name, falling back to `default_gamma` |
| `fermi_degenerate`              | —                                            | not implemented                                                      |
| `relativistic_fermi_degenerate` | —                                            | not implemented                                                      |
