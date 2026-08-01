---
tags:
    - Api
    - Network
---

# Network

`jaff.core.network.Network`

The `Network` class is the most important class in JAFF. It reads a reaction network file, auto-detects its format, validates mass and charge conservation, and assembles the full species and reaction catalogues along with stoichiometry matrices. It also handles optional radiation transport, photochemistry cross-sections, and auxiliary function files.

## Constructor

`#!python Network(fname, config=None, errors=False, label=None, funcfile=True, replace_nH=True, rad_bands=[], rad_powerlaw_index=0, rad_energy_density=False, c=constants.c.cgs.value)`

**Parameters**

**fname** : _str or Path_
: Path to a network file, or the name of a built-in network (a sub-directory of `networks/` containing a single `.jet` file). A built-in network name wins over a same-named path on disk. Supported formats: KIDA, UDFA, PRIZMO, KROME, UCLCHEM, a combination of the above and the `.jaff` file (Refer to [to_jaff](to_jaff.md) for more details).

**config** : _str, Path, or None, optional_
: Path to a TOML configuration file. When `None` (default), JAFF looks for `jaff.toml` in the network file's directory.

**errors** : _bool, optional_
: Exit on validation errors. Default `False`.

**label** : _str or None, optional_
: Network identifier. Defaults to the file stem.

**funcfile** : _bool, str, or Path, optional_
: Path to .jfunc auxiliary functions file. `True` (default) scans the network directory; `False` skips.

**replace_nH** : _bool, optional_
: Replace nH/nHe symbols with species density sums. Default `True`.

**rad_bands** : _list, optional_
: Radiation band boundaries enabling radiation transport. Default `[]`.

**rad_powerlaw_index** : _int or float, optional_
: Spectral power-law index. Default `0`.

**rad_energy_density** : _bool, optional_
: Interpret radiation as energy density. Default `False`.

**c** : _float, optional_
: Speed of light in CGS. Default `constants.c.cgs.value`.

**Raises**

_FileNotFoundError_
: If `fname` does not exist.

## Attributes

| Attribute         | Type                | Description                                                                                                         |
| ----------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `label`           | `str`               | Human-readable network identifier; defaults to the source file stem                                                 |
| `filename`        | `Path`              | Resolved absolute path to the source network file                                                                   |
| `spec`            | `NetworkSpec`       | Normalized construction parameters (resolved `fname`, parsed `config` dict, `funcfile`, `aux_funcs`, ...)           |
| `species`         | `Species`           | Ordered catalogue of the network's core (real) species; special pseudo-species (`_PHOTON`, `_CR`, ...) are excluded |
| `reactions`       | `Reactions`         | Ordered catalogue of all reactions in the network                                                                   |
| `elements`        | `Elements`          | Element catalogue derived from all species; used for composition matrices                                           |
| `reactant_matrix` | `ndarray`           | Shape (n_reactions, n_species) stoichiometry matrix for reactants                                                   |
| `product_matrix`  | `ndarray`           | Shape (n_reactions, n_species) stoichiometry matrix for products                                                    |
| `mass_dict`       | `dict`              | Mapping from element symbol to mass properties, used for conservation checks                                        |
| `dEdt_chem`       | `sympy.Basic`       | Total chemical heating/cooling rate (erg cm⁻³ s⁻¹), accumulated over all reactions                                  |
| `dEdt_other`      | `sympy.Basic`       | Additional heating/cooling rate from the `heatingcoolingrate` auxiliary function, if present                        |
| `dRad_dt_extra`   | `sympy.Basic`       | Extra radiation moment source terms from `@function` definitions                                                    |
| `radiation`       | `Radiation or None` | Radiation field object; `None` when no radiation bands are specified                                                |
| `ndens`           | `sympy.MatrixSymbol`| Symbolic `nden` column vector of species number densities, shape (n_species, 1); `nden[i]` is species `i`           |
| `ntot`            | `sympy.Expr`        | Total number density, `Σ_i nden[i]` over all species                                                                |
| `rho`             | `sympy.Expr`        | Mass density, `Σ_i m_i · nden[i]`; species with unset mass contribute `0`                                           |
