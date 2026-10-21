---
tags:
    - Api
icon: phosphor/atom
---

# jaff.physics

Physical constants and photochemical cross-section lookup for astrochemical calculations.

## Classes

| Class                                                 | Description                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Photochemistry`                                      | Photo cross-section lookup from the bundled databases (see below)                                                   |
| `XsecProps`                                           | One reaction's photo cross sections (dataclass, see below); what `get_xsec` returns and `Reaction.photo.xsec` holds |
| [`RadiationProps`](radiation/index.md#radiationprops) | Radiation-field configuration (bands, per-band spectral index, mode, speed of light)                                |
| [`Radiation`](radiation/index.md#radiation_1)         | Band collection built by `Network` (`net.radiation`); piecewise spectral profiles                                   |
| [`RadiationGroup`](radiation/index.md#radiationgroup) | One band: edges, `profile_idx`, `photden`, `eavg`, per-reaction rate data                                           |

## Submodules

| Submodule                         | Description                                                           |
| --------------------------------- | --------------------------------------------------------------------- |
| [`constants`](constants/index.md) | Physical & astronomical constants as `astropy` Quantities (see below) |

## Photochemistry methods

`jaff.physics.Photochemistry` resolves a reaction's cross sections and
shielding factors from `jaff.db`. It takes the owning `Network`, whose
proxy-photoreaction setting picks the look-up key and whose radiation field
supplies the global `pi_database`. Constructing it downloads the cross-section
and line-shielding data files on first use (cached thereafter), so instantiate
once and reuse:

```python
from jaff.physics import Photochemistry

photo = Photochemistry(net)
photo.get_xsec(rxn)              # XsecProps from the tabulated databases
photo.get_verner_xsec(rxn)       # analytic Verner σ(E) (sympy) or None
Photochemistry.shielding(rxn, net)  # symbolic shielding factor (sympy)
```

| Method                               | Returns              | Description                                                                                                                                                                              |
| ------------------------------------ | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_xsec(reaction, required=False)` | `XsecProps or None`  | Tabulated (Leiden / NORAD: `photon_energy` in eV, `photo_absorption` / `photodecay` in cm²) or Verner symbolic (`photodecay_expr`) cross sections, per `pi_database`; carries `database` |
| `get_verner_xsec(reaction)`          | `sympy.Expr or None` | Analytic Verner (1996) σ(E) expression (symbol `ph_E` in eV, σ in cm²)                                                                                                                   |
| `shielding(reaction, network)`       | `sympy.Expr`         | Dimensionless line-shielding factor; dispatches to the global/local shielding function named by the reaction metadata                                                                    |
| `log_fallback_summary()`             | `None`               | Logs one warning listing the photoionizations `get_xsec` resolved from a fallback database (recorded in `fallbacks` as `(key, requested, used)`), then clears them                       |

## XsecProps

`XsecProps` is a dataclass holding one photo-reaction's cross sections for its
single decay channel. Instances do not define equality (`eq=False`), since they
carry numpy arrays.

| Attribute          | Type                    | Description                                                                                                                                                                                      |
| ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `units`            | `XsecUnits`             | `photon_energy` (`"eV"`) and `cross_section` (`"cm^2"`) unit labels                                                                                                                              |
| `process`          | `XsecProcess`           | `absorption` (weight the rate by `photo_absorption`) and `decay_type` (`"ionization"` / `"dissociation"`)                                                                                        |
| `database`         | `str`                   | Source used: `"norad"`, `"verner"`, `"leiden"` or `"custom"` (a `.jfunc` `photoCrossSection<N>`)                                                                                                                                                 |
| `photon_energy`    | `numpy.ndarray or None` | Tabulated energy grid (eV); `None` for Verner                                                                                                                                                    |
| `photo_absorption` | `numpy.ndarray or None` | Photo-absorption cross section (cm²), or `None` where absent                                                                                                                                     |
| `photodecay`       | `numpy.ndarray or None` | Photodecay cross section (cm²); `None` for Verner                                                                                                                                                |
| `photodecay_expr`  | `sympy.Expr or None`    | Verner σ(E) expression (symbol `ph_E` in eV, σ in cm²); `None` for tabulated sources                                                                                                             |
| `total`            | `float or None`         | Cross section averaged over the whole radiation spectrum (cm²); `0.0` until the rate is computed when radiation is enabled (it stays `0.0` for a custom-rate reaction), `None` without radiation |

## Unit systems

`jaff.physics.constants` exposes physical constants as module-level
[`astropy.units.Quantity`](https://docs.astropy.org/en/stable/units/quantity.html)
objects. There are no separate `cgs`/`si`/`gaussian`/`natural` tables — select
the unit system **per quantity** at the call site with `.cgs`, `.si`, or
`.to(...)`. See [`constants`](constants/index.md) for the full list.

## Example

```python
from jaff.physics import constants as c

# Per-quantity unit selection
c.c.cgs              # speed of light [cm/s]
c.k_B.cgs            # Boltzmann constant [erg/K]
c.c.si               # speed of light [m/s]
c.k_B.to("eV / K")   # Boltzmann constant [eV/K]

# Elementary charge keeps astropy's EM-unit views
c.e.esu              # Gaussian / CGS-ESU [Fr]
c.e.si               # SI [C]

# Bare float (in the chosen unit) when a number is required
float(c.m_e.cgs.value)   # electron mass [g]
```
