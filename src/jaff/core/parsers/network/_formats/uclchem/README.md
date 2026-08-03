# UCLCHEM network parser — rate-coefficient support

UCLCHEM reaction files are comma-delimited, identified by a `NAN` sentinel
column. Each line carries a _mechanism keyword_ in one of its reactant slots
(`FREEZE`, `THERM`, `DESCR`, …) that selects which rate-coefficient formula
UCLCHEM applies at runtime. Reactions with no keyword are ordinary two-body
gas-phase reactions.

This parser reproduces the UCLCHEM rate formulas as SymPy expression strings.
**Not every mechanism can be expressed as a closed-form, per-line rate**: some
need the dust temperature, a surface-diffusion competition formula, or coupling
to another reaction's rate. Those are emitted as `0.0` and listed below.

All formulas are transcribed from UCLCHEM's `src/fortran_src/rates.f90`
(`MODULE RATES`). See [References](#references).

---

## Column layout

```
R1, R2, R3, P1, P2, P3, P4, Alpha, Beta, Gamma, Tmin, Tmax, reduced_mass, extrapolate
```

`Alpha`/`Beta`/`Gamma` → `alpha`/`beta`/`gama` below. `Gamma`'s physical
meaning changes with the mechanism (activation energy for gas reactions,
binding energy for desorption). `reduced_mass` is the species mass used by the
freeze-out rate.

---

## Supported mechanisms

`omega` = grain albedo (UCLCHEM default `0.5`), so `1/(1-omega) = 2`.

| Keyword  | JAFF `type`                             | Rate expression                                                                | UCLCHEM source                                                                       |
| -------- | --------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| _(none)_ | `unknown`/`photo`/`cosmic_ray`/`3_body` | `alpha * (tgas/300)**beta * exp(-gama/tgas)`                                   | two-body Kooij/Arrhenius                                                             |
| `CRP`    | `cosmic_ray`                            | `alpha * crate`                                                                | `alpha * zeta`                                                                       |
| `CRPHOT` | `cosmic_ray`                            | `alpha*gama/(1-omega) * (tgas/300)**beta * crate`                              | `alpha*gama/(1-omega)*(T/300)**beta*zeta`                                            |
| `PHOTON` | `photo`                                 | `alpha/1.7 * chi * exp(-gama*av)`                                              | `alpha*radfield*exp(-gama*Av)/1.7`                                                   |
| `FREEZE` | `freeze`                                | `(1 + beta*1.671e-3/tgas/asize) * nuth*sigmah*sqrt(tgas/m)`                    | `freezeFactor*alpha*v_th*sqrt(T/m)*sigma_grain`                                      |
| `DESCR`  | `desorption_cr`                         | `alpha*4*pi*1.64e-4*(4*sigmah)*phi * crate`                                    | `4*pi*zeta*1.64e-4*surfaceArea*phi*alpha`                                            |
| `DEUVCR` | `desorption_uvcr`                       | `alpha*4.875e3*sigmah*uv_yield*crate * (1 + (chi/uvcreff)/crate*exp(-1.8*av))` | `sigma_grain*uv_yield*4.875e3*zeta*(1 + (radfield/uvcreff)/zeta*exp(-1.8*Av))*alpha` |

### Constants folded into the emitted numbers

- `1/(1-omega) = 2` (CRPHOT) — grain albedo 0.5.
- `/1.7` (PHOTON) — converts the Draine FUV field to Habing units.
- `1.64e-4` (DESCR) — cosmic-ray grain-heating duty factor, Hasegawa & Herbst (1993).
- surface area per H `= 4 * sigmah` (DESCR) — sphere area `4πr²` vs cross-section `πr²`.
- `4.875e3` (DEUVCR) — CR-induced UV photon flux prefactor.
- `1.671e-3` (FREEZE) — Coulomb-focusing factor for charged reactants.

### Free symbols the host model must bind

Beyond the standard runtime symbols `tgas`, `crate` (ζ), `av`, `chi` (FUV
field), the grain mechanisms introduce grain/model parameters as free symbols —
matching the pre-existing `FREEZE` convention in this parser:

| Symbol     | Meaning                                    | Used by               |
| ---------- | ------------------------------------------ | --------------------- |
| `nuth`     | thermal-velocity prefactor `sqrt(8k/πm_u)` | FREEZE                |
| `sigmah`   | grain cross-section per H nucleus          | FREEZE, DESCR, DEUVCR |
| `asize`    | grain radius                               | FREEZE                |
| `m`        | reactant mass                              | FREEZE                |
| `phi`      | cosmic-ray desorption efficiency           | DESCR                 |
| `uv_yield` | photodesorption yield (molecules / photon) | DEUVCR                |
| `uvcreff`  | CR-induced UV field scaling                | DEUVCR                |

---

## Unsupported mechanisms (emitted as `0.0`)

| Keyword    | JAFF `type`                       | Why unsupported                                                                                                                                                                                                  |
| ---------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `THERM`    | `desorption_thermal`              | Needs **dust temperature** and a per-species vibrational frequency `vdiff = sqrt(2·N_sites·k·E_bind/(π²·m))`; `vdiff` is not a per-line quantity. UCLCHEM: `vdiff*exp(-gama/Tdust)*alpha*2*N_sites*surfaceArea`. |
| `DESOH2`   | `desorption_h2`                   | Rate `= epsilon * h2FormEfficiency(Tgas,Tdust) * alpha` — needs the H₂-formation-efficiency function and the dust temperature.                                                                                   |
| `LH`       | `langmuir_hinshelwood`            | Langmuir-Hinshelwood diffusion rate is a **competition formula** (Chang et al. 2007; Garrod & Pauly 2011) over dust temperature and per-species diffusion barriers, not a closed per-line expression.            |
| `LHDES`    | `langmuir_hinshelwood_desorption` | Reactive-desorption branch: rate `= desorptionFraction * rate(corresponding LH reaction)` — **references another reaction's rate**.                                                                              |
| `ER`       | `eley_rideal`                     | Rate `= freezeOutRate * exp(-gama/Tdust)` — needs the freeze-out rate and dust temperature.                                                                                                                      |
| `ERDES`    | `eley_rideal_desorption`          | Reactive-desorption branch coupled to the corresponding `ER` reaction's rate.                                                                                                                                    |
| `H2FORM`   | `h2_formation`                    | Parameterised H₂ formation efficiency `h2FormEfficiency(Tgas,Tdust)` (Cazaux & Tielens 2002, 2004), not `alpha/beta/gama`.                                                                                       |
| `BULKSWAP` | `bulk_swap`                       | Bulk↔surface ice exchange, computed by UCLCHEM's `bulkSurfaceExchangeReactions` from the net freeze/desorption fluxes — coupled, not per-line.                                                                   |
| `SURFSWAP` | `surface_swap`                    | Surface↔bulk exchange, same coupled treatment as `BULKSWAP`.                                                                                                                                                     |

### Three structural blockers

1. **Dust temperature.** `THERM`, `DESOH2`, `LH`, `ER` evaluate at `dustTemp`,
   which can differ from `gasTemp`. A `tdust` symbol exists in JAFF, but these
   rates need more than temperature (see below).
2. **Cross-reaction coupling.** `LHDES`, `ERDES`, `BULKSWAP`, `SURFSWAP` derive
   their rate from _other_ reactions' rates. That requires a second resolution
   pass after all reactions are parsed — the per-line parser cannot express it.
3. **Surface-diffusion machinery.** `LH`/`ER`/`THERM`/`H2FORM` need per-species
   binding/diffusion energies, site density, and vibrational frequencies (the
   competition formula), none of which live in the reaction line.

Implementing these would require a post-parse coupling pass plus a grain-surface
parameter table — tracked as separate follow-up work.

---

## References

- **UCLCHEM 4.0** — Vermariën et al. (2026), arXiv:2606.20265. Source of the
  `rates.f90` formulas transcribed here.
- **UCLCHEM v1** — Holdship et al. (2017), _AJ_ 154, 38. Core rate-equation model.
- **Thermal desorption** — Viti et al. (2004), _MNRAS_ 354, 1141.
- **Cosmic-ray desorption** — Hasegawa & Herbst (1993), _MNRAS_ 261, 83
  (the `1.64e-4` grain-heating factor).
- **Photodesorption** — Roberts et al. (2007); Hollenbach et al. (2009).
- **Surface LH/ER mechanisms + competition** — Quénard et al. (2018),
  _MNRAS_ 474, 2796; Chang et al. (2007); Garrod & Pauly (2011).
- **H₂ formation efficiency** — Cazaux & Tielens (2002, 2004).
