---
tags:
    - User-guide
    - Testing
---

# Collisional Ionization Equilibrium

This is the smallest of the three tests: two reactions, three species, and a fixed temperature. Because the equilibrium state has a closed-form expression, you always know the answer a correct run should converge to, which makes it a good first check for how a generated network should behave.

## The network

**File:** `networks/cie_h/react_cie_h.jet` (PRIZMO format)

```text
# collisional ionization (Cen 1992)
H + E -> H+ + E + E   []   5.85e-11*sqrt(Tgas)*exp(-157809.1/Tgas)

# recombination
H+ + E -> H           []   2.6e-13*(Tgas/1e4)**(-0.8)
```

There are three species: `H`, `H+`, `E`. The generated indices are `idx_h`, `idx_hj`, `idx_e`.

## What it checks

Ionization balances recombination in steady state:

$$k1(T) \times n_H \times n_e = k2(T) \times n_{H+} \times n_e$$

The electron density `n_e` appears on both sides and cancels, which leaves the equilibrium ionization fraction depending on temperature alone — not on density, the initial condition, or the path the system took to get there:

$$x_{\text{eq}} = \frac{n_{\text{H}^+}}{n_{\text{H}} + n_{\text{H}^+}} = \frac{k_1(T)}{k_1(T) + k_2(T)}$$

If a host code reaches the right `x_eq`, its rate coefficients, their temperature dependence, and its handling of equilibrium are all working. If it settles on the wrong constant, the rate coefficients are the place to look; if it never settles at all, the problem is more likely in the integrator.

## Setup

| Quantity       | Value                              | Note                                                              |
| -------------- | ---------------------------------- | ----------------------------------------------------------------- |
| Temperature    | `1.5e4 K`, **fixed**               | Do not evolve the temperature. This network has no thermal terms. |
| Total hydrogen | any, e.g. `1.0 cm⁻³`               | Sets the timescale only — see the scaling note below.             |
| Initial `x`    | `1e-3`                             | `n_H = 0.999 n_tot`, `n_H+ = n_e = 1e-3 n_tot`                    |
| End time       | `t · n_tot ≳ 1e15 s cm⁻³`          | Roughly two decades past equilibration.                           |
| Output         | log-spaced, ≥ 50 points per decade | Only the final value is compared; the curve is for diagnosis.     |

<!-- prettier-ignore -->
!!! warning "Seed the electrons"
    Collisional ionization needs an electron to get going. If you start from `n_e = 0`, the right-hand side is identically zero and the run never leaves its initial state. Nothing is wrong with the integrator in that case — the initial condition simply gives it nothing to do.

<!-- prettier-ignore -->
!!! note "Density only rescales time"
    Both terms are second order in density, so the solution depends on `t` and `n` only through their product. This means you can pick whatever total density suits your host code and rescale the time accordingly — a run at `n = 10³ cm⁻³` reaches the same state a thousand times sooner than one at `n = 1 cm⁻³`. The `tn` column in the reference table is that combined variable.

## What to plot

Ionization fraction `x = n_H+ / (n_H + n_H+)` against `t · n_tot`, both axes logarithmic, with the analytic `x_eq` as a horizontal line.

![Ionization fraction rising from the 1e-3 seed to the analytic equilibrium value 0.5069](../../assets/figures/testing-networks/cie_h_light.png#only-light){ width="720" }
![Ionization fraction rising from the 1e-3 seed to the analytic equilibrium value 0.5069](../../assets/figures/testing-networks/cie_h_dark.png#only-dark){ width="720" }

The curve stays flat at the seed value while the electron pool is still small, then turns over once ionization starts to feed itself, and finally settles onto `x_eq` from below.

## What to expect

At `T = 1.5e4 K`:

```text
    k1 = 1.9333e-13 cm³ s⁻¹
    k2 = 1.8802e-13 cm³ s⁻¹
    x_eq = 0.506942
```

If you would like to run the test at more than one temperature, here are the equilibrium values across a range:

| `T` (K) | `k1` (cm³ s⁻¹) | `k2` (cm³ s⁻¹) | `x_eq`   |
| ------- | -------------- | -------------- | -------- |
| 8.0e3   | 1.418e-17      | 3.108e-13      | 0.000046 |
| 1.0e4   | 8.196e-16      | 2.600e-13      | 0.003142 |
| 1.5e4   | 1.933e-13      | 1.880e-13      | 0.506942 |
| 2.0e4   | 3.097e-12      | 1.493e-13      | 0.953995 |
| 3.0e4   | 5.263e-11      | 1.080e-13      | 0.997953 |
| 5.0e4   | 5.571e-10      | 7.175e-14      | 0.999871 |
| 1.0e5   | 3.818e-09      | 4.121e-14      | 0.999989 |

Suggested acceptance:

| Quantity                                           | Threshold |
| -------------------------------------------------- | --------- |
| relative error of `x_final` against `x_eq`         | < 1e-3    |
| largest departure of `(n_H + n_H+) / n_tot` from 1 | < 1e-10   |
| largest value of `abs(n_e - n_H+) / n_tot`         | < 1e-10   |

The last two check conservation of hydrogen nuclei and of charge. These follow from the structure of the right-hand side rather than from the tolerance you set, so they should hold to near round-off no matter which integrator you use.

## Reference data

`benchmarks/cie_h/cie_h_T1.5e4.txt` — the full trajectory, integrated at `rtol = 1e-12` and agreeing with `x_eq` to 4e-16.

| Column | Meaning                   |
| ------ | ------------------------- |
| `t`    | time (s), for `n_tot = 1` |
| `tn`   | `t · n_tot` (s cm⁻³)      |
| `n_H`  | atomic hydrogen (cm⁻³)    |
| `n_Hp` | ionized hydrogen (cm⁻³)   |
| `n_e`  | electrons (cm⁻³)          |
| `x`    | `n_Hp / (n_H + n_Hp)`     |

Unless you also ran at `n_tot = 1 cm⁻³`, compare against the `tn` column rather than `t`, since `tn` is the density-independent variable (see the scaling note above).

## When it fails

| Symptom                                | Cause                                                                                                                                                            |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nothing happens; `x` stays at its seed | `n_e = 0` initially, or the temperature never made it into the rate evaluation (`T ≈ 0`).                                                                        |
| `x` settles far below `x_eq`           | Running below `1e4 K`. `exp(-157809.1/T)` is `2.7e-5` at 1.5e4 K and `4e-9` at 1e4 K, so the gas really does not ionize. Check the temperature before the rates. |
| `x` settles at a wrong constant        | Rate coefficient wrong, or the temperature the network sees differs from the one you set.                                                                        |
| `x` overshoots 1 or goes negative      | Explicit or under-resolved integration, possibly masked by species clipping.                                                                                     |
| Nuclei conservation drifts             | Renormalization enabled, or a floor/clip large enough to matter.                                                                                                 |
