---
tags:
    - Api
    - Reaction
---

# Reaction

`jaff.core.reaction.Reaction`

The `Reaction` class represents a single chemical reaction, holding its reactants, products, symbolic rate expression, valid temperature bounds, and associated energy and radiation terms.

## Constructor

`#!python Reaction(reactants, products, rate, tmin, tmax, dE, original_string, index, t_cutoff="clip", type="unknown", errors=False, photo=None, rad=None)`

**Parameters**

**reactants** : _list\[Specie\]_
: Reactant species.

**products** : _list\[Specie\]_
: Product species.

**rate** : _sympy.Expr_
: Symbolic rate expression.

**tmin** : _float or None_
: Minimum valid temperature in Kelvin.

**tmax** : _float or None_
: Maximum valid temperature in Kelvin.

**dE** : _sympy.Basic_
: SymPy expression for the internal energy released or absorbed per reaction event (erg).

**original_string** : _str_
: Raw reaction string from the network file.

**index** : _int_
: Position in the network reaction list.

**t_cutoff** : _str, optional_
: Out-of-range behaviour for the rate segment(s): `"clip"` (default) holds the boundary value, `"extrapolate"` evaluates the expression as-is. See [`jaff.toml` temperature cutoffs](../../../user-guide/working-with-networks/jaff-toml.md).

**type** : _str, optional_
: Reaction type concluded by the network-format parser, stored verbatim. Generic gas-phase values are `"photo"`, `"cosmic_ray"`, `"3_body"`, `"unknown"`; grain/ice networks add surface-mechanism types (`"freeze"`, `"desorption_thermal"`, `"desorption_cr"`, `"desorption_uvcr"`, `"desorption_h2"`, `"eley_rideal"`, `"eley_rideal_desorption"`, `"langmuir_hinshelwood"`, `"langmuir_hinshelwood_desorption"`, `"h2_formation"`, `"bulk_swap"`, `"surface_swap"`). Default `"unknown"`.

**errors** : _bool, optional_
: If `True`, terminate the process on mass or charge conservation violations instead of merely logging a warning. Default `False`.

**photo** : _PhotoReactionProps or None, optional_
: Photo-reaction state (see the `photo` attribute). `Network` passes one for every photo reaction. Default `None`.

**rad** : _ReactionRadiationProps or None, optional_
: Radiation coupling (see the `rad` attribute). Default `None`, which gives a zero `dRad`.

## Attributes

| Attribute             | Type            | Description                                                                                                                                      |
| --------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `reactants`           | `Species`       | Ordered species catalogue of reactant species                                                                                                    |
| `products`            | `Species`       | Ordered species catalogue of product species                                                                                                     |
| `rate`                | `sympy.Expr`    | SymPy expression for the rate coefficient (units depend on reaction order; typically cm³ s⁻¹ for two-body reactions)                             |
| `tmin`                | `float or None` | Minimum gas temperature at which the rate is valid (K). `None` means no lower bound                                                              |
| `tmax`                | `float or None` | Maximum gas temperature at which the rate is valid (K). `None` means no upper bound                                                              |
| `t_cutoff`            | `str`           | Out-of-range behaviour (`"clip"` / `"extrapolate"`) for the rate segment(s)                                                                       |
| `rate_segments`       | `RateSegments`  | Per-temperature-range rate pieces; one segment for a single-range reaction, several when merged across disjoint ranges, collapsed into `rate`     |
| `dE`                  | `sympy.Basic`   | SymPy expression for the energy released per reaction event (erg)                                                                                |
| `rad`                 | `ReactionRadiationProps` | Radiation coupling, present on every reaction: `dRad`, the radiation energy the reaction adds to the field per unit photon energy (erg/eV), as a SymPy expression in `E` (eV); zero unless a `.jfunc` `deltaRad` function sets it. The energy symbol must be `E` for it to be integrated over the bands |
| `verbatim`            | `str`           | Human-readable `"R1 + R2 -> P1 + P2"` form                                                                                                       |
| `serialized`          | `str`           | Canonical name-level form `"<sorted_reactants>__<sorted_products>"`                                                                              |
| `serialized_exploded` | `str`           | Like `serialized` but built from atom-level serialized forms of each species (isomer-insensitive)                                                |
| `index`               | `int`           | File-side source number (`source_index`): the file-order row number. Stable per parsed row but **gapped** when rows merge (`duplicate_policy` or a temperature-range merge), so it is **not** a valid array subscript. Used for aux lookups (`chemRateN`) and `Reactions.by_source_index()` |
| `catalogue_index`     | `int`           | Dense `0..count-1` position in the parent `Reactions` catalogue, assigned by `Reactions.add()`. Safe as an array subscript in generated code (what the `reaction_idx` template property emits). `-1` until catalogued |
| `type`                | `str`           | Reaction type concluded by the parser (verbatim): gas-phase `"photo"`/`"cosmic_ray"`/`"3_body"`/`"unknown"`, or a grain surface-mechanism type (`"freeze"`, `"desorption_*"`, `"eley_rideal*"`, `"langmuir_hinshelwood*"`, `"h2_formation"`, `"bulk_swap"`, `"surface_swap"`) |
| `photo`               | `PhotoReactionProps or None` | Photo-reaction state; `None` for non-photo reactions. Holds `xsec` (the reaction's [`XsecProps`](../../physics/index.md#xsecprops) cross sections, or `None` when no database has an entry), `pi_database` (per-reaction photoionization database override), `groups` (radiation bands the reaction contributes to, populated when a radiation field is configured; see [`band_xsecs`](band_xsecs.md)), `custom_rad_rate` (`True` when the rate came from a `.jfunc` aux function rather than the cross sections) and `xsec_tot` (cross section averaged over the whole spectrum, cm²) |
