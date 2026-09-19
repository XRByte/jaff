# Density-alias grammar: `n_X`, `n_X_nuc`, and the `expand_nuclei` flag

**Date:** 2026-09-19
**Branch:** bug-fix/semi-exhaustive (worktree bug-fix-exhaustive)
**Status:** design approved, pending spec review

## Problem

Rate/aux expressions reference species densities through `n_...` alias symbols.
The current resolver (`Network._standardize_symbols`) overloads the `n_` prefix by
an implicit, element-specific convention rather than by rule:

- `n_H` / `n_He` are magic: they mean the **element nucleus sum**
  (Σ atom-count × density over every species bearing that element).
- Every other `n_X` is a single-species lookup.
- Neutral atomic species are reached through a `0` marker hack
  (`n_H0` → species `H`), because a bare `n_H` was stolen for the sum.
- A hardcoded `_simple_map` (`nh0→H`, `nh2→H2`, `ne→e-`, `nhj→H+`), a special
  `nh` case, and three parser-global aliases (`get_hnuclei(n)→nh`,
  `n(idx_h)→nh0`, `n(idx_h2)→nh2`) exist only for hydrogen.

`GOW.jfunc` shows the confusion in the wild: `nH`/`n_H` = H nucleus sum,
`nH0`/`n_H0` = neutral H atom, `nH2` = H2, `n_Hj` = H+, `n_C0`/`n_Cj`/`n_CO` =
those species. The distinction lives entirely in the reader's head.

Two concrete defects fall out:

1. Rate strings are lower-cased upstream (`__normalize_rates`), so
   `core in ["H","He"]` never matched — element sums silently collapsed to a
   single species (`n_H` → `nden[H]` instead of `nden[H] + 2·nden[H2] + …`).
2. An `n_X` that matches no species leaves a stray free symbol in the rate
   (silent wrong physics) instead of failing.

## Goal

Replace the implicit convention with one uniform, self-documenting grammar,
resolved by rule, with loud failures. This is a **breaking change to the
network file format** (`.jfunc` / `.jet`); acceptable because the software is
alpha and every network lives in-repo.

## Grammar

Symbols are parsed case-insensitively (rate strings are lower-cased upstream for
Fortran-style case insensitivity), but resolved by structural rule, never by a
magic list of element names.

| form | resolves to |
|---|---|
| `n_e` | electron species (`e-`) |
| `n_<species>` | that exact species density — `n_H` = neutral atomic H, `n_H2`, `n_CO`, `n_Hj` = H+, `n_Hk` = H−, `n_Hejj` = He++, `n_Hekk` = He−− |
| `n_<element>_nuc` | nucleus sum Σ atom-count × nden over all species bearing that element (`n_H_nuc`, `n_He_nuc`, `n_C_nuc`) |
| `ntot` | total number density (unchanged) |
| `rc_<int>`, `chi_pe` | unchanged |

Charge convention (existing j/k encoding): neutral = no suffix, `j` = +,
`k` = −, `jj` = ++, `kk` = −−. `n_H` is neutral; there is **no** `0` marker.

## Guards (all raise; never emit a stray free symbol)

- `n_<X>j_nuc` / `n_<X>k_nuc` (any charge suffix + `_nuc`) → error: a nucleus
  sum is per-element, so a charged nucleus sum is meaningless.
- `n_<X>_nuc` where `X` is not a known element → error.
- `n_<species>` matching no species in the network → error.
- `n_<X>0` (legacy neutral marker) → error (no longer special; `n_H` is neutral).

A missed migration therefore fails loudly instead of silently collapsing a sum
to a species or leaving a dangling symbol.

## Removed

- `Network._simple_map` (`nh0`, `nh2`, `ne`, `nhj`).
- The `nh` special case in `_standardize_symbols` (→ `n_H_nuc`).
- Parser globals `get_hnuclei(n)→nh`, `n(idx_h)→nh0`, `n(idx_h2)→nh2`
  (`NetworkParser.__seed_globals` / equivalent).

Everything routes through the single `n_...` resolver.

## Flag rename

`replace_nH` → **`expand_nuclei`** (bool).

- `True` (default): `n_X_nuc` expands to the species sum.
- `False`: `n_X_nuc` stays a symbol `nx_nuc` for wrappers that inject nucleus
  densities externally.

`replace_nH` is accepted as a **silent alias** forwarding to `expand_nuclei`
(no deprecation warning — alpha software). Applies to the `Network` constructor,
`NetworkSpec`/args, and the `jaffgen` / `jaffx` CLIs.

## Migration

Rewrite `.jfunc` / `.jet` density aliases in: `GOW`, `GOW++`, `GOW_scpc`,
`GOW.scpc` (`GOW/GOW.scpc.jfunc`), `cie_h`, `h2form`.

Mapping: `nH → n_H_nuc`, `nH0 → n_H`, `nH2 → n_H2`, `nHj → n_Hj`, `ne → n_e`,
`n_H0 → n_H`, `n_C0 → n_C`, `n_O0 → n_O` (drop `0` markers), `n_Cj`/`n_Hj`/
`n_Hejj` unchanged. Function *parameter* names inside `@function` bodies are
local bindings and only need changing where they are also the substituted global
alias at the top-level rate/heating entry points — audit each entry function.

Regenerate affected golden files (`tests/golden/GOW_microphysics/…`, plus any
GOW/cie_h/h2form goldens) once resolution is verified correct.

## Testing (TDD, through full text loading)

New behavior must be proven end-to-end (loading a `.dat` + `.jfunc`), because the
old direct-symbol tests bypassed `__normalize_rates` and hid the collapse bug.

- `n_H_nuc` = element sum over multiple H-bearing species (H + H2 → nden[H] + 2·nden[H2]).
- `n_He_nuc` = element sum (He + HeH+).
- `expand_nuclei=False` → `n_H_nuc` stays symbol `nh_nuc`.
- `n_H`, `n_H2`, `n_CO`, `n_Hj`, `n_e` → correct single species.
- Guards raise: `n_Hj_nuc`, `n_Xx_nuc` (unknown element), `n_H0`, `n_<missing species>`.
- Rewrite `test_decode_neutral_zero_suffix` / `test_decode_neutral_h_vs_sum` to
  the `n_H`/`n_O` forms; `n_H0`/`n_O0` now assert an error.
- Existing charge tests (`test_decode_multi_charge_density`,
  `test_decode_single_cation`, `test_decode_electron`) stay green.
- GOW / cie_h / h2form end-to-end load + golden regeneration.

## Out of scope / follow-ups

- The interim case-insensitive element-sum patch in `_standardize_symbols`
  (added earlier this session) is **superseded** by this resolver and should be
  replaced, along with its `tests/test_density_symbols.py` interim assertions.
- Unrelated open item from earlier this session: a mass-conservation warning
  regression (`H -> H2` stopped emitting the warning in
  `tests/test_reaction_conservation.py::…test_warning_includes_reaction_index`)
  needs separate diagnosis. Not part of this change.
