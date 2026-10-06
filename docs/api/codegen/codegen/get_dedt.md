---
tags:
    - Api
    - Code-generation
---

# get_dedt

`#!python get_dedt(energy="volumetric")`

Generates code for the internal energy time derivative (`dE/dt`). Returns the symbolic energy rate `(dEdt_chem + dEdt_other) / den`, rendered as a target-language expression, where `den` is the normaliser of the chosen form of the network's [EOS](../../core/network/eos.md).

**Parameters**

**energy** : _str, optional_
: Evolved internal-energy form.

    - `"volumetric"` → `den = 1` (erg/cm³/s).
    - `"specific"` → `den = ρ = Σ m_i · nden[i]` (erg/g/s).
    - `"per_particle"` → `den = n_tot = Σ nden[i]` (erg/s per particle).
    - `"molar"` → `den = n_tot / N_A` (erg/mol/s).

    Default `"volumetric"`. Raises `ValueError` for any other value.

**Returns**

_str_
: Energy-equation code string (single target-language expression, no assignment or line terminator).
