# Density-alias grammar (`n_X` / `n_X_nuc`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implicit, H/He-magic `n_` density-alias convention with an explicit rule-based grammar — `n_<species>` = that species, `n_<element>_nuc` = nucleus sum — with loud guards, and migrate the GOW `.jfunc` files onto it.

**Architecture:** All `n_`-prefixed symbols in rate/aux expressions resolve through one new method `Network._resolve_density_alias`, called from `_standardize_symbols`. The KROME/PRIZMO compat layer (`nh`/`nh0`/`nh2`/`ne`/`nhj`, seed-globals) is a *separate namespace* and is left untouched. No `.jet`/`.dat` network-format file is changed. The `replace_nH` flag is renamed to `expand_nuclei` with `replace_nH` kept as a silent constructor alias.

**Tech Stack:** Python, SymPy, pytest, `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-09-19-density-alias-grammar-design.md`

**Working dir (worktree):** `/home/anish/External/programming/research/jaff/worktrees/bug_fix_multiple_temperature_ranges/src/jaff/bug-fix-exhaustive` — run everything from here.

---

## Preliminary: reset the interim patch

The previous session left an uncommitted interim fix in `network.py` and an
untracked interim test file. This plan supersedes both.

- [ ] **Step 0.1: Revert the interim `_standardize_symbols` edit**

Restore the committed version of the file, discarding only the interim element-sum patch:

Run: `git checkout -- src/jaff/core/network/network.py`
Expected: `git status --short` no longer lists `network.py` as modified.

- [ ] **Step 0.2: Remove the interim test file**

Run: `rm -f tests/test_density_symbols.py`
Expected: file gone; it will be recreated with correct assertions in Task 1.

---

## Task 1: New `n_` grammar resolver + guards

**Files:**
- Modify: `src/jaff/core/network/network.py` (the `n_`-prefix branch of `_standardize_symbols`, ~lines 1157-1188; add helper methods; promote `get_element_sum`)
- Test: `tests/test_density_symbols.py` (recreate)

Element-symbol source: `from jaff.common import load_mass_dict` — returns a dict keyed by canonical element symbol (`"H"`, `"He"`, `"C"`, …). `ParserError` is already imported in `network.py` (`from ...errors import ParserError`).

- [ ] **Step 1.1: Write failing tests (through full text loading)**

Recreate `tests/test_density_symbols.py`:

```python
# ABOUTME: n_X / n_X_nuc density-alias grammar resolved through full text loading
# ABOUTME: species-exact lookup, element nucleus sums, and loud guards

import sympy
import pytest

from jaff.errors import ParserError

TGAS = sympy.symbols("tgas")


def _rate(make_network, lines, idx=0, **kw):
    net = make_network(lines, **kw)
    return net, net.reactions[idx].rate


# --- element nucleus sums: n_X_nuc ---------------------------------------- #
def test_n_H_nuc_is_element_sum(make_network):
    net, rate = _rate(
        make_network,
        ["H + H -> H2 [10,1000] 1", "H2 -> H + H [10,1000] n_H_nuc"],
        idx=1,
    )
    nden = net.ndens
    expected = nden[sympy.Idx(net.species["H"].index)] + 2 * nden[
        sympy.Idx(net.species["H2"].index)
    ]
    assert sympy.simplify(rate - expected) == 0


def test_n_He_nuc_is_element_sum(make_network):
    net, rate = _rate(
        make_network,
        ["He + H+ -> HeH+ [10,1000] 1", "HeH+ -> He + H+ [10,1000] n_He_nuc"],
        idx=1,
    )
    nden = net.ndens
    expected = nden[sympy.Idx(net.species["He"].index)] + nden[
        sympy.Idx(net.species["HeH+"].index)
    ]
    assert sympy.simplify(rate - expected) == 0


def test_n_X_nuc_symbol_when_not_expanded(make_network):
    net, rate = _rate(
        make_network,
        ["H + H -> H2 [10,1000] 1", "H2 -> H + H [10,1000] n_H_nuc"],
        idx=1,
        expand_nuclei=False,
    )
    assert rate == sympy.symbols("nh_nuc")


# --- single-species lookup: n_X keeps exact spelling ---------------------- #
def test_n_H_is_neutral_species(make_network):
    net, rate = _rate(make_network, ["H + C -> CH [10,1000] n_H"])
    assert rate == net.ndens[sympy.Idx(net.species["H"].index)]


def test_n_molecule(make_network):
    net, rate = _rate(make_network, ["H + C -> CH [10,1000] n_CH"])
    assert rate == net.ndens[sympy.Idx(net.species["CH"].index)]


def test_n_cation(make_network):
    net, rate = _rate(make_network, ["C + C+ -> C+ + C [10,1000] n_Cj"])
    assert rate == net.ndens[sympy.Idx(net.species["C+"].index)]


def test_n_electron(make_network):
    net, rate = _rate(make_network, ["H -> H+ + e- [10,1000] n_e"])
    assert rate == net.ndens[sympy.Idx(net.species["e-"].index)]


# --- guards: all raise, none leave a stray free symbol -------------------- #
def test_charged_nucleus_sum_raises(make_network):
    with pytest.raises(ParserError):
        make_network(["C + C+ -> C+ + C [10,1000] n_Cj_nuc"])


def test_unknown_element_nucleus_sum_raises(make_network):
    with pytest.raises(ParserError):
        make_network(["H + C -> CH [10,1000] n_Xx_nuc"])


def test_zero_marker_no_longer_special_raises(make_network):
    # n_H0 no longer means neutral H; there is no species 'H0' -> error.
    with pytest.raises(ParserError):
        make_network(["H + C -> CH [10,1000] n_H0"])


def test_missing_species_raises(make_network):
    with pytest.raises(ParserError):
        make_network(["H + C -> CH [10,1000] n_Ne"])
```

- [ ] **Step 1.2: Run tests, verify they fail**

Run: `uv run pytest tests/test_density_symbols.py -q`
Expected: FAIL (old grammar: `n_H_nuc` unresolved / `n_H0` resolves via `0`-marker / guards don't raise).

- [ ] **Step 1.3: Add the element helpers and resolver method**

In `src/jaff/core/network/network.py`, add an import near the other `common` imports:

```python
from ...common import load_mass_dict
```

Add these methods to the `Network` class (place them just above `_standardize_symbols`):

```python
    def _element_symbol(self, low: str) -> str | None:
        """Return the canonical element symbol for a lower-cased token, or None."""
        if not hasattr(self, "_element_lookup"):
            self._element_lookup = {s.lower(): s for s in load_mass_dict()}
        return self._element_lookup.get(low)

    def _element_sum(self, element: str) -> Expr | None:
        """Nucleus sum Sigma count*nden over species bearing *element* (cached)."""
        if element not in self.__element_sums:
            nden = self.ndens
            terms = [
                count * nden[Idx(i)]
                for i, spec in enumerate(self.species)
                if (count := spec.exploded.count(element)) > 0
            ]
            self.__element_sums[element] = sum(terms) if terms else None
        return self.__element_sums[element]

    def _resolve_density_alias(self, name: str, expand_nuclei: bool) -> Expr:
        """Resolve an ``n_...`` density alias.

        Grammar (case-insensitive):
          * ``n_e``             -> electron species density
          * ``n_<element>_nuc`` -> nucleus sum over element-bearing species
          * ``n_<species>``     -> that exact species density
        """
        core_low = name[2:].lower()  # strip "n_"

        if core_low == "e":
            if "e-" in self.species:
                return self.ndens[Idx(self.species["e-"].index)]
            raise ParserError(f"'{name}' used but network has no electron species 'e-'")

        if core_low.endswith("_nuc"):
            base = core_low[:-4]
            if base.endswith(("j", "k")):
                raise ParserError(
                    f"'{name}' is invalid: a nucleus sum is per-element, so a "
                    f"charged nucleus alias ('...j_nuc' / '...k_nuc') is meaningless"
                )
            element = self._element_symbol(base)
            if element is None:
                raise ParserError(
                    f"'{name}' requests a nucleus sum for unknown element '{base}'"
                )
            if not expand_nuclei:
                return symbols(f"n{base}_nuc")
            total = self._element_sum(element)
            if total is None:
                raise ParserError(
                    f"'{name}': no species in the network bears element '{element}'"
                )
            return total

        if self.__charge_reverse is None:
            self.__charge_reverse = self.species.charge_reverse_map()
        sp = self.__charge_reverse.get(core_low)
        if sp is None:
            raise ParserError(
                f"Density symbol '{name}' does not match any species in this network"
            )
        return self.ndens[Idx(sp.index)]
```

Note: `self.__element_sums` and `self.__charge_reverse` are existing instance
attrs (initialised in `__init__`, lines ~307-308). Because name-mangling is
per-class, referencing `self.__element_sums` / `self.__charge_reverse` inside
these `Network` methods resolves correctly.

- [ ] **Step 1.4: Replace the old `n_` branch in `_standardize_symbols`**

Rename the loop parameter `replace_nH` to `expand_nuclei` in the
`_standardize_symbols` signature and body (the `nh` special case becomes
`repl = self.n_hnuc if expand_nuclei else symbols("nh")`). Delete the inner
`get_element_sum` closure (now `self._element_sum`). Replace the whole
`elif low_name.startswith("n_"):` block (the ~30 lines handling H/He magic,
`e`, and charge_reverse) with:

```python
            elif low_name.startswith("n_"):
                repl = self._resolve_density_alias(name, expand_nuclei)
```

Leave the `ntot`, `nh`, `simple_map`, `chi_pe`, and `rc_` branches as they are
(only the `replace_nH`->`expand_nuclei` rename touches `nh`).

- [ ] **Step 1.5: Run tests, verify they pass**

Run: `uv run pytest tests/test_density_symbols.py -q`
Expected: PASS (all 11).

- [ ] **Step 1.6: Update the two charge tests that used the `0` marker**

In `tests/test_charge_convention.py`, `test_decode_neutral_zero_suffix` and
`test_decode_neutral_h_vs_sum` used `n_O0` / `n_H0`. Change them to the exact
species form and keep the same expected species:

```python
def test_decode_neutral_species(make_network):
    import sympy
    net = make_network(["O + O -> O + O [10,1000] 1"])
    expr = net._standardize_symbols(sympy.Symbol("n_O"), True)
    assert expr == net.ndens[sympy.Idx(net.species["O"].index)]


def test_decode_neutral_h(make_network):
    import sympy
    net = make_network(["H + H+ -> H+ + H [10,1000] 1"])
    expr = net._standardize_symbols(sympy.Symbol("n_H"), True)
    assert expr == net.ndens[sympy.Idx(net.species["H"].index)]
```

- [ ] **Step 1.7: Run the charge suite, verify green**

Run: `uv run pytest tests/test_charge_convention.py tests/test_density_symbols.py -q`
Expected: PASS.

- [ ] **Step 1.8: Commit**

```bash
git add src/jaff/core/network/network.py tests/test_density_symbols.py tests/test_charge_convention.py
git commit -m "Rework n_ density-alias grammar: n_X species, n_X_nuc nucleus sum, loud guards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Rename `replace_nH` -> `expand_nuclei` (silent alias)

**Files:**
- Modify: `src/jaff/core/network/network.py` (`__init__` param + `__normalize_network_extras` param + call sites at ~275, 322, 522, 591, 615-633)
- Modify: `src/jaff/core/network/_spec.py` (lines 54, 80, 106)
- Modify: `src/jaff/core/network/_args.py` (line 29)
- Modify: `src/jaff/cli/jaffgen/_engine.py` (lines 249-250, 502-503, 649, 888; docstrings 84, 219)
- Modify: `src/jaff/cli/jaffx/_engine.py` (lines 73, 84, 205)
- Test: `tests/test_network_initialization.py` (add alias test)

- [ ] **Step 2.1: Write failing test for the alias + new name**

Add to `tests/test_network_initialization.py` (it already imports `Network`; add
`import sympy` at the top if absent):

```python
def test_expand_nuclei_alias(tmp_path):
    import sympy
    from jaff import Network

    body = "@format:idx,R,R,P,rate\n1,H,H,H2,1\n2,H2,H,H,n_H_nuc\n"
    p = tmp_path / "n.dat"
    p.write_text(body)
    # legacy replace_nH=False must behave exactly like expand_nuclei=False
    net_legacy = Network(str(p), funcfile=False, replace_nH=False)
    net_new = Network(str(p), funcfile=False, expand_nuclei=False)
    assert (
        net_legacy.reactions[1].rate
        == net_new.reactions[1].rate
        == sympy.symbols("nh_nuc")
    )
```

- [ ] **Step 2.2: Run it, verify failure**

Run: `uv run pytest tests/test_network_initialization.py::test_expand_nuclei_alias -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'expand_nuclei'`.

- [ ] **Step 2.3: Rename in `network.py` with a silent alias**

Change the `Network.__init__` signature `replace_nH: bool = True` to:

```python
        expand_nuclei: bool = True,
        replace_nH: bool | None = None,
```

At the top of `__init__` (before the value is used at ~line 275), add:

```python
        if replace_nH is not None:
            expand_nuclei = replace_nH  # silent legacy alias (alpha software)
```

Rename every downstream use from `replace_nH` to `expand_nuclei`: the
`__normalize_network_extras(replace_nH, ...)` parameter and its body
(lines 591, 615-633), the `_standardize_symbols(..., replace_nH)` calls
(already renamed to `expand_nuclei` in Task 1 signature), the
`self.spec.replace_nH` read at line 522, and the constructor's own
`self.spec` construction path. Store the flag on the spec as
`self.spec.expand_nuclei`.

- [ ] **Step 2.4: Rename in `_spec.py` and `_args.py`**

`_spec.py`: rename the `replace_nH` parameter/attribute to `expand_nuclei`
(lines 54, 80, 106). `_args.py`: rename the dataclass field
`replace_nH: bool = True` to `expand_nuclei: bool = True` (line 29).

- [ ] **Step 2.5: Rename in the CLIs**

`jaffgen/_engine.py`: rename `sn.replace_nH` -> `sn.expand_nuclei` (249-250,
502-503, 649) and the local `replace_nH=replace_nh` kwarg (888) to
`expand_nuclei=replace_nh`; update the TOML key read at 249 to accept
`expand_nuclei` (keep reading `replace_nH` too as a fallback key). Rename the
CLI option `--replace-nh/--no-replace-nh` to
`--expand-nuclei/--no-expand-nuclei` and keep `--replace-nh/--no-replace-nh` as
a hidden alias mapping to the same dest. `jaffx/_engine.py`: same
option rename at line 205; `net_args.expand_nuclei = args.expand_nuclei`
(73) and `expand_nuclei=net_args.expand_nuclei` (84).

- [ ] **Step 2.6: Run the alias test + CLI wiring tests**

Run: `uv run pytest tests/test_network_initialization.py::test_expand_nuclei_alias tests/test_cli_wiring.py tests/test_dust_radiation.py -q`
Expected: PASS. (Fix any remaining `replace_nH` references the failures point to.)

- [ ] **Step 2.7: Commit**

```bash
git add -A
git commit -m "Rename replace_nH -> expand_nuclei (replace_nH kept as silent alias)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Migrate GOW-family `.jfunc` to the `n_X` grammar

**Files:**
- Modify: `networks/GOW/GOW.jfunc`, `networks/GOW++/GOW++.jfunc`,
  `networks/GOW_scpc/GOW_scpc.jfunc`, `networks/GOW/GOW.scpc.jfunc`
- Test: existing `tests/test_codegen_e2e.py` GOW golden tests

Token mapping (apply as whole-word replacements, case-sensitive on the token as
written; do NOT touch `.jet`/`.dat`): `nH -> n_H_nuc`, `nH0 -> n_H`,
`nH2 -> n_H2`, `nHj -> n_Hj`, `ne -> n_e`, `n_H -> n_H_nuc` (only where it meant
H nuclei), `n_He -> n_He_nuc`, `n_H0 -> n_H`, `n_C0 -> n_C`, `n_O0 -> n_O`.
Leave `n_Cj`, `n_Hj`, `n_Hejj`, `n_CO` unchanged. Because `@function`
parameters are local, rename each param at its signature *and* every use in that
body *and* at the call sites passing it — work one function at a time.

- [ ] **Step 3.1: Snapshot current GOW golden output (baseline)**

Run the GOW golden test first to confirm the current baseline is green before migration:

Run: `uv run pytest "tests/test_codegen_e2e.py -k GOW" -q`
Expected: PASS (baseline). If already red, stop and investigate before migrating.

- [ ] **Step 3.2: Migrate `networks/GOW/GOW.jfunc`**

Rewrite density tokens per the mapping above, one `@function` at a time. After
each function, re-read it to confirm no bare `nH`/`nH0`/`nH2`/`ne` or
`0`-marker (`n_H0`, `n_C0`, `n_O0`) remains and that sums use `_nuc`.

- [ ] **Step 3.3: Load GOW and verify resolution is correct**

Run:
```bash
uv run python -c "
from jaff import Network
net = Network('networks/GOW/GOW.jfunc'.replace('.jfunc','.jet'), funcfile='networks/GOW/GOW.jfunc')
print('loaded', len(net.reactions), 'reactions; dEdt_chem free symbols:', net.dEdt_chem.free_symbols)
"
```
Expected: loads without `ParserError`; no stray `n_...`/`nh...` free symbols
remain in `dEdt_chem` / `dRad_dt_extra` (only `nden`, `tgas`, and legitimate
externals like `chi`, `av`, `d2g`, `gradv`, interp-function calls).

- [ ] **Step 3.4: Regenerate + verify the GOW golden**

Confirm the physics is unchanged by comparing regenerated output to the baseline
from Step 3.1. If the golden test has a regeneration switch, use it; otherwise
inspect the diff of generated RHS for GOW and confirm only expected density
substitutions changed.

Run: `uv run pytest "tests/test_codegen_e2e.py -k GOW" -q`
Expected: PASS (regenerate the golden only if the diff is exactly the intended
density-symbol substitutions and numerically equivalent).

- [ ] **Step 3.5: Migrate the remaining GOW `.jfunc` files**

Repeat Steps 3.2-3.4 for `networks/GOW++/GOW++.jfunc`,
`networks/GOW_scpc/GOW_scpc.jfunc`, `networks/GOW/GOW.scpc.jfunc`.

- [ ] **Step 3.6: Commit**

```bash
git add networks/GOW networks/GOW++ networks/GOW_scpc tests/golden
git commit -m "Migrate GOW-family .jfunc to n_X / n_X_nuc density grammar

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Full-suite verification + spec follow-up check

- [ ] **Step 4.1: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS (0 failed). In particular confirm COthin/popsicle-related and
KROME-compat paths still pass (the `nh`/`nh0`/`nh2`/`ne` layer was not touched).

- [ ] **Step 4.2: Re-check the parked mass-warning test**

The spec's follow-up: `tests/test_reaction_conservation.py::TestWarningNamesReaction::test_warning_includes_reaction_index` was failing before this work.

Run: `uv run pytest tests/test_reaction_conservation.py -q`
Expected: If green now, the grammar rework subsumed it — note that in the commit.
If still red, it is confirmed independent; open a separate systematic-debugging
pass (do NOT fold a fix into this branch's commits without its own failing test).

- [ ] **Step 4.3: Final commit if anything changed in 4.2**

```bash
git add -A
git commit -m "Verify full suite after density-alias grammar rework

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: grammar (T1), guards (T1 tests), removed-vs-kept compat (T1 leaves compat untouched), flag rename (T2), migration (T3), testing-through-text-loading (T1), follow-ups (T4.2). Covered.
- The `n_X_nuc` "known element but no bearer" case raises (loud) per spec's fail-loud intent; the empty-sum-as-0 alternative was rejected to catch typos.
- Type consistency: resolver method name `_resolve_density_alias(name, expand_nuclei)` and helpers `_element_symbol` / `_element_sum` are used consistently across T1 steps; flag name `expand_nuclei` consistent across T2.
