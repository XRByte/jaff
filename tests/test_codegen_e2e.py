# ABOUTME: End-to-end code-generation tests against committed golden references
# ABOUTME: Bitwise microphysics output + numeric rates/Jacobian + GOW all-template smoke

# Golden files live in tests/golden/.  To regenerate them after an intentional
# codegen change, run:  JAFF_UPDATE_GOLDEN=1 uv run pytest tests/test_codegen_e2e.py
# and review the diff before committing.

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.matrices.expressions.matexpr import MatrixElement

from jaff import Network
from jaff.cli import JaffGen


def _jaffgen(network_path, template, outdir, lang="cxx"):
    """Drive the ``jaffgen`` engine in-process (same path as the CLI).

    All CLI options default to ``None`` except the ones supplied here, matching
    ``jaffgen --network <net> --template <name> --outdir <dir> --lang <lang>``.
    """
    args = SimpleNamespace(
        network=str(network_path),
        config=None,
        label=None,
        funcfile=None,
        duplicate_policy=None,
        replace_nH=None,
        errors=None,
        network_config=None,
        outdir=str(outdir),
        indir=None,
        files=None,
        template=template,
        lang=lang,
    )
    JaffGen(args)

REPO = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("JAFF_UPDATE_GOLDEN") == "1"

NETWORKS = {
    "h_photo": REPO / "networks" / "h_photoionization" / "h_photo.jet",
    "GOW": REPO / "networks" / "GOW" / "GOW.jet",
}
# Default language per template (needed for extensionless files like _parameters).
ALL_TEMPLATES = {
    "python_solve_ivp": "python",
    "microphysics": "cxx",
    "fortran_dlsodes": "fortran",
    "kokkos_ode": "cxx",
}

# Fixed values for the "unknown" free variables so rates/Jacobian evaluate to
# concrete numbers.  Any scalar symbol not listed falls back to _DEFAULT_CONST.
CONST = {
    "tgas": 100.0,
    "av": 1.0,
    "chi": 1.0,
    "crate": 1.3e-17,
    "d2g": 1.0e-2,
}
_DEFAULT_CONST = 1.5

REL_TOL = 1e-12
ABS_TOL = 1e-300


# --------------------------------------------------------------------------- #
# Numeric evaluation helpers                                                   #
# --------------------------------------------------------------------------- #


def _nden_value(i: int) -> float:
    """Fixed number density for species index *i* (distinct, positive)."""
    return (i + 1) * 100.0


def _undef_value(func: AppliedUndef) -> float:
    """Fixed value for an interpolation-function call (e.g. photorates(...)).

    Keyed on the call's first argument so distinct calls get distinct values.
    """
    try:
        base = float(func.args[0])
    except (TypeError, ValueError):
        base = 1.0
    return 1.0e-11 * (base + 1.0)


def _substitutions(exprs):
    """Map every nden element, scalar symbol, and interp-function call to a value."""
    subs: dict = {}
    for e in exprs:
        if not isinstance(e, sp.Basic):
            continue
        for me in e.atoms(MatrixElement):
            subs[me] = sp.Float(_nden_value(int(me.args[1])))
        for s in e.atoms(sp.Symbol):
            subs[s] = sp.Float(CONST.get(s.name, _DEFAULT_CONST))
        for f in e.atoms(AppliedUndef):
            subs[f] = sp.Float(_undef_value(f))
    return subs


def _numeric_signature(net: Network) -> dict:
    """Evaluate reaction rates and the species Jacobian at the fixed constants.

    Returns
    -------
    dict
        ``{"rates": [float, ...], "jac": [[float, ...], ...]}`` — the rate of
        every reaction and the analytical Jacobian d(dn_i/dt)/d(n_j).
    """
    rates_expr = [r.rate for r in net.reactions]
    rate_subs = _substitutions(rates_expr)
    rates = [
        float(sp.Float(r) if not isinstance(r, sp.Basic) else r.xreplace(rate_subs))
        for r in rates_expr
    ]

    n = net.species.count
    nden = net.ndens
    ys = [sp.Symbol(f"y_{j}") for j in range(n)]
    ymap = {nden[j, 0]: ys[j] for j in range(n)}

    rhs = net.sodes()
    rhs_y = [e.xreplace(ymap) for e in rhs]
    jac = sp.Matrix(rhs_y).jacobian(ys)

    jac_subs = _substitutions(rhs)  # scalars + interp funcs (nden already y-mapped)
    jac_subs.update({ys[j]: sp.Float(_nden_value(j)) for j in range(n)})

    jac_vals = [[float(jac[i, j].xreplace(jac_subs)) for j in range(n)] for i in range(n)]
    return {"rates": rates, "jac": jac_vals}


def _assert_all_finite(sig: dict) -> None:
    assert all(math.isfinite(x) for x in sig["rates"]), "non-finite rate"
    assert all(math.isfinite(x) for row in sig["jac"] for x in row), "non-finite jac"


def _assert_close(actual, expected, label):
    assert len(actual) == len(expected), (
        f"{label}: length {len(actual)} != {len(expected)}"
    )
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
            f"{label}[{i}]: {a!r} != {b!r}"
        )


# --------------------------------------------------------------------------- #
# Test 1: bitwise microphysics output vs golden                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(NETWORKS))
def test_microphysics_output_matches_golden(name, tmp_path):
    """Generated microphysics files match the committed golden byte-for-byte."""
    _jaffgen(NETWORKS[name], template="microphysics", outdir=tmp_path)
    generated = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    gdir = GOLDEN / f"{name}_microphysics"
    if UPDATE:
        gdir.mkdir(parents=True, exist_ok=True)
        for old in gdir.iterdir():
            old.unlink()
        for fn, data in generated.items():
            (gdir / fn).write_bytes(data)
        pytest.skip(f"golden refreshed: {gdir}")

    assert gdir.is_dir(), f"no golden for {name}; run JAFF_UPDATE_GOLDEN=1"
    golden = {p.name: p.read_bytes() for p in gdir.iterdir() if p.is_file()}
    assert set(generated) == set(golden), "generated file set differs from golden"
    for fn in sorted(golden):
        assert generated[fn] == golden[fn], f"{name}/{fn} differs from golden (bitwise)"


# --------------------------------------------------------------------------- #
# Test 2: numeric rates + Jacobian vs golden                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", list(NETWORKS))
def test_rates_and_jacobian_match_golden(name):
    """Rates and Jacobian, evaluated at fixed constants, match the golden numbers."""
    sig = _numeric_signature(Network(str(NETWORKS[name])))
    _assert_all_finite(sig)

    gfile = GOLDEN / f"{name}_numeric.json"
    if UPDATE:
        gfile.parent.mkdir(parents=True, exist_ok=True)
        gfile.write_text(json.dumps(sig, indent=2))
        pytest.skip(f"golden refreshed: {gfile}")

    assert gfile.exists(), f"no golden for {name}; run JAFF_UPDATE_GOLDEN=1"
    ref = json.loads(gfile.read_text())
    _assert_close(sig["rates"], ref["rates"], "rates")
    assert len(sig["jac"]) == len(ref["jac"]), "jacobian row count differs"
    for i, (row, ref_row) in enumerate(zip(sig["jac"], ref["jac"])):
        _assert_close(row, ref_row, f"jac[{i}]")


# --------------------------------------------------------------------------- #
# Test 3: GOW generates for every template                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("template", list(ALL_TEMPLATES))
def test_gow_generates_for_all_templates(template, tmp_path):
    """The GOW network builds without error and emits non-empty files."""
    _jaffgen(NETWORKS["GOW"], template=template, outdir=tmp_path, lang=ALL_TEMPLATES[template])
    files = [p for p in tmp_path.iterdir() if p.is_file()]
    assert files, f"template {template} produced no files"
    assert all(p.stat().st_size > 0 for p in files), f"empty output file for {template}"
