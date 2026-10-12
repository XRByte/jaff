# ABOUTME: Characterization snapshots of the $JAFF template engine: render fixture templates
# ABOUTME: against small networks and compare the generated text with committed renders

# Regression net for the template-parser rewrite.  Each template under
# tests/fixtures/templates/engine/<syntax>/ is rendered with that syntax's engine for
# the networks listed in CASES and compared with the committed render in
# tests/snapshots/template_engine/<network>/.  Directive and ABOUTME lines are
# dropped from both sides before comparing (see generated_text), so a template
# rewritten in the brace syntax must reproduce the legacy render exactly.  The
# legacy syntax renders with the old engine, the brace syntax with the new one
# (directive echo off).
#
# The snapshots pin current behaviour, quirks included (SORT TRUE is a no-op,
# $IDX$ needs $idx$ on the same line, (/ ... /) is not a horizontal array).
# After an intentional change refresh them with
#   JAFF_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_template_engine_snapshots.py
# and review the diff.  Only the legacy syntax writes snapshots.

import os
from functools import cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest

from jaff import Network
from jaff.codegen._template_engine import TemplateParser as LegacyParser
from jaff.codegen.templating import TemplateParser
from jaff.physics.photo_reactions._radiation_props import RadiationProps

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = FIXTURES / "templates" / "engine"
SNAPSHOTS = Path(__file__).parent / "snapshots" / "template_engine"
UPDATE = bool(os.environ.get("JAFF_UPDATE_SNAPSHOTS"))

# Syntax variants of the same templates.  "legacy" is the reference.
SYNTAXES: List[str] = ["legacy", "brace"]
REFERENCE_SYNTAX = "legacy"

NETWORKS: Dict[str, Callable[[], Network]] = {
    # neutral H/D/He with an energy equation (dE/dt from the .jfunc)
    "dedt": lambda: Network(str(FIXTURES / "test_jac_dedt.dat")),
    # charged species and electrons (He++), no radiation
    "cie_hepp": lambda: Network(str(FIXTURES / "react_cie_hepp.jet")),
    # photoionisation with one radiation band
    "h_photo": lambda: Network(
        str(REPO / "networks" / "h_photoionization" / "h_photo.jet"),
        radiation_props=RadiationProps(
            bands=[13.6, float("inf")], profile_index=0, mode="nph"
        ),
    ),
}

ALL = list(NETWORKS)
CASES: Dict[str, List[str]] = {
    "directives.py": ALL,
    "repeat_lists.py": ALL,
    "repeat_expressions.py": ALL,
    "reduce.py": ALL,
    "language.cpp": ALL,
    "language.f90": ALL,
    "modifiers_thermal.py": ["dedt"],
    "modifiers_radiation.py": ["h_photo"],
}


@cache
def network(name: str) -> Network:
    return NETWORKS[name]()


def render(syntax: str, template: str, net: str) -> str:
    path = TEMPLATES / syntax / template
    if syntax == REFERENCE_SYNTAX:
        return LegacyParser(network(net), path).parse_file()

    return TemplateParser(network(net), path, echo_directives=False).parse_file()


def generated_text(rendered: str) -> List[str]:
    """Lines produced by the engine, without directive and ABOUTME lines.

    Directive and END lines are echoed into the legacy output verbatim and each
    fixture's ABOUTME header names its syntax, so those differ between syntaxes
    while everything else must not.
    """
    return [
        line
        for line in rendered.splitlines()
        if "$JAFF" not in line and "ABOUTME:" not in line
    ]


def _params() -> List[Tuple[str, str]]:
    return [(t, n) for t, nets in CASES.items() for n in nets]


@pytest.mark.parametrize("syntax", SYNTAXES)
@pytest.mark.parametrize("template, net", _params())
def test_render_matches_snapshot(syntax: str, template: str, net: str) -> None:
    template_path = TEMPLATES / syntax / template
    if not template_path.exists():
        pytest.skip(f"{syntax} has no {template} yet")

    snapshot = SNAPSHOTS / net / template
    output = render(syntax, template, net)
    if UPDATE and syntax == REFERENCE_SYNTAX:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(output)

    if syntax != REFERENCE_SYNTAX:
        # Rendered with echo off, so no directive may leak into the output.
        leaked = [line for line in output.splitlines() if "$JAFF" in line]
        assert not leaked, f"directive lines in output: {leaked}"

    assert snapshot.exists(), f"no snapshot {snapshot}; run JAFF_UPDATE_SNAPSHOTS=1"
    assert generated_text(output) == generated_text(snapshot.read_text())


def test_snapshot_set_matches_cases() -> None:
    expected = {(n, t) for t, nets in CASES.items() for n in nets}
    found = {(p.parent.name, p.name) for p in SNAPSHOTS.glob("*/*") if p.is_file()}
    assert found == expected, "stale or missing snapshots; run JAFF_UPDATE_SNAPSHOTS=1"


def test_every_reference_template_has_cases() -> None:
    templates = {p.name for p in (TEMPLATES / REFERENCE_SYNTAX).iterdir()}
    assert templates == set(CASES)


def test_other_syntaxes_only_hold_known_templates() -> None:
    for syntax in SYNTAXES:
        templates = {p.name for p in (TEMPLATES / syntax).iterdir()}
        assert templates <= set(CASES), f"{syntax} has templates without CASES"
