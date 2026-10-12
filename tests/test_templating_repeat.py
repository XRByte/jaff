# ABOUTME: Tests for the REPEAT command rendered through the new engine's TemplateParser
# ABOUTME: Vertical/horizontal expansion, the old engine's quirks, CSE naming and options

from pathlib import Path

import pytest

from jaff import Network
from jaff.codegen.codegen import Codegen
from jaff.codegen.templating import TemplateParser
from jaff.errors import ParserError
from jaff.physics.photo_reactions._radiation_props import RadiationProps

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def net() -> Network:
    # species H, e-, H+, He++, He+
    return Network(str(FIXTURES / "react_cie_hepp.jet"))


def render(net, tmp_path, text) -> str:
    template = tmp_path / "t.py"
    template.write_text(text)
    return TemplateParser(net, template, echo_directives=False).parse_file()


def block(header: str, *body: str) -> str:
    return "\n".join([f"$$JAFF {header}", *body, "$$JAFF END", ""])


def test_vertical_writes_one_line_per_item(net, tmp_path):
    out = render(
        net, tmp_path, block("REPEAT idx, specie IN species", '  s[$idx$] = "$specie$"')
    )
    assert out == "".join(
        f'  s[{i}] = "{name}"\n'
        for i, name in enumerate(["H", "e-", "H+", "He++", "He+"])
    )


def test_horizontal_without_idx_builds_an_inline_list(net, tmp_path):
    out = render(net, tmp_path, block("REPEAT specie IN species", 'n = ["$specie$", ]'))
    assert out == 'n = ["H", "e-", "H+", "He++", "He+"]\n'


NAMES = ["H", "e-", "H+", "He++", "He+"]


def test_line_naming_a_variable_without_a_placeholder_is_copied_once(net, tmp_path):
    text = block("REPEAT idx, specie IN species", "# one line per specie", "x")
    assert render(net, tmp_path, text) == "# one line per specie\nx\n"


def test_upper_idx_alone_is_repeated(net, tmp_path):
    text = block("REPEAT idx, specie IN species", "u[$IDX$] = $specie$")
    expected = "".join(f"u[{i}] = {name}\n" for i, name in enumerate(NAMES))
    assert render(net, tmp_path, text) == expected


def test_idx_without_the_item_is_repeated(net, tmp_path):
    text = block("REPEAT idx, specie IN species", "a[$idx$] = 0")
    assert render(net, tmp_path, text) == "".join(f"a[{i}] = 0\n" for i in range(5))


def test_item_without_idx_is_repeated(net, tmp_path):
    text = block("REPEAT idx, specie IN species", "print($specie$)")
    assert render(net, tmp_path, text) == "".join(f"print({n})\n" for n in NAMES)


def test_horizontal_line_without_a_bracketed_placeholder_is_copied(net, tmp_path):
    text = block('REPEAT specie IN species { REPLACE { "n" -> "N" } }', "  n = $specie$")
    assert render(net, tmp_path, text) == "  n = $specie$\n"


def test_vertical_replace_sees_the_indentation(net, tmp_path):
    text = block(
        'REPEAT idx, specie IN species { REPLACE { "  s" -> "S" } }',
        "  s[$idx$] = $specie$",
    )
    assert render(net, tmp_path, text) == (
        "S[0] = H\nS[1] = e-\nS[2] = H+\nS[3] = He++\nS[4] = He+\n"
    )


def test_horizontal_replace_sees_only_the_content(net, tmp_path):
    text = block(
        'REPEAT specie IN species { REPLACE { "^n" -> "N" } }', "  n = [$specie$]"
    )
    assert render(net, tmp_path, text) == "  N = [H, e-, H+, He++, He+]\n"


def test_cse_block_names_temporaries_and_ends_with_a_blank_line(net, tmp_path):
    odes = Codegen(network=net, lang="python").get_indexed_odes(cse_var="t")
    text = block("REPEAT idx, ode, cse IN odes", "t$idx$ = $cse$", "dy[$idx$] = $ode$")
    temporaries = "".join(f"t{i[0]} = {v}\n" for i, v in odes["extras"]["cse"])
    odes_lines = "".join(f"dy[{i[0]}] = {v}\n" for i, v in odes["expressions"])
    assert render(net, tmp_path, text) == temporaries + "\n" + odes_lines


@pytest.mark.parametrize(
    "body, message",
    [
        (["k[$idx$] = $rate$", "c$idx$ = $cse$"], "must be the first line"),
        (["c$idx+1$ = $cse$"], "cannot use an offset index"),
        (["[$idx$] = $cse$"], "must be an identifier"),
    ],
)
def test_bad_cse_line_is_an_error_at_that_line(net, tmp_path, body, message):
    text = block("REPEAT idx, rate, cse IN rates", "x", *body)
    with pytest.raises(ParserError, match=message) as err:
        render(net, tmp_path, text)
    assert err.value.nline == 3


def test_idx_count_must_match_the_dimension(net, tmp_path):
    text = block("REPEAT idx, element IN element_density_matrix", "d[$idx$] = $element$")
    with pytest.raises(ParserError, match="expected 2 \\$idx\\$ placeholders") as err:
        render(net, tmp_path, text)
    assert err.value.nline == 2


def test_unknown_property_and_variable_are_rejected(net, tmp_path):
    with pytest.raises(ParserError, match="REPEAT has no property specis"):
        render(net, tmp_path, block("REPEAT idx IN specis", "x"))
    with pytest.raises(ParserError, match="REPEAT IN species has no variable spec"):
        render(net, tmp_path, block("REPEAT idx, spec IN species", "x"))


def test_option_group_from_use_is_checked_against_the_property(net, tmp_path):
    text = "$$JAFF DEFINE t { THERMAL { MODE -> dtdt } }\n" + block(
        "REPEAT idx, rate IN rates {\n    USE t\n}", "k[$idx$] = $rate$"
    )
    with pytest.raises(ParserError, match="REPEAT IN rates takes no option group") as e:
        render(net, tmp_path, text)
    assert e.value.nline == 3


def test_sign_options_reach_normalized_names(net, tmp_path):
    text = block(
        "REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign "
        "{ SIGN { POS -> p  NEG -> m } }",
        "i_$specie_with_normalized_sign$ = $idx$",
    )
    names = net.species.normalized_names(pos="p", neg="m")
    assert render(net, tmp_path, text) == "".join(
        f"i_{name} = {i}\n" for i, name in enumerate(names)
    )


def test_radodes_takes_the_radiation_order(tmp_path):
    photo = Network(
        str(REPO / "networks" / "h_photoionization" / "h_photo.jet"),
        radiation_props=RadiationProps(
            bands=[13.6, float("inf")], profile_index=0, mode="nph"
        ),
    )
    radodes = Codegen(network=photo, lang="python").get_indexed_radodes(
        order=1, use_cse=False
    )
    text = block(
        "REPEAT idx, radode IN radodes { RADIATION { ORDER -> 1 } }",
        "r[$idx$] = $radode$",
    )
    assert render(photo, tmp_path, text) == "".join(
        f"r[{i[0]}] = {v}\n" for i, v in radodes["expressions"]
    )
