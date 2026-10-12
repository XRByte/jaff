# ABOUTME: Tests for rendering whole templates with the new engine's TemplateParser
# ABOUTME: Echo of directives, DEFINE/USE layering, validation errors and the facade

from pathlib import Path

import pytest

from jaff import Network
from jaff.codegen.templating import TemplateParser
from jaff.errors import ParserError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def net() -> Network:
    return Network(str(FIXTURES / "react_cie_hepp.jet"))


def render(net, tmp_path, text, name="t.py", echo=False, **kw) -> str:
    template = tmp_path / name
    template.write_text(text)
    return TemplateParser(net, template, echo_directives=echo, **kw).parse_file()


def test_text_outside_directives_is_copied_verbatim(net, tmp_path):
    text = "a = $nspec$\n\t# comment  \n"
    assert render(net, tmp_path, text) == text


def test_sub_block_without_echo(net, tmp_path):
    text = "x\n$$JAFF SUB nspec\nn = $nspec$\n$$JAFF END\ny\n"
    assert render(net, tmp_path, text) == "x\nn = 5\ny\n"


@pytest.mark.parametrize(
    "name, comment", [("t.py", "#"), ("t.cpp", "//"), ("t.f90", "!")]
)
def test_echo_comments_header_and_end_in_the_language(net, tmp_path, name, comment):
    text = '  $$JAFF SUB nspec {\n      REPLACE { "5" -> "five" }\n  }\nn = $nspec$\n  $$JAFF END\n'
    out = render(net, tmp_path, text, name=name, echo=True)
    assert out == (
        f"  {comment} $$JAFF SUB nspec {{\n"
        f'      {comment} REPLACE {{ "5" -> "five" }}\n'
        f"  {comment} }}\n"
        "n = five\n"
        f"  {comment} $$JAFF END\n"
    )


def test_echo_adds_missing_final_newline(net, tmp_path):
    out = render(net, tmp_path, "$$JAFF DEFINE m { }", echo=True)
    assert out == "# $$JAFF DEFINE m { }\n"


def test_use_pulls_in_a_define(net, tmp_path):
    text = (
        '$$JAFF DEFINE up {\n    REPLACE { "n" -> "N" }\n}\n'
        "$$JAFF SUB nspec { USE up }\nn = $nspec$\n$$JAFF END\n"
    )
    assert render(net, tmp_path, text) == "N = 5\n"


def test_use_before_define_is_an_error_at_the_use_line(net, tmp_path):
    text = (
        "$$JAFF SUB nspec {\n    USE up\n}\nx\n$$JAFF END\n"
        '$$JAFF DEFINE up { REPLACE { "n" -> "N" } }\n'
    )
    with pytest.raises(ParserError, match="undefined option set up") as err:
        render(net, tmp_path, text)
    assert err.value.nline == 2
    assert err.value.line == "    USE up"


def test_define_twice_is_an_error(net, tmp_path):
    with pytest.raises(ParserError, match="DEFINE m given twice") as err:
        render(net, tmp_path, "$$JAFF DEFINE m { }\n$$JAFF DEFINE m { }\n")
    assert err.value.nline == 2


def test_group_a_command_does_not_take_is_an_error_at_its_line(net, tmp_path):
    text = "$$JAFF SUB nspec {\n    SORT { ENABLED -> True }\n}\nx\n$$JAFF END\n"
    with pytest.raises(ParserError, match="SUB takes no option group SORT") as err:
        render(net, tmp_path, text)
    assert err.value.nline == 2


def test_group_from_use_is_checked_too(net, tmp_path):
    text = (
        "$$JAFF DEFINE m { SORT { ENABLED -> True } }\n"
        "$$JAFF SUB nspec { USE m }\nx\n$$JAFF END\n"
    )
    with pytest.raises(ParserError, match="takes no option group SORT"):
        render(net, tmp_path, text)


def test_unknown_token_is_an_error_at_the_header(net, tmp_path):
    with pytest.raises(ParserError, match="SUB has no token nspecs") as err:
        render(net, tmp_path, "a\n$$JAFF SUB nspecs\nx\n$$JAFF END\n")
    assert err.value.nline == 2


def test_every_block_is_validated_before_rendering(net, tmp_path, monkeypatch):
    from jaff.codegen.templating.commands import Sub

    calls = []
    monkeypatch.setattr(Sub, "render", lambda *a: calls.append(1) or "")
    text = "$$JAFF SUB nspec\nx\n$$JAFF END\n$$JAFF SUB bogus\ny\n$$JAFF END\n"
    with pytest.raises(ParserError, match="bogus"):
        render(net, tmp_path, text)
    assert calls == []


def test_get_and_has_render_through_the_facade(net, tmp_path):
    text = (
        "$$JAFF GET specie_idx IN H+\ni = $specie_idx$\n$$JAFF END\n"
        "$$JAFF HAS specie CO\nh = $specie$\n$$JAFF END\n"
    )
    assert render(net, tmp_path, text) == "i = 2\nh = 0\n"


def test_unknown_extension_needs_a_default_language(net, tmp_path):
    with pytest.raises(RuntimeError, match="xyz files are not yet supported"):
        render(net, tmp_path, "x\n", name="t.xyz")
    assert render(net, tmp_path, "x\n", name="t.xyz", default_lang="python") == "x\n"


def test_get_of_unknown_entity_is_a_parser_error_at_the_header(net, tmp_path):
    text = "x\n$$JAFF GET specie_idx IN Xyz\ni = $specie_idx$\n$$JAFF END\n"
    with pytest.raises(ParserError, match="Xyz is not one") as err:
        render(net, tmp_path, text)
    assert err.value.nline == 2


def test_replace_patterns_are_compiled_once_per_block(net, tmp_path, monkeypatch):
    import jaff.codegen.templating.commands.base as base
    import jaff.codegen.templating.render as render_mod

    calls = []
    original = render_mod.compile_replacements

    def counting(options):
        calls.append(options)
        return original(options)

    monkeypatch.setattr(render_mod, "compile_replacements", counting)
    monkeypatch.setattr(base, "compile_replacements", counting, raising=False)

    text = '$$JAFF SUB nspec { REPLACE { "5" -> "five" } }\nn = $nspec$\n$$JAFF END\n'
    assert render(net, tmp_path, text) == "n = five\n"
    assert len(calls) == 1
