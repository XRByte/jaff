# ABOUTME: Tests for jaff.plotting.themes — style/palette resolution and cycling
# ABOUTME: Built-ins, partial dict overrides, label colour maps, registration

import pytest

from jaff.plotting import themes
from jaff.plotting.spec import SeriesSpec
from jaff.plotting.themes import (
    PALETTES,
    STYLES,
    assign,
    is_marker,
    register_palette,
    register_style,
    resolve_palette,
    resolve_style,
)


@pytest.fixture(autouse=True)
def _restore_registries():
    """Undo registrations made by a test."""
    styles, palettes = dict(STYLES), dict(PALETTES)
    yield
    STYLES.clear()
    STYLES.update(styles)
    PALETTES.clear()
    PALETTES.update(palettes)


def _s(**kw):
    return SeriesSpec.from_dict({"x": [1.0, 2.0], "y": [1.0, 2.0], **kw})


class TestResolveStyle:
    def test_default_is_paper(self):
        assert resolve_style(None) is STYLES["paper"]

    def test_by_name(self):
        assert resolve_style("talk") is STYLES["talk"]

    def test_unknown_name_lists_available(self):
        with pytest.raises(ValueError, match=r"unknown style 'nope'.*paper"):
            resolve_style("nope")

    def test_partial_dict_merges_nested_and_replaces_flat(self):
        s = resolve_style({"base": "talk", "lw": 1.2, "rc": {"font.size": 10}})
        assert s.lw == 1.2
        assert s.rc["font.size"] == 10
        assert s.rc["axes.labelsize"] == STYLES["talk"].rc["axes.labelsize"]
        assert s.ticks == STYLES["talk"].ticks

    def test_dict_base_defaults_to_paper(self):
        s = resolve_style({"cycle": ["-", "o"]})
        assert s.cycle == ("-", "o")
        assert s.panel_size == STYLES["paper"].panel_size

    def test_unknown_field(self):
        with pytest.raises(ValueError, match="unknown field"):
            resolve_style({"linewidth": 2})


@pytest.mark.parametrize(
    "bad",
    [
        {"rc": 5},
        {"cycle": "--"},
        {"cycle": []},
        {"panel_size": 3},
        {"legend_max": 0},
        {"legend_max": 2.5},
        {"legend_max": True},
    ],
)
def test_style_dict_value_validation(bad):
    field = next(iter(bad))
    with pytest.raises(ValueError, match=f"style.{field}"):
        resolve_style(bad)


def test_legend_not_shared_between_builtins():
    legend = STYLES["paper"].legend["title_fontproperties"]
    assert legend is not STYLES["talk"].legend["title_fontproperties"]


def test_unsupported_types():
    with pytest.raises(ValueError, match="expected a name, dict or Style"):
        resolve_style(3)
    with pytest.raises(ValueError, match="palette"):
        resolve_palette(3)


def test_assign_empty_cycle():
    with pytest.raises(ValueError, match="colour cycle is empty"):
        assign([_s()], STYLES["paper"], themes.Palette((), {}))


class TestResolvePalette:
    def test_default(self):
        assert resolve_palette(None) is PALETTES["jaff"]

    def test_list(self):
        p = resolve_palette(["red", "blue"])
        assert p.cycle == ("red", "blue") and p.mapping == {}

    def test_empty_list(self):
        with pytest.raises(ValueError, match="empty"):
            resolve_palette([])

    def test_label_dict_uses_default_cycle(self):
        p = resolve_palette({"ref": "#4d4d4d"})
        assert p.mapping == {"ref": "#4d4d4d"}
        assert p.cycle == PALETTES["jaff"].cycle

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="unknown palette"):
            resolve_palette("rainbow")


class TestRegistration:
    def test_register_style(self):
        register_style("mine", {"base": "talk", "lw": 3})
        assert resolve_style("mine").lw == 3

    def test_register_palette(self):
        register_palette("mine", ["#111111"])
        assert resolve_palette("mine").cycle == ("#111111",)

    def test_cannot_overwrite_builtins(self):
        with pytest.raises(ValueError, match="built-in"):
            register_style("paper", {})
        with pytest.raises(ValueError, match="built-in"):
            register_palette("okabe-ito", ["red"])


class TestAssign:
    def test_colour_first_then_style(self):
        style = resolve_style({"cycle": ["-", "--", "o"]})
        palette = resolve_palette(["r", "g"])
        looks = assign([_s() for _ in range(5)], style, palette)
        assert looks == [("r", "-"), ("g", "-"), ("r", "--"), ("g", "--"), ("r", "o")]

    def test_style_cycle_clamps_at_last(self):
        style = resolve_style({"cycle": ["-"]})
        looks = assign([_s() for _ in range(3)], style, resolve_palette(["r"]))
        assert [ls for _, ls in looks] == ["-", "-", "-"]

    def test_explicit_wins_and_does_not_consume_cycle(self):
        style = resolve_style({"cycle": ["-", ":"]})
        palette = resolve_palette(["r", "g"])
        looks = assign([_s(color="k", style="o"), _s(), _s()], style, palette)
        assert looks == [("k", "o"), ("r", "-"), ("g", "-")]

    def test_label_mapping(self):
        palette = resolve_palette({"ref": "#4d4d4d"})
        looks = assign([_s(label="ref"), _s(label="sim")], STYLES["paper"], palette)
        assert looks[0][0] == "#4d4d4d"
        assert looks[1][0] == palette.cycle[0]


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("o", True),
        ("x", True),
        ("D", True),
        ("-", False),
        ("--", False),
        (":", False),
        ((0, (6, 3)), False),
        ("", False),
        ("None", False),
    ],
)
def test_is_marker(style, expected):
    assert is_marker(style) is expected


def test_builtin_names():
    assert {"paper", "talk"} <= set(themes.STYLES)
    assert {
        "okabe-ito",
        "sunset",
        "vivid",
        "jaff",
        "logo",
        "gow",
        "muted",
        "deep",
    } <= set(themes.PALETTES)


def test_assign_bars_do_not_consume_style_slot():
    style = resolve_style({"cycle": ["-", "--", ":"]})
    bars = _s(kind="bars", x_lo=[0.0, 1.0], x_hi=[1.0, 2.0])
    looks = assign([_s(), bars, _s()], style, resolve_palette(["r"]))
    assert looks[0][1] == "-" and looks[2][1] == "--"


def test_legend_max_default_and_override():
    assert STYLES["paper"].legend_max == 12
    assert STYLES["talk"].legend_max == 12
    assert resolve_style({"legend_max": 20}).legend_max == 20
