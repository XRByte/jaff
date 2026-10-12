# ABOUTME: Tests for jaff.plotting.spec — dict validation into frozen dataclasses
# ABOUTME: Defaults, coercion, and every ValueError path with its spec path

import numpy as np
import pytest

from jaff.plotting.spec import AxesSpec, FigureSpec, SeriesSpec


def _series(**kw):
    return {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], **kw}


def _figure(**kw):
    return {"axes": [{"series": [_series()]}], **kw}


class TestSeriesSpec:
    def test_defaults(self):
        s = SeriesSpec.from_dict(_series())
        assert s.label == "" and s.color is None and s.style is None
        assert s.kind == "line" and s.x_lo is None

    def test_arrays_are_readonly_float_copies(self):
        x = [1, 2, 3]
        s = SeriesSpec.from_dict(_series(x=x))
        assert s.x.dtype == float and not s.x.flags.writeable
        x[0] = 99
        assert s.x[0] == 1.0

    def test_length_mismatch(self):
        with pytest.raises(
            ValueError, match=r"^series: x and y lengths differ \(3 vs 2\)"
        ):
            SeriesSpec.from_dict(_series(y=[1.0, 2.0]))

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown key"):
            SeriesSpec.from_dict(_series(colour="red"))

    def test_missing_y(self):
        with pytest.raises(ValueError, match="missing required key 'y'"):
            SeriesSpec.from_dict({"x": [1.0]})

    def test_not_1d(self):
        with pytest.raises(ValueError, match="expected 1-D"):
            SeriesSpec.from_dict(_series(x=np.ones((3, 1))))

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="not numeric"):
            SeriesSpec.from_dict(_series(x=["a", "b", "c"]))

    def test_bars_need_edges(self):
        with pytest.raises(ValueError, match="requires 'x_lo' and 'x_hi'"):
            SeriesSpec.from_dict(_series(kind="bars"))

    def test_edges_only_for_bars(self):
        with pytest.raises(ValueError, match="only valid for kind 'bars'"):
            SeriesSpec.from_dict(_series(x_lo=[0, 1, 2], x_hi=[1, 2, 3]))

    def test_bars_ok(self):
        s = SeriesSpec.from_dict(
            _series(kind="bars", x_lo=[0, 1, 2], x_hi=[1, 2, np.inf])
        )
        assert s.kind == "bars" and np.isinf(s.x_hi[-1])

    def test_bad_kind(self):
        with pytest.raises(ValueError, match=r"series\.kind: 'area' not in"):
            SeriesSpec.from_dict(_series(kind="area"))


class TestAxesSpec:
    def test_defaults(self):
        a = AxesSpec.from_dict({"series": []})
        assert a.xscale == "linear" and a.tag is None and a.legend is None
        assert a.title == ""

    def test_bad_scale_path(self):
        with pytest.raises(ValueError, match=r"^axes\.yscale: 'symlog' not in"):
            AxesSpec.from_dict({"series": [], "yscale": "symlog"})

    def test_bad_legend(self):
        with pytest.raises(ValueError, match="legend"):
            AxesSpec.from_dict({"series": [], "legend": "figure"})

    def test_series_must_be_list(self):
        with pytest.raises(ValueError, match="expected a list"):
            AxesSpec.from_dict({"series": _series()})

    def test_limits_coerced(self):
        a = AxesSpec.from_dict({"series": [], "xlim": [1, 10]})
        assert a.xlim == (1.0, 10.0)

    def test_bad_limits(self):
        with pytest.raises(ValueError, match="pair of numbers"):
            AxesSpec.from_dict({"series": [], "ylim": 5})


class TestFigureSpec:
    def test_defaults(self):
        f = FigureSpec.from_dict(_figure())
        assert f.grid == (1, 1) and f.legend == "auto"
        assert f.style is None and f.palette is None and f.title is None

    def test_nested_error_path(self):
        spec = {"grid": (1, 2), "axes": [{"series": []}, {"series": [_series(y=[1.0])]}]}
        with pytest.raises(ValueError, match=r"^axes\[1\]\.series\[0\]: x and y"):
            FigureSpec.from_dict(spec)

    def test_too_many_axes(self):
        spec = {"axes": [{"series": []}, {"series": []}]}
        with pytest.raises(ValueError, match=r"2 axes do not fit grid \(1, 1\)"):
            FigureSpec.from_dict(spec)

    @pytest.mark.parametrize("grid", [(0, 1), (1.5, 2), (True, 1), 3, (1, 2, 3)])
    def test_bad_grid(self, grid):
        with pytest.raises(ValueError, match="figure.grid"):
            FigureSpec.from_dict(_figure(grid=grid))

    def test_unknown_figure_key(self):
        with pytest.raises(ValueError, match="^figure: unknown key"):
            FigureSpec.from_dict(_figure(layout="subplots"))

    def test_bad_legend_policy(self):
        with pytest.raises(ValueError, match="figure.legend"):
            FigureSpec.from_dict(_figure(legend="outside"))


class TestHardening:
    def test_none_text_becomes_empty(self):
        s = SeriesSpec.from_dict(_series(label=None))
        assert s.label == ""
        a = AxesSpec.from_dict({"series": [], "xlabel": None, "title": None})
        assert a.xlabel == "" and a.title == ""

    def test_pair_rejects_str(self):
        with pytest.raises(ValueError, match="expected a pair of numbers"):
            AxesSpec.from_dict({"series": [], "xlim": "12"})

    def test_bad_color(self):
        with pytest.raises(ValueError, match=r"series\.color: expected a colour string"):
            SeriesSpec.from_dict(_series(color=0))

    def test_bad_style(self):
        with pytest.raises(ValueError, match=r"series\.style"):
            SeriesSpec.from_dict(_series(style=["-"]))

    @pytest.mark.parametrize("key", ["sharex", "sharey"])
    def test_share_must_be_bool(self, key):
        with pytest.raises(ValueError, match=f"figure.{key}"):
            FigureSpec.from_dict(_figure(**{key: "False"}))

    def test_unknown_style_name(self):
        with pytest.raises(ValueError, match=r"figure\.style: .*unknown style"):
            FigureSpec.from_dict(_figure(style="nope"))

    def test_unknown_palette_name(self):
        with pytest.raises(ValueError, match=r"figure\.palette: .*unknown palette"):
            FigureSpec.from_dict(_figure(palette="nope"))

    def test_style_is_deep_copied(self):
        style = {"rc": {"font.size": 10}}
        f = FigureSpec.from_dict(_figure(style=style))
        style["rc"]["font.size"] = 99
        assert f.style["rc"]["font.size"] == 10


@pytest.mark.parametrize(("key", "scale"), [("sharex", "xscale"), ("sharey", "yscale")])
def test_shared_axes_need_same_scale(key, scale):
    axes = [{"series": []}, {"series": [], scale: "log"}]
    with pytest.raises(ValueError, match=f"figure.{key}: all axes must use the same"):
        FigureSpec.from_dict({"axes": axes, "grid": (1, 2), key: True})
