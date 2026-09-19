# ABOUTME: Tests for RateSegments.evaluate_equivalent_rate multi-range assembly
# ABOUTME: Constant/temperature-dependent ranges, adjacent+separated, both clip modes

from pathlib import Path

import pytest
from sympy import Integer, Rational, symbols

from jaff import Network
from jaff.core.reaction.types import RateSegment, RateSegments

TGAS = symbols("tgas")


def _eval(segments, mode, t):
    """Assemble the piecewise rate and evaluate it at temperature *t*."""
    expr = RateSegments(list(segments), mode).sort().evaluate_equivalent_rate("r")
    return expr.subs(TGAS, t) if hasattr(expr, "subs") else expr


# --------------------------------------------------------------------------- #
# two constant ranges                                                         #
# --------------------------------------------------------------------------- #
def test_two_constant_adjacent_clip():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(Integer(2), 100, 200)]
    assert _eval(segs, "clip", 50) == 1
    assert _eval(segs, "clip", 150) == 2  # bug: returned 1
    assert _eval(segs, "clip", 5) == 1    # clipped low
    assert _eval(segs, "clip", 300) == 2  # clipped high


def test_two_constant_adjacent_extrapolate():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(Integer(2), 100, 200)]
    assert _eval(segs, "extrapolate", 50) == 1
    assert _eval(segs, "extrapolate", 150) == 2
    assert _eval(segs, "extrapolate", 5) == 1
    assert _eval(segs, "extrapolate", 300) == 2


def test_two_constant_separated_clip_interpolates_gap():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(Integer(2), 150, 250)]
    assert _eval(segs, "clip", 50) == 1
    assert _eval(segs, "clip", 200) == 2
    # gap [100, 150]: linear interp, midpoint 125 -> 1.5
    assert _eval(segs, "clip", 125) == Rational(3, 2)


def test_two_constant_separated_extrapolate():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(Integer(2), 150, 250)]
    assert _eval(segs, "extrapolate", 50) == 1
    assert _eval(segs, "extrapolate", 200) == 2
    assert _eval(segs, "extrapolate", 125) == Rational(3, 2)


# --------------------------------------------------------------------------- #
# constant followed by temperature-dependent range                           #
# --------------------------------------------------------------------------- #
def test_constant_then_tempdep_adjacent_clip():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(TGAS, 100, 200)]
    assert _eval(segs, "clip", 50) == 1
    assert _eval(segs, "clip", 150) == 150
    assert _eval(segs, "clip", 5) == 1     # clipped low -> constant
    assert _eval(segs, "clip", 300) == 200  # clipped high -> tmax value


def test_constant_then_tempdep_separated_extrapolate():
    segs = [RateSegment(Integer(1), 10, 100), RateSegment(TGAS, 150, 250)]
    assert _eval(segs, "extrapolate", 50) == 1
    assert _eval(segs, "extrapolate", 200) == 200
    # gap [100,150]: interp between const 1 and tgas; at 125 ->
    # (1*(150-125) + 125*(125-100)) / (150-100) = (25 + 3125)/50 = 63
    assert _eval(segs, "extrapolate", 125) == Rational(25 + 125 * 25, 50)


# --------------------------------------------------------------------------- #
# Network-level regression (the reported repro)                               #
# --------------------------------------------------------------------------- #
def test_two_touching_tempdep_clip():
    """Two touching temperature-dependent ranges assemble each body (issue 045)."""
    segs = [RateSegment(TGAS, 10, 100), RateSegment(2 * TGAS, 100, 200)]
    assert _eval(segs, "clip", 50) == 50
    assert _eval(segs, "clip", 150) == 300   # bug returned 400 (upper clip)
    assert _eval(segs, "clip", 100) == 200   # shared boundary -> upper range
    assert _eval(segs, "clip", 5) == 10      # clipped low -> first tmin value
    assert _eval(segs, "clip", 300) == 400   # clipped high -> last tmax value


def test_two_touching_tempdep_extrapolate():
    segs = [RateSegment(TGAS, 10, 100), RateSegment(2 * TGAS, 100, 200)]
    assert _eval(segs, "extrapolate", 50) == 50
    assert _eval(segs, "extrapolate", 150) == 300
    assert _eval(segs, "extrapolate", 100) == 200
    assert _eval(segs, "extrapolate", 5) == 5      # first body extends down
    assert _eval(segs, "extrapolate", 300) == 600  # last body extends up


def test_three_touching_tempdep_clip():
    segs = [
        RateSegment(TGAS, 10, 100),
        RateSegment(2 * TGAS, 100, 200),
        RateSegment(3 * TGAS, 200, 300),
    ]
    # inside every segment
    assert _eval(segs, "clip", 50) == 50
    assert _eval(segs, "clip", 150) == 300
    assert _eval(segs, "clip", 250) == 750
    # shared boundaries -> upper range governs (strict tgas < tmax)
    assert _eval(segs, "clip", 100) == 200
    assert _eval(segs, "clip", 200) == 600
    # clipped ends
    assert _eval(segs, "clip", 5) == 10
    assert _eval(segs, "clip", 350) == 900


def test_three_touching_tempdep_extrapolate():
    segs = [
        RateSegment(TGAS, 10, 100),
        RateSegment(2 * TGAS, 100, 200),
        RateSegment(3 * TGAS, 200, 300),
    ]
    assert _eval(segs, "extrapolate", 50) == 50
    assert _eval(segs, "extrapolate", 150) == 300
    assert _eval(segs, "extrapolate", 250) == 750
    assert _eval(segs, "extrapolate", 100) == 200
    assert _eval(segs, "extrapolate", 200) == 600
    assert _eval(segs, "extrapolate", 5) == 5       # first extends down
    assert _eval(segs, "extrapolate", 350) == 1050  # last extends up


def test_network_two_touching_tempdep_ranges(tmp_path):
    dat = tmp_path / "tempdep.dat"
    dat.write_text(
        "@format:idx,R,R,P,tmin,tmax,rate\n"
        "1,H,H,H2,10,100,tgas\n"
        "2,H,H,H2,100,200,2*tgas\n"
    )
    net = Network(str(dat), funcfile=False)
    assert net.reactions[0].rate.subs(TGAS, 150) == 300


def test_network_two_constant_ranges(tmp_path):
    dat = tmp_path / "constant.dat"
    dat.write_text(
        "@format:idx,R,R,P,tmin,tmax,rate\n"
        "1,H,H,H2,10,100,1\n"
        "2,H,H,H2,100,200,2\n"
    )
    net = Network(str(dat), funcfile=False)
    rate = net.reactions[0].rate
    assert rate.subs(TGAS, 150) == 2
    assert rate.subs(TGAS, 50) == 1
