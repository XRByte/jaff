# ABOUTME: Tests for the integrator helpers in jaff.common._integrators.
# ABOUTME: smart_integrate dispatches arrays to arr_integrate and SymPy to sym_integrate.
import numpy as np
import pytest
import sympy as sp

from jaff.common import arr_integrate, smart_integrate, sym_integrate

E = sp.Symbol("E")


def test_array_input_matches_arr_integrate():
    x = np.linspace(1.0, 3.0, 201)
    y = x**2
    assert smart_integrate(y, x, (1.5, 2.5)) == arr_integrate(y, x, (1.5, 2.5))


def test_expression_input_matches_sym_integrate():
    expr = sp.Piecewise((E**2, (E >= 1) & (E <= 3)), (0, True))
    got = smart_integrate(expr, E, (0.0, sp.oo))
    assert got == sym_integrate(expr, E, (0.0, sp.oo))
    assert got == pytest.approx(26.0 / 3.0, rel=1e-10)


def test_array_with_symbolic_x_raises():
    with pytest.raises(TypeError, match="ndarray"):
        smart_integrate(np.ones(3), E, (0.0, 1.0))


def test_expression_with_array_x_raises():
    with pytest.raises(TypeError, match="Symbol"):
        smart_integrate(E**2, np.linspace(0, 1, 3), (0.0, 1.0))
