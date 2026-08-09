# ABOUTME: Tests for reaction rate-expression parsing, temperature limits, stoichiometry
# ABOUTME: Operates on freshly built networks via the make_network factory

import pytest
from sympy import Piecewise, symbols


class TestRateExpressions:
    """Rate strings become usable SymPy expressions."""

    @pytest.mark.parametrize(
        "line",
        [
            "H + H -> H2 [10,1000] 1.0e-10",  # constant
            "H + e- -> H- [10,10000] 3e-16 * (tgas/300)**0.5",  # power law
            "C+ + H2 -> CH+ + H [10,41000] 1e-10 * exp(-4640/tgas)",  # exponential
            "O + H -> OH [10,41000] 9.9e-11 * sqrt(tgas) * exp(-100/tgas)",  # complex
        ],
    )
    def test_rate_parses_to_expression(self, make_network, line):
        rea = make_network([line]).reactions[0]
        assert rea.rate is not None
        assert hasattr(rea.rate, "free_symbols") or isinstance(rea.rate, (int, float))

    @pytest.mark.parametrize("rate", ["1e-50", "1e50", "0.0"])
    def test_extreme_rate_values_accepted(self, make_network, rate):
        rea = make_network([f"H -> H+ + e- [10,1000] {rate}"]).reactions[0]
        assert rea.rate is not None

    def test_malformed_rate_expr_is_tolerated(self, make_network):
        # Unknown functions/variables parse as free symbols rather than aborting;
        # the valid reaction still loads.
        net = make_network(
            [
                "H -> H+ + e- [10,1000] invalid_function_name()",
                "He -> He+ + e- [10,1000] 1e-10 * unknown_variable",
                "H2 -> H + H [10,1000] 1e-10",
            ]
        )
        assert net.reactions.count >= 1


class TestTemperatureLimits:
    """tmin/tmax storage and the default 'clip' Piecewise wrapping."""

    def test_limits_are_stored(self, make_network):
        rea = make_network(["H + H -> H2 [10,1000] 1e-10"]).reactions[0]
        assert (rea.tmin, rea.tmax) == (10, 1000)

    @pytest.mark.parametrize(
        "limits",
        ["[0.001,1e10]", "[1000,1000]", "[-1,5000]"],
    )
    def test_extreme_limits_do_not_crash(self, make_network, limits):
        # Open/degenerate bounds may resolve to None; loading must still work.
        rea = make_network([f"H -> H+ + e- {limits} 1e-10"]).reactions[0]
        assert hasattr(rea, "tmin") and hasattr(rea, "tmax")

    def test_clip_wraps_tgas_dependent_rate_in_piecewise(self, make_network):
        # Default t_cutoff="clip": a bounded tgas-dependent rate holds its
        # boundary value, i.e. becomes a Piecewise.
        tgas = symbols("tgas")
        rea = make_network(["H + e- -> H- [10,10000] 3e-16 * (tgas/300)**0.5"]).reactions[
            0
        ]
        assert tgas in rea.rate.free_symbols
        assert rea.t_cutoff == "clip"
        assert rea.rate.has(Piecewise)


class TestStoichiometry:
    """Stoichiometry matrices track reactant/product multiplicity."""

    def test_high_stoichiometry_builds(self, make_network):
        net = make_network(
            [
                "H + H + H + H + H -> H2 + H + H + H [10,1000] 1e-20",
                "H2 -> H + H + H + He + Ne [10,1000] 1e-10",
            ]
        )
        assert net.reactions.count == 2
        assert net.reactant_matrix.shape[0] == 2
        assert net.product_matrix.shape[0] == 2

    def test_matrix_rows_match_reaction_count(self, make_network):
        net = make_network(
            [
                "H + H -> H2 [10,1000] 1e-10",
                "H + O -> OH [10,1000] 1e-11",
                "H2 + O -> H2O [10,1000] 1e-12",
            ]
        )
        n = net.reactions.count
        assert net.reactant_matrix.shape == (n, net.species.count)
        assert net.product_matrix.shape == (n, net.species.count)
