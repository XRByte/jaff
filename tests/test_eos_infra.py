# ABOUTME: Robustness tests for the EOS infrastructure (EosProps, EosFactory, Eos)
# ABOUTME: Validation, per-network Eos identity, no cache leaks, builder registry

import gc
import weakref
from types import SimpleNamespace

import pytest
import sympy as sp

from jaff.physics.eos import Eos, EosFactory, EosProps


def _stub_net(tag: str) -> SimpleNamespace:
    return SimpleNamespace(rho=sp.Symbol(f"rho_{tag}"), ntot=sp.Symbol(f"ntot_{tag}"))


class TestEosIdentity:
    def test_same_expr_different_networks_bind_own_network(self) -> None:
        net_a, net_b = _stub_net("a"), _stub_net("b")
        expr = sp.Symbol("e")
        eos_a = Eos(expr, net_a)
        eos_b = Eos(expr, net_b)
        assert eos_b.specific == expr / net_b.rho
        assert eos_a.specific == expr / net_a.rho


class TestEosFactoryLifetime:
    def test_factory_not_kept_alive_after_generation(self) -> None:
        factory = EosFactory(_stub_net("a"), EosProps("ideal", gamma=5.0 / 3.0))
        factory.ideal()
        ref = weakref.ref(factory)
        del factory
        gc.collect()
        assert ref() is None


class TestEosPropsValidation:
    def test_missing_required_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            EosProps("ideal")

    def test_unknown_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="gama"):
            EosProps("ideal", gamma=5.0 / 3.0, gama=2.0)

    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="bad"):
            EosProps("bad")

    def test_empty_gamma_map_accepted(self) -> None:
        props = EosProps("multi_gamma", default_gamma=5.0 / 3.0, gamma_map={})
        assert props.gamma_map == {}

    def test_non_numeric_gamma_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="gamma"):
            EosProps("ideal", gamma="5/3")

    def test_non_dict_gamma_map_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="gamma_map"):
            EosProps("multi_gamma", default_gamma=5.0 / 3.0, gamma_map=[1.4])

    @pytest.mark.parametrize("gamma", [1.0, 0.0, 0.5])
    def test_gamma_not_above_one_raises(self, gamma: float) -> None:
        with pytest.raises(ValueError, match="gamma"):
            EosProps("ideal", gamma=gamma)

    def test_gamma_map_value_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match="H2"):
            EosProps("multi_gamma", default_gamma=5.0 / 3.0, gamma_map={"H2": 1.0})

    def test_default_gamma_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match="default_gamma"):
            EosProps("multi_gamma", default_gamma=1.0, gamma_map={})


class TestEosFactoryDispatch:
    def test_relativistic_fermi_degenerate_not_implemented(self) -> None:
        factory = EosFactory(_stub_net("a"), EosProps("relativistic_fermi_degenerate"))
        with pytest.raises(NotImplementedError):
            factory.relativistic_fermi_degenerate()

    def test_invalid_type_message_separates_sentences(self) -> None:
        props = EosProps("ideal", gamma=5.0 / 3.0)
        props.type = "bad"
        with pytest.raises(ValueError, match=r"Invalid eos: 'bad'\. Valid eos types"):
            EosFactory(_stub_net("a"), props).generate()

    def test_builder_registry_matches_props_types(self) -> None:
        assert set(EosFactory._BUILDERS) == set(EosProps._REQUIRED)
