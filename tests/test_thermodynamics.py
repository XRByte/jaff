# ABOUTME: Tests for Thermodynamics: cached EOS, stoichiometric rate helpers,
# ABOUTME: quotient-rule dE/dt forms and the dT/dt expression

import pytest
import sympy as sp

from jaff.physics import EosProps
from jaff.physics.constants import k_B
from jaff.physics.thermodynamics import InternalEnergy

GAMMA = 5.0 / 3.0
TWO_SPECIES = "@format:idx,R,R,P,rate\n1,H,H,H2,1\n2,H2,H,H,1\n"


@pytest.fixture
def net(make_network):
    return make_network(
        TWO_SPECIES, funcfile=False, eos_props=EosProps("ideal", gamma=GAMMA)
    )


def test_eos_is_cached_internal_energy(net):
    eos = net.thermodynamics.eos
    assert isinstance(eos, InternalEnergy)
    assert net.thermodynamics.eos is eos


def test_eos_uses_network_eos_props(net):
    sym = net.symbols
    expected = sym.ntot * k_B.cgs.value * sym.tgas / (GAMMA - 1.0)
    assert sp.simplify(net.thermodynamics.eos.volumetric - expected) == 0


def test_network_has_no_eos_method(net):
    assert not hasattr(net, "eos")
