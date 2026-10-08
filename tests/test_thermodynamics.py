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


NUMBER_CONSERVING = "@format:idx,R,R,P,P,rate\n1,H2,O,OH,H,1\n"


def _ids(net):
    return net.species["H"].index, net.species["H2"].index


def test_dntot_dt_counts_particle_change(net):
    # TWO_SPECIES: H + H -> H2 (Δn = -1) ; H2 + H -> H (Δn = -1)
    flux = net.sfluxes()
    assert sp.simplify(net.symbols.dntot_dt - (-flux[0] - flux[1])) == 0


def test_dntot_dt_zero_when_number_conserving(make_network):
    # H2 + O -> OH + H : Δn = 0, so the reaction drops out exactly
    net = make_network(NUMBER_CONSERVING, funcfile=False)
    assert net.symbols.dntot_dt == 0


def test_drho_dt_is_mass_weighted_stoichiometric_sum(net):
    # Never forced to zero: Σ_r (Σ_i m_i ν_ri) F_r for every reaction.
    # TWO_SPECIES: r1 H + H -> H2 (Δm = m_H2 - 2 m_H), r2 H2 + H -> H (Δm = -m_H2)
    flux = net.sfluxes()
    m_h, m_h2 = net.species["H"].mass, net.species["H2"].mass
    expected = (m_h2 - 2 * m_h) * flux[0] + (-m_h2) * flux[1]
    assert sp.simplify(net.symbols.drho_dt - expected) == 0


def test_drho_dt_equals_weighted_rate_of_masses(net):
    masses = [s.mass for s in net.species]
    assert net.symbols.drho_dt == net.symbols.weighted_rate(masses)


def test_drho_dt_does_not_trust_check_mass(make_network, monkeypatch):
    # H -> H+ loses one electron mass; even if the reaction were deemed
    # "conserved", the per-cell mass rate must keep its -m_e term.
    from jaff.core.reaction import Reaction

    monkeypatch.setattr(Reaction, "check_mass", lambda self: True)
    net = make_network("@format:idx,R,P,rate\n1,H,H+,1\n", funcfile=False)
    m_e = 9.10938371e-28
    flux = net.sfluxes()
    expected = (net.species["H+"].mass - net.species["H"].mass) * flux[0]
    assert sp.simplify(net.symbols.drho_dt - expected) == 0
    assert abs(net.species["H+"].mass - net.species["H"].mass + m_e) < 1e-40


def test_weighted_rate_matches_hand_sum(net):
    w_h, w_h2 = sp.symbols("w_h w_h2")
    h, h2 = _ids(net)
    weights = [0, 0]
    weights[h], weights[h2] = w_h, w_h2
    flux = net.sfluxes()
    # r1: H+H -> H2 : ν_H=-2, ν_H2=+1 ; r2: H2+H -> H : ν_H=0, ν_H2=-1
    expected = (-2 * w_h + w_h2) * flux[0] + (-w_h2) * flux[1]
    assert sp.simplify(net.symbols.weighted_rate(weights) - expected) == 0


def test_weighted_rate_rejects_wrong_length(net):
    with pytest.raises(ValueError, match="one weight per species"):
        net.symbols.weighted_rate([1])


def test_rate_helpers_are_cached(net):
    assert net.symbols.dntot_dt is net.symbols.dntot_dt
    assert net.symbols.drho_dt is net.symbols.drho_dt
