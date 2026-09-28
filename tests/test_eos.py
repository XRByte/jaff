# ABOUTME: Unit tests for the symbolic ideal-gas EOS used by the Jacobian energy column
# ABOUTME: Volumetric, per-mass (norm 0) and per-particle (norm 1) internal energies

from pathlib import Path

import pytest
import sympy as sp

from jaff import Network
from jaff.physics.constants import k_B

GAMMA = 5.0 / 3.0
TGAS = sp.symbols("tgas")


@pytest.fixture(scope="module")
def net() -> Network:
    return Network(str(Path(__file__).parent / "fixtures" / "react_cie_hepp.jet"))


def _volumetric(net: Network) -> sp.Expr:
    return net.ntot * k_B.cgs.value * TGAS / (GAMMA - 1.0)


def test_volumetric(net: Network) -> None:
    assert sp.simplify(net.eos(GAMMA, specific=False) - _volumetric(net)) == 0


def test_specific_per_mass_is_default_norm(net: Network) -> None:
    expected = _volumetric(net) / net.rho
    assert sp.simplify(net.eos(GAMMA, specific=True) - expected) == 0
    assert sp.simplify(net.eos(GAMMA, specific=True, norm=0) - expected) == 0


def test_specific_per_particle(net: Network) -> None:
    expected = k_B.cgs.value * TGAS / (GAMMA - 1.0)
    assert sp.simplify(net.eos(GAMMA, specific=True, norm=1) - expected) == 0


def test_invalid_norm_rejected(net: Network) -> None:
    with pytest.raises(ValueError, match="norm"):
        net.eos(GAMMA, specific=True, norm=2)
