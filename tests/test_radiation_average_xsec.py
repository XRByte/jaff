# ABOUTME: Radiation.get_group/total_average_cross_section: photon-number-weighted
# ABOUTME: band and full-spectrum mean cross sections, for tabulated and symbolic σ(E)

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

from jaff.core.reaction import PhotoReactionProps
from jaff.physics import RadiationProps
from jaff.physics.photo_reactions._radiation import Radiation
from tests.xsec_props import make_xsec_props

BANDS = [1.0, 2.0, 3.0]


def _radiation(profile_index=2):
    """Radiation over BANDS with the background field mocked out."""
    with patch("jaff.physics.photo_reactions._radiation.BackgroundField"):
        props = RadiationProps(bands=BANDS, profile_index=profile_index, c=1.0)
        return Radiation(None, props)


def _tabulated(fn):
    """σ(E) sampled densely over the band range, as (σ, E) arrays."""
    energy = np.linspace(BANDS[0], BANDS[-1], 20001)
    return fn(energy), energy


def test_flat_tabulated_xsec_averages_to_itself():
    rad = _radiation()
    sigma, energy = _tabulated(lambda e: np.full_like(e, 1e-18))

    groups = [rad.get_group_average_cross_section(sigma, energy, g) for g in rad.groups]
    assert [float(x) for x in groups] == pytest.approx([1e-18, 1e-18])
    assert float(rad.get_total_average_cross_section(sigma, energy)) == pytest.approx(
        1e-18
    )


def test_flat_symbolic_xsec_averages_to_itself():
    rad = _radiation()
    sigma = sp.Float(1e-18) + 0 * rad.ph_E

    groups = [rad.get_group_average_cross_section(sigma, rad.ph_E, g) for g in rad.groups]
    assert [float(x) for x in groups] == pytest.approx([1e-18, 1e-18])
    total = rad.get_total_average_cross_section(sigma, rad.ph_E)
    assert float(total) == pytest.approx(1e-18)


def test_linear_xsec_band_average_is_band_midpoint():
    """Flat photon spectrum (α=2) and σ ∝ E: <σ>_i is σ at the band midpoint."""
    rad = _radiation(profile_index=2)
    sigma, energy = _tabulated(lambda e: 1e-18 * e)

    tab = [rad.get_group_average_cross_section(sigma, energy, g) for g in rad.groups]
    sym = [
        rad.get_group_average_cross_section(1e-18 * rad.ph_E, rad.ph_E, g)
        for g in rad.groups
    ]
    assert [float(x) for x in tab] == pytest.approx([1.5e-18, 2.5e-18], rel=1e-6)
    assert [float(x) for x in sym] == pytest.approx([1.5e-18, 2.5e-18], rel=1e-9)


def test_tabulated_profile_built_once_per_reaction():
    """The decay total, band and absorption averages share one profile array."""
    rad = _radiation()
    sigma, energy = _tabulated(lambda e: np.full_like(e, 1e-18))
    xsec = make_xsec_props(
        absorption=True, photon_energy=energy, photo_absorption=sigma, photodecay=sigma
    )
    rxn = SimpleNamespace(photo=PhotoReactionProps(xsec=xsec))

    with patch.object(
        rad, "get_photden_profile", wraps=rad.get_photden_profile
    ) as profile:
        rad.get_reaction_rate_props(rxn)

    assert profile.call_count == 1
