# ABOUTME: Validation tests for RadiationProps, focused on per-band profile indices
# ABOUTME: (list profile_index): type, length, divergence checks and warnings.

import logging
from unittest.mock import patch

import pytest

from jaff.errors import ParserError
from jaff.physics import RadiationProps
from jaff.physics.photo_reactions._radiation import Radiation


class TestRadiationPropsProfileIndex:
    def test_scalar_profile_index_accepted(self):
        props = RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=2)
        assert props.profile_index == 2

    def test_list_profile_index_accepted(self):
        props = RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=[2.0, 1.0])
        assert props.profile_index == [2.0, 1.0]

    def test_list_profile_index_accepted_in_energy_mode(self):
        props = RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=[2, 1], mode="u")
        assert props.profile_index == [2, 1]

    @pytest.mark.parametrize("index", [[2.0], [2.0, 1.0, 0.5]])
    def test_list_length_must_match_band_count(self, index):
        with pytest.raises(ParserError, match="profile_index"):
            RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=index)

    def test_empty_list_rejected(self):
        with pytest.raises(ParserError):
            RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=[])

    @pytest.mark.parametrize("index", [True, [True, 1.0], "2", [2.0, "1"]])
    def test_invalid_types_rejected(self, index):
        with pytest.raises(ParserError):
            RadiationProps(bands=[1.0, 2.0, 3.0], profile_index=index)

    def test_list_builds_radiation_with_per_band_index(self):
        props = RadiationProps(bands=[1.0, 2.0, 4.0], profile_index=[2.0, 1.0])
        with patch("jaff.physics.photo_reactions._radiation.BackgroundField"):
            rad = Radiation(None, props)
        assert [grp.profile_idx for grp in rad.groups] == [2.0, 1.0]

    @pytest.mark.parametrize("bands", [[], [1.0]])
    def test_fewer_than_two_band_edges_rejected(self, bands):
        with pytest.raises(ParserError, match="at least two edges"):
            RadiationProps(bands=bands, profile_index=2)


class TestRadiationPropsDivergence:
    def test_zero_lower_edge_checks_first_band_index(self):
        # α0 < 0 → <E> diverges at E→0; last band's index is irrelevant.
        with pytest.raises(RuntimeError, match="bands\\[0\\]"):
            RadiationProps(bands=[0.0, 1.0, 2.0], profile_index=[-1.0, 2.0], mode="u")

    def test_zero_lower_edge_ok_when_first_band_converges(self):
        RadiationProps(bands=[0.0, 1.0, 2.0], profile_index=[2.0, -1.0], mode="u")

    def test_inf_upper_edge_checks_last_band_index(self):
        # α_last > 0 → <E> diverges at E→∞; first band's index is irrelevant.
        with pytest.raises(RuntimeError, match="inf"):
            RadiationProps(bands=[1.0, 2.0, "inf"], profile_index=[-1.0, 2.0], mode="u")

    def test_inf_upper_edge_ok_when_last_band_converges(self):
        RadiationProps(bands=[1.0, 2.0, "inf"], profile_index=[2.0, -1.0], mode="u")

    def test_low_edge_warning_uses_first_band_index(self):
        logger = logging.getLogger()
        with (
            patch.object(logger, "warning") as warn,
            patch("jaff.physics.photo_reactions._radiation_props.JaffLogger") as jl,
        ):
            jl.return_value.get_logger.return_value = logger
            RadiationProps(bands=[0.5, 1.0, 2.0], profile_index=[2.0, 0.0])
            warn.assert_not_called()
            RadiationProps(bands=[0.5, 1.0, 2.0], profile_index=[0.0, 2.0])
            warn.assert_called_once()
