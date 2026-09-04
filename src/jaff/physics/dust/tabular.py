from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from sympy import Basic, Expr

from ...common import arr_integrate
from ...config import DATA_DIR
from ...drivers import HDF5
from ...errors import ParserError
from ...types import HDF5Dict

if TYPE_CHECKING:
    from .dust import Dust


class Tabular:
    def __init__(self, dust: Dust):
        self.dust: Dust = dust
        self._data: HDF5Dict = self._load_cross_sections()

    @cached_property
    def extinction_cross_section_per_hnuc(self) -> np.ndarray:
        # units in cm^2/H
        return self._data["C_ext_per_H"]["_data"]

    @cached_property
    def scattering_cross_section_per_hnuc(self) -> np.ndarray:
        # units in cm^2/H
        return self.albedo * self.extinction_cross_section_per_hnuc

    @cached_property
    def absorption_cross_section_per_hnuc(self) -> np.ndarray:
        # units in cm^2/H
        return (1 - self.albedo) * self.extinction_cross_section_per_hnuc

    @cached_property
    def transport_cross_section_per_hnuc(self) -> np.ndarray:
        # units in cm^2/H
        return (
            1 - self.albedo * self.scattering_cos
        ) * self.extinction_cross_section_per_hnuc

    @cached_property
    def wavelength(self) -> np.ndarray:
        # units in micrometer
        return self._data["wavelength"]["_data"]

    @cached_property
    def photon_energy(self) -> np.ndarray:
        # units in eV, sorted ascending (aligned with all cross-section arrays)
        return self._data["photon_energy"]["_data"]

    @cached_property
    def albedo(self) -> np.ndarray:
        return self._data["albedo"]["_data"]

    @cached_property
    def scattering_cos(self) -> np.ndarray:
        return self._data["cos"]["_data"]

    @cached_property
    def scattering_cos2(self) -> np.ndarray:
        return self._data["cos2"]["_data"]

    def avg_cross_section_per_hnuc(
        self, kind: str, bounds: tuple[float | Basic, float | Basic]
    ) -> Expr | float:
        # bounds must be in eV
        types = {
            "extinction": self.extinction_cross_section_per_hnuc,
            "absorption": self.absorption_cross_section_per_hnuc,
            "scattering": self.scattering_cross_section_per_hnuc,
            "transport": self.transport_cross_section_per_hnuc,
        }

        if kind not in types:
            raise ValueError(f"Invalid type of cross-section: {kind}")

        if self.dust.net.radiation is None:
            raise ParserError(
                "Radiation must be enabled to caculate average dust cross-section"
            )

        lower, upper = bounds
        ph_profile = self.dust.net.radiation.get_photden_profile(self.photon_energy)

        return arr_integrate(
            types[kind] * ph_profile,
            self.photon_energy,
            (lower, upper),
        ) / arr_integrate(
            ph_profile,
            self.photon_energy,
            (lower, upper),
        )

    def _load_cross_sections(self) -> HDF5Dict:
        # rv is validated in Dust.__init__; :.1f matches the dust.hdf5 group
        # names (mw_rv3.1 / mw_rv4.0 / mw_rv5.5).
        data: HDF5Dict = HDF5().to_dict(
            f"{DATA_DIR / 'dust' / 'dust.hdf5'}::mw_rv{self.dust.rv:.1f}"
        )
        return data
