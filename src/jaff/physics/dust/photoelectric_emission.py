from functools import cached_property
from typing import TYPE_CHECKING

from astropy import units as u
from sympy import Expr

from ...common import arr_integrate, smart_integrate
from ..constants import h

if TYPE_CHECKING:
    from ...core.network import Network


class PhotoelectricEmission:
    E_low: u.Quantity = 6.0 * u.eV  # photoelectric emission activation energy in eV
    E_high: u.Quantity = 13.6 * u.eV  # photoelectric emission cutoff energy in eV

    def __init__(self, network: Network):
        self.net: Network = network

    @cached_property
    def chi(self) -> None | Expr | float:
        rad = self.net.radiation
        if rad is None:
            return None

        num_tot = 0.0
        # Background radiation intensity integrated within the photoelectric band
        # and normalized to get the energy density
        den: Expr | float = arr_integrate(
            rad.background_field.intensity / rad.background_field.wavelength,
            rad.background_field.wavelength,  # wavelength is in nm
            (
                self.E_high.to(u.nm, equivalencies=u.spectral()).value,
                self.E_low.to(u.nm, equivalencies=u.spectral()).value,
            ),
        ) * (h.to(u.eV * u.s).value / u.nm.to(u.cm).value)

        for grp in rad.groups:
            lower = max(grp.lower, self.E_low.value)
            upper = (
                min(grp.upper, self.E_high.value)
                if isinstance(grp.upper, (int, float))
                else max(self.E_high.value, grp.lower)
            )
            if upper < lower:
                upper = lower

            energy_frac = smart_integrate(
                rad.energy_profile_sym,
                rad.E_sym,
                (lower, upper),
            ) / smart_integrate(
                rad.energy_profile_sym,
                rad.E_sym,
                (grp.lower, grp.upper),
            )

            num = grp.sym * energy_frac  # type: ignore

            # multiply with average energy in group if number densities are enabled
            if rad.energy_density is False:
                num *= grp.eavg or 0.0

            num_tot += num

        return num_tot / den  # chi
