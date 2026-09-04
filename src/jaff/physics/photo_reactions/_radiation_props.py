import logging
from typing import cast

from sympy import Basic, oo

from ...errors import ParserError
from ...io import JaffLogger
from .. import constants


class RadiationProps:
    # Vaiid modes
    # nph: Photon number density
    # u: Enregy desnity
    _valid_modes: tuple[str, ...] = ("nph", "u")

    _valid_fields: tuple[str, ...] = (
        "bb_4000",
        "bb_10000",
        "bb_20000",
        "draine",
        "habing",
        "mathis",
        "solar",
        "tw_hydra",
    )

    def __init__(
        self,
        bands: list[str | float | Basic] = [],
        profile_index: float = 0.0,
        mode: str = "nph",  # nph or u,
        c: float | str = constants.c.cgs.value,
        background_field: str = "draine",
    ):

        self.logger: logging.Logger = JaffLogger().get_logger()
        self.profile_index: float = self._validate_profile_index(profile_index)
        self.mode: str = self._validate_mode(mode)
        self.bands: list[float | Basic] = self._validate_bands(bands)
        self.c: float | str = self._validate_c(c)
        self.background_field: str = self._validate_field(background_field)

    def _validate_c(self, c) -> float | str:
        if isinstance(c, (float, int, str)):
            return c

        raise ParserError(
            f"Speed of light must be of <float> or <string>. Found {type(c)}"
        )

    def _validate_field(self, field: str) -> str:
        if not isinstance(field, str):
            raise ParserError(
                f"Background radiation field must be a string: Supported fields are: {', '.join(self._valid_fields)}"
            )

        if field.lower() not in self._valid_fields:
            raise ParserError(
                f"Invalid background field {field}. Supported fields are {', '.join(self._valid_fields)}"
            )

        return field.lower()

    def _validate_mode(self, mode: str):
        if not isinstance(mode, str):
            raise ParserError(
                f"Radiation mode must be of type <str>. Valid modes are: {', '.join(self._valid_modes)}"
            )

        if mode.lower() not in self._valid_modes:
            raise ParserError(
                f"Invalid radiation mode {mode}. Valid modes are: {', '.join(self._valid_modes)}"
            )

        return mode.lower()

    def _validate_profile_index(self, index: float) -> float:
        if isinstance(index, (float, int)):
            return index

        raise ParserError(
            f"Invalid type for radiation profile index: {type(index)}\n"
            "Radiation profile index must be an integer or a float"
        )

    def _validate_bands(self, bands: list[float | str | Basic]) -> list[float | Basic]:
        """
        Validate and store the band-edge list, replacing ``"inf"`` with ``sympy.oo``.

        Also checks that the power-law photon-number spectrum is integrable
        over the supplied band range when energy-density mode is active.
        The average energy ``<E>_i = ∫ E·n(E) dE / ∫ n(E) dE`` must
        converge; this requires:

        - The lower edge to be non-zero when the spectral index is steep
          enough to cause a divergence at ``E → 0``.
        - The upper edge to be finite when the spectral index is shallow
          enough to cause a divergence at ``E → ∞``.

        Parameters
        ----------
        bands : list of (float, int, str, or sympy.Basic)
            Mutable band-edge list; modified in-place to replace any
            ``"inf"`` string with ``sympy.oo``.

        Raises
        ------
        RuntimeError
            If the average-energy integral would diverge given the supplied
            band edges and power-law index.
        """
        # Replace the sentinel string "inf" with SymPy's infinity symbol.
        if "inf" in bands:
            inf_index = bands.index("inf")
            bands[inf_index] = oo

        if any(isinstance(v, str) for v in bands):
            raise ParserError(
                f"Only 'inf' is supported as a string entry for radiation bands in the last slot. Found: {bands}"
            )

        # self.bands = cast(list[int | float | Basic], bands)

        if self.mode == "u":
            # The average-energy integral uses the *energy-density* spectrum
            # u(E) ∝ E^(α-1), so the integral ∫ E · u(E) dE ∝ ∫ E^α dE.
            # The effective power-law index for the ∫ E·n(E) dE integral is
            # pl_index = (α-2) + 1 = α - 1.
            pl_index: float = float(self.profile_index) - 1.0

            if pl_index == -1.0:
                # Integrand ~ E^(-1): log-divergence at both E=0 and E=∞.
                if isinstance(bands[0], (float, int)) and float(bands[0]) == 0.0:
                    raise RuntimeError(
                        f"The integral for average energy will diverge since the radiation band starts from bands[0]: {bands[0]}\n"
                        "Please try a non-zero value"
                    )
                if bands[-1] == oo:
                    raise RuntimeError(
                        f'The integral for average energy will diverge since the radiation band ends at bands[{len(bands) - 1}]: "inf"\n'
                        "Please try a non-infinite value or change the profile_index"
                    )
            elif pl_index + 1.0 > 0.0:
                # Integrand ~ E^p with p > -1: diverges at E → ∞.
                if bands[-1] == oo:
                    raise RuntimeError(
                        f'The integral for average energy will diverge since the radiation band ends at bands[{len(bands) - 1}]: "inf"\n'
                        "Please try a non-infinite value or change the profile_index"
                    )
            elif pl_index + 1.0 < 0.0:
                # Integrand ~ E^p with p < -1: diverges at E → 0.
                if isinstance(bands[0], (float, int)) and float(bands[0]) == 0.0:
                    raise RuntimeError(
                        f"The integral for average energy will diverge since the radiation band starts from bands[0]: {bands[0]}\n"
                        "Please try a non-zero value"
                    )

        if (
            float(self.profile_index) <= 1.0
            and isinstance(bands[0], (float, int))
            and float(bands[0]) < 1.0
        ):
            self.logger.warning(
                f"Radiation band starts at bands[0]={bands[0]} eV with "
                f"profile_index={self.profile_index}: the photon-number "
                "normalisation integral ∫E^(α-2)dE is lower-edge divergent "
                "(exponent ≤ -1) and near E→0 becomes ill-conditioned, which "
                "can yield a negative/garbage photon density. Use a non-zero "
                "bands[0] ≳ 1 eV or a larger profile_index."
            )

        return cast(list[float | Basic], bands)
