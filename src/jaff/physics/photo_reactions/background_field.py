from typing import TYPE_CHECKING

from ...config import DATA_DIR
from ...drivers import HDF5

if TYPE_CHECKING:
    import numpy as np


class BackgroundField:
    def __init__(self, type: str = "draine"):
        self.type: str = type

        bgrad = HDF5().to_dict(
            f"{DATA_DIR / 'background_radiation' / 'radiation.hdf5'}::{type}"
        )
        self.wavelength: np.ndarray = bgrad["wavelength"]["_data"]
        self.intensity: np.ndarray = bgrad["intensity"]["_data"]
