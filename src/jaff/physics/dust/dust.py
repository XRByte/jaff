from typing import TYPE_CHECKING

from .photoelectric_emission import PhotoelectricEmission

if TYPE_CHECKING:
    from ...core.network import Network


class Dust:
    def __init__(self, network: Network):
        self.network: Network = network
        self.pe: PhotoelectricEmission = PhotoelectricEmission(network)
