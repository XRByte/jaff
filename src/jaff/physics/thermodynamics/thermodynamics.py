from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from sympy import Expr, Float

from .eos import EosFactory
from .internal_energy import DEDt, InternalEnergy

if TYPE_CHECKING:
    from ...core import Network


class Thermodynamics:
    def __init__(self, net: Network, dEdt_extra: Expr | None = None):
        self.net: Network = net
        self.dEdt_chemical: DEDt = self._get_dEdt_chemical()
        self.dEdt_extra: DEDt = (
            DEDt(dEdt_extra, self.net)
            if dEdt_extra is not None
            else self._get_dEdt_extra()
        )
        self.dEdt_tot: DEDt = self.dEdt_chemical + self.dEdt_extra

    @cached_property
    def eos(self) -> InternalEnergy:
        """Internal energy of the network for its configured EOS (built once).

        Returns
        -------
        InternalEnergy
            Built by :class:`EosFactory` from ``net.eos_props``.
        """
        return EosFactory(self.net, self.net.eos_props).generate()

    def _get_dEdt_chemical(self) -> DEDt:
        dEdt = Float(0.0)
        for r in self.net.reactions:
            _dEdt = r.dE * r.rate
            for s in r.reactants.core:
                _dEdt *= self.net.symbols.ndens[self.net.species[s.name].index]

            dEdt += _dEdt

        return DEDt(self.net.symbols.standardize(dEdt), self.net)

    def _get_dEdt_extra(self) -> DEDt:
        dEdt = Float(0.0)
        if "heatingcoolingrate" in self.net.spec.aux_funcs:
            dEdt = self.net.symbols.standardize(
                self.net.spec.aux_funcs["heatingcoolingrate"]["def"]
            )

        return DEDt(dEdt, self.net)
