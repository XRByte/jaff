from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr, Float

if TYPE_CHECKING:
    from ...core import Network


class Thermodynamics:
    def __init__(self, net: Network):
        self.net: Network = net
        self.dEdt_chemical: Expr = self._get_dEdt_chemical()
        self.dEdt_extra: Expr

    def _get_dEdt_chemical(self) -> Expr:
        dEdt = Float(0.0)
        for r in self.net.reactions:
            _dEdt = r.dE * r.rate
            for s in r.reactants.core:
                _dEdt *= self.net.ndens[self.net.species[s.name].index]

            dEdt += _dEdt

        return self.net._standardize_symbols(dEdt, self.net.spec.expand_nuclei)

    def _get_dEdt_extra(self) -> Expr:
        dEdt = Float(0.0)
        if "heatingcoolingrate" in self.net.spec.aux_funcs:
            dEdt = self.net._standardize_symbols(
                self.net.spec.aux_funcs["heatingcoolingrate"]["def"],
                self.net.spec.expand_nuclei,
            )

        return dEdt
