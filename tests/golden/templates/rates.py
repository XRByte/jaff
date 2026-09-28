# ABOUTME: Reaction rate templates - individual chemrate, deltae, deltarad evaluations
# ABOUTME: Evaluate each reaction's rate and auxiliary quantities

from typing import Dict, List, Any
import numpy as np


def evaluate_rates(network, test_values: Dict[str, Any]) -> Dict[str, List[float]]:
    """Evaluate all reaction rates for given test values.

    Parameters
    ----------
    network : Network
        Network object with reactions.
    test_values : dict
        Test input values (nden, tgas, rho, c_hat).

    Returns
    -------
    dict
        "rates": list of rate coefficients for each reaction
        "deltae": list of energy changes for each reaction (erg/reaction)
        "deltarad": list of radiation changes for each reaction
    """
    nden = test_values["nden"]
    tgas = test_values["tgas"]
    rho = test_values["rho"]
    c_hat = test_values["c_hat"]

    rates = []
    deltae = []
    deltarad = []

    for reaction in network.reactions:
        # Evaluate rate coefficient symbolically, substitute test values
        subs_dict = {
            network.ndens[i, 0]: float(nden[i]) for i in range(len(nden))
        }
        from sympy import symbols
        subs_dict[symbols("tgas")] = float(tgas)

        rate_val = float(reaction.rate.subs(subs_dict))
        rates.append(rate_val)

        # Evaluate deltae (energy change per reaction)
        if reaction.dE != 0:
            de_val = float(reaction.dE.subs(subs_dict))
        else:
            de_val = 0.0
        deltae.append(de_val)

        # Evaluate deltarad (radiation change per reaction)
        if reaction.dRad != 0:
            drad_val = float(reaction.dRad.subs(subs_dict))
        else:
            drad_val = 0.0
        deltarad.append(drad_val)

    return {
        "rates": rates,
        "deltae": deltae,
        "deltarad": deltarad,
    }
