# ABOUTME: Auxiliary function templates - individual chemrate, deltae, deltarad per reaction
# ABOUTME: Validates auxiliary quantity evaluations used in energy/radiation equations

from typing import Dict, List, Any
from sympy import symbols


def evaluate_auxiliary_functions(
    network, test_values: Dict[str, Any]
) -> Dict[str, List[float]]:
    """Evaluate auxiliary functions for each reaction.

    Auxiliary functions are:
    - chemrate{i}: custom rate coefficient (if defined)
    - deltae{i}: energy change per reaction (erg/reaction)
    - deltarad{i}: radiation change per reaction (photons or erg/reaction)

    Parameters
    ----------
    network : Network
        Network object with reactions and auxiliary functions.
    test_values : dict
        Test input values (nden, tgas, rho, c_hat).

    Returns
    -------
    dict
        "chemrate": [chemrate0, chemrate1, ...] for each reaction
        "deltae": [deltae0, deltae1, ...] for each reaction
        "deltarad": [deltarad0, deltarad1, ...] for each reaction
    """
    nden = test_values["nden"]
    tgas = test_values["tgas"]
    rho = test_values["rho"]
    c_hat = test_values["c_hat"]

    # Build substitution dictionary
    subs_dict = {
        network.ndens[i, 0]: float(nden[i]) for i in range(len(nden))
    }
    subs_dict[symbols("tgas")] = float(tgas)

    chemrate = []
    deltae = []
    deltarad = []

    from sympy.core.function import AppliedUndef

    for i, reaction in enumerate(network.reactions):
        # chemrate{i}: use reaction.rate (may be custom via auxiliary function)
        if reaction.rate.has(AppliedUndef):
            rate_val = 0.0  # Skip undefined function rates
        else:
            try:
                rate_val = float(reaction.rate.subs(subs_dict))
            except (TypeError, AttributeError):
                rate_val = 0.0
        chemrate.append(rate_val)

        # deltae{i}: energy change per reaction
        if reaction.dE != 0:
            try:
                de_val = float(reaction.dE.subs(subs_dict))
            except (TypeError, AttributeError):
                de_val = 0.0
        else:
            de_val = 0.0
        deltae.append(de_val)

        # deltarad{i}: radiation change per reaction
        if reaction.dRad != 0:
            try:
                drad_val = float(reaction.dRad.subs(subs_dict))
            except (TypeError, AttributeError):
                drad_val = 0.0
        else:
            drad_val = 0.0
        deltarad.append(drad_val)

    return {
        "chemrate": chemrate,
        "deltae": deltae,
        "deltarad": deltarad,
    }
