# ABOUTME: ODE templates - full right-hand side evaluation for all species
# ABOUTME: Validates complete species evolution equations

from typing import Dict, List, Any
from sympy import symbols


def evaluate_ode(network, test_values: Dict[str, Any]) -> Dict[str, List[float]]:
    """Evaluate ODE right-hand sides for all species.

    Parameters
    ----------
    network : Network
        Network object with species and reactions.
    test_values : dict
        Test input values (nden, tgas, rho, c_hat).

    Returns
    -------
    dict
        "ode": list of dnden[i]/dt for each species (cm⁻³/s)
        "ode_with_energy": includes internal energy as last element (erg/cm³/s)
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

    # Get symbolic ODE right-hand sides
    sodes = network.sodes()

    # Evaluate each species ODE
    ode_values = []
    for sode in sodes:
        try:
            val = float(sode.subs(subs_dict))
        except (TypeError, AttributeError):
            val = 0.0
        ode_values.append(val)

    # Also compute energy equation (dEdt_chem for volumetric energy)
    try:
        dEdt_chem = float(network.dEdt_chem.subs(subs_dict))
    except (TypeError, AttributeError):
        dEdt_chem = 0.0

    try:
        dEdt_other = (
            float(network.dEdt_other.subs(subs_dict))
            if network.dEdt_other != 0
            else 0.0
        )
    except (TypeError, AttributeError):
        dEdt_other = 0.0

    dEdt_total = dEdt_chem + dEdt_other

    return {
        "ode": ode_values,
        "dEdt_total": dEdt_total,
        "ode_with_energy": ode_values + [dEdt_total],
    }
