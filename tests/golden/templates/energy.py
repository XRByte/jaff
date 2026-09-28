# ABOUTME: Energy rate templates - dEdt_chem, dEdt_other, dRad_dt_extra evaluations
# ABOUTME: Separate and combined energy source/sink terms

from typing import Dict, Any
from sympy import symbols


def evaluate_energy_rates(network, test_values: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate chemical heating/cooling rates.

    Parameters
    ----------
    network : Network
        Network object with dEdt_chem, dEdt_other, dRad_dt_extra expressions.
    test_values : dict
        Test input values (nden, tgas, rho, c_hat).

    Returns
    -------
    dict
        "dEdt_chem": chemical heating/cooling rate (erg/cm³/s)
        "dEdt_other": auxiliary heating/cooling rate (erg/cm³/s)
        "dRad_dt_extra": extra radiation source term (photons/cm³/s or erg/cm³/s)
        "dEdt_total": dEdt_chem + dEdt_other
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

    # Evaluate each energy rate, handling undefined functions
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

    try:
        dRad_dt_extra = (
            float(network.dRad_dt_extra.subs(subs_dict))
            if network.dRad_dt_extra != 0
            else 0.0
        )
    except (TypeError, AttributeError):
        dRad_dt_extra = 0.0

    return {
        "dEdt_chem": dEdt_chem,
        "dEdt_other": dEdt_other,
        "dRad_dt_extra": dRad_dt_extra,
        "dEdt_total": dEdt_chem + dEdt_other,
    }
