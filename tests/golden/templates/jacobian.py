# ABOUTME: Jacobian templates - full Jacobian matrix evaluation
# ABOUTME: Validates ∂(dx_i/dt)/∂x_j linearization matrix

from typing import Dict, List, Any
import numpy as np
from sympy import symbols


def evaluate_jacobian(
    network,
    test_values: Dict[str, Any],
    specific_eint: bool = True,
) -> Dict[str, Any]:
    """Evaluate the Jacobian matrix (∂ẋ/∂x) for the ODE system.

    Parameters
    ----------
    network : Network
        Network object with species and reactions.
    test_values : dict
        Test input values (nden, tgas, rho, c_hat).
    specific_eint : bool, optional
        If True, energy column uses specific energy (erg/g).
        If False, energy column uses volumetric energy (erg/cm³).
        Default: True.

    Returns
    -------
    dict
        "jac": (n_species + 1) × (n_species + 1) Jacobian matrix
               rows = d/dt equations, cols = state variables
               Last row is d(dE/dt)/d(state)
        "jac_no_energy": n_species × n_species species-only Jacobian
    """
    nden = test_values["nden"]
    tgas = test_values["tgas"]
    rho = test_values["rho"]
    c_hat = test_values["c_hat"]

    n_species = network.species.count
    nden_sym = network.ndens
    tgas_sym = symbols("tgas")

    # Build substitution dictionary
    subs_dict = {nden_sym[i]: float(nden[i]) for i in range(n_species)}
    subs_dict[tgas_sym] = float(tgas)

    # Get symbolic ODE right-hand sides
    sodes = network.sodes()

    # Compute species × species Jacobian: ∂(dnden[i]/dt) / ∂nden[j]
    jac_no_energy = np.zeros((n_species, n_species), dtype=np.float64)
    for i in range(n_species):
        for j in range(n_species):
            djac = sodes[i].diff(nden_sym[j]).subs(subs_dict)
            try:
                jac_no_energy[i, j] = float(djac)
            except (TypeError, AttributeError):
                jac_no_energy[i, j] = 0.0

    # Compute temperature column: ∂(dnden[i]/dt) / ∂T_gas
    temp_col = np.zeros(n_species, dtype=np.float64)
    for i in range(n_species):
        dtemp = sodes[i].diff(tgas_sym).subs(subs_dict)
        try:
            temp_col[i] = float(dtemp)
        except (TypeError, AttributeError):
            temp_col[i] = 0.0

    # For temperature coupling via energy equation:
    # dẋ_i/dT = (dẋ_i/dT) / (dE/dT) where E is the energy normalization
    if specific_eint:
        eos = network.eos(specific=True)
    else:
        eos = network.eos(specific=False)

    dE_dT = eos.diff(tgas_sym).subs(subs_dict)
    try:
        dE_dT = float(dE_dT)
    except (TypeError, AttributeError):
        dE_dT = 1.0  # Avoid division by zero

    if abs(dE_dT) > 1e-30:
        temp_col_normalized = temp_col / dE_dT
    else:
        temp_col_normalized = np.zeros_like(temp_col)

    # Assemble full Jacobian with temperature column
    jac_full = np.zeros((n_species + 1, n_species + 1), dtype=np.float64)
    jac_full[:n_species, :n_species] = jac_no_energy
    jac_full[:n_species, n_species] = temp_col_normalized

    # Energy equation row: ∂(dE/dt) / ∂nden[j] and ∂(dE/dt) / ∂T
    dEdt = network.dEdt_chem
    if network.dEdt_other != 0:
        dEdt = dEdt + network.dEdt_other

    for j in range(n_species):
        djac = dEdt.diff(nden_sym[j]).subs(subs_dict)
        try:
            jac_full[n_species, j] = float(djac)
        except (TypeError, AttributeError):
            jac_full[n_species, j] = 0.0

    dEdt_dT = dEdt.diff(tgas_sym).subs(subs_dict)
    try:
        jac_full[n_species, n_species] = float(dEdt_dT)
    except (TypeError, AttributeError):
        jac_full[n_species, n_species] = 0.0

    return {
        "jac": jac_full.tolist(),
        "jac_no_energy": jac_no_energy.tolist(),
        "temp_col": temp_col.tolist(),
        "temp_col_normalized": temp_col_normalized.tolist(),
    }
