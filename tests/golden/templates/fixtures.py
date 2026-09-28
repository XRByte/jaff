# ABOUTME: Test fixtures - RNG seeding and high-precision test value generation
# ABOUTME: Provides deterministic, reproducible test inputs for golden template evaluation

import numpy as np
from typing import Dict, Any


# Global RNG seed for reproducibility across all tests
GLOBAL_RNG_SEED = 42


def create_test_rng(seed: int = GLOBAL_RNG_SEED) -> np.random.Generator:
    """Create a seeded RNG for deterministic test value generation."""
    return np.random.default_rng(seed)


def generate_test_values(
    network,
    rng: np.random.Generator | None = None,
    seed: int = GLOBAL_RNG_SEED,
) -> Dict[str, Any]:
    """Generate high-precision test input values for a network.

    Parameters
    ----------
    network : Network
        Network object providing species count and structure.
    rng : np.random.Generator, optional
        RNG instance. If None, creates one with given seed.
    seed : int, optional
        Seed for RNG if not provided. Default: GLOBAL_RNG_SEED.

    Returns
    -------
    dict
        Test values:
        - nden: number densities array (float64, high precision)
        - tgas: gas temperature (K)
        - rho: mass density (g/cm³)
        - c_hat: radiation field strength (if applicable)
    """
    if rng is None:
        rng = create_test_rng(seed)

    n_species = network.species.count

    # Generate number densities: values in range [1e-12, 1e6] cm⁻³ with high precision
    # Using high-precision floats (float64) to catch numerical errors
    nden = rng.uniform(1e-12, 1e6, size=n_species).astype(np.float64)
    # Ensure no exact zeros or infinities
    nden = np.maximum(nden, 1e-15)
    nden = np.minimum(nden, 1e15)

    # Temperature: range [10 K, 1e4 K]
    tgas = float(rng.uniform(10.0, 1e4))

    # Compute mass density from number densities and species masses
    rho = 0.0
    for i, sp in enumerate(network.species):
        if sp.mass is not None:
            rho += sp.mass * nden[i]
    rho = float(np.maximum(rho, 1e-30))  # Avoid zero density

    # Radiation field strength (dimensionless, typically [0.1, 1e3])
    c_hat = float(rng.uniform(0.1, 1e3))

    return {
        "nden": nden,
        "tgas": tgas,
        "rho": rho,
        "c_hat": c_hat,
    }


def get_species_index_map(network) -> Dict[str, int]:
    """Return mapping from species name to nden array index."""
    return {sp.name: sp.index for sp in network.species}
