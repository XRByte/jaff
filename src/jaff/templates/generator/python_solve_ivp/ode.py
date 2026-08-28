import math

import numpy as np
from commons import *


def get_ode(y, tgas, crate, av, dedt=False):
    dy = np.zeros_like(y)

    # species number densities as a column vector (rhs expressions use nden[i, 0])
    nden = y[:nspecs].reshape(-1, 1)

    # full right-hand side: species dn_i/dt (0..nspecs-1) plus energy dE/dt (idx_tgas)
    rhs = np.zeros(nvars)

    # $JAFF REPEAT idx, rhs, cse IN rhses
    x$idx$ = $cse$
    rhs[$idx$] = $rhs$
    # $JAFF END

    dy[:nspecs] = rhs[:nspecs]

    # thermal coupling: evolve gas temperature only when requested at runtime.
    # rhs[idx_tgas] is the net volumetric energy rate dE/dt (erg / cm^3 / s);
    # convert to dT/dt with a simple ideal-gas EOS, e = n_tot * kB * T / (gamma - 1).
    if dedt:
        kB = 1.380649e-16
        gamma = 5.0 / 3.0
        n_tot = y[:nspecs].sum()
        dy[idx_tgas] = rhs[idx_tgas] / (n_tot * kB / (gamma - 1.0))

    return dy
