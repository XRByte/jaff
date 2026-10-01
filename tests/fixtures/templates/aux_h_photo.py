# ABOUTME: jaffgen test template: every auxiliary function of the h_photo network
# ABOUTME: Rendered via GET aux_func; compared numerically against tests/golden
import math


def aux_functions():
    out = {}
    # $JAFF GET aux_func FOR deltae0
    out["deltae0"] = $aux_func$
    # $JAFF END
    # $JAFF GET aux_func FOR deltae1
    out["deltae1"] = $aux_func$
    # $JAFF END
    return out
