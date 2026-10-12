# ABOUTME: jaffgen test template: every auxiliary function of the GOW network
# ABOUTME: Rendered via GET aux_func; compared numerically against tests/golden
import math


def aux_functions():
    out = {}
    $$JAFF GET aux_func IN kcr_h_fac
    out["kcr_h_fac"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN qcr_h
    out["qcr_h"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN qcr_h2
    out["qcr_h2"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN qcr
    out["qcr"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN ncrh2
    out["ncrh2"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN kgr_gong
    out["kgr_gong"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN n2ncr
    out["n2ncr"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN fshield_h2
    out["fshield_h2"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN fshield_c
    out["fshield_c"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate0
    out["chemrate0"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate1
    out["chemrate1"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate4
    out["chemrate4"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate8
    out["chemrate8"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate10
    out["chemrate10"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate13
    out["chemrate13"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate14
    out["chemrate14"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate15
    out["chemrate15"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate16
    out["chemrate16"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate17
    out["chemrate17"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate18
    out["chemrate18"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate19
    out["chemrate19"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate20
    out["chemrate20"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate21
    out["chemrate21"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate22
    out["chemrate22"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate23
    out["chemrate23"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate27
    out["chemrate27"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate28
    out["chemrate28"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate32
    out["chemrate32"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate35
    out["chemrate35"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate37
    out["chemrate37"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate39
    out["chemrate39"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate40
    out["chemrate40"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate41
    out["chemrate41"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate42
    out["chemrate42"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate48
    out["chemrate48"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN chemrate49
    out["chemrate49"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae0
    out["deltae0"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae1
    out["deltae1"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae2
    out["deltae2"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae13
    out["deltae13"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae14
    out["deltae14"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae40
    out["deltae40"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae41
    out["deltae41"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN deltae42
    out["deltae42"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN heating_grainpe
    out["heating_grainpe"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_2level
    out["cooling_2level"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_3level
    out["cooling_3level"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_lya
    out["cooling_lya"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_h2
    out["cooling_h2"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_c0
    out["cooling_c0"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_cplus
    out["cooling_cplus"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_o0
    out["cooling_o0"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_co
    out["cooling_co"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_dust_coll
    out["cooling_dust_coll"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN cooling_dust_rec
    out["cooling_dust_rec"] = $aux_func$
    $$JAFF END
    $$JAFF GET aux_func IN heatingcoolingrate
    out["heatingcoolingrate"] = $aux_func$
    $$JAFF END
    return out
