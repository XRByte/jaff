# ABOUTME: Legacy-syntax template-engine regression fixture: THERMAL and DEDT_TYPE modifiers and the dedt/dtdt SUB tokens (energy network)
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# $JAFF SUB dedt, dtdt
dedt = $dedt$
dtdt = $dtdt$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[THERMAL none]$
f_none[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[THERMAL dedt]$
f_dedt[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[THERMAL dtdt]$
f_dtdt[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[DEDT_TYPE specific]$
f_specific[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[DEDT_TYPE per_particle]$
f_per_particle[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, expr, cse IN jacobian $[THERMAL dtdt DEDT_TYPE specific REPLACE nden\[(\d+)\] y[\1]]$
c$idx$ = $cse$
J[$idx$, $idx$] = $expr$
# $JAFF END

# $JAFF REPEAT idx, expr IN jacobian $[THERMAL none]$
J_none[$idx$, $idx$] = $expr$
# $JAFF END
