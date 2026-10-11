# ABOUTME: Legacy-syntax template-engine regression fixture: radodes, RADIATION and RAD_ORDER modifiers (radiation network)
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# $JAFF REPEAT idx, radode IN radodes
drad[$idx$] = $radode$
# $JAFF END

# $JAFF REPEAT idx, radode, cse IN radodes
r$idx$ = $cse$
drad[$idx$] = $radode$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[RADIATION True]$
f_rad[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[RADIATION FALSE]$
f_norad[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, rhs IN rhses $[RADIATION true RAD_ORDER 1]$
f_rad1[$idx$] = $rhs$
# $JAFF END

# $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE photden\[\s*(\d+)\s*\] rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] xn[\1] REPLACE tgas T REPLACE c_hat chat RADIATION True]$
c$idx$ = $cse$
J[$idx$, $idx$] = $expr$
# $JAFF END
