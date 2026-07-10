module ode
    use commons
    use fluxes
    contains

    ! Right-hand side in the signature ODEPACK/DLSODES expects:
    !   subroutine F(neq, t, y, ydot)
    ! y holds the species number densities (1:nspecs) plus the gas temperature
    ! in slot idx_tgas. FLASH's DLSODES (opkd*) integrates this over the step.
    subroutine fex(neq, t, n, dn)
        implicit none
        integer::neq
        real*8::t
        real*8::n(nspecs+1), dn(nspecs+1)
        real*8::y(nspecs), tgas, flux(nreactions)

        y = n(1:nspecs)
        tgas = n(idx_tgas)

        flux = get_fluxes(y, tgas, common_crate, common_av)

        ! $JAFF REPEAT idx, ode_expression IN ode_expressions
        dn($idx+1$) = $ode_expression$
        ! $JAFF END

        ! Temperature is held fixed across the chemistry substep; FLASH updates
        ! it through the EOS after the network is applied.
        dn(idx_tgas) = 0d0

    end subroutine fex

    ! Dummy Jacobian. With MF=222 DLSODES builds the sparse Jacobian internally
    ! by finite differences, so this is never called - it only satisfies the
    ! argument list of the DLSODES call.
    subroutine jes(neq, t, n, j, ian, jan, pdj)
        implicit none
        integer::neq, j, ian(*), jan(*)
        real*8::t, n(*), pdj(*)
        return
    end subroutine jes

end module ode
