#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>

#include <actual_network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace ArrayUtil;
using namespace network_rp;

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
actual_rhs(const burn_t &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot) {

  using namespace Rates;

  // $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
  const amrex::Real cse0 = state.c_hat*state.xn[0]*state.rn[0];
  const amrex::Real cse1 = 1.6911086202976345e-18*cse0;
  const amrex::Real cse2 = cse1 - 1.6594178159829075e-10*std::pow(state.T, -0.69999999999999996)*state.xn[1]*state.xn[2];

  ydot(1) = -cse2;
  ydot(2) = cse2;
  ydot(3) = cse2;
  ydot(4) = 1.0823095169904861e-29*cse0 - 2.2910735482189851e-26*std::pow(state.T, 0.30000000000000004)*(0.68400000000000005 - 0.041599999999999998*std::log(0.0001*state.T))*state.xn[1]*state.xn[2];
  ydot(5) = -cse1;
  ydot(6) = -1.6911086202976345e-18*state.c_hat*state.xn[0]*state.rn[2*0+1];
  // $JAFF END
  // $JAFF SUB nspec
  ydot(4) /= state.rho;
  // $JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
  const amrex::Real cse0 = 1.6911086202976345e-18*state.c_hat;
  const amrex::Real cse1 = cse0*state.rn[0];
  const amrex::Real cse2 = -cse1;
  const amrex::Real cse3 = std::pow(state.T, -0.69999999999999996);
  const amrex::Real cse4 = 1.6594178159829075e-10*cse3;
  const amrex::Real cse5 = cse4*state.xn[2];
  const amrex::Real cse6 = cse4*state.xn[1];
  const amrex::Real cse7 = (1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2]);
  const amrex::Real cse8 = 7.7439498079206228e-11*cse7*std::pow(state.T, -1.7)*state.xn[1]*state.xn[2];
  const amrex::Real cse9 = cse0*state.xn[0];
  const amrex::Real cse10 = -cse9;
  const amrex::Real cse11 = -cse5;
  const amrex::Real cse12 = -cse6;
  const amrex::Real cse13 = 1.0823095169904861e-29*state.c_hat;
  const amrex::Real cse14 = 0.68400000000000005 - 0.041599999999999998*std::log(0.0001*state.T);
  const amrex::Real cse15 = cse14*state.xn[2];
  const amrex::Real cse16 = 2.2910735482189851e-26*std::pow(state.T, 0.30000000000000004);

  jac(1, 1) = cse2;
  jac(1, 2) = cse5;
  jac(1, 3) = cse6;
  jac(1, 4) = -cse8;
  jac(1, 5) = cse10;
  jac(2, 1) = cse1;
  jac(2, 2) = cse11;
  jac(2, 3) = cse12;
  jac(2, 4) = cse8;
  jac(2, 5) = cse9;
  jac(3, 1) = cse1;
  jac(3, 2) = cse11;
  jac(3, 3) = cse12;
  jac(3, 4) = cse8;
  jac(3, 5) = cse9;
  jac(4, 1) = cse13*state.rn[0];
  jac(4, 2) = -cse15*cse16;
  jac(4, 3) = -cse14*cse16*state.xn[1];
  jac(4, 4) = 0.66666666666670005*cse7*(-6.8732206446569561e-27*cse15*cse3*state.xn[1] + 9.5308659605909776e-28*cse3*state.xn[1]*state.xn[2]);
  jac(4, 5) = cse13*state.xn[0];
  jac(5, 1) = cse2;
  jac(5, 5) = cse10;
  jac(6, 1) = -cse0*state.rn[2*0+1];
  jac(6, 6) = cse10;
  // $JAFF END
  // $JAFF SUB nspec, nbands
  for (int i = 1; i <= INT_NEQS; i++) {
    jac(4, i) /= state.rho;
  }
  // $JAFF END

}

AMREX_INLINE
void actual_rhs_init() {}

#endif
