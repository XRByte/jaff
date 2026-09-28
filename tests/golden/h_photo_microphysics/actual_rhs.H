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

  // $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
  const amrex::Real cse0 = state.c_hat*state.xn[0]*state.rn[2*0];
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

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True SPECIFIC_EINT True]$
  const amrex::Real cse0 = 1.6911086202976345e-18*state.c_hat;
  const amrex::Real cse1 = cse0*state.rn[2*0];
  const amrex::Real cse2 = std::pow(state.T, -1.7);
  const amrex::Real cse3 = 1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2];
  const amrex::Real cse4 = 1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2];
  const amrex::Real cse5 = 1.0/cse4;
  const amrex::Real cse6 = 2.0709734999998963e-16*state.T/cse3;
  const amrex::Real cse7 = cse4*state.T/std::pow(cse3, 2);
  const amrex::Real cse8 = cse6 - 2.5106594999998743e-24*cse7;
  const amrex::Real cse9 = cse1 - 7.7439498079206228e-11*cse2*cse3*cse5*cse8*state.xn[1]*state.xn[2];
  const amrex::Real cse10 = std::pow(state.T, -0.69999999999999996);
  const amrex::Real cse11 = 1.6594178159829075e-10*cse10;
  const amrex::Real cse12 = cse6 - 2.5092930924433742e-24*cse7;
  const amrex::Real cse13 = state.xn[1]*state.xn[2];
  const amrex::Real cse14 = cse3*cse5;
  const amrex::Real cse15 = 7.7439498079206228e-11*cse13*cse14*cse2;
  const amrex::Real cse16 = cse11*state.xn[2] + cse12*cse15;
  const amrex::Real cse17 = cse6 - 1.3664075564999315e-27*cse7;
  const amrex::Real cse18 = cse11*state.xn[1] + cse15*cse17;
  const amrex::Real cse19 = cse0*state.xn[0];
  const amrex::Real cse20 = -cse19;
  const amrex::Real cse21 = -cse16;
  const amrex::Real cse22 = -cse18;
  const amrex::Real cse23 = 1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2];
  const amrex::Real cse24 = 1.0/cse23;
  const amrex::Real cse25 = 0.68400000000000005 - 0.041599999999999998*std::log(0.0001*state.T);
  const amrex::Real cse26 = 2.2910735482189851e-26*cse25*std::pow(state.T, 0.30000000000000004);
  const amrex::Real cse27 = (1.0823095169904861e-29*state.c_hat*state.rn[2*0]*state.xn[0] - cse13*cse26)/std::pow(cse23, 2);
  const amrex::Real cse28 = 0.66666666666670005*cse14*cse24*(-6.8732206446569561e-27*cse10*cse13*cse25 + 9.5308659605909776e-28*cse10*state.xn[1]*state.xn[2]);
  const amrex::Real cse29 = cse24*cse26;

  jac(1, 1) = -cse9;
  jac(1, 2) = cse16;
  jac(1, 3) = cse18;
  jac(1, 4) = -cse15;
  jac(1, 5) = cse20;
  jac(2, 1) = cse9;
  jac(2, 2) = cse21;
  jac(2, 3) = cse22;
  jac(2, 4) = cse15;
  jac(2, 5) = cse19;
  jac(3, 1) = cse9;
  jac(3, 2) = cse21;
  jac(3, 3) = cse22;
  jac(3, 4) = cse15;
  jac(3, 5) = cse19;
  jac(4, 1) = 1.0823095169904861e-29*state.c_hat*cse24*state.rn[2*0] - 1.6737729999999998e-24*cse27 - cse28*cse8;
  jac(4, 2) = -cse12*cse28 - 1.6728620616289998e-24*cse27 - cse29*state.xn[2];
  jac(4, 3) = -cse17*cse28 - 9.1093837099999994e-28*cse27 - cse29*state.xn[1];
  jac(4, 4) = cse28;
  jac(4, 5) = 1.0823095169904861e-29*state.c_hat*cse24*state.xn[0];
  jac(5, 1) = -cse1;
  jac(5, 5) = cse20;
  jac(6, 1) = -cse0*state.rn[2*0+1];
  jac(6, 6) = cse20;
  // $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
