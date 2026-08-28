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
  const amrex::Real cse0 = 6.3999999999999996e-10*state.xn[0]*state.xn[4];
  const amrex::Real cse1 = -cse0;
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = cse2/state.xn[2];
  const amrex::Real cse4 = 0.00011320754716981131*cse3*state.xn[3];
  const amrex::Real cse5 = std::min(10000000000.0, cse4);
  const amrex::Real cse6 = 1.0/(cse5 + 1);
  const amrex::Real cse7 = cse6*state.xn[3]*state.xn[17];
  const amrex::Real cse8 = 1.6000000000000001e-9*cse7;
  const amrex::Real cse9 = cse5*cse8;
  const amrex::Real cse10 = 1.99e-9*std::pow(state.T, -0.19)*state.xn[11]*state.xn[16];
  const amrex::Real cse11 = cse10*cse6;
  const amrex::Real cse12 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse13 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av)*state.xn[15];
  const amrex::Real cse14 = 1.0/state.T;
  const amrex::Real cse15 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse14)*state.xn[7]*state.xn[15];
  const amrex::Real cse16 = M_LN10;
  const amrex::Real cse17 = 1.0/cse16;
  const amrex::Real cse18 = 2*state.xn[3];
  const amrex::Real cse19 = cse18 + state.xn[0] + state.xn[1] + 2*state.xn[4] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16];
  const amrex::Real cse20 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse21 = std::pow(cse16, -2);
  const amrex::Real cse22 = std::pow(cse20, 2)*cse21;
  const amrex::Real cse23 = 1.0/cse19;
  const amrex::Real cse24 = cse23*state.xn[3];
  const amrex::Real cse25 = cse23*state.xn[0];
  const amrex::Real cse26 = std::pow(10, 0.41599999999999998*cse17*cse20 + 0.32700000000000001*cse22 - 3)*cse25 + std::pow(10, 1.3*cse17*cse20 - 1.6200000000000001*cse22 - 4.8449999999999998)*cse24;
  const amrex::Real cse27 = cse19*((cse26 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse25
));
  const amrex::Real cse28 = cse17/(cse27 + 1);
  const amrex::Real cse29 = cse27*cse28;
  const amrex::Real cse30 = state.T > 700.0;
  const amrex::Real cse31 = ((cse30) ? (
   std::pow(10, cse28*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse14)) + cse29*std::log(1.3000000000000001e-9*std::exp(-53300*cse14)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse32 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse33 = ((cse30) ? (
   std::pow(10, cse28*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse14 - 1)) + cse29*std::log(3.5199999999999998e-9*std::exp(-43900*cse14)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse34 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse35 = cse34*state.xn[2]*state.xn[16];
  const amrex::Real cse36 = cse19*d2g;
  const amrex::Real cse37 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[1]*state.xn[11];
  const amrex::Real cse38 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse39 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse38)/cse38;
  const amrex::Real cse40 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse41 = chi*cse40;
  const amrex::Real cse42 = std::pow(state.T, -1.3);
  const amrex::Real cse43 = std::exp(-23*cse14);
  const amrex::Real cse44 = cse42*cse43*state.xn[3]*state.xn[8];
  const amrex::Real cse45 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse46 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av)*state.xn[14];
  const amrex::Real cse47 = cse45 + cse46;
  const amrex::Real cse48 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse14) + 0.62)*state.xn[15];
  const amrex::Real cse49 = 1.3500000000000001e-9*cse48*state.xn[6];
  const amrex::Real cse50 = state.xn[3]*state.xn[6];
  const amrex::Real cse51 = 1.2599999999999999e-13*cse50*std::exp(-22.5*cse14);
  const amrex::Real cse52 = cse49 + cse51;
  const amrex::Real cse53 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse54 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse55 = -cse53 + cse54;
  const amrex::Real cse56 = crate*state.xn[9];
  const amrex::Real cse57 = 6.5199999999999996*cse56;
  const amrex::Real cse58 = 2.2999999999999998*cse24 + 1.5*cse25;
  const amrex::Real cse59 = crate*cse58;
  const amrex::Real cse60 = cse59*state.xn[0];
  const amrex::Real cse61 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse62 = state.xn[0]*state.xn[2];
  const amrex::Real cse63 = cse62*((cse30) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse61, 8) + 0.000111954395*std::pow(cse61, 7) - 0.00263197617*std::pow(cse61, 6) + 0.034825597700000002*std::pow(cse61, 5) - 0.28770560000000001*std::pow(cse61, 4) + 1.56315498*std::pow(cse61, 3) - 5.7393287500000003*std::pow(cse61, 2))
)
: (
   0.0
));
  const amrex::Real cse64 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[2]*state.xn[10];
  const amrex::Real cse65 = 4.8813573300130879e-6*std::pow(cse14, 1.5)*std::pow(114.81344434488582*std::pow(cse14, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse66 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse67 = chi*cse66;
  const amrex::Real cse68 = cse3*cse67;
  const amrex::Real cse69 = 1.7*cse68;
  const amrex::Real cse70 = std::log(state.T);
  const amrex::Real cse71 = 1.2250000000000001e-11*cse36*state.xn[1]/(1.6774414051711492e-5*std::pow(cse68, 1.3779999999999999)*(508.69999999999999*std::pow(cse69, -1.102e-5*cse70 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse72 = -cse57 - cse60 - cse63 + cse64 + cse65 + cse71;
  const amrex::Real cse73 = cse32 - cse37;
  const amrex::Real cse74 = cse18*cse59;
  const amrex::Real cse75 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[2]*state.xn[13];
  const amrex::Real cse76 = 2.0740886949963892e-18*state.c_hat*state.xn[12];
  const amrex::Real cse77 = cse76*state.rn[0];
  const amrex::Real cse78 = -cse77;
  const amrex::Real cse79 = 2.1659999999999999e-12*cse36*state.xn[13]/(1.534817233071468e-7*std::pow(cse68, 1.8740000000000001)*(43750.0*std::pow(cse69, -7.538e-5*cse70 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse80 = 16800.0*crate*cse24*state.xn[12];
  const amrex::Real cse81 = (1.0537934368240927e-9*std::pow(cse14, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse14) + 1.31e-7*std::exp(-26.600000000000001*cse14) + 3.4e-8*std::exp(-7.6200000000000001*cse14) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse14)))*state.xn[7]*state.xn[16];
  const amrex::Real cse82 = cse23*state.xn[2];
  const amrex::Real cse83 = ((cse82 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse84 = cse10*cse83/(cse83 + 1);
  const amrex::Real cse85 = cse11 + 1.3007021411393064e-6*cse35 + cse81 + cse84;
  const amrex::Real cse86 = cse17*cse70;
  const amrex::Real cse87 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse36*state.xn[6]/(7.104740267838494e-7*std::pow(cse68, 1.512)*(5115.0*std::pow(cse69, -5.4939999999999998e-7*cse70 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse86*(cse86*(0.044330000000000001*cse86 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse88 = crate*state.xn[7];
  const amrex::Real cse89 = 3.8500000000000001*cse88;
  const amrex::Real cse90 = 0.15970000000000001*std::exp(-49550.0*cse14);
  const amrex::Real cse91 = ((1.6339999999999999e-6*std::exp(-15230.0*cse14) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse14) + 6.3460000000000002e-9*std::exp(-12.17*cse14))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse90 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse90 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse92 = 4.5579999999999993e-11*cse36*state.xn[8]/(0.011078790689599145*std::pow(cse68, 1.1279999999999999)*(433.10000000000002*std::pow(cse69, -0.00013329999999999999*cse70 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse93 = 1040.0*cse24*cse88;
  const amrex::Real cse94 = 2.808e-22*ncol_H2;
  const amrex::Real cse95 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse94)*std::exp(-1.6000000000000001e-17*ncol_C)*state.xn[7]/(cse94 + 1);
  const amrex::Real cse96 = -cse89 + cse91 + cse92 - cse93 - cse95;
  const amrex::Real cse97 = 3.2999999999999996e-13*cse44;
  const amrex::Real cse98 = cse8 + cse9;
  const amrex::Real cse99 = cse97 + cse98;
  const amrex::Real cse100 = -cse81;
  const amrex::Real cse101 = -cse84;
  const amrex::Real cse102 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse103 = cse33*state.xn[3];
  const amrex::Real cse104 = 5.6999999999999997e-11*cse41;
  const amrex::Real cse105 = cse104*cse39;
  const amrex::Real cse106 = 7.2000000000000002e-15*cse50;
  const amrex::Real cse107 = cse1 + cse106 + cse74;
  const amrex::Real cse108 = -cse54;
  const amrex::Real cse109 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse110 = cse106 + cse109 + cse52 + cse87;
  const amrex::Real cse111 = cse100 + cse53;
  const amrex::Real cse112 = 184.0*cse24*cse56;
  const amrex::Real cse113 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av)*state.xn[9];
  const amrex::Real cse114 = cse112 + cse113 - cse15;
  const amrex::Real cse115 = 9.1500000000000005e-10*cse48*state.xn[8];
  const amrex::Real cse116 = cse102 + cse57 - cse64;
  const amrex::Real cse117 = cse109 - cse45;
  const amrex::Real cse118 = cse101 + cse12 + cse13;
  const amrex::Real cse119 = -cse75 + cse77 - cse79 + cse80;
  const amrex::Real cse120 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse121 = cse120*state.xn[2];
  const amrex::Real cse122 = std::exp(-118400.0*cse14)/(4.4460389837719569e-9*cse121 + 626500000.0);
  const amrex::Real cse123 = cse69 + 50;
  const amrex::Real cse124 = std::log(1.7*cse2*cse67/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse125 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse126 = (1.0/100.0)*state.T;
  const amrex::Real cse127 = std::log(cse126);
  const amrex::Real cse128 = state.T < 500;
  const amrex::Real cse129 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse130 = 4.5299999999999998e-6*cse125 + 7.5799999999999997e-10*std::pow(cse126, 0.0086999999999999994*cse127 + 0.12809999999999999)*state.xn[0] + (0.75*((cse128) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse129
)) + 0.25*((cse128) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse129
)))*state.xn[3];
  const amrex::Real cse131 = cse130*std::exp(-91.209999999999994*cse14)/(cse130 + 2.3e-6);
  const amrex::Real cse132 = 0.001*state.T;
  const amrex::Real cse133 = cse132 + 1;
  const amrex::Real cse134 = std::pow(cse133, -2);
  const amrex::Real cse135 = 1.0/cse133;
  const amrex::Real cse136 = cse23/(std::pow(10, -2.3900000000000001*cse134 + 0.055500000000000001*cse135 - 11.06)*cse25 + std::pow(10, -2.0230000000000001*cse134 - 3.6709999999999998*cse135 - 11.08)*cse24);
  const amrex::Real cse137 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse138 = 1.75*cse137*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse137;
  const amrex::Real cse139 = l0_co_interp1d(state.T);
  const amrex::Real cse140 = 1.0/cse139;
  const amrex::Real cse141 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse23*ncol_H))
));
  const amrex::Real cse142 = 1.0/llte_co_interp2d(state.T, cse141);
  const amrex::Real cse143 = nhalf_co_interp2d(state.T, cse141);
  const amrex::Real cse144 = cse17*std::log(cse19);
  const amrex::Real cse145 = 2.0*cse24*((cse144 < 2) ? (
   10.0
)
: ((cse144 < 4) ? (
   (3.0/2.0)*cse144 + 7
)
: ((cse144 < 7) ? (
   (4.0/3.0)*cse144 + 23.0/3.0
)
: ((cse144 < 10) ? (
   (1.0/3.0)*cse144 + 44.0/3.0
)
: (
   18.0
))))) + cse25*(26.399999999999999*std::sqrt(cse82/(cse82 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse146 = 0.042999999999999997*cse127;
  const amrex::Real cse147 = 4.34e-10*std::pow(cse126, 0.755 - 0.16*cse127)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse126, 0.52500000000000002*cse127 + 1.198) + 5.2499999999999998e-13*std::pow(cse126, cse146 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse148 = 3.5700000000000001e-10*std::pow(cse126, 0.41899999999999998 - 0.0030000000000000001*cse127)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse126, 0.025000000000000001*cse127 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse126, cse146 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse149 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse150 = 3.1899999999999998e-10*std::pow(cse126, 0.36899999999999999 - 0.0060000000000000001*cse127)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse126, 0.041000000000000002*cse127 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse126, 0.058000000000000003*cse127 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse151 = cse148 + 8.9099999999999997e-5;
  const amrex::Real cse152 = cse149*cse150;
  const amrex::Real cse153 = cse147*cse148*cse149 + (1.0/15.0)*cse147*cse150*std::exp(-425.40000000000003*cse14) + cse151*cse152;
  const amrex::Real cse154 = cse147 + 1.7499999999999998e-5;
  const amrex::Real cse155 = cse150 + 1.34e-10;
  const amrex::Real cse156 = (3.0/5.0)*cse148*std::exp(-227.80000000000001*cse14);
  const amrex::Real cse157 = cse152*cse154 + cse154*cse156 + cse155*cse156;
  const amrex::Real cse158 = 1.0/((1.0/3.0)*cse147*cse155*std::exp(-98.800000000000011*cse14) + cse151*cse154 + cse151*cse155 + cse153 + cse157);
  const amrex::Real cse159 = state.T < 100.0;
  const amrex::Real cse160 = state.T < 6000.0;
  const amrex::Real cse161 = state.T < 10;
  const amrex::Real cse162 = std::log(cse132);
  const amrex::Real cse163 = cse162*cse17;
  const amrex::Real cse164 = std::pow(cse16, -8);
  const amrex::Real cse165 = std::pow(cse162, 8)*cse164;
  const amrex::Real cse166 = std::pow(cse16, -7);
  const amrex::Real cse167 = std::pow(cse162, 7)*cse166;
  const amrex::Real cse168 = std::pow(cse16, -6);
  const amrex::Real cse169 = std::pow(cse162, 6)*cse168;
  const amrex::Real cse170 = std::pow(cse16, -5);
  const amrex::Real cse171 = std::pow(cse162, 5)*cse170;
  const amrex::Real cse172 = std::pow(cse16, -4);
  const amrex::Real cse173 = std::pow(cse162, 4)*cse172;
  const amrex::Real cse174 = std::pow(cse16, -3);
  const amrex::Real cse175 = std::pow(cse162, 3);
  const amrex::Real cse176 = cse174*cse175;
  const amrex::Real cse177 = std::pow(cse162, 2)*cse21;
  const amrex::Real cse178 = std::pow(10.0, 16.815729999999999*cse163 + 70.609154000000004*cse165 + 364.14445999999998*cse167 + 801.81246999999996*cse169 + 983.67575999999997*cse171 + 734.71650999999997*cse173 + 343.1918*cse176 + 96.743155000000002*cse177 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse179 = state.xn[5] + state.xn[6];
  const amrex::Real cse180 = std::pow(10.0, 1.5714710999999999*cse163 + 0.32168730000000001*cse171 - 0.51002221000000003*cse173 - 0.23619984999999999*cse176 + 0.015391166*cse177 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse163 + 0.19191374999999999*cse171 - 0.16596184*cse173 + 0.29036281000000003*cse176 - 0.81520437999999995*cse177 - 23.689236999999999)*cse179 + std::pow(10.0, 2.0943374000000001*cse162*cse17 - 0.033638326000000003*cse171 - 0.14913216000000001*cse173 + 0.43693353000000001*cse174*cse175 - 0.77151435999999995*cse177 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse181 = std::pow(10, 37.383713*cse163 + 3.8479610000000002*cse171 + 20.159831000000001*cse173 + 48.656103000000002*cse176 + 58.145166000000003*cse177 - 16.818342000000001)*state.xn[0] + cse180;
  const amrex::Real cse182 = std::pow(10.0, 1.6802758*cse163 - 6.3701156000000001*cse165 + 6.4380698000000001*cse167 + 8.9167182999999994*cse169 - 8.8077017000000009*cse171 - 4.7274035999999997*cse173 + 4.0406627000000004*cse176 + 0.93310621999999999*cse177 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse183 = state.T < 1000.0;
  const amrex::Real cse184 = std::pow(10.0, 3.5692468000000002*cse162*cse17 - 4.2519023000000002*cse171 - 21.328264000000001*cse173 - 27.850082*cse176 - 11.33286*cse177 - 24.311209000000002)*state.xn[0] + cse180;
  const amrex::Real cse185 = std::pow(10.0, 4.6450521*cse163 + 1.5538288*cse171 - 5.5108049000000001*cse173 + 5.9369081000000001*cse176 - 3.7209846*cse177 - 24.311209000000002)*state.xn[0] + cse180;
  const amrex::Real cse186 = std::pow(10.0, 2.8156982240432273*cse17 + 5.9406293599919788*cse170 - 5.2566362076906961*cse172 - 1.3586848802337363*cse174 + 0.049411830040524525*cse21 - 22.089523)*state.xn[1] + std::pow(10.0, 3.7525488682084647*cse17 - 0.62120210234156448*cse170 - 1.5370575959566981*cse172 + 2.5133588394664677*cse174 - 2.4768712409536775*cse21 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.9225864834863131*cse17 + 3.5440891133599637*cse170 - 1.7105157385968939*cse172 + 1.6702447513374001*cse174 - 2.6171337683481011*cse21 - 23.689236999999999)*cse179 + std::pow(10.0, 8.3228160852326614*cse17 + 28.69470131298657*cse170 - 56.79810800956944*cse172 + 34.150687525022562*cse174 - 11.945856385319289*cse21 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse187 = ((cse160) ? (
   cse14
)
: (
   0.00016666666666666666
));
  const amrex::Real cse188 = 1.6e-18*std::exp(-11700.0*cse187) + 6.7e-19*std::exp(-5860.0*cse187) + 3.0e-24*std::exp(-510.0*cse187) + 4.985670872372847e-33*((cse160) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse160) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse160) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse189 = std::log(0.001*((cse160) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse190 = cse170*std::pow(cse189, 5);
  const amrex::Real cse191 = cse172*std::pow(cse189, 4);
  const amrex::Real cse192 = std::pow(cse189, 3);
  const amrex::Real cse193 = std::pow(cse189, 2)*cse21;
  const amrex::Real cse194 = cse17*cse189;
  const amrex::Real cse195 = cse174*cse192;
  const amrex::Real cse196 = cse164*std::pow(cse189, 8);
  const amrex::Real cse197 = cse166*std::pow(cse189, 7);
  const amrex::Real cse198 = cse168*std::pow(cse189, 6);
  const amrex::Real cse199 = 0.045999999999999999*cse127;
  const amrex::Real cse200 = 1.7258000000000001e-6*cse125;
  const amrex::Real cse201 = 8.9000000000000003e-11*std::pow(cse126, cse199 + 0.22800000000000001)*state.xn[0] + cse200*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(0.0070527699999999999*cse70 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse70*(cse70*(cse70*(0.093813800000000003*cse70 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse126, 0.037999999999999999*cse127 + 0.16900000000000001) + 2.15e-11*std::pow(cse126, 0.048000000000000001*cse127 - 0.01))*state.xn[3];
  const amrex::Real cse202 = 2.6400000000000002e-10*std::pow(cse126, cse199 + 0.23100000000000001)*state.xn[0] + cse200*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(0.00235272*cse70 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse70*(cse70*(cse70*(0.097857299999999994*cse70 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse126, 0.031*cse127 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse126, 0.064000000000000001*cse127 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse203 = 2.8763333333333333e-6*cse125*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(-0.00065632500000000003*cse70 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse70*(cse70*(cse70*(0.10508000000000001*cse70 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse126, 0.057000000000000002*cse127 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse126, 0.049000000000000002*cse127 - 0.0040000000000000001) + 1.675e-11*std::pow(cse126, 0.10199999999999999*cse127 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse204 = 5*std::exp(-62.460000000000001*cse14);
  const amrex::Real cse205 = cse203 + 7.8800000000000004e-8;
  const amrex::Real cse206 = cse201*cse204;
  const amrex::Real cse207 = (25.0/3.0)*cse201*cse202*std::exp(-101.30000000000001*cse14) + cse202*cse203*cse204 + cse205*cse206;
  const amrex::Real cse208 = cse202 + 2.65e-7;
  const amrex::Real cse209 = cse201 + 1.81e-14;
  const amrex::Real cse210 = 3*cse203*std::exp(-23.619999999999997*cse14);
  const amrex::Real cse211 = cse206*cse208 + cse208*cse210 + cse209*cse210;
  const amrex::Real cse212 = 1.0/((5.0/3.0)*cse202*cse209*std::exp(-38.840000000000003*cse14) + cse205*cse208 + cse205*cse209 + cse207 + cse211);

  ydot(1) = cse1 + cse11 + cse12 + cse13 + cse15 + cse18*cse33 + 2*cse31 - cse32 + 2.9935562710997168e-6*cse35 - 6.0000000000000005e-15*cse36*state.xn[0] + cse37 + 1.1399999999999999e-10*cse39*cse41*state.xn[3] + 4.2899999999999994e-13*cse44 + cse47 + cse52 + cse55 + 3.2000000000000001e-9*cse7 + cse72 + cse9;
  ydot(2) = cse0 + cse51 + cse60 + cse63 - cse65 - cse71 + cse73;
  ydot(3) = -cse72 + cse74 - cse75 - cse78 - cse79 + cse80 - cse85 - cse87 - cse96 - cse99;
  ydot(4) = -cse100 - cse101 + cse102 - cse103 - cse105*state.xn[3] - cse107 + cse11 + 3.0000000000000002e-15*cse19*d2g*state.xn[0] - cse31 + 4.5427507615910105e-7*cse34*state.xn[2]*state.xn[16] - cse51 - cse55 - cse99;
  ydot(5) = cse107 + cse108;
  ydot(6) = cse110;
  ydot(7) = -cse110;
  ydot(8) = cse111 + cse114 + 9.8999999999999995e-14*cse44 + cse46 + cse96;
  ydot(9) = cse109 - cse115 + cse89 - cse91 - cse92 + cse93 + cse95 - cse97;
  ydot(10) = -cse114 - cse116 - cse117;
  ydot(11) = cse115 + cse116;
  ydot(12) = cse112 + cse113 + cse117 + cse118 + cse73 + cse8;
  ydot(13) = -cse119;
  ydot(14) = cse119;
  ydot(15) = -cse111 + 2.3099999999999997e-13*cse42*cse43*state.xn[3]*state.xn[8] - cse47;
  ydot(16) = -cse115 - cse118 - cse15 - cse49 + cse9;
  ydot(17) = -cse102 - cse108 - cse85;
  ydot(18) = cse49 - cse73 - cse98;
  ydot(19) = 1.7e-24*chi*cse19*cse66*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse123, 0.43099999999999999)*(0.0043*std::pow(cse123, 0.69199999999999995) + 1) + 1) + 5.6999999999999997e-11*chi*cse39*cse40*(6.4087080000000003e-13 + 2.5634832e-11/(cse136*(cse105 + 1.9999999999999999e-7) + 1))*state.xn[3] + 1.602177e-12*crate*cse145*state.xn[0] + 3.2043540000000001e-12*crate*cse145*state.xn[3] + 1.7623947000000002e-12*crate*cse145*state.xn[5]/cse58 - 7.1777529600000012e-12*cse103 - 1.3659996292638161e-10*cse120*cse122*cse62/(1.3338116951315871e-8*cse121*cse122 + 1) - 5.7927337833400002e-20*cse131*state.xn[8]/(2*cse131 + 1) - cse138*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse138*cse142 + cse140*std::pow(cse138/cse143, alpha_co_interp2d(state.T, cse141))*(-cse139*cse142*cse143 + 1) + cse140) + 3.0000000000000002e-15*cse19*d2g*(3.2043540000000002e-13 + 6.7291434000000005e-12/(cse136*(cse104 + 1.9999999999999999e-7) + 1))*state.xn[0] - 7.1777529600000012e-12*cse31 - 1.8998126555754159e-29*cse36*std::pow(state.T, 0.45350000000000001 + 2.234/cse124)*std::exp(cse124*(1.4419999999999999 - 0.050889999999999998*cse124))*state.xn[2] - 2.1789607199999999e-11*cse63 - (2.3872025442750958e-19*cse153*cse158 + 2.8023005140020001e-18*cse157*cse158)*state.xn[11] - ((((cse128 && cse159 && cse160) ? (
   cse161 || cse178 + cse181 < 1.0e-100
)
: (
   ((cse159 && cse160) ? (
      cse161 || cse181 + cse182 < 1.0e-100
   )
   : (
      ((cse128 && cse160 && cse183) ? (
         cse161 || cse178 + cse184 < 1.0e-100
      )
      : (
         ((cse160 && cse183) ? (
            cse161 || cse182 + cse184 < 1.0e-100
         )
         : (
            ((cse128 && cse160) ? (
               cse161 || cse178 + cse185 < 1.0e-100
            )
            : (
               ((cse160) ? (
                  cse161 || cse182 + cse185 < 1.0e-100
               )
               : (
                  ((cse128) ? (
                     cse161 || std::pow(10.0, 7500.646099484944*cse164 + 21588.957206746247*cse166 + 26530.843406310876*cse168 + 30.129743459482278*cse17 + 18165.631967965237*cse170 + 7572.4886742758581*cse172 + 1974.131269262874*cse174 + 310.58441786958315*cse21 - 21.928795999999998)*state.xn[2] + cse186 < 1.0e-100
                  )
                  : (
                     cse161 || std::pow(10.0, -676.68255490510751*cse164 + 381.69251128040054*cse166 + 295.04162851880631*cse168 + 3.0106500755647456*cse17 - 162.65264843551881*cse170 - 48.723840736518795*cse172 + 23.242975457496804*cse174 + 2.9956460707652877*cse21 - 22.921188999999998)*state.xn[2] + cse186 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
))) ? (
   0.0
)
: (
   cse188/(cse188/(std::pow(10.0, 0.19191374999999999*cse190 - 0.16596184*cse191 - 0.81520437999999995*cse193 + 2.1892372*cse194 + 0.29036281000000003*cse195 - 23.689236999999999)*cse179 + std::pow(10.0, 0.32168730000000001*cse190 - 0.51002221000000003*cse191 + 0.015391166*cse193 + 1.5714710999999999*cse194 - 0.23619984999999999*cse195 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse17*cse189 + 0.43693353000000001*cse174*cse192 - 0.033638326000000003*cse190 - 0.14913216000000001*cse191 - 0.77151435999999995*cse193 - 23.962112000000001)*state.xn[3] + ((cse128) ? (
      std::pow(10.0, 983.67575999999997*cse190 + 734.71650999999997*cse191 + 96.743155000000002*cse193 + 16.815729999999999*cse194 + 343.1918*cse195 + 70.609154000000004*cse196 + 364.14445999999998*cse197 + 801.81246999999996*cse198 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse190 - 4.7274035999999997*cse191 + 0.93310621999999999*cse193 + 1.6802758*cse194 + 4.0406627000000004*cse195 - 6.3701156000000001*cse196 + 6.4380698000000001*cse197 + 8.9167182999999994*cse198 - 22.921188999999998)
   ))*state.xn[2] + ((((cse160) ? (
      cse159
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse190 + 20.159831000000001*cse191 + 58.145166000000003*cse193 + 37.383713*cse194 + 48.656103000000002*cse195 - 16.818342000000001)
   )
   : ((((cse160) ? (
      cse183
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse17*cse189 - 4.2519023000000002*cse190 - 21.328264000000001*cse191 - 11.33286*cse193 - 27.850082*cse195 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse190 - 5.5108049000000001*cse191 - 3.7209846*cse193 + 4.6450521*cse194 + 5.9369081000000001*cse195 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse207*cse212 + 2.5697412351439998e-22*cse211*cse212
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse19, state.T), 3.2000000000000002e-32*cse2*cse36*(state.T - 2.1544346900318838*std::pow(85973.158732959506*cse67 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = cse78;
  ydot(21) = -cse76*state.rn[2*0+1];
  // $JAFF END
  // $JAFF SUB nspec
  ydot(19) /= state.rho;
  // $JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
  const amrex::Real cse0 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001);
  const amrex::Real cse1 = cse0*state.xn[14];
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = 1.0/state.T;
  const amrex::Real cse4 = 63590*cse3 + 1;
  const amrex::Real cse5 = std::exp(-cse4);
  const amrex::Real cse6 = std::log(6.6699999999999996e-12*cse2*cse5);
  const amrex::Real cse7 = M_LN10;
  const amrex::Real cse8 = 1.0/cse7;
  const amrex::Real cse9 = 2*state.xn[3];
  const amrex::Real cse10 = cse9 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + 2*state.xn[4];
  const amrex::Real cse11 = 1.0/cse10;
  const amrex::Real cse12 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse13 = std::pow(cse7, -2);
  const amrex::Real cse14 = std::pow(cse12, 2)*cse13;
  const amrex::Real cse15 = std::pow(10, 1.3*cse12*cse8 - 1.6200000000000001*cse14 - 4.8449999999999998);
  const amrex::Real cse16 = cse11*cse15*state.xn[3];
  const amrex::Real cse17 = std::pow(10, 0.41599999999999998*cse12*cse8 + 0.32700000000000001*cse14 - 3);
  const amrex::Real cse18 = cse11*cse17*state.xn[0];
  const amrex::Real cse19 = cse16 + cse18;
  const amrex::Real cse20 = cse19 < 1.0e-50;
  const amrex::Real cse21 = ((cse20) ? (
   1.0e-50
)
: (
   cse19
));
  const amrex::Real cse22 = cse10*cse21;
  const amrex::Real cse23 = cse22 + 1;
  const amrex::Real cse24 = cse8/cse23;
  const amrex::Real cse25 = std::log(3.5199999999999998e-9*std::exp(-43900*cse3));
  const amrex::Real cse26 = cse22*cse24;
  const amrex::Real cse27 = std::pow(10, cse24*cse6 + cse25*cse26);
  const amrex::Real cse28 = state.T > 700.0;
  const amrex::Real cse29 = ((cse28) ? (
   cse27
)
: (
   0.0
));
  const amrex::Real cse30 = std::pow(state.xn[3], 2);
  const amrex::Real cse31 = std::log(1.3000000000000001e-9*std::exp(-53300*cse3));
  const amrex::Real cse32 = cse21*cse24;
  const amrex::Real cse33 = cse31*cse32;
  const amrex::Real cse34 = std::pow(cse10, 2);
  const amrex::Real cse35 = 1.0/cse34;
  const amrex::Real cse36 = cse35*state.xn[3];
  const amrex::Real cse37 = cse15*cse36;
  const amrex::Real cse38 = cse35*state.xn[0];
  const amrex::Real cse39 = cse17*cse38;
  const amrex::Real cse40 = cse37 + cse39;
  const amrex::Real cse41 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse17 - cse40
));
  const amrex::Real cse42 = cse24*cse41;
  const amrex::Real cse43 = std::pow(state.T, 4.1881000000000004);
  const amrex::Real cse44 = 6.7610000000000002e-6*state.T + 1;
  const amrex::Real cse45 = std::pow(cse44, 5.6881000000000004);
  const amrex::Real cse46 = 54657.400000000001*cse3;
  const amrex::Real cse47 = std::exp(-cse46);
  const amrex::Real cse48 = cse47/cse45;
  const amrex::Real cse49 = std::log(5.9959999999999997e-30*cse43*cse48);
  const amrex::Real cse50 = cse8/std::pow(cse23, 2);
  const amrex::Real cse51 = cse50*(-cse21 - cse41);
  const amrex::Real cse52 = cse22*cse51;
  const amrex::Real cse53 = std::pow(10, cse24*cse49 + cse26*cse31);
  const amrex::Real cse54 = cse53*cse7;
  const amrex::Real cse55 = ((cse28) ? (
   cse54*(cse31*cse42 + cse31*cse52 + cse33 + cse49*cse51)
)
: (
   0
));
  const amrex::Real cse56 = 6.0000000000000005e-15*d2g;
  const amrex::Real cse57 = cse56*state.xn[0];
  const amrex::Real cse58 = cse25*cse32;
  const amrex::Real cse59 = cse27*cse7;
  const amrex::Real cse60 = ((cse28) ? (
   cse59*(cse25*cse42 + cse25*cse52 + cse51*cse6 + cse58)
)
: (
   0
));
  const amrex::Real cse61 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse62 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse63 = cse62*state.xn[17];
  const amrex::Real cse64 = 1.5*cse11*state.xn[0] + 2.2999999999999998*cse11*state.xn[3];
  const amrex::Real cse65 = crate*cse64;
  const amrex::Real cse66 = 1.0/state.xn[2];
  const amrex::Real cse67 = cse2*cse66;
  const amrex::Real cse68 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse69 = chi*cse68;
  const amrex::Real cse70 = cse67*cse69;
  const amrex::Real cse71 = 1.7*cse70;
  const amrex::Real cse72 = std::log(state.T);
  const amrex::Real cse73 = 1.102e-5*cse72 + 0.4723;
  const amrex::Real cse74 = std::pow(cse71, -cse73);
  const amrex::Real cse75 = cse74*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse76 = 508.69999999999999*cse75;
  const amrex::Real cse77 = cse76 + 1;
  const amrex::Real cse78 = std::pow(cse70, 1.3779999999999999);
  const amrex::Real cse79 = 1.6774414051711492e-5*cse78;
  const amrex::Real cse80 = cse77*cse79 + 1;
  const amrex::Real cse81 = 1.0/cse80;
  const amrex::Real cse82 = 1.2250000000000001e-11*cse81;
  const amrex::Real cse83 = d2g*state.xn[1];
  const amrex::Real cse84 = cse82*cse83;
  const amrex::Real cse85 = -cse84;
  const amrex::Real cse86 = 2.2999999999999998*cse36 + 1.5*cse38;
  const amrex::Real cse87 = -1.5*cse11 + cse86;
  const amrex::Real cse88 = -cse87;
  const amrex::Real cse89 = crate*state.xn[0];
  const amrex::Real cse90 = cse88*cse89;
  const amrex::Real cse91 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse92 = std::pow(cse91, 2);
  const amrex::Real cse93 = std::pow(cse91, 3);
  const amrex::Real cse94 = std::pow(cse91, 4);
  const amrex::Real cse95 = std::pow(cse91, 5);
  const amrex::Real cse96 = std::pow(cse91, 6);
  const amrex::Real cse97 = std::pow(cse91, 7);
  const amrex::Real cse98 = std::exp(-2.0391498499999999e-6*std::pow(cse91, 8) - 5.7393287500000003*cse92 + 1.56315498*cse93 - 0.28770560000000001*cse94 + 0.034825597700000002*cse95 - 0.00263197617*cse96 + 0.000111954395*cse97);
  const amrex::Real cse99 = 5.9078447946382801e-70*cse98*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse100 = ((cse28) ? (
   cse99
)
: (
   0.0
));
  const amrex::Real cse101 = cse100*state.xn[2];
  const amrex::Real cse102 = cse101 + cse61 + cse63 + cse65 + cse85 + cse90;
  const amrex::Real cse103 = std::exp(-227*cse3);
  const amrex::Real cse104 = cse103*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse105 = cse104*state.xn[11];
  const amrex::Real cse106 = std::pow(cse3, 1.5);
  const amrex::Real cse107 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse108 = std::pow(cse107, -2.242);
  const amrex::Real cse109 = 4.8813573300130879e-6*cse106*cse108;
  const amrex::Real cse110 = cse10*d2g;
  const amrex::Real cse111 = -cse86;
  const amrex::Real cse112 = cse111*cse89;
  const amrex::Real cse113 = -cse112 + cse84;
  const amrex::Real cse114 = cse109*state.xn[2] + cse110*cse82 + cse113;
  const amrex::Real cse115 = cse105 + cse114;
  const amrex::Real cse116 = cse10*((cse20) ? (
   0
)
: (
   -cse40
));
  const amrex::Real cse117 = cse116*cse24;
  const amrex::Real cse118 = cse50*(-cse116 - cse21);
  const amrex::Real cse119 = cse118*cse22;
  const amrex::Real cse120 = cse30*((cse28) ? (
   cse54*(cse117*cse31 + cse118*cse49 + cse119*cse31 + cse33)
)
: (
   0
));
  const amrex::Real cse121 = -cse57;
  const amrex::Real cse122 = ((cse28) ? (
   cse59*(cse117*cse25 + cse118*cse6 + cse119*cse25 + cse58)
)
: (
   0
));
  const amrex::Real cse123 = cse9*state.xn[0];
  const amrex::Real cse124 = 2*cse120 + cse121 + cse122*cse123;
  const amrex::Real cse125 = std::pow(state.xn[2], 2);
  const amrex::Real cse126 = 1.0/cse125;
  const amrex::Real cse127 = 0.00011320754716981131*cse67;
  const amrex::Real cse128 = cse127*state.xn[3];
  const amrex::Real cse129 = std::min(10000000000.0, cse128);
  const amrex::Real cse130 = cse129 + 1;
  const amrex::Real cse131 = std::pow(cse130, -2);
  const amrex::Real cse132 = (((cse128 - 10000000000.0 > 0) ? (
   0
)
: ((cse128 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse133 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse134 = cse126*cse131*cse132*cse133*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse135 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse136 = 2.9935562710997168e-6*cse135;
  const amrex::Real cse137 = cse126*cse2;
  const amrex::Real cse138 = cse132*state.xn[17];
  const amrex::Real cse139 = cse138*cse30;
  const amrex::Real cse140 = cse137*cse139;
  const amrex::Real cse141 = 3.6226415094339619e-13*cse131;
  const amrex::Real cse142 = 1.0/cse130;
  const amrex::Real cse143 = 1.811320754716981e-13*cse142;
  const amrex::Real cse144 = cse140*cse143;
  const amrex::Real cse145 = 1.811320754716981e-13*cse131;
  const amrex::Real cse146 = cse140*cse145;
  const amrex::Real cse147 = cse129*cse146;
  const amrex::Real cse148 = -cse144 + cse147;
  const amrex::Real cse149 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse150 = cse149*state.xn[10];
  const amrex::Real cse151 = cse100*state.xn[0];
  const amrex::Real cse152 = cse77*cse78;
  const amrex::Real cse153 = -cse73;
  const amrex::Real cse154 = 1.2250000000000001e-11*cse110*state.xn[1]/std::pow(cse80, 2);
  const amrex::Real cse155 = cse109*state.xn[1] - cse151 + cse154*(2.3115142563258434e-5*cse152*cse66 + 0.0085331444281056355*cse153*cse66*cse75*cse78);
  const amrex::Real cse156 = cse150 + cse155;
  const amrex::Real cse157 = std::exp(-22.5*cse3);
  const amrex::Real cse158 = 1.2599999999999999e-13*cse157;
  const amrex::Real cse159 = cse158*state.xn[6];
  const amrex::Real cse160 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse161 = cse160*state.xn[4];
  const amrex::Real cse162 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse163 = 1.7599999999999999e-9*cse162;
  const amrex::Real cse164 = cse161*cse163;
  const amrex::Real cse165 = d2g*state.xn[0];
  const amrex::Real cse166 = -1.2000000000000001e-14*cse165;
  const amrex::Real cse167 = cse29*state.xn[0];
  const amrex::Real cse168 = 2*cse21;
  const amrex::Real cse169 = cse24*cse31;
  const amrex::Real cse170 = cse168*cse169;
  const amrex::Real cse171 = cse35*cse9;
  const amrex::Real cse172 = cse15*cse171 + 2*cse39;
  const amrex::Real cse173 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse15 - cse172
));
  const amrex::Real cse174 = cse50*(-cse168 - cse173);
  const amrex::Real cse175 = cse174*cse22;
  const amrex::Real cse176 = cse30*((cse28) ? (
   cse54*(cse169*cse173 + cse170 + cse174*cse49 + cse175*cse31)
)
: (
   0
));
  const amrex::Real cse177 = ((cse28) ? (
   cse53
)
: (
   0.0
));
  const amrex::Real cse178 = cse177*state.xn[3];
  const amrex::Real cse179 = 3.2000000000000001e-9*cse142;
  const amrex::Real cse180 = cse24*cse25;
  const amrex::Real cse181 = cse168*cse180;
  const amrex::Real cse182 = ((cse28) ? (
   cse59*(cse173*cse180 + cse174*cse6 + cse175*cse25 + cse181)
)
: (
   0
));
  const amrex::Real cse183 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse184 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse183)/cse183;
  const amrex::Real cse185 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse186 = chi*cse185;
  const amrex::Real cse187 = cse184*cse186;
  const amrex::Real cse188 = std::exp(-23*cse3);
  const amrex::Real cse189 = cse188*std::pow(state.T, -1.3);
  const amrex::Real cse190 = 4.2899999999999994e-13*cse189;
  const amrex::Real cse191 = cse138*cse67*state.xn[3];
  const amrex::Real cse192 = 1.6000000000000001e-9*cse142;
  const amrex::Real cse193 = cse192*state.xn[17];
  const amrex::Real cse194 = cse145*cse191;
  const amrex::Real cse195 = cse129*cse193 - cse129*cse194 + cse143*cse191;
  const amrex::Real cse196 = cse81*cse83;
  const amrex::Real cse197 = 2.4500000000000003e-11*cse196;
  const amrex::Real cse198 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse199 = -2.2999999999999998*cse11 + cse198;
  const amrex::Real cse200 = -cse199;
  const amrex::Real cse201 = cse200*cse89;
  const amrex::Real cse202 = cse131*cse132*cse66*state.xn[11]*state.xn[16];
  const amrex::Real cse203 = cse133*cse202;
  const amrex::Real cse204 = cse197 - cse201 - cse203;
  const amrex::Real cse205 = cse10*((cse20) ? (
   0
)
: (
   -cse172
));
  const amrex::Real cse206 = cse50*(-cse168 - cse205);
  const amrex::Real cse207 = cse206*cse22;
  const amrex::Real cse208 = cse30*((cse28) ? (
   cse54*(cse169*cse205 + cse170 + cse206*cse49 + cse207*cse31)
)
: (
   0
));
  const amrex::Real cse209 = ((cse28) ? (
   cse59*(cse180*cse205 + cse181 + cse206*cse6 + cse207*cse25)
)
: (
   0
));
  const amrex::Real cse210 = cse160*cse163*state.xn[3];
  const amrex::Real cse211 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse212 = cse210 - cse211;
  const amrex::Real cse213 = -cse198;
  const amrex::Real cse214 = cse213*cse89;
  const amrex::Real cse215 = cse197 - cse214;
  const amrex::Real cse216 = cse158*state.xn[3];
  const amrex::Real cse217 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse218 = 1.3500000000000001e-9*cse217;
  const amrex::Real cse219 = cse218*state.xn[15];
  const amrex::Real cse220 = cse216 + cse219;
  const amrex::Real cse221 = std::exp(0.11*cse3);
  const amrex::Real cse222 = cse221*state.xn[15];
  const amrex::Real cse223 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse224 = cse222*cse223;
  const amrex::Real cse225 = 6.5199999999999996*crate;
  const amrex::Real cse226 = cse149*state.xn[2];
  const amrex::Real cse227 = cse113 + cse226;
  const amrex::Real cse228 = cse142*state.xn[16];
  const amrex::Real cse229 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse230 = cse228*cse229;
  const amrex::Real cse231 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse232 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse233 = cse104*state.xn[1];
  const amrex::Real cse234 = cse232 + cse233;
  const amrex::Real cse235 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse236 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse237 = cse0*state.xn[0];
  const amrex::Real cse238 = -cse237;
  const amrex::Real cse239 = cse113 + cse124;
  const amrex::Real cse240 = cse218*state.xn[6];
  const amrex::Real cse241 = cse221*cse223*state.xn[7];
  const amrex::Real cse242 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse243 = cse240 + cse241 + cse242;
  const amrex::Real cse244 = 3*cse21;
  const amrex::Real cse245 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse246 = cse50*(-cse244 - cse245);
  const amrex::Real cse247 = cse22*cse246;
  const amrex::Real cse248 = cse30*((cse28) ? (
   cse54*(cse169*cse244 + cse169*cse245 + cse246*cse49 + cse247*cse31)
)
: (
   0
));
  const amrex::Real cse249 = ((cse28) ? (
   cse59*(cse180*cse244 + cse180*cse245 + cse246*cse6 + cse247*cse25)
)
: (
   0
));
  const amrex::Real cse250 = cse229*state.xn[11];
  const amrex::Real cse251 = cse142*cse250;
  const amrex::Real cse252 = 6.8999999999999995*cse36 + 4.5*cse38;
  const amrex::Real cse253 = -cse252;
  const amrex::Real cse254 = cse253*cse89;
  const amrex::Real cse255 = 3.6750000000000002e-11*cse196;
  const amrex::Real cse256 = cse251 - cse254 + cse255;
  const amrex::Real cse257 = cse62*state.xn[0];
  const amrex::Real cse258 = cse192*state.xn[3];
  const amrex::Real cse259 = cse129*cse258;
  const amrex::Real cse260 = std::pow(state.T, -2);
  const amrex::Real cse261 = 2.8349999999999999e-12*cse157*cse260*state.xn[3]*state.xn[6];
  const amrex::Real cse262 = -cse261;
  const amrex::Real cse263 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse264 = std::sqrt(cse3);
  const amrex::Real cse265 = std::sqrt(3)*cse264*cse3*state.xn[15];
  const amrex::Real cse266 = 1.7697487500000004e-8*cse265*state.xn[6];
  const amrex::Real cse267 = cse26*cse260;
  const amrex::Real cse268 = cse13*cse3;
  const amrex::Real cse269 = cse12*cse268;
  const amrex::Real cse270 = cse3*cse8;
  const amrex::Real cse271 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(-3.2400000000000002*cse269 + 1.3*cse3*cse8) + cse18*cse7*(0.65400000000000003*cse269 + 0.41599999999999998*cse270)
));
  const amrex::Real cse272 = cse10*cse271;
  const amrex::Real cse273 = cse272*cse50;
  const amrex::Real cse274 = cse21*cse271*cse34*cse50;
  const amrex::Real cse275 = ((cse28) ? (
   cse54*(cse169*cse272 + 1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 + 53300.0*cse267 - cse273*cse49 - cse274*cse31)
)
: (
   0
));
  const amrex::Real cse276 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse277 = 1.0/cse2;
  const amrex::Real cse278 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse279 = ((cse28) ? (
   cse59*(cse180*cse272 + 149925037481.25937*cse24*cse277*(3.3349999999999998e-12*cse277*cse5 + 4.2414529999999999e-7*cse278*cse5)*std::exp(cse4) - cse25*cse274 + 43900.0*cse267 - cse273*cse6)
)
: (
   0
));
  const amrex::Real cse280 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse281 = cse103*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse282 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse283 = cse188*state.xn[3]*state.xn[8];
  const amrex::Real cse284 = cse282*cse283;
  const amrex::Real cse285 = cse161*state.xn[3];
  const amrex::Real cse286 = 3.7768240343347639e-14*cse162*cse285;
  const amrex::Real cse287 = 7.3920000000000002e-11*cse285*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse288 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse289 = 227*cse105*cse260*state.xn[1];
  const amrex::Real cse290 = cse277*cse66;
  const amrex::Real cse291 = cse139*cse290;
  const amrex::Real cse292 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse293 = cse222*state.xn[7];
  const amrex::Real cse294 = 8.7465066538127767e-11*cse293*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse293*std::pow(state.T, -1.339);
  const amrex::Real cse295 = cse292 + cse294;
  const amrex::Real cse296 = state.xn[0]*state.xn[2];
  const amrex::Real cse297 = cse296*((cse28) ? (
   7.9971871901929577e-69*cse98*std::pow(state.T, 12.536555999999999) + cse99*(-11.478657500000001*cse3*cse91 + 4.6894649399999997*cse3*cse92 - 1.1508224*cse3*cse93 + 0.1741279885*cse3*cse94 - 0.015791857020000001*cse3*cse95 + 0.00078368076500000001*cse3*cse96 - 1.6313198799999999e-5*cse3*cse97)
)
: (
   0
));
  const amrex::Real cse298 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse299 = 7.3220359950196314e-6*cse106*cse108*cse298;
  const amrex::Real cse300 = 0.0005114031087522854*std::pow(cse107, -3.242)*cse298*std::pow(cse3, 1.907);
  const amrex::Real cse301 = 0.5*cse3;
  const amrex::Real cse302 = cse3*std::log(cse71);
  const amrex::Real cse303 = cse154*(-1.1557571281629217e-5*cse152*cse3 - cse79*(8.0679819999999989*cse74*std::pow(state.T, -0.98414000000000001) + cse76*(cse153*cse301 - 1.102e-5*cse302)));
  const amrex::Real cse304 = 9.0566037735849048e-14*cse291;
  const amrex::Real cse305 = cse142*cse304;
  const amrex::Real cse306 = cse131*cse304;
  const amrex::Real cse307 = cse129*cse306;
  const amrex::Real cse308 = -cse305 + cse307;
  const amrex::Real cse309 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse310 = 1.1264150943396225e-13*cse202*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse228*cse309;
  const amrex::Real cse311 = cse297 + cse299 - cse300 - cse303 + cse308 + cse310;
  const amrex::Real cse312 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse313 = -cse197;
  const amrex::Real cse314 = cse112 + cse85;
  const amrex::Real cse315 = -cse276 + cse281 + cse289;
  const amrex::Real cse316 = crate*cse9;
  const amrex::Real cse317 = cse316*cse88;
  const amrex::Real cse318 = 5.4939999999999998e-7*cse72 + 0.49559999999999998;
  const amrex::Real cse319 = std::pow(cse71, -cse318);
  const amrex::Real cse320 = cse319*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse321 = 5115.0*cse320;
  const amrex::Real cse322 = cse321 + 1;
  const amrex::Real cse323 = std::pow(cse70, 1.512);
  const amrex::Real cse324 = 7.104740267838494e-7*cse323;
  const amrex::Real cse325 = cse322*cse324 + 1;
  const amrex::Real cse326 = 1.0/cse325;
  const amrex::Real cse327 = 5.5720000000000001e-12*cse326;
  const amrex::Real cse328 = d2g*state.xn[6];
  const amrex::Real cse329 = cse327*cse328;
  const amrex::Real cse330 = crate*cse36;
  const amrex::Real cse331 = cse330*state.xn[7];
  const amrex::Real cse332 = 0.00013329999999999999*cse72 + 0.81200000000000006;
  const amrex::Real cse333 = std::pow(cse71, -cse332);
  const amrex::Real cse334 = cse333*std::pow(state.T, 0.04845);
  const amrex::Real cse335 = 433.10000000000002*cse334;
  const amrex::Real cse336 = cse335 + 1;
  const amrex::Real cse337 = std::pow(cse70, 1.1279999999999999);
  const amrex::Real cse338 = 0.011078790689599145*cse337;
  const amrex::Real cse339 = cse336*cse338 + 1;
  const amrex::Real cse340 = 1.0/cse339;
  const amrex::Real cse341 = 4.5579999999999993e-11*cse340;
  const amrex::Real cse342 = d2g*state.xn[8];
  const amrex::Real cse343 = 1040.0*cse331 + cse341*cse342;
  const amrex::Real cse344 = cse330*state.xn[12];
  const amrex::Real cse345 = 7.538e-5*cse72 + 0.89639999999999997;
  const amrex::Real cse346 = std::pow(cse71, -cse345);
  const amrex::Real cse347 = cse346*std::pow(state.T, 1.635e-6);
  const amrex::Real cse348 = 43750.0*cse347;
  const amrex::Real cse349 = cse348 + 1;
  const amrex::Real cse350 = std::pow(cse70, 1.8740000000000001);
  const amrex::Real cse351 = 1.534817233071468e-7*cse350;
  const amrex::Real cse352 = cse349*cse351 + 1;
  const amrex::Real cse353 = 1.0/cse352;
  const amrex::Real cse354 = 2.1659999999999999e-12*cse353;
  const amrex::Real cse355 = d2g*state.xn[13];
  const amrex::Real cse356 = 16800.0*cse344 + cse354*cse355;
  const amrex::Real cse357 = cse329 + cse343 + cse356;
  const amrex::Real cse358 = cse111*cse316;
  const amrex::Real cse359 = cse357 - cse358;
  const amrex::Real cse360 = cse72*cse8;
  const amrex::Real cse361 = 0.044330000000000001*cse360 - 0.28520000000000001;
  const amrex::Real cse362 = cse360*cse361 - 1.6759999999999999;
  const amrex::Real cse363 = cse360*cse362 + 11.19;
  const amrex::Real cse364 = cse363*state.xn[6];
  const amrex::Real cse365 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse366 = cse322*cse323;
  const amrex::Real cse367 = -cse318;
  const amrex::Real cse368 = 5.5720000000000001e-12*cse110*state.xn[6]/std::pow(cse325, 2);
  const amrex::Real cse369 = cse364*cse365 + cse368*(0.0036340746469993896*cse320*cse323*cse367*cse66 + 1.0742367284971802e-6*cse366*cse66);
  const amrex::Real cse370 = std::exp(-15230.0*cse3);
  const amrex::Real cse371 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse372 = std::exp(-12.17*cse3);
  const amrex::Real cse373 = 1.6339999999999999e-6*cse370 + 9.7930000000000004e-9*cse371 + 6.3460000000000002e-9*cse372;
  const amrex::Real cse374 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse375 = std::exp(-49550.0*cse3);
  const amrex::Real cse376 = 0.15970000000000001*cse375;
  const amrex::Real cse377 = cse376 + 1.7848999999999999;
  const amrex::Real cse378 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse379 = 0.21509999999999996 - cse376;
  const amrex::Real cse380 = std::pow(cse374, -cse377)*std::pow(cse378, -cse379);
  const amrex::Real cse381 = 2.4460185352936309e-10*cse277*cse380;
  const amrex::Real cse382 = cse278*cse373 + cse381;
  const amrex::Real cse383 = cse336*cse337;
  const amrex::Real cse384 = -cse332;
  const amrex::Real cse385 = 4.5579999999999993e-11*cse110*state.xn[8]/std::pow(cse339, 2);
  const amrex::Real cse386 = cse382*state.xn[8] + cse385*(4.7982242476653898*cse334*cse337*cse384*cse66 + 0.012496875897867835*cse383*cse66);
  const amrex::Real cse387 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse388 = cse349*cse350;
  const amrex::Real cse389 = -cse345;
  const amrex::Real cse390 = 2.1659999999999999e-12*cse110*state.xn[13]/std::pow(cse352, 2);
  const amrex::Real cse391 = cse387*state.xn[13] + cse390*(0.0067148253946876721*cse347*cse350*cse389*cse66 + 2.8762474947759312e-7*cse388*cse66);
  const amrex::Real cse392 = cse146 + cse148;
  const amrex::Real cse393 = 1.3007021411393064e-6*cse135;
  const amrex::Real cse394 = cse11*state.xn[2];
  const amrex::Real cse395 = cse394 < 1.0e-50;
  const amrex::Real cse396 = ((cse395) ? (
   0
)
: (
   -0.00011320754716981131*cse137*state.xn[3]
));
  const amrex::Real cse397 = ((cse395) ? (
   10000000000.0
)
: (
   cse128
));
  const amrex::Real cse398 = cse397 + 1;
  const amrex::Real cse399 = 1.0/cse398;
  const amrex::Real cse400 = cse250*state.xn[16];
  const amrex::Real cse401 = cse399*cse400;
  const amrex::Real cse402 = cse396*cse401;
  const amrex::Real cse403 = cse397*cse400/std::pow(cse398, 2);
  const amrex::Real cse404 = cse396*cse403;
  const amrex::Real cse405 = cse402 - cse404;
  const amrex::Real cse406 = cse134 + cse405;
  const amrex::Real cse407 = cse393*state.xn[16] + cse406;
  const amrex::Real cse408 = cse326*cse328;
  const amrex::Real cse409 = 1.1144e-11*cse408;
  const amrex::Real cse410 = 2*cse65;
  const amrex::Real cse411 = cse200*cse316;
  const amrex::Real cse412 = crate*cse11;
  const amrex::Real cse413 = 16800.0*cse412;
  const amrex::Real cse414 = cse353*cse355;
  const amrex::Real cse415 = 33600.0*cse344 + 4.3319999999999997e-12*cse414;
  const amrex::Real cse416 = -cse413*state.xn[12] + cse415;
  const amrex::Real cse417 = 3.2999999999999996e-13*cse189;
  const amrex::Real cse418 = cse417*state.xn[8];
  const amrex::Real cse419 = 1040.0*cse412;
  const amrex::Real cse420 = -cse419*state.xn[7];
  const amrex::Real cse421 = cse340*cse342;
  const amrex::Real cse422 = 2080.0*cse331 + 9.1159999999999986e-11*cse421;
  const amrex::Real cse423 = cse418 + cse420 + cse422;
  const amrex::Real cse424 = ((cse395) ? (
   0
)
: (
   cse127
));
  const amrex::Real cse425 = cse401*cse424;
  const amrex::Real cse426 = cse403*cse424;
  const amrex::Real cse427 = cse425 - cse426;
  const amrex::Real cse428 = cse195 + cse427;
  const amrex::Real cse429 = cse193 - cse194;
  const amrex::Real cse430 = cse213*cse316;
  const amrex::Real cse431 = -cse430;
  const amrex::Real cse432 = 1.1000000000000001*crate;
  const amrex::Real cse433 = cse365*state.xn[2];
  const amrex::Real cse434 = cse110*cse327 + cse363*cse433;
  const amrex::Real cse435 = std::pow(cse3, 0.00231);
  const amrex::Real cse436 = std::pow(state.T, -1.5);
  const amrex::Real cse437 = std::exp(-8110.0*cse3);
  const amrex::Real cse438 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse439 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse440 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse441 = 0.00015100000000000001*cse437 + 1.31e-7*cse438 + 3.4e-8*cse439 + 6.9699999999999997e-9*cse440;
  const amrex::Real cse442 = 1.0537934368240927e-9*cse435 + cse436*cse441;
  const amrex::Real cse443 = cse442*state.xn[16];
  const amrex::Real cse444 = -cse443;
  const amrex::Real cse445 = 2.808e-22*ncol_H2;
  const amrex::Real cse446 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse445)*std::exp(-1.6000000000000001e-17*ncol_C)/(cse445 + 1) + 3.8500000000000001*crate + cse419*state.xn[3];
  const amrex::Real cse447 = cse417*state.xn[3];
  const amrex::Real cse448 = cse110*cse341 + cse382*state.xn[2];
  const amrex::Real cse449 = cse447 + cse448;
  const amrex::Real cse450 = cse397*cse399;
  const amrex::Real cse451 = cse450*state.xn[16];
  const amrex::Real cse452 = cse229*cse451;
  const amrex::Real cse453 = cse230 + cse452;
  const amrex::Real cse454 = -cse453;
  const amrex::Real cse455 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse456 = cse455*state.rn[0];
  const amrex::Real cse457 = cse413*state.xn[3] + cse456;
  const amrex::Real cse458 = cse110*cse354 + cse387*state.xn[2];
  const amrex::Real cse459 = -cse458;
  const amrex::Real cse460 = -cse113 - cse359;
  const amrex::Real cse461 = cse250*cse450;
  const amrex::Real cse462 = cse442*state.xn[7];
  const amrex::Real cse463 = cse393*state.xn[2];
  const amrex::Real cse464 = cse253*cse316;
  const amrex::Real cse465 = -cse464;
  const amrex::Real cse466 = 1.6715999999999999e-11*cse408;
  const amrex::Real cse467 = 3120.0*cse331 + 1.3673999999999998e-10*cse421;
  const amrex::Real cse468 = 50400.0*cse344 + 6.4979999999999992e-12*cse414;
  const amrex::Real cse469 = cse258 + cse259;
  const amrex::Real cse470 = -cse469;
  const amrex::Real cse471 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse435 + cse436*(1.2246100000000002*cse260*cse437 + 3.4846000000000001e-6*cse260*cse438 + 2.5908000000000001e-7*cse260*cse439 + 9.6185999999999982e-9*cse260*cse440) - 1.5*cse441*std::pow(state.T, -2.5));
  const amrex::Real cse472 = -cse471;
  const amrex::Real cse473 = 6.7636511339243933e-7*cse280;
  const amrex::Real cse474 = cse309*cse451;
  const amrex::Real cse475 = ((cse395) ? (
   0
)
: (
   5.6603773584905654e-5*cse290*state.xn[3]
));
  const amrex::Real cse476 = cse401*cse475;
  const amrex::Real cse477 = -cse476;
  const amrex::Real cse478 = cse403*cse475;
  const amrex::Real cse479 = 7913.1350000000002*cse260*cse375;
  const amrex::Real cse480 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse278*cse380 + cse278*(0.024885819999999999*cse260*cse370 + 7.2272340000000005e-7*cse260*cse371 + 7.7230820000000007e-8*cse260*cse372) - 3.0/2.0*cse373/std::pow(state.T, 5.0/2.0) + cse381*(-0.00035870183966449452*cse277*cse377/cse374 - cse479*std::log(cse374)) + cse381*(-6.1221939997287613*cse277*cse379/cse378 + cse479*std::log(cse378)));
  const amrex::Real cse481 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse482 = 4.9999999999999997e-12*cse364*cse436*state.xn[2];
  const amrex::Real cse483 = cse283*cse288;
  const amrex::Real cse484 = 7.5899999999999998e-12*cse483;
  const amrex::Real cse485 = cse433*state.xn[6]*(cse270*cse362 + cse360*(0.044330000000000001*cse268*cse72 + cse270*cse361));
  const amrex::Real cse486 = cse390*(-1.4381237473879656e-7*cse3*cse388 - cse351*(0.071531250000000005*cse346*std::pow(state.T, -0.99999836499999994) + cse348*(cse301*cse389 - 7.538e-5*cse302)));
  const amrex::Real cse487 = cse368*(-5.3711836424859012e-7*cse3*cse366 - cse324*(0.0019963845000000001*cse319*std::pow(state.T, -0.99999960970000001) + cse321*(cse301*cse367 - 5.4939999999999998e-7*cse302)));
  const amrex::Real cse488 = cse385*(-0.0062484379489339175*cse3*cse383 - cse338*(20.983695000000001*cse333*std::pow(state.T, -0.95155000000000001) + cse335*(cse301*cse384 - 0.00013329999999999999*cse302)));
  const amrex::Real cse489 = cse455*state.xn[12];
  const amrex::Real cse490 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse491 = cse29*state.xn[3];
  const amrex::Real cse492 = cse30*cse55;
  const amrex::Real cse493 = -cse1;
  const amrex::Real cse494 = state.xn[0]*state.xn[3];
  const amrex::Real cse495 = cse494*cse60;
  const amrex::Real cse496 = cse317 - cse61;
  const amrex::Real cse497 = cse122*cse494;
  const amrex::Real cse498 = cse120 + cse358 + cse490 + cse497;
  const amrex::Real cse499 = -cse498;
  const amrex::Real cse500 = 4.5427507615910105e-7*cse135;
  const amrex::Real cse501 = 5.6999999999999997e-11*cse186;
  const amrex::Real cse502 = cse184*cse501;
  const amrex::Real cse503 = cse182*cse494;
  const amrex::Real cse504 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse505 = cse159 + cse504;
  const amrex::Real cse506 = cse410 + cse411;
  const amrex::Real cse507 = cse164 + cse203 - cse425 + cse426;
  const amrex::Real cse508 = cse195 + cse429;
  const amrex::Real cse509 = cse209*cse494;
  const amrex::Real cse510 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse511 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse512 = 9.0000000000000011e-15*cse165;
  const amrex::Real cse513 = cse249*cse494;
  const amrex::Real cse514 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse515 = cse251 + cse461 + cse462 + cse514;
  const amrex::Real cse516 = -cse263;
  const amrex::Real cse517 = cse275*cse30;
  const amrex::Real cse518 = cse279*cse494;
  const amrex::Real cse519 = -4.2899999999999994e-13*cse188*cse282*state.xn[3]*state.xn[8] + cse484;
  const amrex::Real cse520 = cse305 - cse307;
  const amrex::Real cse521 = -cse286 + cse287;
  const amrex::Real cse522 = cse310 + cse472 + cse474 + cse477 + cse478 + cse521;
  const amrex::Real cse523 = cse409 + cse505;
  const amrex::Real cse524 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse525 = cse220 + cse434 + cse510 + cse524;
  const amrex::Real cse526 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse527 = cse240 + cse329;
  const amrex::Real cse528 = -cse266;
  const amrex::Real cse529 = cse261 - cse482 + cse485 + cse487 + cse528;
  const amrex::Real cse530 = -cse329;
  const amrex::Real cse531 = cse330*state.xn[9];
  const amrex::Real cse532 = 184.0*cse531;
  const amrex::Real cse533 = -cse532;
  const amrex::Real cse534 = cse343 + cse533;
  const amrex::Real cse535 = 184.0*cse412;
  const amrex::Real cse536 = cse535*state.xn[9];
  const amrex::Real cse537 = 9.8999999999999995e-14*cse189;
  const amrex::Real cse538 = 368.0*cse531;
  const amrex::Real cse539 = -cse538;
  const amrex::Real cse540 = cse422 + cse539;
  const amrex::Real cse541 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av) + cse535*state.xn[3];
  const amrex::Real cse542 = cse236 + cse237;
  const amrex::Real cse543 = 552.0*cse531;
  const amrex::Real cse544 = cse480 + cse488;
  const amrex::Real cse545 = -cse343;
  const amrex::Real cse546 = 9.1500000000000005e-10*cse217;
  const amrex::Real cse547 = cse546*state.xn[15];
  const amrex::Real cse548 = cse546*state.xn[8];
  const amrex::Real cse549 = 1.1994963750000002e-8*cse265*state.xn[8];
  const amrex::Real cse550 = -cse549;
  const amrex::Real cse551 = -cse536 + cse538;
  const amrex::Real cse552 = cse225 + cse511;
  const amrex::Real cse553 = cse526 + cse541;
  const amrex::Real cse554 = cse235 + cse532;
  const amrex::Real cse555 = -cse63;
  const amrex::Real cse556 = -cse231 + cse452;
  const amrex::Real cse557 = -cse474 + cse476 - cse478;
  const amrex::Real cse558 = cse306 + cse315;
  const amrex::Real cse559 = -cse481 + cse486;
  const amrex::Real cse560 = -cse489;
  const amrex::Real cse561 = -cse356;
  const amrex::Real cse562 = 2.3099999999999997e-13*cse189;
  const amrex::Real cse563 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse564 = cse71 + 50;
  const amrex::Real cse565 = 0.0043*std::pow(cse564, 0.69199999999999995) + 1;
  const amrex::Real cse566 = 0.049959999999999997*std::pow(cse564, 0.43099999999999999)*cse565 + 1;
  const amrex::Real cse567 = 1.0/cse566;
  const amrex::Real cse568 = state.xn[2] + 1.0e-50;
  const amrex::Real cse569 = 1.0/cse568;
  const amrex::Real cse570 = cse2*cse69;
  const amrex::Real cse571 = 1.7*cse570;
  const amrex::Real cse572 = cse569*cse571 + 50;
  const amrex::Real cse573 = std::log(cse572);
  const amrex::Real cse574 = 0.45350000000000001 + 2.234/cse573;
  const amrex::Real cse575 = std::pow(state.T, cse574);
  const amrex::Real cse576 = 1.4419999999999999 - 0.050889999999999998*cse573;
  const amrex::Real cse577 = std::exp(cse573*cse576);
  const amrex::Real cse578 = 1.8998126555754159e-29*cse575*cse577;
  const amrex::Real cse579 = d2g*state.xn[2];
  const amrex::Real cse580 = cse578*cse579;
  const amrex::Real cse581 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse582 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse583 = 1.0/cse582;
  const amrex::Real cse584 = cse583*state.xn[2];
  const amrex::Real cse585 = cse581*cse584;
  const amrex::Real cse586 = 4.4460389837719569e-9*cse585 + 626500000.0;
  const amrex::Real cse587 = 1.0/cse586;
  const amrex::Real cse588 = std::exp(-118400.0*cse3);
  const amrex::Real cse589 = cse587*cse588;
  const amrex::Real cse590 = 1.3338116951315871e-8*cse589;
  const amrex::Real cse591 = cse585*cse590 + 1;
  const amrex::Real cse592 = 1.0/cse591;
  const amrex::Real cse593 = cse589*cse592;
  const amrex::Real cse594 = 1.3659996292638161e-10*cse593;
  const amrex::Real cse595 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse596 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse597 = 100.0*d2g;
  const amrex::Real cse598 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse69 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse599 = cse110*cse595;
  const amrex::Real cse600 = -cse596*cse597 + cse598*cse599;
  const amrex::Real cse601 = (((cse600 > 0) ? (
   0
)
: ((cse600 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse602 = cse598*cse601*d2g;
  const amrex::Real cse603 = cse595*cse602;
  const amrex::Real cse604 = (((cse600 < 0) ? (
   0
)
: ((cse600 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse605 = cse604*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse606 = cse597*cse605;
  const amrex::Real cse607 = 0.001*state.T;
  const amrex::Real cse608 = cse607 + 1;
  const amrex::Real cse609 = std::pow(cse608, -2);
  const amrex::Real cse610 = 1.0/cse608;
  const amrex::Real cse611 = std::pow(10, -2.0230000000000001*cse609 - 3.6709999999999998*cse610 - 11.08);
  const amrex::Real cse612 = cse11*cse611;
  const amrex::Real cse613 = cse612*state.xn[3];
  const amrex::Real cse614 = std::pow(10, -2.3900000000000001*cse609 + 0.055500000000000001*cse610 - 11.06);
  const amrex::Real cse615 = cse11*cse614;
  const amrex::Real cse616 = cse615*state.xn[0];
  const amrex::Real cse617 = cse613 + cse616;
  const amrex::Real cse618 = 1.0/cse617;
  const amrex::Real cse619 = cse501 + 1.9999999999999999e-7;
  const amrex::Real cse620 = cse618*cse619;
  const amrex::Real cse621 = cse11*cse620 + 1;
  const amrex::Real cse622 = 3.2043540000000002e-13 + 6.7291434000000005e-12/cse621;
  const amrex::Real cse623 = cse264*state.xn[2];
  const amrex::Real cse624 = (1.0/100.0)*state.T;
  const amrex::Real cse625 = std::log(cse624);
  const amrex::Real cse626 = 0.0086999999999999994*cse625;
  const amrex::Real cse627 = std::pow(cse624, cse626 + 0.12809999999999999);
  const amrex::Real cse628 = 7.5799999999999997e-10*cse627*state.xn[0];
  const amrex::Real cse629 = state.T < 500;
  const amrex::Real cse630 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse631 = 0.75*((cse629) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse630
)) + 0.25*((cse629) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse630
));
  const amrex::Real cse632 = 4.5299999999999998e-6*cse623 + cse628 + cse631*state.xn[3];
  const amrex::Real cse633 = cse632 + 2.3e-6;
  const amrex::Real cse634 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse635 = cse634/cse633;
  const amrex::Real cse636 = 2*cse635;
  const amrex::Real cse637 = cse632*cse636 + 1;
  const amrex::Real cse638 = 1.0/cse637;
  const amrex::Real cse639 = cse635*state.xn[8];
  const amrex::Real cse640 = cse638*cse639;
  const amrex::Real cse641 = cse394 + 0.070000000000000007;
  const amrex::Real cse642 = 1.0/cse641;
  const amrex::Real cse643 = std::sqrt(cse394*cse642);
  const amrex::Real cse644 = 26.399999999999999*cse643;
  const amrex::Real cse645 = cse644 + 6.5;
  const amrex::Real cse646 = cse11*cse645;
  const amrex::Real cse647 = cse8*std::log(cse10);
  const amrex::Real cse648 = cse647 < 2;
  const amrex::Real cse649 = cse647 < 4;
  const amrex::Real cse650 = cse647 < 7;
  const amrex::Real cse651 = cse647 < 10;
  const amrex::Real cse652 = ((cse648) ? (
   10.0
)
: ((cse649) ? (
   (3.0/2.0)*cse647 + 7
)
: ((cse650) ? (
   (4.0/3.0)*cse647 + 23.0/3.0
)
: ((cse651) ? (
   (1.0/3.0)*cse647 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse653 = 2.0*cse11;
  const amrex::Real cse654 = cse652*cse653;
  const amrex::Real cse655 = cse646*state.xn[0] + cse654*state.xn[3];
  const amrex::Real cse656 = std::pow(cse633, -2);
  const amrex::Real cse657 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse658 = 1.75*cse657;
  const amrex::Real cse659 = std::exp(-state.T);
  const amrex::Real cse660 = 1 - cse659;
  const amrex::Real cse661 = std::pow(cse660, 1000);
  const amrex::Real cse662 = l0_co_interp1d(state.T);
  const amrex::Real cse663 = 1.0/cse662;
  const amrex::Real cse664 = 680.10000000000002/cse657;
  const amrex::Real cse665 = cse658*state.xn[0] + cse664*state.xn[2] + state.xn[3];
  const amrex::Real cse666 = M_SQRT2;
  const amrex::Real cse667 = std::max(gradv, 5.7445531974402559e-18*cse2*cse666);
  const amrex::Real cse668 = 100000.0/cse667;
  const amrex::Real cse669 = ncol_H <= 0;
  const amrex::Real cse670 = cse11*ncol_H;
  const amrex::Real cse671 = std::max(1.0000000000000001e-30, cse670);
  const amrex::Real cse672 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse673 = ((cse669) ? (
   cse668*state.xn[9]
)
: (
   cse672/std::sqrt(cse671)
));
  const amrex::Real cse674 = llte_co_interp2d(state.T, cse673);
  const amrex::Real cse675 = 1.0/cse674;
  const amrex::Real cse676 = nhalf_co_interp2d(state.T, cse673);
  const amrex::Real cse677 = cse662*cse675;
  const amrex::Real cse678 = -cse676*cse677 + 1;
  const amrex::Real cse679 = 1.0/cse676;
  const amrex::Real cse680 = cse665*cse679;
  const amrex::Real cse681 = alpha_co_interp2d(state.T, cse673);
  const amrex::Real cse682 = std::pow(cse680, cse681);
  const amrex::Real cse683 = cse663*cse682;
  const amrex::Real cse684 = cse678*cse683;
  const amrex::Real cse685 = cse663 + cse665*cse675 + cse684;
  const amrex::Real cse686 = 1.0/cse685;
  const amrex::Real cse687 = cse661*cse686;
  const amrex::Real cse688 = cse687*state.xn[9];
  const amrex::Real cse689 = std::pow(cse64, -2);
  const amrex::Real cse690 = std::pow(cse621, -2);
  const amrex::Real cse691 = cse35*cse620;
  const amrex::Real cse692 = cse36*cse611;
  const amrex::Real cse693 = cse38*cse614;
  const amrex::Real cse694 = cse692 + cse693;
  const amrex::Real cse695 = std::pow(cse617, -2);
  const amrex::Real cse696 = cse11*cse695;
  const amrex::Real cse697 = cse696*(-cse615 + cse694);
  const amrex::Real cse698 = 1.5159999999999999e-9*cse627;
  const amrex::Real cse699 = cse632*cse634*cse656;
  const amrex::Real cse700 = cse632*cse635;
  const amrex::Real cse701 = 5.7927337833400002e-20*cse700*state.xn[8]/std::pow(cse637, 2);
  const amrex::Real cse702 = cse502 + 1.9999999999999999e-7;
  const amrex::Real cse703 = cse618*cse702;
  const amrex::Real cse704 = cse11*cse703 + 1;
  const amrex::Real cse705 = std::pow(cse704, -2);
  const amrex::Real cse706 = cse35*cse703;
  const amrex::Real cse707 = cse38*cse645;
  const amrex::Real cse708 = cse35*state.xn[2];
  const amrex::Real cse709 = std::pow(cse641, -2);
  const amrex::Real cse710 = 0.5*cse709;
  const amrex::Real cse711 = cse125/std::pow(cse10, 3);
  const amrex::Real cse712 = -0.5*cse642*cse708 + cse710*cse711;
  const amrex::Real cse713 = cse641*cse644*cse66*state.xn[0];
  const amrex::Real cse714 = cse11*cse8;
  const amrex::Real cse715 = ((cse648) ? (
   0
)
: ((cse649) ? (
   (3.0/2.0)*cse714
)
: ((cse650) ? (
   (4.0/3.0)*cse714
)
: ((cse651) ? (
   (1.0/3.0)*cse714
)
: (
   0
)))));
  const amrex::Real cse716 = cse653*state.xn[3];
  const amrex::Real cse717 = cse36*cse652;
  const amrex::Real cse718 = 2.0*cse717;
  const amrex::Real cse719 = cse646 - cse707 + cse712*cse713 + cse715*cse716 - cse718;
  const amrex::Real cse720 = 1.0/cse64;
  const amrex::Real cse721 = cse35*ncol_H*(((cse670 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse670 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse671, 3.0/2.0);
  const amrex::Real cse722 = cse721*ncol_CO;
  const amrex::Real cse723 = ((cse669) ? (
   0
)
: (
   878350727.21550155*cse722
));
  const amrex::Real cse724 = alpha_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse725 = std::log(cse680);
  const amrex::Real cse726 = cse724*cse725;
  const amrex::Real cse727 = cse665/std::pow(cse676, 2);
  const amrex::Real cse728 = nhalf_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse729 = cse723*cse728;
  const amrex::Real cse730 = cse681/cse665;
  const amrex::Real cse731 = cse676*cse730;
  const amrex::Real cse732 = std::pow(cse674, -2);
  const amrex::Real cse733 = llte_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse734 = -cse665*cse723*cse732*cse733 + cse683*(cse662*cse676*cse723*cse732*cse733 - cse677*cse729);
  const amrex::Real cse735 = cse665*state.xn[9];
  const amrex::Real cse736 = cse661*cse735/std::pow(cse685, 2);
  const amrex::Real cse737 = state.T < 100.0;
  const amrex::Real cse738 = state.T < 6000.0;
  const amrex::Real cse739 = state.T < 10;
  const amrex::Real cse740 = std::log(cse607);
  const amrex::Real cse741 = cse740*cse8;
  const amrex::Real cse742 = std::pow(cse7, -8);
  const amrex::Real cse743 = std::pow(cse740, 8)*cse742;
  const amrex::Real cse744 = std::pow(cse7, -7);
  const amrex::Real cse745 = std::pow(cse740, 7)*cse744;
  const amrex::Real cse746 = std::pow(cse7, -6);
  const amrex::Real cse747 = std::pow(cse740, 6)*cse746;
  const amrex::Real cse748 = std::pow(cse7, -5);
  const amrex::Real cse749 = std::pow(cse740, 5)*cse748;
  const amrex::Real cse750 = std::pow(cse7, -4);
  const amrex::Real cse751 = std::pow(cse740, 4)*cse750;
  const amrex::Real cse752 = std::pow(cse7, -3);
  const amrex::Real cse753 = std::pow(cse740, 3);
  const amrex::Real cse754 = cse752*cse753;
  const amrex::Real cse755 = cse13*std::pow(cse740, 2);
  const amrex::Real cse756 = std::pow(10.0, 16.815729999999999*cse741 + 70.609154000000004*cse743 + 364.14445999999998*cse745 + 801.81246999999996*cse747 + 983.67575999999997*cse749 + 734.71650999999997*cse751 + 343.1918*cse754 + 96.743155000000002*cse755 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse757 = state.xn[5] + state.xn[6];
  const amrex::Real cse758 = std::pow(10.0, 1.5714710999999999*cse741 + 0.32168730000000001*cse749 - 0.51002221000000003*cse751 - 0.23619984999999999*cse754 + 0.015391166*cse755 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse741 + 0.19191374999999999*cse749 - 0.16596184*cse751 + 0.29036281000000003*cse754 - 0.81520437999999995*cse755 - 23.689236999999999)*cse757 + std::pow(10.0, 2.0943374000000001*cse740*cse8 - 0.033638326000000003*cse749 - 0.14913216000000001*cse751 + 0.43693353000000001*cse752*cse753 - 0.77151435999999995*cse755 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse759 = std::pow(10, 37.383713*cse741 + 3.8479610000000002*cse749 + 20.159831000000001*cse751 + 48.656103000000002*cse754 + 58.145166000000003*cse755 - 16.818342000000001)*state.xn[0] + cse758;
  const amrex::Real cse760 = std::pow(10.0, 1.6802758*cse741 - 6.3701156000000001*cse743 + 6.4380698000000001*cse745 + 8.9167182999999994*cse747 - 8.8077017000000009*cse749 - 4.7274035999999997*cse751 + 4.0406627000000004*cse754 + 0.93310621999999999*cse755 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse761 = state.T < 1000.0;
  const amrex::Real cse762 = std::pow(10.0, 3.5692468000000002*cse740*cse8 - 4.2519023000000002*cse749 - 21.328264000000001*cse751 - 27.850082*cse754 - 11.33286*cse755 - 24.311209000000002)*state.xn[0] + cse758;
  const amrex::Real cse763 = std::pow(10.0, 4.6450521*cse741 + 1.5538288*cse749 - 5.5108049000000001*cse751 + 5.9369081000000001*cse754 - 3.7209846*cse755 - 24.311209000000002)*state.xn[0] + cse758;
  const amrex::Real cse764 = std::pow(10.0, -11.945856385319289*cse13 + 28.69470131298657*cse748 - 56.79810800956944*cse750 + 34.150687525022562*cse752 + 8.3228160852326614*cse8 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.5440891133599637*cse748 - 1.7105157385968939*cse750 + 1.6702447513374001*cse752 + 3.9225864834863131*cse8 - 23.689236999999999)*cse757 + std::pow(10.0, -2.4768712409536775*cse13 - 0.62120210234156448*cse748 - 1.5370575959566981*cse750 + 2.5133588394664677*cse752 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 5.9406293599919788*cse748 - 5.2566362076906961*cse750 - 1.3586848802337363*cse752 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse765 = ((cse629 && cse737 && cse738) ? (
   cse739 || cse756 + cse759 < 1.0e-100
)
: (
   ((cse737 && cse738) ? (
      cse739 || cse759 + cse760 < 1.0e-100
   )
   : (
      ((cse629 && cse738 && cse761) ? (
         cse739 || cse756 + cse762 < 1.0e-100
      )
      : (
         ((cse738 && cse761) ? (
            cse739 || cse760 + cse762 < 1.0e-100
         )
         : (
            ((cse629 && cse738) ? (
               cse739 || cse756 + cse763 < 1.0e-100
            )
            : (
               ((cse738) ? (
                  cse739 || cse760 + cse763 < 1.0e-100
               )
               : (
                  ((cse629) ? (
                     cse739 || std::pow(10.0, 310.58441786958315*cse13 + 7500.646099484944*cse742 + 21588.957206746247*cse744 + 26530.843406310876*cse746 + 18165.631967965237*cse748 + 7572.4886742758581*cse750 + 1974.131269262874*cse752 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse764 < 1.0e-100
                  )
                  : (
                     cse739 || std::pow(10.0, 2.9956460707652877*cse13 - 676.68255490510751*cse742 + 381.69251128040054*cse744 + 295.04162851880631*cse746 - 162.65264843551881*cse748 - 48.723840736518795*cse750 + 23.242975457496804*cse752 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse764 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse766 = std::log(0.001*((cse738) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse767 = cse766*cse8;
  const amrex::Real cse768 = std::pow(cse766, 5);
  const amrex::Real cse769 = cse748*cse768;
  const amrex::Real cse770 = std::pow(cse766, 4);
  const amrex::Real cse771 = cse750*cse770;
  const amrex::Real cse772 = std::pow(cse766, 3);
  const amrex::Real cse773 = cse752*cse772;
  const amrex::Real cse774 = std::pow(cse766, 2);
  const amrex::Real cse775 = cse13*cse774;
  const amrex::Real cse776 = std::pow(10, 37.383713*cse767 + 3.8479610000000002*cse769 + 20.159831000000001*cse771 + 48.656103000000002*cse773 + 58.145166000000003*cse775 - 16.818342000000001);
  const amrex::Real cse777 = ((cse738) ? (
   cse737
)
: (
   false
));
  const amrex::Real cse778 = std::pow(10.0, 3.5692468000000002*cse766*cse8 - 4.2519023000000002*cse769 - 21.328264000000001*cse771 - 27.850082*cse773 - 11.33286*cse775 - 24.311209000000002);
  const amrex::Real cse779 = ((cse738) ? (
   cse761
)
: (
   false
));
  const amrex::Real cse780 = std::pow(10.0, 4.6450521*cse767 + 1.5538288*cse769 - 5.5108049000000001*cse771 + 5.9369081000000001*cse773 - 3.7209846*cse775 - 24.311209000000002);
  const amrex::Real cse781 = ((cse777) ? (
   cse776
)
: ((cse779) ? (
   cse778
)
: (
   cse780
)));
  const amrex::Real cse782 = std::pow(10.0, 0.43693353000000001*cse752*cse772 + 2.0943374000000001*cse766*cse8 - 0.033638326000000003*cse769 - 0.14913216000000001*cse771 - 0.77151435999999995*cse775 - 23.962112000000001);
  const amrex::Real cse783 = cse782*state.xn[3];
  const amrex::Real cse784 = std::pow(10.0, 1.5714710999999999*cse767 + 0.32168730000000001*cse769 - 0.51002221000000003*cse771 - 0.23619984999999999*cse773 + 0.015391166*cse775 - 22.089523);
  const amrex::Real cse785 = cse784*state.xn[1];
  const amrex::Real cse786 = std::pow(10.0, 2.1892372*cse767 + 0.19191374999999999*cse769 - 0.16596184*cse771 + 0.29036281000000003*cse773 - 0.81520437999999995*cse775 - 23.689236999999999);
  const amrex::Real cse787 = cse757*cse786;
  const amrex::Real cse788 = cse742*std::pow(cse766, 8);
  const amrex::Real cse789 = std::pow(cse766, 7);
  const amrex::Real cse790 = cse744*cse789;
  const amrex::Real cse791 = std::pow(cse766, 6);
  const amrex::Real cse792 = cse746*cse791;
  const amrex::Real cse793 = std::pow(10.0, 16.815729999999999*cse767 + 983.67575999999997*cse769 + 734.71650999999997*cse771 + 343.1918*cse773 + 96.743155000000002*cse775 + 70.609154000000004*cse788 + 364.14445999999998*cse790 + 801.81246999999996*cse792 - 21.928795999999998);
  const amrex::Real cse794 = std::pow(10.0, 1.6802758*cse767 - 8.8077017000000009*cse769 - 4.7274035999999997*cse771 + 4.0406627000000004*cse773 + 0.93310621999999999*cse775 - 6.3701156000000001*cse788 + 6.4380698000000001*cse790 + 8.9167182999999994*cse792 - 22.921188999999998);
  const amrex::Real cse795 = ((cse629) ? (
   cse793
)
: (
   cse794
));
  const amrex::Real cse796 = cse781*state.xn[0] + cse783 + cse785 + cse787 + cse795*state.xn[2];
  const amrex::Real cse797 = std::pow(cse796, -2);
  const amrex::Real cse798 = ((cse738) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse799 = std::exp(-11700.0*cse798);
  const amrex::Real cse800 = std::exp(-5860.0*cse798);
  const amrex::Real cse801 = std::exp(-510.0*cse798);
  const amrex::Real cse802 = std::exp(-2197000.0*((cse738) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse803 = cse802*((cse738) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse804 = 6.0142468035272636e-8*((cse738) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse805 = 1.0/cse804;
  const amrex::Real cse806 = 1.6e-18*cse799 + 6.7e-19*cse800 + 3.0e-24*cse801 + 4.985670872372847e-33*cse803*cse805;
  const amrex::Real cse807 = 1.0/cse796;
  const amrex::Real cse808 = cse806*cse807 + 1;
  const amrex::Real cse809 = std::pow(cse808, -2);
  const amrex::Real cse810 = cse797*std::pow(cse806, 2)*cse809;
  const amrex::Real cse811 = 0.0060000000000000001*cse625;
  const amrex::Real cse812 = std::pow(cse624, 0.36899999999999999 - cse811);
  const amrex::Real cse813 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse814 = 1.08e-14*cse813;
  const amrex::Real cse815 = 0.16*cse625;
  const amrex::Real cse816 = std::pow(cse624, 0.755 - cse815);
  const amrex::Real cse817 = 4.34e-10*cse816;
  const amrex::Real cse818 = cse817*state.xn[0];
  const amrex::Real cse819 = 0.042999999999999997*cse625;
  const amrex::Real cse820 = 5.2499999999999998e-13*std::pow(cse624, cse819 + 0.88900000000000001);
  const amrex::Real cse821 = 0.52500000000000002*cse625;
  const amrex::Real cse822 = 2.2500000000000003e-12*std::pow(cse624, cse821 + 1.198);
  const amrex::Real cse823 = cse820 + cse822;
  const amrex::Real cse824 = cse814*state.xn[2] + cse818 + cse823*state.xn[3];
  const amrex::Real cse825 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse826 = cse824*cse825;
  const amrex::Real cse827 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse828 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse829 = 4.8599999999999998e-10*cse828;
  const amrex::Real cse830 = 3.1899999999999998e-10*cse812;
  const amrex::Real cse831 = cse830*state.xn[0];
  const amrex::Real cse832 = 0.041000000000000002*cse625;
  const amrex::Real cse833 = 4.7499999999999998e-11*std::pow(cse624, cse832 + 0.20300000000000001);
  const amrex::Real cse834 = 0.058000000000000003*cse625;
  const amrex::Real cse835 = 1.6725000000000002e-10*std::pow(cse624, cse834 + 0.23699999999999999);
  const amrex::Real cse836 = cse833 + cse835;
  const amrex::Real cse837 = cse829*state.xn[2] + cse831 + cse836*state.xn[3];
  const amrex::Real cse838 = cse827*cse837;
  const amrex::Real cse839 = 0.0030000000000000001*cse625;
  const amrex::Real cse840 = std::pow(cse624, 0.41899999999999998 - cse839);
  const amrex::Real cse841 = 7.1400000000000007e-11*cse840;
  const amrex::Real cse842 = cse824*cse827;
  const amrex::Real cse843 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse844 = 5.1199999999999999e-10*cse843;
  const amrex::Real cse845 = 3.5700000000000001e-10*cse840;
  const amrex::Real cse846 = cse845*state.xn[0];
  const amrex::Real cse847 = 0.025000000000000001*cse625;
  const amrex::Real cse848 = 3.7250000000000001e-11*std::pow(cse624, cse847 + 0.26400000000000001);
  const amrex::Real cse849 = 1.0274999999999999e-10*std::pow(cse624, cse819 + 0.29599999999999999);
  const amrex::Real cse850 = cse848 + cse849;
  const amrex::Real cse851 = cse844*state.xn[2] + cse846 + cse850*state.xn[3];
  const amrex::Real cse852 = cse827*cse851;
  const amrex::Real cse853 = 8.68e-11*cse816;
  const amrex::Real cse854 = cse825*cse837;
  const amrex::Real cse855 = cse851 + 8.9099999999999997e-5;
  const amrex::Real cse856 = cse827*cse855;
  const amrex::Real cse857 = 6.3800000000000002e-11*cse812;
  const amrex::Real cse858 = 2.1266666666666666e-11*cse812*cse826 + 2.8933333333333332e-11*cse816*cse854 + cse838*cse841 + cse841*cse842 + cse852*cse853 + cse856*cse857;
  const amrex::Real cse859 = cse824 + 1.7499999999999998e-5;
  const amrex::Real cse860 = cse837 + 1.34e-10;
  const amrex::Real cse861 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse862 = cse824*cse861;
  const amrex::Real cse863 = (1.0/3.0)*cse862;
  const amrex::Real cse864 = (1.0/5.0)*cse827;
  const amrex::Real cse865 = cse824*cse864;
  const amrex::Real cse866 = (1.0/15.0)*cse826;
  const amrex::Real cse867 = cse837*cse864;
  const amrex::Real cse868 = cse837*cse866 + cse851*cse865 + cse855*cse867;
  const amrex::Real cse869 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse870 = cse851*cse869;
  const amrex::Real cse871 = (3.0/5.0)*cse870;
  const amrex::Real cse872 = cse859*cse867 + cse859*cse871 + cse860*cse871;
  const amrex::Real cse873 = cse855*cse859 + cse855*cse860 + cse860*cse863 + cse868 + cse872;
  const amrex::Real cse874 = 1.0/cse873;
  const amrex::Real cse875 = 2.3872025442750958e-19*cse874;
  const amrex::Real cse876 = cse827*cse859;
  const amrex::Real cse877 = 2.1419999999999999e-10*cse840;
  const amrex::Real cse878 = cse860*cse869;
  const amrex::Real cse879 = cse859*cse869;
  const amrex::Real cse880 = 1.9139999999999998e-10*cse812*cse870 + 2.604e-10*cse816*cse870 + cse838*cse853 + cse857*cse876 + cse877*cse878 + cse877*cse879;
  const amrex::Real cse881 = 2.8023005140020001e-18*cse874;
  const amrex::Real cse882 = std::pow(cse873, -2);
  const amrex::Real cse883 = cse860*cse861;
  const amrex::Real cse884 = cse882*(-1.0633333333333333e-10*cse812*cse862 - 1.4466666666666666e-10*cse816*cse883 - cse817*cse855 - cse830*cse855 - cse845*cse859 - cse845*cse860 - cse858 - cse880);
  const amrex::Real cse885 = 2.3872025442750958e-19*cse868;
  const amrex::Real cse886 = 2.8023005140020001e-18*cse872;
  const amrex::Real cse887 = 0.045999999999999999*cse625;
  const amrex::Real cse888 = std::pow(cse624, cse887 + 0.22800000000000001);
  const amrex::Real cse889 = std::pow(cse624, cse887 + 0.23100000000000001);
  const amrex::Real cse890 = 2.6400000000000002e-10*cse889;
  const amrex::Real cse891 = cse890*state.xn[0];
  const amrex::Real cse892 = 0.031*cse625;
  const amrex::Real cse893 = 1.11e-10*std::pow(cse624, cse892 + 0.26300000000000001);
  const amrex::Real cse894 = 0.064000000000000001*cse625;
  const amrex::Real cse895 = 4.3749999999999998e-11*std::pow(cse624, cse894 + 0.071999999999999995);
  const amrex::Real cse896 = cse893 + cse895;
  const amrex::Real cse897 = 0.00235272*cse72;
  const amrex::Real cse898 = cse897 - 0.041816600000000002;
  const amrex::Real cse899 = cse72*cse898 + 0.35826400000000003;
  const amrex::Real cse900 = cse72*cse899 - 0.57443;
  const amrex::Real cse901 = 0.097857299999999994*cse72;
  const amrex::Real cse902 = cse901 - 3.1926800000000002;
  const amrex::Real cse903 = cse72*cse902 + 38.504899999999999;
  const amrex::Real cse904 = cse72*cse903 - 202.19300000000001;
  const amrex::Real cse905 = std::exp(((cse761) ? (
   cse72*cse900 - 7.4386999999999999
)
: (
   cse72*cse904 + 386.18599999999998
)));
  const amrex::Real cse906 = 1.7258000000000001e-6*cse623;
  const amrex::Real cse907 = cse905*cse906;
  const amrex::Real cse908 = cse891 + cse896*state.xn[3] + cse907;
  const amrex::Real cse909 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse910 = cse908*cse909;
  const amrex::Real cse911 = 8.9000000000000003e-11*cse888;
  const amrex::Real cse912 = cse911*state.xn[0];
  const amrex::Real cse913 = 0.037999999999999999*cse625;
  const amrex::Real cse914 = 5.1749999999999996e-11*std::pow(cse624, cse913 + 0.16900000000000001);
  const amrex::Real cse915 = 0.048000000000000001*cse625;
  const amrex::Real cse916 = 2.15e-11*std::pow(cse624, cse915 - 0.01);
  const amrex::Real cse917 = cse914 + cse916;
  const amrex::Real cse918 = 0.0070527699999999999*cse72;
  const amrex::Real cse919 = cse918 - 0.11133800000000001;
  const amrex::Real cse920 = cse72*cse919 + 0.69763799999999998;
  const amrex::Real cse921 = cse72*cse920 - 1.3074300000000001;
  const amrex::Real cse922 = 0.093813800000000003*cse72;
  const amrex::Real cse923 = cse922 - 3.0328300000000001;
  const amrex::Real cse924 = cse72*cse923 + 36.180300000000003;
  const amrex::Real cse925 = cse72*cse924 - 187.47399999999999;
  const amrex::Real cse926 = std::exp(((cse761) ? (
   cse72*cse921 - 7.6973500000000001
)
: (
   cse72*cse925 + 350.60899999999998
)));
  const amrex::Real cse927 = cse906*cse926;
  const amrex::Real cse928 = cse912 + cse917*state.xn[3] + cse927;
  const amrex::Real cse929 = cse909*cse928;
  const amrex::Real cse930 = 0.057000000000000002*cse625;
  const amrex::Real cse931 = std::pow(cse624, cse930 + 0.115);
  const amrex::Real cse932 = 1.26e-10*cse931;
  const amrex::Real cse933 = cse932*state.xn[0];
  const amrex::Real cse934 = 0.049000000000000002*cse625;
  const amrex::Real cse935 = 5.3249999999999997e-11*std::pow(cse624, cse934 - 0.0040000000000000001);
  const amrex::Real cse936 = 0.10199999999999999*cse625;
  const amrex::Real cse937 = 1.675e-11*std::pow(cse624, cse936 - 0.085000000000000006);
  const amrex::Real cse938 = cse935 + cse937;
  const amrex::Real cse939 = 0.00065632500000000003*cse72;
  const amrex::Real cse940 = -cse939 - 0.015089200000000001;
  const amrex::Real cse941 = cse72*cse940 + 0.36118400000000001;
  const amrex::Real cse942 = cse72*cse941 - 0.77378199999999997;
  const amrex::Real cse943 = 0.10508000000000001*cse72;
  const amrex::Real cse944 = cse943 - 3.4762;
  const amrex::Real cse945 = cse72*cse944 + 42.594999999999999;
  const amrex::Real cse946 = cse72*cse945 - 227.91300000000001;
  const amrex::Real cse947 = std::exp(((cse761) ? (
   cse72*cse942 - 9.2514099999999999
)
: (
   cse72*cse946 + 444.60000000000002
)));
  const amrex::Real cse948 = 2.8763333333333333e-6*cse947;
  const amrex::Real cse949 = cse623*cse948;
  const amrex::Real cse950 = cse933 + cse938*state.xn[3] + cse949;
  const amrex::Real cse951 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse952 = cse950*cse951;
  const amrex::Real cse953 = 1.32e-9*cse889;
  const amrex::Real cse954 = 6.3e-10*cse931;
  const amrex::Real cse955 = cse928*cse951;
  const amrex::Real cse956 = cse908*cse951;
  const amrex::Real cse957 = cse950 + 7.8800000000000004e-8;
  const amrex::Real cse958 = cse951*cse957;
  const amrex::Real cse959 = 4.4500000000000001e-10*cse888;
  const amrex::Real cse960 = 7.4166666666666679e-10*cse888*cse910 + 2.2000000000000003e-9*cse889*cse929 + cse952*cse953 + cse954*cse955 + cse954*cse956 + cse958*cse959;
  const amrex::Real cse961 = cse908 + 2.65e-7;
  const amrex::Real cse962 = cse928 + 1.81e-14;
  const amrex::Real cse963 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse964 = cse908*cse963;
  const amrex::Real cse965 = (5.0/3.0)*cse962;
  const amrex::Real cse966 = (25.0/3.0)*cse929;
  const amrex::Real cse967 = 5*cse951;
  const amrex::Real cse968 = cse908*cse967;
  const amrex::Real cse969 = cse928*cse967;
  const amrex::Real cse970 = cse908*cse966 + cse950*cse968 + cse957*cse969;
  const amrex::Real cse971 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse972 = cse950*cse971;
  const amrex::Real cse973 = 3*cse972;
  const amrex::Real cse974 = cse961*cse969 + cse961*cse973 + cse962*cse973;
  const amrex::Real cse975 = cse957*cse961 + cse957*cse962 + cse964*cse965 + cse970 + cse974;
  const amrex::Real cse976 = 1.0/cse975;
  const amrex::Real cse977 = 1.4210469458259591e-21*cse976;
  const amrex::Real cse978 = cse951*cse961;
  const amrex::Real cse979 = 3.7800000000000004e-10*cse931;
  const amrex::Real cse980 = cse962*cse971;
  const amrex::Real cse981 = cse961*cse971;
  const amrex::Real cse982 = 2.6700000000000001e-10*cse888*cse972 + 7.9200000000000006e-10*cse889*cse972 + cse953*cse955 + cse959*cse978 + cse979*cse980 + cse979*cse981;
  const amrex::Real cse983 = 2.5697412351439998e-22*cse976;
  const amrex::Real cse984 = std::pow(cse975, -2);
  const amrex::Real cse985 = cse962*cse963;
  const amrex::Real cse986 = cse984*(-1.4833333333333334e-10*cse888*cse964 - 4.4000000000000003e-10*cse889*cse985 - cse890*cse957 - cse911*cse957 - cse932*cse961 - cse932*cse962 - cse960 - cse982);
  const amrex::Real cse987 = 1.4210469458259591e-21*cse970;
  const amrex::Real cse988 = 2.5697412351439998e-22*cse974;
  const amrex::Real cse989 = state.T < 1000000.0;
  const amrex::Real cse990 = cse567*cse69*d2g;
  const amrex::Real cse991 = crate*cse655;
  const amrex::Real cse992 = 1.7623947000000002e-12*cse689*cse991*state.xn[5];
  const amrex::Real cse993 = cse694*cse696;
  const amrex::Real cse994 = 2.0187430200000004e-26*cse165*cse690;
  const amrex::Real cse995 = cse10*cse994;
  const amrex::Real cse996 = 1.461185424e-21*cse187*cse705*state.xn[3];
  const amrex::Real cse997 = 2.0*cse11*cse715*state.xn[3] + 26.399999999999999*cse641*cse643*cse66*cse712*state.xn[0] - cse707 - cse718;
  const amrex::Real cse998 = 1.602177e-12*cse89;
  const amrex::Real cse999 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1000 = 1.7623947000000002e-12*cse720;
  const amrex::Real cse1001 = crate*cse1000*state.xn[5];
  const amrex::Real cse1002 = cse679*cse681;
  const amrex::Real cse1003 = cse1001*cse997 - 7.1777529600000012e-12*cse120 + 3.0000000000000002e-15*cse165*cse622 - 7.1777529600000012e-12*cse497 + 1.7e-24*cse563*cse990 - cse580 - cse603 - cse606 - cse736*(-cse684*(-cse1002*cse729 + cse723*cse724*cse725) - cse734) + cse86*cse992 + cse995*(cse35*cse618*cse619 - cse619*cse993) + cse996*(cse35*cse618*cse702 - cse702*cse993) + cse997*cse998 + cse997*cse999;
  const amrex::Real cse1004 = cse581*cse583;
  const amrex::Real cse1005 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1006 = std::pow(cse582, -2);
  const amrex::Real cse1007 = std::pow(cse586, -2);
  const amrex::Real cse1008 = cse110*cse578;
  const amrex::Real cse1009 = cse575*cse577;
  const amrex::Real cse1010 = 1.0/cse572;
  const amrex::Real cse1011 = std::pow(cse568, -2);
  const amrex::Real cse1012 = cse1010*cse1011;
  const amrex::Real cse1013 = cse72/std::pow(cse573, 2);
  const amrex::Real cse1014 = std::pow(cse566, -2);
  const amrex::Real cse1015 = std::pow(cse564, 0.12299999999999994)*cse69;
  const amrex::Real cse1016 = std::pow(cse564, -0.56899999999999995)*cse565*cse69;
  const amrex::Real cse1017 = 1.3659996292638161e-10*cse1004*cse296;
  const amrex::Real cse1018 = cse1017*cse589/std::pow(cse591, 2);
  const amrex::Real cse1019 = 0.5*cse11*cse642 - cse708*cse710;
  const amrex::Real cse1020 = cse1008*state.xn[2];
  const amrex::Real cse1021 = 9.0599999999999997e-6*cse264;
  const amrex::Real cse1022 = 1.024e-10*cse843;
  const amrex::Real cse1023 = 2.1600000000000004e-15*cse813;
  const amrex::Real cse1024 = 9.7199999999999998e-11*cse828;
  const amrex::Real cse1025 = cse1022*cse838 + cse1022*cse842 + cse1023*cse852 + cse1024*cse856 + 7.1999999999999997e-16*cse813*cse854 + 3.2399999999999999e-11*cse826*cse828;
  const amrex::Real cse1026 = 3.0719999999999998e-10*cse843;
  const amrex::Real cse1027 = cse1023*cse838 + cse1024*cse876 + cse1026*cse878 + cse1026*cse879 + 6.4799999999999996e-15*cse813*cse870 + 2.9159999999999995e-10*cse828*cse870;
  const amrex::Real cse1028 = cse882*(-cse1025 - cse1027 - 3.6000000000000001e-15*cse813*cse883 - cse814*cse855 - 1.6199999999999998e-10*cse828*cse862 - cse829*cse855 - cse844*cse859 - cse844*cse860);
  const amrex::Real cse1029 = cse264*cse905;
  const amrex::Real cse1030 = cse264*cse947;
  const amrex::Real cse1031 = 1.4381666666666666e-5*cse1030;
  const amrex::Real cse1032 = cse264*cse926;
  const amrex::Real cse1033 = 8.6289999999999999e-6*cse1029;
  const amrex::Real cse1034 = 8.6289999999999999e-6*cse1032;
  const amrex::Real cse1035 = 1.4381666666666668e-5*cse1029*cse929 + cse1031*cse955 + cse1031*cse956 + 1.4381666666666668e-5*cse1032*cse910 + cse1033*cse952 + cse1034*cse958;
  const amrex::Real cse1036 = 5.1774000000000006e-6*cse972;
  const amrex::Real cse1037 = 8.6289999999999999e-6*cse1030;
  const amrex::Real cse1038 = cse1029*cse1036 + cse1032*cse1036 + cse1033*cse955 + cse1034*cse978 + cse1037*cse980 + cse1037*cse981;
  const amrex::Real cse1039 = cse264*cse948;
  const amrex::Real cse1040 = 1.7258000000000001e-6*cse957;
  const amrex::Real cse1041 = cse984*(-cse1029*cse1040 - 2.8763333333333337e-6*cse1029*cse985 - cse1032*cse1040 - 2.8763333333333337e-6*cse1032*cse964 - cse1035 - cse1038 - cse1039*cse961 - cse1039*cse962);
  const amrex::Real cse1042 = cse1009*cse579;
  const amrex::Real cse1043 = 3.7996253111508318e-29*cse1042;
  const amrex::Real cse1044 = cse2*cse602;
  const amrex::Real cse1045 = 6.4000000000000004e-32*cse1044;
  const amrex::Real cse1046 = cse605*d2g;
  const amrex::Real cse1047 = 200.0*cse1046;
  const amrex::Real cse1048 = 5.7927337833400002e-20*cse638;
  const amrex::Real cse1049 = cse1048*cse639;
  const amrex::Real cse1050 = -cse631;
  const amrex::Real cse1051 = cse1048*cse699*state.xn[8];
  const amrex::Real cse1052 = cse171*cse611 + 2*cse693;
  const amrex::Real cse1053 = cse696*(cse1052 - cse612);
  const amrex::Real cse1054 = 2*cse699;
  const amrex::Real cse1055 = 2*cse707;
  const amrex::Real cse1056 = cse642*cse708;
  const amrex::Real cse1057 = cse709*cse711;
  const amrex::Real cse1058 = -1.0*cse1056 + 1.0*cse1057;
  const amrex::Real cse1059 = ((cse648) ? (
   0
)
: ((cse649) ? (
   3*cse714
)
: ((cse650) ? (
   (8.0/3.0)*cse714
)
: ((cse651) ? (
   (2.0/3.0)*cse714
)
: (
   0
)))));
  const amrex::Real cse1060 = 4.0*cse717;
  const amrex::Real cse1061 = -cse1055 + cse1058*cse713 + cse1059*cse716 - cse1060 + cse654;
  const amrex::Real cse1062 = ((cse669) ? (
   0
)
: (
   cse672*cse721
));
  const amrex::Real cse1063 = cse1062*cse728;
  const amrex::Real cse1064 = -cse1062*cse665*cse732*cse733 + cse683*(cse1062*cse662*cse676*cse732*cse733 - cse1063*cse677);
  const amrex::Real cse1065 = 1.0/cse808;
  const amrex::Real cse1066 = cse851*cse864;
  const amrex::Real cse1067 = (1.0/15.0)*cse854;
  const amrex::Real cse1068 = cse836*cse855;
  const amrex::Real cse1069 = cse1066*cse823 + cse1067*cse823 + cse1068*cse864 + cse836*cse866 + cse850*cse865 + cse850*cse867;
  const amrex::Real cse1070 = cse850*cse860;
  const amrex::Real cse1071 = (3.0/5.0)*cse869;
  const amrex::Real cse1072 = cse850*cse859;
  const amrex::Real cse1073 = cse859*cse864;
  const amrex::Real cse1074 = cse1070*cse1071 + cse1071*cse1072 + cse1073*cse836 + cse823*cse867 + cse823*cse871 + cse836*cse871;
  const amrex::Real cse1075 = (1.0/3.0)*cse883;
  const amrex::Real cse1076 = cse882*(-cse1068 - cse1069 - cse1070 - cse1072 - cse1074 - cse1075*cse823 - cse823*cse855 - cse836*cse863);
  const amrex::Real cse1077 = cse950*cse967;
  const amrex::Real cse1078 = (25.0/3.0)*cse910;
  const amrex::Real cse1079 = cse917*cse957;
  const amrex::Real cse1080 = cse1077*cse896 + cse1078*cse917 + cse1079*cse967 + cse896*cse966 + cse938*cse968 + cse938*cse969;
  const amrex::Real cse1081 = cse961*cse967;
  const amrex::Real cse1082 = cse938*cse962;
  const amrex::Real cse1083 = 3*cse971;
  const amrex::Real cse1084 = cse938*cse961;
  const amrex::Real cse1085 = cse1081*cse917 + cse1082*cse1083 + cse1083*cse1084 + cse896*cse969 + cse896*cse973 + cse917*cse973;
  const amrex::Real cse1086 = cse963*cse965;
  const amrex::Real cse1087 = (5.0/3.0)*cse964;
  const amrex::Real cse1088 = cse984*(-cse1079 - cse1080 - cse1082 - cse1084 - cse1085 - cse1086*cse896 - cse1087*cse917 - cse896*cse957);
  const amrex::Real cse1089 = cse563*cse990;
  const amrex::Real cse1090 = cse1052*cse696;
  const amrex::Real cse1091 = -cse1055 + 26.399999999999999*cse1058*cse641*cse643*cse66*state.xn[0] + 2.0*cse1059*cse11*state.xn[3] - cse1060;
  const amrex::Real cse1092 = -state.xn[3]*((cse765) ? (
   0
)
: (
   cse786*cse810
));
  const amrex::Real cse1093 = ((cse669) ? (
   cse668
)
: (
   0
));
  const amrex::Real cse1094 = cse1093*cse728;
  const amrex::Real cse1095 = cse696*(3*cse692 + 3*cse693);
  const amrex::Real cse1096 = 2.0*cse11*state.xn[3]*((cse648) ? (
   0
)
: ((cse649) ? (
   (9.0/2.0)*cse714
)
: ((cse650) ? (
   4*cse714
)
: ((cse651) ? (
   cse714
)
: (
   0
))))) + 26.399999999999999*cse641*cse643*cse66*state.xn[0]*(-1.5*cse1056 + 1.5*cse1057) - 3*cse707 - 6.0*cse717;
  const amrex::Real cse1097 = ((cse669) ? (
   0
)
: (
   2635052181.6465044*cse722
));
  const amrex::Real cse1098 = cse1097*cse728;
  const amrex::Real cse1099 = cse589*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1100 = cse296*cse583;
  const amrex::Real cse1101 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1102 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1103 = cse1101*cse584;
  const amrex::Real cse1104 = cse1007*cse588*(2.5500935934394033e-12*cse1006*cse1102*state.xn[2] - 6.6690584756579352e-10*cse1103);
  const amrex::Real cse1105 = cse1010*cse277*cse569*cse69;
  const amrex::Real cse1106 = cse260*cse700;
  const amrex::Real cse1107 = cse3*cse623;
  const amrex::Real cse1108 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1109 = 100*cse3;
  const amrex::Real cse1110 = -2.2649999999999999e-6*cse1107 + cse628*(cse1109*(8.7000000000000001e-5*cse625 + 0.001281) + cse3*cse626) + state.xn[3]*(0.75*((cse629) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1108
)) + 0.25*((cse629) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1108
)));
  const amrex::Real cse1111 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1112 = std::pow(cse608, -3);
  const amrex::Real cse1113 = -cse613*cse7*(0.0040460000000000001*cse1112 + 0.0036709999999999998*cse609) - cse616*cse7*(0.0047800000000000004*cse1112 - 5.5500000000000001e-5*cse609);
  const amrex::Real cse1114 = -cse1110;
  const amrex::Real cse1115 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1116 = std::pow(cse662, -2);
  const amrex::Real cse1117 = ((cse669) ? (
   -2.872276598720128e-13*cse277*cse666*state.xn[9]*(((5.7445531974402559e-18*cse2*cse666 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse666 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse667, 2)
)
: (
   0
));
  const amrex::Real cse1118 = -cse1117*cse733 - llte_co_interp2d_partial_0(state.T, cse673);
  const amrex::Real cse1119 = cse1117*cse728 + nhalf_co_interp2d_partial_0(state.T, cse673);
  const amrex::Real cse1120 = ((cse738) ? (
   -cse260
)
: (
   0
));
  const amrex::Real cse1121 = -1.8719999999999998e-14*cse1120*cse799 - 3.9261999999999998e-15*cse1120*cse800 - 1.53e-21*cse1120*cse801 + 4.985670872372847e-33*cse802*cse805*((cse738) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse803*cse805*((cse738) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse803*((cse738) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse804, 2);
  const amrex::Real cse1122 = ((cse738) ? (
   1
)
: (
   0
));
  const amrex::Real cse1123 = cse1122*cse798;
  const amrex::Real cse1124 = cse1123*cse13*cse766;
  const amrex::Real cse1125 = cse1123*cse748*cse770;
  const amrex::Real cse1126 = cse1123*cse750*cse772;
  const amrex::Real cse1127 = cse1123*cse8;
  const amrex::Real cse1128 = cse1123*cse752*cse774;
  const amrex::Real cse1129 = cse1123*cse742*cse789;
  const amrex::Real cse1130 = cse1123*cse744*cse791;
  const amrex::Real cse1131 = cse1123*cse746*cse768;
  const amrex::Real cse1132 = 65.320000000000007*cse260;
  const amrex::Real cse1133 = cse3*cse819;
  const amrex::Real cse1134 = 0.00042999999999999999*cse625;
  const amrex::Real cse1135 = cse846*(cse1109*(0.0041900000000000001 - 3.0000000000000001e-5*cse625) - cse3*cse839) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse848*(cse1109*(0.00025000000000000001*cse625 + 0.00264) + cse3*cse847) + cse849*(cse1109*(cse1134 + 0.00296) + cse1133));
  const amrex::Real cse1136 = cse831*(cse1109*(0.0036900000000000001 - 6.0000000000000002e-5*cse625) - cse3*cse811) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse833*(cse1109*(0.00041000000000000005*cse625 + 0.0020300000000000001) + cse3*cse832) + cse835*(cse1109*(0.00058*cse625 + 0.0023700000000000001) + cse3*cse834));
  const amrex::Real cse1137 = cse818*(cse1109*(0.0075500000000000003 - 0.0016000000000000001*cse625) - cse3*cse815) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse820*(cse1109*(cse1134 + 0.0088900000000000003) + cse1133) + cse822*(cse1109*(0.0052500000000000003*cse625 + 0.011979999999999999) + cse3*cse821));
  const amrex::Real cse1138 = cse1136*cse855;
  const amrex::Real cse1139 = cse1066*cse1137 + cse1067*cse1137 + cse1132*cse824*cse852 + cse1132*cse837*cse856 + cse1135*cse865 + cse1135*cse867 + cse1136*cse866 + cse1138*cse864 + 28.360000000000003*cse260*cse826*cse837;
  const amrex::Real cse1140 = 136.68000000000001*cse260*cse870;
  const amrex::Real cse1141 = cse1135*cse860;
  const amrex::Real cse1142 = cse1135*cse859;
  const amrex::Real cse1143 = cse1071*cse1141 + cse1071*cse1142 + cse1073*cse1136 + cse1132*cse838*cse859 + cse1136*cse871 + cse1137*cse867 + cse1137*cse871 + cse1140*cse859 + cse1140*cse860;
  const amrex::Real cse1144 = cse882*(-cse1075*cse1137 - cse1136*cse863 - cse1137*cse855 - cse1138 - cse1139 - cse1141 - cse1142 - cse1143 - 32.933333333333337*cse260*cse860*cse862);
  const amrex::Real cse1145 = 312.30000000000001*cse260;
  const amrex::Real cse1146 = cse3*cse887;
  const amrex::Real cse1147 = 0.00046000000000000001*cse625;
  const amrex::Real cse1148 = 8.6290000000000003e-7*cse1107;
  const amrex::Real cse1149 = -cse1148*cse905 + cse891*(cse1109*(cse1147 + 0.00231) + cse1146) + cse907*((cse761) ? (
   cse3*cse900 + cse72*(cse3*cse899 + cse72*(cse3*cse897 + cse3*cse898))
)
: (
   cse3*cse904 + cse72*(cse3*cse903 + cse72*(cse3*cse901 + cse3*cse902))
)) + state.xn[3]*(cse893*(cse1109*(0.00031*cse625 + 0.00263) + cse3*cse892) + cse895*(cse1109*(0.00064000000000000005*cse625 + 0.00071999999999999994) + cse3*cse894));
  const amrex::Real cse1150 = -1.4381666666666666e-6*cse1107*cse947 + cse933*(cse1109*(0.00057000000000000009*cse625 + 0.00115) + cse3*cse930) + cse949*((cse761) ? (
   cse3*cse942 + cse72*(cse3*cse941 + cse72*(-cse3*cse939 + cse3*cse940))
)
: (
   cse3*cse946 + cse72*(cse3*cse945 + cse72*(cse3*cse943 + cse3*cse944))
)) + state.xn[3]*(cse935*(cse1109*(0.00048999999999999998*cse625 - 4.0000000000000003e-5) + cse3*cse934) + cse937*(cse1109*(0.0010199999999999999*cse625 - 0.00085000000000000006) + cse3*cse936));
  const amrex::Real cse1151 = -cse1148*cse926 + cse912*(cse1109*(cse1147 + 0.0022800000000000003) + cse1146) + cse927*((cse761) ? (
   cse3*cse921 + cse72*(cse3*cse920 + cse72*(cse3*cse918 + cse3*cse919))
)
: (
   cse3*cse925 + cse72*(cse3*cse924 + cse72*(cse3*cse922 + cse3*cse923))
)) + state.xn[3]*(cse914*(cse1109*(0.00038000000000000002*cse625 + 0.0016900000000000001) + cse3*cse913) + cse916*(cse1109*(0.00048000000000000001*cse625 - 0.0001) + cse3*cse915));
  const amrex::Real cse1152 = cse1151*cse957;
  const amrex::Real cse1153 = cse1077*cse1149 + cse1078*cse1151 + cse1145*cse908*cse952 + cse1145*cse928*cse958 + cse1149*cse966 + cse1150*cse968 + cse1150*cse969 + cse1152*cse967 + 844.16666666666686*cse260*cse908*cse929;
  const amrex::Real cse1154 = 70.859999999999985*cse260*cse972;
  const amrex::Real cse1155 = cse1150*cse962;
  const amrex::Real cse1156 = cse1150*cse961;
  const amrex::Real cse1157 = cse1081*cse1151 + cse1083*cse1155 + cse1083*cse1156 + cse1145*cse955*cse961 + cse1149*cse969 + cse1149*cse973 + cse1151*cse973 + cse1154*cse961 + cse1154*cse962;
  const amrex::Real cse1158 = cse984*(-cse1086*cse1149 - cse1087*cse1151 - cse1149*cse957 - cse1152 - cse1153 - cse1155 - cse1156 - cse1157 - 64.733333333333348*cse260*cse962*cse964);

  jac(1, 1) = -cse1 - cse10*cse56 - cse102 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3];
  jac(1, 2) = cse115 + cse124;
  jac(1, 3) = cse134 + cse136*state.xn[16] + cse140*cse141 + cse148 + cse156;
  jac(1, 4) = cse123*cse182 - cse141*cse191 + cse159 + cse164 + cse166 + 2*cse167 + 2*cse176 + 4*cse178 + cse179*state.xn[17] + 1.1399999999999999e-10*cse187 + cse190*state.xn[8] + cse195 + cse204;
  jac(1, 5) = cse123*cse209 + cse166 + 2*cse208 + cse212 + cse215;
  jac(1, 7) = cse220;
  jac(1, 8) = cse224;
  jac(1, 9) = cse190*state.xn[3];
  jac(1, 10) = -cse225;
  jac(1, 11) = cse124 + cse227;
  jac(1, 12) = cse230 + cse231 + cse234;
  jac(1, 15) = cse235 + cse236 + cse238 + cse239;
  jac(1, 16) = cse239 + cse243;
  jac(1, 17) = cse123*cse249 + cse136*state.xn[2] - 1.8000000000000002e-14*cse165 + 2*cse248 + cse256;
  jac(1, 18) = cse179*state.xn[3] - cse257 + cse259;
  jac(1, 19) = cse312*(-cse145*cse291 + 9.8669999999999991e-12*cse188*cse288*state.xn[3]*state.xn[8] - cse262 - cse263 - cse266 + 2*cse275*cse30 - cse276 + 2*cse279*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse280 + cse281 - 5.5769999999999994e-13*cse284 - cse286 + cse287 + cse289 - cse295 - cse311);
  jac(2, 1) = cse102;
  jac(2, 2) = -cse115;
  jac(2, 3) = -cse155;
  jac(2, 4) = cse159 + cse201 + cse313;
  jac(2, 5) = cse211 + cse214 + cse313;
  jac(2, 7) = cse216;
  jac(2, 11) = cse314;
  jac(2, 12) = -cse233;
  jac(2, 15) = cse314;
  jac(2, 16) = cse314;
  jac(2, 17) = cse254 - cse255;
  jac(2, 18) = cse257;
  jac(2, 19) = cse312*(-cse262 + cse297 + cse299 - cse300 - cse303 - cse315);
  jac(3, 1) = cse101 + cse317 - cse357 + cse65 - cse84 + cse90;
  jac(3, 2) = -cse114 - cse359;
  jac(3, 3) = -cse156 - cse369 - cse386 - cse391 - cse392 - cse407;
  jac(3, 4) = -cse204 - cse409 + cse410 + cse411 - cse416 - cse423 - cse428 - cse429;
  jac(3, 5) = -cse215 - cse409 - cse415 - cse422 - cse431;
  jac(3, 6) = cse432;
  jac(3, 7) = -cse434;
  jac(3, 8) = cse444 + cse446;
  jac(3, 9) = -cse449;
  jac(3, 10) = cse225;
  jac(3, 11) = -cse227 - cse359;
  jac(3, 12) = cse454;
  jac(3, 13) = cse457;
  jac(3, 14) = cse459;
  jac(3, 15) = cse460;
  jac(3, 16) = cse460;
  jac(3, 17) = -cse256 - cse461 - cse462 - cse463 - cse465 - cse466 - cse467 - cse468;
  jac(3, 18) = cse470;
  jac(3, 19) = cse312*(4.2899999999999994e-13*cse284 + cse292 + cse306 + cse311 + cse472 + cse473 + cse474 + cse477 + cse478 - cse480 + cse481 + cse482 - cse484 - cse485 - cse486 - cse487 - cse488);
  jac(3, 20) = cse489;
  jac(4, 1) = 3.0000000000000002e-15*cse10*d2g - cse490 - cse491 - cse492 - cse493 - cse495 - cse496;
  jac(4, 2) = cse499;
  jac(4, 3) = cse144 - cse146 - cse147 + cse406 + cse500*state.xn[16];
  jac(4, 4) = -cse121 - cse167 - cse176 - cse177*cse9 - cse418 - cse502 - cse503 - cse505 - cse506 - cse507 - cse508;
  jac(4, 5) = -cse121 - cse208 - cse212 - cse430 - cse509;
  jac(4, 7) = -cse216 - cse510;
  jac(4, 8) = cse443;
  jac(4, 9) = -cse447;
  jac(4, 10) = cse511;
  jac(4, 11) = cse499;
  jac(4, 12) = cse453;
  jac(4, 15) = -cse238 - cse498;
  jac(4, 16) = cse499;
  jac(4, 17) = -cse248 + cse465 + cse500*state.xn[2] + cse512 - cse513 + cse515;
  jac(4, 18) = cse470;
  jac(4, 19) = cse312*(-cse261 - 2.3622303960273254e-7*cse280 + cse306 - cse516 - cse517 - cse518 - cse519 - cse520 - cse522);
  jac(5, 1) = cse496;
  jac(5, 2) = cse358;
  jac(5, 4) = -cse164 + cse504 + cse506;
  jac(5, 5) = -cse210 - cse211 - cse431;
  jac(5, 7) = cse510;
  jac(5, 11) = cse358;
  jac(5, 15) = cse358;
  jac(5, 16) = cse358;
  jac(5, 17) = cse464;
  jac(5, 19) = -cse312*cse521;
  jac(6, 1) = cse329;
  jac(6, 2) = cse329;
  jac(6, 3) = cse369;
  jac(6, 4) = cse523;
  jac(6, 5) = cse409;
  jac(6, 6) = -cse432;
  jac(6, 7) = cse525;
  jac(6, 10) = cse526;
  jac(6, 11) = cse329;
  jac(6, 15) = cse329;
  jac(6, 16) = cse527;
  jac(6, 17) = cse466;
  jac(6, 19) = cse312*cse529;
  jac(7, 1) = cse530;
  jac(7, 2) = cse530;
  jac(7, 3) = -cse369;
  jac(7, 4) = -cse523;
  jac(7, 5) = -cse409;
  jac(7, 6) = cse432;
  jac(7, 7) = -cse525;
  jac(7, 10) = -cse526;
  jac(7, 11) = cse530;
  jac(7, 15) = cse530;
  jac(7, 16) = -cse527;
  jac(7, 17) = -cse466;
  jac(7, 19) = -cse312*cse529;
  jac(8, 1) = cse1 + cse534;
  jac(8, 2) = cse534;
  jac(8, 3) = cse386;
  jac(8, 4) = cse420 + cse536 + cse537*state.xn[8] + cse540;
  jac(8, 5) = cse540;
  jac(8, 8) = -cse224 - cse443 - cse446;
  jac(8, 9) = cse448 + cse537*state.xn[3];
  jac(8, 10) = cse541;
  jac(8, 11) = cse534;
  jac(8, 15) = cse534 + cse542;
  jac(8, 16) = -cse241 + cse534;
  jac(8, 17) = -cse462 + cse467 - cse543;
  jac(8, 19) = cse312*(cse263 - 1.287e-13*cse284 + cse294 + cse472 + 2.2769999999999997e-12*cse483 + cse544);
  jac(9, 1) = cse545;
  jac(9, 2) = cse545;
  jac(9, 3) = -cse386;
  jac(9, 4) = -cse423;
  jac(9, 5) = -cse422;
  jac(9, 7) = cse524;
  jac(9, 8) = cse446;
  jac(9, 9) = -cse449 - cse547;
  jac(9, 10) = cse526;
  jac(9, 11) = cse545;
  jac(9, 15) = cse545;
  jac(9, 16) = -cse343 - cse548;
  jac(9, 17) = -cse467;
  jac(9, 19) = cse312*(-cse519 - cse544 - cse550);
  jac(10, 1) = cse532;
  jac(10, 2) = cse532;
  jac(10, 3) = cse150;
  jac(10, 4) = cse551;
  jac(10, 5) = cse538;
  jac(10, 7) = -cse524;
  jac(10, 8) = cse224;
  jac(10, 10) = -cse552 - cse553;
  jac(10, 11) = cse226 + cse532;
  jac(10, 12) = cse232;
  jac(10, 15) = cse554;
  jac(10, 16) = cse241 + cse532;
  jac(10, 17) = -cse514 + cse543;
  jac(10, 19) = -cse295*cse312;
  jac(11, 3) = -cse150;
  jac(11, 9) = cse547;
  jac(11, 10) = cse552;
  jac(11, 11) = -cse226;
  jac(11, 16) = cse548;
  jac(11, 17) = cse514;
  jac(11, 19) = cse312*(cse292 + cse550);
  jac(12, 1) = -cse532 - cse555;
  jac(12, 2) = -cse105 - cse532;
  jac(12, 3) = cse146 - cse402 + cse404;
  jac(12, 4) = cse193 - cse194 - cse427 - cse551;
  jac(12, 5) = cse539;
  jac(12, 7) = cse524;
  jac(12, 10) = cse553;
  jac(12, 11) = cse533;
  jac(12, 12) = -cse234 - cse556;
  jac(12, 15) = -cse554;
  jac(12, 16) = cse242 + cse533;
  jac(12, 17) = -cse461 - cse543;
  jac(12, 18) = cse257 + cse258;
  jac(12, 19) = cse312*(-cse557 - cse558);
  jac(13, 1) = cse356;
  jac(13, 2) = cse356;
  jac(13, 3) = cse391;
  jac(13, 4) = cse416;
  jac(13, 5) = cse415;
  jac(13, 11) = cse356;
  jac(13, 13) = -cse457;
  jac(13, 14) = cse458;
  jac(13, 15) = cse356;
  jac(13, 16) = cse356;
  jac(13, 17) = cse468;
  jac(13, 19) = cse312*cse559;
  jac(13, 20) = cse560;
  jac(14, 1) = cse561;
  jac(14, 2) = cse561;
  jac(14, 3) = -cse391;
  jac(14, 4) = -cse416;
  jac(14, 5) = -cse415;
  jac(14, 11) = cse561;
  jac(14, 13) = cse457;
  jac(14, 14) = cse459;
  jac(14, 15) = cse561;
  jac(14, 16) = cse561;
  jac(14, 17) = -cse468;
  jac(14, 19) = -cse312*cse559;
  jac(14, 20) = cse489;
  jac(15, 1) = cse493;
  jac(15, 4) = cse562*state.xn[8];
  jac(15, 8) = cse443;
  jac(15, 9) = cse562*state.xn[3];
  jac(15, 12) = -cse232;
  jac(15, 15) = -cse235 - cse542;
  jac(15, 17) = cse462;
  jac(15, 19) = cse312*(-3.0029999999999999e-13*cse284 + cse471 + 5.3129999999999997e-12*cse483 + cse516);
  jac(16, 3) = cse148 + cse405;
  jac(16, 4) = cse428;
  jac(16, 7) = -cse219;
  jac(16, 8) = -cse224;
  jac(16, 9) = -cse547;
  jac(16, 12) = cse556;
  jac(16, 16) = -cse243 - cse548;
  jac(16, 17) = cse461;
  jac(16, 18) = cse259;
  jac(16, 19) = cse312*(cse266 + cse294 + cse520 + cse549 + cse557);
  jac(17, 3) = -cse407;
  jac(17, 4) = cse507;
  jac(17, 5) = cse210;
  jac(17, 8) = cse444;
  jac(17, 10) = -cse511;
  jac(17, 12) = cse454;
  jac(17, 17) = -cse463 - cse515;
  jac(17, 19) = cse312*(cse473 + cse522);
  jac(18, 1) = cse555;
  jac(18, 2) = cse105;
  jac(18, 3) = -cse392;
  jac(18, 4) = -cse508;
  jac(18, 7) = cse219;
  jac(18, 12) = cse233;
  jac(18, 16) = cse240;
  jac(18, 18) = -cse257 - cse469;
  jac(18, 19) = cse312*(cse308 + cse528 + cse558);
  jac(19, 1) = 1.461185424e-21*chi*cse184*cse185*cse705*state.xn[3]*(-cse697*cse702 + cse706) + 1.7e-24*chi*cse563*cse567*cse68*d2g + 1.7623947000000002e-12*crate*cse655*cse689*cse87*state.xn[5] + 1.602177e-12*crate*cse655 + 1.7623947000000002e-12*crate*cse719*cse720*state.xn[5] + 1.602177e-12*crate*cse719*state.xn[0] + 3.2043540000000001e-12*crate*cse719*state.xn[3] + 3.0000000000000002e-15*cse10*cse622*d2g + 2.0187430200000004e-26*cse10*cse690*d2g*state.xn[0]*(-cse619*cse697 + cse691) - 2.1789607199999999e-11*cse101 - 7.1777529600000012e-12*cse491 - 7.1777529600000012e-12*cse492 - 7.1777529600000012e-12*cse495 - cse580 - cse585*cse594 - cse603 - cse606 + 3.0000000000000002e-15*cse622*d2g*state.xn[0] + 4.39089220777172e-29*cse627*cse632*cse634*cse638*cse656*state.xn[8] - 4.39089220777172e-29*cse627*cse640 - cse658*cse688 - cse701*(-cse635*cse698 + cse698*cse699) - cse736*(-cse658*cse675 - cse684*(cse723*cse726 + cse731*(cse658*cse679 - cse727*cse729)) - cse734) - state.xn[11]*(cse858*cse875 + cse880*cse881 + cse884*cse885 + cse884*cse886) - state.xn[3]*((cse765) ? (
   0
)
: (
   cse781*cse810
)) - state.xn[7]*((cse989) ? (
   cse960*cse977 + cse982*cse983 + cse986*cse987 + cse986*cse988
)
: (
   0
));
  jac(19, 2) = cse1003 - state.xn[3]*((cse765) ? (
   0
)
: (
   cse784*cse810
));
  jac(19, 3) = 1.7e-24*chi*cse10*cse1014*cse563*cse68*d2g*(0.00025272365919999998*cse1015*cse137 + 0.036605692000000002*cse1016*cse137) + 4.6527220080000002e-11*crate*cse1019*cse641*cse643*cse66*cse720*state.xn[0]*state.xn[5] + 4.22974728e-11*crate*cse1019*cse641*cse643*cse66*std::pow(state.xn[0], 2) + 8.4594945600000001e-11*crate*cse1019*cse641*cse643*cse66*state.xn[0]*state.xn[3] - cse1004*cse594*state.xn[0] + 6.0732876035249664e-19*cse1005*cse1006*cse1007*cse588*cse592*state.xn[0]*state.xn[2] - cse1008 - 7.2151085033443149e-29*cse1009*cse1012*cse1013*cse110*cse570*state.xn[2] - cse1018*(-cse1004*cse590 + 5.9301787935659931e-17*cse1005*cse1006*cse1007*cse588*state.xn[2]) - cse1020*(0.086512999999999993*chi*cse1010*cse1011*cse2*cse573*cse68 - cse1012*cse571*cse576) - 2.1789607199999999e-11*cse151 + 2.6241084038530201e-25*cse264*cse632*cse634*cse638*cse656*state.xn[8] - 2.6241084038530201e-25*cse264*cse640 - cse664*cse688 - cse701*(-cse1021*cse635 + cse1021*cse699) - cse736*(-cse664*cse675 - cse664*cse684*cse730) - state.xn[11]*(cse1025*cse875 + cse1027*cse881 + cse1028*cse885 + cse1028*cse886) - state.xn[3]*((cse765) ? (
   0
)
: (
   cse795*cse810
)) - state.xn[7]*((cse989) ? (
   cse1035*cse977 + cse1038*cse983 + cse1041*cse987 + cse1041*cse988
)
: (
   0
));
  jac(19, 4) = 1.461185424e-21*chi*cse184*cse185*cse705*state.xn[3]*(-cse1053*cse702 + 2*cse706) + 5.6999999999999997e-11*chi*cse184*cse185*(6.4087080000000003e-13 + 2.5634832e-11/cse704) + 3.3999999999999999e-24*chi*cse563*cse567*cse68*d2g + 1.7623947000000002e-12*crate*cse1061*cse720*state.xn[5] + 1.602177e-12*crate*cse1061*state.xn[0] + 3.2043540000000001e-12*crate*cse1061*state.xn[3] + 1.7623947000000002e-12*crate*cse199*cse655*cse689*state.xn[5] + 3.2043540000000001e-12*crate*cse655 + 2.0187430200000004e-26*cse10*cse690*d2g*state.xn[0]*(-cse1053*cse619 + 2*cse691) - cse1043 - cse1045 - cse1047 - cse1049*cse631 - cse1050*cse1051 - 7.1777529600000012e-12*cse167 - 7.1777529600000012e-12*cse176 - 1.4355505920000002e-11*cse178 - 7.1777529600000012e-12*cse503 + 6.0000000000000005e-15*cse622*d2g*state.xn[0] - cse688 - cse701*(-cse1050*cse1054 - cse631*cse636) - cse736*(-cse1064 - cse675 - cse684*(cse1062*cse726 + cse731*(-cse1063*cse727 + cse679))) - state.xn[11]*(cse1069*cse875 + cse1074*cse881 + cse1076*cse885 + cse1076*cse886) - state.xn[3]*((cse765) ? (
   0
)
: (
   cse782*cse810
)) - state.xn[7]*((cse989) ? (
   cse1080*cse977 + cse1085*cse983 + cse1088*cse987 + cse1088*cse988
)
: (
   0
)) - ((cse765) ? (
   0.0
)
: (
   cse1065*cse806
));
  jac(19, 5) = cse1001*cse1091 - cse1043 - cse1045 - cse1047 + 3.3999999999999999e-24*cse1089 + cse1091*cse998 + cse1091*cse999 + cse198*cse992 - 7.1777529600000012e-12*cse208 - 7.1777529600000012e-12*cse509 + cse57*cse622 - cse736*(-cse1064 - cse684*(-cse1002*cse1063 + cse1062*cse724*cse725)) + cse995*(-cse1090*cse619 + 2*cse35*cse618*cse619) + cse996*(-cse1090*cse702 + 2*cse35*cse618*cse702);
  jac(19, 6) = cse1000*cse991 + cse1092;
  jac(19, 7) = cse1092;
  jac(19, 8) = -((cse989) ? (
   cse970*cse977 + cse974*cse983
)
: (
   0.0
));
  jac(19, 9) = -cse1048*cse700;
  jac(19, 10) = -cse665*cse687 - cse736*(cse1093*cse665*cse732*cse733 - cse683*(cse1093*cse662*cse676*cse732*cse733 - cse1094*cse677) - cse684*(-cse1002*cse1094 + cse1093*cse724*cse725));
  jac(19, 11) = cse1003;
  jac(19, 12) = -cse868*cse875 - cse872*cse881;
  jac(19, 15) = cse1003;
  jac(19, 16) = cse1003;
  jac(19, 17) = cse1001*cse1096 - 5.6994379667262474e-29*cse1042 - 9.6000000000000005e-32*cse1044 - 300.0*cse1046 + 5.0999999999999995e-24*cse1089 + cse1096*cse998 + cse1096*cse999 - 7.1777529600000012e-12*cse248 + cse252*cse992 + cse512*cse622 - 7.1777529600000012e-12*cse513 - cse736*(cse1097*cse665*cse732*cse733 - cse683*(cse1097*cse662*cse676*cse732*cse733 - cse1098*cse677) - cse684*(-cse1002*cse1098 + cse1097*cse724*cse725)) + cse995*(-cse1095*cse619 + 3*cse35*cse618*cse619) + cse996*(-cse1095*cse702 + 3*cse35*cse618*cse702);
  jac(19, 19) = cse312*(1.7e-24*chi*cse10*cse1014*cse563*cse68*d2g*(-0.00012636182959999999*cse1015*cse290 - 0.018302846000000001*cse1016*cse290) + 5.6227499999999995e-25*chi*cse10*cse567*cse68*d2g*std::pow(state.T, -0.85299999999999998) + 7.8348995947646122e-14*cse1006*cse1102*cse587*cse588*cse592*state.xn[0]*state.xn[2] - cse1017*cse1104*cse592 - cse1018*(7.6502807803182096e-12*cse1006*cse1102*cse587*cse588*state.xn[2] - 0.0015792330470357991*cse1099*cse584 - 2.0007175426973804e-9*cse1103*cse589 - 1.3338116951315871e-8*cse1104*cse585) - cse1020*(-1.8989*cse1013*cse1105 + cse3*cse574) - cse1020*(-0.043256499999999996*cse1105*cse573 + 0.84999999999999998*cse1105*cse576) - cse1049*cse1110 - cse1051*cse1114 - 1.6173435610483583e-5*cse1099*cse1100*cse592 - 2.0489994438957239e-11*cse1100*cse1101*cse593 - 5.2835524837844135e-18*cse1106*cse638*state.xn[8] - cse1111*cse688 - cse1113*cse619*cse695*cse994 - cse1113*cse696*cse702*cse996 - 2.1789607199999999e-11*cse297 - 7.1777529600000012e-12*cse517 - 7.1777529600000012e-12*cse518 - cse597*cse604*psigd_coll_interp2d_partial_1(cse10, state.T) - cse601*(1.6000000000000001e-32*cse110*cse277*cse598 + cse599) - 1000*cse659*std::pow(cse660, 999)*cse686*cse735 - cse701*(-cse1054*cse1114 - 182.41999999999999*cse1106 - cse1110*cse636) - cse736*(-cse1111*cse675 + cse1115*cse1116*cse678*cse682 + cse1115*cse1116 - cse1118*cse665*cse732 - cse683*(-cse1115*cse675*cse676 - cse1118*cse662*cse676*cse732 - cse1119*cse677) - cse684*(cse725*(cse1117*cse724 + alpha_co_interp2d_partial_0(state.T, cse673)) + cse731*(cse1111*cse679 - cse1119*cse727))) - state.xn[11]*(cse1139*cse875 + cse1143*cse881 + cse1144*cse885 + cse1144*cse886) - state.xn[3]*((cse765) ? (
   0
)
: (
   cse1065*cse1121 + cse806*cse809*(-cse1121*cse807 - cse797*cse806*(-cse783*(3.0182298984217999*cse1122*cse752*cse774*cse798 + 4.8223900769399082*cse1122*cse798*cse8 - 3.5529549287336835*cse1124 - 0.3872755400043702*cse1125 - 1.3735579540080118*cse1126) - cse785*(0.0708789387907936*cse1124 + 3.7035619079275177*cse1125 - 4.6974781513675152*cse1126 + 3.6184459289309552*cse1127 - 1.6316107607322889*cse1128) - cse787*(-3.7541549062629067*cse1124 + 2.2094886994529301*cse1125 - 1.528565035159452*cse1126 + 5.0409049417480247*cse1127 + 2.0057552335975877*cse1128) - state.xn[0]*((cse777) ? (
      cse7*cse776*(116.29033200000001*cse1124 + 19.239805*cse1125 + 80.639324000000002*cse1126 + 37.383713*cse1127 + 145.968309*cse1128)
   )
   : ((cse779) ? (
      cse778*(8.2184944748967013*cse1122*cse798*cse8 - 52.189748993977005*cse1124 - 48.951834264235494*cse1125 - 196.44057098336626*cse1126 - 192.38155095558542*cse1128)
   )
   : (
      cse780*(-17.135767342440825*cse1124 + 17.889115159724135*cse1125 - 50.756388852554174*cse1126 + 10.695627721640689*cse1127 + 41.010708268606813*cse1128)
   ))) - state.xn[2]*((cse629) ? (
      cse793*(445.51869310442481*cse1124 + 11324.985706577943*cse1125 + 6766.9891340104432*cse1126 + 38.719649225812766*cse1127 + 2370.6849681533818*cse1128 + 1300.6686834345674*cse1129 + 5869.3152370465659*cse1130 + 11077.448644792414*cse1131)
   )
   : (
      cse794*(4.297112944704045*cse1124 - 101.40241318979159*cse1125 - 43.540996231705549*cse1126 + 3.8689780091986448*cse1127 + 27.91190909651122*cse1128 - 117.34186576967058*cse1129 + 103.76942484394611*cse1130 + 123.18901581604325*cse1131)
   ))))
)) - state.xn[7]*((cse989) ? (
   cse1153*cse977 + cse1157*cse983 + cse1158*cse987 + cse1158*cse988
)
: (
   0
)));
  jac(20, 13) = -cse456;
  jac(20, 20) = cse560;
  jac(21, 13) = -cse455*state.rn[2*0+1];
  jac(21, 21) = cse560;
  // $JAFF END
  // $JAFF SUB nspec, nbands
  for (int i = 1; i <= INT_NEQS; i++) {
    jac(19, i) /= state.rho;
  }
  // $JAFF END

}

AMREX_INLINE
void actual_rhs_init() {}

#endif
