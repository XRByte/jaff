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
  const amrex::Real cse0 = 6.3999999999999996e-10*state.xn[0]*state.xn[4];
  const amrex::Real cse1 = -cse0;
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = cse2/state.xn[2];
  const amrex::Real cse4 = 0.00011320754716981131*cse3*state.xn[3];
  const amrex::Real cse5 = std::min(10000000000.0, cse4);
  const amrex::Real cse6 = 1.0/(cse5 + 1);
  const amrex::Real cse7 = cse6*state.xn[17]*state.xn[3];
  const amrex::Real cse8 = 1.6000000000000001e-9*cse7;
  const amrex::Real cse9 = cse5*cse8;
  const amrex::Real cse10 = 1.99e-9*std::pow(state.T, -0.19)*state.xn[11]*state.xn[16];
  const amrex::Real cse11 = cse10*cse6;
  const amrex::Real cse12 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse13 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av)*state.xn[15];
  const amrex::Real cse14 = 1.0/state.T;
  const amrex::Real cse15 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse14)*state.xn[15]*state.xn[7];
  const amrex::Real cse16 = M_LN10;
  const amrex::Real cse17 = 1.0/cse16;
  const amrex::Real cse18 = 2*state.xn[3];
  const amrex::Real cse19 = cse18 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + state.xn[1] + 2*state.xn[4];
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
   cse26
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
  const amrex::Real cse35 = cse34*state.xn[16]*state.xn[2];
  const amrex::Real cse36 = cse19*d2g;
  const amrex::Real cse37 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[11]*state.xn[1];
  const amrex::Real cse38 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse39 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse38)/cse38;
  const amrex::Real cse40 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse41 = chi*cse39*cse40;
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
  const amrex::Real cse64 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[10]*state.xn[2];
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
  const amrex::Real cse75 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[13]*state.xn[2];
  const amrex::Real cse76 = 2.0740886949963892e-18*state.c_hat*state.xn[12];
  const amrex::Real cse77 = cse76*state.rn[2*0];
  const amrex::Real cse78 = -cse77;
  const amrex::Real cse79 = 2.1659999999999999e-12*cse36*state.xn[13]/(1.534817233071468e-7*std::pow(cse68, 1.8740000000000001)*(43750.0*std::pow(cse69, -7.538e-5*cse70 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse80 = 16800.0*crate*cse24*state.xn[12];
  const amrex::Real cse81 = (1.0537934368240927e-9*std::pow(cse14, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse14) + 1.31e-7*std::exp(-26.600000000000001*cse14) + 3.4e-8*std::exp(-7.6200000000000001*cse14) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse14)))*state.xn[16]*state.xn[7];
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
  const amrex::Real cse102 = 1.6999999999999999e-9*state.xn[16]*state.xn[9];
  const amrex::Real cse103 = cse33*state.xn[3];
  const amrex::Real cse104 = 5.6999999999999997e-11*cse41;
  const amrex::Real cse105 = 7.2000000000000002e-15*cse50;
  const amrex::Real cse106 = cse1 + cse105 + cse74;
  const amrex::Real cse107 = -cse54;
  const amrex::Real cse108 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse109 = cse105 + cse108 + cse52 + cse87;
  const amrex::Real cse110 = cse100 + cse53;
  const amrex::Real cse111 = 184.0*cse24*cse56;
  const amrex::Real cse112 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av)*state.xn[9];
  const amrex::Real cse113 = cse111 + cse112 - cse15;
  const amrex::Real cse114 = 9.1500000000000005e-10*cse48*state.xn[8];
  const amrex::Real cse115 = cse102 + cse57 - cse64;
  const amrex::Real cse116 = cse108 - cse45;
  const amrex::Real cse117 = cse101 + cse12 + cse13;
  const amrex::Real cse118 = -cse75 + cse77 - cse79 + cse80;
  const amrex::Real cse119 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse120 = cse119*state.xn[2];
  const amrex::Real cse121 = std::exp(-118400.0*cse14)/(4.4460389837719569e-9*cse120 + 626500000.0);
  const amrex::Real cse122 = cse69 + 50;
  const amrex::Real cse123 = std::log(1.7*cse2*cse67/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse124 = 0.001*state.T;
  const amrex::Real cse125 = cse124 + 1;
  const amrex::Real cse126 = std::pow(cse125, -2);
  const amrex::Real cse127 = 1.0/cse125;
  const amrex::Real cse128 = 1.0/(cse23*(cse104 + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse126 + 0.055500000000000001*cse127 - 11.06)*cse25 + std::pow(10, -2.0230000000000001*cse126 - 3.6709999999999998*cse127 - 11.08)*cse24) + 1);
  const amrex::Real cse129 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse130 = (1.0/100.0)*state.T;
  const amrex::Real cse131 = std::log(cse130);
  const amrex::Real cse132 = state.T < 500;
  const amrex::Real cse133 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse134 = 4.5299999999999998e-6*cse129 + 7.5799999999999997e-10*std::pow(cse130, 0.0086999999999999994*cse131 + 0.12809999999999999)*state.xn[0] + (0.75*((cse132) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse133
)) + 0.25*((cse132) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse133
)))*state.xn[3];
  const amrex::Real cse135 = cse134*std::exp(-91.209999999999994*cse14)/(cse134 + 2.3e-6);
  const amrex::Real cse136 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse137 = 1.75*cse136*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse136;
  const amrex::Real cse138 = l0_co_interp1d(state.T);
  const amrex::Real cse139 = 1.0/cse138;
  const amrex::Real cse140 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse23*ncol_H))
));
  const amrex::Real cse141 = 1.0/llte_co_interp2d(state.T, cse140);
  const amrex::Real cse142 = nhalf_co_interp2d(state.T, cse140);
  const amrex::Real cse143 = cse17*std::log(cse19);
  const amrex::Real cse144 = 2.0*cse24*((cse143 < 2) ? (
   10.0
)
: ((cse143 < 4) ? (
   (3.0/2.0)*cse143 + 7
)
: ((cse143 < 7) ? (
   (4.0/3.0)*cse143 + 23.0/3.0
)
: ((cse143 < 10) ? (
   (1.0/3.0)*cse143 + 44.0/3.0
)
: (
   18.0
))))) + cse25*(26.399999999999999*std::sqrt(cse82/(cse82 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse145 = 0.042999999999999997*cse131;
  const amrex::Real cse146 = 4.34e-10*std::pow(cse130, 0.755 - 0.16*cse131)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse130, 0.52500000000000002*cse131 + 1.198) + 5.2499999999999998e-13*std::pow(cse130, cse145 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse147 = 3.5700000000000001e-10*std::pow(cse130, 0.41899999999999998 - 0.0030000000000000001*cse131)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse130, 0.025000000000000001*cse131 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse130, cse145 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse148 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse149 = 3.1899999999999998e-10*std::pow(cse130, 0.36899999999999999 - 0.0060000000000000001*cse131)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse130, 0.041000000000000002*cse131 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse130, 0.058000000000000003*cse131 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse150 = cse147 + 8.9099999999999997e-5;
  const amrex::Real cse151 = cse148*cse149;
  const amrex::Real cse152 = cse146*cse147*cse148 + (1.0/15.0)*cse146*cse149*std::exp(-425.40000000000003*cse14) + cse150*cse151;
  const amrex::Real cse153 = cse146 + 1.7499999999999998e-5;
  const amrex::Real cse154 = cse149 + 1.34e-10;
  const amrex::Real cse155 = (3.0/5.0)*cse147*std::exp(-227.80000000000001*cse14);
  const amrex::Real cse156 = cse151*cse153 + cse153*cse155 + cse154*cse155;
  const amrex::Real cse157 = 1.0/((1.0/3.0)*cse146*cse154*std::exp(-98.800000000000011*cse14) + cse150*cse153 + cse150*cse154 + cse152 + cse156);
  const amrex::Real cse158 = state.T < 100.0;
  const amrex::Real cse159 = state.T < 6000.0;
  const amrex::Real cse160 = state.T < 10;
  const amrex::Real cse161 = std::log(cse124);
  const amrex::Real cse162 = cse161*cse17;
  const amrex::Real cse163 = std::pow(cse16, -8);
  const amrex::Real cse164 = std::pow(cse161, 8)*cse163;
  const amrex::Real cse165 = std::pow(cse16, -7);
  const amrex::Real cse166 = std::pow(cse161, 7)*cse165;
  const amrex::Real cse167 = std::pow(cse16, -6);
  const amrex::Real cse168 = std::pow(cse161, 6)*cse167;
  const amrex::Real cse169 = std::pow(cse16, -5);
  const amrex::Real cse170 = std::pow(cse161, 5)*cse169;
  const amrex::Real cse171 = std::pow(cse16, -4);
  const amrex::Real cse172 = std::pow(cse161, 4)*cse171;
  const amrex::Real cse173 = std::pow(cse16, -3);
  const amrex::Real cse174 = std::pow(cse161, 3);
  const amrex::Real cse175 = cse173*cse174;
  const amrex::Real cse176 = std::pow(cse161, 2)*cse21;
  const amrex::Real cse177 = std::pow(10.0, 16.815729999999999*cse162 + 70.609154000000004*cse164 + 364.14445999999998*cse166 + 801.81246999999996*cse168 + 983.67575999999997*cse170 + 734.71650999999997*cse172 + 343.1918*cse175 + 96.743155000000002*cse176 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse178 = state.xn[5] + state.xn[6];
  const amrex::Real cse179 = std::pow(10.0, 1.5714710999999999*cse162 + 0.32168730000000001*cse170 - 0.51002221000000003*cse172 - 0.23619984999999999*cse175 + 0.015391166*cse176 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse162 + 0.19191374999999999*cse170 - 0.16596184*cse172 + 0.29036281000000003*cse175 - 0.81520437999999995*cse176 - 23.689236999999999)*cse178 + std::pow(10.0, 2.0943374000000001*cse161*cse17 - 0.033638326000000003*cse170 - 0.14913216000000001*cse172 + 0.43693353000000001*cse173*cse174 - 0.77151435999999995*cse176 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse180 = std::pow(10, 37.383713*cse162 + 3.8479610000000002*cse170 + 20.159831000000001*cse172 + 48.656103000000002*cse175 + 58.145166000000003*cse176 - 16.818342000000001)*state.xn[0] + cse179;
  const amrex::Real cse181 = std::pow(10.0, 1.6802758*cse162 - 6.3701156000000001*cse164 + 6.4380698000000001*cse166 + 8.9167182999999994*cse168 - 8.8077017000000009*cse170 - 4.7274035999999997*cse172 + 4.0406627000000004*cse175 + 0.93310621999999999*cse176 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse182 = state.T < 1000.0;
  const amrex::Real cse183 = std::pow(10.0, 3.5692468000000002*cse161*cse17 - 4.2519023000000002*cse170 - 21.328264000000001*cse172 - 27.850082*cse175 - 11.33286*cse176 - 24.311209000000002)*state.xn[0] + cse179;
  const amrex::Real cse184 = std::pow(10.0, 4.6450521*cse162 + 1.5538288*cse170 - 5.5108049000000001*cse172 + 5.9369081000000001*cse175 - 3.7209846*cse176 - 24.311209000000002)*state.xn[0] + cse179;
  const amrex::Real cse185 = std::pow(10.0, -0.62120210234156448*cse169 + 3.7525488682084647*cse17 - 1.5370575959566981*cse171 + 2.5133588394664677*cse173 - 2.4768712409536775*cse21 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse169 + 3.9225864834863131*cse17 - 1.7105157385968939*cse171 + 1.6702447513374001*cse173 - 2.6171337683481011*cse21 - 23.689236999999999)*cse178 + std::pow(10.0, 5.9406293599919788*cse169 + 2.8156982240432273*cse17 - 5.2566362076906961*cse171 - 1.3586848802337363*cse173 + 0.049411830040524525*cse21 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse169 + 8.3228160852326614*cse17 - 56.79810800956944*cse171 + 34.150687525022562*cse173 - 11.945856385319289*cse21 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse186 = ((cse159) ? (
   cse14
)
: (
   0.00016666666666666666
));
  const amrex::Real cse187 = 1.6e-18*std::exp(-11700.0*cse186) + 6.7e-19*std::exp(-5860.0*cse186) + 3.0e-24*std::exp(-510.0*cse186) + 4.985670872372847e-33*((cse159) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse159) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse159) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse188 = std::log(0.001*((cse159) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse189 = cse169*std::pow(cse188, 5);
  const amrex::Real cse190 = cse171*std::pow(cse188, 4);
  const amrex::Real cse191 = std::pow(cse188, 3);
  const amrex::Real cse192 = std::pow(cse188, 2)*cse21;
  const amrex::Real cse193 = cse17*cse188;
  const amrex::Real cse194 = cse173*cse191;
  const amrex::Real cse195 = cse163*std::pow(cse188, 8);
  const amrex::Real cse196 = cse165*std::pow(cse188, 7);
  const amrex::Real cse197 = cse167*std::pow(cse188, 6);
  const amrex::Real cse198 = 0.045999999999999999*cse131;
  const amrex::Real cse199 = 1.7258000000000001e-6*cse129;
  const amrex::Real cse200 = 8.9000000000000003e-11*std::pow(cse130, cse198 + 0.22800000000000001)*state.xn[0] + cse199*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(0.0070527699999999999*cse70 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse70*(cse70*(cse70*(0.093813800000000003*cse70 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse130, 0.037999999999999999*cse131 + 0.16900000000000001) + 2.15e-11*std::pow(cse130, 0.048000000000000001*cse131 - 0.01))*state.xn[3];
  const amrex::Real cse201 = 2.6400000000000002e-10*std::pow(cse130, cse198 + 0.23100000000000001)*state.xn[0] + cse199*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(0.00235272*cse70 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse70*(cse70*(cse70*(0.097857299999999994*cse70 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse130, 0.031*cse131 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse130, 0.064000000000000001*cse131 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse202 = 2.8763333333333333e-6*cse129*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(-0.00065632500000000003*cse70 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse70*(cse70*(cse70*(0.10508000000000001*cse70 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse130, 0.057000000000000002*cse131 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse130, 0.049000000000000002*cse131 - 0.0040000000000000001) + 1.675e-11*std::pow(cse130, 0.10199999999999999*cse131 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse203 = 5*std::exp(-62.460000000000001*cse14);
  const amrex::Real cse204 = cse202 + 7.8800000000000004e-8;
  const amrex::Real cse205 = cse200*cse203;
  const amrex::Real cse206 = (25.0/3.0)*cse200*cse201*std::exp(-101.30000000000001*cse14) + cse201*cse202*cse203 + cse204*cse205;
  const amrex::Real cse207 = cse201 + 2.65e-7;
  const amrex::Real cse208 = cse200 + 1.81e-14;
  const amrex::Real cse209 = 3*cse202*std::exp(-23.619999999999997*cse14);
  const amrex::Real cse210 = cse205*cse207 + cse207*cse209 + cse208*cse209;
  const amrex::Real cse211 = 1.0/((5.0/3.0)*cse201*cse208*std::exp(-38.840000000000003*cse14) + cse204*cse207 + cse204*cse208 + cse206 + cse210);

  ydot(1) = cse1 + cse11 + cse12 + cse13 + cse15 + cse18*cse33 + 2*cse31 - cse32 + 2.9935562710997168e-6*cse35 - 6.0000000000000005e-15*cse36*state.xn[0] + cse37 + 1.1399999999999999e-10*cse41*state.xn[3] + 4.2899999999999994e-13*cse44 + cse47 + cse52 + cse55 + 3.2000000000000001e-9*cse7 + cse72 + cse9;
  ydot(2) = cse0 + cse51 + cse60 + cse63 - cse65 - cse71 + cse73;
  ydot(3) = -cse72 + cse74 - cse75 - cse78 - cse79 + cse80 - cse85 - cse87 - cse96 - cse99;
  ydot(4) = -cse100 - cse101 + cse102 - cse103 - cse104*state.xn[3] - cse106 + cse11 + 3.0000000000000002e-15*cse19*d2g*state.xn[0] - cse31 + 4.5427507615910105e-7*cse34*state.xn[16]*state.xn[2] - cse51 - cse55 - cse99;
  ydot(5) = cse106 + cse107;
  ydot(6) = cse109;
  ydot(7) = -cse109;
  ydot(8) = cse110 + cse113 + 9.8999999999999995e-14*cse44 + cse46 + cse96;
  ydot(9) = cse108 - cse114 + cse89 - cse91 - cse92 + cse93 + cse95 - cse97;
  ydot(10) = -cse113 - cse115 - cse116;
  ydot(11) = cse114 + cse115;
  ydot(12) = cse111 + cse112 + cse116 + cse117 + cse73 + cse8;
  ydot(13) = -cse118;
  ydot(14) = cse118;
  ydot(15) = -cse110 + 2.3099999999999997e-13*cse42*cse43*state.xn[3]*state.xn[8] - cse47;
  ydot(16) = -cse114 - cse117 - cse15 - cse49 + cse9;
  ydot(17) = -cse102 - cse107 - cse85;
  ydot(18) = cse49 - cse73 - cse98;
  ydot(19) = 1.7e-24*chi*cse19*cse66*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse122, 0.43099999999999999)*(0.0043*std::pow(cse122, 0.69199999999999995) + 1) + 1) + 5.6999999999999997e-11*chi*cse39*cse40*(2.5634832e-11*cse128 + 6.4087080000000003e-13)*state.xn[3] + 1.602177e-12*crate*cse144*state.xn[0] + 3.2043540000000001e-12*crate*cse144*state.xn[3] + 1.7623947000000002e-12*crate*cse144*state.xn[5]/cse58 - 7.1777529600000012e-12*cse103 - 1.3659996292638161e-10*cse119*cse121*cse62/(1.3338116951315871e-8*cse120*cse121 + 1) - 5.7927337833400002e-20*cse135*state.xn[8]/(2*cse135 + 1) - cse137*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse137*cse141 + cse139*std::pow(cse137/cse142, alpha_co_interp2d(state.T, cse140))*(-cse138*cse141*cse142 + 1) + cse139) + 3.0000000000000002e-15*cse19*d2g*(6.7291434000000005e-12*cse128 + 3.2043540000000002e-13)*state.xn[0] - 7.1777529600000012e-12*cse31 - 1.8998126555754159e-29*cse36*std::pow(state.T, 0.45350000000000001 + 2.234/cse123)*std::exp(cse123*(1.4419999999999999 - 0.050889999999999998*cse123))*state.xn[2] - 2.1789607199999999e-11*cse63 - (2.3872025442750958e-19*cse152*cse157 + 2.8023005140019997e-18*cse156*cse157)*state.xn[11] - ((((cse132 && cse158 && cse159) ? (
   cse160 || cse177 + cse180 < 1.0e-100
)
: (
   ((cse158 && cse159) ? (
      cse160 || cse180 + cse181 < 1.0e-100
   )
   : (
      ((cse132 && cse159 && cse182) ? (
         cse160 || cse177 + cse183 < 1.0e-100
      )
      : (
         ((cse159 && cse182) ? (
            cse160 || cse181 + cse183 < 1.0e-100
         )
         : (
            ((cse132 && cse159) ? (
               cse160 || cse177 + cse184 < 1.0e-100
            )
            : (
               ((cse159) ? (
                  cse160 || cse181 + cse184 < 1.0e-100
               )
               : (
                  ((cse132) ? (
                     cse160 || std::pow(10.0, 7500.646099484944*cse163 + 21588.957206746247*cse165 + 26530.843406310876*cse167 + 18165.631967965237*cse169 + 30.129743459482278*cse17 + 7572.4886742758581*cse171 + 1974.131269262874*cse173 + 310.58441786958315*cse21 - 21.928795999999998)*state.xn[2] + cse185 < 1.0e-100
                  )
                  : (
                     cse160 || std::pow(10.0, -676.68255490510751*cse163 + 381.69251128040054*cse165 + 295.04162851880631*cse167 - 162.65264843551881*cse169 + 3.0106500755647456*cse17 - 48.723840736518795*cse171 + 23.242975457496804*cse173 + 2.9956460707652877*cse21 - 22.921188999999998)*state.xn[2] + cse185 < 1.0e-100
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
   cse187/(cse187/(std::pow(10.0, 0.19191374999999999*cse189 - 0.16596184*cse190 - 0.81520437999999995*cse192 + 2.1892372*cse193 + 0.29036281000000003*cse194 - 23.689236999999999)*cse178 + std::pow(10.0, 0.32168730000000001*cse189 - 0.51002221000000003*cse190 + 0.015391166*cse192 + 1.5714710999999999*cse193 - 0.23619984999999999*cse194 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse17*cse188 + 0.43693353000000001*cse173*cse191 - 0.033638326000000003*cse189 - 0.14913216000000001*cse190 - 0.77151435999999995*cse192 - 23.962112000000001)*state.xn[3] + ((cse132) ? (
      std::pow(10.0, 983.67575999999997*cse189 + 734.71650999999997*cse190 + 96.743155000000002*cse192 + 16.815729999999999*cse193 + 343.1918*cse194 + 70.609154000000004*cse195 + 364.14445999999998*cse196 + 801.81246999999996*cse197 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse189 - 4.7274035999999997*cse190 + 0.93310621999999999*cse192 + 1.6802758*cse193 + 4.0406627000000004*cse194 - 6.3701156000000001*cse195 + 6.4380698000000001*cse196 + 8.9167182999999994*cse197 - 22.921188999999998)
   ))*state.xn[2] + ((((cse159) ? (
      cse158
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse189 + 20.159831000000001*cse190 + 58.145166000000003*cse192 + 37.383713*cse193 + 48.656103000000002*cse194 - 16.818342000000001)
   )
   : ((((cse159) ? (
      cse182
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse17*cse188 - 4.2519023000000002*cse189 - 21.328264000000001*cse190 - 11.33286*cse192 - 27.850082*cse194 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse189 - 5.5108049000000001*cse190 - 3.7209846*cse192 + 4.6450521*cse193 + 5.9369081000000001*cse194 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse206*cse211 + 2.5697412351439998e-22*cse210*cse211
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

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True SPECIFIC_EINT True]$
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
  const amrex::Real cse61 = 1.6737729999999998e-24*state.xn[0] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse62 = 2.0709734999998963e-16*state.T/cse61;
  const amrex::Real cse63 = 1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9];
  const amrex::Real cse64 = cse63*state.T/std::pow(cse61, 2);
  const amrex::Real cse65 = cse62 - 2.5106594999998743e-24*cse64;
  const amrex::Real cse66 = std::pow(state.T, -2);
  const amrex::Real cse67 = std::exp(-22.5*cse3);
  const amrex::Real cse68 = 2.8349999999999999e-12*cse66*cse67*state.xn[3]*state.xn[6];
  const amrex::Real cse69 = -cse68;
  const amrex::Real cse70 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse71 = std::sqrt(cse3);
  const amrex::Real cse72 = std::sqrt(3)*cse3*cse71*state.xn[15];
  const amrex::Real cse73 = 1.7697487500000004e-8*cse72*state.xn[6];
  const amrex::Real cse74 = cse26*cse66;
  const amrex::Real cse75 = cse13*cse3;
  const amrex::Real cse76 = cse12*cse75;
  const amrex::Real cse77 = cse3*cse8;
  const amrex::Real cse78 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(1.3*cse3*cse8 - 3.2400000000000002*cse76) + cse18*cse7*(0.65400000000000003*cse76 + 0.41599999999999998*cse77)
));
  const amrex::Real cse79 = cse10*cse78;
  const amrex::Real cse80 = cse24*cse31;
  const amrex::Real cse81 = cse50*cse79;
  const amrex::Real cse82 = cse21*cse34*cse50*cse78;
  const amrex::Real cse83 = ((cse28) ? (
   cse54*(1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 - cse31*cse82 - cse49*cse81 + 53300.0*cse74 + cse79*cse80)
)
: (
   0
));
  const amrex::Real cse84 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse85 = 1.0/cse2;
  const amrex::Real cse86 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse87 = cse24*cse25;
  const amrex::Real cse88 = ((cse28) ? (
   cse59*(149925037481.25937*cse24*cse85*(3.3349999999999998e-12*cse5*cse85 + 4.2414529999999999e-7*cse5*cse86)*std::exp(cse4) - cse25*cse82 - cse6*cse81 + 43900.0*cse74 + cse79*cse87)
)
: (
   0
));
  const amrex::Real cse89 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse90 = std::exp(-227*cse3);
  const amrex::Real cse91 = cse90*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse92 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse93 = std::exp(-23*cse3);
  const amrex::Real cse94 = cse93*state.xn[3]*state.xn[8];
  const amrex::Real cse95 = cse92*cse94;
  const amrex::Real cse96 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse97 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse98 = cse97*state.xn[4];
  const amrex::Real cse99 = cse98*state.xn[3];
  const amrex::Real cse100 = 3.7768240343347639e-14*cse96*cse99;
  const amrex::Real cse101 = 7.3920000000000002e-11*cse99*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse102 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse103 = cse90*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse104 = cse103*state.xn[11];
  const amrex::Real cse105 = 227*cse104*cse66*state.xn[1];
  const amrex::Real cse106 = 1.0/state.xn[2];
  const amrex::Real cse107 = cse106*cse85;
  const amrex::Real cse108 = cse106*cse2;
  const amrex::Real cse109 = 0.00011320754716981131*cse108;
  const amrex::Real cse110 = cse109*state.xn[3];
  const amrex::Real cse111 = (((cse110 - 10000000000.0 > 0) ? (
   0
)
: ((cse110 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse112 = cse111*state.xn[17];
  const amrex::Real cse113 = cse112*cse30;
  const amrex::Real cse114 = cse107*cse113;
  const amrex::Real cse115 = std::min(10000000000.0, cse110);
  const amrex::Real cse116 = cse115 + 1;
  const amrex::Real cse117 = std::pow(cse116, -2);
  const amrex::Real cse118 = 1.811320754716981e-13*cse117;
  const amrex::Real cse119 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse120 = std::exp(0.11*cse3);
  const amrex::Real cse121 = cse120*state.xn[15];
  const amrex::Real cse122 = cse121*state.xn[7];
  const amrex::Real cse123 = 8.7465066538127767e-11*cse122*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse122*std::pow(state.T, -1.339);
  const amrex::Real cse124 = cse119 + cse123;
  const amrex::Real cse125 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse126 = std::pow(cse125, 2);
  const amrex::Real cse127 = std::pow(cse125, 3);
  const amrex::Real cse128 = std::pow(cse125, 4);
  const amrex::Real cse129 = std::pow(cse125, 5);
  const amrex::Real cse130 = std::pow(cse125, 6);
  const amrex::Real cse131 = std::pow(cse125, 7);
  const amrex::Real cse132 = std::exp(-2.0391498499999999e-6*std::pow(cse125, 8) - 5.7393287500000003*cse126 + 1.56315498*cse127 - 0.28770560000000001*cse128 + 0.034825597700000002*cse129 - 0.00263197617*cse130 + 0.000111954395*cse131);
  const amrex::Real cse133 = 5.9078447946382801e-70*cse132*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse134 = state.xn[0]*state.xn[2];
  const amrex::Real cse135 = cse134*((cse28) ? (
   7.9971871901929577e-69*cse132*std::pow(state.T, 12.536555999999999) + cse133*(-11.478657500000001*cse125*cse3 + 4.6894649399999997*cse126*cse3 - 1.1508224*cse127*cse3 + 0.1741279885*cse128*cse3 - 0.015791857020000001*cse129*cse3 + 0.00078368076500000001*cse130*cse3 - 1.6313198799999999e-5*cse131*cse3)
)
: (
   0
));
  const amrex::Real cse136 = std::pow(cse3, 1.5);
  const amrex::Real cse137 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse138 = std::pow(cse137, -2.242);
  const amrex::Real cse139 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse140 = 7.3220359950196314e-6*cse136*cse138*cse139;
  const amrex::Real cse141 = 0.0005114031087522854*std::pow(cse137, -3.242)*cse139*std::pow(cse3, 1.907);
  const amrex::Real cse142 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse143 = chi*cse142;
  const amrex::Real cse144 = cse108*cse143;
  const amrex::Real cse145 = std::pow(cse144, 1.3779999999999999);
  const amrex::Real cse146 = 1.7*cse144;
  const amrex::Real cse147 = std::log(state.T);
  const amrex::Real cse148 = 1.102e-5*cse147 + 0.4723;
  const amrex::Real cse149 = std::pow(cse146, -cse148);
  const amrex::Real cse150 = cse149*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse151 = 508.69999999999999*cse150;
  const amrex::Real cse152 = cse151 + 1;
  const amrex::Real cse153 = cse145*cse152;
  const amrex::Real cse154 = -cse148;
  const amrex::Real cse155 = 0.5*cse3;
  const amrex::Real cse156 = cse3*std::log(cse146);
  const amrex::Real cse157 = 1.6774414051711492e-5*cse145;
  const amrex::Real cse158 = cse152*cse157 + 1;
  const amrex::Real cse159 = cse10*d2g;
  const amrex::Real cse160 = 1.2250000000000001e-11*cse159*state.xn[1]/std::pow(cse158, 2);
  const amrex::Real cse161 = cse160*(-1.1557571281629217e-5*cse153*cse3 - cse157*(8.0679819999999989*cse149*std::pow(state.T, -0.98414000000000001) + cse151*(cse154*cse155 - 1.102e-5*cse156)));
  const amrex::Real cse162 = 1.0/cse116;
  const amrex::Real cse163 = 9.0566037735849048e-14*cse114;
  const amrex::Real cse164 = cse162*cse163;
  const amrex::Real cse165 = cse117*cse163;
  const amrex::Real cse166 = cse115*cse165;
  const amrex::Real cse167 = -cse164 + cse166;
  const amrex::Real cse168 = cse162*state.xn[16];
  const amrex::Real cse169 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse170 = cse106*cse111*cse117*state.xn[11]*state.xn[16];
  const amrex::Real cse171 = cse168*cse169 + 1.1264150943396225e-13*cse170*std::pow(state.T, -0.68999999999999995)*state.xn[3];
  const amrex::Real cse172 = cse135 + cse140 - cse141 - cse161 + cse167 + cse171;
  const amrex::Real cse173 = 0.66666666666670005*cse61/cse63;
  const amrex::Real cse174 = cse173*(-cse100 + cse101 + 9.8669999999999991e-12*cse102*cse93*state.xn[3]*state.xn[8] + cse105 - cse114*cse118 - cse124 - cse172 + 2*cse30*cse83 - cse69 - cse70 - cse73 - cse84 + 2*cse88*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse89 + cse91 - 5.5769999999999994e-13*cse95);
  const amrex::Real cse175 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse176 = cse175*state.xn[17];
  const amrex::Real cse177 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse178 = 1.5*cse11*state.xn[0] + 2.2999999999999998*cse11*state.xn[3];
  const amrex::Real cse179 = crate*cse178;
  const amrex::Real cse180 = ((cse28) ? (
   cse133
)
: (
   0.0
));
  const amrex::Real cse181 = cse180*state.xn[2];
  const amrex::Real cse182 = 2.2999999999999998*cse36 + 1.5*cse38;
  const amrex::Real cse183 = -1.5*cse11 + cse182;
  const amrex::Real cse184 = -cse183;
  const amrex::Real cse185 = crate*state.xn[0];
  const amrex::Real cse186 = cse184*cse185;
  const amrex::Real cse187 = 1.0/cse158;
  const amrex::Real cse188 = 1.2250000000000001e-11*cse187;
  const amrex::Real cse189 = d2g*state.xn[1];
  const amrex::Real cse190 = cse188*cse189;
  const amrex::Real cse191 = cse176 + cse177 + cse179 + cse181 + cse186 - cse190;
  const amrex::Real cse192 = cse62 - 2.5092930924433742e-24*cse64;
  const amrex::Real cse193 = 4.8813573300130879e-6*cse136*cse138;
  const amrex::Real cse194 = -cse182;
  const amrex::Real cse195 = -cse185*cse194 + cse190;
  const amrex::Real cse196 = cse159*cse188 + cse193*state.xn[2] + cse195;
  const amrex::Real cse197 = cse104 + cse196;
  const amrex::Real cse198 = cse10*((cse20) ? (
   0
)
: (
   -cse40
));
  const amrex::Real cse199 = cse50*(-cse198 - cse21);
  const amrex::Real cse200 = cse199*cse22;
  const amrex::Real cse201 = cse30*((cse28) ? (
   cse54*(cse198*cse80 + cse199*cse49 + cse200*cse31 + cse33)
)
: (
   0
));
  const amrex::Real cse202 = -cse57;
  const amrex::Real cse203 = ((cse28) ? (
   cse59*(cse198*cse87 + cse199*cse6 + cse200*cse25 + cse58)
)
: (
   0
));
  const amrex::Real cse204 = cse9*state.xn[0];
  const amrex::Real cse205 = 2*cse201 + cse202 + cse203*cse204;
  const amrex::Real cse206 = std::pow(state.xn[2], 2);
  const amrex::Real cse207 = 1.0/cse206;
  const amrex::Real cse208 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse209 = cse111*cse117*cse207*cse208*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse210 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse211 = 2.9935562710997168e-6*cse210;
  const amrex::Real cse212 = cse62 - 1.3664075564999315e-27*cse64;
  const amrex::Real cse213 = cse2*cse207;
  const amrex::Real cse214 = cse113*cse213;
  const amrex::Real cse215 = 3.6226415094339619e-13*cse117;
  const amrex::Real cse216 = 1.811320754716981e-13*cse162;
  const amrex::Real cse217 = cse214*cse216;
  const amrex::Real cse218 = cse118*cse214;
  const amrex::Real cse219 = cse115*cse218;
  const amrex::Real cse220 = -cse217 + cse219;
  const amrex::Real cse221 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse222 = cse221*state.xn[10];
  const amrex::Real cse223 = cse180*state.xn[0];
  const amrex::Real cse224 = cse160*(0.0085331444281056355*cse106*cse145*cse150*cse154 + 2.3115142563258434e-5*cse106*cse153) + cse193*state.xn[1] - cse223;
  const amrex::Real cse225 = cse222 + cse224;
  const amrex::Real cse226 = 1.2599999999999999e-13*cse67;
  const amrex::Real cse227 = cse226*state.xn[6];
  const amrex::Real cse228 = 1.7599999999999999e-9*cse96;
  const amrex::Real cse229 = cse228*cse98;
  const amrex::Real cse230 = cse187*cse189;
  const amrex::Real cse231 = 2.4500000000000003e-11*cse230;
  const amrex::Real cse232 = cse29*state.xn[0];
  const amrex::Real cse233 = 2*cse21;
  const amrex::Real cse234 = cse233*cse80;
  const amrex::Real cse235 = cse35*cse9;
  const amrex::Real cse236 = cse15*cse235 + 2*cse39;
  const amrex::Real cse237 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse15 - cse236
));
  const amrex::Real cse238 = cse50*(-cse233 - cse237);
  const amrex::Real cse239 = cse22*cse238;
  const amrex::Real cse240 = cse30*((cse28) ? (
   cse54*(cse234 + cse237*cse80 + cse238*cse49 + cse239*cse31)
)
: (
   0
));
  const amrex::Real cse241 = ((cse28) ? (
   cse53
)
: (
   0.0
));
  const amrex::Real cse242 = cse241*state.xn[3];
  const amrex::Real cse243 = d2g*state.xn[0];
  const amrex::Real cse244 = 1.2000000000000001e-14*cse243;
  const amrex::Real cse245 = cse233*cse87;
  const amrex::Real cse246 = ((cse28) ? (
   cse59*(cse237*cse87 + cse238*cse6 + cse239*cse25 + cse245)
)
: (
   0
));
  const amrex::Real cse247 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse248 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse247)/cse247;
  const amrex::Real cse249 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse250 = chi*cse248*cse249;
  const amrex::Real cse251 = cse93*std::pow(state.T, -1.3);
  const amrex::Real cse252 = 4.2899999999999994e-13*cse251;
  const amrex::Real cse253 = cse62 - 5.0213189999997486e-24*cse64;
  const amrex::Real cse254 = cse108*cse112*state.xn[3];
  const amrex::Real cse255 = 1.6000000000000001e-9*cse162;
  const amrex::Real cse256 = cse255*state.xn[17];
  const amrex::Real cse257 = cse118*cse254;
  const amrex::Real cse258 = cse115*cse256 - cse115*cse257 + cse216*cse254;
  const amrex::Real cse259 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse260 = -2.2999999999999998*cse11 + cse259;
  const amrex::Real cse261 = -cse260;
  const amrex::Real cse262 = cse185*cse261;
  const amrex::Real cse263 = cse170*cse208;
  const amrex::Real cse264 = -cse262 - cse263;
  const amrex::Real cse265 = cse10*((cse20) ? (
   0
)
: (
   -cse236
));
  const amrex::Real cse266 = cse50*(-cse233 - cse265);
  const amrex::Real cse267 = cse22*cse266;
  const amrex::Real cse268 = ((cse28) ? (
   cse54*(cse234 + cse265*cse80 + cse266*cse49 + cse267*cse31)
)
: (
   0
));
  const amrex::Real cse269 = ((cse28) ? (
   cse59*(cse245 + cse25*cse267 + cse265*cse87 + cse266*cse6)
)
: (
   0
));
  const amrex::Real cse270 = cse228*cse97*state.xn[3];
  const amrex::Real cse271 = cse62 - 5.0199525924432485e-24*cse64;
  const amrex::Real cse272 = -cse231;
  const amrex::Real cse273 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse274 = -cse259;
  const amrex::Real cse275 = cse185*cse274;
  const amrex::Real cse276 = cse272 + cse273 + cse275;
  const amrex::Real cse277 = cse62 - 9.9697094999995006e-24*cse64;
  const amrex::Real cse278 = cse62 - 9.9683430924430005e-24*cse64;
  const amrex::Real cse279 = cse226*state.xn[3];
  const amrex::Real cse280 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse281 = 1.3500000000000001e-9*cse280;
  const amrex::Real cse282 = cse281*state.xn[15];
  const amrex::Real cse283 = cse279 + cse282;
  const amrex::Real cse284 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse285 = cse121*cse284;
  const amrex::Real cse286 = cse62 - 2.9917094999998509e-23*cse64;
  const amrex::Real cse287 = cse62 - 2.991572859244201e-23*cse64;
  const amrex::Real cse288 = 6.5199999999999996*crate;
  const amrex::Real cse289 = cse62 - 6.976853999999652e-23*cse64;
  const amrex::Real cse290 = cse62 - 7.2277833092439893e-23*cse64;
  const amrex::Real cse291 = cse221*state.xn[2];
  const amrex::Real cse292 = cse195 + cse291;
  const amrex::Real cse293 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse294 = cse168*cse293;
  const amrex::Real cse295 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse296 = cse62 - 3.9851444999998012e-23*cse64;
  const amrex::Real cse297 = cse103*state.xn[1];
  const amrex::Real cse298 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse299 = cse297 + cse298;
  const amrex::Real cse300 = cse62 - 6.9955604999996505e-23*cse64;
  const amrex::Real cse301 = cse62 - 6.9954238592440006e-23*cse64;
  const amrex::Real cse302 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse303 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse304 = cse0*state.xn[0];
  const amrex::Real cse305 = -cse304;
  const amrex::Real cse306 = cse62 - 3.242775449999838e-23*cse64;
  const amrex::Real cse307 = cse195 + cse205;
  const amrex::Real cse308 = cse62 - 4.2362104499997889e-23*cse64;
  const amrex::Real cse309 = cse281*state.xn[6];
  const amrex::Real cse310 = cse120*cse284*state.xn[7];
  const amrex::Real cse311 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse312 = cse309 + cse310 + cse311;
  const amrex::Real cse313 = 3*cse21;
  const amrex::Real cse314 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse315 = cse50*(-cse313 - cse314);
  const amrex::Real cse316 = cse22*cse315;
  const amrex::Real cse317 = cse30*((cse28) ? (
   cse54*(cse31*cse316 + cse313*cse80 + cse314*cse80 + cse315*cse49)
)
: (
   0
));
  const amrex::Real cse318 = ((cse28) ? (
   cse59*(cse25*cse316 + cse313*cse87 + cse314*cse87 + cse315*cse6)
)
: (
   0
));
  const amrex::Real cse319 = cse62 - 7.5306120924431232e-24*cse64;
  const amrex::Real cse320 = cse293*state.xn[11];
  const amrex::Real cse321 = cse162*cse320;
  const amrex::Real cse322 = 6.8999999999999995*cse36 + 4.5*cse38;
  const amrex::Real cse323 = -cse322;
  const amrex::Real cse324 = -cse185*cse323 + 3.6750000000000002e-11*cse230;
  const amrex::Real cse325 = cse321 + cse324;
  const amrex::Real cse326 = cse175*state.xn[0];
  const amrex::Real cse327 = cse255*state.xn[3];
  const amrex::Real cse328 = cse115*cse327;
  const amrex::Real cse329 = cse62 - 3.9850078592441519e-23*cse64;
  const amrex::Real cse330 = cse105 - cse84 + cse91;
  const amrex::Real cse331 = cse173*(cse135 + cse140 - cse141 - cse161 - cse330 - cse69);
  const amrex::Real cse332 = crate*cse9;
  const amrex::Real cse333 = cse184*cse332;
  const amrex::Real cse334 = -cse333;
  const amrex::Real cse335 = std::pow(cse3, 0.00231);
  const amrex::Real cse336 = std::exp(-8110.0*cse3);
  const amrex::Real cse337 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse338 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse339 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse340 = 0.00015100000000000001*cse336 + 1.31e-7*cse337 + 3.4e-8*cse338 + 6.9699999999999997e-9*cse339;
  const amrex::Real cse341 = std::pow(state.T, -1.5);
  const amrex::Real cse342 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse335 - 1.5*cse340*std::pow(state.T, -2.5) + cse341*(1.2246100000000002*cse336*cse66 + 3.4846000000000001e-6*cse337*cse66 + 2.5908000000000001e-7*cse338*cse66 + 9.6185999999999982e-9*cse339*cse66));
  const amrex::Real cse343 = -cse342;
  const amrex::Real cse344 = 6.7636511339243933e-7*cse89;
  const amrex::Real cse345 = cse11*state.xn[2];
  const amrex::Real cse346 = cse345 < 1.0e-50;
  const amrex::Real cse347 = ((cse346) ? (
   10000000000.0
)
: (
   cse110
));
  const amrex::Real cse348 = cse347 + 1;
  const amrex::Real cse349 = 1.0/cse348;
  const amrex::Real cse350 = cse347*cse349;
  const amrex::Real cse351 = cse350*state.xn[16];
  const amrex::Real cse352 = cse169*cse351;
  const amrex::Real cse353 = ((cse346) ? (
   0
)
: (
   5.6603773584905654e-5*cse107*state.xn[3]
));
  const amrex::Real cse354 = cse320*state.xn[16];
  const amrex::Real cse355 = cse349*cse354;
  const amrex::Real cse356 = cse353*cse355;
  const amrex::Real cse357 = -cse356;
  const amrex::Real cse358 = cse347*cse354/std::pow(cse348, 2);
  const amrex::Real cse359 = cse353*cse358;
  const amrex::Real cse360 = std::exp(-15230.0*cse3);
  const amrex::Real cse361 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse362 = std::exp(-12.17*cse3);
  const amrex::Real cse363 = 1.6339999999999999e-6*cse360 + 9.7930000000000004e-9*cse361 + 6.3460000000000002e-9*cse362;
  const amrex::Real cse364 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse365 = std::exp(-49550.0*cse3);
  const amrex::Real cse366 = 0.15970000000000001*cse365;
  const amrex::Real cse367 = cse366 + 1.7848999999999999;
  const amrex::Real cse368 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse369 = 0.21509999999999996 - cse366;
  const amrex::Real cse370 = std::pow(cse364, -cse367)*std::pow(cse368, -cse369);
  const amrex::Real cse371 = 7913.1350000000002*cse365*cse66;
  const amrex::Real cse372 = 2.4460185352936309e-10*cse370*cse85;
  const amrex::Real cse373 = state.xn[2]*state.xn[8]*(-3.0/2.0*cse363/std::pow(state.T, 5.0/2.0) - 1.2230092676468155e-10*cse370*cse86 + cse372*(-cse371*std::log(cse364) - 0.00035870183966449452*cse367*cse85/cse364) + cse372*(cse371*std::log(cse368) - 6.1221939997287613*cse369*cse85/cse368) + cse86*(0.024885819999999999*cse360*cse66 + 7.2272340000000005e-7*cse361*cse66 + 7.7230820000000007e-8*cse362*cse66));
  const amrex::Real cse374 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse375 = cse147*cse8;
  const amrex::Real cse376 = 0.044330000000000001*cse375 - 0.28520000000000001;
  const amrex::Real cse377 = cse375*cse376 - 1.6759999999999999;
  const amrex::Real cse378 = cse375*cse377 + 11.19;
  const amrex::Real cse379 = cse378*state.xn[6];
  const amrex::Real cse380 = 4.9999999999999997e-12*cse341*cse379*state.xn[2];
  const amrex::Real cse381 = cse102*cse94;
  const amrex::Real cse382 = 7.5899999999999998e-12*cse381;
  const amrex::Real cse383 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse384 = cse383*state.xn[2];
  const amrex::Real cse385 = cse384*state.xn[6]*(cse375*(0.044330000000000001*cse147*cse75 + cse376*cse77) + cse377*cse77);
  const amrex::Real cse386 = std::pow(cse144, 1.8740000000000001);
  const amrex::Real cse387 = 7.538e-5*cse147 + 0.89639999999999997;
  const amrex::Real cse388 = std::pow(cse146, -cse387);
  const amrex::Real cse389 = cse388*std::pow(state.T, 1.635e-6);
  const amrex::Real cse390 = 43750.0*cse389;
  const amrex::Real cse391 = cse390 + 1;
  const amrex::Real cse392 = cse386*cse391;
  const amrex::Real cse393 = -cse387;
  const amrex::Real cse394 = 1.534817233071468e-7*cse386;
  const amrex::Real cse395 = cse391*cse394 + 1;
  const amrex::Real cse396 = 2.1659999999999999e-12*cse159*state.xn[13]/std::pow(cse395, 2);
  const amrex::Real cse397 = cse396*(-1.4381237473879656e-7*cse3*cse392 - cse394*(0.071531250000000005*cse388*std::pow(state.T, -0.99999836499999994) + cse390*(cse155*cse393 - 7.538e-5*cse156)));
  const amrex::Real cse398 = std::pow(cse144, 1.512);
  const amrex::Real cse399 = 5.4939999999999998e-7*cse147 + 0.49559999999999998;
  const amrex::Real cse400 = std::pow(cse146, -cse399);
  const amrex::Real cse401 = cse400*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse402 = 5115.0*cse401;
  const amrex::Real cse403 = cse402 + 1;
  const amrex::Real cse404 = cse398*cse403;
  const amrex::Real cse405 = -cse399;
  const amrex::Real cse406 = 7.104740267838494e-7*cse398;
  const amrex::Real cse407 = cse403*cse406 + 1;
  const amrex::Real cse408 = 5.5720000000000001e-12*cse159*state.xn[6]/std::pow(cse407, 2);
  const amrex::Real cse409 = cse408*(-5.3711836424859012e-7*cse3*cse404 - cse406*(0.0019963845000000001*cse400*std::pow(state.T, -0.99999960970000001) + cse402*(cse155*cse405 - 5.4939999999999998e-7*cse156)));
  const amrex::Real cse410 = std::pow(cse144, 1.1279999999999999);
  const amrex::Real cse411 = 0.00013329999999999999*cse147 + 0.81200000000000006;
  const amrex::Real cse412 = std::pow(cse146, -cse411);
  const amrex::Real cse413 = cse412*std::pow(state.T, 0.04845);
  const amrex::Real cse414 = 433.10000000000002*cse413;
  const amrex::Real cse415 = cse414 + 1;
  const amrex::Real cse416 = cse410*cse415;
  const amrex::Real cse417 = -cse411;
  const amrex::Real cse418 = 0.011078790689599145*cse410;
  const amrex::Real cse419 = cse415*cse418 + 1;
  const amrex::Real cse420 = 4.5579999999999993e-11*cse159*state.xn[8]/std::pow(cse419, 2);
  const amrex::Real cse421 = cse420*(-0.0062484379489339175*cse3*cse416 - cse418*(20.983695000000001*cse412*std::pow(state.T, -0.95155000000000001) + cse414*(cse155*cse417 - 0.00013329999999999999*cse156)));
  const amrex::Real cse422 = cse173*(cse119 + cse165 + cse172 + cse343 + cse344 + cse352 + cse357 + cse359 - cse373 + cse374 + cse380 - cse382 - cse385 - cse397 - cse409 - cse421 + 4.2899999999999994e-13*cse95);
  const amrex::Real cse423 = 1.0/cse407;
  const amrex::Real cse424 = 5.5720000000000001e-12*cse423;
  const amrex::Real cse425 = d2g*state.xn[6];
  const amrex::Real cse426 = cse424*cse425;
  const amrex::Real cse427 = 1.0/cse419;
  const amrex::Real cse428 = 4.5579999999999993e-11*cse427;
  const amrex::Real cse429 = d2g*state.xn[8];
  const amrex::Real cse430 = cse428*cse429;
  const amrex::Real cse431 = crate*cse36;
  const amrex::Real cse432 = cse431*state.xn[7];
  const amrex::Real cse433 = 1040.0*cse432;
  const amrex::Real cse434 = cse430 + cse433;
  const amrex::Real cse435 = 1.0/cse395;
  const amrex::Real cse436 = 2.1659999999999999e-12*cse435;
  const amrex::Real cse437 = d2g*state.xn[13];
  const amrex::Real cse438 = cse431*state.xn[12];
  const amrex::Real cse439 = cse436*cse437 + 16800.0*cse438;
  const amrex::Real cse440 = cse426 + cse434 + cse439;
  const amrex::Real cse441 = cse194*cse332;
  const amrex::Real cse442 = cse440 - cse441;
  const amrex::Real cse443 = cse379*cse383 + cse408*(0.0036340746469993896*cse106*cse398*cse401*cse405 + 1.0742367284971802e-6*cse106*cse404);
  const amrex::Real cse444 = cse363*cse86 + cse372;
  const amrex::Real cse445 = cse420*(4.7982242476653898*cse106*cse410*cse413*cse417 + 0.012496875897867835*cse106*cse416) + cse444*state.xn[8];
  const amrex::Real cse446 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse447 = cse396*(0.0067148253946876721*cse106*cse386*cse389*cse393 + 2.8762474947759312e-7*cse106*cse392) + cse446*state.xn[13];
  const amrex::Real cse448 = cse218 + cse220;
  const amrex::Real cse449 = 1.3007021411393064e-6*cse210;
  const amrex::Real cse450 = ((cse346) ? (
   0
)
: (
   -0.00011320754716981131*cse213*state.xn[3]
));
  const amrex::Real cse451 = cse355*cse450;
  const amrex::Real cse452 = cse358*cse450;
  const amrex::Real cse453 = cse451 - cse452;
  const amrex::Real cse454 = cse209 + cse449*state.xn[16] + cse453;
  const amrex::Real cse455 = 2*cse179;
  const amrex::Real cse456 = cse261*cse332;
  const amrex::Real cse457 = crate*cse11;
  const amrex::Real cse458 = 16800.0*cse457;
  const amrex::Real cse459 = cse435*cse437;
  const amrex::Real cse460 = 33600.0*cse438 + 4.3319999999999997e-12*cse459;
  const amrex::Real cse461 = -cse458*state.xn[12] + cse460;
  const amrex::Real cse462 = 3.2999999999999996e-13*cse251;
  const amrex::Real cse463 = cse462*state.xn[8];
  const amrex::Real cse464 = 1040.0*cse457;
  const amrex::Real cse465 = cse427*cse429;
  const amrex::Real cse466 = 2080.0*cse432 + 9.1159999999999986e-11*cse465;
  const amrex::Real cse467 = -cse464*state.xn[7] + cse466;
  const amrex::Real cse468 = cse463 + cse467;
  const amrex::Real cse469 = ((cse346) ? (
   0
)
: (
   cse109
));
  const amrex::Real cse470 = cse355*cse469;
  const amrex::Real cse471 = cse358*cse469;
  const amrex::Real cse472 = cse470 - cse471;
  const amrex::Real cse473 = cse258 + cse472;
  const amrex::Real cse474 = cse256 - cse257;
  const amrex::Real cse475 = cse423*cse425;
  const amrex::Real cse476 = 1.1144e-11*cse475;
  const amrex::Real cse477 = cse231 + cse476;
  const amrex::Real cse478 = cse274*cse332;
  const amrex::Real cse479 = -cse478;
  const amrex::Real cse480 = 1.1000000000000001*crate;
  const amrex::Real cse481 = cse159*cse424 + cse378*cse384;
  const amrex::Real cse482 = 1.0537934368240927e-9*cse335 + cse340*cse341;
  const amrex::Real cse483 = cse482*state.xn[16];
  const amrex::Real cse484 = 2.808e-22*ncol_H2;
  const amrex::Real cse485 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse484)*std::exp(-1.6000000000000001e-17*ncol_C)/(cse484 + 1) + 3.8500000000000001*crate + cse464*state.xn[3];
  const amrex::Real cse486 = cse462*state.xn[3];
  const amrex::Real cse487 = cse159*cse428 + cse444*state.xn[2];
  const amrex::Real cse488 = cse486 + cse487;
  const amrex::Real cse489 = cse293*cse351;
  const amrex::Real cse490 = cse294 + cse489;
  const amrex::Real cse491 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse492 = cse491*state.rn[2*0];
  const amrex::Real cse493 = cse458*state.xn[3] + cse492;
  const amrex::Real cse494 = cse159*cse436 + cse446*state.xn[2];
  const amrex::Real cse495 = cse195 + cse442;
  const amrex::Real cse496 = 1.6715999999999999e-11*cse475;
  const amrex::Real cse497 = cse482*state.xn[7];
  const amrex::Real cse498 = cse320*cse350;
  const amrex::Real cse499 = cse449*state.xn[2];
  const amrex::Real cse500 = cse323*cse332;
  const amrex::Real cse501 = -cse500;
  const amrex::Real cse502 = 1.3673999999999998e-10*cse465;
  const amrex::Real cse503 = 3120.0*cse432;
  const amrex::Real cse504 = cse502 + cse503;
  const amrex::Real cse505 = 50400.0*cse438 + 6.4979999999999992e-12*cse459;
  const amrex::Real cse506 = cse327 + cse328;
  const amrex::Real cse507 = cse491*state.xn[12];
  const amrex::Real cse508 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse509 = cse29*state.xn[3];
  const amrex::Real cse510 = state.xn[0]*state.xn[3];
  const amrex::Real cse511 = cse510*cse60;
  const amrex::Real cse512 = cse30*cse55;
  const amrex::Real cse513 = -cse70;
  const amrex::Real cse514 = cse30*cse83;
  const amrex::Real cse515 = cse510*cse88;
  const amrex::Real cse516 = cse382 - 4.2899999999999994e-13*cse92*cse93*state.xn[3]*state.xn[8];
  const amrex::Real cse517 = cse164 - cse166;
  const amrex::Real cse518 = -cse100 + cse101;
  const amrex::Real cse519 = cse171 + cse343 + cse352 + cse357 + cse359 + cse518;
  const amrex::Real cse520 = cse173*(cse165 - cse513 - cse514 - cse515 - cse516 - cse517 - cse519 - cse68 - 2.3622303960273254e-7*cse89);
  const amrex::Real cse521 = cse203*cse510;
  const amrex::Real cse522 = cse201 + cse441 + cse508 + cse521;
  const amrex::Real cse523 = 4.5427507615910105e-7*cse210;
  const amrex::Real cse524 = -cse218 + cse453;
  const amrex::Real cse525 = cse217 - cse219;
  const amrex::Real cse526 = 5.6999999999999997e-11*cse250;
  const amrex::Real cse527 = cse246*cse510;
  const amrex::Real cse528 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse529 = cse227 + cse528;
  const amrex::Real cse530 = cse455 + cse456;
  const amrex::Real cse531 = cse229 + cse263 - cse470 + cse471;
  const amrex::Real cse532 = cse258 + cse474;
  const amrex::Real cse533 = cse269*cse510;
  const amrex::Real cse534 = cse268*cse30;
  const amrex::Real cse535 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse536 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse537 = 9.0000000000000011e-15*cse243;
  const amrex::Real cse538 = cse318*cse510;
  const amrex::Real cse539 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse540 = cse321 + cse497 + cse498 + cse539;
  const amrex::Real cse541 = -cse173*cse518;
  const amrex::Real cse542 = -cse73;
  const amrex::Real cse543 = -cse380 + cse385 + cse409 + cse542 + cse68;
  const amrex::Real cse544 = cse173*cse543;
  const amrex::Real cse545 = cse476 + cse529;
  const amrex::Real cse546 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse547 = cse283 + cse481 + cse535 + cse546;
  const amrex::Real cse548 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse549 = cse309 + cse426;
  const amrex::Real cse550 = -cse173*cse543;
  const amrex::Real cse551 = cse373 + cse421;
  const amrex::Real cse552 = cse173*(cse123 + cse343 + 2.2769999999999997e-12*cse381 + cse551 + cse70 - 1.287e-13*cse95);
  const amrex::Real cse553 = cse431*state.xn[9];
  const amrex::Real cse554 = 184.0*cse553;
  const amrex::Real cse555 = -cse554;
  const amrex::Real cse556 = cse434 + cse555;
  const amrex::Real cse557 = 9.8999999999999995e-14*cse251;
  const amrex::Real cse558 = 184.0*cse457;
  const amrex::Real cse559 = cse558*state.xn[9];
  const amrex::Real cse560 = 368.0*cse553;
  const amrex::Real cse561 = -cse560;
  const amrex::Real cse562 = cse559 + cse561;
  const amrex::Real cse563 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av) + cse558*state.xn[3];
  const amrex::Real cse564 = cse303 + cse304;
  const amrex::Real cse565 = cse310 + cse554;
  const amrex::Real cse566 = 552.0*cse553;
  const amrex::Real cse567 = 1.1994963750000002e-8*cse72*state.xn[8];
  const amrex::Real cse568 = -cse567;
  const amrex::Real cse569 = cse173*(-cse516 - cse551 - cse568);
  const amrex::Real cse570 = 9.1500000000000005e-10*cse280;
  const amrex::Real cse571 = cse570*state.xn[15];
  const amrex::Real cse572 = cse570*state.xn[8];
  const amrex::Real cse573 = -cse124*cse173;
  const amrex::Real cse574 = cse288 + cse536;
  const amrex::Real cse575 = cse548 + cse563;
  const amrex::Real cse576 = cse302 + cse554;
  const amrex::Real cse577 = cse173*(cse119 + cse568);
  const amrex::Real cse578 = -cse352 + cse356 - cse359;
  const amrex::Real cse579 = cse165 + cse330;
  const amrex::Real cse580 = cse173*(-cse578 - cse579);
  const amrex::Real cse581 = -cse374 + cse397;
  const amrex::Real cse582 = cse173*cse581;
  const amrex::Real cse583 = -cse507;
  const amrex::Real cse584 = -cse173*cse581;
  const amrex::Real cse585 = cse173*(cse342 + 5.3129999999999997e-12*cse381 + cse513 - 3.0029999999999999e-13*cse95);
  const amrex::Real cse586 = 2.3099999999999997e-13*cse251;
  const amrex::Real cse587 = cse173*(cse123 + cse517 + cse567 + cse578 + cse73);
  const amrex::Real cse588 = cse173*(cse344 + cse519);
  const amrex::Real cse589 = cse173*(cse167 + cse542 + cse579);
  const amrex::Real cse590 = 1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse591 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse592 = 100.0*d2g;
  const amrex::Real cse593 = cse591*cse592;
  const amrex::Real cse594 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse143 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse595 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse596 = cse159*cse595;
  const amrex::Real cse597 = cse594*cse596;
  const amrex::Real cse598 = 2.1789607199999999e-11*cse181;
  const amrex::Real cse599 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse600 = cse146 + 50;
  const amrex::Real cse601 = 0.0043*std::pow(cse600, 0.69199999999999995) + 1;
  const amrex::Real cse602 = 0.049959999999999997*std::pow(cse600, 0.43099999999999999)*cse601 + 1;
  const amrex::Real cse603 = 1.0/cse602;
  const amrex::Real cse604 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse605 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse606 = 1.0/cse605;
  const amrex::Real cse607 = cse604*cse606;
  const amrex::Real cse608 = cse606*state.xn[2];
  const amrex::Real cse609 = cse604*cse608;
  const amrex::Real cse610 = 4.4460389837719569e-9*cse609 + 626500000.0;
  const amrex::Real cse611 = 1.0/cse610;
  const amrex::Real cse612 = std::exp(-118400.0*cse3);
  const amrex::Real cse613 = cse611*cse612;
  const amrex::Real cse614 = 1.3338116951315871e-8*cse613;
  const amrex::Real cse615 = cse609*cse614 + 1;
  const amrex::Real cse616 = 1.0/cse615;
  const amrex::Real cse617 = cse613*cse616;
  const amrex::Real cse618 = 1.3659996292638161e-10*cse617;
  const amrex::Real cse619 = cse607*cse618;
  const amrex::Real cse620 = state.xn[2] + 1.0e-50;
  const amrex::Real cse621 = cse143/cse620;
  const amrex::Real cse622 = 1.7*cse2;
  const amrex::Real cse623 = cse621*cse622 + 50;
  const amrex::Real cse624 = std::log(cse623);
  const amrex::Real cse625 = 0.45350000000000001 + 2.234/cse624;
  const amrex::Real cse626 = std::pow(state.T, cse625);
  const amrex::Real cse627 = 1.4419999999999999 - 0.050889999999999998*cse624;
  const amrex::Real cse628 = std::exp(cse624*cse627);
  const amrex::Real cse629 = 1.8998126555754159e-29*cse626*cse628;
  const amrex::Real cse630 = cse159*cse629;
  const amrex::Real cse631 = cse630*state.xn[2];
  const amrex::Real cse632 = 0.001*state.T;
  const amrex::Real cse633 = cse632 + 1;
  const amrex::Real cse634 = std::pow(cse633, -2);
  const amrex::Real cse635 = 1.0/cse633;
  const amrex::Real cse636 = std::pow(10, -2.0230000000000001*cse634 - 3.6709999999999998*cse635 - 11.08);
  const amrex::Real cse637 = cse11*cse636;
  const amrex::Real cse638 = cse637*state.xn[3];
  const amrex::Real cse639 = std::pow(10, -2.3900000000000001*cse634 + 0.055500000000000001*cse635 - 11.06);
  const amrex::Real cse640 = cse11*cse639;
  const amrex::Real cse641 = cse640*state.xn[0];
  const amrex::Real cse642 = cse638 + cse641;
  const amrex::Real cse643 = 1.0/cse642;
  const amrex::Real cse644 = cse526 + 1.9999999999999999e-7;
  const amrex::Real cse645 = cse643*cse644;
  const amrex::Real cse646 = cse11*cse645 + 1;
  const amrex::Real cse647 = 1.0/cse646;
  const amrex::Real cse648 = 6.7291434000000005e-12*cse647 + 3.2043540000000002e-13;
  const amrex::Real cse649 = 2.5634832e-11*cse647 + 6.4087080000000003e-13;
  const amrex::Real cse650 = cse71*state.xn[2];
  const amrex::Real cse651 = (1.0/100.0)*state.T;
  const amrex::Real cse652 = std::log(cse651);
  const amrex::Real cse653 = 0.0086999999999999994*cse652;
  const amrex::Real cse654 = std::pow(cse651, cse653 + 0.12809999999999999);
  const amrex::Real cse655 = 7.5799999999999997e-10*cse654*state.xn[0];
  const amrex::Real cse656 = state.T < 500;
  const amrex::Real cse657 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse658 = 0.75*((cse656) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse657
)) + 0.25*((cse656) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse657
));
  const amrex::Real cse659 = 4.5299999999999998e-6*cse650 + cse655 + cse658*state.xn[3];
  const amrex::Real cse660 = cse659 + 2.3e-6;
  const amrex::Real cse661 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse662 = cse661/cse660;
  const amrex::Real cse663 = 2*cse662;
  const amrex::Real cse664 = cse659*cse663 + 1;
  const amrex::Real cse665 = 1.0/cse664;
  const amrex::Real cse666 = cse665*state.xn[8];
  const amrex::Real cse667 = cse659*cse662;
  const amrex::Real cse668 = 5.7927337833400002e-20*cse667;
  const amrex::Real cse669 = cse345 + 0.070000000000000007;
  const amrex::Real cse670 = 1.0/cse669;
  const amrex::Real cse671 = std::sqrt(cse345*cse670);
  const amrex::Real cse672 = 26.399999999999999*cse671;
  const amrex::Real cse673 = cse672 + 6.5;
  const amrex::Real cse674 = cse11*cse673;
  const amrex::Real cse675 = cse8*std::log(cse10);
  const amrex::Real cse676 = cse675 < 2;
  const amrex::Real cse677 = cse675 < 4;
  const amrex::Real cse678 = cse675 < 7;
  const amrex::Real cse679 = cse675 < 10;
  const amrex::Real cse680 = ((cse676) ? (
   10.0
)
: ((cse677) ? (
   (3.0/2.0)*cse675 + 7
)
: ((cse678) ? (
   (4.0/3.0)*cse675 + 23.0/3.0
)
: ((cse679) ? (
   (1.0/3.0)*cse675 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse681 = 2.0*cse11;
  const amrex::Real cse682 = cse680*cse681;
  const amrex::Real cse683 = cse674*state.xn[0] + cse682*state.xn[3];
  const amrex::Real cse684 = 1.0/cse178;
  const amrex::Real cse685 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse686 = 680.10000000000002/cse685;
  const amrex::Real cse687 = 1.75*cse685;
  const amrex::Real cse688 = cse686*state.xn[2] + cse687*state.xn[0] + state.xn[3];
  const amrex::Real cse689 = std::exp(-state.T);
  const amrex::Real cse690 = 1 - cse689;
  const amrex::Real cse691 = std::pow(cse690, 1000);
  const amrex::Real cse692 = l0_co_interp1d(state.T);
  const amrex::Real cse693 = 1.0/cse692;
  const amrex::Real cse694 = M_SQRT2;
  const amrex::Real cse695 = std::max(gradv, 5.7445531974402559e-18*cse2*cse694);
  const amrex::Real cse696 = 100000.0/cse695;
  const amrex::Real cse697 = ncol_H <= 0;
  const amrex::Real cse698 = cse11*ncol_H;
  const amrex::Real cse699 = std::max(1.0000000000000001e-30, cse698);
  const amrex::Real cse700 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse701 = ((cse697) ? (
   cse696*state.xn[9]
)
: (
   cse700/std::sqrt(cse699)
));
  const amrex::Real cse702 = llte_co_interp2d(state.T, cse701);
  const amrex::Real cse703 = 1.0/cse702;
  const amrex::Real cse704 = nhalf_co_interp2d(state.T, cse701);
  const amrex::Real cse705 = cse692*cse703;
  const amrex::Real cse706 = -cse704*cse705 + 1;
  const amrex::Real cse707 = 1.0/cse704;
  const amrex::Real cse708 = cse688*cse707;
  const amrex::Real cse709 = alpha_co_interp2d(state.T, cse701);
  const amrex::Real cse710 = std::pow(cse708, cse709);
  const amrex::Real cse711 = cse693*cse710;
  const amrex::Real cse712 = cse706*cse711;
  const amrex::Real cse713 = cse688*cse703 + cse693 + cse712;
  const amrex::Real cse714 = 1.0/cse713;
  const amrex::Real cse715 = cse691*cse714;
  const amrex::Real cse716 = cse715*state.xn[9];
  const amrex::Real cse717 = 7.1777529600000012e-12*cse509;
  const amrex::Real cse718 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse719 = 1.08e-14*cse718;
  const amrex::Real cse720 = 0.16*cse652;
  const amrex::Real cse721 = std::pow(cse651, 0.755 - cse720);
  const amrex::Real cse722 = 4.34e-10*cse721;
  const amrex::Real cse723 = cse722*state.xn[0];
  const amrex::Real cse724 = 0.042999999999999997*cse652;
  const amrex::Real cse725 = 5.2499999999999998e-13*std::pow(cse651, cse724 + 0.88900000000000001);
  const amrex::Real cse726 = 0.52500000000000002*cse652;
  const amrex::Real cse727 = 2.2500000000000003e-12*std::pow(cse651, cse726 + 1.198);
  const amrex::Real cse728 = cse725 + cse727;
  const amrex::Real cse729 = cse719*state.xn[2] + cse723 + cse728*state.xn[3];
  const amrex::Real cse730 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse731 = 5.1199999999999999e-10*cse730;
  const amrex::Real cse732 = 0.0030000000000000001*cse652;
  const amrex::Real cse733 = std::pow(cse651, 0.41899999999999998 - cse732);
  const amrex::Real cse734 = 3.5700000000000001e-10*cse733;
  const amrex::Real cse735 = cse734*state.xn[0];
  const amrex::Real cse736 = 0.025000000000000001*cse652;
  const amrex::Real cse737 = 3.7250000000000001e-11*std::pow(cse651, cse736 + 0.26400000000000001);
  const amrex::Real cse738 = 1.0274999999999999e-10*std::pow(cse651, cse724 + 0.29599999999999999);
  const amrex::Real cse739 = cse737 + cse738;
  const amrex::Real cse740 = cse731*state.xn[2] + cse735 + cse739*state.xn[3];
  const amrex::Real cse741 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse742 = (1.0/5.0)*cse741;
  const amrex::Real cse743 = cse740*cse742;
  const amrex::Real cse744 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse745 = cse729*cse744;
  const amrex::Real cse746 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse747 = 4.8599999999999998e-10*cse746;
  const amrex::Real cse748 = 0.0060000000000000001*cse652;
  const amrex::Real cse749 = std::pow(cse651, 0.36899999999999999 - cse748);
  const amrex::Real cse750 = 3.1899999999999998e-10*cse749;
  const amrex::Real cse751 = cse750*state.xn[0];
  const amrex::Real cse752 = 0.041000000000000002*cse652;
  const amrex::Real cse753 = 4.7499999999999998e-11*std::pow(cse651, cse752 + 0.20300000000000001);
  const amrex::Real cse754 = 0.058000000000000003*cse652;
  const amrex::Real cse755 = 1.6725000000000002e-10*std::pow(cse651, cse754 + 0.23699999999999999);
  const amrex::Real cse756 = cse753 + cse755;
  const amrex::Real cse757 = cse747*state.xn[2] + cse751 + cse756*state.xn[3];
  const amrex::Real cse758 = (1.0/15.0)*cse757;
  const amrex::Real cse759 = cse740 + 8.9099999999999997e-5;
  const amrex::Real cse760 = cse742*cse757;
  const amrex::Real cse761 = cse729*cse743 + cse745*cse758 + cse759*cse760;
  const amrex::Real cse762 = cse729 + 1.7499999999999998e-5;
  const amrex::Real cse763 = cse757 + 1.34e-10;
  const amrex::Real cse764 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse765 = cse729*cse764;
  const amrex::Real cse766 = (1.0/3.0)*cse765;
  const amrex::Real cse767 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse768 = cse740*cse767;
  const amrex::Real cse769 = (3.0/5.0)*cse768;
  const amrex::Real cse770 = cse760*cse762 + cse762*cse769 + cse763*cse769;
  const amrex::Real cse771 = cse759*cse762 + cse759*cse763 + cse761 + cse763*cse766 + cse770;
  const amrex::Real cse772 = 1.0/cse771;
  const amrex::Real cse773 = 2.3872025442750958e-19*cse772;
  const amrex::Real cse774 = 2.8023005140019997e-18*cse772;
  const amrex::Real cse775 = cse761*cse773 + cse770*cse774;
  const amrex::Real cse776 = state.T < 100.0;
  const amrex::Real cse777 = state.T < 6000.0;
  const amrex::Real cse778 = state.T < 10;
  const amrex::Real cse779 = std::log(cse632);
  const amrex::Real cse780 = cse779*cse8;
  const amrex::Real cse781 = std::pow(cse7, -8);
  const amrex::Real cse782 = std::pow(cse779, 8)*cse781;
  const amrex::Real cse783 = std::pow(cse7, -7);
  const amrex::Real cse784 = std::pow(cse779, 7)*cse783;
  const amrex::Real cse785 = std::pow(cse7, -6);
  const amrex::Real cse786 = std::pow(cse779, 6)*cse785;
  const amrex::Real cse787 = std::pow(cse7, -5);
  const amrex::Real cse788 = std::pow(cse779, 5)*cse787;
  const amrex::Real cse789 = std::pow(cse7, -4);
  const amrex::Real cse790 = std::pow(cse779, 4)*cse789;
  const amrex::Real cse791 = std::pow(cse7, -3);
  const amrex::Real cse792 = std::pow(cse779, 3);
  const amrex::Real cse793 = cse791*cse792;
  const amrex::Real cse794 = cse13*std::pow(cse779, 2);
  const amrex::Real cse795 = std::pow(10.0, 16.815729999999999*cse780 + 70.609154000000004*cse782 + 364.14445999999998*cse784 + 801.81246999999996*cse786 + 983.67575999999997*cse788 + 734.71650999999997*cse790 + 343.1918*cse793 + 96.743155000000002*cse794 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse796 = state.xn[5] + state.xn[6];
  const amrex::Real cse797 = std::pow(10.0, 1.5714710999999999*cse780 + 0.32168730000000001*cse788 - 0.51002221000000003*cse790 - 0.23619984999999999*cse793 + 0.015391166*cse794 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse780 + 0.19191374999999999*cse788 - 0.16596184*cse790 + 0.29036281000000003*cse793 - 0.81520437999999995*cse794 - 23.689236999999999)*cse796 + std::pow(10.0, 2.0943374000000001*cse779*cse8 - 0.033638326000000003*cse788 - 0.14913216000000001*cse790 + 0.43693353000000001*cse791*cse792 - 0.77151435999999995*cse794 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse798 = std::pow(10, 37.383713*cse780 + 3.8479610000000002*cse788 + 20.159831000000001*cse790 + 48.656103000000002*cse793 + 58.145166000000003*cse794 - 16.818342000000001)*state.xn[0] + cse797;
  const amrex::Real cse799 = std::pow(10.0, 1.6802758*cse780 - 6.3701156000000001*cse782 + 6.4380698000000001*cse784 + 8.9167182999999994*cse786 - 8.8077017000000009*cse788 - 4.7274035999999997*cse790 + 4.0406627000000004*cse793 + 0.93310621999999999*cse794 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse800 = state.T < 1000.0;
  const amrex::Real cse801 = std::pow(10.0, 3.5692468000000002*cse779*cse8 - 4.2519023000000002*cse788 - 21.328264000000001*cse790 - 27.850082*cse793 - 11.33286*cse794 - 24.311209000000002)*state.xn[0] + cse797;
  const amrex::Real cse802 = std::pow(10.0, 4.6450521*cse780 + 1.5538288*cse788 - 5.5108049000000001*cse790 + 5.9369081000000001*cse793 - 3.7209846*cse794 - 24.311209000000002)*state.xn[0] + cse797;
  const amrex::Real cse803 = std::pow(10.0, -11.945856385319289*cse13 + 28.69470131298657*cse787 - 56.79810800956944*cse789 + 34.150687525022562*cse791 + 8.3228160852326614*cse8 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.5440891133599637*cse787 - 1.7105157385968939*cse789 + 1.6702447513374001*cse791 + 3.9225864834863131*cse8 - 23.689236999999999)*cse796 + std::pow(10.0, -2.4768712409536775*cse13 - 0.62120210234156448*cse787 - 1.5370575959566981*cse789 + 2.5133588394664677*cse791 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 5.9406293599919788*cse787 - 5.2566362076906961*cse789 - 1.3586848802337363*cse791 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse804 = ((cse656 && cse776 && cse777) ? (
   cse778 || cse795 + cse798 < 1.0e-100
)
: (
   ((cse776 && cse777) ? (
      cse778 || cse798 + cse799 < 1.0e-100
   )
   : (
      ((cse656 && cse777 && cse800) ? (
         cse778 || cse795 + cse801 < 1.0e-100
      )
      : (
         ((cse777 && cse800) ? (
            cse778 || cse799 + cse801 < 1.0e-100
         )
         : (
            ((cse656 && cse777) ? (
               cse778 || cse795 + cse802 < 1.0e-100
            )
            : (
               ((cse777) ? (
                  cse778 || cse799 + cse802 < 1.0e-100
               )
               : (
                  ((cse656) ? (
                     cse778 || std::pow(10.0, 310.58441786958315*cse13 + 7500.646099484944*cse781 + 21588.957206746247*cse783 + 26530.843406310876*cse785 + 18165.631967965237*cse787 + 7572.4886742758581*cse789 + 1974.131269262874*cse791 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse803 < 1.0e-100
                  )
                  : (
                     cse778 || std::pow(10.0, 2.9956460707652877*cse13 - 676.68255490510751*cse781 + 381.69251128040054*cse783 + 295.04162851880631*cse785 - 162.65264843551881*cse787 - 48.723840736518795*cse789 + 23.242975457496804*cse791 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse803 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse805 = ((cse777) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse806 = std::exp(-11700.0*cse805);
  const amrex::Real cse807 = std::exp(-5860.0*cse805);
  const amrex::Real cse808 = std::exp(-510.0*cse805);
  const amrex::Real cse809 = std::exp(-2197000.0*((cse777) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse810 = cse809*((cse777) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse811 = 6.0142468035272636e-8*((cse777) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse812 = 1.0/cse811;
  const amrex::Real cse813 = 1.6e-18*cse806 + 6.7e-19*cse807 + 3.0e-24*cse808 + 4.985670872372847e-33*cse810*cse812;
  const amrex::Real cse814 = std::log(0.001*((cse777) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse815 = std::pow(cse814, 5);
  const amrex::Real cse816 = cse787*cse815;
  const amrex::Real cse817 = std::pow(cse814, 4);
  const amrex::Real cse818 = cse789*cse817;
  const amrex::Real cse819 = std::pow(cse814, 3);
  const amrex::Real cse820 = std::pow(cse814, 2);
  const amrex::Real cse821 = cse13*cse820;
  const amrex::Real cse822 = std::pow(10.0, 0.43693353000000001*cse791*cse819 + 2.0943374000000001*cse8*cse814 - 0.033638326000000003*cse816 - 0.14913216000000001*cse818 - 0.77151435999999995*cse821 - 23.962112000000001);
  const amrex::Real cse823 = cse822*state.xn[3];
  const amrex::Real cse824 = cse8*cse814;
  const amrex::Real cse825 = cse791*cse819;
  const amrex::Real cse826 = std::pow(10.0, 0.32168730000000001*cse816 - 0.51002221000000003*cse818 + 0.015391166*cse821 + 1.5714710999999999*cse824 - 0.23619984999999999*cse825 - 22.089523);
  const amrex::Real cse827 = cse826*state.xn[1];
  const amrex::Real cse828 = std::pow(10.0, 0.19191374999999999*cse816 - 0.16596184*cse818 - 0.81520437999999995*cse821 + 2.1892372*cse824 + 0.29036281000000003*cse825 - 23.689236999999999);
  const amrex::Real cse829 = cse796*cse828;
  const amrex::Real cse830 = std::pow(10, 3.8479610000000002*cse816 + 20.159831000000001*cse818 + 58.145166000000003*cse821 + 37.383713*cse824 + 48.656103000000002*cse825 - 16.818342000000001);
  const amrex::Real cse831 = ((cse777) ? (
   cse776
)
: (
   false
));
  const amrex::Real cse832 = std::pow(10.0, 3.5692468000000002*cse8*cse814 - 4.2519023000000002*cse816 - 21.328264000000001*cse818 - 11.33286*cse821 - 27.850082*cse825 - 24.311209000000002);
  const amrex::Real cse833 = ((cse777) ? (
   cse800
)
: (
   false
));
  const amrex::Real cse834 = std::pow(10.0, 1.5538288*cse816 - 5.5108049000000001*cse818 - 3.7209846*cse821 + 4.6450521*cse824 + 5.9369081000000001*cse825 - 24.311209000000002);
  const amrex::Real cse835 = ((cse831) ? (
   cse830
)
: ((cse833) ? (
   cse832
)
: (
   cse834
)));
  const amrex::Real cse836 = cse781*std::pow(cse814, 8);
  const amrex::Real cse837 = std::pow(cse814, 7);
  const amrex::Real cse838 = cse783*cse837;
  const amrex::Real cse839 = std::pow(cse814, 6);
  const amrex::Real cse840 = cse785*cse839;
  const amrex::Real cse841 = std::pow(10.0, 983.67575999999997*cse816 + 734.71650999999997*cse818 + 96.743155000000002*cse821 + 16.815729999999999*cse824 + 343.1918*cse825 + 70.609154000000004*cse836 + 364.14445999999998*cse838 + 801.81246999999996*cse840 - 21.928795999999998);
  const amrex::Real cse842 = std::pow(10.0, -8.8077017000000009*cse816 - 4.7274035999999997*cse818 + 0.93310621999999999*cse821 + 1.6802758*cse824 + 4.0406627000000004*cse825 - 6.3701156000000001*cse836 + 6.4380698000000001*cse838 + 8.9167182999999994*cse840 - 22.921188999999998);
  const amrex::Real cse843 = ((cse656) ? (
   cse841
)
: (
   cse842
));
  const amrex::Real cse844 = cse823 + cse827 + cse829 + cse835*state.xn[0] + cse843*state.xn[2];
  const amrex::Real cse845 = 1.0/cse844;
  const amrex::Real cse846 = cse813*cse845 + 1;
  const amrex::Real cse847 = 1.0/cse846;
  const amrex::Real cse848 = ((cse804) ? (
   0.0
)
: (
   cse813*cse847
));
  const amrex::Real cse849 = 0.045999999999999999*cse652;
  const amrex::Real cse850 = std::pow(cse651, cse849 + 0.22800000000000001);
  const amrex::Real cse851 = 8.9000000000000003e-11*cse850;
  const amrex::Real cse852 = cse851*state.xn[0];
  const amrex::Real cse853 = 0.037999999999999999*cse652;
  const amrex::Real cse854 = 5.1749999999999996e-11*std::pow(cse651, cse853 + 0.16900000000000001);
  const amrex::Real cse855 = 0.048000000000000001*cse652;
  const amrex::Real cse856 = 2.15e-11*std::pow(cse651, cse855 - 0.01);
  const amrex::Real cse857 = cse854 + cse856;
  const amrex::Real cse858 = 0.0070527699999999999*cse147;
  const amrex::Real cse859 = cse858 - 0.11133800000000001;
  const amrex::Real cse860 = cse147*cse859 + 0.69763799999999998;
  const amrex::Real cse861 = cse147*cse860 - 1.3074300000000001;
  const amrex::Real cse862 = 0.093813800000000003*cse147;
  const amrex::Real cse863 = cse862 - 3.0328300000000001;
  const amrex::Real cse864 = cse147*cse863 + 36.180300000000003;
  const amrex::Real cse865 = cse147*cse864 - 187.47399999999999;
  const amrex::Real cse866 = std::exp(((cse800) ? (
   cse147*cse861 - 7.6973500000000001
)
: (
   cse147*cse865 + 350.60899999999998
)));
  const amrex::Real cse867 = 1.7258000000000001e-6*cse650;
  const amrex::Real cse868 = cse866*cse867;
  const amrex::Real cse869 = cse852 + cse857*state.xn[3] + cse868;
  const amrex::Real cse870 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse871 = cse869*cse870;
  const amrex::Real cse872 = std::pow(cse651, cse849 + 0.23100000000000001);
  const amrex::Real cse873 = 2.6400000000000002e-10*cse872;
  const amrex::Real cse874 = cse873*state.xn[0];
  const amrex::Real cse875 = 0.031*cse652;
  const amrex::Real cse876 = 1.11e-10*std::pow(cse651, cse875 + 0.26300000000000001);
  const amrex::Real cse877 = 0.064000000000000001*cse652;
  const amrex::Real cse878 = 4.3749999999999998e-11*std::pow(cse651, cse877 + 0.071999999999999995);
  const amrex::Real cse879 = cse876 + cse878;
  const amrex::Real cse880 = 0.00235272*cse147;
  const amrex::Real cse881 = cse880 - 0.041816600000000002;
  const amrex::Real cse882 = cse147*cse881 + 0.35826400000000003;
  const amrex::Real cse883 = cse147*cse882 - 0.57443;
  const amrex::Real cse884 = 0.097857299999999994*cse147;
  const amrex::Real cse885 = cse884 - 3.1926800000000002;
  const amrex::Real cse886 = cse147*cse885 + 38.504899999999999;
  const amrex::Real cse887 = cse147*cse886 - 202.19300000000001;
  const amrex::Real cse888 = std::exp(((cse800) ? (
   cse147*cse883 - 7.4386999999999999
)
: (
   cse147*cse887 + 386.18599999999998
)));
  const amrex::Real cse889 = cse867*cse888;
  const amrex::Real cse890 = cse874 + cse879*state.xn[3] + cse889;
  const amrex::Real cse891 = (25.0/3.0)*cse890;
  const amrex::Real cse892 = 0.057000000000000002*cse652;
  const amrex::Real cse893 = std::pow(cse651, cse892 + 0.115);
  const amrex::Real cse894 = 1.26e-10*cse893;
  const amrex::Real cse895 = cse894*state.xn[0];
  const amrex::Real cse896 = 0.049000000000000002*cse652;
  const amrex::Real cse897 = 5.3249999999999997e-11*std::pow(cse651, cse896 - 0.0040000000000000001);
  const amrex::Real cse898 = 0.10199999999999999*cse652;
  const amrex::Real cse899 = 1.675e-11*std::pow(cse651, cse898 - 0.085000000000000006);
  const amrex::Real cse900 = cse897 + cse899;
  const amrex::Real cse901 = 0.00065632500000000003*cse147;
  const amrex::Real cse902 = -cse901 - 0.015089200000000001;
  const amrex::Real cse903 = cse147*cse902 + 0.36118400000000001;
  const amrex::Real cse904 = cse147*cse903 - 0.77378199999999997;
  const amrex::Real cse905 = 0.10508000000000001*cse147;
  const amrex::Real cse906 = cse905 - 3.4762;
  const amrex::Real cse907 = cse147*cse906 + 42.594999999999999;
  const amrex::Real cse908 = cse147*cse907 - 227.91300000000001;
  const amrex::Real cse909 = std::exp(((cse800) ? (
   cse147*cse904 - 9.2514099999999999
)
: (
   cse147*cse908 + 444.60000000000002
)));
  const amrex::Real cse910 = 2.8763333333333333e-6*cse909;
  const amrex::Real cse911 = cse650*cse910;
  const amrex::Real cse912 = cse895 + cse900*state.xn[3] + cse911;
  const amrex::Real cse913 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse914 = 5*cse913;
  const amrex::Real cse915 = cse912*cse914;
  const amrex::Real cse916 = cse912 + 7.8800000000000004e-8;
  const amrex::Real cse917 = cse869*cse914;
  const amrex::Real cse918 = cse871*cse891 + cse890*cse915 + cse916*cse917;
  const amrex::Real cse919 = cse890 + 2.65e-7;
  const amrex::Real cse920 = cse869 + 1.81e-14;
  const amrex::Real cse921 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse922 = cse890*cse921;
  const amrex::Real cse923 = (5.0/3.0)*cse922;
  const amrex::Real cse924 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse925 = cse912*cse924;
  const amrex::Real cse926 = 3*cse925;
  const amrex::Real cse927 = cse917*cse919 + cse919*cse926 + cse920*cse926;
  const amrex::Real cse928 = cse916*cse919 + cse916*cse920 + cse918 + cse920*cse923 + cse927;
  const amrex::Real cse929 = 1.0/cse928;
  const amrex::Real cse930 = 1.4210469458259591e-21*cse929;
  const amrex::Real cse931 = 2.5697412351439998e-22*cse929;
  const amrex::Real cse932 = state.T < 1000000.0;
  const amrex::Real cse933 = ((cse932) ? (
   cse918*cse930 + cse927*cse931
)
: (
   0.0
));
  const amrex::Real cse934 = (1.7e-24*chi*cse10*cse142*cse599*cse603*d2g + 5.6999999999999997e-11*chi*cse248*cse249*cse649*state.xn[3] + 1.7623947000000002e-12*crate*cse683*cse684*state.xn[5] + 1.602177e-12*crate*cse683*state.xn[0] + 3.2043540000000001e-12*crate*cse683*state.xn[3] + 3.0000000000000002e-15*cse10*cse648*d2g*state.xn[0] - cse134*cse619 - 7.1777529600000012e-12*cse241*cse30 - cse598*state.xn[0] - cse631 - cse666*cse668 - cse688*cse716 - cse717*state.xn[0] - cse775*state.xn[11] - cse848*state.xn[3] - cse933*state.xn[7] - std::min(cse593, cse597))/std::pow(cse590, 2);
  const amrex::Real cse935 = 1.0/cse590;
  const amrex::Real cse936 = d2g*state.xn[2];
  const amrex::Real cse937 = cse629*cse936;
  const amrex::Real cse938 = -cse593 + cse597;
  const amrex::Real cse939 = (((cse938 > 0) ? (
   0
)
: ((cse938 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse940 = cse594*cse939*d2g;
  const amrex::Real cse941 = cse595*cse940;
  const amrex::Real cse942 = psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse943 = (((cse938 < 0) ? (
   0
)
: ((cse938 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse944 = cse592*cse943;
  const amrex::Real cse945 = cse942*cse944;
  const amrex::Real cse946 = cse662*cse666;
  const amrex::Real cse947 = std::pow(cse660, -2);
  const amrex::Real cse948 = std::pow(cse178, -2);
  const amrex::Real cse949 = 1.5159999999999999e-9*cse654;
  const amrex::Real cse950 = cse659*cse661*cse947;
  const amrex::Real cse951 = cse668*state.xn[8]/std::pow(cse664, 2);
  const amrex::Real cse952 = std::pow(cse646, -2);
  const amrex::Real cse953 = cse35*cse645;
  const amrex::Real cse954 = cse36*cse636;
  const amrex::Real cse955 = cse38*cse639;
  const amrex::Real cse956 = cse954 + cse955;
  const amrex::Real cse957 = cse644/std::pow(cse642, 2);
  const amrex::Real cse958 = cse11*cse957;
  const amrex::Real cse959 = cse953 - cse958*(-cse640 + cse956);
  const amrex::Real cse960 = cse38*cse673;
  const amrex::Real cse961 = cse35*state.xn[2];
  const amrex::Real cse962 = std::pow(cse669, -2);
  const amrex::Real cse963 = 0.5*cse962;
  const amrex::Real cse964 = cse206/std::pow(cse10, 3);
  const amrex::Real cse965 = -0.5*cse670*cse961 + cse963*cse964;
  const amrex::Real cse966 = cse106*cse669*cse672*state.xn[0];
  const amrex::Real cse967 = cse11*cse8;
  const amrex::Real cse968 = ((cse676) ? (
   0
)
: ((cse677) ? (
   (3.0/2.0)*cse967
)
: ((cse678) ? (
   (4.0/3.0)*cse967
)
: ((cse679) ? (
   (1.0/3.0)*cse967
)
: (
   0
)))));
  const amrex::Real cse969 = cse681*state.xn[3];
  const amrex::Real cse970 = cse36*cse680;
  const amrex::Real cse971 = 2.0*cse970;
  const amrex::Real cse972 = cse674 - cse960 + cse965*cse966 + cse968*cse969 - cse971;
  const amrex::Real cse973 = cse35*ncol_H*(((cse698 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse698 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse699, 3.0/2.0);
  const amrex::Real cse974 = cse973*ncol_CO;
  const amrex::Real cse975 = ((cse697) ? (
   0
)
: (
   878350727.21550155*cse974
));
  const amrex::Real cse976 = alpha_co_interp2d_partial_1(state.T, cse701);
  const amrex::Real cse977 = std::log(cse708);
  const amrex::Real cse978 = cse976*cse977;
  const amrex::Real cse979 = cse688/std::pow(cse704, 2);
  const amrex::Real cse980 = nhalf_co_interp2d_partial_1(state.T, cse701);
  const amrex::Real cse981 = cse975*cse980;
  const amrex::Real cse982 = cse709/cse688;
  const amrex::Real cse983 = cse704*cse982;
  const amrex::Real cse984 = std::pow(cse702, -2);
  const amrex::Real cse985 = llte_co_interp2d_partial_1(state.T, cse701);
  const amrex::Real cse986 = -cse688*cse975*cse984*cse985 + cse711*(cse692*cse704*cse975*cse984*cse985 - cse705*cse981);
  const amrex::Real cse987 = cse688*state.xn[9];
  const amrex::Real cse988 = cse691*cse987/std::pow(cse713, 2);
  const amrex::Real cse989 = std::pow(cse844, -2);
  const amrex::Real cse990 = std::pow(cse846, -2);
  const amrex::Real cse991 = std::pow(cse813, 2)*cse989*cse990;
  const amrex::Real cse992 = cse741*cse757;
  const amrex::Real cse993 = 7.1400000000000007e-11*cse733;
  const amrex::Real cse994 = cse729*cse741;
  const amrex::Real cse995 = 8.68e-11*cse721;
  const amrex::Real cse996 = cse740*cse741;
  const amrex::Real cse997 = cse744*cse757;
  const amrex::Real cse998 = 6.3800000000000002e-11*cse749;
  const amrex::Real cse999 = cse741*cse759;
  const amrex::Real cse1000 = 2.8933333333333332e-11*cse721*cse997 + 2.1266666666666666e-11*cse745*cse749 + cse992*cse993 + cse993*cse994 + cse995*cse996 + cse998*cse999;
  const amrex::Real cse1001 = cse741*cse762;
  const amrex::Real cse1002 = 2.1419999999999999e-10*cse733;
  const amrex::Real cse1003 = cse763*cse767;
  const amrex::Real cse1004 = cse762*cse767;
  const amrex::Real cse1005 = cse1001*cse998 + cse1002*cse1003 + cse1002*cse1004 + 2.604e-10*cse721*cse768 + 1.9139999999999998e-10*cse749*cse768 + cse992*cse995;
  const amrex::Real cse1006 = std::pow(cse771, -2);
  const amrex::Real cse1007 = cse763*cse764;
  const amrex::Real cse1008 = cse1006*(-cse1000 - cse1005 - 1.4466666666666666e-10*cse1007*cse721 - cse722*cse759 - cse734*cse762 - cse734*cse763 - 1.0633333333333333e-10*cse749*cse765 - cse750*cse759);
  const amrex::Real cse1009 = 2.3872025442750958e-19*cse761;
  const amrex::Real cse1010 = 2.8023005140019997e-18*cse770;
  const amrex::Real cse1011 = cse870*cse890;
  const amrex::Real cse1012 = cse912*cse913;
  const amrex::Real cse1013 = 1.32e-9*cse872;
  const amrex::Real cse1014 = 6.3e-10*cse893;
  const amrex::Real cse1015 = cse869*cse913;
  const amrex::Real cse1016 = cse890*cse913;
  const amrex::Real cse1017 = cse913*cse916;
  const amrex::Real cse1018 = 4.4500000000000001e-10*cse850;
  const amrex::Real cse1019 = 7.4166666666666679e-10*cse1011*cse850 + cse1012*cse1013 + cse1014*cse1015 + cse1014*cse1016 + cse1017*cse1018 + 2.2000000000000003e-9*cse871*cse872;
  const amrex::Real cse1020 = cse913*cse919;
  const amrex::Real cse1021 = 3.7800000000000004e-10*cse893;
  const amrex::Real cse1022 = cse920*cse924;
  const amrex::Real cse1023 = cse919*cse924;
  const amrex::Real cse1024 = cse1013*cse1015 + cse1018*cse1020 + cse1021*cse1022 + cse1021*cse1023 + 2.6700000000000001e-10*cse850*cse925 + 7.9200000000000006e-10*cse872*cse925;
  const amrex::Real cse1025 = std::pow(cse928, -2);
  const amrex::Real cse1026 = cse920*cse921;
  const amrex::Real cse1027 = cse1025*(-cse1019 - cse1024 - 4.4000000000000003e-10*cse1026*cse872 - 1.4833333333333334e-10*cse850*cse922 - cse851*cse916 - cse873*cse916 - cse894*cse919 - cse894*cse920);
  const amrex::Real cse1028 = 1.4210469458259591e-21*cse918;
  const amrex::Real cse1029 = 2.5697412351439998e-22*cse927;
  const amrex::Real cse1030 = cse613*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1031 = cse134*cse616;
  const amrex::Real cse1032 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1033 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1034 = std::pow(cse605, -2);
  const amrex::Real cse1035 = cse1032*cse608;
  const amrex::Real cse1036 = std::pow(cse610, -2);
  const amrex::Real cse1037 = cse1036*cse612*(2.5500935934394033e-12*cse1033*cse1034*state.xn[2] - 6.6690584756579352e-10*cse1035);
  const amrex::Real cse1038 = 1.3659996292638161e-10*cse607;
  const amrex::Real cse1039 = std::pow(cse602, -2);
  const amrex::Real cse1040 = cse143*std::pow(cse600, 0.12299999999999994);
  const amrex::Real cse1041 = cse143*std::pow(cse600, -0.56899999999999995)*cse601;
  const amrex::Real cse1042 = cse621*cse85;
  const amrex::Real cse1043 = 1.0/cse623;
  const amrex::Real cse1044 = cse1043*cse147/std::pow(cse624, 2);
  const amrex::Real cse1045 = cse1043*cse627;
  const amrex::Real cse1046 = cse1038*cse134*cse613/std::pow(cse615, 2);
  const amrex::Real cse1047 = cse66*cse667;
  const amrex::Real cse1048 = cse3*cse650;
  const amrex::Real cse1049 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1050 = 100*cse3;
  const amrex::Real cse1051 = -2.2649999999999999e-6*cse1048 + cse655*(cse1050*(8.7000000000000001e-5*cse652 + 0.001281) + cse3*cse653) + state.xn[3]*(0.75*((cse656) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1049
)) + 0.25*((cse656) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1049
)));
  const amrex::Real cse1052 = 5.7927337833400002e-20*cse946;
  const amrex::Real cse1053 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1054 = -cse1051;
  const amrex::Real cse1055 = 5.7927337833400002e-20*cse666*cse950;
  const amrex::Real cse1056 = 2.0187430200000004e-26*cse243;
  const amrex::Real cse1057 = std::pow(cse633, -3);
  const amrex::Real cse1058 = cse952*(-cse638*cse7*(0.0040460000000000001*cse1057 + 0.0036709999999999998*cse634) - cse641*cse7*(0.0047800000000000004*cse1057 - 5.5500000000000001e-5*cse634));
  const amrex::Real cse1059 = 1.461185424e-21*cse250*state.xn[3];
  const amrex::Real cse1060 = 2*cse950;
  const amrex::Real cse1061 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1062 = std::pow(cse692, -2);
  const amrex::Real cse1063 = ((cse697) ? (
   -2.872276598720128e-13*cse694*cse85*state.xn[9]*(((5.7445531974402559e-18*cse2*cse694 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse694 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse695, 2)
)
: (
   0
));
  const amrex::Real cse1064 = -cse1063*cse985 - llte_co_interp2d_partial_0(state.T, cse701);
  const amrex::Real cse1065 = cse1063*cse980 + nhalf_co_interp2d_partial_0(state.T, cse701);
  const amrex::Real cse1066 = ((cse777) ? (
   -cse66
)
: (
   0
));
  const amrex::Real cse1067 = -1.8719999999999998e-14*cse1066*cse806 - 3.9261999999999998e-15*cse1066*cse807 - 1.53e-21*cse1066*cse808 + 4.985670872372847e-33*cse809*cse812*((cse777) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse810*cse812*((cse777) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse810*((cse777) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse811, 2);
  const amrex::Real cse1068 = ((cse777) ? (
   1
)
: (
   0
));
  const amrex::Real cse1069 = cse1068*cse805;
  const amrex::Real cse1070 = cse1069*cse13*cse814;
  const amrex::Real cse1071 = cse1069*cse787*cse817;
  const amrex::Real cse1072 = cse1069*cse789*cse819;
  const amrex::Real cse1073 = cse1069*cse8;
  const amrex::Real cse1074 = cse1069*cse791*cse820;
  const amrex::Real cse1075 = cse1069*cse781*cse837;
  const amrex::Real cse1076 = cse1069*cse783*cse839;
  const amrex::Real cse1077 = cse1069*cse785*cse815;
  const amrex::Real cse1078 = 65.320000000000007*cse992;
  const amrex::Real cse1079 = cse3*cse724;
  const amrex::Real cse1080 = 0.00042999999999999999*cse652;
  const amrex::Real cse1081 = cse735*(cse1050*(0.0041900000000000001 - 3.0000000000000001e-5*cse652) - cse3*cse732) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse737*(cse1050*(0.00025000000000000001*cse652 + 0.00264) + cse3*cse736) + cse738*(cse1050*(cse1080 + 0.00296) + cse1079));
  const amrex::Real cse1082 = cse729*cse742;
  const amrex::Real cse1083 = cse751*(cse1050*(0.0036900000000000001 - 6.0000000000000002e-5*cse652) - cse3*cse748) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse753*(cse1050*(0.00041000000000000005*cse652 + 0.0020300000000000001) + cse3*cse752) + cse755*(cse1050*(0.00058*cse652 + 0.0023700000000000001) + cse3*cse754));
  const amrex::Real cse1084 = (1.0/15.0)*cse745;
  const amrex::Real cse1085 = cse723*(cse1050*(0.0075500000000000003 - 0.0016000000000000001*cse652) - cse3*cse720) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse725*(cse1050*(cse1080 + 0.0088900000000000003) + cse1079) + cse727*(cse1050*(0.0052500000000000003*cse652 + 0.011979999999999999) + cse3*cse726));
  const amrex::Real cse1086 = cse744*cse758;
  const amrex::Real cse1087 = cse1083*cse759;
  const amrex::Real cse1088 = cse1078*cse66*cse759 + cse1081*cse1082 + cse1081*cse760 + cse1083*cse1084 + cse1085*cse1086 + cse1085*cse743 + cse1087*cse742 + 65.320000000000007*cse66*cse729*cse996 + 28.360000000000003*cse66*cse745*cse757;
  const amrex::Real cse1089 = 136.68000000000001*cse768;
  const amrex::Real cse1090 = cse66*cse763;
  const amrex::Real cse1091 = cse66*cse762;
  const amrex::Real cse1092 = cse1081*cse763;
  const amrex::Real cse1093 = (3.0/5.0)*cse767;
  const amrex::Real cse1094 = cse1081*cse762;
  const amrex::Real cse1095 = cse742*cse762;
  const amrex::Real cse1096 = cse1078*cse1091 + cse1083*cse1095 + cse1083*cse769 + cse1085*cse760 + cse1085*cse769 + cse1089*cse1090 + cse1089*cse1091 + cse1092*cse1093 + cse1093*cse1094;
  const amrex::Real cse1097 = (1.0/3.0)*cse1007;
  const amrex::Real cse1098 = cse1006*(-cse1083*cse766 - cse1085*cse1097 - cse1085*cse759 - cse1087 - cse1088 - 32.933333333333337*cse1090*cse765 - cse1092 - cse1094 - cse1096);
  const amrex::Real cse1099 = 312.30000000000001*cse66;
  const amrex::Real cse1100 = cse3*cse849;
  const amrex::Real cse1101 = 0.00046000000000000001*cse652;
  const amrex::Real cse1102 = 8.6290000000000003e-7*cse1048;
  const amrex::Real cse1103 = -cse1102*cse888 + cse874*(cse1050*(cse1101 + 0.00231) + cse1100) + cse889*((cse800) ? (
   cse147*(cse147*(cse3*cse880 + cse3*cse881) + cse3*cse882) + cse3*cse883
)
: (
   cse147*(cse147*(cse3*cse884 + cse3*cse885) + cse3*cse886) + cse3*cse887
)) + state.xn[3]*(cse876*(cse1050*(0.00031*cse652 + 0.00263) + cse3*cse875) + cse878*(cse1050*(0.00064000000000000005*cse652 + 0.00071999999999999994) + cse3*cse877));
  const amrex::Real cse1104 = (25.0/3.0)*cse871;
  const amrex::Real cse1105 = -1.4381666666666666e-6*cse1048*cse909 + cse895*(cse1050*(0.00057000000000000009*cse652 + 0.00115) + cse3*cse892) + cse911*((cse800) ? (
   cse147*(cse147*(-cse3*cse901 + cse3*cse902) + cse3*cse903) + cse3*cse904
)
: (
   cse147*(cse147*(cse3*cse905 + cse3*cse906) + cse3*cse907) + cse3*cse908
)) + state.xn[3]*(cse897*(cse1050*(0.00048999999999999998*cse652 - 4.0000000000000003e-5) + cse3*cse896) + cse899*(cse1050*(0.0010199999999999999*cse652 - 0.00085000000000000006) + cse3*cse898));
  const amrex::Real cse1106 = -cse1102*cse866 + cse852*(cse1050*(cse1101 + 0.0022800000000000003) + cse1100) + cse868*((cse800) ? (
   cse147*(cse147*(cse3*cse858 + cse3*cse859) + cse3*cse860) + cse3*cse861
)
: (
   cse147*(cse147*(cse3*cse862 + cse3*cse863) + cse3*cse864) + cse3*cse865
)) + state.xn[3]*(cse854*(cse1050*(0.00038000000000000002*cse652 + 0.0016900000000000001) + cse3*cse853) + cse856*(cse1050*(0.00048000000000000001*cse652 - 0.0001) + cse3*cse855));
  const amrex::Real cse1107 = cse870*cse891;
  const amrex::Real cse1108 = cse890*cse914;
  const amrex::Real cse1109 = cse1106*cse916;
  const amrex::Real cse1110 = cse1012*cse1099*cse890 + cse1017*cse1099*cse869 + cse1103*cse1104 + cse1103*cse915 + cse1105*cse1108 + cse1105*cse917 + cse1106*cse1107 + cse1109*cse914 + 844.16666666666686*cse66*cse871*cse890;
  const amrex::Real cse1111 = 70.859999999999985*cse66*cse925;
  const amrex::Real cse1112 = cse1105*cse920;
  const amrex::Real cse1113 = 3*cse924;
  const amrex::Real cse1114 = cse914*cse919;
  const amrex::Real cse1115 = cse1105*cse919;
  const amrex::Real cse1116 = cse1015*cse1099*cse919 + cse1103*cse917 + cse1103*cse926 + cse1106*cse1114 + cse1106*cse926 + cse1111*cse919 + cse1111*cse920 + cse1112*cse1113 + cse1113*cse1115;
  const amrex::Real cse1117 = (5.0/3.0)*cse1026;
  const amrex::Real cse1118 = cse1025*(-cse1103*cse1117 - cse1103*cse916 - cse1106*cse923 - cse1109 - cse1110 - cse1112 - cse1115 - cse1116 - 64.733333333333348*cse66*cse920*cse922);
  const amrex::Real cse1119 = cse173*cse935*(1.7e-24*chi*cse10*cse1039*cse142*cse599*d2g*(-0.00012636182959999999*cse1040*cse107 - 0.018302846000000001*cse1041*cse107) + 5.6227499999999995e-25*chi*cse10*cse142*cse603*d2g*std::pow(state.T, -0.85299999999999998) - 1.6173435610483583e-5*cse1030*cse1031*cse606 - cse1031*cse1037*cse1038 - 2.0489994438957239e-11*cse1032*cse134*cse606*cse617 + 7.8348995947646122e-14*cse1033*cse1034*cse611*cse612*cse616*state.xn[0]*state.xn[2] - cse1046*(-0.0015792330470357991*cse1030*cse608 + 7.6502807803182096e-12*cse1033*cse1034*cse611*cse612*state.xn[2] - 2.0007175426973804e-9*cse1035*cse613 - 1.3338116951315871e-8*cse1037*cse609) - 5.2835524837844135e-18*cse1047*cse666 - cse1051*cse1052 - cse1053*cse716 - cse1054*cse1055 - cse1056*cse1058*cse957 - cse1058*cse1059*cse958 - 2.1789607199999999e-11*cse135 - 7.1777529600000012e-12*cse514 - 7.1777529600000012e-12*cse515 - cse631*(-1.8989*cse1042*cse1044 + cse3*cse625) - cse631*(-0.043256499999999996*cse1042*cse1043*cse624 + 0.84999999999999998*cse1042*cse1045) - 1000*cse689*std::pow(cse690, 999)*cse714*cse987 - cse939*(1.6000000000000001e-32*cse159*cse594*cse85 + cse596) - cse944*psigd_coll_interp2d_partial_1(cse10, state.T) - cse951*(-182.41999999999999*cse1047 - cse1051*cse663 - cse1054*cse1060) - cse988*(-cse1053*cse703 + cse1061*cse1062*cse706*cse710 + cse1061*cse1062 - cse1064*cse688*cse984 - cse711*(-cse1061*cse703*cse704 - cse1064*cse692*cse704*cse984 - cse1065*cse705) - cse712*(cse977*(cse1063*cse976 + alpha_co_interp2d_partial_0(state.T, cse701)) + cse983*(cse1053*cse707 - cse1065*cse979))) - state.xn[11]*(cse1009*cse1098 + cse1010*cse1098 + cse1088*cse773 + cse1096*cse774) - state.xn[3]*((cse804) ? (
   0
)
: (
   cse1067*cse847 + cse813*cse990*(-cse1067*cse845 - cse813*cse989*(-cse823*(3.0182298984217999*cse1068*cse791*cse805*cse820 + 4.8223900769399082*cse1068*cse8*cse805 - 3.5529549287336835*cse1070 - 0.3872755400043702*cse1071 - 1.3735579540080118*cse1072) - cse827*(0.0708789387907936*cse1070 + 3.7035619079275177*cse1071 - 4.6974781513675152*cse1072 + 3.6184459289309552*cse1073 - 1.6316107607322889*cse1074) - cse829*(-3.7541549062629067*cse1070 + 2.2094886994529301*cse1071 - 1.528565035159452*cse1072 + 5.0409049417480247*cse1073 + 2.0057552335975877*cse1074) - state.xn[0]*((cse831) ? (
      cse7*cse830*(116.29033200000001*cse1070 + 19.239805*cse1071 + 80.639324000000002*cse1072 + 37.383713*cse1073 + 145.968309*cse1074)
   )
   : ((cse833) ? (
      cse832*(8.2184944748967013*cse1068*cse8*cse805 - 52.189748993977005*cse1070 - 48.951834264235494*cse1071 - 196.44057098336626*cse1072 - 192.38155095558542*cse1074)
   )
   : (
      cse834*(-17.135767342440825*cse1070 + 17.889115159724135*cse1071 - 50.756388852554174*cse1072 + 10.695627721640689*cse1073 + 41.010708268606813*cse1074)
   ))) - state.xn[2]*((cse656) ? (
      cse841*(445.51869310442481*cse1070 + 11324.985706577943*cse1071 + 6766.9891340104432*cse1072 + 38.719649225812766*cse1073 + 2370.6849681533818*cse1074 + 1300.6686834345674*cse1075 + 5869.3152370465659*cse1076 + 11077.448644792414*cse1077)
   )
   : (
      cse842*(4.297112944704045*cse1070 - 101.40241318979159*cse1071 - 43.540996231705549*cse1072 + 3.8689780091986448*cse1073 + 27.91190909651122*cse1074 - 117.34186576967058*cse1075 + 103.76942484394611*cse1076 + 123.18901581604325*cse1077)
   ))))
)) - state.xn[7]*((cse932) ? (
   cse1028*cse1118 + cse1029*cse1118 + cse1110*cse930 + cse1116*cse931
)
: (
   0
)));
  const amrex::Real cse1120 = cse143*cse603*d2g;
  const amrex::Real cse1121 = crate*cse683;
  const amrex::Real cse1122 = 1.7623947000000002e-12*cse1121*cse948*state.xn[5];
  const amrex::Real cse1123 = cse35*cse643*cse644 - cse956*cse958;
  const amrex::Real cse1124 = cse10*cse1056*cse952;
  const amrex::Real cse1125 = cse1059*cse952;
  const amrex::Real cse1126 = 26.399999999999999*cse106*cse669*cse671*cse965*state.xn[0] + 2.0*cse11*cse968*state.xn[3] - cse960 - cse971;
  const amrex::Real cse1127 = 1.602177e-12*cse185;
  const amrex::Real cse1128 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1129 = 1.7623947000000002e-12*cse684;
  const amrex::Real cse1130 = crate*cse1129*state.xn[5];
  const amrex::Real cse1131 = cse707*cse709;
  const amrex::Real cse1132 = 1.7e-24*cse1120*cse599 + cse1122*cse182 + cse1123*cse1124 + cse1123*cse1125 + cse1126*cse1127 + cse1126*cse1128 + cse1126*cse1130 - 7.1777529600000012e-12*cse201 + 3.0000000000000002e-15*cse243*cse648 - 7.1777529600000012e-12*cse521 - cse937 - cse941 - cse945 - cse988*(-cse712*(-cse1131*cse981 + cse975*cse976*cse977) - cse986);
  const amrex::Real cse1133 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1134 = cse626*cse628;
  const amrex::Real cse1135 = std::pow(cse620, -2);
  const amrex::Real cse1136 = cse1135*cse143;
  const amrex::Real cse1137 = 0.5*cse11*cse670 - cse961*cse963;
  const amrex::Real cse1138 = 9.0599999999999997e-6*cse71;
  const amrex::Real cse1139 = 1.024e-10*cse730;
  const amrex::Real cse1140 = 2.1600000000000004e-15*cse718;
  const amrex::Real cse1141 = 9.7199999999999998e-11*cse746;
  const amrex::Real cse1142 = cse1139*cse992 + cse1139*cse994 + cse1140*cse996 + cse1141*cse999 + 7.1999999999999997e-16*cse718*cse997 + 3.2399999999999999e-11*cse745*cse746;
  const amrex::Real cse1143 = 3.0719999999999998e-10*cse730;
  const amrex::Real cse1144 = cse1001*cse1141 + cse1003*cse1143 + cse1004*cse1143 + cse1140*cse992 + 6.4799999999999996e-15*cse718*cse768 + 2.9159999999999995e-10*cse746*cse768;
  const amrex::Real cse1145 = cse1006*(-3.6000000000000001e-15*cse1007*cse718 - cse1142 - cse1144 - cse719*cse759 - cse731*cse762 - cse731*cse763 - 1.6199999999999998e-10*cse746*cse765 - cse747*cse759);
  const amrex::Real cse1146 = cse71*cse888;
  const amrex::Real cse1147 = cse71*cse909;
  const amrex::Real cse1148 = 1.4381666666666666e-5*cse1147;
  const amrex::Real cse1149 = cse71*cse866;
  const amrex::Real cse1150 = 8.6289999999999999e-6*cse1146;
  const amrex::Real cse1151 = 8.6289999999999999e-6*cse1149;
  const amrex::Real cse1152 = 1.4381666666666668e-5*cse1011*cse1149 + cse1012*cse1150 + cse1015*cse1148 + cse1016*cse1148 + cse1017*cse1151 + 1.4381666666666668e-5*cse1146*cse871;
  const amrex::Real cse1153 = 5.1774000000000006e-6*cse925;
  const amrex::Real cse1154 = 8.6289999999999999e-6*cse1147;
  const amrex::Real cse1155 = cse1015*cse1150 + cse1020*cse1151 + cse1022*cse1154 + cse1023*cse1154 + cse1146*cse1153 + cse1149*cse1153;
  const amrex::Real cse1156 = cse71*cse910;
  const amrex::Real cse1157 = 1.7258000000000001e-6*cse916;
  const amrex::Real cse1158 = cse1025*(-2.8763333333333337e-6*cse1026*cse1146 - cse1146*cse1157 - cse1149*cse1157 - 2.8763333333333337e-6*cse1149*cse922 - cse1152 - cse1155 - cse1156*cse919 - cse1156*cse920);
  const amrex::Real cse1159 = cse1134*cse936;
  const amrex::Real cse1160 = 3.7996253111508318e-29*cse1159;
  const amrex::Real cse1161 = cse2*cse940;
  const amrex::Real cse1162 = 6.4000000000000004e-32*cse1161;
  const amrex::Real cse1163 = cse942*cse943*d2g;
  const amrex::Real cse1164 = 200.0*cse1163;
  const amrex::Real cse1165 = -cse658;
  const amrex::Real cse1166 = cse235*cse636 + 2*cse955;
  const amrex::Real cse1167 = 2*cse953 - cse958*(cse1166 - cse637);
  const amrex::Real cse1168 = 2*cse960;
  const amrex::Real cse1169 = cse670*cse961;
  const amrex::Real cse1170 = cse962*cse964;
  const amrex::Real cse1171 = -1.0*cse1169 + 1.0*cse1170;
  const amrex::Real cse1172 = ((cse676) ? (
   0
)
: ((cse677) ? (
   3*cse967
)
: ((cse678) ? (
   (8.0/3.0)*cse967
)
: ((cse679) ? (
   (2.0/3.0)*cse967
)
: (
   0
)))));
  const amrex::Real cse1173 = 4.0*cse970;
  const amrex::Real cse1174 = -cse1168 + cse1171*cse966 + cse1172*cse969 - cse1173 + cse682;
  const amrex::Real cse1175 = ((cse697) ? (
   0
)
: (
   cse700*cse973
));
  const amrex::Real cse1176 = cse1175*cse980;
  const amrex::Real cse1177 = -cse1175*cse688*cse984*cse985 + cse711*(cse1175*cse692*cse704*cse984*cse985 - cse1176*cse705);
  const amrex::Real cse1178 = cse756*cse759;
  const amrex::Real cse1179 = cse1082*cse739 + cse1084*cse756 + cse1086*cse728 + cse1178*cse742 + cse728*cse743 + cse739*cse760;
  const amrex::Real cse1180 = cse739*cse763;
  const amrex::Real cse1181 = cse739*cse762;
  const amrex::Real cse1182 = cse1093*cse1180 + cse1093*cse1181 + cse1095*cse756 + cse728*cse760 + cse728*cse769 + cse756*cse769;
  const amrex::Real cse1183 = cse1006*(-cse1097*cse728 - cse1178 - cse1179 - cse1180 - cse1181 - cse1182 - cse728*cse759 - cse756*cse766);
  const amrex::Real cse1184 = cse857*cse916;
  const amrex::Real cse1185 = cse1104*cse879 + cse1107*cse857 + cse1108*cse900 + cse1184*cse914 + cse879*cse915 + cse900*cse917;
  const amrex::Real cse1186 = cse900*cse920;
  const amrex::Real cse1187 = cse900*cse919;
  const amrex::Real cse1188 = cse1113*cse1186 + cse1113*cse1187 + cse1114*cse857 + cse857*cse926 + cse879*cse917 + cse879*cse926;
  const amrex::Real cse1189 = cse1025*(-cse1117*cse879 - cse1184 - cse1185 - cse1186 - cse1187 - cse1188 - cse857*cse923 - cse879*cse916);
  const amrex::Real cse1190 = cse1120*cse599;
  const amrex::Real cse1191 = -cse1166*cse958 + 2*cse35*cse643*cse644;
  const amrex::Real cse1192 = 26.399999999999999*cse106*cse1171*cse669*cse671*state.xn[0] + 2.0*cse11*cse1172*state.xn[3] - cse1168 - cse1173;
  const amrex::Real cse1193 = state.xn[3]*((cse804) ? (
   0
)
: (
   cse828*cse991
));
  const amrex::Real cse1194 = ((cse697) ? (
   cse696
)
: (
   0
));
  const amrex::Real cse1195 = cse1194*cse980;
  const amrex::Real cse1196 = -cse1132*cse935;
  const amrex::Real cse1197 = 3*cse35*cse643*cse644 - cse958*(3*cse954 + 3*cse955);
  const amrex::Real cse1198 = 26.399999999999999*cse106*cse669*cse671*state.xn[0]*(-1.5*cse1169 + 1.5*cse1170) + 2.0*cse11*state.xn[3]*((cse676) ? (
   0
)
: ((cse677) ? (
   (9.0/2.0)*cse967
)
: ((cse678) ? (
   4*cse967
)
: ((cse679) ? (
   cse967
)
: (
   0
))))) - 3*cse960 - 6.0*cse970;
  const amrex::Real cse1199 = ((cse697) ? (
   0
)
: (
   2635052181.6465044*cse974
));
  const amrex::Real cse1200 = cse1199*cse980;

  jac(1, 1) = -cse1 - cse10*cse56 - cse174*cse65 - cse191 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3];
  jac(1, 2) = -cse174*cse192 + cse197 + cse205;
  jac(1, 3) = -cse174*cse212 + cse209 + cse211*state.xn[16] + cse214*cse215 + cse220 + cse225;
  jac(1, 4) = 3.2000000000000001e-9*cse162*state.xn[17] - cse174*cse253 + cse204*cse246 - cse215*cse254 + cse227 + cse229 + cse231 + 2*cse232 + 2*cse240 + 4*cse242 - cse244 + 1.1399999999999999e-10*cse250 + cse252*state.xn[8] + cse258 + cse264;
  jac(1, 5) = -cse174*cse271 - cse244 + 2*cse268*cse30 + 2*cse269*state.xn[0]*state.xn[3] + cse270 - cse276;
  jac(1, 6) = -cse174*cse277;
  jac(1, 7) = -cse174*cse278 + cse283;
  jac(1, 8) = -cse174*cse286 + cse285;
  jac(1, 9) = -cse174*cse287 + cse252*state.xn[3];
  jac(1, 10) = -cse174*cse289 - cse288;
  jac(1, 11) = -cse174*cse290 + cse205 + cse292;
  jac(1, 12) = -cse174*cse296 + cse294 + cse295 + cse299;
  jac(1, 13) = -cse174*cse300;
  jac(1, 14) = -cse174*cse301;
  jac(1, 15) = -cse174*cse306 + cse302 + cse303 + cse305 + cse307;
  jac(1, 16) = -cse174*cse308 + cse307 + cse312;
  jac(1, 17) = -cse174*cse319 + cse204*cse318 + cse211*state.xn[2] - 1.8000000000000002e-14*cse243 + 2*cse317 + cse325;
  jac(1, 18) = 3.2000000000000001e-9*cse162*state.xn[3] - cse174*cse329 - cse326 + cse328;
  jac(1, 19) = cse174;
  jac(2, 1) = cse191 - cse331*cse65;
  jac(2, 2) = -cse192*cse331 - cse197;
  jac(2, 3) = -cse212*cse331 - cse224;
  jac(2, 4) = cse227 - cse253*cse331 + cse262 + cse272;
  jac(2, 5) = -cse271*cse331 + cse276;
  jac(2, 6) = -cse277*cse331;
  jac(2, 7) = -cse278*cse331 + cse279;
  jac(2, 8) = -cse286*cse331;
  jac(2, 9) = -cse287*cse331;
  jac(2, 10) = -cse289*cse331;
  jac(2, 11) = -cse195 - cse290*cse331;
  jac(2, 12) = -cse296*cse331 - cse297;
  jac(2, 13) = -cse300*cse331;
  jac(2, 14) = -cse301*cse331;
  jac(2, 15) = -cse195 - cse306*cse331;
  jac(2, 16) = -cse195 - cse308*cse331;
  jac(2, 17) = -cse319*cse331 - cse324;
  jac(2, 18) = cse326 - cse329*cse331;
  jac(2, 19) = cse331;
  jac(3, 1) = cse179 + cse181 + cse186 - cse190 - cse334 - cse422*cse65 - cse440;
  jac(3, 2) = -cse192*cse422 - cse196 - cse442;
  jac(3, 3) = -cse212*cse422 - cse225 - cse443 - cse445 - cse447 - cse448 - cse454;
  jac(3, 4) = -cse253*cse422 - cse264 + cse455 + cse456 - cse461 - cse468 - cse473 - cse474 - cse477;
  jac(3, 5) = -cse271*cse422 + cse275 - cse460 - cse466 - cse477 - cse479;
  jac(3, 6) = -cse277*cse422 + cse480;
  jac(3, 7) = -cse278*cse422 - cse481;
  jac(3, 8) = -cse286*cse422 - cse483 + cse485;
  jac(3, 9) = -cse287*cse422 - cse488;
  jac(3, 10) = cse288 - cse289*cse422;
  jac(3, 11) = -cse290*cse422 - cse292 - cse442;
  jac(3, 12) = -cse296*cse422 - cse490;
  jac(3, 13) = -cse300*cse422 + cse493;
  jac(3, 14) = -cse301*cse422 - cse494;
  jac(3, 15) = -cse306*cse422 - cse495;
  jac(3, 16) = -cse308*cse422 - cse495;
  jac(3, 17) = -cse319*cse422 - cse325 - cse496 - cse497 - cse498 - cse499 - cse501 - cse504 - cse505;
  jac(3, 18) = -cse329*cse422 - cse506;
  jac(3, 19) = cse422;
  jac(3, 20) = cse507;
  jac(4, 1) = cse1 + 3.0000000000000002e-15*cse10*d2g + cse177 - cse333 - cse508 - cse509 - cse511 - cse512 - cse520*cse65;
  jac(4, 2) = -cse192*cse520 - cse522;
  jac(4, 3) = cse209 - cse212*cse520 + cse523*state.xn[16] + cse524 + cse525;
  jac(4, 4) = -cse202 - cse232 - cse240 - cse241*cse9 - cse253*cse520 - cse463 - cse526 - cse527 - cse529 - cse530 - cse531 - cse532;
  jac(4, 5) = -cse202 - cse270 - cse271*cse520 + cse273 - cse478 - cse533 - cse534;
  jac(4, 6) = -cse277*cse520;
  jac(4, 7) = -cse278*cse520 - cse279 - cse535;
  jac(4, 8) = -cse286*cse520 + cse483;
  jac(4, 9) = -cse287*cse520 - cse486;
  jac(4, 10) = -cse289*cse520 + cse536;
  jac(4, 11) = -cse290*cse520 - cse522;
  jac(4, 12) = -cse296*cse520 + cse490;
  jac(4, 13) = -cse300*cse520;
  jac(4, 14) = -cse301*cse520;
  jac(4, 15) = -cse305 - cse306*cse520 - cse522;
  jac(4, 16) = -cse308*cse520 - cse522;
  jac(4, 17) = -cse317 - cse319*cse520 + cse501 + cse523*state.xn[2] + cse537 - cse538 + cse540;
  jac(4, 18) = -cse329*cse520 - cse506;
  jac(4, 19) = cse520;
  jac(5, 1) = -cse177 - cse334 - cse541*cse65;
  jac(5, 2) = -cse192*cse541 + cse441;
  jac(5, 3) = -cse212*cse541;
  jac(5, 4) = -cse229 - cse253*cse541 + cse528 + cse530;
  jac(5, 5) = -cse270 - cse271*cse541 - cse273 - cse479;
  jac(5, 6) = -cse277*cse541;
  jac(5, 7) = -cse278*cse541 + cse535;
  jac(5, 8) = -cse286*cse541;
  jac(5, 9) = -cse287*cse541;
  jac(5, 10) = -cse289*cse541;
  jac(5, 11) = -cse290*cse541 + cse441;
  jac(5, 12) = -cse296*cse541;
  jac(5, 13) = -cse300*cse541;
  jac(5, 14) = -cse301*cse541;
  jac(5, 15) = -cse306*cse541 + cse441;
  jac(5, 16) = -cse308*cse541 + cse441;
  jac(5, 17) = -cse319*cse541 + cse500;
  jac(5, 18) = -cse329*cse541;
  jac(5, 19) = cse541;
  jac(6, 1) = cse426 - cse544*cse65;
  jac(6, 2) = -cse192*cse544 + cse426;
  jac(6, 3) = -cse212*cse544 + cse443;
  jac(6, 4) = -cse253*cse544 + cse545;
  jac(6, 5) = -cse271*cse544 + cse476;
  jac(6, 6) = -cse277*cse544 - cse480;
  jac(6, 7) = -cse278*cse544 + cse547;
  jac(6, 8) = -cse286*cse544;
  jac(6, 9) = -cse287*cse544;
  jac(6, 10) = -cse289*cse544 + cse548;
  jac(6, 11) = -cse290*cse544 + cse426;
  jac(6, 12) = -cse296*cse544;
  jac(6, 13) = -cse300*cse544;
  jac(6, 14) = -cse301*cse544;
  jac(6, 15) = -cse306*cse544 + cse426;
  jac(6, 16) = -cse308*cse544 + cse549;
  jac(6, 17) = -cse319*cse544 + cse496;
  jac(6, 18) = -cse329*cse544;
  jac(6, 19) = cse544;
  jac(7, 1) = -cse426 - cse550*cse65;
  jac(7, 2) = -cse192*cse550 - cse426;
  jac(7, 3) = -cse212*cse550 - cse443;
  jac(7, 4) = -cse253*cse550 - cse545;
  jac(7, 5) = -cse271*cse550 - cse476;
  jac(7, 6) = -cse277*cse550 + cse480;
  jac(7, 7) = -cse278*cse550 - cse547;
  jac(7, 8) = -cse286*cse550;
  jac(7, 9) = -cse287*cse550;
  jac(7, 10) = -cse289*cse550 - cse548;
  jac(7, 11) = -cse290*cse550 - cse426;
  jac(7, 12) = -cse296*cse550;
  jac(7, 13) = -cse300*cse550;
  jac(7, 14) = -cse301*cse550;
  jac(7, 15) = -cse306*cse550 - cse426;
  jac(7, 16) = -cse308*cse550 - cse549;
  jac(7, 17) = -cse319*cse550 - cse496;
  jac(7, 18) = -cse329*cse550;
  jac(7, 19) = cse550;
  jac(8, 1) = cse1 - cse552*cse65 + cse556;
  jac(8, 2) = -cse192*cse552 + cse556;
  jac(8, 3) = -cse212*cse552 + cse445;
  jac(8, 4) = -cse253*cse552 + cse467 + cse557*state.xn[8] + cse562;
  jac(8, 5) = -cse271*cse552 + cse466 + cse561;
  jac(8, 6) = -cse277*cse552;
  jac(8, 7) = -cse278*cse552;
  jac(8, 8) = -cse285 - cse286*cse552 - cse483 - cse485;
  jac(8, 9) = -cse287*cse552 + cse487 + cse557*state.xn[3];
  jac(8, 10) = -cse289*cse552 + cse563;
  jac(8, 11) = -cse290*cse552 + cse556;
  jac(8, 12) = -cse296*cse552;
  jac(8, 13) = -cse300*cse552;
  jac(8, 14) = -cse301*cse552;
  jac(8, 15) = -cse306*cse552 + cse556 + cse564;
  jac(8, 16) = -cse308*cse552 + cse430 + cse433 - cse565;
  jac(8, 17) = -cse319*cse552 - cse497 + cse502 + cse503 - cse566;
  jac(8, 18) = -cse329*cse552;
  jac(8, 19) = cse552;
  jac(9, 1) = -cse434 - cse569*cse65;
  jac(9, 2) = -cse192*cse569 - cse434;
  jac(9, 3) = -cse212*cse569 - cse445;
  jac(9, 4) = -cse253*cse569 - cse468;
  jac(9, 5) = -cse271*cse569 - cse466;
  jac(9, 6) = -cse277*cse569;
  jac(9, 7) = -cse278*cse569 + cse546;
  jac(9, 8) = -cse286*cse569 + cse485;
  jac(9, 9) = -cse287*cse569 - cse488 - cse571;
  jac(9, 10) = -cse289*cse569 + cse548;
  jac(9, 11) = -cse290*cse569 - cse434;
  jac(9, 12) = -cse296*cse569;
  jac(9, 13) = -cse300*cse569;
  jac(9, 14) = -cse301*cse569;
  jac(9, 15) = -cse306*cse569 - cse434;
  jac(9, 16) = -cse308*cse569 - cse434 - cse572;
  jac(9, 17) = -cse319*cse569 - cse504;
  jac(9, 18) = -cse329*cse569;
  jac(9, 19) = cse569;
  jac(10, 1) = cse554 - cse573*cse65;
  jac(10, 2) = -cse192*cse573 + cse554;
  jac(10, 3) = -cse212*cse573 + cse222;
  jac(10, 4) = -cse253*cse573 - cse562;
  jac(10, 5) = -cse271*cse573 + cse560;
  jac(10, 6) = -cse277*cse573;
  jac(10, 7) = -cse278*cse573 - cse546;
  jac(10, 8) = cse285 - cse286*cse573;
  jac(10, 9) = -cse287*cse573;
  jac(10, 10) = -cse289*cse573 - cse574 - cse575;
  jac(10, 11) = -cse290*cse573 + cse291 + cse554;
  jac(10, 12) = -cse296*cse573 + cse298;
  jac(10, 13) = -cse300*cse573;
  jac(10, 14) = -cse301*cse573;
  jac(10, 15) = -cse306*cse573 + cse576;
  jac(10, 16) = -cse308*cse573 + cse565;
  jac(10, 17) = -cse319*cse573 - cse539 + cse566;
  jac(10, 18) = -cse329*cse573;
  jac(10, 19) = cse573;
  jac(11, 1) = -cse577*cse65;
  jac(11, 2) = -cse192*cse577;
  jac(11, 3) = -cse212*cse577 - cse222;
  jac(11, 4) = -cse253*cse577;
  jac(11, 5) = -cse271*cse577;
  jac(11, 6) = -cse277*cse577;
  jac(11, 7) = -cse278*cse577;
  jac(11, 8) = -cse286*cse577;
  jac(11, 9) = -cse287*cse577 + cse571;
  jac(11, 10) = -cse289*cse577 + cse574;
  jac(11, 11) = -cse290*cse577 - cse291;
  jac(11, 12) = -cse296*cse577;
  jac(11, 13) = -cse300*cse577;
  jac(11, 14) = -cse301*cse577;
  jac(11, 15) = -cse306*cse577;
  jac(11, 16) = -cse308*cse577 + cse572;
  jac(11, 17) = -cse319*cse577 + cse539;
  jac(11, 18) = -cse329*cse577;
  jac(11, 19) = cse577;
  jac(12, 1) = cse176 - cse554 - cse580*cse65;
  jac(12, 2) = -cse104 - cse192*cse580 - cse554;
  jac(12, 3) = -cse212*cse580 - cse524;
  jac(12, 4) = -cse253*cse580 + cse256 - cse257 - cse472 + cse559 - cse560;
  jac(12, 5) = -cse271*cse580 - cse560;
  jac(12, 6) = -cse277*cse580;
  jac(12, 7) = -cse278*cse580 + cse546;
  jac(12, 8) = -cse286*cse580;
  jac(12, 9) = -cse287*cse580;
  jac(12, 10) = -cse289*cse580 + cse575;
  jac(12, 11) = -cse290*cse580 - cse554;
  jac(12, 12) = cse295 - cse296*cse580 - cse299 - cse489;
  jac(12, 13) = -cse300*cse580;
  jac(12, 14) = -cse301*cse580;
  jac(12, 15) = -cse306*cse580 - cse576;
  jac(12, 16) = -cse308*cse580 + cse311 + cse555;
  jac(12, 17) = -cse319*cse580 - cse498 - cse566;
  jac(12, 18) = cse326 + cse327 - cse329*cse580;
  jac(12, 19) = cse580;
  jac(13, 1) = cse439 - cse582*cse65;
  jac(13, 2) = -cse192*cse582 + cse439;
  jac(13, 3) = -cse212*cse582 + cse447;
  jac(13, 4) = -cse253*cse582 + cse461;
  jac(13, 5) = -cse271*cse582 + cse460;
  jac(13, 6) = -cse277*cse582;
  jac(13, 7) = -cse278*cse582;
  jac(13, 8) = -cse286*cse582;
  jac(13, 9) = -cse287*cse582;
  jac(13, 10) = -cse289*cse582;
  jac(13, 11) = -cse290*cse582 + cse439;
  jac(13, 12) = -cse296*cse582;
  jac(13, 13) = -cse300*cse582 - cse493;
  jac(13, 14) = -cse301*cse582 + cse494;
  jac(13, 15) = -cse306*cse582 + cse439;
  jac(13, 16) = -cse308*cse582 + cse439;
  jac(13, 17) = -cse319*cse582 + cse505;
  jac(13, 18) = -cse329*cse582;
  jac(13, 19) = cse582;
  jac(13, 20) = cse583;
  jac(14, 1) = -cse439 - cse584*cse65;
  jac(14, 2) = -cse192*cse584 - cse439;
  jac(14, 3) = -cse212*cse584 - cse447;
  jac(14, 4) = -cse253*cse584 - cse461;
  jac(14, 5) = -cse271*cse584 - cse460;
  jac(14, 6) = -cse277*cse584;
  jac(14, 7) = -cse278*cse584;
  jac(14, 8) = -cse286*cse584;
  jac(14, 9) = -cse287*cse584;
  jac(14, 10) = -cse289*cse584;
  jac(14, 11) = -cse290*cse584 - cse439;
  jac(14, 12) = -cse296*cse584;
  jac(14, 13) = -cse300*cse584 + cse493;
  jac(14, 14) = -cse301*cse584 - cse494;
  jac(14, 15) = -cse306*cse584 - cse439;
  jac(14, 16) = -cse308*cse584 - cse439;
  jac(14, 17) = -cse319*cse584 - cse505;
  jac(14, 18) = -cse329*cse584;
  jac(14, 19) = cse584;
  jac(14, 20) = cse507;
  jac(15, 1) = -cse1 - cse585*cse65;
  jac(15, 2) = -cse192*cse585;
  jac(15, 3) = -cse212*cse585;
  jac(15, 4) = -cse253*cse585 + cse586*state.xn[8];
  jac(15, 5) = -cse271*cse585;
  jac(15, 6) = -cse277*cse585;
  jac(15, 7) = -cse278*cse585;
  jac(15, 8) = -cse286*cse585 + cse483;
  jac(15, 9) = -cse287*cse585 + cse586*state.xn[3];
  jac(15, 10) = -cse289*cse585;
  jac(15, 11) = -cse290*cse585;
  jac(15, 12) = -cse296*cse585 - cse298;
  jac(15, 13) = -cse300*cse585;
  jac(15, 14) = -cse301*cse585;
  jac(15, 15) = -cse302 - cse306*cse585 - cse564;
  jac(15, 16) = -cse308*cse585;
  jac(15, 17) = -cse319*cse585 + cse497;
  jac(15, 18) = -cse329*cse585;
  jac(15, 19) = cse585;
  jac(16, 1) = -cse587*cse65;
  jac(16, 2) = -cse192*cse587;
  jac(16, 3) = -cse212*cse587 + cse451 - cse452 - cse525;
  jac(16, 4) = -cse253*cse587 + cse473;
  jac(16, 5) = -cse271*cse587;
  jac(16, 6) = -cse277*cse587;
  jac(16, 7) = -cse278*cse587 - cse282;
  jac(16, 8) = -cse285 - cse286*cse587;
  jac(16, 9) = -cse287*cse587 - cse571;
  jac(16, 10) = -cse289*cse587;
  jac(16, 11) = -cse290*cse587;
  jac(16, 12) = -cse295 - cse296*cse587 + cse489;
  jac(16, 13) = -cse300*cse587;
  jac(16, 14) = -cse301*cse587;
  jac(16, 15) = -cse306*cse587;
  jac(16, 16) = -cse308*cse587 - cse312 - cse572;
  jac(16, 17) = -cse319*cse587 + cse498;
  jac(16, 18) = cse328 - cse329*cse587;
  jac(16, 19) = cse587;
  jac(17, 1) = -cse588*cse65;
  jac(17, 2) = -cse192*cse588;
  jac(17, 3) = -cse212*cse588 - cse454;
  jac(17, 4) = -cse253*cse588 + cse531;
  jac(17, 5) = cse270 - cse271*cse588;
  jac(17, 6) = -cse277*cse588;
  jac(17, 7) = -cse278*cse588;
  jac(17, 8) = -cse286*cse588 - cse483;
  jac(17, 9) = -cse287*cse588;
  jac(17, 10) = -cse289*cse588 - cse536;
  jac(17, 11) = -cse290*cse588;
  jac(17, 12) = -cse296*cse588 - cse490;
  jac(17, 13) = -cse300*cse588;
  jac(17, 14) = -cse301*cse588;
  jac(17, 15) = -cse306*cse588;
  jac(17, 16) = -cse308*cse588;
  jac(17, 17) = -cse319*cse588 - cse499 - cse540;
  jac(17, 18) = -cse329*cse588;
  jac(17, 19) = cse588;
  jac(18, 1) = -cse176 - cse589*cse65;
  jac(18, 2) = cse104 - cse192*cse589;
  jac(18, 3) = -cse212*cse589 - cse448;
  jac(18, 4) = -cse253*cse589 - cse532;
  jac(18, 5) = -cse271*cse589;
  jac(18, 6) = -cse277*cse589;
  jac(18, 7) = -cse278*cse589 + cse282;
  jac(18, 8) = -cse286*cse589;
  jac(18, 9) = -cse287*cse589;
  jac(18, 10) = -cse289*cse589;
  jac(18, 11) = -cse290*cse589;
  jac(18, 12) = -cse296*cse589 + cse297;
  jac(18, 13) = -cse300*cse589;
  jac(18, 14) = -cse301*cse589;
  jac(18, 15) = -cse306*cse589;
  jac(18, 16) = -cse308*cse589 + cse309;
  jac(18, 17) = -cse319*cse589;
  jac(18, 18) = -cse326 - cse329*cse589 - cse506;
  jac(18, 19) = cse589;
  jac(19, 1) = -cse1119*cse65 - 1.6737729999999998e-24*cse934 + cse935*(1.7e-24*chi*cse142*cse599*cse603*d2g + 1.461185424e-21*chi*cse248*cse249*cse952*cse959*state.xn[3] + 1.7623947000000002e-12*crate*cse183*cse683*cse948*state.xn[5] + 1.602177e-12*crate*cse683 + 1.7623947000000002e-12*crate*cse684*cse972*state.xn[5] + 1.602177e-12*crate*cse972*state.xn[0] + 3.2043540000000001e-12*crate*cse972*state.xn[3] + 3.0000000000000002e-15*cse10*cse648*d2g + 2.0187430200000004e-26*cse10*cse952*cse959*d2g*state.xn[0] - 7.1777529600000012e-12*cse511 - 7.1777529600000012e-12*cse512 - cse598 - cse609*cse618 + 3.0000000000000002e-15*cse648*d2g*state.xn[0] + 4.39089220777172e-29*cse654*cse659*cse661*cse665*cse947*state.xn[8] - 4.39089220777172e-29*cse654*cse946 - cse687*cse716 - cse717 - cse937 - cse941 - cse945 - cse951*(-cse662*cse949 + cse949*cse950) - cse988*(-cse687*cse703 - cse712*(cse975*cse978 + cse983*(cse687*cse707 - cse979*cse981)) - cse986) - state.xn[11]*(cse1000*cse773 + cse1005*cse774 + cse1008*cse1009 + cse1008*cse1010) - state.xn[3]*((cse804) ? (
   0
)
: (
   cse835*cse991
)) - state.xn[7]*((cse932) ? (
   cse1019*cse930 + cse1024*cse931 + cse1027*cse1028 + cse1027*cse1029
)
: (
   0
)));
  jac(19, 2) = -cse1119*cse192 - 1.6728620616289998e-24*cse934 + cse935*(cse1132 - state.xn[3]*((cse804) ? (
   0
)
: (
   cse826*cse991
)));
  jac(19, 3) = -cse1119*cse212 - 9.1093837099999994e-28*cse934 + cse935*(1.7e-24*chi*cse10*cse1039*cse142*cse599*d2g*(0.00025272365919999998*cse1040*cse213 + 0.036605692000000002*cse1041*cse213) + 4.6527220080000002e-11*crate*cse106*cse1137*cse669*cse671*cse684*state.xn[0]*state.xn[5] + 4.22974728e-11*crate*cse106*cse1137*cse669*cse671*std::pow(state.xn[0], 2) + 8.4594945600000001e-11*crate*cse106*cse1137*cse669*cse671*state.xn[0]*state.xn[3] + 6.0732876035249664e-19*cse1034*cse1036*cse1133*cse612*cse616*state.xn[0]*state.xn[2] - 7.2151085033443149e-29*cse1044*cse1134*cse1136*cse159*cse2*state.xn[2] - cse1046*(5.9301787935659931e-17*cse1034*cse1036*cse1133*cse612*state.xn[2] - cse607*cse614) - 2.1789607199999999e-11*cse223 - cse619*state.xn[0] - cse630 - cse631*(0.086512999999999993*chi*cse1043*cse1135*cse142*cse2*cse624 - cse1045*cse1136*cse622) + 2.6241084038530201e-25*cse659*cse661*cse665*cse71*cse947*state.xn[8] - cse686*cse716 - 2.6241084038530201e-25*cse71*cse946 - cse951*(-cse1138*cse662 + cse1138*cse950) - cse988*(-cse686*cse703 - cse686*cse712*cse982) - state.xn[11]*(cse1009*cse1145 + cse1010*cse1145 + cse1142*cse773 + cse1144*cse774) - state.xn[3]*((cse804) ? (
   0
)
: (
   cse843*cse991
)) - state.xn[7]*((cse932) ? (
   cse1028*cse1158 + cse1029*cse1158 + cse1152*cse930 + cse1155*cse931
)
: (
   0
)));
  jac(19, 4) = -cse1119*cse253 - 3.3475459999999996e-24*cse934 + cse935*(1.461185424e-21*chi*cse1167*cse248*cse249*cse952*state.xn[3] + 3.3999999999999999e-24*chi*cse142*cse599*cse603*d2g + 5.6999999999999997e-11*chi*cse248*cse249*cse649 + 1.7623947000000002e-12*crate*cse1174*cse684*state.xn[5] + 1.602177e-12*crate*cse1174*state.xn[0] + 3.2043540000000001e-12*crate*cse1174*state.xn[3] + 1.7623947000000002e-12*crate*cse260*cse683*cse948*state.xn[5] + 3.2043540000000001e-12*crate*cse683 + 2.0187430200000004e-26*cse10*cse1167*cse952*d2g*state.xn[0] - cse1052*cse658 - cse1055*cse1165 - cse1160 - cse1162 - cse1164 - 7.1777529600000012e-12*cse232 - 7.1777529600000012e-12*cse240 - 1.4355505920000002e-11*cse242 - 7.1777529600000012e-12*cse527 + 6.0000000000000005e-15*cse648*d2g*state.xn[0] - cse716 - cse848 - cse951*(-cse1060*cse1165 - cse658*cse663) - cse988*(-cse1177 - cse703 - cse712*(cse1175*cse978 + cse983*(-cse1176*cse979 + cse707))) - state.xn[11]*(cse1009*cse1183 + cse1010*cse1183 + cse1179*cse773 + cse1182*cse774) - state.xn[3]*((cse804) ? (
   0
)
: (
   cse822*cse991
)) - state.xn[7]*((cse932) ? (
   cse1028*cse1189 + cse1029*cse1189 + cse1185*cse930 + cse1188*cse931
)
: (
   0
)));
  jac(19, 5) = -cse1119*cse271 - 3.3466350616289996e-24*cse934 + cse935*(cse1122*cse259 + cse1124*cse1191 + cse1125*cse1191 + cse1127*cse1192 + cse1128*cse1192 + cse1130*cse1192 - cse1160 - cse1162 - cse1164 + 3.3999999999999999e-24*cse1190 - 7.1777529600000012e-12*cse533 - 7.1777529600000012e-12*cse534 + cse57*cse648 - cse988*(-cse1177 - cse712*(-cse1131*cse1176 + cse1175*cse976*cse977)));
  jac(19, 6) = -cse1119*cse277 - 6.6464729999999996e-24*cse934 + cse935*(cse1121*cse1129 - cse1193);
  jac(19, 7) = -cse1119*cse278 - cse1193*cse935 - 6.6455620616289996e-24*cse934;
  jac(19, 8) = -cse1119*cse286 - cse933*cse935 - 1.9944730000000003e-23*cse934;
  jac(19, 9) = -cse1119*cse287 - cse665*cse668*cse935 - 1.9943819061629003e-23*cse934;
  jac(19, 10) = -cse1119*cse289 - 4.6512360000000008e-23*cse934 + cse935*(-cse688*cse715 - cse988*(cse1194*cse688*cse984*cse985 - cse711*(cse1194*cse692*cse704*cse984*cse985 - cse1195*cse705) - cse712*(-cse1131*cse1195 + cse1194*cse976*cse977)));
  jac(19, 11) = -cse1119*cse290 - cse1196 - 4.8185222061629005e-23*cse934;
  jac(19, 12) = -cse1119*cse296 - cse775*cse935 - 2.6567630000000005e-23*cse934;
  jac(19, 13) = -cse1119*cse300 - 4.6637070000000001e-23*cse934;
  jac(19, 14) = -cse1119*cse301 - 4.6636159061629004e-23*cse934;
  jac(19, 15) = -cse1119*cse306 - cse1196 - 2.1618503000000003e-23*cse934;
  jac(19, 16) = -cse1119*cse308 - cse1196 - 2.8241403000000005e-23*cse934;
  jac(19, 17) = -cse1119*cse319 - 5.0204080616289998e-24*cse934 + cse935*(cse1122*cse322 + cse1124*cse1197 + cse1125*cse1197 + cse1127*cse1198 + cse1128*cse1198 + cse1130*cse1198 - 5.6994379667262474e-29*cse1159 - 9.6000000000000005e-32*cse1161 - 300.0*cse1163 + 5.0999999999999995e-24*cse1190 - 7.1777529600000012e-12*cse317 + cse537*cse648 - 7.1777529600000012e-12*cse538 - cse988*(cse1199*cse688*cse984*cse985 - cse711*(cse1199*cse692*cse704*cse984*cse985 - cse1200*cse705) - cse712*(-cse1131*cse1200 + cse1199*cse976*cse977)));
  jac(19, 18) = -cse1119*cse329 - 2.6566719061629008e-23*cse934;
  jac(19, 19) = cse1119;
  jac(20, 13) = -cse492;
  jac(20, 20) = cse583;
  jac(21, 13) = -cse491*state.rn[2*0+1];
  jac(21, 21) = cse583;
  // $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
