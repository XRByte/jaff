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
  const amrex::Real cse35 = cse34*state.xn[2]*state.xn[16];
  const amrex::Real cse36 = cse19*d2g;
  const amrex::Real cse37 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[1]*state.xn[11];
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
  const amrex::Real cse77 = cse76*state.rn[2*0];
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
  const amrex::Real cse124 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse125 = (1.0/100.0)*state.T;
  const amrex::Real cse126 = std::log(cse125);
  const amrex::Real cse127 = state.T < 500;
  const amrex::Real cse128 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse129 = 4.5299999999999998e-6*cse124 + 7.5799999999999997e-10*std::pow(cse125, 0.0086999999999999994*cse126 + 0.12809999999999999)*state.xn[0] + (0.75*((cse127) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse128
)) + 0.25*((cse127) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse128
)))*state.xn[3];
  const amrex::Real cse130 = cse129*std::exp(-91.209999999999994*cse14)/(cse129 + 2.3e-6);
  const amrex::Real cse131 = 0.001*state.T;
  const amrex::Real cse132 = cse131 + 1;
  const amrex::Real cse133 = std::pow(cse132, -2);
  const amrex::Real cse134 = 1.0/cse132;
  const amrex::Real cse135 = 1.0/(cse23*(cse104 + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse133 + 0.055500000000000001*cse134 - 11.06)*cse25 + std::pow(10, -2.0230000000000001*cse133 - 3.6709999999999998*cse134 - 11.08)*cse24) + 1);
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
  const amrex::Real cse145 = 0.042999999999999997*cse126;
  const amrex::Real cse146 = 4.34e-10*std::pow(cse125, 0.755 - 0.16*cse126)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse125, 0.52500000000000002*cse126 + 1.198) + 5.2499999999999998e-13*std::pow(cse125, cse145 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse147 = 3.5700000000000001e-10*std::pow(cse125, 0.41899999999999998 - 0.0030000000000000001*cse126)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse125, 0.025000000000000001*cse126 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse125, cse145 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse148 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse149 = 3.1899999999999998e-10*std::pow(cse125, 0.36899999999999999 - 0.0060000000000000001*cse126)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse125, 0.041000000000000002*cse126 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse125, 0.058000000000000003*cse126 + 0.23699999999999999))*state.xn[3];
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
  const amrex::Real cse161 = std::log(cse131);
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
  const amrex::Real cse198 = 0.045999999999999999*cse126;
  const amrex::Real cse199 = 1.7258000000000001e-6*cse124;
  const amrex::Real cse200 = 8.9000000000000003e-11*std::pow(cse125, cse198 + 0.22800000000000001)*state.xn[0] + cse199*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(0.0070527699999999999*cse70 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse70*(cse70*(cse70*(0.093813800000000003*cse70 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse125, 0.037999999999999999*cse126 + 0.16900000000000001) + 2.15e-11*std::pow(cse125, 0.048000000000000001*cse126 - 0.01))*state.xn[3];
  const amrex::Real cse201 = 2.6400000000000002e-10*std::pow(cse125, cse198 + 0.23100000000000001)*state.xn[0] + cse199*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(0.00235272*cse70 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse70*(cse70*(cse70*(0.097857299999999994*cse70 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse125, 0.031*cse126 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse125, 0.064000000000000001*cse126 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse202 = 2.8763333333333333e-6*cse124*std::exp(((cse182) ? (
   cse70*(cse70*(cse70*(-0.00065632500000000003*cse70 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse70*(cse70*(cse70*(0.10508000000000001*cse70 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse125, 0.057000000000000002*cse126 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse125, 0.049000000000000002*cse126 - 0.0040000000000000001) + 1.675e-11*std::pow(cse125, 0.10199999999999999*cse126 - 0.085000000000000006))*state.xn[3];
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
  ydot(4) = -cse100 - cse101 + cse102 - cse103 - cse104*state.xn[3] - cse106 + cse11 + 3.0000000000000002e-15*cse19*d2g*state.xn[0] - cse31 + 4.5427507615910105e-7*cse34*state.xn[2]*state.xn[16] - cse51 - cse55 - cse99;
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
  ydot(19) = 1.7e-24*chi*cse19*cse66*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse122, 0.43099999999999999)*(0.0043*std::pow(cse122, 0.69199999999999995) + 1) + 1) + 5.6999999999999997e-11*chi*cse39*cse40*(2.5634832e-11*cse135 + 6.4087080000000003e-13)*state.xn[3] + 1.602177e-12*crate*cse144*state.xn[0] + 3.2043540000000001e-12*crate*cse144*state.xn[3] + 1.7623947000000002e-12*crate*cse144*state.xn[5]/cse58 - 7.1777529600000012e-12*cse103 - 1.3659996292638161e-10*cse119*cse121*cse62/(1.3338116951315871e-8*cse120*cse121 + 1) - 5.7927337833400002e-20*cse130*state.xn[8]/(2*cse130 + 1) - cse137*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse137*cse141 + cse139*std::pow(cse137/cse142, alpha_co_interp2d(state.T, cse140))*(-cse138*cse141*cse142 + 1) + cse139) + 3.0000000000000002e-15*cse19*d2g*(6.7291434000000005e-12*cse135 + 3.2043540000000002e-13)*state.xn[0] - 7.1777529600000012e-12*cse31 - 1.8998126555754159e-29*cse36*std::pow(state.T, 0.45350000000000001 + 2.234/cse123)*std::exp(cse123*(1.4419999999999999 - 0.050889999999999998*cse123))*state.xn[2] - 2.1789607199999999e-11*cse63 - (2.3872025442750958e-19*cse152*cse157 + 2.8023005140019997e-18*cse156*cse157)*state.xn[11] - ((((cse127 && cse158 && cse159) ? (
   cse160 || cse177 + cse180 < 1.0e-100
)
: (
   ((cse158 && cse159) ? (
      cse160 || cse180 + cse181 < 1.0e-100
   )
   : (
      ((cse127 && cse159 && cse182) ? (
         cse160 || cse177 + cse183 < 1.0e-100
      )
      : (
         ((cse159 && cse182) ? (
            cse160 || cse181 + cse183 < 1.0e-100
         )
         : (
            ((cse127 && cse159) ? (
               cse160 || cse177 + cse184 < 1.0e-100
            )
            : (
               ((cse159) ? (
                  cse160 || cse181 + cse184 < 1.0e-100
               )
               : (
                  ((cse127) ? (
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
   cse187/(cse187/(std::pow(10.0, 0.19191374999999999*cse189 - 0.16596184*cse190 - 0.81520437999999995*cse192 + 2.1892372*cse193 + 0.29036281000000003*cse194 - 23.689236999999999)*cse178 + std::pow(10.0, 0.32168730000000001*cse189 - 0.51002221000000003*cse190 + 0.015391166*cse192 + 1.5714710999999999*cse193 - 0.23619984999999999*cse194 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse17*cse188 + 0.43693353000000001*cse173*cse191 - 0.033638326000000003*cse189 - 0.14913216000000001*cse190 - 0.77151435999999995*cse192 - 23.962112000000001)*state.xn[3] + ((cse127) ? (
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
  const amrex::Real cse186 = chi*cse184*cse185;
  const amrex::Real cse187 = std::exp(-23*cse3);
  const amrex::Real cse188 = cse187*std::pow(state.T, -1.3);
  const amrex::Real cse189 = 4.2899999999999994e-13*cse188;
  const amrex::Real cse190 = cse138*cse67*state.xn[3];
  const amrex::Real cse191 = 1.6000000000000001e-9*cse142;
  const amrex::Real cse192 = cse191*state.xn[17];
  const amrex::Real cse193 = cse145*cse190;
  const amrex::Real cse194 = cse129*cse192 - cse129*cse193 + cse143*cse190;
  const amrex::Real cse195 = cse81*cse83;
  const amrex::Real cse196 = 2.4500000000000003e-11*cse195;
  const amrex::Real cse197 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse198 = -2.2999999999999998*cse11 + cse197;
  const amrex::Real cse199 = -cse198;
  const amrex::Real cse200 = cse199*cse89;
  const amrex::Real cse201 = cse131*cse132*cse66*state.xn[11]*state.xn[16];
  const amrex::Real cse202 = cse133*cse201;
  const amrex::Real cse203 = cse196 - cse200 - cse202;
  const amrex::Real cse204 = cse10*((cse20) ? (
   0
)
: (
   -cse172
));
  const amrex::Real cse205 = cse50*(-cse168 - cse204);
  const amrex::Real cse206 = cse205*cse22;
  const amrex::Real cse207 = cse30*((cse28) ? (
   cse54*(cse169*cse204 + cse170 + cse205*cse49 + cse206*cse31)
)
: (
   0
));
  const amrex::Real cse208 = ((cse28) ? (
   cse59*(cse180*cse204 + cse181 + cse205*cse6 + cse206*cse25)
)
: (
   0
));
  const amrex::Real cse209 = cse160*cse163*state.xn[3];
  const amrex::Real cse210 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse211 = cse209 - cse210;
  const amrex::Real cse212 = -cse197;
  const amrex::Real cse213 = cse212*cse89;
  const amrex::Real cse214 = cse196 - cse213;
  const amrex::Real cse215 = cse158*state.xn[3];
  const amrex::Real cse216 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse217 = 1.3500000000000001e-9*cse216;
  const amrex::Real cse218 = cse217*state.xn[15];
  const amrex::Real cse219 = cse215 + cse218;
  const amrex::Real cse220 = std::exp(0.11*cse3);
  const amrex::Real cse221 = cse220*state.xn[15];
  const amrex::Real cse222 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse223 = cse221*cse222;
  const amrex::Real cse224 = 6.5199999999999996*crate;
  const amrex::Real cse225 = cse149*state.xn[2];
  const amrex::Real cse226 = cse113 + cse225;
  const amrex::Real cse227 = cse142*state.xn[16];
  const amrex::Real cse228 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse229 = cse227*cse228;
  const amrex::Real cse230 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse231 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse232 = cse104*state.xn[1];
  const amrex::Real cse233 = cse231 + cse232;
  const amrex::Real cse234 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse235 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse236 = cse0*state.xn[0];
  const amrex::Real cse237 = -cse236;
  const amrex::Real cse238 = cse113 + cse124;
  const amrex::Real cse239 = cse217*state.xn[6];
  const amrex::Real cse240 = cse220*cse222*state.xn[7];
  const amrex::Real cse241 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse242 = cse239 + cse240 + cse241;
  const amrex::Real cse243 = 3*cse21;
  const amrex::Real cse244 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse245 = cse50*(-cse243 - cse244);
  const amrex::Real cse246 = cse22*cse245;
  const amrex::Real cse247 = cse30*((cse28) ? (
   cse54*(cse169*cse243 + cse169*cse244 + cse245*cse49 + cse246*cse31)
)
: (
   0
));
  const amrex::Real cse248 = ((cse28) ? (
   cse59*(cse180*cse243 + cse180*cse244 + cse245*cse6 + cse246*cse25)
)
: (
   0
));
  const amrex::Real cse249 = cse228*state.xn[11];
  const amrex::Real cse250 = cse142*cse249;
  const amrex::Real cse251 = 6.8999999999999995*cse36 + 4.5*cse38;
  const amrex::Real cse252 = -cse251;
  const amrex::Real cse253 = cse252*cse89;
  const amrex::Real cse254 = 3.6750000000000002e-11*cse195;
  const amrex::Real cse255 = cse250 - cse253 + cse254;
  const amrex::Real cse256 = cse62*state.xn[0];
  const amrex::Real cse257 = cse191*state.xn[3];
  const amrex::Real cse258 = cse129*cse257;
  const amrex::Real cse259 = std::pow(state.T, -2);
  const amrex::Real cse260 = 2.8349999999999999e-12*cse157*cse259*state.xn[3]*state.xn[6];
  const amrex::Real cse261 = -cse260;
  const amrex::Real cse262 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse263 = std::sqrt(cse3);
  const amrex::Real cse264 = std::sqrt(3)*cse263*cse3*state.xn[15];
  const amrex::Real cse265 = 1.7697487500000004e-8*cse264*state.xn[6];
  const amrex::Real cse266 = cse259*cse26;
  const amrex::Real cse267 = cse13*cse3;
  const amrex::Real cse268 = cse12*cse267;
  const amrex::Real cse269 = cse3*cse8;
  const amrex::Real cse270 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(-3.2400000000000002*cse268 + 1.3*cse3*cse8) + cse18*cse7*(0.65400000000000003*cse268 + 0.41599999999999998*cse269)
));
  const amrex::Real cse271 = cse10*cse270;
  const amrex::Real cse272 = cse271*cse50;
  const amrex::Real cse273 = cse21*cse270*cse34*cse50;
  const amrex::Real cse274 = ((cse28) ? (
   cse54*(cse169*cse271 + 1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 + 53300.0*cse266 - cse272*cse49 - cse273*cse31)
)
: (
   0
));
  const amrex::Real cse275 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse276 = 1.0/cse2;
  const amrex::Real cse277 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse278 = ((cse28) ? (
   cse59*(cse180*cse271 + 149925037481.25937*cse24*cse276*(3.3349999999999998e-12*cse276*cse5 + 4.2414529999999999e-7*cse277*cse5)*std::exp(cse4) - cse25*cse273 + 43900.0*cse266 - cse272*cse6)
)
: (
   0
));
  const amrex::Real cse279 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse280 = cse103*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse281 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse282 = cse187*state.xn[3]*state.xn[8];
  const amrex::Real cse283 = cse281*cse282;
  const amrex::Real cse284 = cse161*state.xn[3];
  const amrex::Real cse285 = 3.7768240343347639e-14*cse162*cse284;
  const amrex::Real cse286 = 7.3920000000000002e-11*cse284*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse287 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse288 = 227*cse105*cse259*state.xn[1];
  const amrex::Real cse289 = cse276*cse66;
  const amrex::Real cse290 = cse139*cse289;
  const amrex::Real cse291 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse292 = cse221*state.xn[7];
  const amrex::Real cse293 = 8.7465066538127767e-11*cse292*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse292*std::pow(state.T, -1.339);
  const amrex::Real cse294 = cse291 + cse293;
  const amrex::Real cse295 = state.xn[0]*state.xn[2];
  const amrex::Real cse296 = cse295*((cse28) ? (
   7.9971871901929577e-69*cse98*std::pow(state.T, 12.536555999999999) + cse99*(-11.478657500000001*cse3*cse91 + 4.6894649399999997*cse3*cse92 - 1.1508224*cse3*cse93 + 0.1741279885*cse3*cse94 - 0.015791857020000001*cse3*cse95 + 0.00078368076500000001*cse3*cse96 - 1.6313198799999999e-5*cse3*cse97)
)
: (
   0
));
  const amrex::Real cse297 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse298 = 7.3220359950196314e-6*cse106*cse108*cse297;
  const amrex::Real cse299 = 0.0005114031087522854*std::pow(cse107, -3.242)*cse297*std::pow(cse3, 1.907);
  const amrex::Real cse300 = 0.5*cse3;
  const amrex::Real cse301 = cse3*std::log(cse71);
  const amrex::Real cse302 = cse154*(-1.1557571281629217e-5*cse152*cse3 - cse79*(8.0679819999999989*cse74*std::pow(state.T, -0.98414000000000001) + cse76*(cse153*cse300 - 1.102e-5*cse301)));
  const amrex::Real cse303 = 9.0566037735849048e-14*cse290;
  const amrex::Real cse304 = cse142*cse303;
  const amrex::Real cse305 = cse131*cse303;
  const amrex::Real cse306 = cse129*cse305;
  const amrex::Real cse307 = -cse304 + cse306;
  const amrex::Real cse308 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse309 = 1.1264150943396225e-13*cse201*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse227*cse308;
  const amrex::Real cse310 = cse296 + cse298 - cse299 - cse302 + cse307 + cse309;
  const amrex::Real cse311 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse312 = -cse196;
  const amrex::Real cse313 = cse112 + cse85;
  const amrex::Real cse314 = -cse275 + cse280 + cse288;
  const amrex::Real cse315 = crate*cse9;
  const amrex::Real cse316 = cse315*cse88;
  const amrex::Real cse317 = 5.4939999999999998e-7*cse72 + 0.49559999999999998;
  const amrex::Real cse318 = std::pow(cse71, -cse317);
  const amrex::Real cse319 = cse318*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse320 = 5115.0*cse319;
  const amrex::Real cse321 = cse320 + 1;
  const amrex::Real cse322 = std::pow(cse70, 1.512);
  const amrex::Real cse323 = 7.104740267838494e-7*cse322;
  const amrex::Real cse324 = cse321*cse323 + 1;
  const amrex::Real cse325 = 1.0/cse324;
  const amrex::Real cse326 = 5.5720000000000001e-12*cse325;
  const amrex::Real cse327 = d2g*state.xn[6];
  const amrex::Real cse328 = cse326*cse327;
  const amrex::Real cse329 = crate*cse36;
  const amrex::Real cse330 = cse329*state.xn[7];
  const amrex::Real cse331 = 0.00013329999999999999*cse72 + 0.81200000000000006;
  const amrex::Real cse332 = std::pow(cse71, -cse331);
  const amrex::Real cse333 = cse332*std::pow(state.T, 0.04845);
  const amrex::Real cse334 = 433.10000000000002*cse333;
  const amrex::Real cse335 = cse334 + 1;
  const amrex::Real cse336 = std::pow(cse70, 1.1279999999999999);
  const amrex::Real cse337 = 0.011078790689599145*cse336;
  const amrex::Real cse338 = cse335*cse337 + 1;
  const amrex::Real cse339 = 1.0/cse338;
  const amrex::Real cse340 = 4.5579999999999993e-11*cse339;
  const amrex::Real cse341 = d2g*state.xn[8];
  const amrex::Real cse342 = 1040.0*cse330 + cse340*cse341;
  const amrex::Real cse343 = cse329*state.xn[12];
  const amrex::Real cse344 = 7.538e-5*cse72 + 0.89639999999999997;
  const amrex::Real cse345 = std::pow(cse71, -cse344);
  const amrex::Real cse346 = cse345*std::pow(state.T, 1.635e-6);
  const amrex::Real cse347 = 43750.0*cse346;
  const amrex::Real cse348 = cse347 + 1;
  const amrex::Real cse349 = std::pow(cse70, 1.8740000000000001);
  const amrex::Real cse350 = 1.534817233071468e-7*cse349;
  const amrex::Real cse351 = cse348*cse350 + 1;
  const amrex::Real cse352 = 1.0/cse351;
  const amrex::Real cse353 = 2.1659999999999999e-12*cse352;
  const amrex::Real cse354 = d2g*state.xn[13];
  const amrex::Real cse355 = 16800.0*cse343 + cse353*cse354;
  const amrex::Real cse356 = cse328 + cse342 + cse355;
  const amrex::Real cse357 = cse111*cse315;
  const amrex::Real cse358 = cse356 - cse357;
  const amrex::Real cse359 = cse72*cse8;
  const amrex::Real cse360 = 0.044330000000000001*cse359 - 0.28520000000000001;
  const amrex::Real cse361 = cse359*cse360 - 1.6759999999999999;
  const amrex::Real cse362 = cse359*cse361 + 11.19;
  const amrex::Real cse363 = cse362*state.xn[6];
  const amrex::Real cse364 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse365 = cse321*cse322;
  const amrex::Real cse366 = -cse317;
  const amrex::Real cse367 = 5.5720000000000001e-12*cse110*state.xn[6]/std::pow(cse324, 2);
  const amrex::Real cse368 = cse363*cse364 + cse367*(0.0036340746469993896*cse319*cse322*cse366*cse66 + 1.0742367284971802e-6*cse365*cse66);
  const amrex::Real cse369 = std::exp(-15230.0*cse3);
  const amrex::Real cse370 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse371 = std::exp(-12.17*cse3);
  const amrex::Real cse372 = 1.6339999999999999e-6*cse369 + 9.7930000000000004e-9*cse370 + 6.3460000000000002e-9*cse371;
  const amrex::Real cse373 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse374 = std::exp(-49550.0*cse3);
  const amrex::Real cse375 = 0.15970000000000001*cse374;
  const amrex::Real cse376 = cse375 + 1.7848999999999999;
  const amrex::Real cse377 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse378 = 0.21509999999999996 - cse375;
  const amrex::Real cse379 = std::pow(cse373, -cse376)*std::pow(cse377, -cse378);
  const amrex::Real cse380 = 2.4460185352936309e-10*cse276*cse379;
  const amrex::Real cse381 = cse277*cse372 + cse380;
  const amrex::Real cse382 = cse335*cse336;
  const amrex::Real cse383 = -cse331;
  const amrex::Real cse384 = 4.5579999999999993e-11*cse110*state.xn[8]/std::pow(cse338, 2);
  const amrex::Real cse385 = cse381*state.xn[8] + cse384*(4.7982242476653898*cse333*cse336*cse383*cse66 + 0.012496875897867835*cse382*cse66);
  const amrex::Real cse386 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse387 = cse348*cse349;
  const amrex::Real cse388 = -cse344;
  const amrex::Real cse389 = 2.1659999999999999e-12*cse110*state.xn[13]/std::pow(cse351, 2);
  const amrex::Real cse390 = cse386*state.xn[13] + cse389*(0.0067148253946876721*cse346*cse349*cse388*cse66 + 2.8762474947759312e-7*cse387*cse66);
  const amrex::Real cse391 = cse146 + cse148;
  const amrex::Real cse392 = 1.3007021411393064e-6*cse135;
  const amrex::Real cse393 = cse11*state.xn[2];
  const amrex::Real cse394 = cse393 < 1.0e-50;
  const amrex::Real cse395 = ((cse394) ? (
   0
)
: (
   -0.00011320754716981131*cse137*state.xn[3]
));
  const amrex::Real cse396 = ((cse394) ? (
   10000000000.0
)
: (
   cse128
));
  const amrex::Real cse397 = cse396 + 1;
  const amrex::Real cse398 = 1.0/cse397;
  const amrex::Real cse399 = cse249*state.xn[16];
  const amrex::Real cse400 = cse398*cse399;
  const amrex::Real cse401 = cse395*cse400;
  const amrex::Real cse402 = cse396*cse399/std::pow(cse397, 2);
  const amrex::Real cse403 = cse395*cse402;
  const amrex::Real cse404 = cse401 - cse403;
  const amrex::Real cse405 = cse134 + cse404;
  const amrex::Real cse406 = cse392*state.xn[16] + cse405;
  const amrex::Real cse407 = cse325*cse327;
  const amrex::Real cse408 = 1.1144e-11*cse407;
  const amrex::Real cse409 = 2*cse65;
  const amrex::Real cse410 = cse199*cse315;
  const amrex::Real cse411 = crate*cse11;
  const amrex::Real cse412 = 16800.0*cse411;
  const amrex::Real cse413 = cse352*cse354;
  const amrex::Real cse414 = 33600.0*cse343 + 4.3319999999999997e-12*cse413;
  const amrex::Real cse415 = -cse412*state.xn[12] + cse414;
  const amrex::Real cse416 = 3.2999999999999996e-13*cse188;
  const amrex::Real cse417 = cse416*state.xn[8];
  const amrex::Real cse418 = 1040.0*cse411;
  const amrex::Real cse419 = -cse418*state.xn[7];
  const amrex::Real cse420 = cse339*cse341;
  const amrex::Real cse421 = 2080.0*cse330 + 9.1159999999999986e-11*cse420;
  const amrex::Real cse422 = cse417 + cse419 + cse421;
  const amrex::Real cse423 = ((cse394) ? (
   0
)
: (
   cse127
));
  const amrex::Real cse424 = cse400*cse423;
  const amrex::Real cse425 = cse402*cse423;
  const amrex::Real cse426 = cse424 - cse425;
  const amrex::Real cse427 = cse194 + cse426;
  const amrex::Real cse428 = cse192 - cse193;
  const amrex::Real cse429 = cse212*cse315;
  const amrex::Real cse430 = -cse429;
  const amrex::Real cse431 = 1.1000000000000001*crate;
  const amrex::Real cse432 = cse364*state.xn[2];
  const amrex::Real cse433 = cse110*cse326 + cse362*cse432;
  const amrex::Real cse434 = std::pow(cse3, 0.00231);
  const amrex::Real cse435 = std::pow(state.T, -1.5);
  const amrex::Real cse436 = std::exp(-8110.0*cse3);
  const amrex::Real cse437 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse438 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse439 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse440 = 0.00015100000000000001*cse436 + 1.31e-7*cse437 + 3.4e-8*cse438 + 6.9699999999999997e-9*cse439;
  const amrex::Real cse441 = 1.0537934368240927e-9*cse434 + cse435*cse440;
  const amrex::Real cse442 = cse441*state.xn[16];
  const amrex::Real cse443 = -cse442;
  const amrex::Real cse444 = 2.808e-22*ncol_H2;
  const amrex::Real cse445 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse444)*std::exp(-1.6000000000000001e-17*ncol_C)/(cse444 + 1) + 3.8500000000000001*crate + cse418*state.xn[3];
  const amrex::Real cse446 = cse416*state.xn[3];
  const amrex::Real cse447 = cse110*cse340 + cse381*state.xn[2];
  const amrex::Real cse448 = cse446 + cse447;
  const amrex::Real cse449 = cse396*cse398;
  const amrex::Real cse450 = cse449*state.xn[16];
  const amrex::Real cse451 = cse228*cse450;
  const amrex::Real cse452 = cse229 + cse451;
  const amrex::Real cse453 = -cse452;
  const amrex::Real cse454 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse455 = cse454*state.rn[2*0];
  const amrex::Real cse456 = cse412*state.xn[3] + cse455;
  const amrex::Real cse457 = cse110*cse353 + cse386*state.xn[2];
  const amrex::Real cse458 = -cse457;
  const amrex::Real cse459 = -cse113 - cse358;
  const amrex::Real cse460 = cse249*cse449;
  const amrex::Real cse461 = cse441*state.xn[7];
  const amrex::Real cse462 = cse392*state.xn[2];
  const amrex::Real cse463 = cse252*cse315;
  const amrex::Real cse464 = -cse463;
  const amrex::Real cse465 = 1.6715999999999999e-11*cse407;
  const amrex::Real cse466 = 3120.0*cse330 + 1.3673999999999998e-10*cse420;
  const amrex::Real cse467 = 50400.0*cse343 + 6.4979999999999992e-12*cse413;
  const amrex::Real cse468 = cse257 + cse258;
  const amrex::Real cse469 = -cse468;
  const amrex::Real cse470 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse434 + cse435*(1.2246100000000002*cse259*cse436 + 3.4846000000000001e-6*cse259*cse437 + 2.5908000000000001e-7*cse259*cse438 + 9.6185999999999982e-9*cse259*cse439) - 1.5*cse440*std::pow(state.T, -2.5));
  const amrex::Real cse471 = -cse470;
  const amrex::Real cse472 = 6.7636511339243933e-7*cse279;
  const amrex::Real cse473 = cse308*cse450;
  const amrex::Real cse474 = ((cse394) ? (
   0
)
: (
   5.6603773584905654e-5*cse289*state.xn[3]
));
  const amrex::Real cse475 = cse400*cse474;
  const amrex::Real cse476 = -cse475;
  const amrex::Real cse477 = cse402*cse474;
  const amrex::Real cse478 = 7913.1350000000002*cse259*cse374;
  const amrex::Real cse479 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse277*cse379 + cse277*(0.024885819999999999*cse259*cse369 + 7.2272340000000005e-7*cse259*cse370 + 7.7230820000000007e-8*cse259*cse371) - 3.0/2.0*cse372/std::pow(state.T, 5.0/2.0) + cse380*(-0.00035870183966449452*cse276*cse376/cse373 - cse478*std::log(cse373)) + cse380*(-6.1221939997287613*cse276*cse378/cse377 + cse478*std::log(cse377)));
  const amrex::Real cse480 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse481 = 4.9999999999999997e-12*cse363*cse435*state.xn[2];
  const amrex::Real cse482 = cse282*cse287;
  const amrex::Real cse483 = 7.5899999999999998e-12*cse482;
  const amrex::Real cse484 = cse432*state.xn[6]*(cse269*cse361 + cse359*(0.044330000000000001*cse267*cse72 + cse269*cse360));
  const amrex::Real cse485 = cse389*(-1.4381237473879656e-7*cse3*cse387 - cse350*(0.071531250000000005*cse345*std::pow(state.T, -0.99999836499999994) + cse347*(cse300*cse388 - 7.538e-5*cse301)));
  const amrex::Real cse486 = cse367*(-5.3711836424859012e-7*cse3*cse365 - cse323*(0.0019963845000000001*cse318*std::pow(state.T, -0.99999960970000001) + cse320*(cse300*cse366 - 5.4939999999999998e-7*cse301)));
  const amrex::Real cse487 = cse384*(-0.0062484379489339175*cse3*cse382 - cse337*(20.983695000000001*cse332*std::pow(state.T, -0.95155000000000001) + cse334*(cse300*cse383 - 0.00013329999999999999*cse301)));
  const amrex::Real cse488 = cse454*state.xn[12];
  const amrex::Real cse489 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse490 = cse29*state.xn[3];
  const amrex::Real cse491 = cse30*cse55;
  const amrex::Real cse492 = -cse1;
  const amrex::Real cse493 = state.xn[0]*state.xn[3];
  const amrex::Real cse494 = cse493*cse60;
  const amrex::Real cse495 = cse316 - cse61;
  const amrex::Real cse496 = cse122*cse493;
  const amrex::Real cse497 = cse120 + cse357 + cse489 + cse496;
  const amrex::Real cse498 = -cse497;
  const amrex::Real cse499 = 4.5427507615910105e-7*cse135;
  const amrex::Real cse500 = 5.6999999999999997e-11*cse186;
  const amrex::Real cse501 = cse182*cse493;
  const amrex::Real cse502 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse503 = cse159 + cse502;
  const amrex::Real cse504 = cse409 + cse410;
  const amrex::Real cse505 = cse164 + cse202 - cse424 + cse425;
  const amrex::Real cse506 = cse194 + cse428;
  const amrex::Real cse507 = cse208*cse493;
  const amrex::Real cse508 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse509 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse510 = 9.0000000000000011e-15*cse165;
  const amrex::Real cse511 = cse248*cse493;
  const amrex::Real cse512 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse513 = cse250 + cse460 + cse461 + cse512;
  const amrex::Real cse514 = -cse262;
  const amrex::Real cse515 = cse274*cse30;
  const amrex::Real cse516 = cse278*cse493;
  const amrex::Real cse517 = -4.2899999999999994e-13*cse187*cse281*state.xn[3]*state.xn[8] + cse483;
  const amrex::Real cse518 = cse304 - cse306;
  const amrex::Real cse519 = -cse285 + cse286;
  const amrex::Real cse520 = cse309 + cse471 + cse473 + cse476 + cse477 + cse519;
  const amrex::Real cse521 = cse408 + cse503;
  const amrex::Real cse522 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse523 = cse219 + cse433 + cse508 + cse522;
  const amrex::Real cse524 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse525 = cse239 + cse328;
  const amrex::Real cse526 = -cse265;
  const amrex::Real cse527 = cse260 - cse481 + cse484 + cse486 + cse526;
  const amrex::Real cse528 = -cse328;
  const amrex::Real cse529 = cse329*state.xn[9];
  const amrex::Real cse530 = 184.0*cse529;
  const amrex::Real cse531 = -cse530;
  const amrex::Real cse532 = cse342 + cse531;
  const amrex::Real cse533 = 184.0*cse411;
  const amrex::Real cse534 = cse533*state.xn[9];
  const amrex::Real cse535 = 9.8999999999999995e-14*cse188;
  const amrex::Real cse536 = 368.0*cse529;
  const amrex::Real cse537 = -cse536;
  const amrex::Real cse538 = cse421 + cse537;
  const amrex::Real cse539 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av) + cse533*state.xn[3];
  const amrex::Real cse540 = cse235 + cse236;
  const amrex::Real cse541 = 552.0*cse529;
  const amrex::Real cse542 = cse479 + cse487;
  const amrex::Real cse543 = -cse342;
  const amrex::Real cse544 = 9.1500000000000005e-10*cse216;
  const amrex::Real cse545 = cse544*state.xn[15];
  const amrex::Real cse546 = cse544*state.xn[8];
  const amrex::Real cse547 = 1.1994963750000002e-8*cse264*state.xn[8];
  const amrex::Real cse548 = -cse547;
  const amrex::Real cse549 = -cse534 + cse536;
  const amrex::Real cse550 = cse224 + cse509;
  const amrex::Real cse551 = cse524 + cse539;
  const amrex::Real cse552 = cse234 + cse530;
  const amrex::Real cse553 = -cse63;
  const amrex::Real cse554 = -cse230 + cse451;
  const amrex::Real cse555 = -cse473 + cse475 - cse477;
  const amrex::Real cse556 = cse305 + cse314;
  const amrex::Real cse557 = -cse480 + cse485;
  const amrex::Real cse558 = -cse488;
  const amrex::Real cse559 = -cse355;
  const amrex::Real cse560 = 2.3099999999999997e-13*cse188;
  const amrex::Real cse561 = 1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse562 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse563 = 100.0*d2g;
  const amrex::Real cse564 = cse562*cse563;
  const amrex::Real cse565 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse69 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse566 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse567 = cse110*cse566;
  const amrex::Real cse568 = cse565*cse567;
  const amrex::Real cse569 = 2.1789607199999999e-11*cse101;
  const amrex::Real cse570 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse571 = cse71 + 50;
  const amrex::Real cse572 = 0.0043*std::pow(cse571, 0.69199999999999995) + 1;
  const amrex::Real cse573 = 0.049959999999999997*std::pow(cse571, 0.43099999999999999)*cse572 + 1;
  const amrex::Real cse574 = 1.0/cse573;
  const amrex::Real cse575 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse576 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse577 = 1.0/cse576;
  const amrex::Real cse578 = cse575*cse577;
  const amrex::Real cse579 = cse577*state.xn[2];
  const amrex::Real cse580 = cse575*cse579;
  const amrex::Real cse581 = 4.4460389837719569e-9*cse580 + 626500000.0;
  const amrex::Real cse582 = 1.0/cse581;
  const amrex::Real cse583 = std::exp(-118400.0*cse3);
  const amrex::Real cse584 = cse582*cse583;
  const amrex::Real cse585 = 1.3338116951315871e-8*cse584;
  const amrex::Real cse586 = cse580*cse585 + 1;
  const amrex::Real cse587 = 1.0/cse586;
  const amrex::Real cse588 = cse584*cse587;
  const amrex::Real cse589 = 1.3659996292638161e-10*cse588;
  const amrex::Real cse590 = cse578*cse589;
  const amrex::Real cse591 = state.xn[2] + 1.0e-50;
  const amrex::Real cse592 = 1.0/cse591;
  const amrex::Real cse593 = cse2*cse69;
  const amrex::Real cse594 = 1.7*cse593;
  const amrex::Real cse595 = cse592*cse594 + 50;
  const amrex::Real cse596 = std::log(cse595);
  const amrex::Real cse597 = 0.45350000000000001 + 2.234/cse596;
  const amrex::Real cse598 = std::pow(state.T, cse597);
  const amrex::Real cse599 = 1.4419999999999999 - 0.050889999999999998*cse596;
  const amrex::Real cse600 = std::exp(cse596*cse599);
  const amrex::Real cse601 = 1.8998126555754159e-29*cse598*cse600;
  const amrex::Real cse602 = cse110*cse601;
  const amrex::Real cse603 = cse602*state.xn[2];
  const amrex::Real cse604 = 0.001*state.T;
  const amrex::Real cse605 = cse604 + 1;
  const amrex::Real cse606 = std::pow(cse605, -2);
  const amrex::Real cse607 = 1.0/cse605;
  const amrex::Real cse608 = std::pow(10, -2.0230000000000001*cse606 - 3.6709999999999998*cse607 - 11.08);
  const amrex::Real cse609 = cse11*cse608;
  const amrex::Real cse610 = cse609*state.xn[3];
  const amrex::Real cse611 = std::pow(10, -2.3900000000000001*cse606 + 0.055500000000000001*cse607 - 11.06);
  const amrex::Real cse612 = cse11*cse611;
  const amrex::Real cse613 = cse612*state.xn[0];
  const amrex::Real cse614 = cse610 + cse613;
  const amrex::Real cse615 = 1.0/cse614;
  const amrex::Real cse616 = cse500 + 1.9999999999999999e-7;
  const amrex::Real cse617 = cse615*cse616;
  const amrex::Real cse618 = cse11*cse617 + 1;
  const amrex::Real cse619 = 1.0/cse618;
  const amrex::Real cse620 = 6.7291434000000005e-12*cse619 + 3.2043540000000002e-13;
  const amrex::Real cse621 = 2.5634832e-11*cse619 + 6.4087080000000003e-13;
  const amrex::Real cse622 = cse263*state.xn[2];
  const amrex::Real cse623 = (1.0/100.0)*state.T;
  const amrex::Real cse624 = std::log(cse623);
  const amrex::Real cse625 = 0.0086999999999999994*cse624;
  const amrex::Real cse626 = std::pow(cse623, cse625 + 0.12809999999999999);
  const amrex::Real cse627 = 7.5799999999999997e-10*cse626*state.xn[0];
  const amrex::Real cse628 = state.T < 500;
  const amrex::Real cse629 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse630 = 0.75*((cse628) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse629
)) + 0.25*((cse628) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse629
));
  const amrex::Real cse631 = 4.5299999999999998e-6*cse622 + cse627 + cse630*state.xn[3];
  const amrex::Real cse632 = cse631 + 2.3e-6;
  const amrex::Real cse633 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse634 = cse633/cse632;
  const amrex::Real cse635 = 2*cse634;
  const amrex::Real cse636 = cse631*cse635 + 1;
  const amrex::Real cse637 = 1.0/cse636;
  const amrex::Real cse638 = cse637*state.xn[8];
  const amrex::Real cse639 = cse631*cse634;
  const amrex::Real cse640 = 5.7927337833400002e-20*cse639;
  const amrex::Real cse641 = cse393 + 0.070000000000000007;
  const amrex::Real cse642 = 1.0/cse641;
  const amrex::Real cse643 = std::sqrt(cse393*cse642);
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
  const amrex::Real cse656 = 1.0/cse64;
  const amrex::Real cse657 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse658 = 680.10000000000002/cse657;
  const amrex::Real cse659 = 1.75*cse657;
  const amrex::Real cse660 = cse658*state.xn[2] + cse659*state.xn[0] + state.xn[3];
  const amrex::Real cse661 = std::exp(-state.T);
  const amrex::Real cse662 = 1 - cse661;
  const amrex::Real cse663 = std::pow(cse662, 1000);
  const amrex::Real cse664 = l0_co_interp1d(state.T);
  const amrex::Real cse665 = 1.0/cse664;
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
  const amrex::Real cse677 = cse664*cse675;
  const amrex::Real cse678 = -cse676*cse677 + 1;
  const amrex::Real cse679 = 1.0/cse676;
  const amrex::Real cse680 = cse660*cse679;
  const amrex::Real cse681 = alpha_co_interp2d(state.T, cse673);
  const amrex::Real cse682 = std::pow(cse680, cse681);
  const amrex::Real cse683 = cse665*cse682;
  const amrex::Real cse684 = cse678*cse683;
  const amrex::Real cse685 = cse660*cse675 + cse665 + cse684;
  const amrex::Real cse686 = 1.0/cse685;
  const amrex::Real cse687 = cse663*cse686;
  const amrex::Real cse688 = cse687*state.xn[9];
  const amrex::Real cse689 = 7.1777529600000012e-12*cse490;
  const amrex::Real cse690 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse691 = 5.1199999999999999e-10*cse690;
  const amrex::Real cse692 = 0.0030000000000000001*cse624;
  const amrex::Real cse693 = std::pow(cse623, 0.41899999999999998 - cse692);
  const amrex::Real cse694 = 3.5700000000000001e-10*cse693;
  const amrex::Real cse695 = cse694*state.xn[0];
  const amrex::Real cse696 = 0.025000000000000001*cse624;
  const amrex::Real cse697 = 3.7250000000000001e-11*std::pow(cse623, cse696 + 0.26400000000000001);
  const amrex::Real cse698 = 0.042999999999999997*cse624;
  const amrex::Real cse699 = 1.0274999999999999e-10*std::pow(cse623, cse698 + 0.29599999999999999);
  const amrex::Real cse700 = cse697 + cse699;
  const amrex::Real cse701 = cse691*state.xn[2] + cse695 + cse700*state.xn[3];
  const amrex::Real cse702 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse703 = 1.08e-14*cse702;
  const amrex::Real cse704 = 0.16*cse624;
  const amrex::Real cse705 = std::pow(cse623, 0.755 - cse704);
  const amrex::Real cse706 = 4.34e-10*cse705;
  const amrex::Real cse707 = cse706*state.xn[0];
  const amrex::Real cse708 = 5.2499999999999998e-13*std::pow(cse623, cse698 + 0.88900000000000001);
  const amrex::Real cse709 = 0.52500000000000002*cse624;
  const amrex::Real cse710 = 2.2500000000000003e-12*std::pow(cse623, cse709 + 1.198);
  const amrex::Real cse711 = cse708 + cse710;
  const amrex::Real cse712 = cse703*state.xn[2] + cse707 + cse711*state.xn[3];
  const amrex::Real cse713 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse714 = (1.0/5.0)*cse713;
  const amrex::Real cse715 = cse712*cse714;
  const amrex::Real cse716 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse717 = 4.8599999999999998e-10*cse716;
  const amrex::Real cse718 = 0.0060000000000000001*cse624;
  const amrex::Real cse719 = std::pow(cse623, 0.36899999999999999 - cse718);
  const amrex::Real cse720 = 3.1899999999999998e-10*cse719;
  const amrex::Real cse721 = cse720*state.xn[0];
  const amrex::Real cse722 = 0.041000000000000002*cse624;
  const amrex::Real cse723 = 4.7499999999999998e-11*std::pow(cse623, cse722 + 0.20300000000000001);
  const amrex::Real cse724 = 0.058000000000000003*cse624;
  const amrex::Real cse725 = 1.6725000000000002e-10*std::pow(cse623, cse724 + 0.23699999999999999);
  const amrex::Real cse726 = cse723 + cse725;
  const amrex::Real cse727 = cse717*state.xn[2] + cse721 + cse726*state.xn[3];
  const amrex::Real cse728 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse729 = cse712*cse728;
  const amrex::Real cse730 = (1.0/15.0)*cse729;
  const amrex::Real cse731 = cse701 + 8.9099999999999997e-5;
  const amrex::Real cse732 = cse714*cse727;
  const amrex::Real cse733 = cse701*cse715 + cse727*cse730 + cse731*cse732;
  const amrex::Real cse734 = cse712 + 1.7499999999999998e-5;
  const amrex::Real cse735 = cse727 + 1.34e-10;
  const amrex::Real cse736 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse737 = cse712*cse736;
  const amrex::Real cse738 = (1.0/3.0)*cse737;
  const amrex::Real cse739 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse740 = cse701*cse739;
  const amrex::Real cse741 = (3.0/5.0)*cse740;
  const amrex::Real cse742 = cse732*cse734 + cse734*cse741 + cse735*cse741;
  const amrex::Real cse743 = cse731*cse734 + cse731*cse735 + cse733 + cse735*cse738 + cse742;
  const amrex::Real cse744 = 1.0/cse743;
  const amrex::Real cse745 = 2.3872025442750958e-19*cse744;
  const amrex::Real cse746 = 2.8023005140019997e-18*cse744;
  const amrex::Real cse747 = cse733*cse745 + cse742*cse746;
  const amrex::Real cse748 = state.T < 100.0;
  const amrex::Real cse749 = state.T < 6000.0;
  const amrex::Real cse750 = state.T < 10;
  const amrex::Real cse751 = std::log(cse604);
  const amrex::Real cse752 = cse751*cse8;
  const amrex::Real cse753 = std::pow(cse7, -8);
  const amrex::Real cse754 = std::pow(cse751, 8)*cse753;
  const amrex::Real cse755 = std::pow(cse7, -7);
  const amrex::Real cse756 = std::pow(cse751, 7)*cse755;
  const amrex::Real cse757 = std::pow(cse7, -6);
  const amrex::Real cse758 = std::pow(cse751, 6)*cse757;
  const amrex::Real cse759 = std::pow(cse7, -5);
  const amrex::Real cse760 = std::pow(cse751, 5)*cse759;
  const amrex::Real cse761 = std::pow(cse7, -4);
  const amrex::Real cse762 = std::pow(cse751, 4)*cse761;
  const amrex::Real cse763 = std::pow(cse7, -3);
  const amrex::Real cse764 = std::pow(cse751, 3);
  const amrex::Real cse765 = cse763*cse764;
  const amrex::Real cse766 = cse13*std::pow(cse751, 2);
  const amrex::Real cse767 = std::pow(10.0, 16.815729999999999*cse752 + 70.609154000000004*cse754 + 364.14445999999998*cse756 + 801.81246999999996*cse758 + 983.67575999999997*cse760 + 734.71650999999997*cse762 + 343.1918*cse765 + 96.743155000000002*cse766 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse768 = state.xn[5] + state.xn[6];
  const amrex::Real cse769 = std::pow(10.0, 1.5714710999999999*cse752 + 0.32168730000000001*cse760 - 0.51002221000000003*cse762 - 0.23619984999999999*cse765 + 0.015391166*cse766 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse752 + 0.19191374999999999*cse760 - 0.16596184*cse762 + 0.29036281000000003*cse765 - 0.81520437999999995*cse766 - 23.689236999999999)*cse768 + std::pow(10.0, 2.0943374000000001*cse751*cse8 - 0.033638326000000003*cse760 - 0.14913216000000001*cse762 + 0.43693353000000001*cse763*cse764 - 0.77151435999999995*cse766 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse770 = std::pow(10, 37.383713*cse752 + 3.8479610000000002*cse760 + 20.159831000000001*cse762 + 48.656103000000002*cse765 + 58.145166000000003*cse766 - 16.818342000000001)*state.xn[0] + cse769;
  const amrex::Real cse771 = std::pow(10.0, 1.6802758*cse752 - 6.3701156000000001*cse754 + 6.4380698000000001*cse756 + 8.9167182999999994*cse758 - 8.8077017000000009*cse760 - 4.7274035999999997*cse762 + 4.0406627000000004*cse765 + 0.93310621999999999*cse766 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse772 = state.T < 1000.0;
  const amrex::Real cse773 = std::pow(10.0, 3.5692468000000002*cse751*cse8 - 4.2519023000000002*cse760 - 21.328264000000001*cse762 - 27.850082*cse765 - 11.33286*cse766 - 24.311209000000002)*state.xn[0] + cse769;
  const amrex::Real cse774 = std::pow(10.0, 4.6450521*cse752 + 1.5538288*cse760 - 5.5108049000000001*cse762 + 5.9369081000000001*cse765 - 3.7209846*cse766 - 24.311209000000002)*state.xn[0] + cse769;
  const amrex::Real cse775 = std::pow(10.0, -11.945856385319289*cse13 + 28.69470131298657*cse759 - 56.79810800956944*cse761 + 34.150687525022562*cse763 + 8.3228160852326614*cse8 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.5440891133599637*cse759 - 1.7105157385968939*cse761 + 1.6702447513374001*cse763 + 3.9225864834863131*cse8 - 23.689236999999999)*cse768 + std::pow(10.0, -2.4768712409536775*cse13 - 0.62120210234156448*cse759 - 1.5370575959566981*cse761 + 2.5133588394664677*cse763 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 5.9406293599919788*cse759 - 5.2566362076906961*cse761 - 1.3586848802337363*cse763 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse776 = ((cse628 && cse748 && cse749) ? (
   cse750 || cse767 + cse770 < 1.0e-100
)
: (
   ((cse748 && cse749) ? (
      cse750 || cse770 + cse771 < 1.0e-100
   )
   : (
      ((cse628 && cse749 && cse772) ? (
         cse750 || cse767 + cse773 < 1.0e-100
      )
      : (
         ((cse749 && cse772) ? (
            cse750 || cse771 + cse773 < 1.0e-100
         )
         : (
            ((cse628 && cse749) ? (
               cse750 || cse767 + cse774 < 1.0e-100
            )
            : (
               ((cse749) ? (
                  cse750 || cse771 + cse774 < 1.0e-100
               )
               : (
                  ((cse628) ? (
                     cse750 || std::pow(10.0, 310.58441786958315*cse13 + 7500.646099484944*cse753 + 21588.957206746247*cse755 + 26530.843406310876*cse757 + 18165.631967965237*cse759 + 7572.4886742758581*cse761 + 1974.131269262874*cse763 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse775 < 1.0e-100
                  )
                  : (
                     cse750 || std::pow(10.0, 2.9956460707652877*cse13 - 676.68255490510751*cse753 + 381.69251128040054*cse755 + 295.04162851880631*cse757 - 162.65264843551881*cse759 - 48.723840736518795*cse761 + 23.242975457496804*cse763 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse775 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse777 = ((cse749) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse778 = std::exp(-11700.0*cse777);
  const amrex::Real cse779 = std::exp(-5860.0*cse777);
  const amrex::Real cse780 = std::exp(-510.0*cse777);
  const amrex::Real cse781 = std::exp(-2197000.0*((cse749) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse782 = cse781*((cse749) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse783 = 6.0142468035272636e-8*((cse749) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse784 = 1.0/cse783;
  const amrex::Real cse785 = 1.6e-18*cse778 + 6.7e-19*cse779 + 3.0e-24*cse780 + 4.985670872372847e-33*cse782*cse784;
  const amrex::Real cse786 = std::log(0.001*((cse749) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse787 = std::pow(cse786, 5);
  const amrex::Real cse788 = cse759*cse787;
  const amrex::Real cse789 = std::pow(cse786, 4);
  const amrex::Real cse790 = cse761*cse789;
  const amrex::Real cse791 = std::pow(cse786, 3);
  const amrex::Real cse792 = std::pow(cse786, 2);
  const amrex::Real cse793 = cse13*cse792;
  const amrex::Real cse794 = std::pow(10.0, 0.43693353000000001*cse763*cse791 + 2.0943374000000001*cse786*cse8 - 0.033638326000000003*cse788 - 0.14913216000000001*cse790 - 0.77151435999999995*cse793 - 23.962112000000001);
  const amrex::Real cse795 = cse794*state.xn[3];
  const amrex::Real cse796 = cse786*cse8;
  const amrex::Real cse797 = cse763*cse791;
  const amrex::Real cse798 = std::pow(10.0, 0.32168730000000001*cse788 - 0.51002221000000003*cse790 + 0.015391166*cse793 + 1.5714710999999999*cse796 - 0.23619984999999999*cse797 - 22.089523);
  const amrex::Real cse799 = cse798*state.xn[1];
  const amrex::Real cse800 = std::pow(10.0, 0.19191374999999999*cse788 - 0.16596184*cse790 - 0.81520437999999995*cse793 + 2.1892372*cse796 + 0.29036281000000003*cse797 - 23.689236999999999);
  const amrex::Real cse801 = cse768*cse800;
  const amrex::Real cse802 = std::pow(10, 3.8479610000000002*cse788 + 20.159831000000001*cse790 + 58.145166000000003*cse793 + 37.383713*cse796 + 48.656103000000002*cse797 - 16.818342000000001);
  const amrex::Real cse803 = ((cse749) ? (
   cse748
)
: (
   false
));
  const amrex::Real cse804 = std::pow(10.0, 3.5692468000000002*cse786*cse8 - 4.2519023000000002*cse788 - 21.328264000000001*cse790 - 11.33286*cse793 - 27.850082*cse797 - 24.311209000000002);
  const amrex::Real cse805 = ((cse749) ? (
   cse772
)
: (
   false
));
  const amrex::Real cse806 = std::pow(10.0, 1.5538288*cse788 - 5.5108049000000001*cse790 - 3.7209846*cse793 + 4.6450521*cse796 + 5.9369081000000001*cse797 - 24.311209000000002);
  const amrex::Real cse807 = ((cse803) ? (
   cse802
)
: ((cse805) ? (
   cse804
)
: (
   cse806
)));
  const amrex::Real cse808 = cse753*std::pow(cse786, 8);
  const amrex::Real cse809 = std::pow(cse786, 7);
  const amrex::Real cse810 = cse755*cse809;
  const amrex::Real cse811 = std::pow(cse786, 6);
  const amrex::Real cse812 = cse757*cse811;
  const amrex::Real cse813 = std::pow(10.0, 983.67575999999997*cse788 + 734.71650999999997*cse790 + 96.743155000000002*cse793 + 16.815729999999999*cse796 + 343.1918*cse797 + 70.609154000000004*cse808 + 364.14445999999998*cse810 + 801.81246999999996*cse812 - 21.928795999999998);
  const amrex::Real cse814 = std::pow(10.0, -8.8077017000000009*cse788 - 4.7274035999999997*cse790 + 0.93310621999999999*cse793 + 1.6802758*cse796 + 4.0406627000000004*cse797 - 6.3701156000000001*cse808 + 6.4380698000000001*cse810 + 8.9167182999999994*cse812 - 22.921188999999998);
  const amrex::Real cse815 = ((cse628) ? (
   cse813
)
: (
   cse814
));
  const amrex::Real cse816 = cse795 + cse799 + cse801 + cse807*state.xn[0] + cse815*state.xn[2];
  const amrex::Real cse817 = 1.0/cse816;
  const amrex::Real cse818 = cse785*cse817 + 1;
  const amrex::Real cse819 = 1.0/cse818;
  const amrex::Real cse820 = ((cse776) ? (
   0.0
)
: (
   cse785*cse819
));
  const amrex::Real cse821 = 0.045999999999999999*cse624;
  const amrex::Real cse822 = std::pow(cse623, cse821 + 0.23100000000000001);
  const amrex::Real cse823 = 2.6400000000000002e-10*cse822;
  const amrex::Real cse824 = cse823*state.xn[0];
  const amrex::Real cse825 = 0.031*cse624;
  const amrex::Real cse826 = 1.11e-10*std::pow(cse623, cse825 + 0.26300000000000001);
  const amrex::Real cse827 = 0.064000000000000001*cse624;
  const amrex::Real cse828 = 4.3749999999999998e-11*std::pow(cse623, cse827 + 0.071999999999999995);
  const amrex::Real cse829 = cse826 + cse828;
  const amrex::Real cse830 = 0.00235272*cse72;
  const amrex::Real cse831 = cse830 - 0.041816600000000002;
  const amrex::Real cse832 = cse72*cse831 + 0.35826400000000003;
  const amrex::Real cse833 = cse72*cse832 - 0.57443;
  const amrex::Real cse834 = 0.097857299999999994*cse72;
  const amrex::Real cse835 = cse834 - 3.1926800000000002;
  const amrex::Real cse836 = cse72*cse835 + 38.504899999999999;
  const amrex::Real cse837 = cse72*cse836 - 202.19300000000001;
  const amrex::Real cse838 = std::exp(((cse772) ? (
   cse72*cse833 - 7.4386999999999999
)
: (
   cse72*cse837 + 386.18599999999998
)));
  const amrex::Real cse839 = 1.7258000000000001e-6*cse622;
  const amrex::Real cse840 = cse838*cse839;
  const amrex::Real cse841 = cse824 + cse829*state.xn[3] + cse840;
  const amrex::Real cse842 = std::pow(cse623, cse821 + 0.22800000000000001);
  const amrex::Real cse843 = 8.9000000000000003e-11*cse842;
  const amrex::Real cse844 = cse843*state.xn[0];
  const amrex::Real cse845 = 0.037999999999999999*cse624;
  const amrex::Real cse846 = 5.1749999999999996e-11*std::pow(cse623, cse845 + 0.16900000000000001);
  const amrex::Real cse847 = 0.048000000000000001*cse624;
  const amrex::Real cse848 = 2.15e-11*std::pow(cse623, cse847 - 0.01);
  const amrex::Real cse849 = cse846 + cse848;
  const amrex::Real cse850 = 0.0070527699999999999*cse72;
  const amrex::Real cse851 = cse850 - 0.11133800000000001;
  const amrex::Real cse852 = cse72*cse851 + 0.69763799999999998;
  const amrex::Real cse853 = cse72*cse852 - 1.3074300000000001;
  const amrex::Real cse854 = 0.093813800000000003*cse72;
  const amrex::Real cse855 = cse854 - 3.0328300000000001;
  const amrex::Real cse856 = cse72*cse855 + 36.180300000000003;
  const amrex::Real cse857 = cse72*cse856 - 187.47399999999999;
  const amrex::Real cse858 = std::exp(((cse772) ? (
   cse72*cse853 - 7.6973500000000001
)
: (
   cse72*cse857 + 350.60899999999998
)));
  const amrex::Real cse859 = cse839*cse858;
  const amrex::Real cse860 = cse844 + cse849*state.xn[3] + cse859;
  const amrex::Real cse861 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse862 = cse860*cse861;
  const amrex::Real cse863 = (25.0/3.0)*cse862;
  const amrex::Real cse864 = 0.057000000000000002*cse624;
  const amrex::Real cse865 = std::pow(cse623, cse864 + 0.115);
  const amrex::Real cse866 = 1.26e-10*cse865;
  const amrex::Real cse867 = cse866*state.xn[0];
  const amrex::Real cse868 = 0.049000000000000002*cse624;
  const amrex::Real cse869 = 5.3249999999999997e-11*std::pow(cse623, cse868 - 0.0040000000000000001);
  const amrex::Real cse870 = 0.10199999999999999*cse624;
  const amrex::Real cse871 = 1.675e-11*std::pow(cse623, cse870 - 0.085000000000000006);
  const amrex::Real cse872 = cse869 + cse871;
  const amrex::Real cse873 = 0.00065632500000000003*cse72;
  const amrex::Real cse874 = -cse873 - 0.015089200000000001;
  const amrex::Real cse875 = cse72*cse874 + 0.36118400000000001;
  const amrex::Real cse876 = cse72*cse875 - 0.77378199999999997;
  const amrex::Real cse877 = 0.10508000000000001*cse72;
  const amrex::Real cse878 = cse877 - 3.4762;
  const amrex::Real cse879 = cse72*cse878 + 42.594999999999999;
  const amrex::Real cse880 = cse72*cse879 - 227.91300000000001;
  const amrex::Real cse881 = std::exp(((cse772) ? (
   cse72*cse876 - 9.2514099999999999
)
: (
   cse72*cse880 + 444.60000000000002
)));
  const amrex::Real cse882 = 2.8763333333333333e-6*cse881;
  const amrex::Real cse883 = cse622*cse882;
  const amrex::Real cse884 = cse867 + cse872*state.xn[3] + cse883;
  const amrex::Real cse885 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse886 = 5*cse885;
  const amrex::Real cse887 = cse841*cse886;
  const amrex::Real cse888 = cse884 + 7.8800000000000004e-8;
  const amrex::Real cse889 = cse860*cse886;
  const amrex::Real cse890 = cse841*cse863 + cse884*cse887 + cse888*cse889;
  const amrex::Real cse891 = cse841 + 2.65e-7;
  const amrex::Real cse892 = cse860 + 1.81e-14;
  const amrex::Real cse893 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse894 = cse841*cse893;
  const amrex::Real cse895 = (5.0/3.0)*cse892;
  const amrex::Real cse896 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse897 = cse884*cse896;
  const amrex::Real cse898 = 3*cse897;
  const amrex::Real cse899 = cse889*cse891 + cse891*cse898 + cse892*cse898;
  const amrex::Real cse900 = cse888*cse891 + cse888*cse892 + cse890 + cse894*cse895 + cse899;
  const amrex::Real cse901 = 1.0/cse900;
  const amrex::Real cse902 = 1.4210469458259591e-21*cse901;
  const amrex::Real cse903 = 2.5697412351439998e-22*cse901;
  const amrex::Real cse904 = state.T < 1000000.0;
  const amrex::Real cse905 = ((cse904) ? (
   cse890*cse902 + cse899*cse903
)
: (
   0.0
));
  const amrex::Real cse906 = (1.7e-24*chi*cse10*cse570*cse574*cse68*d2g + 5.6999999999999997e-11*chi*cse184*cse185*cse621*state.xn[3] + 1.7623947000000002e-12*crate*cse655*cse656*state.xn[5] + 1.602177e-12*crate*cse655*state.xn[0] + 3.2043540000000001e-12*crate*cse655*state.xn[3] + 3.0000000000000002e-15*cse10*cse620*d2g*state.xn[0] - 7.1777529600000012e-12*cse177*cse30 - cse295*cse590 - cse569*state.xn[0] - cse603 - cse638*cse640 - cse660*cse688 - cse689*state.xn[0] - cse747*state.xn[11] - cse820*state.xn[3] - cse905*state.xn[7] - std::min(cse564, cse568))/std::pow(cse561, 2);
  const amrex::Real cse907 = 1.0/cse561;
  const amrex::Real cse908 = d2g*state.xn[2];
  const amrex::Real cse909 = cse601*cse908;
  const amrex::Real cse910 = -cse564 + cse568;
  const amrex::Real cse911 = (((cse910 > 0) ? (
   0
)
: ((cse910 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse912 = cse565*cse911*d2g;
  const amrex::Real cse913 = cse566*cse912;
  const amrex::Real cse914 = (((cse910 < 0) ? (
   0
)
: ((cse910 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse915 = cse914*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse916 = cse563*cse915;
  const amrex::Real cse917 = cse634*cse638;
  const amrex::Real cse918 = std::pow(cse632, -2);
  const amrex::Real cse919 = std::pow(cse64, -2);
  const amrex::Real cse920 = 1.5159999999999999e-9*cse626;
  const amrex::Real cse921 = cse631*cse633*cse918;
  const amrex::Real cse922 = cse640*state.xn[8]/std::pow(cse636, 2);
  const amrex::Real cse923 = std::pow(cse618, -2);
  const amrex::Real cse924 = cse35*cse617;
  const amrex::Real cse925 = cse36*cse608;
  const amrex::Real cse926 = cse38*cse611;
  const amrex::Real cse927 = cse925 + cse926;
  const amrex::Real cse928 = cse616/std::pow(cse614, 2);
  const amrex::Real cse929 = cse11*cse928;
  const amrex::Real cse930 = cse924 - cse929*(-cse612 + cse927);
  const amrex::Real cse931 = cse38*cse645;
  const amrex::Real cse932 = cse35*state.xn[2];
  const amrex::Real cse933 = std::pow(cse641, -2);
  const amrex::Real cse934 = 0.5*cse933;
  const amrex::Real cse935 = cse125/std::pow(cse10, 3);
  const amrex::Real cse936 = -0.5*cse642*cse932 + cse934*cse935;
  const amrex::Real cse937 = cse641*cse644*cse66*state.xn[0];
  const amrex::Real cse938 = cse11*cse8;
  const amrex::Real cse939 = ((cse648) ? (
   0
)
: ((cse649) ? (
   (3.0/2.0)*cse938
)
: ((cse650) ? (
   (4.0/3.0)*cse938
)
: ((cse651) ? (
   (1.0/3.0)*cse938
)
: (
   0
)))));
  const amrex::Real cse940 = cse653*state.xn[3];
  const amrex::Real cse941 = cse36*cse652;
  const amrex::Real cse942 = 2.0*cse941;
  const amrex::Real cse943 = cse646 - cse931 + cse936*cse937 + cse939*cse940 - cse942;
  const amrex::Real cse944 = cse35*ncol_H*(((cse670 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse670 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse671, 3.0/2.0);
  const amrex::Real cse945 = cse944*ncol_CO;
  const amrex::Real cse946 = ((cse669) ? (
   0
)
: (
   878350727.21550155*cse945
));
  const amrex::Real cse947 = alpha_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse948 = std::log(cse680);
  const amrex::Real cse949 = cse947*cse948;
  const amrex::Real cse950 = cse660/std::pow(cse676, 2);
  const amrex::Real cse951 = nhalf_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse952 = cse946*cse951;
  const amrex::Real cse953 = cse681/cse660;
  const amrex::Real cse954 = cse676*cse953;
  const amrex::Real cse955 = std::pow(cse674, -2);
  const amrex::Real cse956 = llte_co_interp2d_partial_1(state.T, cse673);
  const amrex::Real cse957 = -cse660*cse946*cse955*cse956 + cse683*(cse664*cse676*cse946*cse955*cse956 - cse677*cse952);
  const amrex::Real cse958 = cse660*state.xn[9];
  const amrex::Real cse959 = cse663*cse958/std::pow(cse685, 2);
  const amrex::Real cse960 = std::pow(cse816, -2);
  const amrex::Real cse961 = std::pow(cse818, -2);
  const amrex::Real cse962 = std::pow(cse785, 2)*cse960*cse961;
  const amrex::Real cse963 = cse713*cse727;
  const amrex::Real cse964 = 7.1400000000000007e-11*cse693;
  const amrex::Real cse965 = cse712*cse713;
  const amrex::Real cse966 = 8.68e-11*cse705;
  const amrex::Real cse967 = cse701*cse713;
  const amrex::Real cse968 = cse727*cse728;
  const amrex::Real cse969 = 6.3800000000000002e-11*cse719;
  const amrex::Real cse970 = cse713*cse731;
  const amrex::Real cse971 = 2.8933333333333332e-11*cse705*cse968 + 2.1266666666666666e-11*cse719*cse729 + cse963*cse964 + cse964*cse965 + cse966*cse967 + cse969*cse970;
  const amrex::Real cse972 = cse713*cse734;
  const amrex::Real cse973 = 2.1419999999999999e-10*cse693;
  const amrex::Real cse974 = cse735*cse739;
  const amrex::Real cse975 = cse734*cse739;
  const amrex::Real cse976 = 2.604e-10*cse705*cse740 + 1.9139999999999998e-10*cse719*cse740 + cse963*cse966 + cse969*cse972 + cse973*cse974 + cse973*cse975;
  const amrex::Real cse977 = std::pow(cse743, -2);
  const amrex::Real cse978 = cse735*cse736;
  const amrex::Real cse979 = cse977*(-cse694*cse734 - cse694*cse735 - 1.4466666666666666e-10*cse705*cse978 - cse706*cse731 - 1.0633333333333333e-10*cse719*cse737 - cse720*cse731 - cse971 - cse976);
  const amrex::Real cse980 = 2.3872025442750958e-19*cse733;
  const amrex::Real cse981 = 2.8023005140019997e-18*cse742;
  const amrex::Real cse982 = cse841*cse861;
  const amrex::Real cse983 = cse884*cse885;
  const amrex::Real cse984 = 1.32e-9*cse822;
  const amrex::Real cse985 = 6.3e-10*cse865;
  const amrex::Real cse986 = cse860*cse885;
  const amrex::Real cse987 = cse841*cse885;
  const amrex::Real cse988 = cse885*cse888;
  const amrex::Real cse989 = 4.4500000000000001e-10*cse842;
  const amrex::Real cse990 = 2.2000000000000003e-9*cse822*cse862 + 7.4166666666666679e-10*cse842*cse982 + cse983*cse984 + cse985*cse986 + cse985*cse987 + cse988*cse989;
  const amrex::Real cse991 = cse885*cse891;
  const amrex::Real cse992 = 3.7800000000000004e-10*cse865;
  const amrex::Real cse993 = cse892*cse896;
  const amrex::Real cse994 = cse891*cse896;
  const amrex::Real cse995 = 7.9200000000000006e-10*cse822*cse897 + 2.6700000000000001e-10*cse842*cse897 + cse984*cse986 + cse989*cse991 + cse992*cse993 + cse992*cse994;
  const amrex::Real cse996 = std::pow(cse900, -2);
  const amrex::Real cse997 = cse892*cse893;
  const amrex::Real cse998 = cse996*(-4.4000000000000003e-10*cse822*cse997 - cse823*cse888 - 1.4833333333333334e-10*cse842*cse894 - cse843*cse888 - cse866*cse891 - cse866*cse892 - cse990 - cse995);
  const amrex::Real cse999 = 1.4210469458259591e-21*cse890;
  const amrex::Real cse1000 = 2.5697412351439998e-22*cse899;
  const amrex::Real cse1001 = cse574*cse69*d2g;
  const amrex::Real cse1002 = crate*cse655;
  const amrex::Real cse1003 = 1.7623947000000002e-12*cse1002*cse919*state.xn[5];
  const amrex::Real cse1004 = cse35*cse615*cse616 - cse927*cse929;
  const amrex::Real cse1005 = 2.0187430200000004e-26*cse165;
  const amrex::Real cse1006 = cse10*cse1005*cse923;
  const amrex::Real cse1007 = 1.461185424e-21*cse186*cse923*state.xn[3];
  const amrex::Real cse1008 = 2.0*cse11*cse939*state.xn[3] + 26.399999999999999*cse641*cse643*cse66*cse936*state.xn[0] - cse931 - cse942;
  const amrex::Real cse1009 = 1.602177e-12*cse89;
  const amrex::Real cse1010 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1011 = 1.7623947000000002e-12*cse656;
  const amrex::Real cse1012 = crate*cse1011*state.xn[5];
  const amrex::Real cse1013 = cse679*cse681;
  const amrex::Real cse1014 = 1.7e-24*cse1001*cse570 + cse1003*cse86 + cse1004*cse1006 + cse1004*cse1007 + cse1008*cse1009 + cse1008*cse1010 + cse1008*cse1012 - 7.1777529600000012e-12*cse120 + 3.0000000000000002e-15*cse165*cse620 - 7.1777529600000012e-12*cse496 - cse909 - cse913 - cse916 - cse959*(-cse684*(-cse1013*cse952 + cse946*cse947*cse948) - cse957);
  const amrex::Real cse1015 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1016 = std::pow(cse576, -2);
  const amrex::Real cse1017 = std::pow(cse581, -2);
  const amrex::Real cse1018 = cse598*cse600;
  const amrex::Real cse1019 = 1.0/cse595;
  const amrex::Real cse1020 = std::pow(cse591, -2);
  const amrex::Real cse1021 = cse1019*cse1020;
  const amrex::Real cse1022 = cse72/std::pow(cse596, 2);
  const amrex::Real cse1023 = std::pow(cse573, -2);
  const amrex::Real cse1024 = std::pow(cse571, 0.12299999999999994)*cse69;
  const amrex::Real cse1025 = std::pow(cse571, -0.56899999999999995)*cse572*cse69;
  const amrex::Real cse1026 = 1.3659996292638161e-10*cse295*cse578;
  const amrex::Real cse1027 = cse1026*cse584/std::pow(cse586, 2);
  const amrex::Real cse1028 = 0.5*cse11*cse642 - cse932*cse934;
  const amrex::Real cse1029 = 9.0599999999999997e-6*cse263;
  const amrex::Real cse1030 = 1.024e-10*cse690;
  const amrex::Real cse1031 = 2.1600000000000004e-15*cse702;
  const amrex::Real cse1032 = 9.7199999999999998e-11*cse716;
  const amrex::Real cse1033 = cse1030*cse963 + cse1030*cse965 + cse1031*cse967 + cse1032*cse970 + 7.1999999999999997e-16*cse702*cse968 + 3.2399999999999999e-11*cse716*cse729;
  const amrex::Real cse1034 = 3.0719999999999998e-10*cse690;
  const amrex::Real cse1035 = cse1031*cse963 + cse1032*cse972 + cse1034*cse974 + cse1034*cse975 + 6.4799999999999996e-15*cse702*cse740 + 2.9159999999999995e-10*cse716*cse740;
  const amrex::Real cse1036 = cse977*(-cse1033 - cse1035 - cse691*cse734 - cse691*cse735 - 3.6000000000000001e-15*cse702*cse978 - cse703*cse731 - 1.6199999999999998e-10*cse716*cse737 - cse717*cse731);
  const amrex::Real cse1037 = cse263*cse838;
  const amrex::Real cse1038 = cse263*cse881;
  const amrex::Real cse1039 = 1.4381666666666666e-5*cse1038;
  const amrex::Real cse1040 = cse263*cse858;
  const amrex::Real cse1041 = 8.6289999999999999e-6*cse1037;
  const amrex::Real cse1042 = 8.6289999999999999e-6*cse1040;
  const amrex::Real cse1043 = 1.4381666666666668e-5*cse1037*cse862 + cse1039*cse986 + cse1039*cse987 + 1.4381666666666668e-5*cse1040*cse982 + cse1041*cse983 + cse1042*cse988;
  const amrex::Real cse1044 = 5.1774000000000006e-6*cse897;
  const amrex::Real cse1045 = 8.6289999999999999e-6*cse1038;
  const amrex::Real cse1046 = cse1037*cse1044 + cse1040*cse1044 + cse1041*cse986 + cse1042*cse991 + cse1045*cse993 + cse1045*cse994;
  const amrex::Real cse1047 = cse263*cse882;
  const amrex::Real cse1048 = 1.7258000000000001e-6*cse888;
  const amrex::Real cse1049 = cse996*(-cse1037*cse1048 - 2.8763333333333337e-6*cse1037*cse997 - cse1040*cse1048 - 2.8763333333333337e-6*cse1040*cse894 - cse1043 - cse1046 - cse1047*cse891 - cse1047*cse892);
  const amrex::Real cse1050 = cse1018*cse908;
  const amrex::Real cse1051 = 3.7996253111508318e-29*cse1050;
  const amrex::Real cse1052 = cse2*cse912;
  const amrex::Real cse1053 = 6.4000000000000004e-32*cse1052;
  const amrex::Real cse1054 = cse915*d2g;
  const amrex::Real cse1055 = 200.0*cse1054;
  const amrex::Real cse1056 = 5.7927337833400002e-20*cse917;
  const amrex::Real cse1057 = -cse630;
  const amrex::Real cse1058 = 5.7927337833400002e-20*cse638*cse921;
  const amrex::Real cse1059 = 2*cse921;
  const amrex::Real cse1060 = cse171*cse608 + 2*cse926;
  const amrex::Real cse1061 = 2*cse924 - cse929*(cse1060 - cse609);
  const amrex::Real cse1062 = 2*cse931;
  const amrex::Real cse1063 = cse642*cse932;
  const amrex::Real cse1064 = cse933*cse935;
  const amrex::Real cse1065 = -1.0*cse1063 + 1.0*cse1064;
  const amrex::Real cse1066 = ((cse648) ? (
   0
)
: ((cse649) ? (
   3*cse938
)
: ((cse650) ? (
   (8.0/3.0)*cse938
)
: ((cse651) ? (
   (2.0/3.0)*cse938
)
: (
   0
)))));
  const amrex::Real cse1067 = 4.0*cse941;
  const amrex::Real cse1068 = -cse1062 + cse1065*cse937 + cse1066*cse940 - cse1067 + cse654;
  const amrex::Real cse1069 = ((cse669) ? (
   0
)
: (
   cse672*cse944
));
  const amrex::Real cse1070 = cse1069*cse951;
  const amrex::Real cse1071 = -cse1069*cse660*cse955*cse956 + cse683*(cse1069*cse664*cse676*cse955*cse956 - cse1070*cse677);
  const amrex::Real cse1072 = cse701*cse714;
  const amrex::Real cse1073 = (1.0/15.0)*cse968;
  const amrex::Real cse1074 = cse726*cse731;
  const amrex::Real cse1075 = cse1072*cse711 + cse1073*cse711 + cse1074*cse714 + cse700*cse715 + cse700*cse732 + cse726*cse730;
  const amrex::Real cse1076 = cse700*cse735;
  const amrex::Real cse1077 = (3.0/5.0)*cse739;
  const amrex::Real cse1078 = cse700*cse734;
  const amrex::Real cse1079 = cse714*cse734;
  const amrex::Real cse1080 = cse1076*cse1077 + cse1077*cse1078 + cse1079*cse726 + cse711*cse732 + cse711*cse741 + cse726*cse741;
  const amrex::Real cse1081 = (1.0/3.0)*cse978;
  const amrex::Real cse1082 = cse977*(-cse1074 - cse1075 - cse1076 - cse1078 - cse1080 - cse1081*cse711 - cse711*cse731 - cse726*cse738);
  const amrex::Real cse1083 = cse884*cse886;
  const amrex::Real cse1084 = (25.0/3.0)*cse982;
  const amrex::Real cse1085 = cse849*cse888;
  const amrex::Real cse1086 = cse1083*cse829 + cse1084*cse849 + cse1085*cse886 + cse829*cse863 + cse872*cse887 + cse872*cse889;
  const amrex::Real cse1087 = cse886*cse891;
  const amrex::Real cse1088 = cse872*cse892;
  const amrex::Real cse1089 = 3*cse896;
  const amrex::Real cse1090 = cse872*cse891;
  const amrex::Real cse1091 = cse1087*cse849 + cse1088*cse1089 + cse1089*cse1090 + cse829*cse889 + cse829*cse898 + cse849*cse898;
  const amrex::Real cse1092 = cse893*cse895;
  const amrex::Real cse1093 = (5.0/3.0)*cse894;
  const amrex::Real cse1094 = cse996*(-cse1085 - cse1086 - cse1088 - cse1090 - cse1091 - cse1092*cse829 - cse1093*cse849 - cse829*cse888);
  const amrex::Real cse1095 = cse1001*cse570;
  const amrex::Real cse1096 = -cse1060*cse929 + 2*cse35*cse615*cse616;
  const amrex::Real cse1097 = -cse1062 + 26.399999999999999*cse1065*cse641*cse643*cse66*state.xn[0] + 2.0*cse1066*cse11*state.xn[3] - cse1067;
  const amrex::Real cse1098 = state.xn[3]*((cse776) ? (
   0
)
: (
   cse800*cse962
));
  const amrex::Real cse1099 = ((cse669) ? (
   cse668
)
: (
   0
));
  const amrex::Real cse1100 = cse1099*cse951;
  const amrex::Real cse1101 = cse1014*cse907;
  const amrex::Real cse1102 = 3*cse35*cse615*cse616 - cse929*(3*cse925 + 3*cse926);
  const amrex::Real cse1103 = 2.0*cse11*state.xn[3]*((cse648) ? (
   0
)
: ((cse649) ? (
   (9.0/2.0)*cse938
)
: ((cse650) ? (
   4*cse938
)
: ((cse651) ? (
   cse938
)
: (
   0
))))) + 26.399999999999999*cse641*cse643*cse66*state.xn[0]*(-1.5*cse1063 + 1.5*cse1064) - 3*cse931 - 6.0*cse941;
  const amrex::Real cse1104 = ((cse669) ? (
   0
)
: (
   2635052181.6465044*cse945
));
  const amrex::Real cse1105 = cse1104*cse951;
  const amrex::Real cse1106 = cse584*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1107 = cse295*cse577;
  const amrex::Real cse1108 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1109 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1110 = cse1108*cse579;
  const amrex::Real cse1111 = cse1017*cse583*(2.5500935934394033e-12*cse1016*cse1109*state.xn[2] - 6.6690584756579352e-10*cse1110);
  const amrex::Real cse1112 = cse1019*cse276*cse592*cse69;
  const amrex::Real cse1113 = cse259*cse639;
  const amrex::Real cse1114 = cse3*cse622;
  const amrex::Real cse1115 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1116 = 100*cse3;
  const amrex::Real cse1117 = -2.2649999999999999e-6*cse1114 + cse627*(cse1116*(8.7000000000000001e-5*cse624 + 0.001281) + cse3*cse625) + state.xn[3]*(0.75*((cse628) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1115
)) + 0.25*((cse628) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1115
)));
  const amrex::Real cse1118 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1119 = -cse1117;
  const amrex::Real cse1120 = std::pow(cse605, -3);
  const amrex::Real cse1121 = -cse610*cse7*(0.0040460000000000001*cse1120 + 0.0036709999999999998*cse606) - cse613*cse7*(0.0047800000000000004*cse1120 - 5.5500000000000001e-5*cse606);
  const amrex::Real cse1122 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1123 = std::pow(cse664, -2);
  const amrex::Real cse1124 = ((cse669) ? (
   -2.872276598720128e-13*cse276*cse666*state.xn[9]*(((5.7445531974402559e-18*cse2*cse666 - gradv < 0) ? (
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
  const amrex::Real cse1125 = -cse1124*cse956 - llte_co_interp2d_partial_0(state.T, cse673);
  const amrex::Real cse1126 = cse1124*cse951 + nhalf_co_interp2d_partial_0(state.T, cse673);
  const amrex::Real cse1127 = ((cse749) ? (
   -cse259
)
: (
   0
));
  const amrex::Real cse1128 = -1.8719999999999998e-14*cse1127*cse778 - 3.9261999999999998e-15*cse1127*cse779 - 1.53e-21*cse1127*cse780 + 4.985670872372847e-33*cse781*cse784*((cse749) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse782*cse784*((cse749) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse782*((cse749) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse783, 2);
  const amrex::Real cse1129 = ((cse749) ? (
   1
)
: (
   0
));
  const amrex::Real cse1130 = cse1129*cse777;
  const amrex::Real cse1131 = cse1130*cse13*cse786;
  const amrex::Real cse1132 = cse1130*cse759*cse789;
  const amrex::Real cse1133 = cse1130*cse761*cse791;
  const amrex::Real cse1134 = cse1130*cse8;
  const amrex::Real cse1135 = cse1130*cse763*cse792;
  const amrex::Real cse1136 = cse1130*cse753*cse809;
  const amrex::Real cse1137 = cse1130*cse755*cse811;
  const amrex::Real cse1138 = cse1130*cse757*cse787;
  const amrex::Real cse1139 = 65.320000000000007*cse963;
  const amrex::Real cse1140 = cse3*cse698;
  const amrex::Real cse1141 = 0.00042999999999999999*cse624;
  const amrex::Real cse1142 = cse695*(cse1116*(0.0041900000000000001 - 3.0000000000000001e-5*cse624) - cse3*cse692) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse697*(cse1116*(0.00025000000000000001*cse624 + 0.00264) + cse3*cse696) + cse699*(cse1116*(cse1141 + 0.00296) + cse1140));
  const amrex::Real cse1143 = cse721*(cse1116*(0.0036900000000000001 - 6.0000000000000002e-5*cse624) - cse3*cse718) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse723*(cse1116*(0.00041000000000000005*cse624 + 0.0020300000000000001) + cse3*cse722) + cse725*(cse1116*(0.00058*cse624 + 0.0023700000000000001) + cse3*cse724));
  const amrex::Real cse1144 = cse707*(cse1116*(0.0075500000000000003 - 0.0016000000000000001*cse624) - cse3*cse704) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse708*(cse1116*(cse1141 + 0.0088900000000000003) + cse1140) + cse710*(cse1116*(0.0052500000000000003*cse624 + 0.011979999999999999) + cse3*cse709));
  const amrex::Real cse1145 = cse1143*cse731;
  const amrex::Real cse1146 = cse1072*cse1144 + cse1073*cse1144 + cse1139*cse259*cse731 + cse1142*cse715 + cse1142*cse732 + cse1143*cse730 + cse1145*cse714 + 65.320000000000007*cse259*cse712*cse967 + 28.360000000000003*cse259*cse727*cse729;
  const amrex::Real cse1147 = 136.68000000000001*cse740;
  const amrex::Real cse1148 = cse259*cse735;
  const amrex::Real cse1149 = cse259*cse734;
  const amrex::Real cse1150 = cse1142*cse735;
  const amrex::Real cse1151 = cse1142*cse734;
  const amrex::Real cse1152 = cse1077*cse1150 + cse1077*cse1151 + cse1079*cse1143 + cse1139*cse1149 + cse1143*cse741 + cse1144*cse732 + cse1144*cse741 + cse1147*cse1148 + cse1147*cse1149;
  const amrex::Real cse1153 = cse977*(-cse1081*cse1144 - cse1143*cse738 - cse1144*cse731 - cse1145 - cse1146 - 32.933333333333337*cse1148*cse737 - cse1150 - cse1151 - cse1152);
  const amrex::Real cse1154 = 312.30000000000001*cse259;
  const amrex::Real cse1155 = cse3*cse821;
  const amrex::Real cse1156 = 0.00046000000000000001*cse624;
  const amrex::Real cse1157 = 8.6290000000000003e-7*cse1114;
  const amrex::Real cse1158 = -cse1157*cse838 + cse824*(cse1116*(cse1156 + 0.00231) + cse1155) + cse840*((cse772) ? (
   cse3*cse833 + cse72*(cse3*cse832 + cse72*(cse3*cse830 + cse3*cse831))
)
: (
   cse3*cse837 + cse72*(cse3*cse836 + cse72*(cse3*cse834 + cse3*cse835))
)) + state.xn[3]*(cse826*(cse1116*(0.00031*cse624 + 0.00263) + cse3*cse825) + cse828*(cse1116*(0.00064000000000000005*cse624 + 0.00071999999999999994) + cse3*cse827));
  const amrex::Real cse1159 = -1.4381666666666666e-6*cse1114*cse881 + cse867*(cse1116*(0.00057000000000000009*cse624 + 0.00115) + cse3*cse864) + cse883*((cse772) ? (
   cse3*cse876 + cse72*(cse3*cse875 + cse72*(-cse3*cse873 + cse3*cse874))
)
: (
   cse3*cse880 + cse72*(cse3*cse879 + cse72*(cse3*cse877 + cse3*cse878))
)) + state.xn[3]*(cse869*(cse1116*(0.00048999999999999998*cse624 - 4.0000000000000003e-5) + cse3*cse868) + cse871*(cse1116*(0.0010199999999999999*cse624 - 0.00085000000000000006) + cse3*cse870));
  const amrex::Real cse1160 = -cse1157*cse858 + cse844*(cse1116*(cse1156 + 0.0022800000000000003) + cse1155) + cse859*((cse772) ? (
   cse3*cse853 + cse72*(cse3*cse852 + cse72*(cse3*cse850 + cse3*cse851))
)
: (
   cse3*cse857 + cse72*(cse3*cse856 + cse72*(cse3*cse854 + cse3*cse855))
)) + state.xn[3]*(cse846*(cse1116*(0.00038000000000000002*cse624 + 0.0016900000000000001) + cse3*cse845) + cse848*(cse1116*(0.00048000000000000001*cse624 - 0.0001) + cse3*cse847));
  const amrex::Real cse1161 = cse1160*cse888;
  const amrex::Real cse1162 = cse1083*cse1158 + cse1084*cse1160 + cse1154*cse841*cse983 + cse1154*cse860*cse988 + cse1158*cse863 + cse1159*cse887 + cse1159*cse889 + cse1161*cse886 + 844.16666666666686*cse259*cse841*cse862;
  const amrex::Real cse1163 = 70.859999999999985*cse259*cse897;
  const amrex::Real cse1164 = cse1159*cse892;
  const amrex::Real cse1165 = cse1159*cse891;
  const amrex::Real cse1166 = cse1087*cse1160 + cse1089*cse1164 + cse1089*cse1165 + cse1154*cse891*cse986 + cse1158*cse889 + cse1158*cse898 + cse1160*cse898 + cse1163*cse891 + cse1163*cse892;
  const amrex::Real cse1167 = cse996*(-cse1092*cse1158 - cse1093*cse1160 - cse1158*cse888 - cse1161 - cse1162 - cse1164 - cse1165 - cse1166 - 64.733333333333348*cse259*cse892*cse894);

  jac(1, 1) = -cse1 - cse10*cse56 - cse102 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3];
  jac(1, 2) = cse115 + cse124;
  jac(1, 3) = cse134 + cse136*state.xn[16] + cse140*cse141 + cse148 + cse156;
  jac(1, 4) = cse123*cse182 - cse141*cse190 + cse159 + cse164 + cse166 + 2*cse167 + 2*cse176 + 4*cse178 + cse179*state.xn[17] + 1.1399999999999999e-10*cse186 + cse189*state.xn[8] + cse194 + cse203;
  jac(1, 5) = cse123*cse208 + cse166 + 2*cse207 + cse211 + cse214;
  jac(1, 7) = cse219;
  jac(1, 8) = cse223;
  jac(1, 9) = cse189*state.xn[3];
  jac(1, 10) = -cse224;
  jac(1, 11) = cse124 + cse226;
  jac(1, 12) = cse229 + cse230 + cse233;
  jac(1, 15) = cse234 + cse235 + cse237 + cse238;
  jac(1, 16) = cse238 + cse242;
  jac(1, 17) = cse123*cse248 + cse136*state.xn[2] - 1.8000000000000002e-14*cse165 + 2*cse247 + cse255;
  jac(1, 18) = cse179*state.xn[3] - cse256 + cse258;
  jac(1, 19) = cse311*(-cse145*cse290 + 9.8669999999999991e-12*cse187*cse287*state.xn[3]*state.xn[8] - cse261 - cse262 - cse265 + 2*cse274*cse30 - cse275 + 2*cse278*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse279 + cse280 - 5.5769999999999994e-13*cse283 - cse285 + cse286 + cse288 - cse294 - cse310);
  jac(2, 1) = cse102;
  jac(2, 2) = -cse115;
  jac(2, 3) = -cse155;
  jac(2, 4) = cse159 + cse200 + cse312;
  jac(2, 5) = cse210 + cse213 + cse312;
  jac(2, 7) = cse215;
  jac(2, 11) = cse313;
  jac(2, 12) = -cse232;
  jac(2, 15) = cse313;
  jac(2, 16) = cse313;
  jac(2, 17) = cse253 - cse254;
  jac(2, 18) = cse256;
  jac(2, 19) = cse311*(-cse261 + cse296 + cse298 - cse299 - cse302 - cse314);
  jac(3, 1) = cse101 + cse316 - cse356 + cse65 - cse84 + cse90;
  jac(3, 2) = -cse114 - cse358;
  jac(3, 3) = -cse156 - cse368 - cse385 - cse390 - cse391 - cse406;
  jac(3, 4) = -cse203 - cse408 + cse409 + cse410 - cse415 - cse422 - cse427 - cse428;
  jac(3, 5) = -cse214 - cse408 - cse414 - cse421 - cse430;
  jac(3, 6) = cse431;
  jac(3, 7) = -cse433;
  jac(3, 8) = cse443 + cse445;
  jac(3, 9) = -cse448;
  jac(3, 10) = cse224;
  jac(3, 11) = -cse226 - cse358;
  jac(3, 12) = cse453;
  jac(3, 13) = cse456;
  jac(3, 14) = cse458;
  jac(3, 15) = cse459;
  jac(3, 16) = cse459;
  jac(3, 17) = -cse255 - cse460 - cse461 - cse462 - cse464 - cse465 - cse466 - cse467;
  jac(3, 18) = cse469;
  jac(3, 19) = cse311*(4.2899999999999994e-13*cse283 + cse291 + cse305 + cse310 + cse471 + cse472 + cse473 + cse476 + cse477 - cse479 + cse480 + cse481 - cse483 - cse484 - cse485 - cse486 - cse487);
  jac(3, 20) = cse488;
  jac(4, 1) = 3.0000000000000002e-15*cse10*d2g - cse489 - cse490 - cse491 - cse492 - cse494 - cse495;
  jac(4, 2) = cse498;
  jac(4, 3) = cse144 - cse146 - cse147 + cse405 + cse499*state.xn[16];
  jac(4, 4) = -cse121 - cse167 - cse176 - cse177*cse9 - cse417 - cse500 - cse501 - cse503 - cse504 - cse505 - cse506;
  jac(4, 5) = -cse121 - cse207 - cse211 - cse429 - cse507;
  jac(4, 7) = -cse215 - cse508;
  jac(4, 8) = cse442;
  jac(4, 9) = -cse446;
  jac(4, 10) = cse509;
  jac(4, 11) = cse498;
  jac(4, 12) = cse452;
  jac(4, 15) = -cse237 - cse497;
  jac(4, 16) = cse498;
  jac(4, 17) = -cse247 + cse464 + cse499*state.xn[2] + cse510 - cse511 + cse513;
  jac(4, 18) = cse469;
  jac(4, 19) = cse311*(-cse260 - 2.3622303960273254e-7*cse279 + cse305 - cse514 - cse515 - cse516 - cse517 - cse518 - cse520);
  jac(5, 1) = cse495;
  jac(5, 2) = cse357;
  jac(5, 4) = -cse164 + cse502 + cse504;
  jac(5, 5) = -cse209 - cse210 - cse430;
  jac(5, 7) = cse508;
  jac(5, 11) = cse357;
  jac(5, 15) = cse357;
  jac(5, 16) = cse357;
  jac(5, 17) = cse463;
  jac(5, 19) = -cse311*cse519;
  jac(6, 1) = cse328;
  jac(6, 2) = cse328;
  jac(6, 3) = cse368;
  jac(6, 4) = cse521;
  jac(6, 5) = cse408;
  jac(6, 6) = -cse431;
  jac(6, 7) = cse523;
  jac(6, 10) = cse524;
  jac(6, 11) = cse328;
  jac(6, 15) = cse328;
  jac(6, 16) = cse525;
  jac(6, 17) = cse465;
  jac(6, 19) = cse311*cse527;
  jac(7, 1) = cse528;
  jac(7, 2) = cse528;
  jac(7, 3) = -cse368;
  jac(7, 4) = -cse521;
  jac(7, 5) = -cse408;
  jac(7, 6) = cse431;
  jac(7, 7) = -cse523;
  jac(7, 10) = -cse524;
  jac(7, 11) = cse528;
  jac(7, 15) = cse528;
  jac(7, 16) = -cse525;
  jac(7, 17) = -cse465;
  jac(7, 19) = -cse311*cse527;
  jac(8, 1) = cse1 + cse532;
  jac(8, 2) = cse532;
  jac(8, 3) = cse385;
  jac(8, 4) = cse419 + cse534 + cse535*state.xn[8] + cse538;
  jac(8, 5) = cse538;
  jac(8, 8) = -cse223 - cse442 - cse445;
  jac(8, 9) = cse447 + cse535*state.xn[3];
  jac(8, 10) = cse539;
  jac(8, 11) = cse532;
  jac(8, 15) = cse532 + cse540;
  jac(8, 16) = -cse240 + cse532;
  jac(8, 17) = -cse461 + cse466 - cse541;
  jac(8, 19) = cse311*(cse262 - 1.287e-13*cse283 + cse293 + cse471 + 2.2769999999999997e-12*cse482 + cse542);
  jac(9, 1) = cse543;
  jac(9, 2) = cse543;
  jac(9, 3) = -cse385;
  jac(9, 4) = -cse422;
  jac(9, 5) = -cse421;
  jac(9, 7) = cse522;
  jac(9, 8) = cse445;
  jac(9, 9) = -cse448 - cse545;
  jac(9, 10) = cse524;
  jac(9, 11) = cse543;
  jac(9, 15) = cse543;
  jac(9, 16) = -cse342 - cse546;
  jac(9, 17) = -cse466;
  jac(9, 19) = cse311*(-cse517 - cse542 - cse548);
  jac(10, 1) = cse530;
  jac(10, 2) = cse530;
  jac(10, 3) = cse150;
  jac(10, 4) = cse549;
  jac(10, 5) = cse536;
  jac(10, 7) = -cse522;
  jac(10, 8) = cse223;
  jac(10, 10) = -cse550 - cse551;
  jac(10, 11) = cse225 + cse530;
  jac(10, 12) = cse231;
  jac(10, 15) = cse552;
  jac(10, 16) = cse240 + cse530;
  jac(10, 17) = -cse512 + cse541;
  jac(10, 19) = -cse294*cse311;
  jac(11, 3) = -cse150;
  jac(11, 9) = cse545;
  jac(11, 10) = cse550;
  jac(11, 11) = -cse225;
  jac(11, 16) = cse546;
  jac(11, 17) = cse512;
  jac(11, 19) = cse311*(cse291 + cse548);
  jac(12, 1) = -cse530 - cse553;
  jac(12, 2) = -cse105 - cse530;
  jac(12, 3) = cse146 - cse401 + cse403;
  jac(12, 4) = cse192 - cse193 - cse426 - cse549;
  jac(12, 5) = cse537;
  jac(12, 7) = cse522;
  jac(12, 10) = cse551;
  jac(12, 11) = cse531;
  jac(12, 12) = -cse233 - cse554;
  jac(12, 15) = -cse552;
  jac(12, 16) = cse241 + cse531;
  jac(12, 17) = -cse460 - cse541;
  jac(12, 18) = cse256 + cse257;
  jac(12, 19) = cse311*(-cse555 - cse556);
  jac(13, 1) = cse355;
  jac(13, 2) = cse355;
  jac(13, 3) = cse390;
  jac(13, 4) = cse415;
  jac(13, 5) = cse414;
  jac(13, 11) = cse355;
  jac(13, 13) = -cse456;
  jac(13, 14) = cse457;
  jac(13, 15) = cse355;
  jac(13, 16) = cse355;
  jac(13, 17) = cse467;
  jac(13, 19) = cse311*cse557;
  jac(13, 20) = cse558;
  jac(14, 1) = cse559;
  jac(14, 2) = cse559;
  jac(14, 3) = -cse390;
  jac(14, 4) = -cse415;
  jac(14, 5) = -cse414;
  jac(14, 11) = cse559;
  jac(14, 13) = cse456;
  jac(14, 14) = cse458;
  jac(14, 15) = cse559;
  jac(14, 16) = cse559;
  jac(14, 17) = -cse467;
  jac(14, 19) = -cse311*cse557;
  jac(14, 20) = cse488;
  jac(15, 1) = cse492;
  jac(15, 4) = cse560*state.xn[8];
  jac(15, 8) = cse442;
  jac(15, 9) = cse560*state.xn[3];
  jac(15, 12) = -cse231;
  jac(15, 15) = -cse234 - cse540;
  jac(15, 17) = cse461;
  jac(15, 19) = cse311*(-3.0029999999999999e-13*cse283 + cse470 + 5.3129999999999997e-12*cse482 + cse514);
  jac(16, 3) = cse148 + cse404;
  jac(16, 4) = cse427;
  jac(16, 7) = -cse218;
  jac(16, 8) = -cse223;
  jac(16, 9) = -cse545;
  jac(16, 12) = cse554;
  jac(16, 16) = -cse242 - cse546;
  jac(16, 17) = cse460;
  jac(16, 18) = cse258;
  jac(16, 19) = cse311*(cse265 + cse293 + cse518 + cse547 + cse555);
  jac(17, 3) = -cse406;
  jac(17, 4) = cse505;
  jac(17, 5) = cse209;
  jac(17, 8) = cse443;
  jac(17, 10) = -cse509;
  jac(17, 12) = cse453;
  jac(17, 17) = -cse462 - cse513;
  jac(17, 19) = cse311*(cse472 + cse520);
  jac(18, 1) = cse553;
  jac(18, 2) = cse105;
  jac(18, 3) = -cse391;
  jac(18, 4) = -cse506;
  jac(18, 7) = cse218;
  jac(18, 12) = cse232;
  jac(18, 16) = cse239;
  jac(18, 18) = -cse256 - cse468;
  jac(18, 19) = cse311*(cse307 + cse526 + cse556);
  jac(19, 1) = -1.6737729999999998e-24*cse906 + cse907*(1.461185424e-21*chi*cse184*cse185*cse923*cse930*state.xn[3] + 1.7e-24*chi*cse570*cse574*cse68*d2g + 1.7623947000000002e-12*crate*cse655*cse87*cse919*state.xn[5] + 1.602177e-12*crate*cse655 + 1.7623947000000002e-12*crate*cse656*cse943*state.xn[5] + 1.602177e-12*crate*cse943*state.xn[0] + 3.2043540000000001e-12*crate*cse943*state.xn[3] + 3.0000000000000002e-15*cse10*cse620*d2g + 2.0187430200000004e-26*cse10*cse923*cse930*d2g*state.xn[0] - 7.1777529600000012e-12*cse491 - 7.1777529600000012e-12*cse494 - cse569 - cse580*cse589 + 3.0000000000000002e-15*cse620*d2g*state.xn[0] + 4.39089220777172e-29*cse626*cse631*cse633*cse637*cse918*state.xn[8] - 4.39089220777172e-29*cse626*cse917 - cse659*cse688 - cse689 - cse909 - cse913 - cse916 - cse922*(-cse634*cse920 + cse920*cse921) - cse959*(-cse659*cse675 - cse684*(cse946*cse949 + cse954*(cse659*cse679 - cse950*cse952)) - cse957) - state.xn[11]*(cse745*cse971 + cse746*cse976 + cse979*cse980 + cse979*cse981) - state.xn[3]*((cse776) ? (
   0
)
: (
   cse807*cse962
)) - state.xn[7]*((cse904) ? (
   cse1000*cse998 + cse902*cse990 + cse903*cse995 + cse998*cse999
)
: (
   0
)));
  jac(19, 2) = -1.6728620616289998e-24*cse906 + cse907*(cse1014 - state.xn[3]*((cse776) ? (
   0
)
: (
   cse798*cse962
)));
  jac(19, 3) = -9.1093837099999994e-28*cse906 + cse907*(1.7e-24*chi*cse10*cse1023*cse570*cse68*d2g*(0.00025272365919999998*cse1024*cse137 + 0.036605692000000002*cse1025*cse137) + 4.6527220080000002e-11*crate*cse1028*cse641*cse643*cse656*cse66*state.xn[0]*state.xn[5] + 4.22974728e-11*crate*cse1028*cse641*cse643*cse66*std::pow(state.xn[0], 2) + 8.4594945600000001e-11*crate*cse1028*cse641*cse643*cse66*state.xn[0]*state.xn[3] + 6.0732876035249664e-19*cse1015*cse1016*cse1017*cse583*cse587*state.xn[0]*state.xn[2] - 7.2151085033443149e-29*cse1018*cse1021*cse1022*cse110*cse593*state.xn[2] - cse1027*(5.9301787935659931e-17*cse1015*cse1016*cse1017*cse583*state.xn[2] - cse578*cse585) - 2.1789607199999999e-11*cse151 + 2.6241084038530201e-25*cse263*cse631*cse633*cse637*cse918*state.xn[8] - 2.6241084038530201e-25*cse263*cse917 - cse590*state.xn[0] - cse602 - cse603*(0.086512999999999993*chi*cse1019*cse1020*cse2*cse596*cse68 - cse1021*cse594*cse599) - cse658*cse688 - cse922*(-cse1029*cse634 + cse1029*cse921) - cse959*(-cse658*cse675 - cse658*cse684*cse953) - state.xn[11]*(cse1033*cse745 + cse1035*cse746 + cse1036*cse980 + cse1036*cse981) - state.xn[3]*((cse776) ? (
   0
)
: (
   cse815*cse962
)) - state.xn[7]*((cse904) ? (
   cse1000*cse1049 + cse1043*cse902 + cse1046*cse903 + cse1049*cse999
)
: (
   0
)));
  jac(19, 4) = -3.3475459999999996e-24*cse906 + cse907*(1.461185424e-21*chi*cse1061*cse184*cse185*cse923*state.xn[3] + 5.6999999999999997e-11*chi*cse184*cse185*cse621 + 3.3999999999999999e-24*chi*cse570*cse574*cse68*d2g + 1.7623947000000002e-12*crate*cse1068*cse656*state.xn[5] + 1.602177e-12*crate*cse1068*state.xn[0] + 3.2043540000000001e-12*crate*cse1068*state.xn[3] + 1.7623947000000002e-12*crate*cse198*cse655*cse919*state.xn[5] + 3.2043540000000001e-12*crate*cse655 + 2.0187430200000004e-26*cse10*cse1061*cse923*d2g*state.xn[0] - cse1051 - cse1053 - cse1055 - cse1056*cse630 - cse1057*cse1058 - 7.1777529600000012e-12*cse167 - 7.1777529600000012e-12*cse176 - 1.4355505920000002e-11*cse178 - 7.1777529600000012e-12*cse501 + 6.0000000000000005e-15*cse620*d2g*state.xn[0] - cse688 - cse820 - cse922*(-cse1057*cse1059 - cse630*cse635) - cse959*(-cse1071 - cse675 - cse684*(cse1069*cse949 + cse954*(-cse1070*cse950 + cse679))) - state.xn[11]*(cse1075*cse745 + cse1080*cse746 + cse1082*cse980 + cse1082*cse981) - state.xn[3]*((cse776) ? (
   0
)
: (
   cse794*cse962
)) - state.xn[7]*((cse904) ? (
   cse1000*cse1094 + cse1086*cse902 + cse1091*cse903 + cse1094*cse999
)
: (
   0
)));
  jac(19, 5) = -3.3466350616289996e-24*cse906 + cse907*(cse1003*cse197 + cse1006*cse1096 + cse1007*cse1096 + cse1009*cse1097 + cse1010*cse1097 + cse1012*cse1097 - cse1051 - cse1053 - cse1055 + 3.3999999999999999e-24*cse1095 - 7.1777529600000012e-12*cse207 - 7.1777529600000012e-12*cse507 + cse57*cse620 - cse959*(-cse1071 - cse684*(-cse1013*cse1070 + cse1069*cse947*cse948)));
  jac(19, 6) = -6.6464729999999996e-24*cse906 + cse907*(cse1002*cse1011 - cse1098);
  jac(19, 7) = -cse1098*cse907 - 6.6455620616289996e-24*cse906;
  jac(19, 8) = -cse905*cse907 - 1.9944730000000003e-23*cse906;
  jac(19, 9) = -cse637*cse640*cse907 - 1.9943819061629003e-23*cse906;
  jac(19, 10) = -4.6512360000000008e-23*cse906 + cse907*(-cse660*cse687 - cse959*(cse1099*cse660*cse955*cse956 - cse683*(cse1099*cse664*cse676*cse955*cse956 - cse1100*cse677) - cse684*(-cse1013*cse1100 + cse1099*cse947*cse948)));
  jac(19, 11) = cse1101 - 4.8185222061629005e-23*cse906;
  jac(19, 12) = -cse747*cse907 - 2.6567630000000005e-23*cse906;
  jac(19, 13) = -4.6637070000000001e-23*cse906;
  jac(19, 14) = -4.6636159061629004e-23*cse906;
  jac(19, 15) = cse1101 - 2.1618503000000003e-23*cse906;
  jac(19, 16) = cse1101 - 2.8241403000000005e-23*cse906;
  jac(19, 17) = -5.0204080616289998e-24*cse906 + cse907*(cse1003*cse251 + cse1006*cse1102 + cse1007*cse1102 + cse1009*cse1103 + cse1010*cse1103 + cse1012*cse1103 - 5.6994379667262474e-29*cse1050 - 9.6000000000000005e-32*cse1052 - 300.0*cse1054 + 5.0999999999999995e-24*cse1095 - 7.1777529600000012e-12*cse247 + cse510*cse620 - 7.1777529600000012e-12*cse511 - cse959*(cse1104*cse660*cse955*cse956 - cse683*(cse1104*cse664*cse676*cse955*cse956 - cse1105*cse677) - cse684*(-cse1013*cse1105 + cse1104*cse947*cse948)));
  jac(19, 18) = -2.6566719061629008e-23*cse906;
  jac(19, 19) = cse311*cse907*(1.7e-24*chi*cse10*cse1023*cse570*cse68*d2g*(-0.00012636182959999999*cse1024*cse289 - 0.018302846000000001*cse1025*cse289) + 5.6227499999999995e-25*chi*cse10*cse574*cse68*d2g*std::pow(state.T, -0.85299999999999998) - cse1005*cse1121*cse923*cse928 - cse1007*cse1121*cse929 + 7.8348995947646122e-14*cse1016*cse1109*cse582*cse583*cse587*state.xn[0]*state.xn[2] - cse1026*cse1111*cse587 - cse1027*(7.6502807803182096e-12*cse1016*cse1109*cse582*cse583*state.xn[2] - 0.0015792330470357991*cse1106*cse579 - 2.0007175426973804e-9*cse1110*cse584 - 1.3338116951315871e-8*cse1111*cse580) - cse1056*cse1117 - cse1058*cse1119 - 1.6173435610483583e-5*cse1106*cse1107*cse587 - 2.0489994438957239e-11*cse1107*cse1108*cse588 - 5.2835524837844135e-18*cse1113*cse638 - cse1118*cse688 - 2.1789607199999999e-11*cse296 - 7.1777529600000012e-12*cse515 - 7.1777529600000012e-12*cse516 - cse563*cse914*psigd_coll_interp2d_partial_1(cse10, state.T) - cse603*(-1.8989*cse1022*cse1112 + cse3*cse597) - cse603*(-0.043256499999999996*cse1112*cse596 + 0.84999999999999998*cse1112*cse599) - 1000*cse661*std::pow(cse662, 999)*cse686*cse958 - cse911*(1.6000000000000001e-32*cse110*cse276*cse565 + cse567) - cse922*(-cse1059*cse1119 - 182.41999999999999*cse1113 - cse1117*cse635) - cse959*(-cse1118*cse675 + cse1122*cse1123*cse678*cse682 + cse1122*cse1123 - cse1125*cse660*cse955 - cse683*(-cse1122*cse675*cse676 - cse1125*cse664*cse676*cse955 - cse1126*cse677) - cse684*(cse948*(cse1124*cse947 + alpha_co_interp2d_partial_0(state.T, cse673)) + cse954*(cse1118*cse679 - cse1126*cse950))) - state.xn[11]*(cse1146*cse745 + cse1152*cse746 + cse1153*cse980 + cse1153*cse981) - state.xn[3]*((cse776) ? (
   0
)
: (
   cse1128*cse819 + cse785*cse961*(-cse1128*cse817 - cse785*cse960*(-cse795*(3.0182298984217999*cse1129*cse763*cse777*cse792 + 4.8223900769399082*cse1129*cse777*cse8 - 3.5529549287336835*cse1131 - 0.3872755400043702*cse1132 - 1.3735579540080118*cse1133) - cse799*(0.0708789387907936*cse1131 + 3.7035619079275177*cse1132 - 4.6974781513675152*cse1133 + 3.6184459289309552*cse1134 - 1.6316107607322889*cse1135) - cse801*(-3.7541549062629067*cse1131 + 2.2094886994529301*cse1132 - 1.528565035159452*cse1133 + 5.0409049417480247*cse1134 + 2.0057552335975877*cse1135) - state.xn[0]*((cse803) ? (
      cse7*cse802*(116.29033200000001*cse1131 + 19.239805*cse1132 + 80.639324000000002*cse1133 + 37.383713*cse1134 + 145.968309*cse1135)
   )
   : ((cse805) ? (
      cse804*(8.2184944748967013*cse1129*cse777*cse8 - 52.189748993977005*cse1131 - 48.951834264235494*cse1132 - 196.44057098336626*cse1133 - 192.38155095558542*cse1135)
   )
   : (
      cse806*(-17.135767342440825*cse1131 + 17.889115159724135*cse1132 - 50.756388852554174*cse1133 + 10.695627721640689*cse1134 + 41.010708268606813*cse1135)
   ))) - state.xn[2]*((cse628) ? (
      cse813*(445.51869310442481*cse1131 + 11324.985706577943*cse1132 + 6766.9891340104432*cse1133 + 38.719649225812766*cse1134 + 2370.6849681533818*cse1135 + 1300.6686834345674*cse1136 + 5869.3152370465659*cse1137 + 11077.448644792414*cse1138)
   )
   : (
      cse814*(4.297112944704045*cse1131 - 101.40241318979159*cse1132 - 43.540996231705549*cse1133 + 3.8689780091986448*cse1134 + 27.91190909651122*cse1135 - 117.34186576967058*cse1136 + 103.76942484394611*cse1137 + 123.18901581604325*cse1138)
   ))))
)) - state.xn[7]*((cse904) ? (
   cse1000*cse1167 + cse1162*cse902 + cse1166*cse903 + cse1167*cse999
)
: (
   0
)));
  jac(20, 13) = -cse455;
  jac(20, 20) = cse558;
  jac(21, 13) = -cse454*state.rn[2*0+1];
  jac(21, 21) = cse558;
  // $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
