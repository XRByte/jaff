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
  const amrex::Real cse21 = std::pow(cse20, 2)/std::pow(cse16, 2);
  const amrex::Real cse22 = 1.0/cse19;
  const amrex::Real cse23 = cse22*state.xn[3];
  const amrex::Real cse24 = cse22*state.xn[0];
  const amrex::Real cse25 = std::pow(10, 0.41599999999999998*cse17*cse20 + 0.32700000000000001*cse21 - 3)*cse24 + std::pow(10, 1.3*cse17*cse20 - 1.6200000000000001*cse21 - 4.8449999999999998)*cse23;
  const amrex::Real cse26 = cse19*((cse25 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse25
));
  const amrex::Real cse27 = cse17/(cse26 + 1);
  const amrex::Real cse28 = cse26*cse27;
  const amrex::Real cse29 = state.T > 700.0;
  const amrex::Real cse30 = ((cse29) ? (
   std::pow(10, cse27*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse14)) + cse28*std::log(1.3000000000000001e-9*std::exp(-53300*cse14)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse31 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse32 = ((cse29) ? (
   std::pow(10, cse27*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse14 - 1)) + cse28*std::log(3.5199999999999998e-9*std::exp(-43900*cse14)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse33 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse34 = cse33*state.xn[2]*state.xn[16];
  const amrex::Real cse35 = state.c_hat*state.rn[0];
  const amrex::Real cse36 = cse35*state.xn[3];
  const amrex::Real cse37 = cse19*d2g;
  const amrex::Real cse38 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[1]*state.xn[11];
  const amrex::Real cse39 = std::pow(state.T, -1.3);
  const amrex::Real cse40 = std::exp(-23*cse14);
  const amrex::Real cse41 = cse39*cse40*state.xn[3]*state.xn[8];
  const amrex::Real cse42 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse43 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av)*state.xn[14];
  const amrex::Real cse44 = cse42 + cse43;
  const amrex::Real cse45 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse14) + 0.62)*state.xn[15];
  const amrex::Real cse46 = 1.3500000000000001e-9*cse45*state.xn[6];
  const amrex::Real cse47 = state.xn[3]*state.xn[6];
  const amrex::Real cse48 = 1.2599999999999999e-13*cse47*std::exp(-22.5*cse14);
  const amrex::Real cse49 = cse46 + cse48;
  const amrex::Real cse50 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse51 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse52 = -cse50 + cse51;
  const amrex::Real cse53 = crate*state.xn[9];
  const amrex::Real cse54 = 6.5199999999999996*cse53;
  const amrex::Real cse55 = 2.2999999999999998*cse23 + 1.5*cse24;
  const amrex::Real cse56 = crate*cse55;
  const amrex::Real cse57 = cse56*state.xn[0];
  const amrex::Real cse58 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse59 = state.xn[0]*state.xn[2];
  const amrex::Real cse60 = cse59*((cse29) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse58, 8) + 0.000111954395*std::pow(cse58, 7) - 0.00263197617*std::pow(cse58, 6) + 0.034825597700000002*std::pow(cse58, 5) - 0.28770560000000001*std::pow(cse58, 4) + 1.56315498*std::pow(cse58, 3) - 5.7393287500000003*std::pow(cse58, 2))
)
: (
   0.0
));
  const amrex::Real cse61 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[2]*state.xn[10];
  const amrex::Real cse62 = 4.8813573300130879e-6*std::pow(cse14, 1.5)*std::pow(114.81344434488582*std::pow(cse14, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse63 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse64 = chi*cse63;
  const amrex::Real cse65 = cse3*cse64;
  const amrex::Real cse66 = 1.7*cse65;
  const amrex::Real cse67 = std::log(state.T);
  const amrex::Real cse68 = 1.2250000000000001e-11*cse37*state.xn[1]/(1.6774414051711492e-5*std::pow(cse65, 1.3779999999999999)*(508.69999999999999*std::pow(cse66, -1.102e-5*cse67 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse69 = -cse54 - cse57 - cse60 + cse61 + cse62 + cse68;
  const amrex::Real cse70 = cse31 - cse38;
  const amrex::Real cse71 = cse18*cse56;
  const amrex::Real cse72 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[2]*state.xn[13];
  const amrex::Real cse73 = 2.0740886949963892e-18*state.xn[12];
  const amrex::Real cse74 = cse35*cse73;
  const amrex::Real cse75 = 2.1659999999999999e-12*cse37*state.xn[13]/(1.534817233071468e-7*std::pow(cse65, 1.8740000000000001)*(43750.0*std::pow(cse66, -7.538e-5*cse67 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse76 = 16800.0*crate*cse23*state.xn[12];
  const amrex::Real cse77 = (1.0537934368240927e-9*std::pow(cse14, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse14) + 1.31e-7*std::exp(-26.600000000000001*cse14) + 3.4e-8*std::exp(-7.6200000000000001*cse14) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse14)))*state.xn[7]*state.xn[16];
  const amrex::Real cse78 = cse22*state.xn[2];
  const amrex::Real cse79 = ((cse78 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse80 = cse10*cse79/(cse79 + 1);
  const amrex::Real cse81 = cse11 + 1.3007021411393064e-6*cse34 + cse77 + cse80;
  const amrex::Real cse82 = cse17*cse67;
  const amrex::Real cse83 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse37*state.xn[6]/(7.104740267838494e-7*std::pow(cse65, 1.512)*(5115.0*std::pow(cse66, -5.4939999999999998e-7*cse67 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse82*(cse82*(0.044330000000000001*cse82 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse84 = crate*state.xn[7];
  const amrex::Real cse85 = 3.8500000000000001*cse84;
  const amrex::Real cse86 = 0.15970000000000001*std::exp(-49550.0*cse14);
  const amrex::Real cse87 = ((1.6339999999999999e-6*std::exp(-15230.0*cse14) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse14) + 6.3460000000000002e-9*std::exp(-12.17*cse14))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse86 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse86 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse88 = 7.1303347066783361e-18*state.xn[7];
  const amrex::Real cse89 = cse35*cse88;
  const amrex::Real cse90 = 4.5579999999999993e-11*cse37*state.xn[8]/(0.011078790689599145*std::pow(cse65, 1.1279999999999999)*(433.10000000000002*std::pow(cse66, -0.00013329999999999999*cse67 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse91 = 1040.0*cse23*cse84;
  const amrex::Real cse92 = -cse85 + cse87 - cse89 + cse90 - cse91;
  const amrex::Real cse93 = 3.2999999999999996e-13*cse41;
  const amrex::Real cse94 = cse8 + cse9;
  const amrex::Real cse95 = cse93 + cse94;
  const amrex::Real cse96 = -cse77;
  const amrex::Real cse97 = -cse80;
  const amrex::Real cse98 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse99 = cse32*state.xn[3];
  const amrex::Real cse100 = 7.2000000000000002e-15*cse47;
  const amrex::Real cse101 = cse1 + cse100 + cse71;
  const amrex::Real cse102 = -cse51;
  const amrex::Real cse103 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse104 = cse100 + cse103 + cse49 + cse83;
  const amrex::Real cse105 = cse50 + cse96;
  const amrex::Real cse106 = 1.5094998137029966e-18*state.xn[9];
  const amrex::Real cse107 = cse106*cse35;
  const amrex::Real cse108 = 184.0*cse23*cse53;
  const amrex::Real cse109 = cse107 + cse108 - cse15;
  const amrex::Real cse110 = 9.1500000000000005e-10*cse45*state.xn[8];
  const amrex::Real cse111 = cse54 - cse61 + cse98;
  const amrex::Real cse112 = cse103 - cse42;
  const amrex::Real cse113 = cse12 + cse13 + cse97;
  const amrex::Real cse114 = -cse72 + cse74 - cse75 + cse76;
  const amrex::Real cse115 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse116 = cse115*state.xn[2];
  const amrex::Real cse117 = std::exp(-118400.0*cse14)/(4.4460389837719569e-9*cse116 + 626500000.0);
  const amrex::Real cse118 = cse66 + 50;
  const amrex::Real cse119 = std::log(1.7*cse2*cse64/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse120 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse121 = 1.75*cse120*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse120;
  const amrex::Real cse122 = l0_co_interp1d(state.T);
  const amrex::Real cse123 = 1.0/cse122;
  const amrex::Real cse124 = 100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2);
  const amrex::Real cse125 = 1.0/llte_co_interp2d(state.T, cse124);
  const amrex::Real cse126 = nhalf_co_interp2d(state.T, cse124);
  const amrex::Real cse127 = 0.001*state.T;
  const amrex::Real cse128 = cse127 + 1;
  const amrex::Real cse129 = std::pow(cse128, -2);
  const amrex::Real cse130 = 1.0/cse128;
  const amrex::Real cse131 = 1.0/(cse22*(5.6999999999999997e-11*chi*std::exp(-4.1799999999999997*av) + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse129 + 0.055500000000000001*cse130 - 11.06)*cse24 + std::pow(10, -2.0230000000000001*cse129 - 3.6709999999999998*cse130 - 11.08)*cse23) + 1);
  const amrex::Real cse132 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse133 = (1.0/100.0)*state.T;
  const amrex::Real cse134 = std::log(cse133);
  const amrex::Real cse135 = state.T < 500;
  const amrex::Real cse136 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse137 = 4.5299999999999998e-6*cse132 + 7.5799999999999997e-10*std::pow(cse133, 0.0086999999999999994*cse134 + 0.12809999999999999)*state.xn[0] + (0.75*((cse135) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse136
)) + 0.25*((cse135) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse136
)))*state.xn[3];
  const amrex::Real cse138 = cse137*std::exp(-91.209999999999994*cse14)/(cse137 + 2.3e-6);
  const amrex::Real cse139 = cse17*std::log(cse19);
  const amrex::Real cse140 = cse23*((cse139 < 2) ? (
   10.0
)
: ((cse139 < 4) ? (
   (3.0/2.0)*cse139 + 7
)
: ((cse139 < 7) ? (
   (4.0/3.0)*cse139 + 23.0/3.0
)
: ((cse139 < 10) ? (
   (1.0/3.0)*cse139 + 44.0/3.0
)
: (
   18.0
))))) + cse24*(26.399999999999999*std::sqrt(cse78/(cse78 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse141 = state.T < 100.0;
  const amrex::Real cse142 = state.T < 6000.0;
  const amrex::Real cse143 = state.T < 10;
  const amrex::Real cse144 = std::log(cse127);
  const amrex::Real cse145 = std::pow(cse144, 2);
  const amrex::Real cse146 = std::pow(cse144, 3);
  const amrex::Real cse147 = std::pow(cse144, 4);
  const amrex::Real cse148 = std::pow(cse144, 5);
  const amrex::Real cse149 = std::pow(cse144, 6);
  const amrex::Real cse150 = std::pow(cse144, 7);
  const amrex::Real cse151 = std::pow(cse144, 8);
  const amrex::Real cse152 = std::pow(10.0, 16.815729999999999*cse144 + 96.743155000000002*cse145 + 343.1918*cse146 + 734.71650999999997*cse147 + 983.67575999999997*cse148 + 801.81246999999996*cse149 + 364.14445999999998*cse150 + 70.609154000000004*cse151 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse153 = state.xn[5] + state.xn[6];
  const amrex::Real cse154 = std::pow(10.0, 1.5714710999999999*cse144 + 0.015391166*cse145 - 0.23619984999999999*cse146 - 0.51002221000000003*cse147 + 0.32168730000000001*cse148 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse144 - 0.77151435999999995*cse145 + 0.43693353000000001*cse146 - 0.14913216000000001*cse147 - 0.033638326000000003*cse148 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse144 - 0.81520437999999995*cse145 + 0.29036281000000003*cse146 - 0.16596184*cse147 + 0.19191374999999999*cse148 - 23.689236999999999)*cse153;
  const amrex::Real cse155 = std::pow(10, 37.383713*cse144 + 58.145166000000003*cse145 + 48.656103000000002*cse146 + 20.159831000000001*cse147 + 3.8479610000000002*cse148 - 16.818342000000001)*state.xn[0] + cse154;
  const amrex::Real cse156 = std::pow(10.0, 1.6802758*cse144 + 0.93310621999999999*cse145 + 4.0406627000000004*cse146 - 4.7274035999999997*cse147 - 8.8077017000000009*cse148 + 8.9167182999999994*cse149 + 6.4380698000000001*cse150 - 6.3701156000000001*cse151 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse157 = state.T < 1000.0;
  const amrex::Real cse158 = std::pow(10.0, 3.5692468000000002*cse144 - 11.33286*cse145 - 27.850082*cse146 - 21.328264000000001*cse147 - 4.2519023000000002*cse148 - 24.311209000000002)*state.xn[0] + cse154;
  const amrex::Real cse159 = std::pow(10.0, 4.6450521*cse144 - 3.7209846*cse145 + 5.9369081000000001*cse146 - 5.5108049000000001*cse147 + 1.5538288*cse148 - 24.311209000000002)*state.xn[0] + cse154;
  const amrex::Real cse160 = 1.2972734453571459e-22*state.xn[0] + 1.2615234450058559e-20*state.xn[1] + 4.6629930528575411e-23*state.xn[3] + 1.3183594641975411e-19*state.xn[5] + 1.3183594641975411e-19*state.xn[6];
  const amrex::Real cse161 = ((cse142) ? (
   cse14
)
: (
   0.00016666666666666666
));
  const amrex::Real cse162 = 1.6e-18*std::exp(-11700.0*cse161) + 6.7e-19*std::exp(-5860.0*cse161) + 3.0e-24*std::exp(-510.0*cse161) + 4.985670872372847e-33*((cse142) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse142) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse142) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse163 = std::log(0.001*((cse142) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse164 = std::pow(cse163, 2);
  const amrex::Real cse165 = std::pow(cse163, 3);
  const amrex::Real cse166 = std::pow(cse163, 4);
  const amrex::Real cse167 = std::pow(cse163, 5);
  const amrex::Real cse168 = std::pow(cse163, 6);
  const amrex::Real cse169 = std::pow(cse163, 7);
  const amrex::Real cse170 = std::pow(cse163, 8);
  const amrex::Real cse171 = 0.042999999999999997*cse134;
  const amrex::Real cse172 = 4.34e-10*std::pow(cse133, 0.755 - 0.16*cse134)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse133, 0.52500000000000002*cse134 + 1.198) + 5.2499999999999998e-13*std::pow(cse133, cse171 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse173 = 3.5700000000000001e-10*std::pow(cse133, 0.41899999999999998 - 0.0030000000000000001*cse134)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse133, 0.025000000000000001*cse134 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse133, cse171 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse174 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse175 = 3.1899999999999998e-10*std::pow(cse133, 0.36899999999999999 - 0.0060000000000000001*cse134)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse133, 0.041000000000000002*cse134 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse133, 0.058000000000000003*cse134 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse176 = cse173 + 8.9099999999999997e-5;
  const amrex::Real cse177 = cse174*cse175;
  const amrex::Real cse178 = cse172*cse173*cse174 + (1.0/15.0)*cse172*cse175*std::exp(-425.40000000000003*cse14) + cse176*cse177;
  const amrex::Real cse179 = cse172 + 1.7499999999999998e-5;
  const amrex::Real cse180 = cse175 + 1.34e-10;
  const amrex::Real cse181 = (3.0/5.0)*cse173*std::exp(-227.80000000000001*cse14);
  const amrex::Real cse182 = cse177*cse179 + cse179*cse181 + cse180*cse181;
  const amrex::Real cse183 = 1.0/((1.0/3.0)*cse172*cse180*std::exp(-98.800000000000011*cse14) + cse176*cse179 + cse176*cse180 + cse178 + cse182);
  const amrex::Real cse184 = 0.045999999999999999*cse134;
  const amrex::Real cse185 = 1.7258000000000001e-6*cse132;
  const amrex::Real cse186 = 8.9000000000000003e-11*std::pow(cse133, cse184 + 0.22800000000000001)*state.xn[0] + cse185*std::exp(((cse157) ? (
   cse67*(cse67*(cse67*(0.0070527699999999999*cse67 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse67*(cse67*(cse67*(0.093813800000000003*cse67 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse133, 0.037999999999999999*cse134 + 0.16900000000000001) + 2.15e-11*std::pow(cse133, 0.048000000000000001*cse134 - 0.01))*state.xn[3];
  const amrex::Real cse187 = 2.6400000000000002e-10*std::pow(cse133, cse184 + 0.23100000000000001)*state.xn[0] + cse185*std::exp(((cse157) ? (
   cse67*(cse67*(cse67*(0.00235272*cse67 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse67*(cse67*(cse67*(0.097857299999999994*cse67 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse133, 0.031*cse134 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse133, 0.064000000000000001*cse134 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse188 = 2.8763333333333333e-6*cse132*std::exp(((cse157) ? (
   cse67*(cse67*(cse67*(-0.00065632500000000003*cse67 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse67*(cse67*(cse67*(0.10508000000000001*cse67 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse133, 0.057000000000000002*cse134 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse133, 0.049000000000000002*cse134 - 0.0040000000000000001) + 1.675e-11*std::pow(cse133, 0.10199999999999999*cse134 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse189 = 5*std::exp(-62.460000000000001*cse14);
  const amrex::Real cse190 = cse188 + 7.8800000000000004e-8;
  const amrex::Real cse191 = cse186*cse189;
  const amrex::Real cse192 = (25.0/3.0)*cse186*cse187*std::exp(-101.30000000000001*cse14) + cse187*cse188*cse189 + cse190*cse191;
  const amrex::Real cse193 = cse187 + 2.65e-7;
  const amrex::Real cse194 = cse186 + 1.81e-14;
  const amrex::Real cse195 = 3*cse188*std::exp(-23.619999999999997*cse14);
  const amrex::Real cse196 = cse191*cse193 + cse193*cse195 + cse194*cse195;
  const amrex::Real cse197 = 1.0/((5.0/3.0)*cse187*cse194*std::exp(-38.840000000000003*cse14) + cse190*cse193 + cse190*cse194 + cse192 + cse196);
  const amrex::Real cse198 = state.c_hat*state.rn[2*0+1];

  ydot(1) = cse1 + cse11 + cse12 + cse13 + cse15 + cse18*cse32 + 2*cse30 - cse31 + 2.9935562710997168e-6*cse34 + 2.9899673672605791e-18*cse36 - 6.0000000000000005e-15*cse37*state.xn[0] + cse38 + 4.2899999999999994e-13*cse41 + cse44 + cse49 + cse52 + cse69 + 3.2000000000000001e-9*cse7 + cse9;
  ydot(2) = cse0 + cse48 + cse57 + cse60 - cse62 - cse68 + cse70;
  ydot(3) = -cse69 + cse71 - cse72 + cse74 - cse75 + cse76 - cse81 - cse83 - cse92 - cse95;
  ydot(4) = -cse101 + cse11 + 3.0000000000000002e-15*cse19*d2g*state.xn[0] - cse30 + 4.5427507615910105e-7*cse33*state.xn[2]*state.xn[16] - 1.4949836836302895e-18*cse36 - cse48 - cse52 - cse95 - cse96 - cse97 + cse98 - cse99;
  ydot(5) = cse101 + cse102;
  ydot(6) = cse104;
  ydot(7) = -cse104;
  ydot(8) = cse105 + cse109 + 9.8999999999999995e-14*cse41 + cse43 + cse92;
  ydot(9) = cse103 - cse110 + cse85 - cse87 + cse89 - cse90 + cse91 - cse93;
  ydot(10) = -cse109 - cse111 - cse112;
  ydot(11) = cse110 + cse111;
  ydot(12) = cse107 + cse108 + cse112 + cse113 + cse70 + cse8;
  ydot(13) = -cse114;
  ydot(14) = cse114;
  ydot(15) = -cse105 + 2.3099999999999997e-13*cse39*cse40*state.xn[3]*state.xn[8] - cse44;
  ydot(16) = -cse110 - cse113 - cse15 - cse46 + cse9;
  ydot(17) = -cse102 - cse81 - cse98;
  ydot(18) = cse46 - cse70 - cse94;
  ydot(19) = 1.4949836836302895e-18*state.c_hat*(2.5634832e-11*cse131 + 6.4087080000000003e-13)*state.xn[3]*state.rn[0] + 1.7e-24*chi*cse19*cse63*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse118, 0.43099999999999999)*(0.0043*std::pow(cse118, 0.69199999999999995) + 1) + 1) + 1.602177e-12*crate*cse140*state.xn[0] + 3.2043540000000001e-12*crate*cse140*state.xn[3] + 1.7623947000000002e-12*crate*cse140*state.xn[5]/cse55 - 1.3659996292638161e-10*cse115*cse117*cse59/(1.3338116951315871e-8*cse116*cse117 + 1) - cse121*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse121*cse125 + cse123*std::pow(cse121/cse126, alpha_co_interp2d(state.T, cse124))*(-cse122*cse125*cse126 + 1) + cse123) - 5.7927337833400002e-20*cse138*state.xn[8]/(2*cse138 + 1) + 3.0000000000000002e-15*cse19*d2g*(6.7291434000000005e-12*cse131 + 3.2043540000000002e-13)*state.xn[0] - 7.1777529600000012e-12*cse30 - 1.8998126555754159e-29*cse37*std::pow(state.T, 0.45350000000000001 + 2.234/cse119)*std::exp(cse119*(1.4419999999999999 - 0.050889999999999998*cse119))*state.xn[2] - 2.1789607199999999e-11*cse60 - 7.1777529600000012e-12*cse99 - (2.3872025442750958e-19*cse178*cse183 + 2.8023005140020001e-18*cse182*cse183)*state.xn[11] - ((((cse135 && cse141 && cse142) ? (
   cse143 || cse152 + cse155 < 1.0e-100
)
: (
   ((cse141 && cse142) ? (
      cse143 || cse155 + cse156 < 1.0e-100
   )
   : (
      ((cse135 && cse142 && cse157) ? (
         cse143 || cse152 + cse158 < 1.0e-100
      )
      : (
         ((cse142 && cse157) ? (
            cse143 || cse156 + cse158 < 1.0e-100
         )
         : (
            ((cse135 && cse142) ? (
               cse143 || cse152 + cse159 < 1.0e-100
            )
            : (
               ((cse142) ? (
                  cse143 || cse156 + cse159 < 1.0e-100
               )
               : (
                  ((cse135) ? (
                     cse143 || cse160 + 3.0478204245257306e+83651*state.xn[2] < 1.0e-100
                  )
                  : (
                     cse143 || cse160 + 1.0073452104972926e-205*state.xn[2] < 1.0e-100
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
   cse162/(cse162/(std::pow(10.0, 1.5714710999999999*cse163 + 0.015391166*cse164 - 0.23619984999999999*cse165 - 0.51002221000000003*cse166 + 0.32168730000000001*cse167 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse163 - 0.77151435999999995*cse164 + 0.43693353000000001*cse165 - 0.14913216000000001*cse166 - 0.033638326000000003*cse167 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse163 - 0.81520437999999995*cse164 + 0.29036281000000003*cse165 - 0.16596184*cse166 + 0.19191374999999999*cse167 - 23.689236999999999)*cse153 + ((cse135) ? (
      std::pow(10.0, 16.815729999999999*cse163 + 96.743155000000002*cse164 + 343.1918*cse165 + 734.71650999999997*cse166 + 983.67575999999997*cse167 + 801.81246999999996*cse168 + 364.14445999999998*cse169 + 70.609154000000004*cse170 - 21.928795999999998)
   )
   : (
      std::pow(10.0, 1.6802758*cse163 + 0.93310621999999999*cse164 + 4.0406627000000004*cse165 - 4.7274035999999997*cse166 - 8.8077017000000009*cse167 + 8.9167182999999994*cse168 + 6.4380698000000001*cse169 - 6.3701156000000001*cse170 - 22.921188999999998)
   ))*state.xn[2] + ((((cse142) ? (
      cse141
   )
   : (
      false
   ))) ? (
      std::pow(10, 37.383713*cse163 + 58.145166000000003*cse164 + 48.656103000000002*cse165 + 20.159831000000001*cse166 + 3.8479610000000002*cse167 - 16.818342000000001)
   )
   : ((((cse142) ? (
      cse157
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse163 - 11.33286*cse164 - 27.850082*cse165 - 21.328264000000001*cse166 - 4.2519023000000002*cse167 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 4.6450521*cse163 - 3.7209846*cse164 + 5.9369081000000001*cse165 - 5.5108049000000001*cse166 + 1.5538288*cse167 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse192*cse197 + 2.5697412351439998e-22*cse196*cse197
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse19, state.T), 3.2000000000000002e-32*cse2*cse37*(state.T - 2.1544346900318838*std::pow(85973.158732959506*cse64 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = -cse107 - 5.4061971382371991e-18*cse36 - cse74 - cse89;
  ydot(21) = -cse106*cse198 - cse198*cse73 - cse198*cse88 - 5.4061971382371991e-18*cse198*state.xn[3];
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
  const amrex::Real cse180 = 2.9899673672605791e-18*state.c_hat;
  const amrex::Real cse181 = cse24*cse25;
  const amrex::Real cse182 = cse168*cse181;
  const amrex::Real cse183 = ((cse28) ? (
   cse59*(cse173*cse181 + cse174*cse6 + cse175*cse25 + cse182)
)
: (
   0
));
  const amrex::Real cse184 = std::exp(-23*cse3);
  const amrex::Real cse185 = cse184*std::pow(state.T, -1.3);
  const amrex::Real cse186 = 4.2899999999999994e-13*cse185;
  const amrex::Real cse187 = cse138*cse67*state.xn[3];
  const amrex::Real cse188 = 1.6000000000000001e-9*cse142;
  const amrex::Real cse189 = cse188*state.xn[17];
  const amrex::Real cse190 = cse145*cse187;
  const amrex::Real cse191 = cse129*cse189 - cse129*cse190 + cse143*cse187;
  const amrex::Real cse192 = cse81*cse83;
  const amrex::Real cse193 = 2.4500000000000003e-11*cse192;
  const amrex::Real cse194 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse195 = -2.2999999999999998*cse11 + cse194;
  const amrex::Real cse196 = -cse195;
  const amrex::Real cse197 = cse196*cse89;
  const amrex::Real cse198 = cse131*cse132*cse66*state.xn[11]*state.xn[16];
  const amrex::Real cse199 = cse133*cse198;
  const amrex::Real cse200 = cse193 - cse197 - cse199;
  const amrex::Real cse201 = cse10*((cse20) ? (
   0
)
: (
   -cse172
));
  const amrex::Real cse202 = cse50*(-cse168 - cse201);
  const amrex::Real cse203 = cse202*cse22;
  const amrex::Real cse204 = cse30*((cse28) ? (
   cse54*(cse169*cse201 + cse170 + cse202*cse49 + cse203*cse31)
)
: (
   0
));
  const amrex::Real cse205 = ((cse28) ? (
   cse59*(cse181*cse201 + cse182 + cse202*cse6 + cse203*cse25)
)
: (
   0
));
  const amrex::Real cse206 = cse160*cse163*state.xn[3];
  const amrex::Real cse207 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse208 = cse206 - cse207;
  const amrex::Real cse209 = -cse194;
  const amrex::Real cse210 = cse209*cse89;
  const amrex::Real cse211 = cse193 - cse210;
  const amrex::Real cse212 = cse158*state.xn[3];
  const amrex::Real cse213 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse214 = 1.3500000000000001e-9*cse213;
  const amrex::Real cse215 = cse214*state.xn[15];
  const amrex::Real cse216 = cse212 + cse215;
  const amrex::Real cse217 = std::exp(0.11*cse3);
  const amrex::Real cse218 = cse217*state.xn[15];
  const amrex::Real cse219 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse220 = cse218*cse219;
  const amrex::Real cse221 = 6.5199999999999996*crate;
  const amrex::Real cse222 = cse149*state.xn[2];
  const amrex::Real cse223 = cse113 + cse222;
  const amrex::Real cse224 = cse142*state.xn[16];
  const amrex::Real cse225 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse226 = cse224*cse225;
  const amrex::Real cse227 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse228 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse229 = cse104*state.xn[1];
  const amrex::Real cse230 = cse228 + cse229;
  const amrex::Real cse231 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse232 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse233 = cse0*state.xn[0];
  const amrex::Real cse234 = -cse233;
  const amrex::Real cse235 = cse113 + cse124;
  const amrex::Real cse236 = cse214*state.xn[6];
  const amrex::Real cse237 = cse217*cse219*state.xn[7];
  const amrex::Real cse238 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse239 = cse236 + cse237 + cse238;
  const amrex::Real cse240 = 3*cse21;
  const amrex::Real cse241 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse242 = cse50*(-cse240 - cse241);
  const amrex::Real cse243 = cse22*cse242;
  const amrex::Real cse244 = cse30*((cse28) ? (
   cse54*(cse169*cse240 + cse169*cse241 + cse242*cse49 + cse243*cse31)
)
: (
   0
));
  const amrex::Real cse245 = ((cse28) ? (
   cse59*(cse181*cse240 + cse181*cse241 + cse242*cse6 + cse243*cse25)
)
: (
   0
));
  const amrex::Real cse246 = cse225*state.xn[11];
  const amrex::Real cse247 = cse142*cse246;
  const amrex::Real cse248 = 6.8999999999999995*cse36 + 4.5*cse38;
  const amrex::Real cse249 = -cse248;
  const amrex::Real cse250 = cse249*cse89;
  const amrex::Real cse251 = 3.6750000000000002e-11*cse192;
  const amrex::Real cse252 = cse247 - cse250 + cse251;
  const amrex::Real cse253 = cse62*state.xn[0];
  const amrex::Real cse254 = cse188*state.xn[3];
  const amrex::Real cse255 = cse129*cse254;
  const amrex::Real cse256 = std::pow(state.T, -2);
  const amrex::Real cse257 = 2.8349999999999999e-12*cse157*cse256*state.xn[3]*state.xn[6];
  const amrex::Real cse258 = -cse257;
  const amrex::Real cse259 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse260 = std::sqrt(cse3);
  const amrex::Real cse261 = std::sqrt(3)*cse260*cse3*state.xn[15];
  const amrex::Real cse262 = 1.7697487500000004e-8*cse261*state.xn[6];
  const amrex::Real cse263 = cse256*cse26;
  const amrex::Real cse264 = cse13*cse3;
  const amrex::Real cse265 = cse12*cse264;
  const amrex::Real cse266 = cse3*cse8;
  const amrex::Real cse267 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(-3.2400000000000002*cse265 + 1.3*cse3*cse8) + cse18*cse7*(0.65400000000000003*cse265 + 0.41599999999999998*cse266)
));
  const amrex::Real cse268 = cse10*cse267;
  const amrex::Real cse269 = cse268*cse50;
  const amrex::Real cse270 = cse21*cse267*cse34*cse50;
  const amrex::Real cse271 = ((cse28) ? (
   cse54*(cse169*cse268 + 1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 + 53300.0*cse263 - cse269*cse49 - cse270*cse31)
)
: (
   0
));
  const amrex::Real cse272 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse273 = 1.0/cse2;
  const amrex::Real cse274 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse275 = ((cse28) ? (
   cse59*(cse181*cse268 + 149925037481.25937*cse24*cse273*(3.3349999999999998e-12*cse273*cse5 + 4.2414529999999999e-7*cse274*cse5)*std::exp(cse4) - cse25*cse270 + 43900.0*cse263 - cse269*cse6)
)
: (
   0
));
  const amrex::Real cse276 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse277 = cse103*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse278 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse279 = cse184*state.xn[3]*state.xn[8];
  const amrex::Real cse280 = cse278*cse279;
  const amrex::Real cse281 = cse161*state.xn[3];
  const amrex::Real cse282 = 3.7768240343347639e-14*cse162*cse281;
  const amrex::Real cse283 = 7.3920000000000002e-11*cse281*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse284 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse285 = 227*cse105*cse256*state.xn[1];
  const amrex::Real cse286 = cse273*cse66;
  const amrex::Real cse287 = cse139*cse286;
  const amrex::Real cse288 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse289 = cse218*state.xn[7];
  const amrex::Real cse290 = 8.7465066538127767e-11*cse289*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse289*std::pow(state.T, -1.339);
  const amrex::Real cse291 = cse288 + cse290;
  const amrex::Real cse292 = state.xn[0]*state.xn[2];
  const amrex::Real cse293 = cse292*((cse28) ? (
   7.9971871901929577e-69*cse98*std::pow(state.T, 12.536555999999999) + cse99*(-11.478657500000001*cse3*cse91 + 4.6894649399999997*cse3*cse92 - 1.1508224*cse3*cse93 + 0.1741279885*cse3*cse94 - 0.015791857020000001*cse3*cse95 + 0.00078368076500000001*cse3*cse96 - 1.6313198799999999e-5*cse3*cse97)
)
: (
   0
));
  const amrex::Real cse294 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse295 = 7.3220359950196314e-6*cse106*cse108*cse294;
  const amrex::Real cse296 = 0.0005114031087522854*std::pow(cse107, -3.242)*cse294*std::pow(cse3, 1.907);
  const amrex::Real cse297 = 0.5*cse3;
  const amrex::Real cse298 = cse3*std::log(cse71);
  const amrex::Real cse299 = cse154*(-1.1557571281629217e-5*cse152*cse3 - cse79*(8.0679819999999989*cse74*std::pow(state.T, -0.98414000000000001) + cse76*(cse153*cse297 - 1.102e-5*cse298)));
  const amrex::Real cse300 = 9.0566037735849048e-14*cse287;
  const amrex::Real cse301 = cse142*cse300;
  const amrex::Real cse302 = cse131*cse300;
  const amrex::Real cse303 = cse129*cse302;
  const amrex::Real cse304 = -cse301 + cse303;
  const amrex::Real cse305 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse306 = 1.1264150943396225e-13*cse198*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse224*cse305;
  const amrex::Real cse307 = cse293 + cse295 - cse296 - cse299 + cse304 + cse306;
  const amrex::Real cse308 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse309 = -cse193;
  const amrex::Real cse310 = cse112 + cse85;
  const amrex::Real cse311 = -cse272 + cse277 + cse285;
  const amrex::Real cse312 = crate*cse9;
  const amrex::Real cse313 = cse312*cse88;
  const amrex::Real cse314 = 5.4939999999999998e-7*cse72 + 0.49559999999999998;
  const amrex::Real cse315 = std::pow(cse71, -cse314);
  const amrex::Real cse316 = cse315*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse317 = 5115.0*cse316;
  const amrex::Real cse318 = cse317 + 1;
  const amrex::Real cse319 = std::pow(cse70, 1.512);
  const amrex::Real cse320 = 7.104740267838494e-7*cse319;
  const amrex::Real cse321 = cse318*cse320 + 1;
  const amrex::Real cse322 = 1.0/cse321;
  const amrex::Real cse323 = 5.5720000000000001e-12*cse322;
  const amrex::Real cse324 = d2g*state.xn[6];
  const amrex::Real cse325 = cse323*cse324;
  const amrex::Real cse326 = 1040.0*state.xn[7];
  const amrex::Real cse327 = crate*cse36;
  const amrex::Real cse328 = 0.00013329999999999999*cse72 + 0.81200000000000006;
  const amrex::Real cse329 = std::pow(cse71, -cse328);
  const amrex::Real cse330 = cse329*std::pow(state.T, 0.04845);
  const amrex::Real cse331 = 433.10000000000002*cse330;
  const amrex::Real cse332 = cse331 + 1;
  const amrex::Real cse333 = std::pow(cse70, 1.1279999999999999);
  const amrex::Real cse334 = 0.011078790689599145*cse333;
  const amrex::Real cse335 = cse332*cse334 + 1;
  const amrex::Real cse336 = 1.0/cse335;
  const amrex::Real cse337 = 4.5579999999999993e-11*cse336;
  const amrex::Real cse338 = d2g*state.xn[8];
  const amrex::Real cse339 = cse326*cse327 + cse337*cse338;
  const amrex::Real cse340 = 16800.0*state.xn[12];
  const amrex::Real cse341 = 7.538e-5*cse72 + 0.89639999999999997;
  const amrex::Real cse342 = std::pow(cse71, -cse341);
  const amrex::Real cse343 = cse342*std::pow(state.T, 1.635e-6);
  const amrex::Real cse344 = 43750.0*cse343;
  const amrex::Real cse345 = cse344 + 1;
  const amrex::Real cse346 = std::pow(cse70, 1.8740000000000001);
  const amrex::Real cse347 = 1.534817233071468e-7*cse346;
  const amrex::Real cse348 = cse345*cse347 + 1;
  const amrex::Real cse349 = 1.0/cse348;
  const amrex::Real cse350 = 2.1659999999999999e-12*cse349;
  const amrex::Real cse351 = d2g*state.xn[13];
  const amrex::Real cse352 = cse327*cse340 + cse350*cse351;
  const amrex::Real cse353 = cse325 + cse339 + cse352;
  const amrex::Real cse354 = cse111*cse312;
  const amrex::Real cse355 = cse353 - cse354;
  const amrex::Real cse356 = cse72*cse8;
  const amrex::Real cse357 = 0.044330000000000001*cse356 - 0.28520000000000001;
  const amrex::Real cse358 = cse356*cse357 - 1.6759999999999999;
  const amrex::Real cse359 = cse356*cse358 + 11.19;
  const amrex::Real cse360 = cse359*state.xn[6];
  const amrex::Real cse361 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse362 = cse318*cse319;
  const amrex::Real cse363 = -cse314;
  const amrex::Real cse364 = 5.5720000000000001e-12*cse110*state.xn[6]/std::pow(cse321, 2);
  const amrex::Real cse365 = cse360*cse361 + cse364*(0.0036340746469993896*cse316*cse319*cse363*cse66 + 1.0742367284971802e-6*cse362*cse66);
  const amrex::Real cse366 = std::exp(-15230.0*cse3);
  const amrex::Real cse367 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse368 = std::exp(-12.17*cse3);
  const amrex::Real cse369 = 1.6339999999999999e-6*cse366 + 9.7930000000000004e-9*cse367 + 6.3460000000000002e-9*cse368;
  const amrex::Real cse370 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse371 = std::exp(-49550.0*cse3);
  const amrex::Real cse372 = 0.15970000000000001*cse371;
  const amrex::Real cse373 = cse372 + 1.7848999999999999;
  const amrex::Real cse374 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse375 = 0.21509999999999996 - cse372;
  const amrex::Real cse376 = std::pow(cse370, -cse373)*std::pow(cse374, -cse375);
  const amrex::Real cse377 = 2.4460185352936309e-10*cse273*cse376;
  const amrex::Real cse378 = cse274*cse369 + cse377;
  const amrex::Real cse379 = cse332*cse333;
  const amrex::Real cse380 = -cse328;
  const amrex::Real cse381 = 4.5579999999999993e-11*cse110*state.xn[8]/std::pow(cse335, 2);
  const amrex::Real cse382 = cse378*state.xn[8] + cse381*(4.7982242476653898*cse330*cse333*cse380*cse66 + 0.012496875897867835*cse379*cse66);
  const amrex::Real cse383 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse384 = cse345*cse346;
  const amrex::Real cse385 = -cse341;
  const amrex::Real cse386 = 2.1659999999999999e-12*cse110*state.xn[13]/std::pow(cse348, 2);
  const amrex::Real cse387 = cse383*state.xn[13] + cse386*(0.0067148253946876721*cse343*cse346*cse385*cse66 + 2.8762474947759312e-7*cse384*cse66);
  const amrex::Real cse388 = cse146 + cse148;
  const amrex::Real cse389 = 1.3007021411393064e-6*cse135;
  const amrex::Real cse390 = cse11*state.xn[2];
  const amrex::Real cse391 = cse390 < 1.0e-50;
  const amrex::Real cse392 = ((cse391) ? (
   0
)
: (
   -0.00011320754716981131*cse137*state.xn[3]
));
  const amrex::Real cse393 = ((cse391) ? (
   10000000000.0
)
: (
   cse128
));
  const amrex::Real cse394 = cse393 + 1;
  const amrex::Real cse395 = 1.0/cse394;
  const amrex::Real cse396 = cse246*state.xn[16];
  const amrex::Real cse397 = cse395*cse396;
  const amrex::Real cse398 = cse392*cse397;
  const amrex::Real cse399 = cse393*cse396/std::pow(cse394, 2);
  const amrex::Real cse400 = cse392*cse399;
  const amrex::Real cse401 = cse398 - cse400;
  const amrex::Real cse402 = cse134 + cse401;
  const amrex::Real cse403 = cse389*state.xn[16] + cse402;
  const amrex::Real cse404 = cse322*cse324;
  const amrex::Real cse405 = 1.1144e-11*cse404;
  const amrex::Real cse406 = 2*cse65;
  const amrex::Real cse407 = cse196*cse312;
  const amrex::Real cse408 = crate*cse11;
  const amrex::Real cse409 = cse327*state.xn[12];
  const amrex::Real cse410 = cse349*cse351;
  const amrex::Real cse411 = 33600.0*cse409 + 4.3319999999999997e-12*cse410;
  const amrex::Real cse412 = -cse340*cse408 + cse411;
  const amrex::Real cse413 = 3.2999999999999996e-13*cse185;
  const amrex::Real cse414 = cse413*state.xn[8];
  const amrex::Real cse415 = -cse326*cse408;
  const amrex::Real cse416 = cse327*state.xn[7];
  const amrex::Real cse417 = cse336*cse338;
  const amrex::Real cse418 = 2080.0*cse416 + 9.1159999999999986e-11*cse417;
  const amrex::Real cse419 = cse414 + cse415 + cse418;
  const amrex::Real cse420 = ((cse391) ? (
   0
)
: (
   cse127
));
  const amrex::Real cse421 = cse397*cse420;
  const amrex::Real cse422 = cse399*cse420;
  const amrex::Real cse423 = cse421 - cse422;
  const amrex::Real cse424 = cse191 + cse423;
  const amrex::Real cse425 = cse189 - cse190;
  const amrex::Real cse426 = cse209*cse312;
  const amrex::Real cse427 = -cse426;
  const amrex::Real cse428 = 1.1000000000000001*crate;
  const amrex::Real cse429 = cse361*state.xn[2];
  const amrex::Real cse430 = cse110*cse323 + cse359*cse429;
  const amrex::Real cse431 = std::pow(cse3, 0.00231);
  const amrex::Real cse432 = std::pow(state.T, -1.5);
  const amrex::Real cse433 = std::exp(-8110.0*cse3);
  const amrex::Real cse434 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse435 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse436 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse437 = 0.00015100000000000001*cse433 + 1.31e-7*cse434 + 3.4e-8*cse435 + 6.9699999999999997e-9*cse436;
  const amrex::Real cse438 = 1.0537934368240927e-9*cse431 + cse432*cse437;
  const amrex::Real cse439 = cse438*state.xn[16];
  const amrex::Real cse440 = -cse439;
  const amrex::Real cse441 = state.c_hat*state.rn[0];
  const amrex::Real cse442 = 7.1303347066783361e-18*cse441;
  const amrex::Real cse443 = cse11*state.xn[3];
  const amrex::Real cse444 = crate*cse443;
  const amrex::Real cse445 = 3.8500000000000001*crate + cse442 + 1040.0*cse444;
  const amrex::Real cse446 = cse413*state.xn[3];
  const amrex::Real cse447 = cse110*cse337 + cse378*state.xn[2];
  const amrex::Real cse448 = cse446 + cse447;
  const amrex::Real cse449 = cse393*cse395;
  const amrex::Real cse450 = cse449*state.xn[16];
  const amrex::Real cse451 = cse225*cse450;
  const amrex::Real cse452 = cse226 + cse451;
  const amrex::Real cse453 = -cse452;
  const amrex::Real cse454 = 2.0740886949963892e-18*cse441;
  const amrex::Real cse455 = 16800.0*cse444 + cse454;
  const amrex::Real cse456 = cse110*cse350 + cse383*state.xn[2];
  const amrex::Real cse457 = -cse456;
  const amrex::Real cse458 = -cse113 - cse355;
  const amrex::Real cse459 = cse246*cse449;
  const amrex::Real cse460 = cse438*state.xn[7];
  const amrex::Real cse461 = cse389*state.xn[2];
  const amrex::Real cse462 = cse249*cse312;
  const amrex::Real cse463 = -cse462;
  const amrex::Real cse464 = 1.6715999999999999e-11*cse404;
  const amrex::Real cse465 = 3120.0*cse416 + 1.3673999999999998e-10*cse417;
  const amrex::Real cse466 = 50400.0*cse409 + 6.4979999999999992e-12*cse410;
  const amrex::Real cse467 = cse254 + cse255;
  const amrex::Real cse468 = -cse467;
  const amrex::Real cse469 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse431 + cse432*(1.2246100000000002*cse256*cse433 + 3.4846000000000001e-6*cse256*cse434 + 2.5908000000000001e-7*cse256*cse435 + 9.6185999999999982e-9*cse256*cse436) - 1.5*cse437*std::pow(state.T, -2.5));
  const amrex::Real cse470 = -cse469;
  const amrex::Real cse471 = 6.7636511339243933e-7*cse276;
  const amrex::Real cse472 = cse305*cse450;
  const amrex::Real cse473 = ((cse391) ? (
   0
)
: (
   5.6603773584905654e-5*cse286*state.xn[3]
));
  const amrex::Real cse474 = cse397*cse473;
  const amrex::Real cse475 = -cse474;
  const amrex::Real cse476 = cse399*cse473;
  const amrex::Real cse477 = 7913.1350000000002*cse256*cse371;
  const amrex::Real cse478 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse274*cse376 + cse274*(0.024885819999999999*cse256*cse366 + 7.2272340000000005e-7*cse256*cse367 + 7.7230820000000007e-8*cse256*cse368) - 3.0/2.0*cse369/std::pow(state.T, 5.0/2.0) + cse377*(-0.00035870183966449452*cse273*cse373/cse370 - cse477*std::log(cse370)) + cse377*(-6.1221939997287613*cse273*cse375/cse374 + cse477*std::log(cse374)));
  const amrex::Real cse479 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse480 = 4.9999999999999997e-12*cse360*cse432*state.xn[2];
  const amrex::Real cse481 = cse279*cse284;
  const amrex::Real cse482 = 7.5899999999999998e-12*cse481;
  const amrex::Real cse483 = cse429*state.xn[6]*(cse266*cse358 + cse356*(0.044330000000000001*cse264*cse72 + cse266*cse357));
  const amrex::Real cse484 = cse386*(-1.4381237473879656e-7*cse3*cse384 - cse347*(0.071531250000000005*cse342*std::pow(state.T, -0.99999836499999994) + cse344*(cse297*cse385 - 7.538e-5*cse298)));
  const amrex::Real cse485 = cse364*(-5.3711836424859012e-7*cse3*cse362 - cse320*(0.0019963845000000001*cse315*std::pow(state.T, -0.99999960970000001) + cse317*(cse297*cse363 - 5.4939999999999998e-7*cse298)));
  const amrex::Real cse486 = cse381*(-0.0062484379489339175*cse3*cse379 - cse334*(20.983695000000001*cse329*std::pow(state.T, -0.95155000000000001) + cse331*(cse297*cse380 - 0.00013329999999999999*cse298)));
  const amrex::Real cse487 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse488 = cse487*state.xn[12];
  const amrex::Real cse489 = 7.1303347066783361e-18*state.c_hat;
  const amrex::Real cse490 = cse489*state.xn[7];
  const amrex::Real cse491 = cse488 + cse490;
  const amrex::Real cse492 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse493 = cse29*state.xn[3];
  const amrex::Real cse494 = cse30*cse55;
  const amrex::Real cse495 = -cse1;
  const amrex::Real cse496 = state.xn[0]*state.xn[3];
  const amrex::Real cse497 = cse496*cse60;
  const amrex::Real cse498 = cse313 - cse61;
  const amrex::Real cse499 = cse122*cse496;
  const amrex::Real cse500 = cse120 + cse354 + cse492 + cse499;
  const amrex::Real cse501 = -cse500;
  const amrex::Real cse502 = 4.5427507615910105e-7*cse135;
  const amrex::Real cse503 = cse183*cse496;
  const amrex::Real cse504 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse505 = cse159 + cse504;
  const amrex::Real cse506 = cse406 + cse407;
  const amrex::Real cse507 = cse164 + cse199 - cse421 + cse422;
  const amrex::Real cse508 = cse191 + cse425;
  const amrex::Real cse509 = cse205*cse496;
  const amrex::Real cse510 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse511 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse512 = 9.0000000000000011e-15*cse165;
  const amrex::Real cse513 = cse245*cse496;
  const amrex::Real cse514 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse515 = cse247 + cse459 + cse460 + cse514;
  const amrex::Real cse516 = -cse259;
  const amrex::Real cse517 = cse271*cse30;
  const amrex::Real cse518 = cse275*cse496;
  const amrex::Real cse519 = -4.2899999999999994e-13*cse184*cse278*state.xn[3]*state.xn[8] + cse482;
  const amrex::Real cse520 = cse301 - cse303;
  const amrex::Real cse521 = -cse282 + cse283;
  const amrex::Real cse522 = cse306 + cse470 + cse472 + cse475 + cse476 + cse521;
  const amrex::Real cse523 = state.c_hat*state.xn[3];
  const amrex::Real cse524 = 1.4949836836302895e-18*cse523;
  const amrex::Real cse525 = cse405 + cse505;
  const amrex::Real cse526 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse527 = cse216 + cse430 + cse510 + cse526;
  const amrex::Real cse528 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse529 = cse236 + cse325;
  const amrex::Real cse530 = -cse262;
  const amrex::Real cse531 = cse257 - cse480 + cse483 + cse485 + cse530;
  const amrex::Real cse532 = -cse325;
  const amrex::Real cse533 = 184.0*state.xn[9];
  const amrex::Real cse534 = cse327*cse533;
  const amrex::Real cse535 = -cse534;
  const amrex::Real cse536 = cse339 + cse535;
  const amrex::Real cse537 = cse408*cse533;
  const amrex::Real cse538 = 9.8999999999999995e-14*cse185;
  const amrex::Real cse539 = cse327*state.xn[9];
  const amrex::Real cse540 = 368.0*cse539;
  const amrex::Real cse541 = -cse540;
  const amrex::Real cse542 = cse418 + cse541;
  const amrex::Real cse543 = 1.5094998137029966e-18*cse441;
  const amrex::Real cse544 = 184.0*cse444 + cse543;
  const amrex::Real cse545 = cse232 + cse233;
  const amrex::Real cse546 = 552.0*cse539;
  const amrex::Real cse547 = cse478 + cse486;
  const amrex::Real cse548 = 1.5094998137029966e-18*state.c_hat;
  const amrex::Real cse549 = cse548*state.xn[9];
  const amrex::Real cse550 = -cse549;
  const amrex::Real cse551 = -cse339;
  const amrex::Real cse552 = 9.1500000000000005e-10*cse213;
  const amrex::Real cse553 = cse552*state.xn[15];
  const amrex::Real cse554 = cse552*state.xn[8];
  const amrex::Real cse555 = 1.1994963750000002e-8*cse261*state.xn[8];
  const amrex::Real cse556 = -cse555;
  const amrex::Real cse557 = -cse537 + cse540;
  const amrex::Real cse558 = cse221 + cse511;
  const amrex::Real cse559 = cse528 + cse544;
  const amrex::Real cse560 = cse231 + cse534;
  const amrex::Real cse561 = -cse63;
  const amrex::Real cse562 = -cse227 + cse451;
  const amrex::Real cse563 = -cse472 + cse474 - cse476;
  const amrex::Real cse564 = cse302 + cse311;
  const amrex::Real cse565 = -cse479 + cse484;
  const amrex::Real cse566 = -cse352;
  const amrex::Real cse567 = 2.3099999999999997e-13*cse185;
  const amrex::Real cse568 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse569 = cse71 + 50;
  const amrex::Real cse570 = 0.0043*std::pow(cse569, 0.69199999999999995) + 1;
  const amrex::Real cse571 = 0.049959999999999997*std::pow(cse569, 0.43099999999999999)*cse570 + 1;
  const amrex::Real cse572 = 1.0/cse571;
  const amrex::Real cse573 = state.xn[2] + 1.0e-50;
  const amrex::Real cse574 = 1.0/cse573;
  const amrex::Real cse575 = cse2*cse69;
  const amrex::Real cse576 = 1.7*cse575;
  const amrex::Real cse577 = cse574*cse576 + 50;
  const amrex::Real cse578 = std::log(cse577);
  const amrex::Real cse579 = 0.45350000000000001 + 2.234/cse578;
  const amrex::Real cse580 = std::pow(state.T, cse579);
  const amrex::Real cse581 = 1.4419999999999999 - 0.050889999999999998*cse578;
  const amrex::Real cse582 = std::exp(cse578*cse581);
  const amrex::Real cse583 = 1.8998126555754159e-29*cse580*cse582;
  const amrex::Real cse584 = d2g*state.xn[2];
  const amrex::Real cse585 = cse583*cse584;
  const amrex::Real cse586 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse587 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse588 = 1.0/cse587;
  const amrex::Real cse589 = cse588*state.xn[2];
  const amrex::Real cse590 = cse586*cse589;
  const amrex::Real cse591 = 4.4460389837719569e-9*cse590 + 626500000.0;
  const amrex::Real cse592 = 1.0/cse591;
  const amrex::Real cse593 = std::exp(-118400.0*cse3);
  const amrex::Real cse594 = cse592*cse593;
  const amrex::Real cse595 = 1.3338116951315871e-8*cse594;
  const amrex::Real cse596 = cse590*cse595 + 1;
  const amrex::Real cse597 = 1.0/cse596;
  const amrex::Real cse598 = cse594*cse597;
  const amrex::Real cse599 = 1.3659996292638161e-10*cse598;
  const amrex::Real cse600 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse601 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse602 = 100.0*d2g;
  const amrex::Real cse603 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse69 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse604 = cse110*cse600;
  const amrex::Real cse605 = -cse601*cse602 + cse603*cse604;
  const amrex::Real cse606 = (((cse605 > 0) ? (
   0
)
: ((cse605 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse607 = cse603*cse606*d2g;
  const amrex::Real cse608 = cse600*cse607;
  const amrex::Real cse609 = (((cse605 < 0) ? (
   0
)
: ((cse605 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse610 = cse609*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse611 = cse602*cse610;
  const amrex::Real cse612 = 0.001*state.T;
  const amrex::Real cse613 = cse612 + 1;
  const amrex::Real cse614 = std::pow(cse613, -2);
  const amrex::Real cse615 = 1.0/cse613;
  const amrex::Real cse616 = std::pow(10, -2.0230000000000001*cse614 - 3.6709999999999998*cse615 - 11.08);
  const amrex::Real cse617 = cse11*cse616;
  const amrex::Real cse618 = cse617*state.xn[3];
  const amrex::Real cse619 = std::pow(10, -2.3900000000000001*cse614 + 0.055500000000000001*cse615 - 11.06);
  const amrex::Real cse620 = cse11*cse619;
  const amrex::Real cse621 = cse620*state.xn[0];
  const amrex::Real cse622 = cse618 + cse621;
  const amrex::Real cse623 = 1.0/cse622;
  const amrex::Real cse624 = 5.6999999999999997e-11*chi*std::exp(-4.1799999999999997*av) + 1.9999999999999999e-7;
  const amrex::Real cse625 = cse623*cse624;
  const amrex::Real cse626 = cse11*cse625 + 1;
  const amrex::Real cse627 = 1.0/cse626;
  const amrex::Real cse628 = 6.7291434000000005e-12*cse627 + 3.2043540000000002e-13;
  const amrex::Real cse629 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse630 = 1.75*cse629;
  const amrex::Real cse631 = std::exp(-state.T);
  const amrex::Real cse632 = 1 - cse631;
  const amrex::Real cse633 = std::pow(cse632, 1000);
  const amrex::Real cse634 = l0_co_interp1d(state.T);
  const amrex::Real cse635 = 1.0/cse634;
  const amrex::Real cse636 = 680.10000000000002/cse629;
  const amrex::Real cse637 = cse630*state.xn[0] + cse636*state.xn[2] + state.xn[3];
  const amrex::Real cse638 = M_SQRT2;
  const amrex::Real cse639 = std::max(gradv, 5.7445531974402559e-18*cse2*cse638);
  const amrex::Real cse640 = 1.0/cse639;
  const amrex::Real cse641 = 100000.0*cse640;
  const amrex::Real cse642 = cse641*state.xn[9];
  const amrex::Real cse643 = llte_co_interp2d(state.T, cse642);
  const amrex::Real cse644 = 1.0/cse643;
  const amrex::Real cse645 = nhalf_co_interp2d(state.T, cse642);
  const amrex::Real cse646 = cse634*cse644;
  const amrex::Real cse647 = -cse645*cse646 + 1;
  const amrex::Real cse648 = 1.0/cse645;
  const amrex::Real cse649 = cse637*cse648;
  const amrex::Real cse650 = alpha_co_interp2d(state.T, cse642);
  const amrex::Real cse651 = std::pow(cse649, cse650);
  const amrex::Real cse652 = cse635*cse651;
  const amrex::Real cse653 = cse647*cse652;
  const amrex::Real cse654 = cse635 + cse637*cse644 + cse653;
  const amrex::Real cse655 = 1.0/cse654;
  const amrex::Real cse656 = cse633*cse655;
  const amrex::Real cse657 = cse656*state.xn[9];
  const amrex::Real cse658 = cse260*state.xn[2];
  const amrex::Real cse659 = (1.0/100.0)*state.T;
  const amrex::Real cse660 = std::log(cse659);
  const amrex::Real cse661 = 0.0086999999999999994*cse660;
  const amrex::Real cse662 = std::pow(cse659, cse661 + 0.12809999999999999);
  const amrex::Real cse663 = 7.5799999999999997e-10*cse662*state.xn[0];
  const amrex::Real cse664 = state.T < 500;
  const amrex::Real cse665 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse666 = 0.75*((cse664) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse665
)) + 0.25*((cse664) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse665
));
  const amrex::Real cse667 = 4.5299999999999998e-6*cse658 + cse663 + cse666*state.xn[3];
  const amrex::Real cse668 = cse667 + 2.3e-6;
  const amrex::Real cse669 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse670 = cse669/cse668;
  const amrex::Real cse671 = 2*cse670;
  const amrex::Real cse672 = cse667*cse671 + 1;
  const amrex::Real cse673 = 1.0/cse672;
  const amrex::Real cse674 = cse670*state.xn[8];
  const amrex::Real cse675 = cse673*cse674;
  const amrex::Real cse676 = cse390 + 0.070000000000000007;
  const amrex::Real cse677 = 1.0/cse676;
  const amrex::Real cse678 = std::sqrt(cse390*cse677);
  const amrex::Real cse679 = 26.399999999999999*cse678;
  const amrex::Real cse680 = cse679 + 6.5;
  const amrex::Real cse681 = cse11*cse680;
  const amrex::Real cse682 = cse8*std::log(cse10);
  const amrex::Real cse683 = cse682 < 2;
  const amrex::Real cse684 = cse682 < 4;
  const amrex::Real cse685 = cse682 < 7;
  const amrex::Real cse686 = cse682 < 10;
  const amrex::Real cse687 = ((cse683) ? (
   10.0
)
: ((cse684) ? (
   (3.0/2.0)*cse682 + 7
)
: ((cse685) ? (
   (4.0/3.0)*cse682 + 23.0/3.0
)
: ((cse686) ? (
   (1.0/3.0)*cse682 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse688 = cse11*cse687;
  const amrex::Real cse689 = cse681*state.xn[0] + cse688*state.xn[3];
  const amrex::Real cse690 = std::pow(cse668, -2);
  const amrex::Real cse691 = cse650/cse637;
  const amrex::Real cse692 = cse653*cse691;
  const amrex::Real cse693 = cse637*state.xn[9];
  const amrex::Real cse694 = cse633*cse693/std::pow(cse654, 2);
  const amrex::Real cse695 = std::pow(cse64, -2);
  const amrex::Real cse696 = std::pow(cse626, -2);
  const amrex::Real cse697 = cse35*cse625;
  const amrex::Real cse698 = cse36*cse616;
  const amrex::Real cse699 = cse38*cse619;
  const amrex::Real cse700 = cse698 + cse699;
  const amrex::Real cse701 = cse624/std::pow(cse622, 2);
  const amrex::Real cse702 = cse11*cse701;
  const amrex::Real cse703 = cse697 - cse702*(-cse620 + cse700);
  const amrex::Real cse704 = 1.5159999999999999e-9*cse662;
  const amrex::Real cse705 = cse667*cse669*cse690;
  const amrex::Real cse706 = cse667*cse670;
  const amrex::Real cse707 = 5.7927337833400002e-20*cse706*state.xn[8]/std::pow(cse672, 2);
  const amrex::Real cse708 = cse38*cse680;
  const amrex::Real cse709 = cse35*state.xn[2];
  const amrex::Real cse710 = std::pow(cse676, -2);
  const amrex::Real cse711 = 0.5*cse710;
  const amrex::Real cse712 = cse125/std::pow(cse10, 3);
  const amrex::Real cse713 = -0.5*cse677*cse709 + cse711*cse712;
  const amrex::Real cse714 = cse66*cse676*cse679*state.xn[0];
  const amrex::Real cse715 = cse11*cse8;
  const amrex::Real cse716 = ((cse683) ? (
   0
)
: ((cse684) ? (
   (3.0/2.0)*cse715
)
: ((cse685) ? (
   (4.0/3.0)*cse715
)
: ((cse686) ? (
   (1.0/3.0)*cse715
)
: (
   0
)))));
  const amrex::Real cse717 = cse36*cse687;
  const amrex::Real cse718 = cse443*cse716 + cse681 - cse708 + cse713*cse714 - cse717;
  const amrex::Real cse719 = 1.0/cse64;
  const amrex::Real cse720 = state.T < 100.0;
  const amrex::Real cse721 = state.T < 6000.0;
  const amrex::Real cse722 = state.T < 10;
  const amrex::Real cse723 = std::log(cse612);
  const amrex::Real cse724 = std::pow(cse723, 2);
  const amrex::Real cse725 = std::pow(cse723, 3);
  const amrex::Real cse726 = std::pow(cse723, 4);
  const amrex::Real cse727 = std::pow(cse723, 5);
  const amrex::Real cse728 = std::pow(cse723, 6);
  const amrex::Real cse729 = std::pow(cse723, 7);
  const amrex::Real cse730 = std::pow(cse723, 8);
  const amrex::Real cse731 = std::pow(10.0, 16.815729999999999*cse723 + 96.743155000000002*cse724 + 343.1918*cse725 + 734.71650999999997*cse726 + 983.67575999999997*cse727 + 801.81246999999996*cse728 + 364.14445999999998*cse729 + 70.609154000000004*cse730 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse732 = state.xn[5] + state.xn[6];
  const amrex::Real cse733 = std::pow(10.0, 1.5714710999999999*cse723 + 0.015391166*cse724 - 0.23619984999999999*cse725 - 0.51002221000000003*cse726 + 0.32168730000000001*cse727 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse723 - 0.77151435999999995*cse724 + 0.43693353000000001*cse725 - 0.14913216000000001*cse726 - 0.033638326000000003*cse727 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse723 - 0.81520437999999995*cse724 + 0.29036281000000003*cse725 - 0.16596184*cse726 + 0.19191374999999999*cse727 - 23.689236999999999)*cse732;
  const amrex::Real cse734 = std::pow(10, 37.383713*cse723 + 58.145166000000003*cse724 + 48.656103000000002*cse725 + 20.159831000000001*cse726 + 3.8479610000000002*cse727 - 16.818342000000001)*state.xn[0] + cse733;
  const amrex::Real cse735 = std::pow(10.0, 1.6802758*cse723 + 0.93310621999999999*cse724 + 4.0406627000000004*cse725 - 4.7274035999999997*cse726 - 8.8077017000000009*cse727 + 8.9167182999999994*cse728 + 6.4380698000000001*cse729 - 6.3701156000000001*cse730 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse736 = state.T < 1000.0;
  const amrex::Real cse737 = std::pow(10.0, 3.5692468000000002*cse723 - 11.33286*cse724 - 27.850082*cse725 - 21.328264000000001*cse726 - 4.2519023000000002*cse727 - 24.311209000000002)*state.xn[0] + cse733;
  const amrex::Real cse738 = std::pow(10.0, 4.6450521*cse723 - 3.7209846*cse724 + 5.9369081000000001*cse725 - 5.5108049000000001*cse726 + 1.5538288*cse727 - 24.311209000000002)*state.xn[0] + cse733;
  const amrex::Real cse739 = 1.2972734453571459e-22*state.xn[0] + 1.2615234450058559e-20*state.xn[1] + 4.6629930528575411e-23*state.xn[3] + 1.3183594641975411e-19*state.xn[5] + 1.3183594641975411e-19*state.xn[6];
  const amrex::Real cse740 = ((cse664 && cse720 && cse721) ? (
   cse722 || cse731 + cse734 < 1.0e-100
)
: (
   ((cse720 && cse721) ? (
      cse722 || cse734 + cse735 < 1.0e-100
   )
   : (
      ((cse664 && cse721 && cse736) ? (
         cse722 || cse731 + cse737 < 1.0e-100
      )
      : (
         ((cse721 && cse736) ? (
            cse722 || cse735 + cse737 < 1.0e-100
         )
         : (
            ((cse664 && cse721) ? (
               cse722 || cse731 + cse738 < 1.0e-100
            )
            : (
               ((cse721) ? (
                  cse722 || cse735 + cse738 < 1.0e-100
               )
               : (
                  ((cse664) ? (
                     cse722 || cse739 + 3.0478204245257306e+83651*state.xn[2] < 1.0e-100
                  )
                  : (
                     cse722 || cse739 + 1.0073452104972926e-205*state.xn[2] < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse741 = std::log(0.001*((cse721) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse742 = std::pow(cse741, 2);
  const amrex::Real cse743 = std::pow(cse741, 3);
  const amrex::Real cse744 = std::pow(cse741, 4);
  const amrex::Real cse745 = std::pow(cse741, 5);
  const amrex::Real cse746 = std::pow(10, 37.383713*cse741 + 58.145166000000003*cse742 + 48.656103000000002*cse743 + 20.159831000000001*cse744 + 3.8479610000000002*cse745 - 16.818342000000001);
  const amrex::Real cse747 = ((cse721) ? (
   cse720
)
: (
   false
));
  const amrex::Real cse748 = std::pow(10.0, 3.5692468000000002*cse741 - 11.33286*cse742 - 27.850082*cse743 - 21.328264000000001*cse744 - 4.2519023000000002*cse745 - 24.311209000000002);
  const amrex::Real cse749 = ((cse721) ? (
   cse736
)
: (
   false
));
  const amrex::Real cse750 = std::pow(10.0, 4.6450521*cse741 - 3.7209846*cse742 + 5.9369081000000001*cse743 - 5.5108049000000001*cse744 + 1.5538288*cse745 - 24.311209000000002);
  const amrex::Real cse751 = ((cse747) ? (
   cse746
)
: ((cse749) ? (
   cse748
)
: (
   cse750
)));
  const amrex::Real cse752 = std::pow(10.0, 2.0943374000000001*cse741 - 0.77151435999999995*cse742 + 0.43693353000000001*cse743 - 0.14913216000000001*cse744 - 0.033638326000000003*cse745 - 23.962112000000001);
  const amrex::Real cse753 = cse752*state.xn[3];
  const amrex::Real cse754 = std::pow(10.0, 1.5714710999999999*cse741 + 0.015391166*cse742 - 0.23619984999999999*cse743 - 0.51002221000000003*cse744 + 0.32168730000000001*cse745 - 22.089523);
  const amrex::Real cse755 = cse754*state.xn[1];
  const amrex::Real cse756 = std::pow(10.0, 2.1892372*cse741 - 0.81520437999999995*cse742 + 0.29036281000000003*cse743 - 0.16596184*cse744 + 0.19191374999999999*cse745 - 23.689236999999999);
  const amrex::Real cse757 = cse732*cse756;
  const amrex::Real cse758 = std::pow(cse741, 6);
  const amrex::Real cse759 = std::pow(cse741, 7);
  const amrex::Real cse760 = std::pow(cse741, 8);
  const amrex::Real cse761 = std::pow(10.0, 16.815729999999999*cse741 + 96.743155000000002*cse742 + 343.1918*cse743 + 734.71650999999997*cse744 + 983.67575999999997*cse745 + 801.81246999999996*cse758 + 364.14445999999998*cse759 + 70.609154000000004*cse760 - 21.928795999999998);
  const amrex::Real cse762 = std::pow(10.0, 1.6802758*cse741 + 0.93310621999999999*cse742 + 4.0406627000000004*cse743 - 4.7274035999999997*cse744 - 8.8077017000000009*cse745 + 8.9167182999999994*cse758 + 6.4380698000000001*cse759 - 6.3701156000000001*cse760 - 22.921188999999998);
  const amrex::Real cse763 = ((cse664) ? (
   cse761
)
: (
   cse762
));
  const amrex::Real cse764 = cse751*state.xn[0] + cse753 + cse755 + cse757 + cse763*state.xn[2];
  const amrex::Real cse765 = std::pow(cse764, -2);
  const amrex::Real cse766 = ((cse721) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse767 = std::exp(-11700.0*cse766);
  const amrex::Real cse768 = std::exp(-5860.0*cse766);
  const amrex::Real cse769 = std::exp(-510.0*cse766);
  const amrex::Real cse770 = std::exp(-2197000.0*((cse721) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse771 = cse770*((cse721) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse772 = 6.0142468035272636e-8*((cse721) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse773 = 1.0/cse772;
  const amrex::Real cse774 = 1.6e-18*cse767 + 6.7e-19*cse768 + 3.0e-24*cse769 + 4.985670872372847e-33*cse771*cse773;
  const amrex::Real cse775 = 1.0/cse764;
  const amrex::Real cse776 = cse774*cse775 + 1;
  const amrex::Real cse777 = std::pow(cse776, -2);
  const amrex::Real cse778 = cse765*std::pow(cse774, 2)*cse777;
  const amrex::Real cse779 = 0.0060000000000000001*cse660;
  const amrex::Real cse780 = std::pow(cse659, 0.36899999999999999 - cse779);
  const amrex::Real cse781 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse782 = 1.08e-14*cse781;
  const amrex::Real cse783 = 0.16*cse660;
  const amrex::Real cse784 = std::pow(cse659, 0.755 - cse783);
  const amrex::Real cse785 = 4.34e-10*cse784;
  const amrex::Real cse786 = cse785*state.xn[0];
  const amrex::Real cse787 = 0.042999999999999997*cse660;
  const amrex::Real cse788 = 5.2499999999999998e-13*std::pow(cse659, cse787 + 0.88900000000000001);
  const amrex::Real cse789 = 0.52500000000000002*cse660;
  const amrex::Real cse790 = 2.2500000000000003e-12*std::pow(cse659, cse789 + 1.198);
  const amrex::Real cse791 = cse788 + cse790;
  const amrex::Real cse792 = cse782*state.xn[2] + cse786 + cse791*state.xn[3];
  const amrex::Real cse793 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse794 = cse792*cse793;
  const amrex::Real cse795 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse796 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse797 = 4.8599999999999998e-10*cse796;
  const amrex::Real cse798 = 3.1899999999999998e-10*cse780;
  const amrex::Real cse799 = cse798*state.xn[0];
  const amrex::Real cse800 = 0.041000000000000002*cse660;
  const amrex::Real cse801 = 4.7499999999999998e-11*std::pow(cse659, cse800 + 0.20300000000000001);
  const amrex::Real cse802 = 0.058000000000000003*cse660;
  const amrex::Real cse803 = 1.6725000000000002e-10*std::pow(cse659, cse802 + 0.23699999999999999);
  const amrex::Real cse804 = cse801 + cse803;
  const amrex::Real cse805 = cse797*state.xn[2] + cse799 + cse804*state.xn[3];
  const amrex::Real cse806 = cse795*cse805;
  const amrex::Real cse807 = 0.0030000000000000001*cse660;
  const amrex::Real cse808 = std::pow(cse659, 0.41899999999999998 - cse807);
  const amrex::Real cse809 = 7.1400000000000007e-11*cse808;
  const amrex::Real cse810 = cse792*cse795;
  const amrex::Real cse811 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse812 = 5.1199999999999999e-10*cse811;
  const amrex::Real cse813 = 3.5700000000000001e-10*cse808;
  const amrex::Real cse814 = cse813*state.xn[0];
  const amrex::Real cse815 = 0.025000000000000001*cse660;
  const amrex::Real cse816 = 3.7250000000000001e-11*std::pow(cse659, cse815 + 0.26400000000000001);
  const amrex::Real cse817 = 1.0274999999999999e-10*std::pow(cse659, cse787 + 0.29599999999999999);
  const amrex::Real cse818 = cse816 + cse817;
  const amrex::Real cse819 = cse812*state.xn[2] + cse814 + cse818*state.xn[3];
  const amrex::Real cse820 = cse795*cse819;
  const amrex::Real cse821 = 8.68e-11*cse784;
  const amrex::Real cse822 = cse793*cse805;
  const amrex::Real cse823 = cse819 + 8.9099999999999997e-5;
  const amrex::Real cse824 = cse795*cse823;
  const amrex::Real cse825 = 6.3800000000000002e-11*cse780;
  const amrex::Real cse826 = 2.1266666666666666e-11*cse780*cse794 + 2.8933333333333332e-11*cse784*cse822 + cse806*cse809 + cse809*cse810 + cse820*cse821 + cse824*cse825;
  const amrex::Real cse827 = cse792 + 1.7499999999999998e-5;
  const amrex::Real cse828 = cse805 + 1.34e-10;
  const amrex::Real cse829 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse830 = cse792*cse829;
  const amrex::Real cse831 = (1.0/3.0)*cse830;
  const amrex::Real cse832 = (1.0/5.0)*cse795;
  const amrex::Real cse833 = cse792*cse832;
  const amrex::Real cse834 = (1.0/15.0)*cse794;
  const amrex::Real cse835 = cse805*cse832;
  const amrex::Real cse836 = cse805*cse834 + cse819*cse833 + cse823*cse835;
  const amrex::Real cse837 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse838 = cse819*cse837;
  const amrex::Real cse839 = (3.0/5.0)*cse838;
  const amrex::Real cse840 = cse827*cse835 + cse827*cse839 + cse828*cse839;
  const amrex::Real cse841 = cse823*cse827 + cse823*cse828 + cse828*cse831 + cse836 + cse840;
  const amrex::Real cse842 = 1.0/cse841;
  const amrex::Real cse843 = 2.3872025442750958e-19*cse842;
  const amrex::Real cse844 = cse795*cse827;
  const amrex::Real cse845 = 2.1419999999999999e-10*cse808;
  const amrex::Real cse846 = cse828*cse837;
  const amrex::Real cse847 = cse827*cse837;
  const amrex::Real cse848 = 1.9139999999999998e-10*cse780*cse838 + 2.604e-10*cse784*cse838 + cse806*cse821 + cse825*cse844 + cse845*cse846 + cse845*cse847;
  const amrex::Real cse849 = 2.8023005140020001e-18*cse842;
  const amrex::Real cse850 = std::pow(cse841, -2);
  const amrex::Real cse851 = cse828*cse829;
  const amrex::Real cse852 = cse850*(-1.0633333333333333e-10*cse780*cse830 - 1.4466666666666666e-10*cse784*cse851 - cse785*cse823 - cse798*cse823 - cse813*cse827 - cse813*cse828 - cse826 - cse848);
  const amrex::Real cse853 = 2.3872025442750958e-19*cse836;
  const amrex::Real cse854 = 2.8023005140020001e-18*cse840;
  const amrex::Real cse855 = 0.045999999999999999*cse660;
  const amrex::Real cse856 = std::pow(cse659, cse855 + 0.22800000000000001);
  const amrex::Real cse857 = std::pow(cse659, cse855 + 0.23100000000000001);
  const amrex::Real cse858 = 2.6400000000000002e-10*cse857;
  const amrex::Real cse859 = cse858*state.xn[0];
  const amrex::Real cse860 = 0.031*cse660;
  const amrex::Real cse861 = 1.11e-10*std::pow(cse659, cse860 + 0.26300000000000001);
  const amrex::Real cse862 = 0.064000000000000001*cse660;
  const amrex::Real cse863 = 4.3749999999999998e-11*std::pow(cse659, cse862 + 0.071999999999999995);
  const amrex::Real cse864 = cse861 + cse863;
  const amrex::Real cse865 = 0.00235272*cse72;
  const amrex::Real cse866 = cse865 - 0.041816600000000002;
  const amrex::Real cse867 = cse72*cse866 + 0.35826400000000003;
  const amrex::Real cse868 = cse72*cse867 - 0.57443;
  const amrex::Real cse869 = 0.097857299999999994*cse72;
  const amrex::Real cse870 = cse869 - 3.1926800000000002;
  const amrex::Real cse871 = cse72*cse870 + 38.504899999999999;
  const amrex::Real cse872 = cse72*cse871 - 202.19300000000001;
  const amrex::Real cse873 = std::exp(((cse736) ? (
   cse72*cse868 - 7.4386999999999999
)
: (
   cse72*cse872 + 386.18599999999998
)));
  const amrex::Real cse874 = 1.7258000000000001e-6*cse658;
  const amrex::Real cse875 = cse873*cse874;
  const amrex::Real cse876 = cse859 + cse864*state.xn[3] + cse875;
  const amrex::Real cse877 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse878 = cse876*cse877;
  const amrex::Real cse879 = 8.9000000000000003e-11*cse856;
  const amrex::Real cse880 = cse879*state.xn[0];
  const amrex::Real cse881 = 0.037999999999999999*cse660;
  const amrex::Real cse882 = 5.1749999999999996e-11*std::pow(cse659, cse881 + 0.16900000000000001);
  const amrex::Real cse883 = 0.048000000000000001*cse660;
  const amrex::Real cse884 = 2.15e-11*std::pow(cse659, cse883 - 0.01);
  const amrex::Real cse885 = cse882 + cse884;
  const amrex::Real cse886 = 0.0070527699999999999*cse72;
  const amrex::Real cse887 = cse886 - 0.11133800000000001;
  const amrex::Real cse888 = cse72*cse887 + 0.69763799999999998;
  const amrex::Real cse889 = cse72*cse888 - 1.3074300000000001;
  const amrex::Real cse890 = 0.093813800000000003*cse72;
  const amrex::Real cse891 = cse890 - 3.0328300000000001;
  const amrex::Real cse892 = cse72*cse891 + 36.180300000000003;
  const amrex::Real cse893 = cse72*cse892 - 187.47399999999999;
  const amrex::Real cse894 = std::exp(((cse736) ? (
   cse72*cse889 - 7.6973500000000001
)
: (
   cse72*cse893 + 350.60899999999998
)));
  const amrex::Real cse895 = cse874*cse894;
  const amrex::Real cse896 = cse880 + cse885*state.xn[3] + cse895;
  const amrex::Real cse897 = cse877*cse896;
  const amrex::Real cse898 = 0.057000000000000002*cse660;
  const amrex::Real cse899 = std::pow(cse659, cse898 + 0.115);
  const amrex::Real cse900 = 1.26e-10*cse899;
  const amrex::Real cse901 = cse900*state.xn[0];
  const amrex::Real cse902 = 0.049000000000000002*cse660;
  const amrex::Real cse903 = 5.3249999999999997e-11*std::pow(cse659, cse902 - 0.0040000000000000001);
  const amrex::Real cse904 = 0.10199999999999999*cse660;
  const amrex::Real cse905 = 1.675e-11*std::pow(cse659, cse904 - 0.085000000000000006);
  const amrex::Real cse906 = cse903 + cse905;
  const amrex::Real cse907 = 0.00065632500000000003*cse72;
  const amrex::Real cse908 = -cse907 - 0.015089200000000001;
  const amrex::Real cse909 = cse72*cse908 + 0.36118400000000001;
  const amrex::Real cse910 = cse72*cse909 - 0.77378199999999997;
  const amrex::Real cse911 = 0.10508000000000001*cse72;
  const amrex::Real cse912 = cse911 - 3.4762;
  const amrex::Real cse913 = cse72*cse912 + 42.594999999999999;
  const amrex::Real cse914 = cse72*cse913 - 227.91300000000001;
  const amrex::Real cse915 = std::exp(((cse736) ? (
   cse72*cse910 - 9.2514099999999999
)
: (
   cse72*cse914 + 444.60000000000002
)));
  const amrex::Real cse916 = 2.8763333333333333e-6*cse915;
  const amrex::Real cse917 = cse658*cse916;
  const amrex::Real cse918 = cse901 + cse906*state.xn[3] + cse917;
  const amrex::Real cse919 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse920 = cse918*cse919;
  const amrex::Real cse921 = 1.32e-9*cse857;
  const amrex::Real cse922 = 6.3e-10*cse899;
  const amrex::Real cse923 = cse896*cse919;
  const amrex::Real cse924 = cse876*cse919;
  const amrex::Real cse925 = cse918 + 7.8800000000000004e-8;
  const amrex::Real cse926 = cse919*cse925;
  const amrex::Real cse927 = 4.4500000000000001e-10*cse856;
  const amrex::Real cse928 = 7.4166666666666679e-10*cse856*cse878 + 2.2000000000000003e-9*cse857*cse897 + cse920*cse921 + cse922*cse923 + cse922*cse924 + cse926*cse927;
  const amrex::Real cse929 = cse876 + 2.65e-7;
  const amrex::Real cse930 = cse896 + 1.81e-14;
  const amrex::Real cse931 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse932 = cse876*cse931;
  const amrex::Real cse933 = (5.0/3.0)*cse930;
  const amrex::Real cse934 = (25.0/3.0)*cse897;
  const amrex::Real cse935 = 5*cse919;
  const amrex::Real cse936 = cse876*cse935;
  const amrex::Real cse937 = cse896*cse935;
  const amrex::Real cse938 = cse876*cse934 + cse918*cse936 + cse925*cse937;
  const amrex::Real cse939 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse940 = cse918*cse939;
  const amrex::Real cse941 = 3*cse940;
  const amrex::Real cse942 = cse929*cse937 + cse929*cse941 + cse930*cse941;
  const amrex::Real cse943 = cse925*cse929 + cse925*cse930 + cse932*cse933 + cse938 + cse942;
  const amrex::Real cse944 = 1.0/cse943;
  const amrex::Real cse945 = 1.4210469458259591e-21*cse944;
  const amrex::Real cse946 = cse919*cse929;
  const amrex::Real cse947 = 3.7800000000000004e-10*cse899;
  const amrex::Real cse948 = cse930*cse939;
  const amrex::Real cse949 = cse929*cse939;
  const amrex::Real cse950 = 2.6700000000000001e-10*cse856*cse940 + 7.9200000000000006e-10*cse857*cse940 + cse921*cse923 + cse927*cse946 + cse947*cse948 + cse947*cse949;
  const amrex::Real cse951 = 2.5697412351439998e-22*cse944;
  const amrex::Real cse952 = std::pow(cse943, -2);
  const amrex::Real cse953 = cse930*cse931;
  const amrex::Real cse954 = cse952*(-1.4833333333333334e-10*cse856*cse932 - 4.4000000000000003e-10*cse857*cse953 - cse858*cse925 - cse879*cse925 - cse900*cse929 - cse900*cse930 - cse928 - cse950);
  const amrex::Real cse955 = 1.4210469458259591e-21*cse938;
  const amrex::Real cse956 = 2.5697412351439998e-22*cse942;
  const amrex::Real cse957 = state.T < 1000000.0;
  const amrex::Real cse958 = cse572*cse69*d2g;
  const amrex::Real cse959 = crate*cse689;
  const amrex::Real cse960 = 1.7623947000000002e-12*cse695*cse959*state.xn[5];
  const amrex::Real cse961 = 3.8323655572603624e-29*cse441;
  const amrex::Real cse962 = cse961*state.xn[3];
  const amrex::Real cse963 = cse696*(cse35*cse623*cse624 - cse700*cse702);
  const amrex::Real cse964 = 2.0187430200000004e-26*cse165;
  const amrex::Real cse965 = cse10*cse964;
  const amrex::Real cse966 = cse11*cse716*state.xn[3] + 26.399999999999999*cse66*cse676*cse678*cse713*state.xn[0] - cse708 - cse717;
  const amrex::Real cse967 = 1.602177e-12*cse89;
  const amrex::Real cse968 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse969 = 1.7623947000000002e-12*cse719;
  const amrex::Real cse970 = crate*cse969*state.xn[5];
  const amrex::Real cse971 = -7.1777529600000012e-12*cse120 + 3.0000000000000002e-15*cse165*cse628 - 7.1777529600000012e-12*cse499 + 1.7e-24*cse568*cse958 - cse585 - cse608 - cse611 + cse86*cse960 + cse962*cse963 + cse963*cse965 + cse966*cse967 + cse966*cse968 + cse966*cse970;
  const amrex::Real cse972 = cse586*cse588;
  const amrex::Real cse973 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse974 = std::pow(cse587, -2);
  const amrex::Real cse975 = std::pow(cse591, -2);
  const amrex::Real cse976 = cse110*cse583;
  const amrex::Real cse977 = cse580*cse582;
  const amrex::Real cse978 = 1.0/cse577;
  const amrex::Real cse979 = std::pow(cse573, -2);
  const amrex::Real cse980 = cse978*cse979;
  const amrex::Real cse981 = cse72/std::pow(cse578, 2);
  const amrex::Real cse982 = std::pow(cse571, -2);
  const amrex::Real cse983 = std::pow(cse569, 0.12299999999999994)*cse69;
  const amrex::Real cse984 = std::pow(cse569, -0.56899999999999995)*cse570*cse69;
  const amrex::Real cse985 = 1.3659996292638161e-10*cse292*cse972;
  const amrex::Real cse986 = cse594*cse985/std::pow(cse596, 2);
  const amrex::Real cse987 = 0.5*cse11*cse677 - cse709*cse711;
  const amrex::Real cse988 = cse976*state.xn[2];
  const amrex::Real cse989 = 9.0599999999999997e-6*cse260;
  const amrex::Real cse990 = 1.024e-10*cse811;
  const amrex::Real cse991 = 2.1600000000000004e-15*cse781;
  const amrex::Real cse992 = 9.7199999999999998e-11*cse796;
  const amrex::Real cse993 = 7.1999999999999997e-16*cse781*cse822 + 3.2399999999999999e-11*cse794*cse796 + cse806*cse990 + cse810*cse990 + cse820*cse991 + cse824*cse992;
  const amrex::Real cse994 = 3.0719999999999998e-10*cse811;
  const amrex::Real cse995 = 6.4799999999999996e-15*cse781*cse838 + 2.9159999999999995e-10*cse796*cse838 + cse806*cse991 + cse844*cse992 + cse846*cse994 + cse847*cse994;
  const amrex::Real cse996 = cse850*(-3.6000000000000001e-15*cse781*cse851 - cse782*cse823 - 1.6199999999999998e-10*cse796*cse830 - cse797*cse823 - cse812*cse827 - cse812*cse828 - cse993 - cse995);
  const amrex::Real cse997 = cse260*cse873;
  const amrex::Real cse998 = cse260*cse915;
  const amrex::Real cse999 = 1.4381666666666666e-5*cse998;
  const amrex::Real cse1000 = cse260*cse894;
  const amrex::Real cse1001 = 8.6289999999999999e-6*cse997;
  const amrex::Real cse1002 = 8.6289999999999999e-6*cse1000;
  const amrex::Real cse1003 = 1.4381666666666668e-5*cse1000*cse878 + cse1001*cse920 + cse1002*cse926 + 1.4381666666666668e-5*cse897*cse997 + cse923*cse999 + cse924*cse999;
  const amrex::Real cse1004 = 5.1774000000000006e-6*cse940;
  const amrex::Real cse1005 = 8.6289999999999999e-6*cse998;
  const amrex::Real cse1006 = cse1000*cse1004 + cse1001*cse923 + cse1002*cse946 + cse1004*cse997 + cse1005*cse948 + cse1005*cse949;
  const amrex::Real cse1007 = cse260*cse916;
  const amrex::Real cse1008 = 1.7258000000000001e-6*cse925;
  const amrex::Real cse1009 = cse952*(-cse1000*cse1008 - 2.8763333333333337e-6*cse1000*cse932 - cse1003 - cse1006 - cse1007*cse929 - cse1007*cse930 - cse1008*cse997 - 2.8763333333333337e-6*cse953*cse997);
  const amrex::Real cse1010 = cse584*cse977;
  const amrex::Real cse1011 = 3.7996253111508318e-29*cse1010;
  const amrex::Real cse1012 = cse2*cse607;
  const amrex::Real cse1013 = 6.4000000000000004e-32*cse1012;
  const amrex::Real cse1014 = cse610*d2g;
  const amrex::Real cse1015 = 200.0*cse1014;
  const amrex::Real cse1016 = 2.5634832e-11*cse627 + 6.4087080000000003e-13;
  const amrex::Real cse1017 = 5.7927337833400002e-20*cse673;
  const amrex::Real cse1018 = cse1017*cse674;
  const amrex::Real cse1019 = -cse666;
  const amrex::Real cse1020 = cse1017*cse705*state.xn[8];
  const amrex::Real cse1021 = cse171*cse616 + 2*cse699;
  const amrex::Real cse1022 = 2*cse697 - cse702*(cse1021 - cse617);
  const amrex::Real cse1023 = 2*cse705;
  const amrex::Real cse1024 = 2*cse708;
  const amrex::Real cse1025 = cse677*cse709;
  const amrex::Real cse1026 = cse710*cse712;
  const amrex::Real cse1027 = -1.0*cse1025 + 1.0*cse1026;
  const amrex::Real cse1028 = ((cse683) ? (
   0
)
: ((cse684) ? (
   3*cse715
)
: ((cse685) ? (
   (8.0/3.0)*cse715
)
: ((cse686) ? (
   (2.0/3.0)*cse715
)
: (
   0
)))));
  const amrex::Real cse1029 = cse171*cse687;
  const amrex::Real cse1030 = -cse1024 + cse1027*cse714 + cse1028*cse443 - cse1029 + cse688;
  const amrex::Real cse1031 = 1.0/cse776;
  const amrex::Real cse1032 = cse819*cse832;
  const amrex::Real cse1033 = (1.0/15.0)*cse822;
  const amrex::Real cse1034 = cse804*cse823;
  const amrex::Real cse1035 = cse1032*cse791 + cse1033*cse791 + cse1034*cse832 + cse804*cse834 + cse818*cse833 + cse818*cse835;
  const amrex::Real cse1036 = cse818*cse828;
  const amrex::Real cse1037 = (3.0/5.0)*cse837;
  const amrex::Real cse1038 = cse818*cse827;
  const amrex::Real cse1039 = cse827*cse832;
  const amrex::Real cse1040 = cse1036*cse1037 + cse1037*cse1038 + cse1039*cse804 + cse791*cse835 + cse791*cse839 + cse804*cse839;
  const amrex::Real cse1041 = (1.0/3.0)*cse851;
  const amrex::Real cse1042 = cse850*(-cse1034 - cse1035 - cse1036 - cse1038 - cse1040 - cse1041*cse791 - cse791*cse823 - cse804*cse831);
  const amrex::Real cse1043 = cse918*cse935;
  const amrex::Real cse1044 = (25.0/3.0)*cse878;
  const amrex::Real cse1045 = cse885*cse925;
  const amrex::Real cse1046 = cse1043*cse864 + cse1044*cse885 + cse1045*cse935 + cse864*cse934 + cse906*cse936 + cse906*cse937;
  const amrex::Real cse1047 = cse929*cse935;
  const amrex::Real cse1048 = cse906*cse930;
  const amrex::Real cse1049 = 3*cse939;
  const amrex::Real cse1050 = cse906*cse929;
  const amrex::Real cse1051 = cse1047*cse885 + cse1048*cse1049 + cse1049*cse1050 + cse864*cse937 + cse864*cse941 + cse885*cse941;
  const amrex::Real cse1052 = cse931*cse933;
  const amrex::Real cse1053 = (5.0/3.0)*cse932;
  const amrex::Real cse1054 = cse952*(-cse1045 - cse1046 - cse1048 - cse1050 - cse1051 - cse1052*cse864 - cse1053*cse885 - cse864*cse925);
  const amrex::Real cse1055 = cse568*cse958;
  const amrex::Real cse1056 = cse696*(-cse1021*cse702 + 2*cse35*cse623*cse624);
  const amrex::Real cse1057 = -cse1024 + 26.399999999999999*cse1027*cse66*cse676*cse678*state.xn[0] + cse1028*cse11*state.xn[3] - cse1029;
  const amrex::Real cse1058 = -state.xn[3]*((cse740) ? (
   0
)
: (
   cse756*cse778
));
  const amrex::Real cse1059 = std::pow(cse643, -2);
  const amrex::Real cse1060 = llte_co_interp2d_partial_1(state.T, cse642);
  const amrex::Real cse1061 = nhalf_co_interp2d_partial_1(state.T, cse642);
  const amrex::Real cse1062 = cse1061*cse641;
  const amrex::Real cse1063 = alpha_co_interp2d_partial_1(state.T, cse642);
  const amrex::Real cse1064 = std::log(cse649);
  const amrex::Real cse1065 = cse696*(3*cse35*cse623*cse624 - cse702*(3*cse698 + 3*cse699));
  const amrex::Real cse1066 = cse11*state.xn[3]*((cse683) ? (
   0
)
: ((cse684) ? (
   (9.0/2.0)*cse715
)
: ((cse685) ? (
   4*cse715
)
: ((cse686) ? (
   cse715
)
: (
   0
))))) + 26.399999999999999*cse66*cse676*cse678*state.xn[0]*(-1.5*cse1025 + 1.5*cse1026) - 3*cse708 - 3*cse717;
  const amrex::Real cse1067 = cse594*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1068 = cse292*cse588;
  const amrex::Real cse1069 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1070 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1071 = cse1069*cse589;
  const amrex::Real cse1072 = cse593*cse975*(2.5500935934394033e-12*cse1070*cse974*state.xn[2] - 6.6690584756579352e-10*cse1071);
  const amrex::Real cse1073 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1074 = cse273*cse574*cse69*cse978;
  const amrex::Real cse1075 = cse256*cse706;
  const amrex::Real cse1076 = cse3*cse658;
  const amrex::Real cse1077 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1078 = 100*cse3;
  const amrex::Real cse1079 = -2.2649999999999999e-6*cse1076 + cse663*(cse1078*(8.7000000000000001e-5*cse660 + 0.001281) + cse3*cse661) + state.xn[3]*(0.75*((cse664) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1077
)) + 0.25*((cse664) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1077
)));
  const amrex::Real cse1080 = std::pow(cse613, -3);
  const amrex::Real cse1081 = cse696*cse701*(-cse618*cse7*(0.0040460000000000001*cse1080 + 0.0036709999999999998*cse614) - cse621*cse7*(0.0047800000000000004*cse1080 - 5.5500000000000001e-5*cse614));
  const amrex::Real cse1082 = -cse1079;
  const amrex::Real cse1083 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1084 = std::pow(cse634, -2);
  const amrex::Real cse1085 = std::pow(cse639, -2);
  const amrex::Real cse1086 = (((5.7445531974402559e-18*cse2*cse638 - gradv < 0) ? (
   0
)
: ((5.7445531974402559e-18*cse2*cse638 - gradv == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1087 = cse1059*(2.872276598720128e-13*cse1060*cse1085*cse1086*cse273*cse638*state.xn[9] - llte_co_interp2d_partial_0(state.T, cse642));
  const amrex::Real cse1088 = 2.872276598720128e-13*cse1085*cse1086*cse273*cse638*state.xn[9];
  const amrex::Real cse1089 = -cse1061*cse1088 + nhalf_co_interp2d_partial_0(state.T, cse642);
  const amrex::Real cse1090 = ((cse721) ? (
   -cse256
)
: (
   0
));
  const amrex::Real cse1091 = -1.8719999999999998e-14*cse1090*cse767 - 3.9261999999999998e-15*cse1090*cse768 - 1.53e-21*cse1090*cse769 + 4.985670872372847e-33*cse770*cse773*((cse721) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse771*cse773*((cse721) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse771*((cse721) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse772, 2);
  const amrex::Real cse1092 = ((cse721) ? (
   1
)
: (
   0
));
  const amrex::Real cse1093 = cse1092*cse766;
  const amrex::Real cse1094 = cse1093*cse741;
  const amrex::Real cse1095 = cse1093*cse743;
  const amrex::Real cse1096 = cse1093*cse744;
  const amrex::Real cse1097 = cse1093*cse742;
  const amrex::Real cse1098 = cse1093*cse745;
  const amrex::Real cse1099 = cse1093*cse758;
  const amrex::Real cse1100 = cse1093*cse759;
  const amrex::Real cse1101 = 65.320000000000007*cse256;
  const amrex::Real cse1102 = cse3*cse787;
  const amrex::Real cse1103 = 0.00042999999999999999*cse660;
  const amrex::Real cse1104 = cse814*(cse1078*(0.0041900000000000001 - 3.0000000000000001e-5*cse660) - cse3*cse807) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse816*(cse1078*(0.00025000000000000001*cse660 + 0.00264) + cse3*cse815) + cse817*(cse1078*(cse1103 + 0.00296) + cse1102));
  const amrex::Real cse1105 = cse799*(cse1078*(0.0036900000000000001 - 6.0000000000000002e-5*cse660) - cse3*cse779) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse801*(cse1078*(0.00041000000000000005*cse660 + 0.0020300000000000001) + cse3*cse800) + cse803*(cse1078*(0.00058*cse660 + 0.0023700000000000001) + cse3*cse802));
  const amrex::Real cse1106 = cse786*(cse1078*(0.0075500000000000003 - 0.0016000000000000001*cse660) - cse3*cse783) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse788*(cse1078*(cse1103 + 0.0088900000000000003) + cse1102) + cse790*(cse1078*(0.0052500000000000003*cse660 + 0.011979999999999999) + cse3*cse789));
  const amrex::Real cse1107 = cse1105*cse823;
  const amrex::Real cse1108 = cse1032*cse1106 + cse1033*cse1106 + cse1101*cse792*cse820 + cse1101*cse805*cse824 + cse1104*cse833 + cse1104*cse835 + cse1105*cse834 + cse1107*cse832 + 28.360000000000003*cse256*cse794*cse805;
  const amrex::Real cse1109 = 136.68000000000001*cse256*cse838;
  const amrex::Real cse1110 = cse1104*cse828;
  const amrex::Real cse1111 = cse1104*cse827;
  const amrex::Real cse1112 = cse1037*cse1110 + cse1037*cse1111 + cse1039*cse1105 + cse1101*cse806*cse827 + cse1105*cse839 + cse1106*cse835 + cse1106*cse839 + cse1109*cse827 + cse1109*cse828;
  const amrex::Real cse1113 = cse850*(-cse1041*cse1106 - cse1105*cse831 - cse1106*cse823 - cse1107 - cse1108 - cse1110 - cse1111 - cse1112 - 32.933333333333337*cse256*cse828*cse830);
  const amrex::Real cse1114 = 312.30000000000001*cse256;
  const amrex::Real cse1115 = cse3*cse855;
  const amrex::Real cse1116 = 0.00046000000000000001*cse660;
  const amrex::Real cse1117 = 8.6290000000000003e-7*cse1076;
  const amrex::Real cse1118 = -cse1117*cse873 + cse859*(cse1078*(cse1116 + 0.00231) + cse1115) + cse875*((cse736) ? (
   cse3*cse868 + cse72*(cse3*cse867 + cse72*(cse3*cse865 + cse3*cse866))
)
: (
   cse3*cse872 + cse72*(cse3*cse871 + cse72*(cse3*cse869 + cse3*cse870))
)) + state.xn[3]*(cse861*(cse1078*(0.00031*cse660 + 0.00263) + cse3*cse860) + cse863*(cse1078*(0.00064000000000000005*cse660 + 0.00071999999999999994) + cse3*cse862));
  const amrex::Real cse1119 = -1.4381666666666666e-6*cse1076*cse915 + cse901*(cse1078*(0.00057000000000000009*cse660 + 0.00115) + cse3*cse898) + cse917*((cse736) ? (
   cse3*cse910 + cse72*(cse3*cse909 + cse72*(-cse3*cse907 + cse3*cse908))
)
: (
   cse3*cse914 + cse72*(cse3*cse913 + cse72*(cse3*cse911 + cse3*cse912))
)) + state.xn[3]*(cse903*(cse1078*(0.00048999999999999998*cse660 - 4.0000000000000003e-5) + cse3*cse902) + cse905*(cse1078*(0.0010199999999999999*cse660 - 0.00085000000000000006) + cse3*cse904));
  const amrex::Real cse1120 = -cse1117*cse894 + cse880*(cse1078*(cse1116 + 0.0022800000000000003) + cse1115) + cse895*((cse736) ? (
   cse3*cse889 + cse72*(cse3*cse888 + cse72*(cse3*cse886 + cse3*cse887))
)
: (
   cse3*cse893 + cse72*(cse3*cse892 + cse72*(cse3*cse890 + cse3*cse891))
)) + state.xn[3]*(cse882*(cse1078*(0.00038000000000000002*cse660 + 0.0016900000000000001) + cse3*cse881) + cse884*(cse1078*(0.00048000000000000001*cse660 - 0.0001) + cse3*cse883));
  const amrex::Real cse1121 = cse1120*cse925;
  const amrex::Real cse1122 = cse1043*cse1118 + cse1044*cse1120 + cse1114*cse876*cse920 + cse1114*cse896*cse926 + cse1118*cse934 + cse1119*cse936 + cse1119*cse937 + cse1121*cse935 + 844.16666666666686*cse256*cse876*cse897;
  const amrex::Real cse1123 = 70.859999999999985*cse256*cse940;
  const amrex::Real cse1124 = cse1119*cse930;
  const amrex::Real cse1125 = cse1119*cse929;
  const amrex::Real cse1126 = cse1047*cse1120 + cse1049*cse1124 + cse1049*cse1125 + cse1114*cse923*cse929 + cse1118*cse937 + cse1118*cse941 + cse1120*cse941 + cse1123*cse929 + cse1123*cse930;
  const amrex::Real cse1127 = cse952*(-cse1052*cse1118 - cse1053*cse1120 - cse1118*cse925 - cse1121 - cse1122 - cse1124 - cse1125 - cse1126 - 64.733333333333348*cse256*cse930*cse932);
  const amrex::Real cse1128 = -cse491 - 5.4061971382371991e-18*cse523 - cse549;

  jac(1, 1) = -cse1 - cse10*cse56 - cse102 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3];
  jac(1, 2) = cse115 + cse124;
  jac(1, 3) = cse134 + cse136*state.xn[16] + cse140*cse141 + cse148 + cse156;
  jac(1, 4) = cse123*cse183 - cse141*cse187 + cse159 + cse164 + cse166 + 2*cse167 + 2*cse176 + 4*cse178 + cse179*state.xn[17] + cse180*state.rn[0] + cse186*state.xn[8] + cse191 + cse200;
  jac(1, 5) = cse123*cse205 + cse166 + 2*cse204 + cse208 + cse211;
  jac(1, 7) = cse216;
  jac(1, 8) = cse220;
  jac(1, 9) = cse186*state.xn[3];
  jac(1, 10) = -cse221;
  jac(1, 11) = cse124 + cse223;
  jac(1, 12) = cse226 + cse227 + cse230;
  jac(1, 15) = cse231 + cse232 + cse234 + cse235;
  jac(1, 16) = cse235 + cse239;
  jac(1, 17) = cse123*cse245 + cse136*state.xn[2] - 1.8000000000000002e-14*cse165 + 2*cse244 + cse252;
  jac(1, 18) = cse179*state.xn[3] - cse253 + cse255;
  jac(1, 19) = cse308*(-cse145*cse287 + 9.8669999999999991e-12*cse184*cse284*state.xn[3]*state.xn[8] - cse258 - cse259 - cse262 + 2*cse271*cse30 - cse272 + 2*cse275*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse276 + cse277 - 5.5769999999999994e-13*cse280 - cse282 + cse283 + cse285 - cse291 - cse307);
  jac(1, 20) = cse180*state.xn[3];
  jac(2, 1) = cse102;
  jac(2, 2) = -cse115;
  jac(2, 3) = -cse155;
  jac(2, 4) = cse159 + cse197 + cse309;
  jac(2, 5) = cse207 + cse210 + cse309;
  jac(2, 7) = cse212;
  jac(2, 11) = cse310;
  jac(2, 12) = -cse229;
  jac(2, 15) = cse310;
  jac(2, 16) = cse310;
  jac(2, 17) = cse250 - cse251;
  jac(2, 18) = cse253;
  jac(2, 19) = cse308*(-cse258 + cse293 + cse295 - cse296 - cse299 - cse311);
  jac(3, 1) = cse101 + cse313 - cse353 + cse65 - cse84 + cse90;
  jac(3, 2) = -cse114 - cse355;
  jac(3, 3) = -cse156 - cse365 - cse382 - cse387 - cse388 - cse403;
  jac(3, 4) = -cse200 - cse405 + cse406 + cse407 - cse412 - cse419 - cse424 - cse425;
  jac(3, 5) = -cse211 - cse405 - cse411 - cse418 - cse427;
  jac(3, 6) = cse428;
  jac(3, 7) = -cse430;
  jac(3, 8) = cse440 + cse445;
  jac(3, 9) = -cse448;
  jac(3, 10) = cse221;
  jac(3, 11) = -cse223 - cse355;
  jac(3, 12) = cse453;
  jac(3, 13) = cse455;
  jac(3, 14) = cse457;
  jac(3, 15) = cse458;
  jac(3, 16) = cse458;
  jac(3, 17) = -cse252 - cse459 - cse460 - cse461 - cse463 - cse464 - cse465 - cse466;
  jac(3, 18) = cse468;
  jac(3, 19) = cse308*(4.2899999999999994e-13*cse280 + cse288 + cse302 + cse307 + cse470 + cse471 + cse472 + cse475 + cse476 - cse478 + cse479 + cse480 - cse482 - cse483 - cse484 - cse485 - cse486);
  jac(3, 20) = cse491;
  jac(4, 1) = 3.0000000000000002e-15*cse10*d2g - cse492 - cse493 - cse494 - cse495 - cse497 - cse498;
  jac(4, 2) = cse501;
  jac(4, 3) = cse144 - cse146 - cse147 + cse402 + cse502*state.xn[16];
  jac(4, 4) = -cse121 - cse167 - cse176 - cse177*cse9 - cse414 - 1.4949836836302895e-18*cse441 - cse503 - cse505 - cse506 - cse507 - cse508;
  jac(4, 5) = -cse121 - cse204 - cse208 - cse426 - cse509;
  jac(4, 7) = -cse212 - cse510;
  jac(4, 8) = cse439;
  jac(4, 9) = -cse446;
  jac(4, 10) = cse511;
  jac(4, 11) = cse501;
  jac(4, 12) = cse452;
  jac(4, 15) = -cse234 - cse500;
  jac(4, 16) = cse501;
  jac(4, 17) = -cse244 + cse463 + cse502*state.xn[2] + cse512 - cse513 + cse515;
  jac(4, 18) = cse468;
  jac(4, 19) = cse308*(-cse257 - 2.3622303960273254e-7*cse276 + cse302 - cse516 - cse517 - cse518 - cse519 - cse520 - cse522);
  jac(4, 20) = -cse524;
  jac(5, 1) = cse498;
  jac(5, 2) = cse354;
  jac(5, 4) = -cse164 + cse504 + cse506;
  jac(5, 5) = -cse206 - cse207 - cse427;
  jac(5, 7) = cse510;
  jac(5, 11) = cse354;
  jac(5, 15) = cse354;
  jac(5, 16) = cse354;
  jac(5, 17) = cse462;
  jac(5, 19) = -cse308*cse521;
  jac(6, 1) = cse325;
  jac(6, 2) = cse325;
  jac(6, 3) = cse365;
  jac(6, 4) = cse525;
  jac(6, 5) = cse405;
  jac(6, 6) = -cse428;
  jac(6, 7) = cse527;
  jac(6, 10) = cse528;
  jac(6, 11) = cse325;
  jac(6, 15) = cse325;
  jac(6, 16) = cse529;
  jac(6, 17) = cse464;
  jac(6, 19) = cse308*cse531;
  jac(7, 1) = cse532;
  jac(7, 2) = cse532;
  jac(7, 3) = -cse365;
  jac(7, 4) = -cse525;
  jac(7, 5) = -cse405;
  jac(7, 6) = cse428;
  jac(7, 7) = -cse527;
  jac(7, 10) = -cse528;
  jac(7, 11) = cse532;
  jac(7, 15) = cse532;
  jac(7, 16) = -cse529;
  jac(7, 17) = -cse464;
  jac(7, 19) = -cse308*cse531;
  jac(8, 1) = cse1 + cse536;
  jac(8, 2) = cse536;
  jac(8, 3) = cse382;
  jac(8, 4) = cse415 + cse537 + cse538*state.xn[8] + cse542;
  jac(8, 5) = cse542;
  jac(8, 8) = -cse220 - cse439 - cse445;
  jac(8, 9) = cse447 + cse538*state.xn[3];
  jac(8, 10) = cse544;
  jac(8, 11) = cse536;
  jac(8, 15) = cse536 + cse545;
  jac(8, 16) = -cse237 + cse536;
  jac(8, 17) = -cse460 + cse465 - cse546;
  jac(8, 19) = cse308*(cse259 - 1.287e-13*cse280 + cse290 + cse470 + 2.2769999999999997e-12*cse481 + cse547);
  jac(8, 20) = -cse490 - cse550;
  jac(9, 1) = cse551;
  jac(9, 2) = cse551;
  jac(9, 3) = -cse382;
  jac(9, 4) = -cse419;
  jac(9, 5) = -cse418;
  jac(9, 7) = cse526;
  jac(9, 8) = cse445;
  jac(9, 9) = -cse448 - cse553;
  jac(9, 10) = cse528;
  jac(9, 11) = cse551;
  jac(9, 15) = cse551;
  jac(9, 16) = -cse339 - cse554;
  jac(9, 17) = -cse465;
  jac(9, 19) = cse308*(-cse519 - cse547 - cse556);
  jac(9, 20) = cse490;
  jac(10, 1) = cse534;
  jac(10, 2) = cse534;
  jac(10, 3) = cse150;
  jac(10, 4) = cse557;
  jac(10, 5) = cse540;
  jac(10, 7) = -cse526;
  jac(10, 8) = cse220;
  jac(10, 10) = -cse558 - cse559;
  jac(10, 11) = cse222 + cse534;
  jac(10, 12) = cse228;
  jac(10, 15) = cse560;
  jac(10, 16) = cse237 + cse534;
  jac(10, 17) = -cse514 + cse546;
  jac(10, 19) = -cse291*cse308;
  jac(10, 20) = cse550;
  jac(11, 3) = -cse150;
  jac(11, 9) = cse553;
  jac(11, 10) = cse558;
  jac(11, 11) = -cse222;
  jac(11, 16) = cse554;
  jac(11, 17) = cse514;
  jac(11, 19) = cse308*(cse288 + cse556);
  jac(12, 1) = -cse534 - cse561;
  jac(12, 2) = -cse105 - cse534;
  jac(12, 3) = cse146 - cse398 + cse400;
  jac(12, 4) = cse189 - cse190 - cse423 - cse557;
  jac(12, 5) = cse541;
  jac(12, 7) = cse526;
  jac(12, 10) = cse559;
  jac(12, 11) = cse535;
  jac(12, 12) = -cse230 - cse562;
  jac(12, 15) = -cse560;
  jac(12, 16) = cse238 + cse535;
  jac(12, 17) = -cse459 - cse546;
  jac(12, 18) = cse253 + cse254;
  jac(12, 19) = cse308*(-cse563 - cse564);
  jac(12, 20) = cse549;
  jac(13, 1) = cse352;
  jac(13, 2) = cse352;
  jac(13, 3) = cse387;
  jac(13, 4) = cse412;
  jac(13, 5) = cse411;
  jac(13, 11) = cse352;
  jac(13, 13) = -cse455;
  jac(13, 14) = cse456;
  jac(13, 15) = cse352;
  jac(13, 16) = cse352;
  jac(13, 17) = cse466;
  jac(13, 19) = cse308*cse565;
  jac(13, 20) = -cse488;
  jac(14, 1) = cse566;
  jac(14, 2) = cse566;
  jac(14, 3) = -cse387;
  jac(14, 4) = -cse412;
  jac(14, 5) = -cse411;
  jac(14, 11) = cse566;
  jac(14, 13) = cse455;
  jac(14, 14) = cse457;
  jac(14, 15) = cse566;
  jac(14, 16) = cse566;
  jac(14, 17) = -cse466;
  jac(14, 19) = -cse308*cse565;
  jac(14, 20) = cse488;
  jac(15, 1) = cse495;
  jac(15, 4) = cse567*state.xn[8];
  jac(15, 8) = cse439;
  jac(15, 9) = cse567*state.xn[3];
  jac(15, 12) = -cse228;
  jac(15, 15) = -cse231 - cse545;
  jac(15, 17) = cse460;
  jac(15, 19) = cse308*(-3.0029999999999999e-13*cse280 + cse469 + 5.3129999999999997e-12*cse481 + cse516);
  jac(16, 3) = cse148 + cse401;
  jac(16, 4) = cse424;
  jac(16, 7) = -cse215;
  jac(16, 8) = -cse220;
  jac(16, 9) = -cse553;
  jac(16, 12) = cse562;
  jac(16, 16) = -cse239 - cse554;
  jac(16, 17) = cse459;
  jac(16, 18) = cse255;
  jac(16, 19) = cse308*(cse262 + cse290 + cse520 + cse555 + cse563);
  jac(17, 3) = -cse403;
  jac(17, 4) = cse507;
  jac(17, 5) = cse206;
  jac(17, 8) = cse440;
  jac(17, 10) = -cse511;
  jac(17, 12) = cse453;
  jac(17, 17) = -cse461 - cse515;
  jac(17, 19) = cse308*(cse471 + cse522);
  jac(18, 1) = cse561;
  jac(18, 2) = cse105;
  jac(18, 3) = -cse388;
  jac(18, 4) = -cse508;
  jac(18, 7) = cse215;
  jac(18, 12) = cse229;
  jac(18, 16) = cse236;
  jac(18, 18) = -cse253 - cse467;
  jac(18, 19) = cse308*(cse304 + cse530 + cse564);
  jac(19, 1) = 3.8323655572603624e-29*state.c_hat*cse696*cse703*state.rn[0]*state.xn[3] + 1.7e-24*chi*cse568*cse572*cse68*d2g + 1.7623947000000002e-12*crate*cse689*cse695*cse87*state.xn[5] + 1.602177e-12*crate*cse689 + 1.7623947000000002e-12*crate*cse718*cse719*state.xn[5] + 1.602177e-12*crate*cse718*state.xn[0] + 3.2043540000000001e-12*crate*cse718*state.xn[3] + 3.0000000000000002e-15*cse10*cse628*d2g + 2.0187430200000004e-26*cse10*cse696*cse703*d2g*state.xn[0] - 2.1789607199999999e-11*cse101 - 7.1777529600000012e-12*cse493 - 7.1777529600000012e-12*cse494 - 7.1777529600000012e-12*cse497 - cse585 - cse590*cse599 - cse608 - cse611 + 3.0000000000000002e-15*cse628*d2g*state.xn[0] - cse630*cse657 + 4.39089220777172e-29*cse662*cse667*cse669*cse673*cse690*state.xn[8] - 4.39089220777172e-29*cse662*cse675 - cse694*(-cse630*cse644 - cse630*cse692) - cse707*(-cse670*cse704 + cse704*cse705) - state.xn[11]*(cse826*cse843 + cse848*cse849 + cse852*cse853 + cse852*cse854) - state.xn[3]*((cse740) ? (
   0
)
: (
   cse751*cse778
)) - state.xn[7]*((cse957) ? (
   cse928*cse945 + cse950*cse951 + cse954*cse955 + cse954*cse956
)
: (
   0
));
  jac(19, 2) = cse971 - state.xn[3]*((cse740) ? (
   0
)
: (
   cse754*cse778
));
  jac(19, 3) = 1.7e-24*chi*cse10*cse568*cse68*cse982*d2g*(0.00025272365919999998*cse137*cse983 + 0.036605692000000002*cse137*cse984) + 4.6527220080000002e-11*crate*cse66*cse676*cse678*cse719*cse987*state.xn[0]*state.xn[5] + 4.22974728e-11*crate*cse66*cse676*cse678*cse987*std::pow(state.xn[0], 2) + 8.4594945600000001e-11*crate*cse66*cse676*cse678*cse987*state.xn[0]*state.xn[3] - 7.2151085033443149e-29*cse110*cse575*cse977*cse980*cse981*state.xn[2] - 2.1789607199999999e-11*cse151 + 2.6241084038530201e-25*cse260*cse667*cse669*cse673*cse690*state.xn[8] - 2.6241084038530201e-25*cse260*cse675 + 6.0732876035249664e-19*cse593*cse597*cse973*cse974*cse975*state.xn[0]*state.xn[2] - cse599*cse972*state.xn[0] - cse636*cse657 - cse694*(-cse636*cse644 - cse636*cse692) - cse707*(-cse670*cse989 + cse705*cse989) - cse976 - cse986*(5.9301787935659931e-17*cse593*cse973*cse974*cse975*state.xn[2] - cse595*cse972) - cse988*(0.086512999999999993*chi*cse2*cse578*cse68*cse978*cse979 - cse576*cse581*cse980) - state.xn[11]*(cse843*cse993 + cse849*cse995 + cse853*cse996 + cse854*cse996) - state.xn[3]*((cse740) ? (
   0
)
: (
   cse763*cse778
)) - state.xn[7]*((cse957) ? (
   cse1003*cse945 + cse1006*cse951 + cse1009*cse955 + cse1009*cse956
)
: (
   0
));
  jac(19, 4) = 1.4949836836302895e-18*state.c_hat*cse1016*state.rn[0] + 3.8323655572603624e-29*state.c_hat*cse1022*cse696*state.rn[0]*state.xn[3] + 3.3999999999999999e-24*chi*cse568*cse572*cse68*d2g + 1.7623947000000002e-12*crate*cse1030*cse719*state.xn[5] + 1.602177e-12*crate*cse1030*state.xn[0] + 3.2043540000000001e-12*crate*cse1030*state.xn[3] + 1.7623947000000002e-12*crate*cse195*cse689*cse695*state.xn[5] + 3.2043540000000001e-12*crate*cse689 + 2.0187430200000004e-26*cse10*cse1022*cse696*d2g*state.xn[0] - cse1011 - cse1013 - cse1015 - cse1018*cse666 - cse1019*cse1020 - 7.1777529600000012e-12*cse167 - 7.1777529600000012e-12*cse176 - 1.4355505920000002e-11*cse178 - 7.1777529600000012e-12*cse503 + 6.0000000000000005e-15*cse628*d2g*state.xn[0] - cse657 - cse694*(-cse644 - cse692) - cse707*(-cse1019*cse1023 - cse666*cse671) - state.xn[11]*(cse1035*cse843 + cse1040*cse849 + cse1042*cse853 + cse1042*cse854) - state.xn[3]*((cse740) ? (
   0
)
: (
   cse752*cse778
)) - state.xn[7]*((cse957) ? (
   cse1046*cse945 + cse1051*cse951 + cse1054*cse955 + cse1054*cse956
)
: (
   0
)) - ((cse740) ? (
   0.0
)
: (
   cse1031*cse774
));
  jac(19, 5) = -cse1011 - cse1013 - cse1015 + 3.3999999999999999e-24*cse1055 + cse1056*cse962 + cse1056*cse965 + cse1057*cse967 + cse1057*cse968 + cse1057*cse970 + cse194*cse960 - 7.1777529600000012e-12*cse204 - 7.1777529600000012e-12*cse509 + cse57*cse628;
  jac(19, 6) = cse1058 + cse959*cse969;
  jac(19, 7) = cse1058;
  jac(19, 8) = -((cse957) ? (
   cse938*cse945 + cse942*cse951
)
: (
   0.0
));
  jac(19, 9) = -cse1017*cse706;
  jac(19, 10) = -cse637*cse656 - cse694*(100000.0*cse1059*cse1060*cse637*cse640 - cse652*(100000.0*cse1059*cse1060*cse634*cse640*cse645 - cse1062*cse646) - cse653*(-cse1062*cse648*cse650 + 100000.0*cse1063*cse1064*cse640));
  jac(19, 11) = cse971;
  jac(19, 12) = -cse836*cse843 - cse840*cse849;
  jac(19, 15) = cse971;
  jac(19, 16) = cse971;
  jac(19, 17) = -5.6994379667262474e-29*cse1010 - 9.6000000000000005e-32*cse1012 - 300.0*cse1014 + 5.0999999999999995e-24*cse1055 + cse1065*cse962 + cse1065*cse965 + cse1066*cse967 + cse1066*cse968 + cse1066*cse970 - 7.1777529600000012e-12*cse244 + cse248*cse960 + cse512*cse628 - 7.1777529600000012e-12*cse513;
  jac(19, 19) = cse308*(1.7e-24*chi*cse10*cse568*cse68*cse982*d2g*(-0.00012636182959999999*cse286*cse983 - 0.018302846000000001*cse286*cse984) + 5.6227499999999995e-25*chi*cse10*cse572*cse68*d2g*std::pow(state.T, -0.85299999999999998) - cse1018*cse1079 - cse1020*cse1082 - 1.6173435610483583e-5*cse1067*cse1068*cse597 - 2.0489994438957239e-11*cse1068*cse1069*cse598 + 7.8348995947646122e-14*cse1070*cse592*cse593*cse597*cse974*state.xn[0]*state.xn[2] - cse1072*cse597*cse985 - cse1073*cse657 - 5.2835524837844135e-18*cse1075*cse673*state.xn[8] - cse1081*cse443*cse961 - cse1081*cse964 - 2.1789607199999999e-11*cse293 - 7.1777529600000012e-12*cse517 - 7.1777529600000012e-12*cse518 - cse602*cse609*psigd_coll_interp2d_partial_1(cse10, state.T) - cse606*(1.6000000000000001e-32*cse110*cse273*cse603 + cse604) - 1000*cse631*std::pow(cse632, 999)*cse655*cse693 - cse694*(-cse1073*cse644 + cse1083*cse1084*cse647*cse651 + cse1083*cse1084 - cse1087*cse637 - cse652*(-cse1083*cse644*cse645 - cse1087*cse634*cse645 - cse1089*cse646) - cse653*(cse1064*(-cse1063*cse1088 + alpha_co_interp2d_partial_0(state.T, cse642)) + cse645*cse691*(cse1073*cse648 - cse1089*cse637/std::pow(cse645, 2)))) - cse707*(-cse1023*cse1082 - 182.41999999999999*cse1075 - cse1079*cse671) - cse986*(-0.0015792330470357991*cse1067*cse589 + 7.6502807803182096e-12*cse1070*cse592*cse593*cse974*state.xn[2] - 2.0007175426973804e-9*cse1071*cse594 - 1.3338116951315871e-8*cse1072*cse590) - cse988*(-0.043256499999999996*cse1074*cse578 + 0.84999999999999998*cse1074*cse581) - cse988*(-1.8989*cse1074*cse981 + cse3*cse579) - state.xn[11]*(cse1108*cse843 + cse1112*cse849 + cse1113*cse853 + cse1113*cse854) - state.xn[3]*((cse740) ? (
   0
)
: (
   cse1031*cse1091 + cse774*cse777*(-cse1091*cse775 - cse765*cse774*(-cse753*(3.0182298984217999*cse1092*cse742*cse766 + 4.8223900769399082*cse1092*cse766 - 3.5529549287336835*cse1094 - 1.3735579540080118*cse1095 - 0.3872755400043702*cse1096) - cse755*(3.6184459289309552*cse1093 + 0.0708789387907936*cse1094 - 4.6974781513675152*cse1095 + 3.7035619079275177*cse1096 - 1.6316107607322889*cse1097) - cse757*(5.0409049417480247*cse1093 - 3.7541549062629067*cse1094 - 1.528565035159452*cse1095 + 2.2094886994529301*cse1096 + 2.0057552335975877*cse1097) - state.xn[0]*((cse747) ? (
      cse7*cse746*(37.383713*cse1093 + 116.29033200000001*cse1094 + 80.639324000000002*cse1095 + 19.239805*cse1096 + 145.968309*cse1097)
   )
   : ((cse749) ? (
      cse748*(8.2184944748967013*cse1092*cse766 - 52.189748993977005*cse1094 - 196.44057098336626*cse1095 - 48.951834264235494*cse1096 - 192.38155095558542*cse1097)
   )
   : (
      cse750*(10.695627721640689*cse1093 - 17.135767342440825*cse1094 - 50.756388852554174*cse1095 + 17.889115159724135*cse1096 + 41.010708268606813*cse1097)
   ))) - state.xn[2]*((cse664) ? (
      cse761*(38.719649225812766*cse1093 + 445.51869310442481*cse1094 + 6766.9891340104432*cse1095 + 11324.985706577943*cse1096 + 2370.6849681533818*cse1097 + 11077.448644792414*cse1098 + 5869.3152370465659*cse1099 + 1300.6686834345674*cse1100)
   )
   : (
      cse762*(3.8689780091986448*cse1093 + 4.297112944704045*cse1094 - 43.540996231705549*cse1095 - 101.40241318979159*cse1096 + 27.91190909651122*cse1097 + 123.18901581604325*cse1098 + 103.76942484394611*cse1099 - 117.34186576967058*cse1100)
   ))))
)) - state.xn[7]*((cse957) ? (
   cse1122*cse945 + cse1126*cse951 + cse1127*cse955 + cse1127*cse956
)
: (
   0
)));
  jac(19, 20) = cse1016*cse524;
  jac(20, 4) = -5.4061971382371991e-18*cse441;
  jac(20, 8) = -cse442;
  jac(20, 10) = -cse543;
  jac(20, 13) = -cse454;
  jac(20, 20) = cse1128;
  jac(21, 4) = -5.4061971382371991e-18*state.c_hat*state.rn[2*0+1];
  jac(21, 8) = -cse489*state.rn[2*0+1];
  jac(21, 10) = -cse548*state.rn[2*0+1];
  jac(21, 13) = -cse487*state.rn[2*0+1];
  jac(21, 21) = cse1128;
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
