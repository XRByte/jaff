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
  const amrex::Real cse25 = std::pow(10, 0.41599999999999998*cse17*cse20 + 0.32700000000000001*cse22 - 3) + std::pow(10, 1.3*cse17*cse20 - 1.6200000000000001*cse22 - 4.8449999999999998)*cse24;
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
  const amrex::Real cse31 = std::pow(state.xn[0], 2);
  const amrex::Real cse32 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse33 = ((cse29) ? (
   std::pow(10, cse27*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse14 - 1)) + cse28*std::log(3.5199999999999998e-9*std::exp(-43900*cse14)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse34 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse35 = cse34*state.xn[2]*state.xn[16];
  const amrex::Real cse36 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[1]*state.xn[11];
  const amrex::Real cse37 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse38 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse37)/cse37;
  const amrex::Real cse39 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse40 = chi*cse38*cse39;
  const amrex::Real cse41 = std::pow(state.T, -1.3);
  const amrex::Real cse42 = std::exp(-23*cse14);
  const amrex::Real cse43 = cse41*cse42*state.xn[3]*state.xn[8];
  const amrex::Real cse44 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse45 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av)*state.xn[14];
  const amrex::Real cse46 = cse44 + cse45;
  const amrex::Real cse47 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse14) + 0.62)*state.xn[15];
  const amrex::Real cse48 = 1.3500000000000001e-9*cse47*state.xn[6];
  const amrex::Real cse49 = state.xn[3]*state.xn[6];
  const amrex::Real cse50 = 1.2599999999999999e-13*cse49*std::exp(-22.5*cse14);
  const amrex::Real cse51 = cse48 + cse50;
  const amrex::Real cse52 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse53 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse54 = -cse52 + cse53;
  const amrex::Real cse55 = 2.2999999999999998*cse24 + 1.5;
  const amrex::Real cse56 = crate*cse55;
  const amrex::Real cse57 = cse56*state.xn[0];
  const amrex::Real cse58 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse59 = ((cse29) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse58, 8) + 0.000111954395*std::pow(cse58, 7) - 0.00263197617*std::pow(cse58, 6) + 0.034825597700000002*std::pow(cse58, 5) - 0.28770560000000001*std::pow(cse58, 4) + 1.56315498*std::pow(cse58, 3) - 5.7393287500000003*std::pow(cse58, 2))
)
: (
   0.0
))*state.xn[0]*state.xn[2];
  const amrex::Real cse60 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[2]*state.xn[10];
  const amrex::Real cse61 = 4.8813573300130879e-6*std::pow(cse14, 1.5)*std::pow(114.81344434488582*std::pow(cse14, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse62 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse63 = chi*cse62;
  const amrex::Real cse64 = cse3*cse63;
  const amrex::Real cse65 = 1.7*cse64;
  const amrex::Real cse66 = std::log(state.T);
  const amrex::Real cse67 = cse19*d2g;
  const amrex::Real cse68 = 1.2250000000000001e-11*cse67*state.xn[1]/(1.6774414051711492e-5*std::pow(cse64, 1.3779999999999999)*(508.69999999999999*std::pow(cse65, -1.102e-5*cse66 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse69 = crate*state.xn[9];
  const amrex::Real cse70 = 6.5199999999999996*cse23*cse69*state.xn[0];
  const amrex::Real cse71 = -cse57 - cse59 + cse60 + cse61 + cse68 - cse70;
  const amrex::Real cse72 = cse32 - cse36;
  const amrex::Real cse73 = cse18*cse56;
  const amrex::Real cse74 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[2]*state.xn[13];
  const amrex::Real cse75 = 2.0740886949963892e-18*state.c_hat*state.xn[12];
  const amrex::Real cse76 = cse75*state.rn[2*0];
  const amrex::Real cse77 = -cse76;
  const amrex::Real cse78 = 2.1659999999999999e-12*cse67*state.xn[13]/(1.534817233071468e-7*std::pow(cse64, 1.8740000000000001)*(43750.0*std::pow(cse65, -7.538e-5*cse66 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse79 = 16800.0*crate*cse24*state.xn[12];
  const amrex::Real cse80 = (1.0537934368240927e-9*std::pow(cse14, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse14) + 1.31e-7*std::exp(-26.600000000000001*cse14) + 3.4e-8*std::exp(-7.6200000000000001*cse14) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse14)))*state.xn[7]*state.xn[16];
  const amrex::Real cse81 = cse23*state.xn[2];
  const amrex::Real cse82 = ((cse81 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse83 = cse10*cse82/(cse82 + 1);
  const amrex::Real cse84 = cse11 + 1.3007021411393064e-6*cse35 + cse80 + cse83;
  const amrex::Real cse85 = cse17*cse66;
  const amrex::Real cse86 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse67*state.xn[6]/(7.104740267838494e-7*std::pow(cse64, 1.512)*(5115.0*std::pow(cse65, -5.4939999999999998e-7*cse66 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse85*(cse85*(0.044330000000000001*cse85 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse87 = crate*state.xn[7];
  const amrex::Real cse88 = 3.8500000000000001*cse87;
  const amrex::Real cse89 = 0.15970000000000001*std::exp(-49550.0*cse14);
  const amrex::Real cse90 = ((1.6339999999999999e-6*std::exp(-15230.0*cse14) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse14) + 6.3460000000000002e-9*std::exp(-12.17*cse14))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse89 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse89 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse91 = 4.5579999999999993e-11*cse67*state.xn[8]/(0.011078790689599145*std::pow(cse64, 1.1279999999999999)*(433.10000000000002*std::pow(cse65, -0.00013329999999999999*cse66 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse92 = 1040.0*cse24*cse87;
  const amrex::Real cse93 = 2.808e-22*ncol_H2;
  const amrex::Real cse94 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse93)*std::exp(-1.6000000000000001e-17*ncol_C)*state.xn[7]/(cse93 + 1);
  const amrex::Real cse95 = -cse88 + cse90 + cse91 - cse92 - cse94;
  const amrex::Real cse96 = 3.2999999999999996e-13*cse43;
  const amrex::Real cse97 = cse8 + cse9;
  const amrex::Real cse98 = cse96 + cse97;
  const amrex::Real cse99 = -cse80;
  const amrex::Real cse100 = -cse83;
  const amrex::Real cse101 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse102 = cse33*state.xn[3];
  const amrex::Real cse103 = 5.6999999999999997e-11*cse40;
  const amrex::Real cse104 = 7.2000000000000002e-15*cse49;
  const amrex::Real cse105 = cse1 + cse104 + cse73;
  const amrex::Real cse106 = -cse53;
  const amrex::Real cse107 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse108 = cse104 + cse107 + cse51 + cse86;
  const amrex::Real cse109 = cse52 + cse99;
  const amrex::Real cse110 = 184.0*cse24*cse69;
  const amrex::Real cse111 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av)*state.xn[9];
  const amrex::Real cse112 = cse110 + cse111 - cse15;
  const amrex::Real cse113 = 9.1500000000000005e-10*cse47*state.xn[8];
  const amrex::Real cse114 = cse101 - cse60 + cse70;
  const amrex::Real cse115 = cse107 - cse44;
  const amrex::Real cse116 = cse100 + cse12 + cse13;
  const amrex::Real cse117 = -cse74 + cse76 - cse78 + cse79;
  const amrex::Real cse118 = cse65 + 50;
  const amrex::Real cse119 = std::log(1.7*cse2*cse63/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse120 = std::pow(state.T, 0.14999999999999999)*state.xn[2]/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse121 = cse120*std::exp(-118400.0*cse14)/(4.4460389837719569e-9*cse120 + 626500000.0);
  const amrex::Real cse122 = 0.001*state.T;
  const amrex::Real cse123 = cse122 + 1;
  const amrex::Real cse124 = std::pow(cse123, -2);
  const amrex::Real cse125 = 1.0/cse123;
  const amrex::Real cse126 = 1.0/(cse23*(cse103 + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse124 + 0.055500000000000001*cse125 - 11.06) + std::pow(10, -2.0230000000000001*cse124 - 3.6709999999999998*cse125 - 11.08)*cse24) + 1);
  const amrex::Real cse127 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse128 = state.T < 500;
  const amrex::Real cse129 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse130 = (1.0/100.0)*state.T;
  const amrex::Real cse131 = std::log(cse130);
  const amrex::Real cse132 = 4.5299999999999998e-6*cse127 + 7.5799999999999997e-10*std::pow(cse130, 0.0086999999999999994*cse131 + 0.12809999999999999)*cse19 + (0.75*((cse128) ? (
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
  const amrex::Real cse133 = cse132*std::exp(-91.209999999999994*cse14)/(cse132 + 2.3e-6);
  const amrex::Real cse134 = cse17*std::log(cse19);
  const amrex::Real cse135 = 2.0*cse24*((cse134 < 2) ? (
   10.0
)
: ((cse134 < 4) ? (
   (3.0/2.0)*cse134 + 7
)
: ((cse134 < 7) ? (
   (4.0/3.0)*cse134 + 23.0/3.0
)
: ((cse134 < 10) ? (
   (1.0/3.0)*cse134 + 44.0/3.0
)
: (
   18.0
))))) + 26.399999999999999*std::sqrt(cse81/(cse81 + 0.070000000000000007)) + 6.5;
  const amrex::Real cse136 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse137 = 1.75*cse136*cse19 + state.xn[3] + 680.10000000000002*state.xn[2]/cse136;
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
  const amrex::Real cse143 = state.T < 100.0;
  const amrex::Real cse144 = state.T < 6000.0;
  const amrex::Real cse145 = state.T < 10;
  const amrex::Real cse146 = std::log(cse122);
  const amrex::Real cse147 = cse146*cse17;
  const amrex::Real cse148 = std::pow(cse16, -8);
  const amrex::Real cse149 = std::pow(cse146, 8)*cse148;
  const amrex::Real cse150 = std::pow(cse16, -7);
  const amrex::Real cse151 = std::pow(cse146, 7)*cse150;
  const amrex::Real cse152 = std::pow(cse16, -6);
  const amrex::Real cse153 = std::pow(cse146, 6)*cse152;
  const amrex::Real cse154 = std::pow(cse16, -5);
  const amrex::Real cse155 = std::pow(cse146, 5)*cse154;
  const amrex::Real cse156 = std::pow(cse16, -4);
  const amrex::Real cse157 = std::pow(cse146, 4)*cse156;
  const amrex::Real cse158 = std::pow(cse16, -3);
  const amrex::Real cse159 = std::pow(cse146, 3);
  const amrex::Real cse160 = cse158*cse159;
  const amrex::Real cse161 = std::pow(cse146, 2)*cse21;
  const amrex::Real cse162 = std::pow(10.0, 16.815729999999999*cse147 + 70.609154000000004*cse149 + 364.14445999999998*cse151 + 801.81246999999996*cse153 + 983.67575999999997*cse155 + 734.71650999999997*cse157 + 343.1918*cse160 + 96.743155000000002*cse161 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse163 = state.xn[5] + state.xn[6];
  const amrex::Real cse164 = std::pow(10.0, 1.5714710999999999*cse147 + 0.32168730000000001*cse155 - 0.51002221000000003*cse157 - 0.23619984999999999*cse160 + 0.015391166*cse161 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse147 + 0.19191374999999999*cse155 - 0.16596184*cse157 + 0.29036281000000003*cse160 - 0.81520437999999995*cse161 - 23.689236999999999)*cse163 + std::pow(10.0, 2.0943374000000001*cse146*cse17 - 0.033638326000000003*cse155 - 0.14913216000000001*cse157 + 0.43693353000000001*cse158*cse159 - 0.77151435999999995*cse161 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse165 = std::pow(10, 37.383713*cse147 + 3.8479610000000002*cse155 + 20.159831000000001*cse157 + 48.656103000000002*cse160 + 58.145166000000003*cse161 - 16.818342000000001)*cse19 + cse164;
  const amrex::Real cse166 = std::pow(10.0, 1.6802758*cse147 - 6.3701156000000001*cse149 + 6.4380698000000001*cse151 + 8.9167182999999994*cse153 - 8.8077017000000009*cse155 - 4.7274035999999997*cse157 + 4.0406627000000004*cse160 + 0.93310621999999999*cse161 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse167 = state.T < 1000.0;
  const amrex::Real cse168 = std::pow(10.0, 3.5692468000000002*cse146*cse17 - 4.2519023000000002*cse155 - 21.328264000000001*cse157 - 27.850082*cse160 - 11.33286*cse161 - 24.311209000000002)*cse19 + cse164;
  const amrex::Real cse169 = std::pow(10.0, 4.6450521*cse147 + 1.5538288*cse155 - 5.5108049000000001*cse157 + 5.9369081000000001*cse160 - 3.7209846*cse161 - 24.311209000000002)*cse19 + cse164;
  const amrex::Real cse170 = std::pow(10.0, -0.62120210234156448*cse154 - 1.5370575959566981*cse156 + 2.5133588394664677*cse158 + 3.7525488682084647*cse17 - 2.4768712409536775*cse21 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse154 - 1.7105157385968939*cse156 + 1.6702447513374001*cse158 + 3.9225864834863131*cse17 - 2.6171337683481011*cse21 - 23.689236999999999)*cse163 + std::pow(10.0, 5.9406293599919788*cse154 - 5.2566362076906961*cse156 - 1.3586848802337363*cse158 + 2.8156982240432273*cse17 + 0.049411830040524525*cse21 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse154 - 56.79810800956944*cse156 + 34.150687525022562*cse158 + 8.3228160852326614*cse17 - 11.945856385319289*cse21 - 24.311209000000002)*cse19;
  const amrex::Real cse171 = ((cse144) ? (
   cse14
)
: (
   0.00016666666666666666
));
  const amrex::Real cse172 = 1.6e-18*std::exp(-11700.0*cse171) + 6.7e-19*std::exp(-5860.0*cse171) + 3.0e-24*std::exp(-510.0*cse171) + 4.985670872372847e-33*((cse144) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse144) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse144) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse173 = std::log(0.001*((cse144) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse174 = cse154*std::pow(cse173, 5);
  const amrex::Real cse175 = cse156*std::pow(cse173, 4);
  const amrex::Real cse176 = std::pow(cse173, 3);
  const amrex::Real cse177 = std::pow(cse173, 2)*cse21;
  const amrex::Real cse178 = cse17*cse173;
  const amrex::Real cse179 = cse158*cse176;
  const amrex::Real cse180 = cse148*std::pow(cse173, 8);
  const amrex::Real cse181 = cse150*std::pow(cse173, 7);
  const amrex::Real cse182 = cse152*std::pow(cse173, 6);
  const amrex::Real cse183 = 0.042999999999999997*cse131;
  const amrex::Real cse184 = 4.34e-10*std::pow(cse130, 0.755 - 0.16*cse131)*cse19 + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse130, 0.52500000000000002*cse131 + 1.198) + 5.2499999999999998e-13*std::pow(cse130, cse183 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse185 = 3.5700000000000001e-10*std::pow(cse130, 0.41899999999999998 - 0.0030000000000000001*cse131)*cse19 + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse130, 0.025000000000000001*cse131 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse130, cse183 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse186 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse187 = 3.1899999999999998e-10*std::pow(cse130, 0.36899999999999999 - 0.0060000000000000001*cse131)*cse19 + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse130, 0.041000000000000002*cse131 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse130, 0.058000000000000003*cse131 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse188 = cse185 + 8.9099999999999997e-5;
  const amrex::Real cse189 = cse186*cse187;
  const amrex::Real cse190 = cse184*cse185*cse186 + (1.0/15.0)*cse184*cse187*std::exp(-425.40000000000003*cse14) + cse188*cse189;
  const amrex::Real cse191 = cse184 + 1.7499999999999998e-5;
  const amrex::Real cse192 = cse187 + 1.34e-10;
  const amrex::Real cse193 = (3.0/5.0)*cse185*std::exp(-227.80000000000001*cse14);
  const amrex::Real cse194 = cse189*cse191 + cse191*cse193 + cse192*cse193;
  const amrex::Real cse195 = 1.0/((1.0/3.0)*cse184*cse192*std::exp(-98.800000000000011*cse14) + cse188*cse191 + cse188*cse192 + cse190 + cse194);
  const amrex::Real cse196 = 1.7258000000000001e-6*cse127;
  const amrex::Real cse197 = 0.045999999999999999*cse131;
  const amrex::Real cse198 = 8.9000000000000003e-11*std::pow(cse130, cse197 + 0.22800000000000001)*cse19 + cse196*std::exp(((cse167) ? (
   cse66*(cse66*(cse66*(0.0070527699999999999*cse66 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse66*(cse66*(cse66*(0.093813800000000003*cse66 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse130, 0.037999999999999999*cse131 + 0.16900000000000001) + 2.15e-11*std::pow(cse130, 0.048000000000000001*cse131 - 0.01))*state.xn[3];
  const amrex::Real cse199 = 2.6400000000000002e-10*std::pow(cse130, cse197 + 0.23100000000000001)*cse19 + cse196*std::exp(((cse167) ? (
   cse66*(cse66*(cse66*(0.00235272*cse66 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse66*(cse66*(cse66*(0.097857299999999994*cse66 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse130, 0.031*cse131 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse130, 0.064000000000000001*cse131 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse200 = 2.8763333333333333e-6*cse127*std::exp(((cse167) ? (
   cse66*(cse66*(cse66*(-0.00065632500000000003*cse66 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse66*(cse66*(cse66*(0.10508000000000001*cse66 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse130, 0.057000000000000002*cse131 + 0.115)*cse19 + (5.3249999999999997e-11*std::pow(cse130, 0.049000000000000002*cse131 - 0.0040000000000000001) + 1.675e-11*std::pow(cse130, 0.10199999999999999*cse131 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse201 = 5*std::exp(-62.460000000000001*cse14);
  const amrex::Real cse202 = cse200 + 7.8800000000000004e-8;
  const amrex::Real cse203 = cse198*cse201;
  const amrex::Real cse204 = (25.0/3.0)*cse198*cse199*std::exp(-101.30000000000001*cse14) + cse199*cse200*cse201 + cse202*cse203;
  const amrex::Real cse205 = cse199 + 2.65e-7;
  const amrex::Real cse206 = cse198 + 1.81e-14;
  const amrex::Real cse207 = 3*cse200*std::exp(-23.619999999999997*cse14);
  const amrex::Real cse208 = cse203*cse205 + cse205*cse207 + cse206*cse207;
  const amrex::Real cse209 = 1.0/((5.0/3.0)*cse199*cse206*std::exp(-38.840000000000003*cse14) + cse202*cse205 + cse202*cse206 + cse204 + cse208);

  ydot(1) = cse1 + cse11 + cse12 + cse13 + cse15 + cse18*cse33 + 2*cse30 - 6.0000000000000005e-15*cse31*d2g - cse32 + 2.9935562710997168e-6*cse35 + cse36 + 1.1399999999999999e-10*cse40*state.xn[3] + 4.2899999999999994e-13*cse43 + cse46 + cse51 + cse54 + 3.2000000000000001e-9*cse7 + cse71 + cse9;
  ydot(2) = cse0 + cse50 + cse57 + cse59 - cse61 - cse68 + cse72;
  ydot(3) = -cse71 + cse73 - cse74 - cse77 - cse78 + cse79 - cse84 - cse86 - cse95 - cse98;
  ydot(4) = -cse100 + cse101 - cse102 - cse103*state.xn[3] - cse105 + cse11 - cse30 + 3.0000000000000002e-15*cse31*d2g + 4.5427507615910105e-7*cse34*state.xn[2]*state.xn[16] - cse50 - cse54 - cse98 - cse99;
  ydot(5) = cse105 + cse106;
  ydot(6) = cse108;
  ydot(7) = -cse108;
  ydot(8) = cse109 + cse112 + 9.8999999999999995e-14*cse43 + cse45 + cse95;
  ydot(9) = cse107 - cse113 + cse88 - cse90 - cse91 + cse92 + cse94 - cse96;
  ydot(10) = -cse112 - cse114 - cse115;
  ydot(11) = cse113 + cse114;
  ydot(12) = cse110 + cse111 + cse115 + cse116 + cse72 + cse8;
  ydot(13) = -cse117;
  ydot(14) = cse117;
  ydot(15) = -cse109 + 2.3099999999999997e-13*cse41*cse42*state.xn[3]*state.xn[8] - cse46;
  ydot(16) = -cse113 - cse116 - cse15 - cse48 + cse9;
  ydot(17) = -cse101 - cse106 - cse84;
  ydot(18) = cse48 - cse72 - cse97;
  ydot(19) = 1.7e-24*chi*cse19*cse62*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse118, 0.43099999999999999)*(0.0043*std::pow(cse118, 0.69199999999999995) + 1) + 1) + 5.6999999999999997e-11*chi*cse38*cse39*(2.5634832e-11*cse126 + 6.4087080000000003e-13)*state.xn[3] + 1.602177e-12*crate*cse135*state.xn[0] + 3.2043540000000001e-12*crate*cse135*state.xn[3] + 1.7623947000000002e-12*crate*cse135*state.xn[5]/cse55 - 7.1777529600000012e-12*cse102 - 1.3659996292638161e-10*cse121*cse19/(1.3338116951315871e-8*cse121 + 1) - 5.7927337833400002e-20*cse133*state.xn[8]/(2*cse133 + 1) - cse137*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse137*cse141 + cse139*std::pow(cse137/cse142, alpha_co_interp2d(state.T, cse140))*(-cse138*cse141*cse142 + 1) + cse139) - 7.1777529600000012e-12*cse30 + 3.0000000000000002e-15*cse31*d2g*(6.7291434000000005e-12*cse126 + 3.2043540000000002e-13) - 2.1789607199999999e-11*cse59 - 1.8998126555754159e-29*cse67*std::pow(state.T, 0.45350000000000001 + 2.234/cse119)*std::exp(cse119*(1.4419999999999999 - 0.050889999999999998*cse119))*state.xn[2] - (2.3872025442750958e-19*cse190*cse195 + 2.8023005140020001e-18*cse194*cse195)*state.xn[11] - ((((cse128 && cse143 && cse144) ? (
   cse145 || cse162 + cse165 < 1.0e-100
)
: (
   ((cse143 && cse144) ? (
      cse145 || cse165 + cse166 < 1.0e-100
   )
   : (
      ((cse128 && cse144 && cse167) ? (
         cse145 || cse162 + cse168 < 1.0e-100
      )
      : (
         ((cse144 && cse167) ? (
            cse145 || cse166 + cse168 < 1.0e-100
         )
         : (
            ((cse128 && cse144) ? (
               cse145 || cse162 + cse169 < 1.0e-100
            )
            : (
               ((cse144) ? (
                  cse145 || cse166 + cse169 < 1.0e-100
               )
               : (
                  ((cse128) ? (
                     cse145 || std::pow(10.0, 7500.646099484944*cse148 + 21588.957206746247*cse150 + 26530.843406310876*cse152 + 18165.631967965237*cse154 + 7572.4886742758581*cse156 + 1974.131269262874*cse158 + 30.129743459482278*cse17 + 310.58441786958315*cse21 - 21.928795999999998)*state.xn[2] + cse170 < 1.0e-100
                  )
                  : (
                     cse145 || std::pow(10.0, -676.68255490510751*cse148 + 381.69251128040054*cse150 + 295.04162851880631*cse152 - 162.65264843551881*cse154 - 48.723840736518795*cse156 + 23.242975457496804*cse158 + 3.0106500755647456*cse17 + 2.9956460707652877*cse21 - 22.921188999999998)*state.xn[2] + cse170 < 1.0e-100
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
   cse172/(cse172/(std::pow(10.0, 0.19191374999999999*cse174 - 0.16596184*cse175 - 0.81520437999999995*cse177 + 2.1892372*cse178 + 0.29036281000000003*cse179 - 23.689236999999999)*cse163 + std::pow(10.0, 0.32168730000000001*cse174 - 0.51002221000000003*cse175 + 0.015391166*cse177 + 1.5714710999999999*cse178 - 0.23619984999999999*cse179 - 22.089523)*state.xn[1] + std::pow(10.0, 0.43693353000000001*cse158*cse176 + 2.0943374000000001*cse17*cse173 - 0.033638326000000003*cse174 - 0.14913216000000001*cse175 - 0.77151435999999995*cse177 - 23.962112000000001)*state.xn[3] + cse19*((((cse144) ? (
      cse143
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse174 + 20.159831000000001*cse175 + 58.145166000000003*cse177 + 37.383713*cse178 + 48.656103000000002*cse179 - 16.818342000000001)
   )
   : ((((cse144) ? (
      cse167
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse17*cse173 - 4.2519023000000002*cse174 - 21.328264000000001*cse175 - 11.33286*cse177 - 27.850082*cse179 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse174 - 5.5108049000000001*cse175 - 3.7209846*cse177 + 4.6450521*cse178 + 5.9369081000000001*cse179 - 24.311209000000002)
   ))) + ((cse128) ? (
      std::pow(10.0, 983.67575999999997*cse174 + 734.71650999999997*cse175 + 96.743155000000002*cse177 + 16.815729999999999*cse178 + 343.1918*cse179 + 70.609154000000004*cse180 + 364.14445999999998*cse181 + 801.81246999999996*cse182 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse174 - 4.7274035999999997*cse175 + 0.93310621999999999*cse177 + 1.6802758*cse178 + 4.0406627000000004*cse179 - 6.3701156000000001*cse180 + 6.4380698000000001*cse181 + 8.9167182999999994*cse182 - 22.921188999999998)
   ))*state.xn[2]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse204*cse209 + 2.5697412351439998e-22*cse208*cse209
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse19, state.T), 3.2000000000000002e-32*cse2*cse67*(state.T - 2.1544346900318838*std::pow(85973.158732959506*cse63 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = cse77;
  ydot(21) = -cse75*state.rn[2*0+1];
  // $JAFF END
  // $JAFF SUB nspec
  ydot(19) /= state.rho;
  // $JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
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
  const amrex::Real cse11 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse12 = std::pow(cse7, -2);
  const amrex::Real cse13 = std::pow(cse11, 2)*cse12;
  const amrex::Real cse14 = std::pow(10, 0.41599999999999998*cse11*cse8 + 0.32700000000000001*cse13 - 3);
  const amrex::Real cse15 = 1.0/cse10;
  const amrex::Real cse16 = std::pow(10, 1.3*cse11*cse8 - 1.6200000000000001*cse13 - 4.8449999999999998);
  const amrex::Real cse17 = cse15*cse16*state.xn[3];
  const amrex::Real cse18 = cse14 + cse17;
  const amrex::Real cse19 = cse18 < 1.0e-50;
  const amrex::Real cse20 = ((cse19) ? (
   1.0e-50
)
: (
   cse18
));
  const amrex::Real cse21 = cse10*cse20;
  const amrex::Real cse22 = cse21 + 1;
  const amrex::Real cse23 = cse8/cse22;
  const amrex::Real cse24 = std::log(3.5199999999999998e-9*std::exp(-43900*cse3));
  const amrex::Real cse25 = cse21*cse23;
  const amrex::Real cse26 = std::pow(10, cse23*cse6 + cse24*cse25);
  const amrex::Real cse27 = state.T > 700.0;
  const amrex::Real cse28 = ((cse27) ? (
   cse26
)
: (
   0.0
));
  const amrex::Real cse29 = std::pow(state.xn[3], 2);
  const amrex::Real cse30 = std::log(1.3000000000000001e-9*std::exp(-53300*cse3));
  const amrex::Real cse31 = std::pow(cse10, 2);
  const amrex::Real cse32 = 1.0/cse31;
  const amrex::Real cse33 = cse32*state.xn[3];
  const amrex::Real cse34 = cse16*cse33;
  const amrex::Real cse35 = cse10*((cse19) ? (
   0
)
: (
   -cse34
));
  const amrex::Real cse36 = cse23*cse35;
  const amrex::Real cse37 = cse20*cse23;
  const amrex::Real cse38 = std::pow(state.T, 4.1881000000000004);
  const amrex::Real cse39 = 6.7610000000000002e-6*state.T + 1;
  const amrex::Real cse40 = std::pow(cse39, 5.6881000000000004);
  const amrex::Real cse41 = 54657.400000000001*cse3;
  const amrex::Real cse42 = std::exp(-cse41);
  const amrex::Real cse43 = cse42/cse40;
  const amrex::Real cse44 = std::log(5.9959999999999997e-30*cse38*cse43);
  const amrex::Real cse45 = cse8/std::pow(cse22, 2);
  const amrex::Real cse46 = cse45*(-cse20 - cse35);
  const amrex::Real cse47 = cse21*cse46;
  const amrex::Real cse48 = std::pow(10, cse23*cse44 + cse25*cse30);
  const amrex::Real cse49 = cse48*cse7;
  const amrex::Real cse50 = ((cse27) ? (
   cse49*(cse30*cse36 + cse30*cse37 + cse30*cse47 + cse44*cse46)
)
: (
   0
));
  const amrex::Real cse51 = d2g*state.xn[0];
  const amrex::Real cse52 = cse26*cse7;
  const amrex::Real cse53 = ((cse27) ? (
   cse52*(cse24*cse36 + cse24*cse37 + cse24*cse47 + cse46*cse6)
)
: (
   0
));
  const amrex::Real cse54 = 6.5199999999999996*cse15;
  const amrex::Real cse55 = crate*state.xn[9];
  const amrex::Real cse56 = cse54*cse55;
  const amrex::Real cse57 = crate*state.xn[0];
  const amrex::Real cse58 = cse32*cse57*state.xn[9];
  const amrex::Real cse59 = 6.5199999999999996*cse58;
  const amrex::Real cse60 = -cse59;
  const amrex::Real cse61 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse62 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse63 = cse62*state.xn[17];
  const amrex::Real cse64 = cse33*cse57;
  const amrex::Real cse65 = 2.2999999999999998*cse64;
  const amrex::Real cse66 = 2.2999999999999998*cse15*state.xn[3] + 1.5;
  const amrex::Real cse67 = crate*cse66;
  const amrex::Real cse68 = 1.0/state.xn[2];
  const amrex::Real cse69 = cse2*cse68;
  const amrex::Real cse70 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse71 = chi*cse70;
  const amrex::Real cse72 = cse69*cse71;
  const amrex::Real cse73 = 1.7*cse72;
  const amrex::Real cse74 = std::log(state.T);
  const amrex::Real cse75 = 1.102e-5*cse74 + 0.4723;
  const amrex::Real cse76 = std::pow(cse73, -cse75);
  const amrex::Real cse77 = cse76*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse78 = 508.69999999999999*cse77;
  const amrex::Real cse79 = cse78 + 1;
  const amrex::Real cse80 = std::pow(cse72, 1.3779999999999999);
  const amrex::Real cse81 = 1.6774414051711492e-5*cse80;
  const amrex::Real cse82 = cse79*cse81 + 1;
  const amrex::Real cse83 = 1.0/cse82;
  const amrex::Real cse84 = 1.2250000000000001e-11*cse83*d2g;
  const amrex::Real cse85 = cse84*state.xn[1];
  const amrex::Real cse86 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse87 = std::pow(cse86, 2);
  const amrex::Real cse88 = std::pow(cse86, 3);
  const amrex::Real cse89 = std::pow(cse86, 4);
  const amrex::Real cse90 = std::pow(cse86, 5);
  const amrex::Real cse91 = std::pow(cse86, 6);
  const amrex::Real cse92 = std::pow(cse86, 7);
  const amrex::Real cse93 = std::exp(-2.0391498499999999e-6*std::pow(cse86, 8) - 5.7393287500000003*cse87 + 1.56315498*cse88 - 0.28770560000000001*cse89 + 0.034825597700000002*cse90 - 0.00263197617*cse91 + 0.000111954395*cse92);
  const amrex::Real cse94 = 5.9078447946382801e-70*cse93*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse95 = ((cse27) ? (
   cse94
)
: (
   0.0
));
  const amrex::Real cse96 = cse95*state.xn[2];
  const amrex::Real cse97 = cse61 + cse63 - cse65 + cse67 - cse85 + cse96;
  const amrex::Real cse98 = std::exp(-227*cse3);
  const amrex::Real cse99 = cse98*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse100 = cse99*state.xn[11];
  const amrex::Real cse101 = std::pow(cse3, 1.5);
  const amrex::Real cse102 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse103 = std::pow(cse102, -2.242);
  const amrex::Real cse104 = 4.8813573300130879e-6*cse101*cse103;
  const amrex::Real cse105 = cse65 + cse85;
  const amrex::Real cse106 = cse10*cse84 + cse104*state.xn[2] + cse105;
  const amrex::Real cse107 = cse100 + cse106;
  const amrex::Real cse108 = cse29*cse50;
  const amrex::Real cse109 = cse9*state.xn[0];
  const amrex::Real cse110 = 2*cse108 + cse109*cse53;
  const amrex::Real cse111 = cse110 + cse59;
  const amrex::Real cse112 = std::pow(state.xn[2], 2);
  const amrex::Real cse113 = 1.0/cse112;
  const amrex::Real cse114 = 0.00011320754716981131*cse69;
  const amrex::Real cse115 = cse114*state.xn[3];
  const amrex::Real cse116 = std::min(10000000000.0, cse115);
  const amrex::Real cse117 = cse116 + 1;
  const amrex::Real cse118 = std::pow(cse117, -2);
  const amrex::Real cse119 = (((cse115 - 10000000000.0 > 0) ? (
   0
)
: ((cse115 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse120 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse121 = cse113*cse118*cse119*cse120*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse122 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse123 = 2.9935562710997168e-6*cse122;
  const amrex::Real cse124 = cse113*cse2;
  const amrex::Real cse125 = cse119*state.xn[17];
  const amrex::Real cse126 = cse125*cse29;
  const amrex::Real cse127 = cse124*cse126;
  const amrex::Real cse128 = 3.6226415094339619e-13*cse118;
  const amrex::Real cse129 = 1.0/cse117;
  const amrex::Real cse130 = 1.811320754716981e-13*cse129;
  const amrex::Real cse131 = cse127*cse130;
  const amrex::Real cse132 = 1.811320754716981e-13*cse118;
  const amrex::Real cse133 = cse127*cse132;
  const amrex::Real cse134 = cse116*cse133;
  const amrex::Real cse135 = -cse131 + cse134;
  const amrex::Real cse136 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse137 = cse136*state.xn[10];
  const amrex::Real cse138 = cse95*state.xn[0];
  const amrex::Real cse139 = cse79*cse80;
  const amrex::Real cse140 = -cse75;
  const amrex::Real cse141 = d2g*state.xn[1];
  const amrex::Real cse142 = 1.2250000000000001e-11*cse10*cse141/std::pow(cse82, 2);
  const amrex::Real cse143 = cse104*state.xn[1] - cse138 + cse142*(2.3115142563258434e-5*cse139*cse68 + 0.0085331444281056355*cse140*cse68*cse77*cse80);
  const amrex::Real cse144 = cse137 + cse143;
  const amrex::Real cse145 = 13.039999999999999*cse58;
  const amrex::Real cse146 = std::exp(-22.5*cse3);
  const amrex::Real cse147 = 1.2599999999999999e-13*cse146;
  const amrex::Real cse148 = cse147*state.xn[6];
  const amrex::Real cse149 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse150 = cse149*state.xn[4];
  const amrex::Real cse151 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse152 = 1.7599999999999999e-9*cse151;
  const amrex::Real cse153 = cse150*cse152;
  const amrex::Real cse154 = cse28*state.xn[0];
  const amrex::Real cse155 = 2*cse20;
  const amrex::Real cse156 = cse23*cse30;
  const amrex::Real cse157 = cse155*cse156;
  const amrex::Real cse158 = cse32*cse9;
  const amrex::Real cse159 = cse158*cse16;
  const amrex::Real cse160 = cse10*((cse19) ? (
   0
)
: (
   cse15*cse16 - cse159
));
  const amrex::Real cse161 = cse45*(-cse155 - cse160);
  const amrex::Real cse162 = cse161*cse21;
  const amrex::Real cse163 = cse29*((cse27) ? (
   cse49*(cse156*cse160 + cse157 + cse161*cse44 + cse162*cse30)
)
: (
   0
));
  const amrex::Real cse164 = ((cse27) ? (
   cse48
)
: (
   0.0
));
  const amrex::Real cse165 = cse164*state.xn[3];
  const amrex::Real cse166 = 3.2000000000000001e-9*cse129;
  const amrex::Real cse167 = cse23*cse24;
  const amrex::Real cse168 = cse155*cse167;
  const amrex::Real cse169 = ((cse27) ? (
   cse52*(cse160*cse167 + cse161*cse6 + cse162*cse24 + cse168)
)
: (
   0
));
  const amrex::Real cse170 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse171 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse170)/cse170;
  const amrex::Real cse172 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse173 = chi*cse171*cse172;
  const amrex::Real cse174 = std::exp(-23*cse3);
  const amrex::Real cse175 = cse174*std::pow(state.T, -1.3);
  const amrex::Real cse176 = 4.2899999999999994e-13*cse175;
  const amrex::Real cse177 = cse125*cse69*state.xn[3];
  const amrex::Real cse178 = 1.6000000000000001e-9*cse129;
  const amrex::Real cse179 = cse178*state.xn[17];
  const amrex::Real cse180 = cse132*cse177;
  const amrex::Real cse181 = cse116*cse179 - cse116*cse180 + cse130*cse177;
  const amrex::Real cse182 = cse141*cse83;
  const amrex::Real cse183 = 2.4500000000000003e-11*cse182;
  const amrex::Real cse184 = 4.5999999999999996*cse33;
  const amrex::Real cse185 = -2.2999999999999998*cse15 + cse184;
  const amrex::Real cse186 = -cse185;
  const amrex::Real cse187 = cse186*cse57;
  const amrex::Real cse188 = cse118*cse119*cse68*state.xn[11]*state.xn[16];
  const amrex::Real cse189 = cse120*cse188;
  const amrex::Real cse190 = cse183 - cse187 - cse189;
  const amrex::Real cse191 = cse149*cse152*state.xn[3];
  const amrex::Real cse192 = cse10*((cse19) ? (
   0
)
: (
   -cse159
));
  const amrex::Real cse193 = cse45*(-cse155 - cse192);
  const amrex::Real cse194 = cse193*cse21;
  const amrex::Real cse195 = cse29*((cse27) ? (
   cse49*(cse156*cse192 + cse157 + cse193*cse44 + cse194*cse30)
)
: (
   0
));
  const amrex::Real cse196 = ((cse27) ? (
   cse52*(cse167*cse192 + cse168 + cse193*cse6 + cse194*cse24)
)
: (
   0
));
  const amrex::Real cse197 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse198 = -cse197;
  const amrex::Real cse199 = cse183 + cse184*cse57;
  const amrex::Real cse200 = cse198 + cse199;
  const amrex::Real cse201 = cse147*state.xn[3];
  const amrex::Real cse202 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse203 = 1.3500000000000001e-9*cse202;
  const amrex::Real cse204 = cse203*state.xn[15];
  const amrex::Real cse205 = cse201 + cse204;
  const amrex::Real cse206 = std::exp(0.11*cse3);
  const amrex::Real cse207 = cse206*state.xn[15];
  const amrex::Real cse208 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse209 = cse207*cse208;
  const amrex::Real cse210 = cse54*cse57;
  const amrex::Real cse211 = cse136*state.xn[2];
  const amrex::Real cse212 = cse211 + cse59;
  const amrex::Real cse213 = cse105 + cse212;
  const amrex::Real cse214 = cse129*state.xn[16];
  const amrex::Real cse215 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse216 = cse214*cse215;
  const amrex::Real cse217 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse218 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse219 = cse99*state.xn[1];
  const amrex::Real cse220 = cse218 + cse219;
  const amrex::Real cse221 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse222 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse223 = cse0*state.xn[0];
  const amrex::Real cse224 = cse105 + cse111;
  const amrex::Real cse225 = cse203*state.xn[6];
  const amrex::Real cse226 = cse206*cse208*state.xn[7];
  const amrex::Real cse227 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse228 = cse225 + cse226 + cse227;
  const amrex::Real cse229 = cse10*((cse19) ? (
   0
)
: (
   -3*cse34
));
  const amrex::Real cse230 = 3*cse20;
  const amrex::Real cse231 = cse45*(-cse229 - cse230);
  const amrex::Real cse232 = cse21*cse231;
  const amrex::Real cse233 = cse29*((cse27) ? (
   cse49*(cse156*cse229 + cse156*cse230 + cse231*cse44 + cse232*cse30)
)
: (
   0
));
  const amrex::Real cse234 = ((cse27) ? (
   cse52*(cse167*cse229 + cse167*cse230 + cse231*cse6 + cse232*cse24)
)
: (
   0
));
  const amrex::Real cse235 = 19.559999999999999*cse58;
  const amrex::Real cse236 = cse215*state.xn[11];
  const amrex::Real cse237 = cse129*cse236;
  const amrex::Real cse238 = 3.6750000000000002e-11*cse182 + 6.8999999999999995*cse64;
  const amrex::Real cse239 = cse235 + cse237 + cse238;
  const amrex::Real cse240 = cse62*state.xn[0];
  const amrex::Real cse241 = cse178*state.xn[3];
  const amrex::Real cse242 = cse116*cse241;
  const amrex::Real cse243 = std::pow(state.T, -2);
  const amrex::Real cse244 = 2.8349999999999999e-12*cse146*cse243*state.xn[3]*state.xn[6];
  const amrex::Real cse245 = -cse244;
  const amrex::Real cse246 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse247 = std::sqrt(cse3);
  const amrex::Real cse248 = std::sqrt(3)*cse247*cse3*state.xn[15];
  const amrex::Real cse249 = 1.7697487500000004e-8*cse248*state.xn[6];
  const amrex::Real cse250 = cse243*cse25;
  const amrex::Real cse251 = cse3*cse8;
  const amrex::Real cse252 = cse12*cse3;
  const amrex::Real cse253 = cse11*cse252;
  const amrex::Real cse254 = ((cse19) ? (
   0
)
: (
   cse14*cse7*(0.41599999999999998*cse251 + 0.65400000000000003*cse253) + cse17*cse7*(-3.2400000000000002*cse253 + 1.3*cse3*cse8)
));
  const amrex::Real cse255 = cse10*cse254;
  const amrex::Real cse256 = cse255*cse45;
  const amrex::Real cse257 = cse20*cse254*cse31*cse45;
  const amrex::Real cse258 = ((cse27) ? (
   cse49*(cse156*cse255 + 1.6677785190126752e+29*cse23*cse40*(-2.3058963562359999e-34*cse38*std::pow(cse39, -6.6881000000000004)*cse42 + 3.2772577039999999e-25*cse43*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse43*std::pow(state.T, 3.1881000000000004))*std::exp(cse41)/cse38 + 53300.0*cse250 - cse256*cse44 - cse257*cse30)
)
: (
   0
));
  const amrex::Real cse259 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse260 = 1.0/cse2;
  const amrex::Real cse261 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse262 = ((cse27) ? (
   cse52*(cse167*cse255 + 149925037481.25937*cse23*cse260*(3.3349999999999998e-12*cse260*cse5 + 4.2414529999999999e-7*cse261*cse5)*std::exp(cse4) - cse24*cse257 + 43900.0*cse250 - cse256*cse6)
)
: (
   0
));
  const amrex::Real cse263 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse264 = cse98*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse265 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse266 = cse174*state.xn[3]*state.xn[8];
  const amrex::Real cse267 = cse265*cse266;
  const amrex::Real cse268 = cse150*state.xn[3];
  const amrex::Real cse269 = 3.7768240343347639e-14*cse151*cse268;
  const amrex::Real cse270 = 7.3920000000000002e-11*cse268*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse271 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse272 = 227*cse100*cse243*state.xn[1];
  const amrex::Real cse273 = cse260*cse68;
  const amrex::Real cse274 = cse126*cse273;
  const amrex::Real cse275 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse276 = cse207*state.xn[7];
  const amrex::Real cse277 = 8.7465066538127767e-11*cse276*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse276*std::pow(state.T, -1.339);
  const amrex::Real cse278 = cse275 + cse277;
  const amrex::Real cse279 = state.xn[0]*state.xn[2]*((cse27) ? (
   7.9971871901929577e-69*cse93*std::pow(state.T, 12.536555999999999) + cse94*(-11.478657500000001*cse3*cse86 + 4.6894649399999997*cse3*cse87 - 1.1508224*cse3*cse88 + 0.1741279885*cse3*cse89 - 0.015791857020000001*cse3*cse90 + 0.00078368076500000001*cse3*cse91 - 1.6313198799999999e-5*cse3*cse92)
)
: (
   0
));
  const amrex::Real cse280 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse281 = 7.3220359950196314e-6*cse101*cse103*cse280;
  const amrex::Real cse282 = 0.0005114031087522854*std::pow(cse102, -3.242)*cse280*std::pow(cse3, 1.907);
  const amrex::Real cse283 = 0.5*cse3;
  const amrex::Real cse284 = cse3*std::log(cse73);
  const amrex::Real cse285 = cse142*(-1.1557571281629217e-5*cse139*cse3 - cse81*(8.0679819999999989*cse76*std::pow(state.T, -0.98414000000000001) + cse78*(cse140*cse283 - 1.102e-5*cse284)));
  const amrex::Real cse286 = 9.0566037735849048e-14*cse274;
  const amrex::Real cse287 = cse129*cse286;
  const amrex::Real cse288 = cse118*cse286;
  const amrex::Real cse289 = cse116*cse288;
  const amrex::Real cse290 = -cse287 + cse289;
  const amrex::Real cse291 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse292 = 1.1264150943396225e-13*cse188*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse214*cse291;
  const amrex::Real cse293 = cse279 + cse281 - cse282 - cse285 + cse290 + cse292;
  const amrex::Real cse294 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse295 = -cse105;
  const amrex::Real cse296 = -cse259 + cse264 + cse272;
  const amrex::Real cse297 = -cse56;
  const amrex::Real cse298 = cse297 + cse59;
  const amrex::Real cse299 = crate*cse29*cse32;
  const amrex::Real cse300 = 4.5999999999999996*cse299;
  const amrex::Real cse301 = 5.4939999999999998e-7*cse74 + 0.49559999999999998;
  const amrex::Real cse302 = std::pow(cse73, -cse301);
  const amrex::Real cse303 = cse302*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse304 = 5115.0*cse303;
  const amrex::Real cse305 = cse304 + 1;
  const amrex::Real cse306 = std::pow(cse72, 1.512);
  const amrex::Real cse307 = 7.104740267838494e-7*cse306;
  const amrex::Real cse308 = cse305*cse307 + 1;
  const amrex::Real cse309 = 1.0/cse308;
  const amrex::Real cse310 = cse309*d2g*state.xn[6];
  const amrex::Real cse311 = 5.5720000000000001e-12*cse310;
  const amrex::Real cse312 = crate*cse33;
  const amrex::Real cse313 = cse312*state.xn[7];
  const amrex::Real cse314 = 0.00013329999999999999*cse74 + 0.81200000000000006;
  const amrex::Real cse315 = std::pow(cse73, -cse314);
  const amrex::Real cse316 = cse315*std::pow(state.T, 0.04845);
  const amrex::Real cse317 = 433.10000000000002*cse316;
  const amrex::Real cse318 = cse317 + 1;
  const amrex::Real cse319 = std::pow(cse72, 1.1279999999999999);
  const amrex::Real cse320 = 0.011078790689599145*cse319;
  const amrex::Real cse321 = cse318*cse320 + 1;
  const amrex::Real cse322 = 1.0/cse321;
  const amrex::Real cse323 = cse322*d2g*state.xn[8];
  const amrex::Real cse324 = 1040.0*cse313 + 4.5579999999999993e-11*cse323;
  const amrex::Real cse325 = cse312*state.xn[12];
  const amrex::Real cse326 = 7.538e-5*cse74 + 0.89639999999999997;
  const amrex::Real cse327 = std::pow(cse73, -cse326);
  const amrex::Real cse328 = cse327*std::pow(state.T, 1.635e-6);
  const amrex::Real cse329 = 43750.0*cse328;
  const amrex::Real cse330 = cse329 + 1;
  const amrex::Real cse331 = std::pow(cse72, 1.8740000000000001);
  const amrex::Real cse332 = 1.534817233071468e-7*cse331;
  const amrex::Real cse333 = cse330*cse332 + 1;
  const amrex::Real cse334 = 1.0/cse333;
  const amrex::Real cse335 = cse334*d2g*state.xn[13];
  const amrex::Real cse336 = 16800.0*cse325 + 2.1659999999999999e-12*cse335;
  const amrex::Real cse337 = cse300 + cse311 + cse324 + cse336;
  const amrex::Real cse338 = cse337 + cse59;
  const amrex::Real cse339 = cse74*cse8;
  const amrex::Real cse340 = 0.044330000000000001*cse339 - 0.28520000000000001;
  const amrex::Real cse341 = cse339*cse340 - 1.6759999999999999;
  const amrex::Real cse342 = cse339*cse341 + 11.19;
  const amrex::Real cse343 = cse342*state.xn[6];
  const amrex::Real cse344 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse345 = cse305*cse306;
  const amrex::Real cse346 = -cse301;
  const amrex::Real cse347 = cse10*d2g;
  const amrex::Real cse348 = 5.5720000000000001e-12*cse347;
  const amrex::Real cse349 = cse348*state.xn[6]/std::pow(cse308, 2);
  const amrex::Real cse350 = cse343*cse344 + cse349*(0.0036340746469993896*cse303*cse306*cse346*cse68 + 1.0742367284971802e-6*cse345*cse68);
  const amrex::Real cse351 = std::exp(-15230.0*cse3);
  const amrex::Real cse352 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse353 = std::exp(-12.17*cse3);
  const amrex::Real cse354 = 1.6339999999999999e-6*cse351 + 9.7930000000000004e-9*cse352 + 6.3460000000000002e-9*cse353;
  const amrex::Real cse355 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse356 = std::exp(-49550.0*cse3);
  const amrex::Real cse357 = 0.15970000000000001*cse356;
  const amrex::Real cse358 = cse357 + 1.7848999999999999;
  const amrex::Real cse359 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse360 = 0.21509999999999996 - cse357;
  const amrex::Real cse361 = std::pow(cse355, -cse358)*std::pow(cse359, -cse360);
  const amrex::Real cse362 = 2.4460185352936309e-10*cse260*cse361;
  const amrex::Real cse363 = cse261*cse354 + cse362;
  const amrex::Real cse364 = cse318*cse319;
  const amrex::Real cse365 = -cse314;
  const amrex::Real cse366 = 4.5579999999999993e-11*cse347;
  const amrex::Real cse367 = cse366*state.xn[8]/std::pow(cse321, 2);
  const amrex::Real cse368 = cse363*state.xn[8] + cse367*(4.7982242476653898*cse316*cse319*cse365*cse68 + 0.012496875897867835*cse364*cse68);
  const amrex::Real cse369 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse370 = cse330*cse331;
  const amrex::Real cse371 = -cse326;
  const amrex::Real cse372 = 2.1659999999999999e-12*cse347;
  const amrex::Real cse373 = cse372*state.xn[13]/std::pow(cse333, 2);
  const amrex::Real cse374 = cse369*state.xn[13] + cse373*(0.0067148253946876721*cse328*cse331*cse371*cse68 + 2.8762474947759312e-7*cse370*cse68);
  const amrex::Real cse375 = cse133 + cse135;
  const amrex::Real cse376 = 1.3007021411393064e-6*cse122;
  const amrex::Real cse377 = cse15*state.xn[2];
  const amrex::Real cse378 = cse377 < 1.0e-50;
  const amrex::Real cse379 = ((cse378) ? (
   0
)
: (
   -0.00011320754716981131*cse124*state.xn[3]
));
  const amrex::Real cse380 = ((cse378) ? (
   10000000000.0
)
: (
   cse115
));
  const amrex::Real cse381 = cse380 + 1;
  const amrex::Real cse382 = 1.0/cse381;
  const amrex::Real cse383 = cse236*state.xn[16];
  const amrex::Real cse384 = cse382*cse383;
  const amrex::Real cse385 = cse379*cse384;
  const amrex::Real cse386 = cse380*cse383/std::pow(cse381, 2);
  const amrex::Real cse387 = cse379*cse386;
  const amrex::Real cse388 = cse385 - cse387;
  const amrex::Real cse389 = cse121 + cse388;
  const amrex::Real cse390 = cse376*state.xn[16] + cse389;
  const amrex::Real cse391 = 2*cse67;
  const amrex::Real cse392 = crate*cse186*cse9;
  const amrex::Real cse393 = crate*cse15;
  const amrex::Real cse394 = 16800.0*cse393;
  const amrex::Real cse395 = 33600.0*cse325 + 4.3319999999999997e-12*cse335;
  const amrex::Real cse396 = -cse394*state.xn[12] + cse395;
  const amrex::Real cse397 = 3.2999999999999996e-13*cse175;
  const amrex::Real cse398 = cse397*state.xn[8];
  const amrex::Real cse399 = 1040.0*cse393;
  const amrex::Real cse400 = -cse399*state.xn[7];
  const amrex::Real cse401 = 2080.0*cse313 + 9.1159999999999986e-11*cse323;
  const amrex::Real cse402 = cse398 + cse400 + cse401;
  const amrex::Real cse403 = ((cse378) ? (
   0
)
: (
   cse114
));
  const amrex::Real cse404 = cse384*cse403;
  const amrex::Real cse405 = cse386*cse403;
  const amrex::Real cse406 = cse404 - cse405;
  const amrex::Real cse407 = cse181 + cse406;
  const amrex::Real cse408 = cse179 - cse180;
  const amrex::Real cse409 = 1.1144e-11*cse310;
  const amrex::Real cse410 = cse145 + cse409;
  const amrex::Real cse411 = 9.1999999999999993*cse299;
  const amrex::Real cse412 = 1.1000000000000001*crate;
  const amrex::Real cse413 = cse344*state.xn[2];
  const amrex::Real cse414 = cse309*cse348 + cse342*cse413;
  const amrex::Real cse415 = std::pow(cse3, 0.00231);
  const amrex::Real cse416 = std::pow(state.T, -1.5);
  const amrex::Real cse417 = std::exp(-8110.0*cse3);
  const amrex::Real cse418 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse419 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse420 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse421 = 0.00015100000000000001*cse417 + 1.31e-7*cse418 + 3.4e-8*cse419 + 6.9699999999999997e-9*cse420;
  const amrex::Real cse422 = 1.0537934368240927e-9*cse415 + cse416*cse421;
  const amrex::Real cse423 = cse422*state.xn[16];
  const amrex::Real cse424 = -cse423;
  const amrex::Real cse425 = 2.808e-22*ncol_H2;
  const amrex::Real cse426 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse425)*std::exp(-1.6000000000000001e-17*ncol_C)/(cse425 + 1) + 3.8500000000000001*crate + cse399*state.xn[3];
  const amrex::Real cse427 = cse397*state.xn[3];
  const amrex::Real cse428 = cse322*cse366 + cse363*state.xn[2];
  const amrex::Real cse429 = cse427 + cse428;
  const amrex::Real cse430 = cse380*cse382;
  const amrex::Real cse431 = cse430*state.xn[16];
  const amrex::Real cse432 = cse215*cse431;
  const amrex::Real cse433 = cse216 + cse432;
  const amrex::Real cse434 = -cse433;
  const amrex::Real cse435 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse436 = cse435*state.rn[2*0];
  const amrex::Real cse437 = cse394*state.xn[3] + cse436;
  const amrex::Real cse438 = cse334*cse372 + cse369*state.xn[2];
  const amrex::Real cse439 = -cse438;
  const amrex::Real cse440 = -cse105 - cse338;
  const amrex::Real cse441 = cse236*cse430;
  const amrex::Real cse442 = cse422*state.xn[7];
  const amrex::Real cse443 = cse376*state.xn[2];
  const amrex::Real cse444 = 13.799999999999999*cse299;
  const amrex::Real cse445 = 1.6715999999999999e-11*cse310;
  const amrex::Real cse446 = 3120.0*cse313 + 1.3673999999999998e-10*cse323;
  const amrex::Real cse447 = 50400.0*cse325 + 6.4979999999999992e-12*cse335;
  const amrex::Real cse448 = cse241 + cse242;
  const amrex::Real cse449 = -cse448;
  const amrex::Real cse450 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse415 + cse416*(1.2246100000000002*cse243*cse417 + 3.4846000000000001e-6*cse243*cse418 + 2.5908000000000001e-7*cse243*cse419 + 9.6185999999999982e-9*cse243*cse420) - 1.5*cse421*std::pow(state.T, -2.5));
  const amrex::Real cse451 = -cse450;
  const amrex::Real cse452 = 6.7636511339243933e-7*cse263;
  const amrex::Real cse453 = cse291*cse431;
  const amrex::Real cse454 = ((cse378) ? (
   0
)
: (
   5.6603773584905654e-5*cse273*state.xn[3]
));
  const amrex::Real cse455 = cse384*cse454;
  const amrex::Real cse456 = -cse455;
  const amrex::Real cse457 = cse386*cse454;
  const amrex::Real cse458 = 7913.1350000000002*cse243*cse356;
  const amrex::Real cse459 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse261*cse361 + cse261*(0.024885819999999999*cse243*cse351 + 7.2272340000000005e-7*cse243*cse352 + 7.7230820000000007e-8*cse243*cse353) - 3.0/2.0*cse354/std::pow(state.T, 5.0/2.0) + cse362*(-0.00035870183966449452*cse260*cse358/cse355 - cse458*std::log(cse355)) + cse362*(-6.1221939997287613*cse260*cse360/cse359 + cse458*std::log(cse359)));
  const amrex::Real cse460 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse461 = 4.9999999999999997e-12*cse343*cse416*state.xn[2];
  const amrex::Real cse462 = cse266*cse271;
  const amrex::Real cse463 = 7.5899999999999998e-12*cse462;
  const amrex::Real cse464 = cse413*state.xn[6]*(cse251*cse341 + cse339*(cse251*cse340 + 0.044330000000000001*cse252*cse74));
  const amrex::Real cse465 = cse373*(-1.4381237473879656e-7*cse3*cse370 - cse332*(0.071531250000000005*cse327*std::pow(state.T, -0.99999836499999994) + cse329*(cse283*cse371 - 7.538e-5*cse284)));
  const amrex::Real cse466 = cse349*(-5.3711836424859012e-7*cse3*cse345 - cse307*(0.0019963845000000001*cse302*std::pow(state.T, -0.99999960970000001) + cse304*(cse283*cse346 - 5.4939999999999998e-7*cse284)));
  const amrex::Real cse467 = cse367*(-0.0062484379489339175*cse3*cse364 - cse320*(20.983695000000001*cse315*std::pow(state.T, -0.95155000000000001) + cse317*(cse283*cse365 - 0.00013329999999999999*cse284)));
  const amrex::Real cse468 = cse435*state.xn[12];
  const amrex::Real cse469 = cse28*state.xn[3];
  const amrex::Real cse470 = cse300 + cse61;
  const amrex::Real cse471 = state.xn[0]*state.xn[3];
  const amrex::Real cse472 = cse471*cse53;
  const amrex::Real cse473 = -cse108 - cse472;
  const amrex::Real cse474 = -cse300;
  const amrex::Real cse475 = -cse108 - cse472 - cse474;
  const amrex::Real cse476 = 4.5427507615910105e-7*cse122;
  const amrex::Real cse477 = 5.6999999999999997e-11*cse173;
  const amrex::Real cse478 = cse169*cse471;
  const amrex::Real cse479 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse480 = cse148 + cse479;
  const amrex::Real cse481 = cse391 + cse392;
  const amrex::Real cse482 = cse153 + cse189 - cse404 + cse405;
  const amrex::Real cse483 = cse181 + cse408;
  const amrex::Real cse484 = cse196*cse471;
  const amrex::Real cse485 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse486 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse487 = cse234*cse471;
  const amrex::Real cse488 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse489 = cse237 + cse441 + cse442 + cse488;
  const amrex::Real cse490 = -cse246;
  const amrex::Real cse491 = cse258*cse29;
  const amrex::Real cse492 = cse262*cse471;
  const amrex::Real cse493 = -4.2899999999999994e-13*cse174*cse265*state.xn[3]*state.xn[8] + cse463;
  const amrex::Real cse494 = cse287 - cse289;
  const amrex::Real cse495 = -cse269 + cse270;
  const amrex::Real cse496 = cse292 + cse451 + cse453 + cse456 + cse457 + cse495;
  const amrex::Real cse497 = cse409 + cse480;
  const amrex::Real cse498 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse499 = cse205 + cse414 + cse485 + cse498;
  const amrex::Real cse500 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse501 = cse225 + cse311;
  const amrex::Real cse502 = -cse249;
  const amrex::Real cse503 = cse244 - cse461 + cse464 + cse466 + cse502;
  const amrex::Real cse504 = -cse311;
  const amrex::Real cse505 = 184.0*cse55;
  const amrex::Real cse506 = cse33*cse505;
  const amrex::Real cse507 = -cse506;
  const amrex::Real cse508 = cse324 + cse507;
  const amrex::Real cse509 = cse15*cse505;
  const amrex::Real cse510 = 9.8999999999999995e-14*cse175;
  const amrex::Real cse511 = cse33*cse55;
  const amrex::Real cse512 = 368.0*cse511;
  const amrex::Real cse513 = -cse512;
  const amrex::Real cse514 = cse401 + cse513;
  const amrex::Real cse515 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av) + 184.0*cse393*state.xn[3];
  const amrex::Real cse516 = cse222 + cse223;
  const amrex::Real cse517 = 552.0*cse511;
  const amrex::Real cse518 = cse459 + cse467;
  const amrex::Real cse519 = -cse324;
  const amrex::Real cse520 = 9.1500000000000005e-10*cse202;
  const amrex::Real cse521 = cse520*state.xn[15];
  const amrex::Real cse522 = cse520*state.xn[8];
  const amrex::Real cse523 = 1.1994963750000002e-8*cse248*state.xn[8];
  const amrex::Real cse524 = -cse523;
  const amrex::Real cse525 = cse506 + cse59;
  const amrex::Real cse526 = -cse509;
  const amrex::Real cse527 = cse145 + cse512;
  const amrex::Real cse528 = cse210 + cse486;
  const amrex::Real cse529 = cse500 + cse515;
  const amrex::Real cse530 = cse235 - cse488;
  const amrex::Real cse531 = -cse145;
  const amrex::Real cse532 = -cse63;
  const amrex::Real cse533 = -cse217 + cse432;
  const amrex::Real cse534 = -cse453 + cse455 - cse457;
  const amrex::Real cse535 = cse288 + cse296;
  const amrex::Real cse536 = -cse460 + cse465;
  const amrex::Real cse537 = -cse468;
  const amrex::Real cse538 = -cse336;
  const amrex::Real cse539 = 2.3099999999999997e-13*cse175;
  const amrex::Real cse540 = cse377 + 0.070000000000000007;
  const amrex::Real cse541 = 1.0/cse540;
  const amrex::Real cse542 = std::sqrt(cse377*cse541);
  const amrex::Real cse543 = 26.399999999999999*cse542;
  const amrex::Real cse544 = cse8*std::log(cse10);
  const amrex::Real cse545 = cse544 < 2;
  const amrex::Real cse546 = cse544 < 4;
  const amrex::Real cse547 = cse544 < 7;
  const amrex::Real cse548 = cse544 < 10;
  const amrex::Real cse549 = ((cse545) ? (
   10.0
)
: ((cse546) ? (
   (3.0/2.0)*cse544 + 7
)
: ((cse547) ? (
   (4.0/3.0)*cse544 + 23.0/3.0
)
: ((cse548) ? (
   (1.0/3.0)*cse544 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse550 = 2.0*cse15;
  const amrex::Real cse551 = cse549*cse550;
  const amrex::Real cse552 = cse543 + cse551*state.xn[3] + 6.5;
  const amrex::Real cse553 = 0.001*state.T;
  const amrex::Real cse554 = cse553 + 1;
  const amrex::Real cse555 = std::pow(cse554, -2);
  const amrex::Real cse556 = 1.0/cse554;
  const amrex::Real cse557 = std::pow(10, -2.3900000000000001*cse555 + 0.055500000000000001*cse556 - 11.06);
  const amrex::Real cse558 = std::pow(10, -2.0230000000000001*cse555 - 3.6709999999999998*cse556 - 11.08);
  const amrex::Real cse559 = cse15*cse558;
  const amrex::Real cse560 = cse559*state.xn[3];
  const amrex::Real cse561 = cse557 + cse560;
  const amrex::Real cse562 = 1.0/cse561;
  const amrex::Real cse563 = cse477 + 1.9999999999999999e-7;
  const amrex::Real cse564 = cse15*cse562*cse563 + 1;
  const amrex::Real cse565 = 1.0/cse564;
  const amrex::Real cse566 = state.T < 100.0;
  const amrex::Real cse567 = state.T < 6000.0;
  const amrex::Real cse568 = state.T < 500;
  const amrex::Real cse569 = state.T < 10;
  const amrex::Real cse570 = std::log(cse553);
  const amrex::Real cse571 = cse570*cse8;
  const amrex::Real cse572 = std::pow(cse7, -8);
  const amrex::Real cse573 = std::pow(cse570, 8)*cse572;
  const amrex::Real cse574 = std::pow(cse7, -7);
  const amrex::Real cse575 = std::pow(cse570, 7)*cse574;
  const amrex::Real cse576 = std::pow(cse7, -6);
  const amrex::Real cse577 = std::pow(cse570, 6)*cse576;
  const amrex::Real cse578 = std::pow(cse7, -5);
  const amrex::Real cse579 = std::pow(cse570, 5)*cse578;
  const amrex::Real cse580 = std::pow(cse7, -4);
  const amrex::Real cse581 = std::pow(cse570, 4)*cse580;
  const amrex::Real cse582 = std::pow(cse7, -3);
  const amrex::Real cse583 = std::pow(cse570, 3);
  const amrex::Real cse584 = cse582*cse583;
  const amrex::Real cse585 = cse12*std::pow(cse570, 2);
  const amrex::Real cse586 = std::pow(10.0, 16.815729999999999*cse571 + 70.609154000000004*cse573 + 364.14445999999998*cse575 + 801.81246999999996*cse577 + 983.67575999999997*cse579 + 734.71650999999997*cse581 + 343.1918*cse584 + 96.743155000000002*cse585 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse587 = state.xn[5] + state.xn[6];
  const amrex::Real cse588 = std::pow(10.0, 1.5714710999999999*cse571 + 0.32168730000000001*cse579 - 0.51002221000000003*cse581 - 0.23619984999999999*cse584 + 0.015391166*cse585 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse571 + 0.19191374999999999*cse579 - 0.16596184*cse581 + 0.29036281000000003*cse584 - 0.81520437999999995*cse585 - 23.689236999999999)*cse587 + std::pow(10.0, 2.0943374000000001*cse570*cse8 - 0.033638326000000003*cse579 - 0.14913216000000001*cse581 + 0.43693353000000001*cse582*cse583 - 0.77151435999999995*cse585 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse589 = std::pow(10, 37.383713*cse571 + 3.8479610000000002*cse579 + 20.159831000000001*cse581 + 48.656103000000002*cse584 + 58.145166000000003*cse585 - 16.818342000000001)*cse10 + cse588;
  const amrex::Real cse590 = std::pow(10.0, 1.6802758*cse571 - 6.3701156000000001*cse573 + 6.4380698000000001*cse575 + 8.9167182999999994*cse577 - 8.8077017000000009*cse579 - 4.7274035999999997*cse581 + 4.0406627000000004*cse584 + 0.93310621999999999*cse585 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse591 = state.T < 1000.0;
  const amrex::Real cse592 = std::pow(10.0, 3.5692468000000002*cse570*cse8 - 4.2519023000000002*cse579 - 21.328264000000001*cse581 - 27.850082*cse584 - 11.33286*cse585 - 24.311209000000002)*cse10 + cse588;
  const amrex::Real cse593 = std::pow(10.0, 4.6450521*cse571 + 1.5538288*cse579 - 5.5108049000000001*cse581 + 5.9369081000000001*cse584 - 3.7209846*cse585 - 24.311209000000002)*cse10 + cse588;
  const amrex::Real cse594 = std::pow(10.0, -11.945856385319289*cse12 + 28.69470131298657*cse578 - 56.79810800956944*cse580 + 34.150687525022562*cse582 + 8.3228160852326614*cse8 - 24.311209000000002)*cse10 + std::pow(10.0, -2.6171337683481011*cse12 + 3.5440891133599637*cse578 - 1.7105157385968939*cse580 + 1.6702447513374001*cse582 + 3.9225864834863131*cse8 - 23.689236999999999)*cse587 + std::pow(10.0, -2.4768712409536775*cse12 - 0.62120210234156448*cse578 - 1.5370575959566981*cse580 + 2.5133588394664677*cse582 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse12 + 5.9406293599919788*cse578 - 5.2566362076906961*cse580 - 1.3586848802337363*cse582 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse595 = ((cse566 && cse567 && cse568) ? (
   cse569 || cse586 + cse589 < 1.0e-100
)
: (
   ((cse566 && cse567) ? (
      cse569 || cse589 + cse590 < 1.0e-100
   )
   : (
      ((cse567 && cse568 && cse591) ? (
         cse569 || cse586 + cse592 < 1.0e-100
      )
      : (
         ((cse567 && cse591) ? (
            cse569 || cse590 + cse592 < 1.0e-100
         )
         : (
            ((cse567 && cse568) ? (
               cse569 || cse586 + cse593 < 1.0e-100
            )
            : (
               ((cse567) ? (
                  cse569 || cse590 + cse593 < 1.0e-100
               )
               : (
                  ((cse568) ? (
                     cse569 || std::pow(10.0, 310.58441786958315*cse12 + 7500.646099484944*cse572 + 21588.957206746247*cse574 + 26530.843406310876*cse576 + 18165.631967965237*cse578 + 7572.4886742758581*cse580 + 1974.131269262874*cse582 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse594 < 1.0e-100
                  )
                  : (
                     cse569 || std::pow(10.0, 2.9956460707652877*cse12 - 676.68255490510751*cse572 + 381.69251128040054*cse574 + 295.04162851880631*cse576 - 162.65264843551881*cse578 - 48.723840736518795*cse580 + 23.242975457496804*cse582 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse594 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse596 = std::log(0.001*((cse567) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse597 = cse596*cse8;
  const amrex::Real cse598 = std::pow(cse596, 5);
  const amrex::Real cse599 = cse578*cse598;
  const amrex::Real cse600 = std::pow(cse596, 4);
  const amrex::Real cse601 = cse580*cse600;
  const amrex::Real cse602 = std::pow(cse596, 3);
  const amrex::Real cse603 = cse582*cse602;
  const amrex::Real cse604 = std::pow(cse596, 2);
  const amrex::Real cse605 = cse12*cse604;
  const amrex::Real cse606 = std::pow(10, 37.383713*cse597 + 3.8479610000000002*cse599 + 20.159831000000001*cse601 + 48.656103000000002*cse603 + 58.145166000000003*cse605 - 16.818342000000001);
  const amrex::Real cse607 = ((cse567) ? (
   cse566
)
: (
   false
));
  const amrex::Real cse608 = std::pow(10.0, 3.5692468000000002*cse596*cse8 - 4.2519023000000002*cse599 - 21.328264000000001*cse601 - 27.850082*cse603 - 11.33286*cse605 - 24.311209000000002);
  const amrex::Real cse609 = ((cse567) ? (
   cse591
)
: (
   false
));
  const amrex::Real cse610 = std::pow(10.0, 4.6450521*cse597 + 1.5538288*cse599 - 5.5108049000000001*cse601 + 5.9369081000000001*cse603 - 3.7209846*cse605 - 24.311209000000002);
  const amrex::Real cse611 = ((cse607) ? (
   cse606
)
: ((cse609) ? (
   cse608
)
: (
   cse610
)));
  const amrex::Real cse612 = std::pow(10.0, 0.43693353000000001*cse582*cse602 + 2.0943374000000001*cse596*cse8 - 0.033638326000000003*cse599 - 0.14913216000000001*cse601 - 0.77151435999999995*cse605 - 23.962112000000001);
  const amrex::Real cse613 = cse612*state.xn[3];
  const amrex::Real cse614 = std::pow(10.0, 1.5714710999999999*cse597 + 0.32168730000000001*cse599 - 0.51002221000000003*cse601 - 0.23619984999999999*cse603 + 0.015391166*cse605 - 22.089523);
  const amrex::Real cse615 = cse614*state.xn[1];
  const amrex::Real cse616 = std::pow(10.0, 2.1892372*cse597 + 0.19191374999999999*cse599 - 0.16596184*cse601 + 0.29036281000000003*cse603 - 0.81520437999999995*cse605 - 23.689236999999999);
  const amrex::Real cse617 = cse587*cse616;
  const amrex::Real cse618 = cse572*std::pow(cse596, 8);
  const amrex::Real cse619 = std::pow(cse596, 7);
  const amrex::Real cse620 = cse574*cse619;
  const amrex::Real cse621 = std::pow(cse596, 6);
  const amrex::Real cse622 = cse576*cse621;
  const amrex::Real cse623 = std::pow(10.0, 16.815729999999999*cse597 + 983.67575999999997*cse599 + 734.71650999999997*cse601 + 343.1918*cse603 + 96.743155000000002*cse605 + 70.609154000000004*cse618 + 364.14445999999998*cse620 + 801.81246999999996*cse622 - 21.928795999999998);
  const amrex::Real cse624 = std::pow(10.0, 1.6802758*cse597 - 8.8077017000000009*cse599 - 4.7274035999999997*cse601 + 4.0406627000000004*cse603 + 0.93310621999999999*cse605 - 6.3701156000000001*cse618 + 6.4380698000000001*cse620 + 8.9167182999999994*cse622 - 22.921188999999998);
  const amrex::Real cse625 = ((cse568) ? (
   cse623
)
: (
   cse624
));
  const amrex::Real cse626 = cse10*cse611 + cse613 + cse615 + cse617 + cse625*state.xn[2];
  const amrex::Real cse627 = std::pow(cse626, -2);
  const amrex::Real cse628 = ((cse567) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse629 = std::exp(-11700.0*cse628);
  const amrex::Real cse630 = std::exp(-5860.0*cse628);
  const amrex::Real cse631 = std::exp(-510.0*cse628);
  const amrex::Real cse632 = std::exp(-2197000.0*((cse567) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse633 = cse632*((cse567) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse634 = 6.0142468035272636e-8*((cse567) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse635 = 1.0/cse634;
  const amrex::Real cse636 = 1.6e-18*cse629 + 6.7e-19*cse630 + 3.0e-24*cse631 + 4.985670872372847e-33*cse633*cse635;
  const amrex::Real cse637 = 1.0/cse626;
  const amrex::Real cse638 = cse636*cse637 + 1;
  const amrex::Real cse639 = std::pow(cse638, -2);
  const amrex::Real cse640 = cse627*std::pow(cse636, 2)*cse639;
  const amrex::Real cse641 = cse611*cse640;
  const amrex::Real cse642 = (1.0/100.0)*state.T;
  const amrex::Real cse643 = std::log(cse642);
  const amrex::Real cse644 = 0.0060000000000000001*cse643;
  const amrex::Real cse645 = std::pow(cse642, 0.36899999999999999 - cse644);
  const amrex::Real cse646 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse647 = 1.08e-14*cse646;
  const amrex::Real cse648 = 0.16*cse643;
  const amrex::Real cse649 = std::pow(cse642, 0.755 - cse648);
  const amrex::Real cse650 = 4.34e-10*cse649;
  const amrex::Real cse651 = cse10*cse650;
  const amrex::Real cse652 = 0.042999999999999997*cse643;
  const amrex::Real cse653 = 5.2499999999999998e-13*std::pow(cse642, cse652 + 0.88900000000000001);
  const amrex::Real cse654 = 0.52500000000000002*cse643;
  const amrex::Real cse655 = 2.2500000000000003e-12*std::pow(cse642, cse654 + 1.198);
  const amrex::Real cse656 = cse653 + cse655;
  const amrex::Real cse657 = cse647*state.xn[2] + cse651 + cse656*state.xn[3];
  const amrex::Real cse658 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse659 = cse657*cse658;
  const amrex::Real cse660 = cse645*cse659;
  const amrex::Real cse661 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse662 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse663 = 4.8599999999999998e-10*cse662;
  const amrex::Real cse664 = 3.1899999999999998e-10*cse645;
  const amrex::Real cse665 = cse10*cse664;
  const amrex::Real cse666 = 0.041000000000000002*cse643;
  const amrex::Real cse667 = 4.7499999999999998e-11*std::pow(cse642, cse666 + 0.20300000000000001);
  const amrex::Real cse668 = 0.058000000000000003*cse643;
  const amrex::Real cse669 = 1.6725000000000002e-10*std::pow(cse642, cse668 + 0.23699999999999999);
  const amrex::Real cse670 = cse667 + cse669;
  const amrex::Real cse671 = cse663*state.xn[2] + cse665 + cse670*state.xn[3];
  const amrex::Real cse672 = cse661*cse671;
  const amrex::Real cse673 = 0.0030000000000000001*cse643;
  const amrex::Real cse674 = std::pow(cse642, 0.41899999999999998 - cse673);
  const amrex::Real cse675 = 7.1400000000000007e-11*cse674;
  const amrex::Real cse676 = cse657*cse661;
  const amrex::Real cse677 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse678 = 5.1199999999999999e-10*cse677;
  const amrex::Real cse679 = 3.5700000000000001e-10*cse674;
  const amrex::Real cse680 = cse10*cse679;
  const amrex::Real cse681 = 0.025000000000000001*cse643;
  const amrex::Real cse682 = 3.7250000000000001e-11*std::pow(cse642, cse681 + 0.26400000000000001);
  const amrex::Real cse683 = 1.0274999999999999e-10*std::pow(cse642, cse652 + 0.29599999999999999);
  const amrex::Real cse684 = cse682 + cse683;
  const amrex::Real cse685 = cse678*state.xn[2] + cse680 + cse684*state.xn[3];
  const amrex::Real cse686 = cse661*cse685;
  const amrex::Real cse687 = 8.68e-11*cse649;
  const amrex::Real cse688 = cse658*cse671;
  const amrex::Real cse689 = cse649*cse688;
  const amrex::Real cse690 = cse685 + 8.9099999999999997e-5;
  const amrex::Real cse691 = cse645*cse690;
  const amrex::Real cse692 = 6.3800000000000002e-11*cse661;
  const amrex::Real cse693 = 2.1266666666666666e-11*cse660 + cse672*cse675 + cse675*cse676 + cse686*cse687 + 2.8933333333333332e-11*cse689 + cse691*cse692;
  const amrex::Real cse694 = cse657 + 1.7499999999999998e-5;
  const amrex::Real cse695 = cse671 + 1.34e-10;
  const amrex::Real cse696 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse697 = cse657*cse696;
  const amrex::Real cse698 = (1.0/3.0)*cse697;
  const amrex::Real cse699 = (1.0/5.0)*cse661;
  const amrex::Real cse700 = cse657*cse699;
  const amrex::Real cse701 = (1.0/15.0)*cse659;
  const amrex::Real cse702 = cse671*cse699;
  const amrex::Real cse703 = cse671*cse701 + cse685*cse700 + cse690*cse702;
  const amrex::Real cse704 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse705 = cse685*cse704;
  const amrex::Real cse706 = (3.0/5.0)*cse705;
  const amrex::Real cse707 = cse694*cse702 + cse694*cse706 + cse695*cse706;
  const amrex::Real cse708 = cse690*cse694 + cse690*cse695 + cse695*cse698 + cse703 + cse707;
  const amrex::Real cse709 = 1.0/cse708;
  const amrex::Real cse710 = 2.3872025442750958e-19*cse709;
  const amrex::Real cse711 = cse645*cse705;
  const amrex::Real cse712 = cse649*cse705;
  const amrex::Real cse713 = cse645*cse694;
  const amrex::Real cse714 = 2.1419999999999999e-10*cse674;
  const amrex::Real cse715 = cse695*cse704;
  const amrex::Real cse716 = cse694*cse704;
  const amrex::Real cse717 = cse672*cse687 + cse692*cse713 + 1.9139999999999998e-10*cse711 + 2.604e-10*cse712 + cse714*cse715 + cse714*cse716;
  const amrex::Real cse718 = 2.8023005140020001e-18*cse709;
  const amrex::Real cse719 = std::pow(cse708, -2);
  const amrex::Real cse720 = cse645*cse697;
  const amrex::Real cse721 = cse695*cse696;
  const amrex::Real cse722 = cse649*cse721;
  const amrex::Real cse723 = cse719*(-cse650*cse690 - cse664*cse690 - cse679*cse694 - cse679*cse695 - cse693 - cse717 - 1.0633333333333333e-10*cse720 - 1.4466666666666666e-10*cse722);
  const amrex::Real cse724 = 2.3872025442750958e-19*cse703;
  const amrex::Real cse725 = 2.8023005140020001e-18*cse707;
  const amrex::Real cse726 = 0.045999999999999999*cse643;
  const amrex::Real cse727 = std::pow(cse642, cse726 + 0.22800000000000001);
  const amrex::Real cse728 = std::pow(cse642, cse726 + 0.23100000000000001);
  const amrex::Real cse729 = 2.6400000000000002e-10*cse728;
  const amrex::Real cse730 = cse10*cse729;
  const amrex::Real cse731 = 0.031*cse643;
  const amrex::Real cse732 = 1.11e-10*std::pow(cse642, cse731 + 0.26300000000000001);
  const amrex::Real cse733 = 0.064000000000000001*cse643;
  const amrex::Real cse734 = 4.3749999999999998e-11*std::pow(cse642, cse733 + 0.071999999999999995);
  const amrex::Real cse735 = cse732 + cse734;
  const amrex::Real cse736 = 0.00235272*cse74;
  const amrex::Real cse737 = cse736 - 0.041816600000000002;
  const amrex::Real cse738 = cse737*cse74 + 0.35826400000000003;
  const amrex::Real cse739 = cse738*cse74 - 0.57443;
  const amrex::Real cse740 = 0.097857299999999994*cse74;
  const amrex::Real cse741 = cse740 - 3.1926800000000002;
  const amrex::Real cse742 = cse74*cse741 + 38.504899999999999;
  const amrex::Real cse743 = cse74*cse742 - 202.19300000000001;
  const amrex::Real cse744 = std::exp(((cse591) ? (
   cse739*cse74 - 7.4386999999999999
)
: (
   cse74*cse743 + 386.18599999999998
)));
  const amrex::Real cse745 = cse247*state.xn[2];
  const amrex::Real cse746 = 1.7258000000000001e-6*cse745;
  const amrex::Real cse747 = cse744*cse746;
  const amrex::Real cse748 = cse730 + cse735*state.xn[3] + cse747;
  const amrex::Real cse749 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse750 = cse748*cse749;
  const amrex::Real cse751 = cse727*cse750;
  const amrex::Real cse752 = 0.057000000000000002*cse643;
  const amrex::Real cse753 = std::pow(cse642, cse752 + 0.115);
  const amrex::Real cse754 = 1.26e-10*cse753;
  const amrex::Real cse755 = cse10*cse754;
  const amrex::Real cse756 = 0.049000000000000002*cse643;
  const amrex::Real cse757 = 5.3249999999999997e-11*std::pow(cse642, cse756 - 0.0040000000000000001);
  const amrex::Real cse758 = 0.10199999999999999*cse643;
  const amrex::Real cse759 = 1.675e-11*std::pow(cse642, cse758 - 0.085000000000000006);
  const amrex::Real cse760 = cse757 + cse759;
  const amrex::Real cse761 = 0.00065632500000000003*cse74;
  const amrex::Real cse762 = -cse761 - 0.015089200000000001;
  const amrex::Real cse763 = cse74*cse762 + 0.36118400000000001;
  const amrex::Real cse764 = cse74*cse763 - 0.77378199999999997;
  const amrex::Real cse765 = 0.10508000000000001*cse74;
  const amrex::Real cse766 = cse765 - 3.4762;
  const amrex::Real cse767 = cse74*cse766 + 42.594999999999999;
  const amrex::Real cse768 = cse74*cse767 - 227.91300000000001;
  const amrex::Real cse769 = std::exp(((cse591) ? (
   cse74*cse764 - 9.2514099999999999
)
: (
   cse74*cse768 + 444.60000000000002
)));
  const amrex::Real cse770 = 2.8763333333333333e-6*cse769;
  const amrex::Real cse771 = cse745*cse770;
  const amrex::Real cse772 = cse755 + cse760*state.xn[3] + cse771;
  const amrex::Real cse773 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse774 = cse728*cse773;
  const amrex::Real cse775 = 1.32e-9*cse774;
  const amrex::Real cse776 = 8.9000000000000003e-11*cse727;
  const amrex::Real cse777 = cse10*cse776;
  const amrex::Real cse778 = 0.037999999999999999*cse643;
  const amrex::Real cse779 = 5.1749999999999996e-11*std::pow(cse642, cse778 + 0.16900000000000001);
  const amrex::Real cse780 = 0.048000000000000001*cse643;
  const amrex::Real cse781 = 2.15e-11*std::pow(cse642, cse780 - 0.01);
  const amrex::Real cse782 = cse779 + cse781;
  const amrex::Real cse783 = 0.0070527699999999999*cse74;
  const amrex::Real cse784 = cse783 - 0.11133800000000001;
  const amrex::Real cse785 = cse74*cse784 + 0.69763799999999998;
  const amrex::Real cse786 = cse74*cse785 - 1.3074300000000001;
  const amrex::Real cse787 = 0.093813800000000003*cse74;
  const amrex::Real cse788 = cse787 - 3.0328300000000001;
  const amrex::Real cse789 = cse74*cse788 + 36.180300000000003;
  const amrex::Real cse790 = cse74*cse789 - 187.47399999999999;
  const amrex::Real cse791 = std::exp(((cse591) ? (
   cse74*cse786 - 7.6973500000000001
)
: (
   cse74*cse790 + 350.60899999999998
)));
  const amrex::Real cse792 = cse746*cse791;
  const amrex::Real cse793 = cse777 + cse782*state.xn[3] + cse792;
  const amrex::Real cse794 = cse749*cse793;
  const amrex::Real cse795 = cse728*cse794;
  const amrex::Real cse796 = cse753*cse773;
  const amrex::Real cse797 = 6.3e-10*cse796;
  const amrex::Real cse798 = cse772 + 7.8800000000000004e-8;
  const amrex::Real cse799 = cse727*cse798;
  const amrex::Real cse800 = 4.4500000000000001e-10*cse773;
  const amrex::Real cse801 = cse748*cse797 + 7.4166666666666679e-10*cse751 + cse772*cse775 + cse793*cse797 + 2.2000000000000003e-9*cse795 + cse799*cse800;
  const amrex::Real cse802 = cse748 + 2.65e-7;
  const amrex::Real cse803 = cse793 + 1.81e-14;
  const amrex::Real cse804 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse805 = cse748*cse804;
  const amrex::Real cse806 = (5.0/3.0)*cse805;
  const amrex::Real cse807 = 5*cse773;
  const amrex::Real cse808 = cse748*cse807;
  const amrex::Real cse809 = (25.0/3.0)*cse748;
  const amrex::Real cse810 = cse793*cse807;
  const amrex::Real cse811 = cse772*cse808 + cse794*cse809 + cse798*cse810;
  const amrex::Real cse812 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse813 = cse772*cse812;
  const amrex::Real cse814 = 3*cse813;
  const amrex::Real cse815 = cse802*cse810 + cse802*cse814 + cse803*cse814;
  const amrex::Real cse816 = cse798*cse802 + cse798*cse803 + cse803*cse806 + cse811 + cse815;
  const amrex::Real cse817 = 1.0/cse816;
  const amrex::Real cse818 = 1.4210469458259591e-21*cse817;
  const amrex::Real cse819 = cse727*cse813;
  const amrex::Real cse820 = 7.9200000000000006e-10*cse728;
  const amrex::Real cse821 = cse727*cse802;
  const amrex::Real cse822 = 3.7800000000000004e-10*cse753;
  const amrex::Real cse823 = cse802*cse822;
  const amrex::Real cse824 = cse803*cse822;
  const amrex::Real cse825 = cse775*cse793 + cse800*cse821 + cse812*cse823 + cse812*cse824 + cse813*cse820 + 2.6700000000000001e-10*cse819;
  const amrex::Real cse826 = 2.5697412351439998e-22*cse817;
  const amrex::Real cse827 = std::pow(cse816, -2);
  const amrex::Real cse828 = cse727*cse805;
  const amrex::Real cse829 = cse803*cse804;
  const amrex::Real cse830 = cse728*cse829;
  const amrex::Real cse831 = cse827*(-cse729*cse798 - cse754*cse802 - cse754*cse803 - cse776*cse798 - cse801 - cse825 - 1.4833333333333334e-10*cse828 - 4.4000000000000003e-10*cse830);
  const amrex::Real cse832 = 1.4210469458259591e-21*cse811;
  const amrex::Real cse833 = 2.5697412351439998e-22*cse815;
  const amrex::Real cse834 = state.T < 1000000.0;
  const amrex::Real cse835 = 100.0*d2g;
  const amrex::Real cse836 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse837 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse71 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse838 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse839 = cse347*cse838;
  const amrex::Real cse840 = -cse835*cse836 + cse837*cse839;
  const amrex::Real cse841 = (((cse840 < 0) ? (
   0
)
: ((cse840 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse842 = cse841*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse843 = cse32*state.xn[2];
  const amrex::Real cse844 = std::pow(cse540, -2);
  const amrex::Real cse845 = 0.5*cse844;
  const amrex::Real cse846 = std::pow(cse10, -3);
  const amrex::Real cse847 = cse112*cse846;
  const amrex::Real cse848 = cse10*cse540*cse543*cse68;
  const amrex::Real cse849 = cse15*cse8;
  const amrex::Real cse850 = cse550*state.xn[3];
  const amrex::Real cse851 = cse33*cse549;
  const amrex::Real cse852 = cse848*(-0.5*cse541*cse843 + cse845*cse847) + cse850*((cse545) ? (
   0
)
: ((cse546) ? (
   (3.0/2.0)*cse849
)
: ((cse547) ? (
   (4.0/3.0)*cse849
)
: ((cse548) ? (
   (1.0/3.0)*cse849
)
: (
   0
))))) - 2.0*cse851;
  const amrex::Real cse853 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse854 = 1.75*cse853;
  const amrex::Real cse855 = std::exp(-state.T);
  const amrex::Real cse856 = 1 - cse855;
  const amrex::Real cse857 = std::pow(cse856, 1000);
  const amrex::Real cse858 = l0_co_interp1d(state.T);
  const amrex::Real cse859 = 1.0/cse858;
  const amrex::Real cse860 = 680.10000000000002/cse853;
  const amrex::Real cse861 = cse10*cse854 + cse860*state.xn[2] + state.xn[3];
  const amrex::Real cse862 = M_SQRT2;
  const amrex::Real cse863 = std::max(gradv, 5.7445531974402559e-18*cse2*cse862);
  const amrex::Real cse864 = 100000.0/cse863;
  const amrex::Real cse865 = ncol_H <= 0;
  const amrex::Real cse866 = cse15*ncol_H;
  const amrex::Real cse867 = std::max(1.0000000000000001e-30, cse866);
  const amrex::Real cse868 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse869 = ((cse865) ? (
   cse864*state.xn[9]
)
: (
   cse868/std::sqrt(cse867)
));
  const amrex::Real cse870 = llte_co_interp2d(state.T, cse869);
  const amrex::Real cse871 = 1.0/cse870;
  const amrex::Real cse872 = nhalf_co_interp2d(state.T, cse869);
  const amrex::Real cse873 = cse858*cse871;
  const amrex::Real cse874 = -cse872*cse873 + 1;
  const amrex::Real cse875 = 1.0/cse872;
  const amrex::Real cse876 = cse861*cse875;
  const amrex::Real cse877 = alpha_co_interp2d(state.T, cse869);
  const amrex::Real cse878 = std::pow(cse876, cse877);
  const amrex::Real cse879 = cse859*cse878;
  const amrex::Real cse880 = cse874*cse879;
  const amrex::Real cse881 = cse859 + cse861*cse871 + cse880;
  const amrex::Real cse882 = 1.0/cse881;
  const amrex::Real cse883 = cse857*cse882;
  const amrex::Real cse884 = cse883*state.xn[9];
  const amrex::Real cse885 = (((cse840 > 0) ? (
   0
)
: ((cse840 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse886 = cse837*cse885*d2g;
  const amrex::Real cse887 = state.xn[2] + 1.0e-50;
  const amrex::Real cse888 = 1.0/cse887;
  const amrex::Real cse889 = cse2*cse71;
  const amrex::Real cse890 = 1.7*cse889;
  const amrex::Real cse891 = cse888*cse890 + 50;
  const amrex::Real cse892 = std::log(cse891);
  const amrex::Real cse893 = 0.45350000000000001 + 2.234/cse892;
  const amrex::Real cse894 = std::pow(state.T, cse893);
  const amrex::Real cse895 = 1.4419999999999999 - 0.050889999999999998*cse892;
  const amrex::Real cse896 = std::exp(cse892*cse895);
  const amrex::Real cse897 = 1.8998126555754159e-29*cse894*cse896;
  const amrex::Real cse898 = d2g*state.xn[2];
  const amrex::Real cse899 = std::pow(state.xn[0], 2);
  const amrex::Real cse900 = std::pow(cse564, -2);
  const amrex::Real cse901 = cse563/std::pow(cse561, 2);
  const amrex::Real cse902 = cse558*cse846*cse901;
  const amrex::Real cse903 = cse902*state.xn[3];
  const amrex::Real cse904 = cse32*cse562*cse563 - cse903;
  const amrex::Real cse905 = 1.0/cse66;
  const amrex::Real cse906 = std::pow(cse870, -2);
  const amrex::Real cse907 = llte_co_interp2d_partial_1(state.T, cse869);
  const amrex::Real cse908 = cse32*ncol_H*(((cse866 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse866 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse867, 3.0/2.0);
  const amrex::Real cse909 = cse908*ncol_CO;
  const amrex::Real cse910 = ((cse865) ? (
   0
)
: (
   878350727.21550155*cse909
));
  const amrex::Real cse911 = nhalf_co_interp2d_partial_1(state.T, cse869);
  const amrex::Real cse912 = cse910*cse911;
  const amrex::Real cse913 = alpha_co_interp2d_partial_1(state.T, cse869);
  const amrex::Real cse914 = std::log(cse876);
  const amrex::Real cse915 = cse913*cse914;
  const amrex::Real cse916 = cse861/std::pow(cse872, 2);
  const amrex::Real cse917 = cse877/cse861;
  const amrex::Real cse918 = cse872*cse917;
  const amrex::Real cse919 = cse861*state.xn[9];
  const amrex::Real cse920 = cse857*cse919/std::pow(cse881, 2);
  const amrex::Real cse921 = 0.0086999999999999994*cse643;
  const amrex::Real cse922 = std::pow(cse642, cse921 + 0.12809999999999999);
  const amrex::Real cse923 = 7.5799999999999997e-10*cse10*cse922;
  const amrex::Real cse924 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse925 = 0.75*((cse568) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse924
)) + 0.25*((cse568) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse924
));
  const amrex::Real cse926 = 4.5299999999999998e-6*cse745 + cse923 + cse925*state.xn[3];
  const amrex::Real cse927 = cse926 + 2.3e-6;
  const amrex::Real cse928 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse929 = cse928/cse927;
  const amrex::Real cse930 = cse922*cse929;
  const amrex::Real cse931 = 2*cse929;
  const amrex::Real cse932 = cse926*cse931 + 1;
  const amrex::Real cse933 = 1.0/cse932;
  const amrex::Real cse934 = cse933*state.xn[8];
  const amrex::Real cse935 = cse930*cse934;
  const amrex::Real cse936 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse937 = cse73 + 50;
  const amrex::Real cse938 = 0.0043*std::pow(cse937, 0.69199999999999995) + 1;
  const amrex::Real cse939 = 0.049959999999999997*std::pow(cse937, 0.43099999999999999)*cse938 + 1;
  const amrex::Real cse940 = 1.0/cse939;
  const amrex::Real cse941 = 1.5159999999999999e-9*cse922;
  const amrex::Real cse942 = std::pow(cse927, -2);
  const amrex::Real cse943 = cse926*cse928*cse942;
  const amrex::Real cse944 = cse926*cse929;
  const amrex::Real cse945 = 5.7927337833400002e-20*cse944*state.xn[8]/std::pow(cse932, 2);
  const amrex::Real cse946 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse947 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse948 = 1.0/cse947;
  const amrex::Real cse949 = cse948*state.xn[2];
  const amrex::Real cse950 = cse946*cse949;
  const amrex::Real cse951 = 4.4460389837719569e-9*cse950 + 626500000.0;
  const amrex::Real cse952 = 1.0/cse951;
  const amrex::Real cse953 = std::exp(-118400.0*cse3);
  const amrex::Real cse954 = cse952*cse953;
  const amrex::Real cse955 = 1.3338116951315871e-8*cse954;
  const amrex::Real cse956 = cse950*cse955 + 1;
  const amrex::Real cse957 = 1.0/cse956;
  const amrex::Real cse958 = 1.3659996292638161e-10*cse954*cse957;
  const amrex::Real cse959 = std::pow(cse66, -2);
  const amrex::Real cse960 = -1.461185424e-21*chi*cse171*cse172*cse900*cse904*state.xn[3] - 1.7e-24*chi*cse70*cse936*cse940*d2g - 4.0535078100000002e-12*crate*cse32*cse552*cse959*state.xn[3]*state.xn[5] - 1.7623947000000002e-12*crate*cse852*cse905*state.xn[5] - 1.602177e-12*crate*cse852*state.xn[0] - 3.2043540000000001e-12*crate*cse852*state.xn[3] + 7.1777529600000012e-12*cse108 + 7.1777529600000012e-12*cse472 + cse835*cse842 + cse838*cse886 + cse854*cse884 + cse897*cse898 - 2.0187430200000004e-26*cse899*cse900*cse904*d2g + cse920*(-cse854*cse871 + cse861*cse906*cse907*cse910 - cse879*(cse858*cse872*cse906*cse907*cse910 - cse873*cse912) - cse880*(cse910*cse915 + cse918*(cse854*cse875 - cse912*cse916))) - 4.39089220777172e-29*cse922*cse926*cse928*cse933*cse942*state.xn[8] + 4.39089220777172e-29*cse935 + cse945*(-cse929*cse941 + cse941*cse943) + cse950*cse958 + state.xn[11]*(cse693*cse710 + cse717*cse718 + cse723*cse724 + cse723*cse725) + state.xn[7]*((cse834) ? (
   cse801*cse818 + cse825*cse826 + cse831*cse832 + cse831*cse833
)
: (
   0
));
  const amrex::Real cse961 = cse960 + state.xn[3]*((cse595) ? (
   0
)
: (
   cse641
));
  const amrex::Real cse962 = cse347*cse897;
  const amrex::Real cse963 = cse946*cse948;
  const amrex::Real cse964 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse965 = std::pow(cse947, -2);
  const amrex::Real cse966 = std::pow(cse951, -2);
  const amrex::Real cse967 = cse894*cse896;
  const amrex::Real cse968 = 1.0/cse891;
  const amrex::Real cse969 = std::pow(cse887, -2);
  const amrex::Real cse970 = cse968*cse969;
  const amrex::Real cse971 = cse74/std::pow(cse892, 2);
  const amrex::Real cse972 = std::pow(cse939, -2);
  const amrex::Real cse973 = cse71*std::pow(cse937, 0.12299999999999994);
  const amrex::Real cse974 = cse71*std::pow(cse937, -0.56899999999999995)*cse938;
  const amrex::Real cse975 = cse950*cse954;
  const amrex::Real cse976 = 1.3659996292638161e-10*cse10*cse975/std::pow(cse956, 2);
  const amrex::Real cse977 = 0.5*cse15*cse541 - cse843*cse845;
  const amrex::Real cse978 = cse962*state.xn[2];
  const amrex::Real cse979 = 9.0599999999999997e-6*cse247;
  const amrex::Real cse980 = 1.024e-10*cse677;
  const amrex::Real cse981 = 2.1600000000000004e-15*cse646;
  const amrex::Real cse982 = 9.7199999999999998e-11*cse661*cse662;
  const amrex::Real cse983 = 7.1999999999999997e-16*cse646*cse688 + 3.2399999999999999e-11*cse659*cse662 + cse672*cse980 + cse676*cse980 + cse686*cse981 + cse690*cse982;
  const amrex::Real cse984 = 3.0719999999999998e-10*cse677;
  const amrex::Real cse985 = 6.4799999999999996e-15*cse646*cse705 + 2.9159999999999995e-10*cse662*cse705 + cse672*cse981 + cse694*cse982 + cse715*cse984 + cse716*cse984;
  const amrex::Real cse986 = cse719*(-3.6000000000000001e-15*cse646*cse721 - cse647*cse690 - 1.6199999999999998e-10*cse662*cse697 - cse663*cse690 - cse678*cse694 - cse678*cse695 - cse983 - cse985);
  const amrex::Real cse987 = cse247*cse744;
  const amrex::Real cse988 = 8.6289999999999999e-6*cse773;
  const amrex::Real cse989 = cse987*cse988;
  const amrex::Real cse990 = cse247*cse791;
  const amrex::Real cse991 = cse748*cse773;
  const amrex::Real cse992 = cse247*cse769;
  const amrex::Real cse993 = 1.4381666666666666e-5*cse992;
  const amrex::Real cse994 = cse773*cse793;
  const amrex::Real cse995 = cse988*cse990;
  const amrex::Real cse996 = 1.4381666666666668e-5*cse750*cse990 + cse772*cse989 + 1.4381666666666668e-5*cse794*cse987 + cse798*cse995 + cse991*cse993 + cse993*cse994;
  const amrex::Real cse997 = 5.1774000000000006e-6*cse813;
  const amrex::Real cse998 = 8.6289999999999999e-6*cse812*cse992;
  const amrex::Real cse999 = cse793*cse989 + cse802*cse995 + cse802*cse998 + cse803*cse998 + cse987*cse997 + cse990*cse997;
  const amrex::Real cse1000 = cse247*cse770;
  const amrex::Real cse1001 = 1.7258000000000001e-6*cse798;
  const amrex::Real cse1002 = cse827*(-cse1000*cse802 - cse1000*cse803 - cse1001*cse987 - cse1001*cse990 - 2.8763333333333337e-6*cse805*cse990 - 2.8763333333333337e-6*cse829*cse987 - cse996 - cse999);
  const amrex::Real cse1003 = 6.3799999999999997e-10*cse645;
  const amrex::Real cse1004 = cse1003 + cse670;
  const amrex::Real cse1005 = 7.1400000000000002e-10*cse674;
  const amrex::Real cse1006 = cse1005 + cse684;
  const amrex::Real cse1007 = 8.68e-10*cse649;
  const amrex::Real cse1008 = cse1007 + cse656;
  const amrex::Real cse1009 = cse685*cse699;
  const amrex::Real cse1010 = (1.0/15.0)*cse688;
  const amrex::Real cse1011 = cse1004*cse690;
  const amrex::Real cse1012 = cse1004*cse701 + cse1006*cse700 + cse1006*cse702 + cse1008*cse1009 + cse1008*cse1010 + cse1011*cse699;
  const amrex::Real cse1013 = cse694*cse699;
  const amrex::Real cse1014 = cse1006*cse695;
  const amrex::Real cse1015 = (3.0/5.0)*cse704;
  const amrex::Real cse1016 = cse1006*cse694;
  const amrex::Real cse1017 = cse1004*cse1013 + cse1004*cse706 + cse1008*cse702 + cse1008*cse706 + cse1014*cse1015 + cse1015*cse1016;
  const amrex::Real cse1018 = (1.0/3.0)*cse721;
  const amrex::Real cse1019 = cse719*(-cse1004*cse698 - cse1008*cse1018 - cse1008*cse690 - cse1011 - cse1012 - cse1014 - cse1016 - cse1017);
  const amrex::Real cse1020 = 2*cse611;
  const amrex::Real cse1021 = 5.2800000000000004e-10*cse728;
  const amrex::Real cse1022 = cse1021 + cse735;
  const amrex::Real cse1023 = cse772*cse807;
  const amrex::Real cse1024 = (25.0/3.0)*cse794;
  const amrex::Real cse1025 = 1.7800000000000001e-10*cse727;
  const amrex::Real cse1026 = cse1025 + cse782;
  const amrex::Real cse1027 = cse749*cse809;
  const amrex::Real cse1028 = 2.5200000000000001e-10*cse753;
  const amrex::Real cse1029 = cse1028 + cse760;
  const amrex::Real cse1030 = cse1026*cse798;
  const amrex::Real cse1031 = cse1022*cse1023 + cse1022*cse1024 + cse1026*cse1027 + cse1029*cse808 + cse1029*cse810 + cse1030*cse807;
  const amrex::Real cse1032 = cse802*cse807;
  const amrex::Real cse1033 = cse1029*cse802;
  const amrex::Real cse1034 = 3*cse812;
  const amrex::Real cse1035 = cse1029*cse803;
  const amrex::Real cse1036 = cse1022*cse810 + cse1022*cse814 + cse1026*cse1032 + cse1026*cse814 + cse1033*cse1034 + cse1034*cse1035;
  const amrex::Real cse1037 = (5.0/3.0)*cse829;
  const amrex::Real cse1038 = cse827*(-cse1022*cse1037 - cse1022*cse798 - cse1026*cse806 - cse1030 - cse1031 - cse1033 - cse1035 - cse1036);
  const amrex::Real cse1039 = cse541*cse843;
  const amrex::Real cse1040 = cse844*cse847;
  const amrex::Real cse1041 = cse848*(-1.0*cse1039 + 1.0*cse1040) + cse850*((cse545) ? (
   0
)
: ((cse546) ? (
   3*cse849
)
: ((cse547) ? (
   (8.0/3.0)*cse849
)
: ((cse548) ? (
   (2.0/3.0)*cse849
)
: (
   0
))))) - 4.0*cse851;
  const amrex::Real cse1042 = cse1041 + cse551;
  const amrex::Real cse1043 = 3.5*cse853;
  const amrex::Real cse1044 = cse1043 + 1;
  const amrex::Real cse1045 = -2*cse32*cse562*cse563;
  const amrex::Real cse1046 = cse15*cse901;
  const amrex::Real cse1047 = -cse1045 - cse1046*(cse158*cse558 - cse559);
  const amrex::Real cse1048 = ((cse865) ? (
   0
)
: (
   cse868*cse908
));
  const amrex::Real cse1049 = cse1048*cse915;
  const amrex::Real cse1050 = cse1048*cse911;
  const amrex::Real cse1051 = -cse1050*cse916;
  const amrex::Real cse1052 = -cse1048*cse861*cse906*cse907 + cse879*(cse1048*cse858*cse872*cse906*cse907 - cse1050*cse873);
  const amrex::Real cse1053 = cse925 + cse941;
  const amrex::Real cse1054 = 5.7927337833400002e-20*cse933;
  const amrex::Real cse1055 = cse1054*state.xn[8];
  const amrex::Real cse1056 = cse1055*cse929;
  const amrex::Real cse1057 = -cse1053;
  const amrex::Real cse1058 = cse1055*cse943;
  const amrex::Real cse1059 = 2*cse943;
  const amrex::Real cse1060 = 1.0/cse638;
  const amrex::Real cse1061 = cse842*d2g;
  const amrex::Real cse1062 = cse2*cse886;
  const amrex::Real cse1063 = cse898*cse967;
  const amrex::Real cse1064 = cse957*cse975;
  const amrex::Real cse1065 = -3.3999999999999999e-24*chi*cse70*cse936*cse940*d2g + 200.0*cse1061 + 6.4000000000000004e-32*cse1062 + 3.7996253111508318e-29*cse1063 + 2.7319992585276321e-10*cse1064;
  const amrex::Real cse1066 = 1.4280000000000001e-10*cse674;
  const amrex::Real cse1067 = 1.736e-10*cse649;
  const amrex::Real cse1068 = 1.276e-10*cse661;
  const amrex::Real cse1069 = cse1066*cse672 + cse1066*cse676 + cse1067*cse686 + cse1068*cse691 + 4.2533333333333333e-11*cse660 + 5.7866666666666665e-11*cse689;
  const amrex::Real cse1070 = 4.2839999999999999e-10*cse674;
  const amrex::Real cse1071 = cse1067*cse672 + cse1068*cse713 + cse1070*cse715 + cse1070*cse716 + 3.8279999999999996e-10*cse711 + 5.208e-10*cse712;
  const amrex::Real cse1072 = cse719*(-cse1003*cse690 - cse1005*cse694 - cse1005*cse695 - cse1007*cse690 - cse1069 - cse1071 - 2.1266666666666666e-10*cse720 - 2.8933333333333332e-10*cse722);
  const amrex::Real cse1073 = 2.64e-9*cse774;
  const amrex::Real cse1074 = 1.26e-9*cse796;
  const amrex::Real cse1075 = 8.9000000000000003e-10*cse773;
  const amrex::Real cse1076 = cse1073*cse772 + cse1074*cse748 + cse1074*cse793 + cse1075*cse799 + 1.4833333333333336e-9*cse751 + 4.4000000000000005e-9*cse795;
  const amrex::Real cse1077 = cse728*cse813;
  const amrex::Real cse1078 = cse753*cse812;
  const amrex::Real cse1079 = 7.5600000000000008e-10*cse1078;
  const amrex::Real cse1080 = cse1073*cse793 + cse1075*cse821 + 1.5840000000000001e-9*cse1077 + cse1079*cse802 + cse1079*cse803 + 5.3400000000000002e-10*cse819;
  const amrex::Real cse1081 = cse827*(-cse1021*cse798 - cse1025*cse798 - cse1028*cse802 - cse1028*cse803 - cse1076 - cse1080 - 2.9666666666666668e-10*cse828 - 8.8000000000000006e-10*cse830);
  const amrex::Real cse1082 = -cse1045 - cse9*cse902;
  const amrex::Real cse1083 = cse922*cse943;
  const amrex::Real cse1084 = -state.xn[3]*((cse595) ? (
   0
)
: (
   cse616*cse640
));
  const amrex::Real cse1085 = ((cse865) ? (
   cse864
)
: (
   0
));
  const amrex::Real cse1086 = cse1085*cse911;
  const amrex::Real cse1087 = -cse961;
  const amrex::Real cse1088 = 5.25*cse853;
  const amrex::Real cse1089 = 3*cse32*cse562*cse563 - 3*cse903;
  const amrex::Real cse1090 = cse848*(-1.5*cse1039 + 1.5*cse1040) + cse850*((cse545) ? (
   0
)
: ((cse546) ? (
   (9.0/2.0)*cse849
)
: ((cse547) ? (
   4*cse849
)
: ((cse548) ? (
   cse849
)
: (
   0
))))) - 6.0*cse851;
  const amrex::Real cse1091 = ((cse865) ? (
   0
)
: (
   2635052181.6465044*cse909
));
  const amrex::Real cse1092 = cse1091*cse911;
  const amrex::Real cse1093 = 2.1420000000000005e-10*cse674;
  const amrex::Real cse1094 = 2.604e-10*cse649;
  const amrex::Real cse1095 = 1.9139999999999998e-10*cse661;
  const amrex::Real cse1096 = cse1093*cse672 + cse1093*cse676 + cse1094*cse686 + cse1095*cse691 + 6.3799999999999989e-11*cse660 + cse687*cse688;
  const amrex::Real cse1097 = 6.4260000000000004e-10*cse674;
  const amrex::Real cse1098 = cse1094*cse672 + cse1095*cse713 + cse1097*cse715 + cse1097*cse716 + 5.7419999999999994e-10*cse711 + 7.8120000000000005e-10*cse712;
  const amrex::Real cse1099 = 1.0710000000000001e-9*cse674;
  const amrex::Real cse1100 = cse719*(-cse1096 - cse1098 - cse1099*cse694 - cse1099*cse695 - 1.3020000000000001e-9*cse649*cse690 - cse650*cse721 - 9.569999999999999e-10*cse691 - 3.1899999999999993e-10*cse720);
  const amrex::Real cse1101 = 3.9600000000000004e-9*cse774;
  const amrex::Real cse1102 = 1.8900000000000003e-9*cse796;
  const amrex::Real cse1103 = 1.335e-9*cse773;
  const amrex::Real cse1104 = cse1101*cse772 + cse1102*cse748 + cse1102*cse793 + cse1103*cse799 + 2.2250000000000001e-9*cse751 + 6.6000000000000012e-9*cse795;
  const amrex::Real cse1105 = 1.134e-9*cse1078;
  const amrex::Real cse1106 = 2.3760000000000001e-9*cse1077 + cse1101*cse793 + cse1103*cse821 + cse1105*cse802 + cse1105*cse803 + 8.0100000000000003e-10*cse819;
  const amrex::Real cse1107 = cse827*(-cse1104 - cse1106 - cse798*cse820 - 2.6700000000000001e-10*cse799 - cse823 - cse824 - 4.4500000000000001e-10*cse828 - 1.3200000000000002e-9*cse830);
  const amrex::Real cse1108 = cse949*cse954*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1109 = cse10*cse957;
  const amrex::Real cse1110 = cse949*std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1111 = cse1110*cse954;
  const amrex::Real cse1112 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1113 = cse950*cse953*cse966*(-6.6690584756579352e-10*cse1110 + 2.5500935934394033e-12*cse1112*cse965*state.xn[2]);
  const amrex::Real cse1114 = cse260*cse71*cse888*cse968;
  const amrex::Real cse1115 = cse243*cse944;
  const amrex::Real cse1116 = cse3*cse745;
  const amrex::Real cse1117 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1118 = 100*cse3;
  const amrex::Real cse1119 = -2.2649999999999999e-6*cse1116 + cse923*(cse1118*(8.7000000000000001e-5*cse643 + 0.001281) + cse3*cse921) + state.xn[3]*(0.75*((cse568) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1117
)) + 0.25*((cse568) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1117
)));
  const amrex::Real cse1120 = std::pow(cse554, -3);
  const amrex::Real cse1121 = cse1046*(-cse557*cse7*(0.0047800000000000004*cse1120 - 5.5500000000000001e-5*cse555) - cse560*cse7*(0.0040460000000000001*cse1120 + 0.0036709999999999998*cse555));
  const amrex::Real cse1122 = 0.4375*cse10/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1123 = -cse1119;
  const amrex::Real cse1124 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1125 = std::pow(cse858, -2);
  const amrex::Real cse1126 = ((cse865) ? (
   -2.872276598720128e-13*cse260*cse862*state.xn[9]*(((5.7445531974402559e-18*cse2*cse862 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse862 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse863, 2)
)
: (
   0
));
  const amrex::Real cse1127 = -cse1126*cse907 - llte_co_interp2d_partial_0(state.T, cse869);
  const amrex::Real cse1128 = cse1126*cse911 + nhalf_co_interp2d_partial_0(state.T, cse869);
  const amrex::Real cse1129 = ((cse567) ? (
   -cse243
)
: (
   0
));
  const amrex::Real cse1130 = -1.8719999999999998e-14*cse1129*cse629 - 3.9261999999999998e-15*cse1129*cse630 - 1.53e-21*cse1129*cse631 + 4.985670872372847e-33*cse632*cse635*((cse567) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse633*cse635*((cse567) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse633*((cse567) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse634, 2);
  const amrex::Real cse1131 = ((cse567) ? (
   1
)
: (
   0
));
  const amrex::Real cse1132 = cse1131*cse628;
  const amrex::Real cse1133 = cse1132*cse12*cse596;
  const amrex::Real cse1134 = cse1132*cse578*cse600;
  const amrex::Real cse1135 = cse1132*cse580*cse602;
  const amrex::Real cse1136 = cse1132*cse8;
  const amrex::Real cse1137 = cse1132*cse582*cse604;
  const amrex::Real cse1138 = cse1132*cse572*cse619;
  const amrex::Real cse1139 = cse1132*cse574*cse621;
  const amrex::Real cse1140 = cse1132*cse576*cse598;
  const amrex::Real cse1141 = 65.320000000000007*cse243;
  const amrex::Real cse1142 = cse1141*cse672;
  const amrex::Real cse1143 = cse3*cse652;
  const amrex::Real cse1144 = 0.00042999999999999999*cse643;
  const amrex::Real cse1145 = cse680*(cse1118*(0.0041900000000000001 - 3.0000000000000001e-5*cse643) - cse3*cse673) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse682*(cse1118*(0.00025000000000000001*cse643 + 0.00264) + cse3*cse681) + cse683*(cse1118*(cse1144 + 0.00296) + cse1143));
  const amrex::Real cse1146 = cse665*(cse1118*(0.0036900000000000001 - 6.0000000000000002e-5*cse643) - cse3*cse644) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse667*(cse1118*(0.00041000000000000005*cse643 + 0.0020300000000000001) + cse3*cse666) + cse669*(cse1118*(0.00058*cse643 + 0.0023700000000000001) + cse3*cse668));
  const amrex::Real cse1147 = cse651*(cse1118*(0.0075500000000000003 - 0.0016000000000000001*cse643) - cse3*cse648) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse653*(cse1118*(cse1144 + 0.0088900000000000003) + cse1143) + cse655*(cse1118*(0.0052500000000000003*cse643 + 0.011979999999999999) + cse3*cse654));
  const amrex::Real cse1148 = cse1146*cse690;
  const amrex::Real cse1149 = cse1009*cse1147 + cse1010*cse1147 + cse1141*cse657*cse686 + cse1142*cse690 + cse1145*cse700 + cse1145*cse702 + cse1146*cse701 + cse1148*cse699 + 28.360000000000003*cse243*cse659*cse671;
  const amrex::Real cse1150 = 136.68000000000001*cse243*cse705;
  const amrex::Real cse1151 = cse1145*cse695;
  const amrex::Real cse1152 = cse1145*cse694;
  const amrex::Real cse1153 = cse1013*cse1146 + cse1015*cse1151 + cse1015*cse1152 + cse1142*cse694 + cse1146*cse706 + cse1147*cse702 + cse1147*cse706 + cse1150*cse694 + cse1150*cse695;
  const amrex::Real cse1154 = cse719*(-cse1018*cse1147 - cse1146*cse698 - cse1147*cse690 - cse1148 - cse1149 - cse1151 - cse1152 - cse1153 - 32.933333333333337*cse243*cse695*cse697);
  const amrex::Real cse1155 = 312.30000000000001*cse243;
  const amrex::Real cse1156 = cse1155*cse994;
  const amrex::Real cse1157 = cse3*cse726;
  const amrex::Real cse1158 = 0.00046000000000000001*cse643;
  const amrex::Real cse1159 = 8.6290000000000003e-7*cse1116;
  const amrex::Real cse1160 = -cse1159*cse744 + cse730*(cse1118*(cse1158 + 0.00231) + cse1157) + cse747*((cse591) ? (
   cse3*cse739 + cse74*(cse3*cse738 + cse74*(cse3*cse736 + cse3*cse737))
)
: (
   cse3*cse743 + cse74*(cse3*cse742 + cse74*(cse3*cse740 + cse3*cse741))
)) + state.xn[3]*(cse732*(cse1118*(0.00031*cse643 + 0.00263) + cse3*cse731) + cse734*(cse1118*(0.00064000000000000005*cse643 + 0.00071999999999999994) + cse3*cse733));
  const amrex::Real cse1161 = -1.4381666666666666e-6*cse1116*cse769 + cse755*(cse1118*(0.00057000000000000009*cse643 + 0.00115) + cse3*cse752) + cse771*((cse591) ? (
   cse3*cse764 + cse74*(cse3*cse763 + cse74*(-cse3*cse761 + cse3*cse762))
)
: (
   cse3*cse768 + cse74*(cse3*cse767 + cse74*(cse3*cse765 + cse3*cse766))
)) + state.xn[3]*(cse757*(cse1118*(0.00048999999999999998*cse643 - 4.0000000000000003e-5) + cse3*cse756) + cse759*(cse1118*(0.0010199999999999999*cse643 - 0.00085000000000000006) + cse3*cse758));
  const amrex::Real cse1162 = -cse1159*cse791 + cse777*(cse1118*(cse1158 + 0.0022800000000000003) + cse1157) + cse792*((cse591) ? (
   cse3*cse786 + cse74*(cse3*cse785 + cse74*(cse3*cse783 + cse3*cse784))
)
: (
   cse3*cse790 + cse74*(cse3*cse789 + cse74*(cse3*cse787 + cse3*cse788))
)) + state.xn[3]*(cse779*(cse1118*(0.00038000000000000002*cse643 + 0.0016900000000000001) + cse3*cse778) + cse781*(cse1118*(0.00048000000000000001*cse643 - 0.0001) + cse3*cse780));
  const amrex::Real cse1163 = cse1162*cse798;
  const amrex::Real cse1164 = cse1023*cse1160 + cse1024*cse1160 + cse1027*cse1162 + cse1155*cse772*cse991 + cse1156*cse798 + cse1161*cse808 + cse1161*cse810 + cse1163*cse807 + 844.16666666666686*cse243*cse748*cse794;
  const amrex::Real cse1165 = 70.859999999999985*cse243*cse813;
  const amrex::Real cse1166 = cse1161*cse802;
  const amrex::Real cse1167 = cse1161*cse803;
  const amrex::Real cse1168 = cse1032*cse1162 + cse1034*cse1166 + cse1034*cse1167 + cse1156*cse802 + cse1160*cse810 + cse1160*cse814 + cse1162*cse814 + cse1165*cse802 + cse1165*cse803;
  const amrex::Real cse1169 = cse827*(-cse1037*cse1160 - cse1160*cse798 - cse1162*cse806 - cse1163 - cse1164 - cse1166 - cse1167 - cse1168 - 64.733333333333348*cse243*cse803*cse805);

  jac(1, 1) = -cse1 + 2*cse28*state.xn[3] + 2*cse29*cse50 - 1.2000000000000001e-14*cse51 + 2*cse53*state.xn[0]*state.xn[3] - cse56 - cse60 - cse97;
  jac(1, 2) = cse107 + cse111;
  jac(1, 3) = cse121 + cse123*state.xn[16] + cse127*cse128 + cse135 + cse144;
  jac(1, 4) = cse109*cse169 - cse128*cse177 + cse145 + cse148 + cse153 + 2*cse154 + 2*cse163 + 4*cse165 + cse166*state.xn[17] + 1.1399999999999999e-10*cse173 + cse176*state.xn[8] + cse181 + cse190;
  jac(1, 5) = cse109*cse196 + cse145 + cse191 + 2*cse195 + cse200;
  jac(1, 7) = cse205;
  jac(1, 8) = cse209;
  jac(1, 9) = cse176*state.xn[3];
  jac(1, 10) = -cse210;
  jac(1, 11) = cse110 + cse213;
  jac(1, 12) = cse216 + cse217 + cse220;
  jac(1, 15) = cse221 + cse222 - cse223 + cse224;
  jac(1, 16) = cse224 + cse228;
  jac(1, 17) = cse109*cse234 + cse123*state.xn[2] + 2*cse233 + cse239;
  jac(1, 18) = cse166*state.xn[3] - cse240 + cse242;
  jac(1, 19) = cse294*(-cse132*cse274 + 9.8669999999999991e-12*cse174*cse271*state.xn[3]*state.xn[8] - cse245 - cse246 - cse249 + 2*cse258*cse29 - cse259 + 2*cse262*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse263 + cse264 - 5.5769999999999994e-13*cse267 - cse269 + cse270 + cse272 - cse278 - cse293);
  jac(2, 1) = cse97;
  jac(2, 2) = -cse107;
  jac(2, 3) = -cse143;
  jac(2, 4) = cse148 - cse183 + cse187;
  jac(2, 5) = -cse200;
  jac(2, 7) = cse201;
  jac(2, 11) = cse295;
  jac(2, 12) = -cse219;
  jac(2, 15) = cse295;
  jac(2, 16) = cse295;
  jac(2, 17) = -cse238;
  jac(2, 18) = cse240;
  jac(2, 19) = cse294*(-cse245 + cse279 + cse281 - cse282 - cse285 - cse296);
  jac(3, 1) = -cse105 - cse298 - cse337 + cse67 + cse96;
  jac(3, 2) = -cse106 - cse338;
  jac(3, 3) = -cse144 - cse350 - cse368 - cse374 - cse375 - cse390;
  jac(3, 4) = -cse190 + cse391 + cse392 - cse396 - cse402 - cse407 - cse408 - cse410;
  jac(3, 5) = -cse199 - cse395 - cse401 - cse410 - cse411;
  jac(3, 6) = cse412;
  jac(3, 7) = -cse414;
  jac(3, 8) = cse424 + cse426;
  jac(3, 9) = -cse429;
  jac(3, 10) = cse210;
  jac(3, 11) = -cse213 - cse337;
  jac(3, 12) = cse434;
  jac(3, 13) = cse437;
  jac(3, 14) = cse439;
  jac(3, 15) = cse440;
  jac(3, 16) = cse440;
  jac(3, 17) = -cse239 - cse441 - cse442 - cse443 - cse444 - cse445 - cse446 - cse447;
  jac(3, 18) = cse449;
  jac(3, 19) = cse294*(4.2899999999999994e-13*cse267 + cse275 + cse288 + cse293 + cse451 + cse452 + cse453 + cse456 + cse457 - cse459 + cse460 + cse461 - cse463 - cse464 - cse465 - cse466 - cse467);
  jac(3, 20) = cse468;
  jac(4, 1) = cse1 - cse469 + cse470 + cse473 + 6.0000000000000005e-15*cse51;
  jac(4, 2) = cse475;
  jac(4, 3) = cse131 - cse133 - cse134 + cse389 + cse476*state.xn[16];
  jac(4, 4) = -cse154 - cse163 - cse164*cse9 - cse398 - cse477 - cse478 - cse480 - cse481 - cse482 - cse483;
  jac(4, 5) = -cse191 - cse195 - cse198 + cse411 - cse484;
  jac(4, 7) = -cse201 - cse485;
  jac(4, 8) = cse423;
  jac(4, 9) = -cse427;
  jac(4, 10) = cse486;
  jac(4, 11) = cse475;
  jac(4, 12) = cse433;
  jac(4, 15) = cse223 + cse300 + cse473;
  jac(4, 16) = cse475;
  jac(4, 17) = -cse233 + cse444 + cse476*state.xn[2] - cse487 + cse489;
  jac(4, 18) = cse449;
  jac(4, 19) = cse294*(-cse244 - 2.3622303960273254e-7*cse263 + cse288 - cse490 - cse491 - cse492 - cse493 - cse494 - cse496);
  jac(5, 1) = -cse470;
  jac(5, 2) = cse474;
  jac(5, 4) = -cse153 + cse479 + cse481;
  jac(5, 5) = -cse191 - cse197 - cse411;
  jac(5, 7) = cse485;
  jac(5, 11) = cse474;
  jac(5, 15) = cse474;
  jac(5, 16) = cse474;
  jac(5, 17) = -cse444;
  jac(5, 19) = -cse294*cse495;
  jac(6, 1) = cse311;
  jac(6, 2) = cse311;
  jac(6, 3) = cse350;
  jac(6, 4) = cse497;
  jac(6, 5) = cse409;
  jac(6, 6) = -cse412;
  jac(6, 7) = cse499;
  jac(6, 10) = cse500;
  jac(6, 11) = cse311;
  jac(6, 15) = cse311;
  jac(6, 16) = cse501;
  jac(6, 17) = cse445;
  jac(6, 19) = cse294*cse503;
  jac(7, 1) = cse504;
  jac(7, 2) = cse504;
  jac(7, 3) = -cse350;
  jac(7, 4) = -cse497;
  jac(7, 5) = -cse409;
  jac(7, 6) = cse412;
  jac(7, 7) = -cse499;
  jac(7, 10) = -cse500;
  jac(7, 11) = cse504;
  jac(7, 15) = cse504;
  jac(7, 16) = -cse501;
  jac(7, 17) = -cse445;
  jac(7, 19) = -cse294*cse503;
  jac(8, 1) = cse1 + cse508;
  jac(8, 2) = cse508;
  jac(8, 3) = cse368;
  jac(8, 4) = cse400 + cse509 + cse510*state.xn[8] + cse514;
  jac(8, 5) = cse514;
  jac(8, 8) = -cse209 - cse423 - cse426;
  jac(8, 9) = cse428 + cse510*state.xn[3];
  jac(8, 10) = cse515;
  jac(8, 11) = cse508;
  jac(8, 15) = cse508 + cse516;
  jac(8, 16) = -cse226 + cse508;
  jac(8, 17) = -cse442 + cse446 - cse517;
  jac(8, 19) = cse294*(cse246 - 1.287e-13*cse267 + cse277 + cse451 + 2.2769999999999997e-12*cse462 + cse518);
  jac(9, 1) = cse519;
  jac(9, 2) = cse519;
  jac(9, 3) = -cse368;
  jac(9, 4) = -cse402;
  jac(9, 5) = -cse401;
  jac(9, 7) = cse498;
  jac(9, 8) = cse426;
  jac(9, 9) = -cse429 - cse521;
  jac(9, 10) = cse500;
  jac(9, 11) = cse519;
  jac(9, 15) = cse519;
  jac(9, 16) = -cse324 - cse522;
  jac(9, 17) = -cse446;
  jac(9, 19) = cse294*(-cse493 - cse518 - cse524);
  jac(10, 1) = cse297 + cse525;
  jac(10, 2) = cse525;
  jac(10, 3) = cse137;
  jac(10, 4) = cse526 + cse527;
  jac(10, 5) = cse527;
  jac(10, 7) = -cse498;
  jac(10, 8) = cse209;
  jac(10, 10) = -cse528 - cse529;
  jac(10, 11) = cse211 + cse525;
  jac(10, 12) = cse218;
  jac(10, 15) = cse221 + cse525;
  jac(10, 16) = cse226 + cse525;
  jac(10, 17) = cse517 + cse530;
  jac(10, 19) = -cse278*cse294;
  jac(11, 1) = -cse298;
  jac(11, 2) = cse60;
  jac(11, 3) = -cse137;
  jac(11, 4) = cse531;
  jac(11, 5) = cse531;
  jac(11, 9) = cse521;
  jac(11, 10) = cse528;
  jac(11, 11) = -cse212;
  jac(11, 15) = cse60;
  jac(11, 16) = cse522 - cse59;
  jac(11, 17) = -cse530;
  jac(11, 19) = cse294*(cse275 + cse524);
  jac(12, 1) = -cse506 - cse532;
  jac(12, 2) = -cse100 - cse506;
  jac(12, 3) = cse133 - cse385 + cse387;
  jac(12, 4) = cse179 - cse180 - cse406 - cse512 - cse526;
  jac(12, 5) = cse513;
  jac(12, 7) = cse498;
  jac(12, 10) = cse529;
  jac(12, 11) = cse507;
  jac(12, 12) = -cse220 - cse533;
  jac(12, 15) = -cse221 - cse506;
  jac(12, 16) = cse227 + cse507;
  jac(12, 17) = -cse441 - cse517;
  jac(12, 18) = cse240 + cse241;
  jac(12, 19) = cse294*(-cse534 - cse535);
  jac(13, 1) = cse336;
  jac(13, 2) = cse336;
  jac(13, 3) = cse374;
  jac(13, 4) = cse396;
  jac(13, 5) = cse395;
  jac(13, 11) = cse336;
  jac(13, 13) = -cse437;
  jac(13, 14) = cse438;
  jac(13, 15) = cse336;
  jac(13, 16) = cse336;
  jac(13, 17) = cse447;
  jac(13, 19) = cse294*cse536;
  jac(13, 20) = cse537;
  jac(14, 1) = cse538;
  jac(14, 2) = cse538;
  jac(14, 3) = -cse374;
  jac(14, 4) = -cse396;
  jac(14, 5) = -cse395;
  jac(14, 11) = cse538;
  jac(14, 13) = cse437;
  jac(14, 14) = cse439;
  jac(14, 15) = cse538;
  jac(14, 16) = cse538;
  jac(14, 17) = -cse447;
  jac(14, 19) = -cse294*cse536;
  jac(14, 20) = cse468;
  jac(15, 1) = -cse1;
  jac(15, 4) = cse539*state.xn[8];
  jac(15, 8) = cse423;
  jac(15, 9) = cse539*state.xn[3];
  jac(15, 12) = -cse218;
  jac(15, 15) = -cse221 - cse516;
  jac(15, 17) = cse442;
  jac(15, 19) = cse294*(-3.0029999999999999e-13*cse267 + cse450 + 5.3129999999999997e-12*cse462 + cse490);
  jac(16, 3) = cse135 + cse388;
  jac(16, 4) = cse407;
  jac(16, 7) = -cse204;
  jac(16, 8) = -cse209;
  jac(16, 9) = -cse521;
  jac(16, 12) = cse533;
  jac(16, 16) = -cse228 - cse522;
  jac(16, 17) = cse441;
  jac(16, 18) = cse242;
  jac(16, 19) = cse294*(cse249 + cse277 + cse494 + cse523 + cse534);
  jac(17, 3) = -cse390;
  jac(17, 4) = cse482;
  jac(17, 5) = cse191;
  jac(17, 8) = cse424;
  jac(17, 10) = -cse486;
  jac(17, 12) = cse434;
  jac(17, 17) = -cse443 - cse489;
  jac(17, 19) = cse294*(cse452 + cse496);
  jac(18, 1) = cse532;
  jac(18, 2) = cse100;
  jac(18, 3) = -cse375;
  jac(18, 4) = -cse483;
  jac(18, 7) = cse204;
  jac(18, 12) = cse219;
  jac(18, 16) = cse225;
  jac(18, 18) = -cse240 - cse448;
  jac(18, 19) = cse294*(cse290 + cse502 + cse535);
  jac(19, 1) = 1.602177e-12*crate*cse552 - 7.1777529600000012e-12*cse469 - 2.1789607199999999e-11*cse96 - cse961 + 6.0000000000000005e-15*d2g*state.xn[0]*(6.7291434000000005e-12*cse565 + 3.2043540000000002e-13);
  jac(19, 2) = -cse960 - state.xn[3]*((cse595) ? (
   0
)
: (
   -cse640*(-cse611 - cse614)
));
  jac(19, 3) = 1.7e-24*chi*cse10*cse70*cse936*cse972*d2g*(0.00025272365919999998*cse124*cse973 + 0.036605692000000002*cse124*cse974) + 4.6527220080000002e-11*crate*cse10*cse540*cse542*cse68*cse905*cse977*state.xn[5] + 4.22974728e-11*crate*cse10*cse540*cse542*cse68*cse977*state.xn[0] + 8.4594945600000001e-11*crate*cse10*cse540*cse542*cse68*cse977*state.xn[3] + 6.0732876035249664e-19*cse10*cse953*cse957*cse964*cse965*cse966*state.xn[2] - cse10*cse958*cse963 - 2.1789607199999999e-11*cse138 + 2.6241084038530201e-25*cse247*cse926*cse928*cse933*cse942*state.xn[8] - 2.6241084038530201e-25*cse247*cse929*cse934 - 7.2151085033443149e-29*cse347*cse889*cse967*cse970*cse971*state.xn[2] - cse860*cse884 - cse920*(-cse860*cse871 - cse860*cse880*cse917) - cse945*(-cse929*cse979 + cse943*cse979) - cse962 - cse976*(5.9301787935659931e-17*cse953*cse964*cse965*cse966*state.xn[2] - cse955*cse963) - cse978*(0.086512999999999993*chi*cse2*cse70*cse892*cse968*cse969 - cse890*cse895*cse970) - state.xn[11]*(cse710*cse983 + cse718*cse985 + cse724*cse986 + cse725*cse986) - state.xn[3]*((cse595) ? (
   0
)
: (
   cse625*cse640
)) - state.xn[7]*((cse834) ? (
   cse1002*cse832 + cse1002*cse833 + cse818*cse996 + cse826*cse999
)
: (
   0
));
  jac(19, 4) = 1.461185424e-21*chi*cse1047*cse171*cse172*cse900*state.xn[3] + 5.6999999999999997e-11*chi*cse171*cse172*(2.5634832e-11*cse565 + 6.4087080000000003e-13) + 1.7623947000000002e-12*crate*cse1042*cse905*state.xn[5] + 1.602177e-12*crate*cse1042*state.xn[0] + 3.2043540000000001e-12*crate*cse1042*state.xn[3] + 1.7623947000000002e-12*crate*cse185*cse552*cse959*state.xn[5] + 3.2043540000000001e-12*crate*cse552 - cse1044*cse884 + 2.0187430200000004e-26*cse1047*cse899*cse900*d2g - cse1053*cse1056 - cse1057*cse1058 - cse1065 - 7.1777529600000012e-12*cse154 - 7.1777529600000012e-12*cse163 - 1.4355505920000002e-11*cse165 - 7.1777529600000012e-12*cse478 - cse920*(-cse1044*cse871 - cse1052 - cse880*(cse1049 + cse918*(cse1044*cse875 + cse1051))) - cse945*(-cse1053*cse931 - cse1057*cse1059) - state.xn[11]*(cse1012*cse710 + cse1017*cse718 + cse1019*cse724 + cse1019*cse725) - state.xn[3]*((cse595) ? (
   0
)
: (
   -cse640*(-cse1020 - cse612)
)) - state.xn[7]*((cse834) ? (
   cse1031*cse818 + cse1036*cse826 + cse1038*cse832 + cse1038*cse833
)
: (
   0
)) - ((cse595) ? (
   0.0
)
: (
   cse1060*cse636
));
  jac(19, 5) = 1.461185424e-21*chi*cse1082*cse171*cse172*cse900*state.xn[3] + 1.7623947000000002e-12*crate*cse1041*cse905*state.xn[5] + 1.602177e-12*crate*cse1041*state.xn[0] + 3.2043540000000001e-12*crate*cse1041*state.xn[3] + 8.1070156200000004e-12*crate*cse32*cse552*cse959*state.xn[3]*state.xn[5] - cse1043*cse884 - cse1065 + 2.0187430200000004e-26*cse1082*cse899*cse900*d2g - 7.1777529600000012e-12*cse195 - 7.1777529600000012e-12*cse484 - cse920*(-cse1043*cse871 - cse1052 - cse880*(cse1049 + cse918*(cse1043*cse875 + cse1051))) + 8.78178441554344e-29*cse922*cse926*cse928*cse933*cse942*state.xn[8] - 8.78178441554344e-29*cse935 - cse945*(3.0319999999999999e-9*cse1083 - 3.0319999999999999e-9*cse930) - state.xn[11]*(cse1069*cse710 + cse1071*cse718 + cse1072*cse724 + cse1072*cse725) - state.xn[3]*((cse595) ? (
   0
)
: (
   cse1020*cse640
)) - state.xn[7]*((cse834) ? (
   cse1076*cse818 + cse1080*cse826 + cse1081*cse832 + cse1081*cse833
)
: (
   0
));
  jac(19, 6) = 1.7623947000000002e-12*crate*cse552*cse905 + cse1084;
  jac(19, 7) = cse1084;
  jac(19, 8) = -((cse834) ? (
   cse811*cse818 + cse815*cse826
)
: (
   0.0
));
  jac(19, 9) = -cse1054*cse944;
  jac(19, 10) = -cse861*cse883 - cse920*(cse1085*cse861*cse906*cse907 - cse879*(cse1085*cse858*cse872*cse906*cse907 - cse1086*cse873) - cse880*(cse1085*cse913*cse914 - cse1086*cse875*cse877));
  jac(19, 11) = cse1087;
  jac(19, 12) = -cse703*cse710 - cse707*cse718;
  jac(19, 15) = cse1087;
  jac(19, 16) = cse1087;
  jac(19, 17) = 1.461185424e-21*chi*cse1089*cse171*cse172*cse900*state.xn[3] + 5.0999999999999995e-24*chi*cse70*cse936*cse940*d2g + 1.7623947000000002e-12*crate*cse1090*cse905*state.xn[5] + 1.602177e-12*crate*cse1090*state.xn[0] + 3.2043540000000001e-12*crate*cse1090*state.xn[3] + 1.2160523430000001e-11*crate*cse32*cse552*cse959*state.xn[3]*state.xn[5] - 300.0*cse1061 - 9.6000000000000005e-32*cse1062 - 5.6994379667262474e-29*cse1063 - 4.097998887791448e-10*cse1064 - cse1088*cse884 + 2.0187430200000004e-26*cse1089*cse899*cse900*d2g - 7.1777529600000012e-12*cse233 - 7.1777529600000012e-12*cse487 - cse920*(-cse1088*cse871 + cse1091*cse861*cse906*cse907 - cse879*(cse1091*cse858*cse872*cse906*cse907 - cse1092*cse873) - cse880*(cse1091*cse915 + cse918*(cse1088*cse875 - cse1092*cse916))) + 1.3172676623315158e-28*cse922*cse926*cse928*cse933*cse942*state.xn[8] - 1.3172676623315158e-28*cse935 - cse945*(4.5479999999999996e-9*cse1083 - 4.5479999999999996e-9*cse930) - state.xn[11]*(cse1096*cse710 + cse1098*cse718 + cse1100*cse724 + cse1100*cse725) - state.xn[3]*((cse595) ? (
   0
)
: (
   3*cse641
)) - state.xn[7]*((cse834) ? (
   cse1104*cse818 + cse1106*cse826 + cse1107*cse832 + cse1107*cse833
)
: (
   0
));
  jac(19, 19) = cse294*(1.7e-24*chi*cse10*cse70*cse936*cse972*d2g*(-0.00012636182959999999*cse273*cse973 - 0.018302846000000001*cse273*cse974) + 5.6227499999999995e-25*chi*cse10*cse70*cse940*d2g*std::pow(state.T, -0.85299999999999998) + 7.8348995947646122e-14*cse10*cse1112*cse952*cse953*cse957*cse965*state.xn[2] - cse1056*cse1119 - cse1058*cse1123 - 1.6173435610483583e-5*cse1108*cse1109 - 2.0489994438957239e-11*cse1109*cse1111 - 1.3659996292638161e-10*cse1109*cse1113 - 5.2835524837844135e-18*cse1115*cse934 - 1.461185424e-21*cse1121*cse173*cse900*state.xn[3] - 2.0187430200000004e-26*cse1121*cse899*cse900*d2g - cse1122*cse884 - 2.1789607199999999e-11*cse279 - 7.1777529600000012e-12*cse491 - 7.1777529600000012e-12*cse492 - cse835*cse841*psigd_coll_interp2d_partial_1(cse10, state.T) - 1000*cse855*std::pow(cse856, 999)*cse882*cse919 - cse885*(1.6000000000000001e-32*cse260*cse347*cse837 + cse839) - cse920*(-cse1122*cse871 + cse1124*cse1125*cse874*cse878 + cse1124*cse1125 - cse1127*cse861*cse906 - cse879*(-cse1124*cse871*cse872 - cse1127*cse858*cse872*cse906 - cse1128*cse873) - cse880*(cse914*(cse1126*cse913 + alpha_co_interp2d_partial_0(state.T, cse869)) + cse918*(cse1122*cse875 - cse1128*cse916))) - cse945*(-cse1059*cse1123 - 182.41999999999999*cse1115 - cse1119*cse931) - cse976*(-0.0015792330470357991*cse1108 - 2.0007175426973804e-9*cse1111 + 7.6502807803182096e-12*cse1112*cse952*cse953*cse965*state.xn[2] - 1.3338116951315871e-8*cse1113) - cse978*(-0.043256499999999996*cse1114*cse892 + 0.84999999999999998*cse1114*cse895) - cse978*(-1.8989*cse1114*cse971 + cse3*cse893) - state.xn[11]*(cse1149*cse710 + cse1153*cse718 + cse1154*cse724 + cse1154*cse725) - state.xn[3]*((cse595) ? (
   0
)
: (
   cse1060*cse1130 + cse636*cse639*(-cse1130*cse637 - cse627*cse636*(-cse10*((cse607) ? (
      cse606*cse7*(116.29033200000001*cse1133 + 19.239805*cse1134 + 80.639324000000002*cse1135 + 37.383713*cse1136 + 145.968309*cse1137)
   )
   : ((cse609) ? (
      cse608*(8.2184944748967013*cse1131*cse628*cse8 - 52.189748993977005*cse1133 - 48.951834264235494*cse1134 - 196.44057098336626*cse1135 - 192.38155095558542*cse1137)
   )
   : (
      cse610*(-17.135767342440825*cse1133 + 17.889115159724135*cse1134 - 50.756388852554174*cse1135 + 10.695627721640689*cse1136 + 41.010708268606813*cse1137)
   ))) - cse613*(3.0182298984217999*cse1131*cse582*cse604*cse628 + 4.8223900769399082*cse1131*cse628*cse8 - 3.5529549287336835*cse1133 - 0.3872755400043702*cse1134 - 1.3735579540080118*cse1135) - cse615*(0.0708789387907936*cse1133 + 3.7035619079275177*cse1134 - 4.6974781513675152*cse1135 + 3.6184459289309552*cse1136 - 1.6316107607322889*cse1137) - cse617*(-3.7541549062629067*cse1133 + 2.2094886994529301*cse1134 - 1.528565035159452*cse1135 + 5.0409049417480247*cse1136 + 2.0057552335975877*cse1137) - state.xn[2]*((cse568) ? (
      cse623*(445.51869310442481*cse1133 + 11324.985706577943*cse1134 + 6766.9891340104432*cse1135 + 38.719649225812766*cse1136 + 2370.6849681533818*cse1137 + 1300.6686834345674*cse1138 + 5869.3152370465659*cse1139 + 11077.448644792414*cse1140)
   )
   : (
      cse624*(4.297112944704045*cse1133 - 101.40241318979159*cse1134 - 43.540996231705549*cse1135 + 3.8689780091986448*cse1136 + 27.91190909651122*cse1137 - 117.34186576967058*cse1138 + 103.76942484394611*cse1139 + 123.18901581604325*cse1140)
   ))))
)) - state.xn[7]*((cse834) ? (
   cse1164*cse818 + cse1168*cse826 + cse1169*cse832 + cse1169*cse833
)
: (
   0
)));
  jac(20, 13) = -cse436;
  jac(20, 20) = cse537;
  jac(21, 13) = -cse435*state.rn[2*0+1];
  jac(21, 21) = cse537;
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
