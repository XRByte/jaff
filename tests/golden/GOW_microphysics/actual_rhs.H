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
  const amrex::Real cse10 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse11 = state.T < 10.0;
  const amrex::Real cse12 = 1.0/state.T;
  const amrex::Real cse13 = state.T < 500.0;
  const amrex::Real cse14 = ((cse11) ? (
   3.6831481232786064e-10
)
: ((cse13) ? (
   7.9513696852843416e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse12)
)
: (
   9.6735673353022342e-11
)))*state.xn[7]*state.xn[15];
  const amrex::Real cse15 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av)*state.xn[15];
  const amrex::Real cse16 = M_LN10;
  const amrex::Real cse17 = 1.0/cse16;
  const amrex::Real cse18 = 2*state.xn[3];
  const amrex::Real cse19 = 3*state.xn[16];
  const amrex::Real cse20 = cse18 + cse19 + state.xn[0] + state.xn[1] + 2*state.xn[4] + state.xn[10] + state.xn[14] + state.xn[15];
  const amrex::Real cse21 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse22 = std::pow(cse21, 2)/std::pow(cse16, 2);
  const amrex::Real cse23 = 1.0/cse20;
  const amrex::Real cse24 = cse23*state.xn[3];
  const amrex::Real cse25 = cse23*state.xn[0];
  const amrex::Real cse26 = std::pow(10, 0.41599999999999998*cse17*cse21 + 0.32700000000000001*cse22 - 3)*cse25 + std::pow(10, 1.3*cse17*cse21 - 1.6200000000000001*cse22 - 4.8449999999999998)*cse24;
  const amrex::Real cse27 = cse20*((cse26 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse26
));
  const amrex::Real cse28 = cse17/(cse27 + 1);
  const amrex::Real cse29 = cse27*cse28;
  const amrex::Real cse30 = state.T > 700.0;
  const amrex::Real cse31 = ((cse30) ? (
   std::pow(10, cse28*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse12)) + cse29*std::log(1.3000000000000001e-9*std::exp(-53300*cse12)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse32 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse33 = ((cse30) ? (
   std::pow(10, cse28*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse12 - 1)) + cse29*std::log(3.5199999999999998e-9*std::exp(-43900*cse12)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse34 = 19.413464793123975*std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse35 = state.T < 300.0;
  const amrex::Real cse36 = ((cse11) ? (
   2.5561688710818002e-7
)
: ((cse35) ? (
   4.36e-8*cse34
)
: (
   4.36e-8
)));
  const amrex::Real cse37 = state.c_hat*state.rn[0];
  const amrex::Real cse38 = cse37*state.xn[3];
  const amrex::Real cse39 = cse20*d2g;
  const amrex::Real cse40 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse12)*state.xn[1]*state.xn[11];
  const amrex::Real cse41 = std::pow(state.T, -1.3);
  const amrex::Real cse42 = std::exp(-23*cse12);
  const amrex::Real cse43 = cse41*cse42*state.xn[3]*state.xn[8];
  const amrex::Real cse44 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse45 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av)*state.xn[14];
  const amrex::Real cse46 = cse44 + cse45;
  const amrex::Real cse47 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse12) + 0.62;
  const amrex::Real cse48 = state.T < 800.0;
  const amrex::Real cse49 = ((cse11) ? (
   2.0223626229831418e-8
)
: ((cse48) ? (
   1.3500000000000001e-9*cse47
)
: (
   3.0044907052141756e-9
)))*state.xn[6]*state.xn[15];
  const amrex::Real cse50 = state.xn[3]*state.xn[6];
  const amrex::Real cse51 = 1.2599999999999999e-13*cse50*std::exp(-22.5*cse12);
  const amrex::Real cse52 = cse49 + cse51;
  const amrex::Real cse53 = state.xn[2]*state.xn[16];
  const amrex::Real cse54 = cse53*((cse11) ? (
   1.371888797782434e-7
)
: ((cse35) ? (
   2.3400000000000001e-8*cse34
)
: (
   2.3400000000000001e-8
)));
  const amrex::Real cse55 = 1.99e-9*std::pow(state.T, -0.19)*state.xn[11]*state.xn[16];
  const amrex::Real cse56 = cse55*cse6;
  const amrex::Real cse57 = cse54 + cse56;
  const amrex::Real cse58 = ((cse11) ? (
   5.1211813936333398e-11
)
: ((cse13) ? (
   2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)
)
: (
   1.4161275116197014e-10
)))*state.xn[0]*state.xn[14];
  const amrex::Real cse59 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse60 = -cse58 + cse59;
  const amrex::Real cse61 = crate*state.xn[9];
  const amrex::Real cse62 = 6.5199999999999996*cse61;
  const amrex::Real cse63 = ((cse11) ? (
   2.4248802529616757e-6
)
: ((cse35) ? (
   1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)
)
: (
   2.7500000000000001e-7
)))*state.xn[2]*state.xn[10];
  const amrex::Real cse64 = 2.2999999999999998*cse24 + 1.5*cse25;
  const amrex::Real cse65 = crate*cse64;
  const amrex::Real cse66 = cse65*state.xn[0];
  const amrex::Real cse67 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse68 = state.xn[0]*state.xn[2];
  const amrex::Real cse69 = cse68*((cse30) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse67, 8) + 0.000111954395*std::pow(cse67, 7) - 0.00263197617*std::pow(cse67, 6) + 0.034825597700000002*std::pow(cse67, 5) - 0.28770560000000001*std::pow(cse67, 4) + 1.56315498*std::pow(cse67, 3) - 5.7393287500000003*std::pow(cse67, 2))
)
: (
   0.0
));
  const amrex::Real cse70 = 4.8813573300130879e-6*std::pow(cse12, 1.5)*std::pow(114.81344434488582*std::pow(cse12, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse71 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse72 = chi*cse71;
  const amrex::Real cse73 = cse3*cse72;
  const amrex::Real cse74 = 1.7*cse73;
  const amrex::Real cse75 = std::log(state.T);
  const amrex::Real cse76 = 1.2250000000000001e-11*cse39*state.xn[1]/(1.6774414051711492e-5*std::pow(cse73, 1.3779999999999999)*(508.69999999999999*std::pow(cse74, -1.102e-5*cse75 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse77 = -cse62 + cse63 - cse66 - cse69 + cse70 + cse76;
  const amrex::Real cse78 = cse32 - cse40;
  const amrex::Real cse79 = ((cse11) ? (
   3.5010967150614577e-11
)
: ((cse35) ? (
   1.4595000290776864e-10*std::pow(state.T, -0.62)
)
: (
   4.2499999999999999e-12
)))*state.xn[2]*state.xn[13];
  const amrex::Real cse80 = cse18*cse65;
  const amrex::Real cse81 = 2.0740886949963892e-18*state.xn[12];
  const amrex::Real cse82 = cse37*cse81;
  const amrex::Real cse83 = 2.1659999999999999e-12*cse39*state.xn[13]/(1.534817233071468e-7*std::pow(cse73, 1.8740000000000001)*(43750.0*std::pow(cse74, -7.538e-5*cse75 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse84 = 16800.0*crate*cse24*state.xn[12];
  const amrex::Real cse85 = cse17*cse75;
  const amrex::Real cse86 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse39*state.xn[6]/(7.104740267838494e-7*std::pow(cse73, 1.512)*(5115.0*std::pow(cse74, -5.4939999999999998e-7*cse75 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse85*(cse85*(0.044330000000000001*cse85 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse87 = (1.0537934368240927e-9*std::pow(cse12, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse12) + 1.31e-7*std::exp(-26.600000000000001*cse12) + 3.4e-8*std::exp(-7.6200000000000001*cse12) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse12)))*state.xn[7]*state.xn[16];
  const amrex::Real cse88 = cse23*state.xn[2];
  const amrex::Real cse89 = ((cse88 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse90 = cse55*cse89/(cse89 + 1);
  const amrex::Real cse91 = cse36*cse53 + cse57 + cse87 + cse90;
  const amrex::Real cse92 = crate*state.xn[7];
  const amrex::Real cse93 = 3.8500000000000001*cse92;
  const amrex::Real cse94 = 0.15970000000000001*std::exp(-49550.0*cse12);
  const amrex::Real cse95 = ((1.6339999999999999e-6*std::exp(-15230.0*cse12) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse12) + 6.3460000000000002e-9*std::exp(-12.17*cse12))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse94 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse94 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse96 = 7.1303347066783361e-18*state.xn[7];
  const amrex::Real cse97 = cse37*cse96;
  const amrex::Real cse98 = 4.5579999999999993e-11*cse39*state.xn[8]/(0.011078790689599145*std::pow(cse73, 1.1279999999999999)*(433.10000000000002*std::pow(cse74, -0.00013329999999999999*cse75 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse99 = 1040.0*cse24*cse92;
  const amrex::Real cse100 = -cse93 + cse95 - cse97 + cse98 - cse99;
  const amrex::Real cse101 = 3.2999999999999996e-13*cse43;
  const amrex::Real cse102 = cse8 + cse9;
  const amrex::Real cse103 = cse101 + cse102;
  const amrex::Real cse104 = -cse87;
  const amrex::Real cse105 = -cse90;
  const amrex::Real cse106 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse107 = cse33*state.xn[3];
  const amrex::Real cse108 = 7.2000000000000002e-15*cse50;
  const amrex::Real cse109 = cse1 + cse108 + cse80;
  const amrex::Real cse110 = -cse59;
  const amrex::Real cse111 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse112 = cse108 + cse111 + cse52 + cse86;
  const amrex::Real cse113 = cse104 + cse58;
  const amrex::Real cse114 = 1.5094998137029966e-18*state.xn[9];
  const amrex::Real cse115 = cse114*cse37;
  const amrex::Real cse116 = 184.0*cse24*cse61;
  const amrex::Real cse117 = cse115 + cse116 - cse14;
  const amrex::Real cse118 = ((cse11) ? (
   1.3707124444663516e-8
)
: ((cse48) ? (
   9.1500000000000005e-10*cse47
)
: (
   2.0363770335340524e-9
)))*state.xn[8]*state.xn[15];
  const amrex::Real cse119 = cse106 + cse62 - cse63;
  const amrex::Real cse120 = cse111 - cse44;
  const amrex::Real cse121 = cse10 + cse105 + cse15;
  const amrex::Real cse122 = -cse79 + cse82 - cse83 + cse84;
  const amrex::Real cse123 = std::log(1.7*cse2*cse72/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse124 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse125 = cse124*state.xn[2];
  const amrex::Real cse126 = std::exp(-118400.0*cse12)/(4.4460389837719569e-9*cse125 + 626500000.0);
  const amrex::Real cse127 = cse74 + 50;
  const amrex::Real cse128 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse129 = 1.75*cse128*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse128;
  const amrex::Real cse130 = l0_co_interp1d(state.T);
  const amrex::Real cse131 = 1.0/cse130;
  const amrex::Real cse132 = 100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2);
  const amrex::Real cse133 = 1.0/llte_co_interp2d(state.T, cse132);
  const amrex::Real cse134 = nhalf_co_interp2d(state.T, cse132);
  const amrex::Real cse135 = 0.001*state.T;
  const amrex::Real cse136 = cse135 + 1;
  const amrex::Real cse137 = std::pow(cse136, -2);
  const amrex::Real cse138 = 1.0/cse136;
  const amrex::Real cse139 = 1.0/(cse23*(5.6999999999999997e-11*chi*std::exp(-4.1799999999999997*av) + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse137 + 0.055500000000000001*cse138 - 11.06)*cse25 + std::pow(10, -2.0230000000000001*cse137 - 3.6709999999999998*cse138 - 11.08)*cse24) + 1);
  const amrex::Real cse140 = std::sqrt(cse12)*state.xn[2];
  const amrex::Real cse141 = (1.0/100.0)*state.T;
  const amrex::Real cse142 = std::log(cse141);
  const amrex::Real cse143 = state.T < 500;
  const amrex::Real cse144 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse145 = 4.5299999999999998e-6*cse140 + 7.5799999999999997e-10*std::pow(cse141, 0.0086999999999999994*cse142 + 0.12809999999999999)*state.xn[0] + (0.75*((cse143) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse144
)) + 0.25*((cse143) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse144
)))*state.xn[3];
  const amrex::Real cse146 = cse145*std::exp(-91.209999999999994*cse12)/(cse145 + 2.3e-6);
  const amrex::Real cse147 = cse17*std::log(cse20);
  const amrex::Real cse148 = cse24*((cse147 < 2) ? (
   10.0
)
: ((cse147 < 4) ? (
   (3.0/2.0)*cse147 + 7
)
: ((cse147 < 7) ? (
   (4.0/3.0)*cse147 + 23.0/3.0
)
: ((cse147 < 10) ? (
   (1.0/3.0)*cse147 + 44.0/3.0
)
: (
   18.0
))))) + cse25*(26.399999999999999*std::sqrt(cse88/(cse88 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse149 = state.T < 100.0;
  const amrex::Real cse150 = state.T < 6000.0;
  const amrex::Real cse151 = state.T < 10;
  const amrex::Real cse152 = std::log(cse135);
  const amrex::Real cse153 = std::pow(cse152, 2);
  const amrex::Real cse154 = std::pow(cse152, 3);
  const amrex::Real cse155 = std::pow(cse152, 4);
  const amrex::Real cse156 = std::pow(cse152, 5);
  const amrex::Real cse157 = std::pow(cse152, 6);
  const amrex::Real cse158 = std::pow(cse152, 7);
  const amrex::Real cse159 = std::pow(cse152, 8);
  const amrex::Real cse160 = std::pow(10.0, 16.815729999999999*cse152 + 96.743155000000002*cse153 + 343.1918*cse154 + 734.71650999999997*cse155 + 983.67575999999997*cse156 + 801.81246999999996*cse157 + 364.14445999999998*cse158 + 70.609154000000004*cse159 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse161 = state.xn[5] + state.xn[6];
  const amrex::Real cse162 = std::pow(10.0, 1.5714710999999999*cse152 + 0.015391166*cse153 - 0.23619984999999999*cse154 - 0.51002221000000003*cse155 + 0.32168730000000001*cse156 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse152 - 0.77151435999999995*cse153 + 0.43693353000000001*cse154 - 0.14913216000000001*cse155 - 0.033638326000000003*cse156 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse152 - 0.81520437999999995*cse153 + 0.29036281000000003*cse154 - 0.16596184*cse155 + 0.19191374999999999*cse156 - 23.689236999999999)*cse161;
  const amrex::Real cse163 = std::pow(10, 37.383713*cse152 + 58.145166000000003*cse153 + 48.656103000000002*cse154 + 20.159831000000001*cse155 + 3.8479610000000002*cse156 - 16.818342000000001)*state.xn[0] + cse162;
  const amrex::Real cse164 = std::pow(10.0, 1.6802758*cse152 + 0.93310621999999999*cse153 + 4.0406627000000004*cse154 - 4.7274035999999997*cse155 - 8.8077017000000009*cse156 + 8.9167182999999994*cse157 + 6.4380698000000001*cse158 - 6.3701156000000001*cse159 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse165 = state.T < 1000.0;
  const amrex::Real cse166 = std::pow(10.0, 3.5692468000000002*cse152 - 11.33286*cse153 - 27.850082*cse154 - 21.328264000000001*cse155 - 4.2519023000000002*cse156 - 24.311209000000002)*state.xn[0] + cse162;
  const amrex::Real cse167 = std::pow(10.0, 4.6450521*cse152 - 3.7209846*cse153 + 5.9369081000000001*cse154 - 5.5108049000000001*cse155 + 1.5538288*cse156 - 24.311209000000002)*state.xn[0] + cse162;
  const amrex::Real cse168 = 1.2972734453571459e-22*state.xn[0] + 1.2615234450058559e-20*state.xn[1] + 4.6629930528575411e-23*state.xn[3] + 1.3183594641975411e-19*state.xn[5] + 1.3183594641975411e-19*state.xn[6];
  const amrex::Real cse169 = ((cse150) ? (
   cse12
)
: (
   0.00016666666666666666
));
  const amrex::Real cse170 = 1.6e-18*std::exp(-11700.0*cse169) + 6.7e-19*std::exp(-5860.0*cse169) + 3.0e-24*std::exp(-510.0*cse169) + 4.985670872372847e-33*((cse150) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse150) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse150) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse171 = std::log(0.001*((cse150) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse172 = std::pow(cse171, 2);
  const amrex::Real cse173 = std::pow(cse171, 3);
  const amrex::Real cse174 = std::pow(cse171, 4);
  const amrex::Real cse175 = std::pow(cse171, 5);
  const amrex::Real cse176 = std::pow(cse171, 6);
  const amrex::Real cse177 = std::pow(cse171, 7);
  const amrex::Real cse178 = std::pow(cse171, 8);
  const amrex::Real cse179 = 0.042999999999999997*cse142;
  const amrex::Real cse180 = 4.34e-10*std::pow(cse141, 0.755 - 0.16*cse142)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse141, 0.52500000000000002*cse142 + 1.198) + 5.2499999999999998e-13*std::pow(cse141, cse179 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse181 = 3.5700000000000001e-10*std::pow(cse141, 0.41899999999999998 - 0.0030000000000000001*cse142)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse141, 0.025000000000000001*cse142 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse141, cse179 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse182 = (1.0/5.0)*std::exp(-326.60000000000002*cse12);
  const amrex::Real cse183 = 3.1899999999999998e-10*std::pow(cse141, 0.36899999999999999 - 0.0060000000000000001*cse142)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse141, 0.041000000000000002*cse142 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse141, 0.058000000000000003*cse142 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse184 = cse181 + 8.9099999999999997e-5;
  const amrex::Real cse185 = cse182*cse183;
  const amrex::Real cse186 = cse180*cse181*cse182 + (1.0/15.0)*cse180*cse183*std::exp(-425.40000000000003*cse12) + cse184*cse185;
  const amrex::Real cse187 = cse180 + 1.7499999999999998e-5;
  const amrex::Real cse188 = cse183 + 1.34e-10;
  const amrex::Real cse189 = (3.0/5.0)*cse181*std::exp(-227.80000000000001*cse12);
  const amrex::Real cse190 = cse185*cse187 + cse187*cse189 + cse188*cse189;
  const amrex::Real cse191 = 1.0/((1.0/3.0)*cse180*cse188*std::exp(-98.800000000000011*cse12) + cse184*cse187 + cse184*cse188 + cse186 + cse190);
  const amrex::Real cse192 = 0.045999999999999999*cse142;
  const amrex::Real cse193 = 1.7258000000000001e-6*cse140;
  const amrex::Real cse194 = 8.9000000000000003e-11*std::pow(cse141, cse192 + 0.22800000000000001)*state.xn[0] + cse193*std::exp(((cse165) ? (
   cse75*(cse75*(cse75*(0.0070527699999999999*cse75 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse75*(cse75*(cse75*(0.093813800000000003*cse75 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse141, 0.037999999999999999*cse142 + 0.16900000000000001) + 2.15e-11*std::pow(cse141, 0.048000000000000001*cse142 - 0.01))*state.xn[3];
  const amrex::Real cse195 = 2.6400000000000002e-10*std::pow(cse141, cse192 + 0.23100000000000001)*state.xn[0] + cse193*std::exp(((cse165) ? (
   cse75*(cse75*(cse75*(0.00235272*cse75 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse75*(cse75*(cse75*(0.097857299999999994*cse75 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse141, 0.031*cse142 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse141, 0.064000000000000001*cse142 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse196 = 2.8763333333333333e-6*cse140*std::exp(((cse165) ? (
   cse75*(cse75*(cse75*(-0.00065632500000000003*cse75 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse75*(cse75*(cse75*(0.10508000000000001*cse75 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse141, 0.057000000000000002*cse142 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse141, 0.049000000000000002*cse142 - 0.0040000000000000001) + 1.675e-11*std::pow(cse141, 0.10199999999999999*cse142 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse197 = 5*std::exp(-62.460000000000001*cse12);
  const amrex::Real cse198 = cse196 + 7.8800000000000004e-8;
  const amrex::Real cse199 = cse194*cse197;
  const amrex::Real cse200 = (25.0/3.0)*cse194*cse195*std::exp(-101.30000000000001*cse12) + cse195*cse196*cse197 + cse198*cse199;
  const amrex::Real cse201 = cse195 + 2.65e-7;
  const amrex::Real cse202 = cse194 + 1.81e-14;
  const amrex::Real cse203 = 3*cse196*std::exp(-23.619999999999997*cse12);
  const amrex::Real cse204 = cse199*cse201 + cse201*cse203 + cse202*cse203;
  const amrex::Real cse205 = 1.0/((5.0/3.0)*cse195*cse202*std::exp(-38.840000000000003*cse12) + cse198*cse201 + cse198*cse202 + cse200 + cse204);
  const amrex::Real cse206 = state.c_hat*state.rn[2*0+1];

  ydot(1) = cse1 + cse10 + cse14 + cse15 + cse18*cse33 + cse19*cse36*state.xn[2] + 2*cse31 - cse32 + 2.9899673672605791e-18*cse38 - 6.0000000000000005e-15*cse39*state.xn[0] + cse40 + 4.2899999999999994e-13*cse43 + cse46 + cse52 + cse57 + cse60 + 3.2000000000000001e-9*cse7 + cse77 + cse9;
  ydot(2) = cse0 + cse51 + cse66 + cse69 - cse70 - cse76 + cse78;
  ydot(3) = -cse100 - cse103 - cse77 - cse79 + cse80 + cse82 - cse83 + cse84 - cse86 - cse91;
  ydot(4) = -cse103 - cse104 - cse105 + cse106 - cse107 - cse109 + 3.0000000000000002e-15*cse20*d2g*state.xn[0] - cse31 - 1.4949836836302895e-18*cse38 - cse51 + cse54 + cse56 - cse60;
  ydot(5) = cse109 + cse110;
  ydot(6) = cse112;
  ydot(7) = -cse112;
  ydot(8) = cse100 + cse113 + cse117 + 9.8999999999999995e-14*cse43 + cse45;
  ydot(9) = -cse101 + cse111 - cse118 + cse93 - cse95 + cse97 - cse98 + cse99;
  ydot(10) = -cse117 - cse119 - cse120;
  ydot(11) = cse118 + cse119;
  ydot(12) = cse115 + cse116 + cse120 + cse121 + cse78 + cse8;
  ydot(13) = -cse122;
  ydot(14) = cse122;
  ydot(15) = -cse113 + 2.3099999999999997e-13*cse41*cse42*state.xn[3]*state.xn[8] - cse46;
  ydot(16) = -cse118 - cse121 - cse14 - cse49 + cse9;
  ydot(17) = -cse106 - cse110 - cse91;
  ydot(18) = -cse102 + cse49 - cse78;
  ydot(19) = 1.4949836836302895e-18*state.c_hat*(2.5634832e-11*cse139 + 6.4087080000000003e-13)*state.xn[3]*state.rn[0] + 1.7e-24*chi*cse20*cse71*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse127, 0.43099999999999999)*(0.0043*std::pow(cse127, 0.69199999999999995) + 1) + 1) + 1.602177e-12*crate*cse148*state.xn[0] + 3.2043540000000001e-12*crate*cse148*state.xn[3] + 1.7623947000000002e-12*crate*cse148*state.xn[5]/cse64 - 7.1777529600000012e-12*cse107 - 1.3659996292638161e-10*cse124*cse126*cse68/(1.3338116951315871e-8*cse125*cse126 + 1) - cse129*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse129*cse133 + cse131*std::pow(cse129/cse134, alpha_co_interp2d(state.T, cse132))*(-cse130*cse133*cse134 + 1) + cse131) - 5.7927337833400002e-20*cse146*state.xn[8]/(2*cse146 + 1) + 3.0000000000000002e-15*cse20*d2g*(6.7291434000000005e-12*cse139 + 3.2043540000000002e-13)*state.xn[0] - 7.1777529600000012e-12*cse31 - 2.1789607199999999e-11*cse69 - 1.8998126555754159e-29*d2g*std::pow(state.T, 0.45350000000000001 + 2.234/cse123)*std::exp(cse123*(1.4419999999999999 - 0.050889999999999998*cse123))*state.xn[2] - (2.3872025442750958e-19*cse186*cse191 + 2.8023005140020001e-18*cse190*cse191)*state.xn[11] - ((((cse143 && cse149 && cse150) ? (
   cse151 || cse160 + cse163 < 1.0e-100
)
: (
   ((cse149 && cse150) ? (
      cse151 || cse163 + cse164 < 1.0e-100
   )
   : (
      ((cse143 && cse150 && cse165) ? (
         cse151 || cse160 + cse166 < 1.0e-100
      )
      : (
         ((cse150 && cse165) ? (
            cse151 || cse164 + cse166 < 1.0e-100
         )
         : (
            ((cse143 && cse150) ? (
               cse151 || cse160 + cse167 < 1.0e-100
            )
            : (
               ((cse150) ? (
                  cse151 || cse164 + cse167 < 1.0e-100
               )
               : (
                  ((cse143) ? (
                     cse151 || cse168 + 3.0478204245257306e+83651*state.xn[2] < 1.0e-100
                  )
                  : (
                     cse151 || cse168 + 1.0073452104972926e-205*state.xn[2] < 1.0e-100
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
   cse170/(cse170/(std::pow(10.0, 1.5714710999999999*cse171 + 0.015391166*cse172 - 0.23619984999999999*cse173 - 0.51002221000000003*cse174 + 0.32168730000000001*cse175 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse171 - 0.77151435999999995*cse172 + 0.43693353000000001*cse173 - 0.14913216000000001*cse174 - 0.033638326000000003*cse175 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse171 - 0.81520437999999995*cse172 + 0.29036281000000003*cse173 - 0.16596184*cse174 + 0.19191374999999999*cse175 - 23.689236999999999)*cse161 + ((cse143) ? (
      std::pow(10.0, 16.815729999999999*cse171 + 96.743155000000002*cse172 + 343.1918*cse173 + 734.71650999999997*cse174 + 983.67575999999997*cse175 + 801.81246999999996*cse176 + 364.14445999999998*cse177 + 70.609154000000004*cse178 - 21.928795999999998)
   )
   : (
      std::pow(10.0, 1.6802758*cse171 + 0.93310621999999999*cse172 + 4.0406627000000004*cse173 - 4.7274035999999997*cse174 - 8.8077017000000009*cse175 + 8.9167182999999994*cse176 + 6.4380698000000001*cse177 - 6.3701156000000001*cse178 - 22.921188999999998)
   ))*state.xn[2] + ((((cse150) ? (
      cse149
   )
   : (
      false
   ))) ? (
      std::pow(10, 37.383713*cse171 + 58.145166000000003*cse172 + 48.656103000000002*cse173 + 20.159831000000001*cse174 + 3.8479610000000002*cse175 - 16.818342000000001)
   )
   : ((((cse150) ? (
      cse165
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse171 - 11.33286*cse172 - 27.850082*cse173 - 21.328264000000001*cse174 - 4.2519023000000002*cse175 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 4.6450521*cse171 - 3.7209846*cse172 + 5.9369081000000001*cse173 - 5.5108049000000001*cse174 + 1.5538288*cse175 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse200*cse205 + 2.5697412351439998e-22*cse204*cse205
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse20, state.T), 3.2000000000000002e-32*cse2*cse39*(state.T - 2.1544346900318838*std::pow(85973.158732959506*cse72 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = -cse115 - 5.4061971382371991e-18*cse38 - cse82 - cse97;
  ydot(21) = -cse114*cse206 - cse206*cse81 - cse206*cse96 - 5.4061971382371991e-18*cse206*state.xn[3];
  // $JAFF END
  // $JAFF SUB nspec
  ydot(19) /= state.rho;
  // $JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {

  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
  const amrex::Real cse0 = state.T < 10.0;
  const amrex::Real cse1 = state.T < 500.0;
  const amrex::Real cse2 = ((cse0) ? (
   5.1211813936333398e-11
)
: ((cse1) ? (
   2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)
)
: (
   1.4161275116197014e-10
)));
  const amrex::Real cse3 = cse2*state.xn[14];
  const amrex::Real cse4 = std::sqrt(state.T);
  const amrex::Real cse5 = 1.0/state.T;
  const amrex::Real cse6 = 63590*cse5 + 1;
  const amrex::Real cse7 = std::exp(-cse6);
  const amrex::Real cse8 = std::log(6.6699999999999996e-12*cse4*cse7);
  const amrex::Real cse9 = M_LN10;
  const amrex::Real cse10 = 1.0/cse9;
  const amrex::Real cse11 = 3*state.xn[16];
  const amrex::Real cse12 = 2*state.xn[3];
  const amrex::Real cse13 = cse11 + cse12 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 2*state.xn[4];
  const amrex::Real cse14 = 1.0/cse13;
  const amrex::Real cse15 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse16 = std::pow(cse9, -2);
  const amrex::Real cse17 = std::pow(cse15, 2)*cse16;
  const amrex::Real cse18 = std::pow(10, 1.3*cse10*cse15 - 1.6200000000000001*cse17 - 4.8449999999999998);
  const amrex::Real cse19 = cse14*cse18*state.xn[3];
  const amrex::Real cse20 = std::pow(10, 0.41599999999999998*cse10*cse15 + 0.32700000000000001*cse17 - 3);
  const amrex::Real cse21 = cse14*cse20*state.xn[0];
  const amrex::Real cse22 = cse19 + cse21;
  const amrex::Real cse23 = cse22 < 1.0e-50;
  const amrex::Real cse24 = ((cse23) ? (
   1.0e-50
)
: (
   cse22
));
  const amrex::Real cse25 = cse13*cse24;
  const amrex::Real cse26 = cse25 + 1;
  const amrex::Real cse27 = cse10/cse26;
  const amrex::Real cse28 = std::log(3.5199999999999998e-9*std::exp(-43900*cse5));
  const amrex::Real cse29 = cse25*cse27;
  const amrex::Real cse30 = std::pow(10, cse27*cse8 + cse28*cse29);
  const amrex::Real cse31 = state.T > 700.0;
  const amrex::Real cse32 = ((cse31) ? (
   cse30
)
: (
   0.0
));
  const amrex::Real cse33 = std::pow(state.xn[3], 2);
  const amrex::Real cse34 = std::log(1.3000000000000001e-9*std::exp(-53300*cse5));
  const amrex::Real cse35 = cse24*cse27;
  const amrex::Real cse36 = cse34*cse35;
  const amrex::Real cse37 = std::pow(cse13, 2);
  const amrex::Real cse38 = 1.0/cse37;
  const amrex::Real cse39 = cse38*state.xn[3];
  const amrex::Real cse40 = cse18*cse39;
  const amrex::Real cse41 = cse38*state.xn[0];
  const amrex::Real cse42 = cse20*cse41;
  const amrex::Real cse43 = cse40 + cse42;
  const amrex::Real cse44 = cse13*((cse23) ? (
   0
)
: (
   cse14*cse20 - cse43
));
  const amrex::Real cse45 = cse27*cse44;
  const amrex::Real cse46 = std::pow(state.T, 4.1881000000000004);
  const amrex::Real cse47 = 6.7610000000000002e-6*state.T + 1;
  const amrex::Real cse48 = std::pow(cse47, 5.6881000000000004);
  const amrex::Real cse49 = 54657.400000000001*cse5;
  const amrex::Real cse50 = std::exp(-cse49);
  const amrex::Real cse51 = cse50/cse48;
  const amrex::Real cse52 = std::log(5.9959999999999997e-30*cse46*cse51);
  const amrex::Real cse53 = cse10/std::pow(cse26, 2);
  const amrex::Real cse54 = cse53*(-cse24 - cse44);
  const amrex::Real cse55 = cse25*cse54;
  const amrex::Real cse56 = std::pow(10, cse27*cse52 + cse29*cse34);
  const amrex::Real cse57 = cse56*cse9;
  const amrex::Real cse58 = ((cse31) ? (
   cse57*(cse34*cse45 + cse34*cse55 + cse36 + cse52*cse54)
)
: (
   0
));
  const amrex::Real cse59 = 6.0000000000000005e-15*d2g;
  const amrex::Real cse60 = cse59*state.xn[0];
  const amrex::Real cse61 = cse28*cse35;
  const amrex::Real cse62 = cse30*cse9;
  const amrex::Real cse63 = ((cse31) ? (
   cse62*(cse28*cse45 + cse28*cse55 + cse54*cse8 + cse61)
)
: (
   0
));
  const amrex::Real cse64 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse65 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse66 = cse65*state.xn[17];
  const amrex::Real cse67 = 1.5*cse14*state.xn[0] + 2.2999999999999998*cse14*state.xn[3];
  const amrex::Real cse68 = crate*cse67;
  const amrex::Real cse69 = 1.0/state.xn[2];
  const amrex::Real cse70 = cse4*cse69;
  const amrex::Real cse71 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse72 = chi*cse71;
  const amrex::Real cse73 = cse70*cse72;
  const amrex::Real cse74 = 1.7*cse73;
  const amrex::Real cse75 = std::log(state.T);
  const amrex::Real cse76 = 1.102e-5*cse75 + 0.4723;
  const amrex::Real cse77 = std::pow(cse74, -cse76);
  const amrex::Real cse78 = cse77*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse79 = 508.69999999999999*cse78;
  const amrex::Real cse80 = cse79 + 1;
  const amrex::Real cse81 = std::pow(cse73, 1.3779999999999999);
  const amrex::Real cse82 = 1.6774414051711492e-5*cse81;
  const amrex::Real cse83 = cse80*cse82 + 1;
  const amrex::Real cse84 = 1.0/cse83;
  const amrex::Real cse85 = 1.2250000000000001e-11*cse84;
  const amrex::Real cse86 = d2g*state.xn[1];
  const amrex::Real cse87 = cse85*cse86;
  const amrex::Real cse88 = -cse87;
  const amrex::Real cse89 = 2.2999999999999998*cse39 + 1.5*cse41;
  const amrex::Real cse90 = -1.5*cse14 + cse89;
  const amrex::Real cse91 = -cse90;
  const amrex::Real cse92 = crate*state.xn[0];
  const amrex::Real cse93 = cse91*cse92;
  const amrex::Real cse94 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse95 = std::pow(cse94, 2);
  const amrex::Real cse96 = std::pow(cse94, 3);
  const amrex::Real cse97 = std::pow(cse94, 4);
  const amrex::Real cse98 = std::pow(cse94, 5);
  const amrex::Real cse99 = std::pow(cse94, 6);
  const amrex::Real cse100 = std::pow(cse94, 7);
  const amrex::Real cse101 = std::exp(0.000111954395*cse100 - 2.0391498499999999e-6*std::pow(cse94, 8) - 5.7393287500000003*cse95 + 1.56315498*cse96 - 0.28770560000000001*cse97 + 0.034825597700000002*cse98 - 0.00263197617*cse99);
  const amrex::Real cse102 = 5.9078447946382801e-70*cse101*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse103 = ((cse31) ? (
   cse102
)
: (
   0.0
));
  const amrex::Real cse104 = cse103*state.xn[2];
  const amrex::Real cse105 = cse104 + cse64 + cse66 + cse68 + cse88 + cse93;
  const amrex::Real cse106 = std::exp(-227*cse5);
  const amrex::Real cse107 = cse106*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse108 = cse107*state.xn[11];
  const amrex::Real cse109 = std::pow(cse5, 1.5);
  const amrex::Real cse110 = 114.81344434488582*std::pow(cse5, 0.40699999999999997) + 1;
  const amrex::Real cse111 = std::pow(cse110, -2.242);
  const amrex::Real cse112 = 4.8813573300130879e-6*cse109*cse111;
  const amrex::Real cse113 = cse13*d2g;
  const amrex::Real cse114 = -cse89;
  const amrex::Real cse115 = cse114*cse92;
  const amrex::Real cse116 = -cse115 + cse87;
  const amrex::Real cse117 = cse112*state.xn[2] + cse113*cse85 + cse116;
  const amrex::Real cse118 = cse108 + cse117;
  const amrex::Real cse119 = cse13*((cse23) ? (
   0
)
: (
   -cse43
));
  const amrex::Real cse120 = cse119*cse27;
  const amrex::Real cse121 = cse53*(-cse119 - cse24);
  const amrex::Real cse122 = cse121*cse25;
  const amrex::Real cse123 = cse33*((cse31) ? (
   cse57*(cse120*cse34 + cse121*cse52 + cse122*cse34 + cse36)
)
: (
   0
));
  const amrex::Real cse124 = -cse60;
  const amrex::Real cse125 = ((cse31) ? (
   cse62*(cse120*cse28 + cse121*cse8 + cse122*cse28 + cse61)
)
: (
   0
));
  const amrex::Real cse126 = cse12*state.xn[0];
  const amrex::Real cse127 = 2*cse123 + cse124 + cse125*cse126;
  const amrex::Real cse128 = 19.413464793123975*std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse129 = state.T < 300.0;
  const amrex::Real cse130 = ((cse0) ? (
   2.5561688710818002e-7
)
: ((cse129) ? (
   4.36e-8*cse128
)
: (
   4.36e-8
)));
  const amrex::Real cse131 = std::pow(state.xn[2], 2);
  const amrex::Real cse132 = 1.0/cse131;
  const amrex::Real cse133 = cse132*cse4;
  const amrex::Real cse134 = 0.00011320754716981131*cse70;
  const amrex::Real cse135 = cse134*state.xn[3];
  const amrex::Real cse136 = (((cse135 - 10000000000.0 > 0) ? (
   0
)
: ((cse135 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse137 = cse136*state.xn[17];
  const amrex::Real cse138 = cse137*cse33;
  const amrex::Real cse139 = cse133*cse138;
  const amrex::Real cse140 = std::min(10000000000.0, cse135);
  const amrex::Real cse141 = cse140 + 1;
  const amrex::Real cse142 = std::pow(cse141, -2);
  const amrex::Real cse143 = 3.6226415094339619e-13*cse142;
  const amrex::Real cse144 = 1.0/cse141;
  const amrex::Real cse145 = 1.811320754716981e-13*cse144;
  const amrex::Real cse146 = cse139*cse145;
  const amrex::Real cse147 = 1.811320754716981e-13*cse142;
  const amrex::Real cse148 = cse139*cse147;
  const amrex::Real cse149 = cse140*cse148;
  const amrex::Real cse150 = -cse146 + cse149;
  const amrex::Real cse151 = ((cse0) ? (
   1.371888797782434e-7
)
: ((cse129) ? (
   2.3400000000000001e-8*cse128
)
: (
   2.3400000000000001e-8
)));
  const amrex::Real cse152 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse153 = cse132*cse136*cse142*cse152*state.xn[11]*state.xn[16]*state.xn[3] + cse151*state.xn[16];
  const amrex::Real cse154 = ((cse0) ? (
   2.4248802529616757e-6
)
: ((cse129) ? (
   1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)
)
: (
   2.7500000000000001e-7
)));
  const amrex::Real cse155 = cse154*state.xn[10];
  const amrex::Real cse156 = cse103*state.xn[0];
  const amrex::Real cse157 = cse80*cse81;
  const amrex::Real cse158 = -cse76;
  const amrex::Real cse159 = 1.2250000000000001e-11*cse113*state.xn[1]/std::pow(cse83, 2);
  const amrex::Real cse160 = cse112*state.xn[1] - cse156 + cse159*(2.3115142563258434e-5*cse157*cse69 + 0.0085331444281056355*cse158*cse69*cse78*cse81);
  const amrex::Real cse161 = cse155 + cse160;
  const amrex::Real cse162 = std::exp(-22.5*cse5);
  const amrex::Real cse163 = 1.2599999999999999e-13*cse162;
  const amrex::Real cse164 = cse163*state.xn[6];
  const amrex::Real cse165 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse166 = cse165*state.xn[4];
  const amrex::Real cse167 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse168 = 1.7599999999999999e-9*cse167;
  const amrex::Real cse169 = cse166*cse168;
  const amrex::Real cse170 = d2g*state.xn[0];
  const amrex::Real cse171 = -1.2000000000000001e-14*cse170;
  const amrex::Real cse172 = cse32*state.xn[0];
  const amrex::Real cse173 = 2*cse24;
  const amrex::Real cse174 = cse27*cse34;
  const amrex::Real cse175 = cse173*cse174;
  const amrex::Real cse176 = cse12*cse38;
  const amrex::Real cse177 = cse176*cse18 + 2*cse42;
  const amrex::Real cse178 = cse13*((cse23) ? (
   0
)
: (
   cse14*cse18 - cse177
));
  const amrex::Real cse179 = cse53*(-cse173 - cse178);
  const amrex::Real cse180 = cse179*cse25;
  const amrex::Real cse181 = cse33*((cse31) ? (
   cse57*(cse174*cse178 + cse175 + cse179*cse52 + cse180*cse34)
)
: (
   0
));
  const amrex::Real cse182 = ((cse31) ? (
   cse56
)
: (
   0.0
));
  const amrex::Real cse183 = cse182*state.xn[3];
  const amrex::Real cse184 = 3.2000000000000001e-9*cse144;
  const amrex::Real cse185 = 2.9899673672605791e-18*state.c_hat;
  const amrex::Real cse186 = cse27*cse28;
  const amrex::Real cse187 = cse173*cse186;
  const amrex::Real cse188 = ((cse31) ? (
   cse62*(cse178*cse186 + cse179*cse8 + cse180*cse28 + cse187)
)
: (
   0
));
  const amrex::Real cse189 = std::exp(-23*cse5);
  const amrex::Real cse190 = cse189*std::pow(state.T, -1.3);
  const amrex::Real cse191 = 4.2899999999999994e-13*cse190;
  const amrex::Real cse192 = cse137*cse70*state.xn[3];
  const amrex::Real cse193 = 1.6000000000000001e-9*cse144;
  const amrex::Real cse194 = cse193*state.xn[17];
  const amrex::Real cse195 = cse147*cse192;
  const amrex::Real cse196 = cse140*cse194 - cse140*cse195 + cse145*cse192;
  const amrex::Real cse197 = cse84*cse86;
  const amrex::Real cse198 = 2.4500000000000003e-11*cse197;
  const amrex::Real cse199 = 4.5999999999999996*cse39 + 3.0*cse41;
  const amrex::Real cse200 = -2.2999999999999998*cse14 + cse199;
  const amrex::Real cse201 = -cse200;
  const amrex::Real cse202 = cse201*cse92;
  const amrex::Real cse203 = cse136*cse142*cse69*state.xn[11]*state.xn[16];
  const amrex::Real cse204 = cse152*cse203;
  const amrex::Real cse205 = cse198 - cse202 - cse204;
  const amrex::Real cse206 = cse13*((cse23) ? (
   0
)
: (
   -cse177
));
  const amrex::Real cse207 = cse53*(-cse173 - cse206);
  const amrex::Real cse208 = cse207*cse25;
  const amrex::Real cse209 = cse33*((cse31) ? (
   cse57*(cse174*cse206 + cse175 + cse207*cse52 + cse208*cse34)
)
: (
   0
));
  const amrex::Real cse210 = ((cse31) ? (
   cse62*(cse186*cse206 + cse187 + cse207*cse8 + cse208*cse28)
)
: (
   0
));
  const amrex::Real cse211 = cse165*cse168*state.xn[3];
  const amrex::Real cse212 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse213 = cse211 - cse212;
  const amrex::Real cse214 = -cse199;
  const amrex::Real cse215 = cse214*cse92;
  const amrex::Real cse216 = cse198 - cse215;
  const amrex::Real cse217 = cse163*state.xn[3];
  const amrex::Real cse218 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse5) + 0.62;
  const amrex::Real cse219 = state.T < 800.0;
  const amrex::Real cse220 = ((cse0) ? (
   2.0223626229831418e-8
)
: ((cse219) ? (
   1.3500000000000001e-9*cse218
)
: (
   3.0044907052141756e-9
)));
  const amrex::Real cse221 = cse220*state.xn[15];
  const amrex::Real cse222 = cse217 + cse221;
  const amrex::Real cse223 = std::exp(0.11*cse5);
  const amrex::Real cse224 = std::pow(state.T, 0.33900000000000002);
  const amrex::Real cse225 = ((cse0) ? (
   3.6831481232786064e-10
)
: ((cse1) ? (
   7.9513696852843416e-10*cse223/cse224
)
: (
   9.6735673353022342e-11
)));
  const amrex::Real cse226 = cse225*state.xn[15];
  const amrex::Real cse227 = 6.5199999999999996*crate;
  const amrex::Real cse228 = cse154*state.xn[2];
  const amrex::Real cse229 = cse116 + cse228;
  const amrex::Real cse230 = cse144*state.xn[16];
  const amrex::Real cse231 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse232 = cse230*cse231;
  const amrex::Real cse233 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse234 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse235 = cse107*state.xn[1];
  const amrex::Real cse236 = cse234 + cse235;
  const amrex::Real cse237 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse238 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse239 = cse2*state.xn[0];
  const amrex::Real cse240 = -cse239;
  const amrex::Real cse241 = cse116 + cse127;
  const amrex::Real cse242 = cse220*state.xn[6];
  const amrex::Real cse243 = cse225*state.xn[7];
  const amrex::Real cse244 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse245 = cse242 + cse243 + cse244;
  const amrex::Real cse246 = 3*cse24;
  const amrex::Real cse247 = cse13*((cse23) ? (
   0
)
: (
   -3*cse40 - 3*cse42
));
  const amrex::Real cse248 = cse53*(-cse246 - cse247);
  const amrex::Real cse249 = cse248*cse25;
  const amrex::Real cse250 = cse33*((cse31) ? (
   cse57*(cse174*cse246 + cse174*cse247 + cse248*cse52 + cse249*cse34)
)
: (
   0
));
  const amrex::Real cse251 = cse130*state.xn[2];
  const amrex::Real cse252 = ((cse31) ? (
   cse62*(cse186*cse246 + cse186*cse247 + cse248*cse8 + cse249*cse28)
)
: (
   0
));
  const amrex::Real cse253 = 6.8999999999999995*cse39 + 4.5*cse41;
  const amrex::Real cse254 = -cse253;
  const amrex::Real cse255 = cse254*cse92;
  const amrex::Real cse256 = 3.6750000000000002e-11*cse197;
  const amrex::Real cse257 = cse231*state.xn[11];
  const amrex::Real cse258 = cse144*cse257 + cse151*state.xn[2];
  const amrex::Real cse259 = -cse255 + cse256 + cse258;
  const amrex::Real cse260 = cse65*state.xn[0];
  const amrex::Real cse261 = cse193*state.xn[3];
  const amrex::Real cse262 = cse140*cse261;
  const amrex::Real cse263 = std::pow(state.T, -2);
  const amrex::Real cse264 = cse263*cse29;
  const amrex::Real cse265 = cse16*cse5;
  const amrex::Real cse266 = cse15*cse265;
  const amrex::Real cse267 = cse10*cse5;
  const amrex::Real cse268 = ((cse23) ? (
   0
)
: (
   cse19*cse9*(1.3*cse10*cse5 - 3.2400000000000002*cse266) + cse21*cse9*(0.65400000000000003*cse266 + 0.41599999999999998*cse267)
));
  const amrex::Real cse269 = cse13*cse268;
  const amrex::Real cse270 = cse269*cse53;
  const amrex::Real cse271 = cse24*cse268*cse37*cse53;
  const amrex::Real cse272 = cse33*((cse31) ? (
   cse57*(cse174*cse269 + 53300.0*cse264 + 1.6677785190126752e+29*cse27*cse48*(-2.3058963562359999e-34*cse46*std::pow(cse47, -6.6881000000000004)*cse50 + 3.2772577039999999e-25*cse51*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse51*std::pow(state.T, 3.1881000000000004))*std::exp(cse49)/cse46 - cse270*cse52 - cse271*cse34)
)
: (
   0
));
  const amrex::Real cse273 = 1.0/cse4;
  const amrex::Real cse274 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse275 = ((cse31) ? (
   cse62*(cse186*cse269 + 43900.0*cse264 + 149925037481.25937*cse27*cse273*(3.3349999999999998e-12*cse273*cse7 + 4.2414529999999999e-7*cse274*cse7)*std::exp(cse6) - cse270*cse8 - cse271*cse28)
)
: (
   0
));
  const amrex::Real cse276 = std::pow(state.T, -1.52);
  const amrex::Real cse277 = ((cse0) ? (
   0
)
: ((cse129) ? (
   -4.4014207378970674e-7*cse276
)
: (
   0
)));
  const amrex::Real cse278 = cse189*state.xn[3]*state.xn[8];
  const amrex::Real cse279 = cse278*std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse280 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse281 = cse278*cse280;
  const amrex::Real cse282 = cse273*cse69;
  const amrex::Real cse283 = cse138*cse282;
  const amrex::Real cse284 = cse166*state.xn[3];
  const amrex::Real cse285 = -3.7768240343347639e-14*cse167*cse284 + 7.3920000000000002e-11*cse284*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse286 = state.xn[10]*state.xn[2]*((cse0) ? (
   0
)
: ((cse129) ? (
   -6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)
)
: (
   0
)));
  const amrex::Real cse287 = state.xn[15]*state.xn[7]*((cse0) ? (
   0
)
: ((cse1) ? (
   -2.6955143233113918e-10*cse223*std::pow(state.T, -1.339) - 8.7465066538127767e-11*cse223*cse263/cse224
)
: (
   0
)));
  const amrex::Real cse288 = cse286 + cse287;
  const amrex::Real cse289 = std::sqrt(cse5);
  const amrex::Real cse290 = std::sqrt(3)*cse289*cse5;
  const amrex::Real cse291 = state.xn[15]*state.xn[6]*((cse0) ? (
   0
)
: ((cse219) ? (
   -1.7697487500000004e-8*cse290
)
: (
   0
)));
  const amrex::Real cse292 = 2.8349999999999999e-12*cse162*cse263*state.xn[3]*state.xn[6];
  const amrex::Real cse293 = cse291 + cse292;
  const amrex::Real cse294 = cse106*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998)) + 227*cse108*cse263*state.xn[1] - state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse295 = state.xn[0]*state.xn[2];
  const amrex::Real cse296 = cse295*((cse31) ? (
   7.9971871901929577e-69*cse101*std::pow(state.T, 12.536555999999999) + cse102*(-1.6313198799999999e-5*cse100*cse5 - 11.478657500000001*cse5*cse94 + 4.6894649399999997*cse5*cse95 - 1.1508224*cse5*cse96 + 0.1741279885*cse5*cse97 - 0.015791857020000001*cse5*cse98 + 0.00078368076500000001*cse5*cse99)
)
: (
   0
));
  const amrex::Real cse297 = cse5*state.xn[1]*state.xn[2];
  const amrex::Real cse298 = 0.5*cse5;
  const amrex::Real cse299 = cse5*std::log(cse74);
  const amrex::Real cse300 = -7.3220359950196314e-6*cse109*cse111*cse297 + 0.0005114031087522854*std::pow(cse110, -3.242)*cse297*std::pow(cse5, 1.907) + cse159*(-1.1557571281629217e-5*cse157*cse5 - cse82*(8.0679819999999989*cse77*std::pow(state.T, -0.98414000000000001) + cse79*(cse158*cse298 - 1.102e-5*cse299))) - cse296;
  const amrex::Real cse301 = cse294 + cse300;
  const amrex::Real cse302 = state.xn[0]*state.xn[14]*((cse0) ? (
   0
)
: ((cse1) ? (
   7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)
)
: (
   0
)));
  const amrex::Real cse303 = -cse302;
  const amrex::Real cse304 = 9.0566037735849048e-14*cse283;
  const amrex::Real cse305 = cse144*cse304;
  const amrex::Real cse306 = cse142*cse304;
  const amrex::Real cse307 = cse140*cse306;
  const amrex::Real cse308 = -cse307;
  const amrex::Real cse309 = cse303 + cse305 + cse308;
  const amrex::Real cse310 = state.xn[16]*state.xn[2];
  const amrex::Real cse311 = cse310*((cse0) ? (
   0
)
: ((cse129) ? (
   -2.3622303960273252e-7*cse276
)
: (
   0
)));
  const amrex::Real cse312 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse313 = cse230*cse312;
  const amrex::Real cse314 = 1.1264150943396225e-13*cse203*std::pow(state.T, -0.68999999999999995)*state.xn[3];
  const amrex::Real cse315 = cse311 - cse313 - cse314;
  const amrex::Real cse316 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse317 = -cse198;
  const amrex::Real cse318 = cse115 + cse88;
  const amrex::Real cse319 = crate*cse12;
  const amrex::Real cse320 = cse319*cse91;
  const amrex::Real cse321 = 5.4939999999999998e-7*cse75 + 0.49559999999999998;
  const amrex::Real cse322 = std::pow(cse74, -cse321);
  const amrex::Real cse323 = cse322*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse324 = 5115.0*cse323;
  const amrex::Real cse325 = cse324 + 1;
  const amrex::Real cse326 = std::pow(cse73, 1.512);
  const amrex::Real cse327 = 7.104740267838494e-7*cse326;
  const amrex::Real cse328 = cse325*cse327 + 1;
  const amrex::Real cse329 = 1.0/cse328;
  const amrex::Real cse330 = 5.5720000000000001e-12*cse329;
  const amrex::Real cse331 = d2g*state.xn[6];
  const amrex::Real cse332 = cse330*cse331;
  const amrex::Real cse333 = 1040.0*state.xn[7];
  const amrex::Real cse334 = crate*cse39;
  const amrex::Real cse335 = 0.00013329999999999999*cse75 + 0.81200000000000006;
  const amrex::Real cse336 = std::pow(cse74, -cse335);
  const amrex::Real cse337 = cse336*std::pow(state.T, 0.04845);
  const amrex::Real cse338 = 433.10000000000002*cse337;
  const amrex::Real cse339 = cse338 + 1;
  const amrex::Real cse340 = std::pow(cse73, 1.1279999999999999);
  const amrex::Real cse341 = 0.011078790689599145*cse340;
  const amrex::Real cse342 = cse339*cse341 + 1;
  const amrex::Real cse343 = 1.0/cse342;
  const amrex::Real cse344 = 4.5579999999999993e-11*cse343;
  const amrex::Real cse345 = d2g*state.xn[8];
  const amrex::Real cse346 = cse333*cse334 + cse344*cse345;
  const amrex::Real cse347 = 16800.0*state.xn[12];
  const amrex::Real cse348 = 7.538e-5*cse75 + 0.89639999999999997;
  const amrex::Real cse349 = std::pow(cse74, -cse348);
  const amrex::Real cse350 = cse349*std::pow(state.T, 1.635e-6);
  const amrex::Real cse351 = 43750.0*cse350;
  const amrex::Real cse352 = cse351 + 1;
  const amrex::Real cse353 = std::pow(cse73, 1.8740000000000001);
  const amrex::Real cse354 = 1.534817233071468e-7*cse353;
  const amrex::Real cse355 = cse352*cse354 + 1;
  const amrex::Real cse356 = 1.0/cse355;
  const amrex::Real cse357 = 2.1659999999999999e-12*cse356;
  const amrex::Real cse358 = d2g*state.xn[13];
  const amrex::Real cse359 = cse334*cse347 + cse357*cse358;
  const amrex::Real cse360 = cse332 + cse346 + cse359;
  const amrex::Real cse361 = cse114*cse319;
  const amrex::Real cse362 = cse360 - cse361;
  const amrex::Real cse363 = cse10*cse75;
  const amrex::Real cse364 = 0.044330000000000001*cse363 - 0.28520000000000001;
  const amrex::Real cse365 = cse363*cse364 - 1.6759999999999999;
  const amrex::Real cse366 = cse363*cse365 + 11.19;
  const amrex::Real cse367 = cse366*state.xn[6];
  const amrex::Real cse368 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse369 = cse325*cse326;
  const amrex::Real cse370 = -cse321;
  const amrex::Real cse371 = 5.5720000000000001e-12*cse113*state.xn[6]/std::pow(cse328, 2);
  const amrex::Real cse372 = cse367*cse368 + cse371*(0.0036340746469993896*cse323*cse326*cse370*cse69 + 1.0742367284971802e-6*cse369*cse69);
  const amrex::Real cse373 = std::exp(-15230.0*cse5);
  const amrex::Real cse374 = std::exp(-73.799999999999997*cse5);
  const amrex::Real cse375 = std::exp(-12.17*cse5);
  const amrex::Real cse376 = 1.6339999999999999e-6*cse373 + 9.7930000000000004e-9*cse374 + 6.3460000000000002e-9*cse375;
  const amrex::Real cse377 = 0.00071740367932898903*cse4 + 1;
  const amrex::Real cse378 = std::exp(-49550.0*cse5);
  const amrex::Real cse379 = 0.15970000000000001*cse378;
  const amrex::Real cse380 = cse379 + 1.7848999999999999;
  const amrex::Real cse381 = 12.244387999457523*cse4 + 1;
  const amrex::Real cse382 = 0.21509999999999996 - cse379;
  const amrex::Real cse383 = std::pow(cse377, -cse380)*std::pow(cse381, -cse382);
  const amrex::Real cse384 = 2.4460185352936309e-10*cse273*cse383;
  const amrex::Real cse385 = cse274*cse376 + cse384;
  const amrex::Real cse386 = cse339*cse340;
  const amrex::Real cse387 = -cse335;
  const amrex::Real cse388 = 4.5579999999999993e-11*cse113*state.xn[8]/std::pow(cse342, 2);
  const amrex::Real cse389 = cse385*state.xn[8] + cse388*(4.7982242476653898*cse337*cse340*cse387*cse69 + 0.012496875897867835*cse386*cse69);
  const amrex::Real cse390 = ((cse0) ? (
   3.5010967150614577e-11
)
: ((cse129) ? (
   1.4595000290776864e-10*std::pow(state.T, -0.62)
)
: (
   4.2499999999999999e-12
)));
  const amrex::Real cse391 = cse352*cse353;
  const amrex::Real cse392 = -cse348;
  const amrex::Real cse393 = 2.1659999999999999e-12*cse113*state.xn[13]/std::pow(cse355, 2);
  const amrex::Real cse394 = cse390*state.xn[13] + cse393*(0.0067148253946876721*cse350*cse353*cse392*cse69 + 2.8762474947759312e-7*cse391*cse69);
  const amrex::Real cse395 = cse148 + cse150;
  const amrex::Real cse396 = cse14*state.xn[2];
  const amrex::Real cse397 = cse396 < 1.0e-50;
  const amrex::Real cse398 = ((cse397) ? (
   0
)
: (
   -0.00011320754716981131*cse133*state.xn[3]
));
  const amrex::Real cse399 = ((cse397) ? (
   10000000000.0
)
: (
   cse135
));
  const amrex::Real cse400 = cse399 + 1;
  const amrex::Real cse401 = 1.0/cse400;
  const amrex::Real cse402 = cse257*state.xn[16];
  const amrex::Real cse403 = cse401*cse402;
  const amrex::Real cse404 = cse398*cse403;
  const amrex::Real cse405 = cse399*cse402/std::pow(cse400, 2);
  const amrex::Real cse406 = cse398*cse405;
  const amrex::Real cse407 = cse404 - cse406;
  const amrex::Real cse408 = cse153 + cse407;
  const amrex::Real cse409 = cse130*state.xn[16] + cse408;
  const amrex::Real cse410 = cse329*cse331;
  const amrex::Real cse411 = 1.1144e-11*cse410;
  const amrex::Real cse412 = 2*cse68;
  const amrex::Real cse413 = cse201*cse319;
  const amrex::Real cse414 = crate*cse14;
  const amrex::Real cse415 = cse334*state.xn[12];
  const amrex::Real cse416 = cse356*cse358;
  const amrex::Real cse417 = 33600.0*cse415 + 4.3319999999999997e-12*cse416;
  const amrex::Real cse418 = -cse347*cse414 + cse417;
  const amrex::Real cse419 = 3.2999999999999996e-13*cse190;
  const amrex::Real cse420 = cse419*state.xn[8];
  const amrex::Real cse421 = -cse333*cse414;
  const amrex::Real cse422 = cse334*state.xn[7];
  const amrex::Real cse423 = cse343*cse345;
  const amrex::Real cse424 = 2080.0*cse422 + 9.1159999999999986e-11*cse423;
  const amrex::Real cse425 = cse420 + cse421 + cse424;
  const amrex::Real cse426 = ((cse397) ? (
   0
)
: (
   cse134
));
  const amrex::Real cse427 = cse403*cse426;
  const amrex::Real cse428 = cse405*cse426;
  const amrex::Real cse429 = cse427 - cse428;
  const amrex::Real cse430 = cse196 + cse429;
  const amrex::Real cse431 = cse194 - cse195;
  const amrex::Real cse432 = cse214*cse319;
  const amrex::Real cse433 = -cse432;
  const amrex::Real cse434 = 1.1000000000000001*crate;
  const amrex::Real cse435 = cse368*state.xn[2];
  const amrex::Real cse436 = cse113*cse330 + cse366*cse435;
  const amrex::Real cse437 = std::pow(cse5, 0.00231);
  const amrex::Real cse438 = std::pow(state.T, -1.5);
  const amrex::Real cse439 = std::exp(-8110.0*cse5);
  const amrex::Real cse440 = std::exp(-26.600000000000001*cse5);
  const amrex::Real cse441 = std::exp(-7.6200000000000001*cse5);
  const amrex::Real cse442 = std::exp(-1.3799999999999999*cse5);
  const amrex::Real cse443 = 0.00015100000000000001*cse439 + 1.31e-7*cse440 + 3.4e-8*cse441 + 6.9699999999999997e-9*cse442;
  const amrex::Real cse444 = 1.0537934368240927e-9*cse437 + cse438*cse443;
  const amrex::Real cse445 = cse444*state.xn[16];
  const amrex::Real cse446 = -cse445;
  const amrex::Real cse447 = state.c_hat*state.rn[0];
  const amrex::Real cse448 = 7.1303347066783361e-18*cse447;
  const amrex::Real cse449 = cse14*state.xn[3];
  const amrex::Real cse450 = crate*cse449;
  const amrex::Real cse451 = 3.8500000000000001*crate + cse448 + 1040.0*cse450;
  const amrex::Real cse452 = cse419*state.xn[3];
  const amrex::Real cse453 = cse113*cse344 + cse385*state.xn[2];
  const amrex::Real cse454 = cse452 + cse453;
  const amrex::Real cse455 = cse399*cse401;
  const amrex::Real cse456 = cse455*state.xn[16];
  const amrex::Real cse457 = cse231*cse456;
  const amrex::Real cse458 = cse232 + cse457;
  const amrex::Real cse459 = -cse458;
  const amrex::Real cse460 = 2.0740886949963892e-18*cse447;
  const amrex::Real cse461 = 16800.0*cse450 + cse460;
  const amrex::Real cse462 = cse113*cse357 + cse390*state.xn[2];
  const amrex::Real cse463 = -cse462;
  const amrex::Real cse464 = -cse116 - cse362;
  const amrex::Real cse465 = cse257*cse455;
  const amrex::Real cse466 = cse444*state.xn[7];
  const amrex::Real cse467 = cse254*cse319;
  const amrex::Real cse468 = -cse467;
  const amrex::Real cse469 = 1.6715999999999999e-11*cse410;
  const amrex::Real cse470 = 3120.0*cse422 + 1.3673999999999998e-10*cse423;
  const amrex::Real cse471 = 50400.0*cse415 + 6.4979999999999992e-12*cse416;
  const amrex::Real cse472 = cse261 + cse262;
  const amrex::Real cse473 = -cse472;
  const amrex::Real cse474 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse437*cse5 + cse438*(1.2246100000000002*cse263*cse439 + 3.4846000000000001e-6*cse263*cse440 + 2.5908000000000001e-7*cse263*cse441 + 9.6185999999999982e-9*cse263*cse442) - 1.5*cse443*std::pow(state.T, -2.5));
  const amrex::Real cse475 = -cse306;
  const amrex::Real cse476 = cse277*cse310;
  const amrex::Real cse477 = cse393*(-cse354*(0.071531250000000005*cse349*std::pow(state.T, -0.99999836499999994) + cse351*(cse298*cse392 - 7.538e-5*cse299)) - 1.4381237473879656e-7*cse391*cse5) + state.xn[13]*state.xn[2]*((cse0) ? (
   0
)
: ((cse129) ? (
   -9.0489001802816551e-11*std::pow(state.T, -1.6200000000000001)
)
: (
   0
)));
  const amrex::Real cse478 = -4.9999999999999997e-12*cse367*cse438*state.xn[2] + cse371*(-cse327*(0.0019963845000000001*cse322*std::pow(state.T, -0.99999960970000001) + cse324*(cse298*cse370 - 5.4939999999999998e-7*cse299)) - 5.3711836424859012e-7*cse369*cse5) + cse435*state.xn[6]*(cse267*cse365 + cse363*(0.044330000000000001*cse265*cse75 + cse267*cse364));
  const amrex::Real cse479 = 7913.1350000000002*cse263*cse378;
  const amrex::Real cse480 = cse388*(-cse341*(20.983695000000001*cse336*std::pow(state.T, -0.95155000000000001) + cse338*(cse298*cse387 - 0.00013329999999999999*cse299)) - 0.0062484379489339175*cse386*cse5) + state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse274*cse383 + cse274*(0.024885819999999999*cse263*cse373 + 7.2272340000000005e-7*cse263*cse374 + 7.7230820000000007e-8*cse263*cse375) - 3.0/2.0*cse376/std::pow(state.T, 5.0/2.0) + cse384*(-0.00035870183966449452*cse273*cse380/cse377 - cse479*std::log(cse377)) + cse384*(-6.1221939997287613*cse273*cse382/cse381 + cse479*std::log(cse381)));
  const amrex::Real cse481 = -4.2899999999999994e-13*cse189*cse280*state.xn[3]*state.xn[8] + 7.5899999999999998e-12*cse279;
  const amrex::Real cse482 = cse480 + cse481;
  const amrex::Real cse483 = ((cse397) ? (
   0
)
: (
   5.6603773584905654e-5*cse282*state.xn[3]
));
  const amrex::Real cse484 = cse403*cse483;
  const amrex::Real cse485 = cse312*cse456;
  const amrex::Real cse486 = cse405*cse483;
  const amrex::Real cse487 = cse484 - cse485 - cse486;
  const amrex::Real cse488 = 2.0740886949963892e-18*state.c_hat;
  const amrex::Real cse489 = cse488*state.xn[12];
  const amrex::Real cse490 = 7.1303347066783361e-18*state.c_hat;
  const amrex::Real cse491 = cse490*state.xn[7];
  const amrex::Real cse492 = cse489 + cse491;
  const amrex::Real cse493 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse494 = cse32*state.xn[3];
  const amrex::Real cse495 = cse33*cse58;
  const amrex::Real cse496 = -cse3;
  const amrex::Real cse497 = state.xn[0]*state.xn[3];
  const amrex::Real cse498 = cse497*cse63;
  const amrex::Real cse499 = cse320 - cse64;
  const amrex::Real cse500 = cse125*cse497;
  const amrex::Real cse501 = cse123 + cse361 + cse493 + cse500;
  const amrex::Real cse502 = -cse501;
  const amrex::Real cse503 = cse188*cse497;
  const amrex::Real cse504 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse505 = cse164 + cse504;
  const amrex::Real cse506 = cse412 + cse413;
  const amrex::Real cse507 = cse169 + cse204 - cse427 + cse428;
  const amrex::Real cse508 = cse196 + cse431;
  const amrex::Real cse509 = cse210*cse497;
  const amrex::Real cse510 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse511 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse512 = 9.0000000000000011e-15*cse170;
  const amrex::Real cse513 = cse252*cse497;
  const amrex::Real cse514 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse515 = cse258 + cse465 + cse466 + cse514;
  const amrex::Real cse516 = cse275*cse497;
  const amrex::Real cse517 = -cse474;
  const amrex::Real cse518 = -cse484 + cse485 + cse486;
  const amrex::Real cse519 = cse285 - cse311 + cse313 + cse314 + cse517 + cse518;
  const amrex::Real cse520 = state.c_hat*state.xn[3];
  const amrex::Real cse521 = 1.4949836836302895e-18*cse520;
  const amrex::Real cse522 = cse411 + cse505;
  const amrex::Real cse523 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse524 = cse222 + cse436 + cse510 + cse523;
  const amrex::Real cse525 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse526 = cse242 + cse332;
  const amrex::Real cse527 = cse293 + cse478;
  const amrex::Real cse528 = -cse332;
  const amrex::Real cse529 = 184.0*state.xn[9];
  const amrex::Real cse530 = cse334*cse529;
  const amrex::Real cse531 = -cse530;
  const amrex::Real cse532 = cse346 + cse531;
  const amrex::Real cse533 = cse414*cse529;
  const amrex::Real cse534 = 9.8999999999999995e-14*cse190;
  const amrex::Real cse535 = cse334*state.xn[9];
  const amrex::Real cse536 = 368.0*cse535;
  const amrex::Real cse537 = -cse536;
  const amrex::Real cse538 = cse424 + cse537;
  const amrex::Real cse539 = 1.5094998137029966e-18*cse447;
  const amrex::Real cse540 = 184.0*cse450 + cse539;
  const amrex::Real cse541 = cse238 + cse239;
  const amrex::Real cse542 = 552.0*cse535;
  const amrex::Real cse543 = 1.5094998137029966e-18*state.c_hat;
  const amrex::Real cse544 = cse543*state.xn[9];
  const amrex::Real cse545 = -cse544;
  const amrex::Real cse546 = -cse346;
  const amrex::Real cse547 = ((cse0) ? (
   1.3707124444663516e-8
)
: ((cse219) ? (
   9.1500000000000005e-10*cse218
)
: (
   2.0363770335340524e-9
)));
  const amrex::Real cse548 = cse547*state.xn[15];
  const amrex::Real cse549 = cse547*state.xn[8];
  const amrex::Real cse550 = state.xn[15]*state.xn[8]*((cse0) ? (
   0
)
: ((cse219) ? (
   -1.1994963750000002e-8*cse290
)
: (
   0
)));
  const amrex::Real cse551 = -cse533 + cse536;
  const amrex::Real cse552 = cse227 + cse511;
  const amrex::Real cse553 = cse525 + cse540;
  const amrex::Real cse554 = cse237 + cse530;
  const amrex::Real cse555 = -cse66;
  const amrex::Real cse556 = -cse233 + cse457;
  const amrex::Real cse557 = cse294 + cse306;
  const amrex::Real cse558 = -cse359;
  const amrex::Real cse559 = 2.3099999999999997e-13*cse190;
  const amrex::Real cse560 = cse291 - cse305 + cse307;
  const amrex::Real cse561 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse562 = cse74 + 50;
  const amrex::Real cse563 = 0.0043*std::pow(cse562, 0.69199999999999995) + 1;
  const amrex::Real cse564 = 0.049959999999999997*std::pow(cse562, 0.43099999999999999)*cse563 + 1;
  const amrex::Real cse565 = 1.0/cse564;
  const amrex::Real cse566 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse567 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse568 = 1.0/cse567;
  const amrex::Real cse569 = cse568*state.xn[2];
  const amrex::Real cse570 = cse566*cse569;
  const amrex::Real cse571 = 4.4460389837719569e-9*cse570 + 626500000.0;
  const amrex::Real cse572 = 1.0/cse571;
  const amrex::Real cse573 = std::exp(-118400.0*cse5);
  const amrex::Real cse574 = cse572*cse573;
  const amrex::Real cse575 = 1.3338116951315871e-8*cse574;
  const amrex::Real cse576 = cse570*cse575 + 1;
  const amrex::Real cse577 = 1.0/cse576;
  const amrex::Real cse578 = cse574*cse577;
  const amrex::Real cse579 = 1.3659996292638161e-10*cse578;
  const amrex::Real cse580 = 3.2000000000000002e-32*cse4;
  const amrex::Real cse581 = psigd_coll_interp2d(cse13, state.T);
  const amrex::Real cse582 = 100.0*d2g;
  const amrex::Real cse583 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse72 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse584 = cse113*cse580;
  const amrex::Real cse585 = -cse581*cse582 + cse583*cse584;
  const amrex::Real cse586 = (((cse585 > 0) ? (
   0
)
: ((cse585 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse587 = cse583*cse586*d2g;
  const amrex::Real cse588 = cse580*cse587;
  const amrex::Real cse589 = (((cse585 < 0) ? (
   0
)
: ((cse585 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse590 = cse589*psigd_coll_interp2d_partial_0(cse13, state.T);
  const amrex::Real cse591 = cse582*cse590;
  const amrex::Real cse592 = 0.001*state.T;
  const amrex::Real cse593 = cse592 + 1;
  const amrex::Real cse594 = std::pow(cse593, -2);
  const amrex::Real cse595 = 1.0/cse593;
  const amrex::Real cse596 = std::pow(10, -2.0230000000000001*cse594 - 3.6709999999999998*cse595 - 11.08);
  const amrex::Real cse597 = cse14*cse596;
  const amrex::Real cse598 = cse597*state.xn[3];
  const amrex::Real cse599 = std::pow(10, -2.3900000000000001*cse594 + 0.055500000000000001*cse595 - 11.06);
  const amrex::Real cse600 = cse14*cse599;
  const amrex::Real cse601 = cse600*state.xn[0];
  const amrex::Real cse602 = cse598 + cse601;
  const amrex::Real cse603 = 1.0/cse602;
  const amrex::Real cse604 = 5.6999999999999997e-11*chi*std::exp(-4.1799999999999997*av) + 1.9999999999999999e-7;
  const amrex::Real cse605 = cse603*cse604;
  const amrex::Real cse606 = cse14*cse605 + 1;
  const amrex::Real cse607 = 1.0/cse606;
  const amrex::Real cse608 = 6.7291434000000005e-12*cse607 + 3.2043540000000002e-13;
  const amrex::Real cse609 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse610 = 1.75*cse609;
  const amrex::Real cse611 = std::exp(-state.T);
  const amrex::Real cse612 = 1 - cse611;
  const amrex::Real cse613 = std::pow(cse612, 1000);
  const amrex::Real cse614 = l0_co_interp1d(state.T);
  const amrex::Real cse615 = 1.0/cse614;
  const amrex::Real cse616 = 680.10000000000002/cse609;
  const amrex::Real cse617 = cse610*state.xn[0] + cse616*state.xn[2] + state.xn[3];
  const amrex::Real cse618 = M_SQRT2;
  const amrex::Real cse619 = std::max(gradv, 5.7445531974402559e-18*cse4*cse618);
  const amrex::Real cse620 = 1.0/cse619;
  const amrex::Real cse621 = 100000.0*cse620;
  const amrex::Real cse622 = cse621*state.xn[9];
  const amrex::Real cse623 = llte_co_interp2d(state.T, cse622);
  const amrex::Real cse624 = 1.0/cse623;
  const amrex::Real cse625 = nhalf_co_interp2d(state.T, cse622);
  const amrex::Real cse626 = cse614*cse624;
  const amrex::Real cse627 = -cse625*cse626 + 1;
  const amrex::Real cse628 = 1.0/cse625;
  const amrex::Real cse629 = cse617*cse628;
  const amrex::Real cse630 = alpha_co_interp2d(state.T, cse622);
  const amrex::Real cse631 = std::pow(cse629, cse630);
  const amrex::Real cse632 = cse615*cse631;
  const amrex::Real cse633 = cse627*cse632;
  const amrex::Real cse634 = cse615 + cse617*cse624 + cse633;
  const amrex::Real cse635 = 1.0/cse634;
  const amrex::Real cse636 = cse613*cse635;
  const amrex::Real cse637 = cse636*state.xn[9];
  const amrex::Real cse638 = cse289*state.xn[2];
  const amrex::Real cse639 = (1.0/100.0)*state.T;
  const amrex::Real cse640 = std::log(cse639);
  const amrex::Real cse641 = 0.0086999999999999994*cse640;
  const amrex::Real cse642 = std::pow(cse639, cse641 + 0.12809999999999999);
  const amrex::Real cse643 = 7.5799999999999997e-10*cse642*state.xn[0];
  const amrex::Real cse644 = state.T < 500;
  const amrex::Real cse645 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse646 = 0.75*((cse644) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse645
)) + 0.25*((cse644) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse645
));
  const amrex::Real cse647 = 4.5299999999999998e-6*cse638 + cse643 + cse646*state.xn[3];
  const amrex::Real cse648 = cse647 + 2.3e-6;
  const amrex::Real cse649 = std::exp(-91.209999999999994*cse5);
  const amrex::Real cse650 = cse649/cse648;
  const amrex::Real cse651 = 2*cse650;
  const amrex::Real cse652 = cse647*cse651 + 1;
  const amrex::Real cse653 = 1.0/cse652;
  const amrex::Real cse654 = cse650*state.xn[8];
  const amrex::Real cse655 = cse653*cse654;
  const amrex::Real cse656 = cse396 + 0.070000000000000007;
  const amrex::Real cse657 = 1.0/cse656;
  const amrex::Real cse658 = std::sqrt(cse396*cse657);
  const amrex::Real cse659 = 26.399999999999999*cse658;
  const amrex::Real cse660 = cse659 + 6.5;
  const amrex::Real cse661 = cse14*cse660;
  const amrex::Real cse662 = cse10*std::log(cse13);
  const amrex::Real cse663 = cse662 < 2;
  const amrex::Real cse664 = cse662 < 4;
  const amrex::Real cse665 = cse662 < 7;
  const amrex::Real cse666 = cse662 < 10;
  const amrex::Real cse667 = ((cse663) ? (
   10.0
)
: ((cse664) ? (
   (3.0/2.0)*cse662 + 7
)
: ((cse665) ? (
   (4.0/3.0)*cse662 + 23.0/3.0
)
: ((cse666) ? (
   (1.0/3.0)*cse662 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse668 = cse14*cse667;
  const amrex::Real cse669 = cse661*state.xn[0] + cse668*state.xn[3];
  const amrex::Real cse670 = std::pow(cse648, -2);
  const amrex::Real cse671 = cse630/cse617;
  const amrex::Real cse672 = cse633*cse671;
  const amrex::Real cse673 = cse617*state.xn[9];
  const amrex::Real cse674 = cse613*cse673/std::pow(cse634, 2);
  const amrex::Real cse675 = std::pow(cse67, -2);
  const amrex::Real cse676 = std::pow(cse606, -2);
  const amrex::Real cse677 = cse38*cse605;
  const amrex::Real cse678 = cse39*cse596;
  const amrex::Real cse679 = cse41*cse599;
  const amrex::Real cse680 = cse678 + cse679;
  const amrex::Real cse681 = cse604/std::pow(cse602, 2);
  const amrex::Real cse682 = cse14*cse681;
  const amrex::Real cse683 = cse677 - cse682*(-cse600 + cse680);
  const amrex::Real cse684 = 1.5159999999999999e-9*cse642;
  const amrex::Real cse685 = cse647*cse649*cse670;
  const amrex::Real cse686 = cse647*cse650;
  const amrex::Real cse687 = 5.7927337833400002e-20*cse686*state.xn[8]/std::pow(cse652, 2);
  const amrex::Real cse688 = cse41*cse660;
  const amrex::Real cse689 = cse38*state.xn[2];
  const amrex::Real cse690 = std::pow(cse656, -2);
  const amrex::Real cse691 = 0.5*cse690;
  const amrex::Real cse692 = cse131/std::pow(cse13, 3);
  const amrex::Real cse693 = -0.5*cse657*cse689 + cse691*cse692;
  const amrex::Real cse694 = cse656*cse659*cse69*state.xn[0];
  const amrex::Real cse695 = cse10*cse14;
  const amrex::Real cse696 = ((cse663) ? (
   0
)
: ((cse664) ? (
   (3.0/2.0)*cse695
)
: ((cse665) ? (
   (4.0/3.0)*cse695
)
: ((cse666) ? (
   (1.0/3.0)*cse695
)
: (
   0
)))));
  const amrex::Real cse697 = cse39*cse667;
  const amrex::Real cse698 = cse449*cse696 + cse661 - cse688 + cse693*cse694 - cse697;
  const amrex::Real cse699 = 1.0/cse67;
  const amrex::Real cse700 = state.T < 100.0;
  const amrex::Real cse701 = state.T < 6000.0;
  const amrex::Real cse702 = state.T < 10;
  const amrex::Real cse703 = std::log(cse592);
  const amrex::Real cse704 = std::pow(cse703, 2);
  const amrex::Real cse705 = std::pow(cse703, 3);
  const amrex::Real cse706 = std::pow(cse703, 4);
  const amrex::Real cse707 = std::pow(cse703, 5);
  const amrex::Real cse708 = std::pow(cse703, 6);
  const amrex::Real cse709 = std::pow(cse703, 7);
  const amrex::Real cse710 = std::pow(cse703, 8);
  const amrex::Real cse711 = std::pow(10.0, 16.815729999999999*cse703 + 96.743155000000002*cse704 + 343.1918*cse705 + 734.71650999999997*cse706 + 983.67575999999997*cse707 + 801.81246999999996*cse708 + 364.14445999999998*cse709 + 70.609154000000004*cse710 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse712 = state.xn[5] + state.xn[6];
  const amrex::Real cse713 = std::pow(10.0, 1.5714710999999999*cse703 + 0.015391166*cse704 - 0.23619984999999999*cse705 - 0.51002221000000003*cse706 + 0.32168730000000001*cse707 - 22.089523)*state.xn[1] + std::pow(10.0, 2.0943374000000001*cse703 - 0.77151435999999995*cse704 + 0.43693353000000001*cse705 - 0.14913216000000001*cse706 - 0.033638326000000003*cse707 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 2.1892372*cse703 - 0.81520437999999995*cse704 + 0.29036281000000003*cse705 - 0.16596184*cse706 + 0.19191374999999999*cse707 - 23.689236999999999)*cse712;
  const amrex::Real cse714 = std::pow(10, 37.383713*cse703 + 58.145166000000003*cse704 + 48.656103000000002*cse705 + 20.159831000000001*cse706 + 3.8479610000000002*cse707 - 16.818342000000001)*state.xn[0] + cse713;
  const amrex::Real cse715 = std::pow(10.0, 1.6802758*cse703 + 0.93310621999999999*cse704 + 4.0406627000000004*cse705 - 4.7274035999999997*cse706 - 8.8077017000000009*cse707 + 8.9167182999999994*cse708 + 6.4380698000000001*cse709 - 6.3701156000000001*cse710 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse716 = state.T < 1000.0;
  const amrex::Real cse717 = std::pow(10.0, 3.5692468000000002*cse703 - 11.33286*cse704 - 27.850082*cse705 - 21.328264000000001*cse706 - 4.2519023000000002*cse707 - 24.311209000000002)*state.xn[0] + cse713;
  const amrex::Real cse718 = std::pow(10.0, 4.6450521*cse703 - 3.7209846*cse704 + 5.9369081000000001*cse705 - 5.5108049000000001*cse706 + 1.5538288*cse707 - 24.311209000000002)*state.xn[0] + cse713;
  const amrex::Real cse719 = 1.2972734453571459e-22*state.xn[0] + 1.2615234450058559e-20*state.xn[1] + 4.6629930528575411e-23*state.xn[3] + 1.3183594641975411e-19*state.xn[5] + 1.3183594641975411e-19*state.xn[6];
  const amrex::Real cse720 = ((cse644 && cse700 && cse701) ? (
   cse702 || cse711 + cse714 < 1.0e-100
)
: (
   ((cse700 && cse701) ? (
      cse702 || cse714 + cse715 < 1.0e-100
   )
   : (
      ((cse644 && cse701 && cse716) ? (
         cse702 || cse711 + cse717 < 1.0e-100
      )
      : (
         ((cse701 && cse716) ? (
            cse702 || cse715 + cse717 < 1.0e-100
         )
         : (
            ((cse644 && cse701) ? (
               cse702 || cse711 + cse718 < 1.0e-100
            )
            : (
               ((cse701) ? (
                  cse702 || cse715 + cse718 < 1.0e-100
               )
               : (
                  ((cse644) ? (
                     cse702 || cse719 + 3.0478204245257306e+83651*state.xn[2] < 1.0e-100
                  )
                  : (
                     cse702 || cse719 + 1.0073452104972926e-205*state.xn[2] < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse721 = std::log(0.001*((cse701) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse722 = std::pow(cse721, 2);
  const amrex::Real cse723 = std::pow(cse721, 3);
  const amrex::Real cse724 = std::pow(cse721, 4);
  const amrex::Real cse725 = std::pow(cse721, 5);
  const amrex::Real cse726 = std::pow(10, 37.383713*cse721 + 58.145166000000003*cse722 + 48.656103000000002*cse723 + 20.159831000000001*cse724 + 3.8479610000000002*cse725 - 16.818342000000001);
  const amrex::Real cse727 = ((cse701) ? (
   cse700
)
: (
   false
));
  const amrex::Real cse728 = std::pow(10.0, 3.5692468000000002*cse721 - 11.33286*cse722 - 27.850082*cse723 - 21.328264000000001*cse724 - 4.2519023000000002*cse725 - 24.311209000000002);
  const amrex::Real cse729 = ((cse701) ? (
   cse716
)
: (
   false
));
  const amrex::Real cse730 = std::pow(10.0, 4.6450521*cse721 - 3.7209846*cse722 + 5.9369081000000001*cse723 - 5.5108049000000001*cse724 + 1.5538288*cse725 - 24.311209000000002);
  const amrex::Real cse731 = ((cse727) ? (
   cse726
)
: ((cse729) ? (
   cse728
)
: (
   cse730
)));
  const amrex::Real cse732 = std::pow(10.0, 2.0943374000000001*cse721 - 0.77151435999999995*cse722 + 0.43693353000000001*cse723 - 0.14913216000000001*cse724 - 0.033638326000000003*cse725 - 23.962112000000001);
  const amrex::Real cse733 = cse732*state.xn[3];
  const amrex::Real cse734 = std::pow(10.0, 1.5714710999999999*cse721 + 0.015391166*cse722 - 0.23619984999999999*cse723 - 0.51002221000000003*cse724 + 0.32168730000000001*cse725 - 22.089523);
  const amrex::Real cse735 = cse734*state.xn[1];
  const amrex::Real cse736 = std::pow(10.0, 2.1892372*cse721 - 0.81520437999999995*cse722 + 0.29036281000000003*cse723 - 0.16596184*cse724 + 0.19191374999999999*cse725 - 23.689236999999999);
  const amrex::Real cse737 = cse712*cse736;
  const amrex::Real cse738 = std::pow(cse721, 6);
  const amrex::Real cse739 = std::pow(cse721, 7);
  const amrex::Real cse740 = std::pow(cse721, 8);
  const amrex::Real cse741 = std::pow(10.0, 16.815729999999999*cse721 + 96.743155000000002*cse722 + 343.1918*cse723 + 734.71650999999997*cse724 + 983.67575999999997*cse725 + 801.81246999999996*cse738 + 364.14445999999998*cse739 + 70.609154000000004*cse740 - 21.928795999999998);
  const amrex::Real cse742 = std::pow(10.0, 1.6802758*cse721 + 0.93310621999999999*cse722 + 4.0406627000000004*cse723 - 4.7274035999999997*cse724 - 8.8077017000000009*cse725 + 8.9167182999999994*cse738 + 6.4380698000000001*cse739 - 6.3701156000000001*cse740 - 22.921188999999998);
  const amrex::Real cse743 = ((cse644) ? (
   cse741
)
: (
   cse742
));
  const amrex::Real cse744 = cse731*state.xn[0] + cse733 + cse735 + cse737 + cse743*state.xn[2];
  const amrex::Real cse745 = std::pow(cse744, -2);
  const amrex::Real cse746 = ((cse701) ? (
   cse5
)
: (
   0.00016666666666666666
));
  const amrex::Real cse747 = std::exp(-11700.0*cse746);
  const amrex::Real cse748 = std::exp(-5860.0*cse746);
  const amrex::Real cse749 = std::exp(-510.0*cse746);
  const amrex::Real cse750 = std::exp(-2197000.0*((cse701) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse751 = cse750*((cse701) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse752 = 6.0142468035272636e-8*((cse701) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse753 = 1.0/cse752;
  const amrex::Real cse754 = 1.6e-18*cse747 + 6.7e-19*cse748 + 3.0e-24*cse749 + 4.985670872372847e-33*cse751*cse753;
  const amrex::Real cse755 = 1.0/cse744;
  const amrex::Real cse756 = cse754*cse755 + 1;
  const amrex::Real cse757 = std::pow(cse756, -2);
  const amrex::Real cse758 = cse745*std::pow(cse754, 2)*cse757;
  const amrex::Real cse759 = 0.0060000000000000001*cse640;
  const amrex::Real cse760 = std::pow(cse639, 0.36899999999999999 - cse759);
  const amrex::Real cse761 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse762 = 1.08e-14*cse761;
  const amrex::Real cse763 = 0.16*cse640;
  const amrex::Real cse764 = std::pow(cse639, 0.755 - cse763);
  const amrex::Real cse765 = 4.34e-10*cse764;
  const amrex::Real cse766 = cse765*state.xn[0];
  const amrex::Real cse767 = 0.042999999999999997*cse640;
  const amrex::Real cse768 = 5.2499999999999998e-13*std::pow(cse639, cse767 + 0.88900000000000001);
  const amrex::Real cse769 = 0.52500000000000002*cse640;
  const amrex::Real cse770 = 2.2500000000000003e-12*std::pow(cse639, cse769 + 1.198);
  const amrex::Real cse771 = cse768 + cse770;
  const amrex::Real cse772 = cse762*state.xn[2] + cse766 + cse771*state.xn[3];
  const amrex::Real cse773 = std::exp(-425.40000000000003*cse5);
  const amrex::Real cse774 = cse772*cse773;
  const amrex::Real cse775 = std::exp(-326.60000000000002*cse5);
  const amrex::Real cse776 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse777 = 4.8599999999999998e-10*cse776;
  const amrex::Real cse778 = 3.1899999999999998e-10*cse760;
  const amrex::Real cse779 = cse778*state.xn[0];
  const amrex::Real cse780 = 0.041000000000000002*cse640;
  const amrex::Real cse781 = 4.7499999999999998e-11*std::pow(cse639, cse780 + 0.20300000000000001);
  const amrex::Real cse782 = 0.058000000000000003*cse640;
  const amrex::Real cse783 = 1.6725000000000002e-10*std::pow(cse639, cse782 + 0.23699999999999999);
  const amrex::Real cse784 = cse781 + cse783;
  const amrex::Real cse785 = cse777*state.xn[2] + cse779 + cse784*state.xn[3];
  const amrex::Real cse786 = cse775*cse785;
  const amrex::Real cse787 = 0.0030000000000000001*cse640;
  const amrex::Real cse788 = std::pow(cse639, 0.41899999999999998 - cse787);
  const amrex::Real cse789 = 7.1400000000000007e-11*cse788;
  const amrex::Real cse790 = cse772*cse775;
  const amrex::Real cse791 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse792 = 5.1199999999999999e-10*cse791;
  const amrex::Real cse793 = 3.5700000000000001e-10*cse788;
  const amrex::Real cse794 = cse793*state.xn[0];
  const amrex::Real cse795 = 0.025000000000000001*cse640;
  const amrex::Real cse796 = 3.7250000000000001e-11*std::pow(cse639, cse795 + 0.26400000000000001);
  const amrex::Real cse797 = 1.0274999999999999e-10*std::pow(cse639, cse767 + 0.29599999999999999);
  const amrex::Real cse798 = cse796 + cse797;
  const amrex::Real cse799 = cse792*state.xn[2] + cse794 + cse798*state.xn[3];
  const amrex::Real cse800 = cse775*cse799;
  const amrex::Real cse801 = 8.68e-11*cse764;
  const amrex::Real cse802 = cse773*cse785;
  const amrex::Real cse803 = cse799 + 8.9099999999999997e-5;
  const amrex::Real cse804 = cse775*cse803;
  const amrex::Real cse805 = 6.3800000000000002e-11*cse760;
  const amrex::Real cse806 = 2.1266666666666666e-11*cse760*cse774 + 2.8933333333333332e-11*cse764*cse802 + cse786*cse789 + cse789*cse790 + cse800*cse801 + cse804*cse805;
  const amrex::Real cse807 = cse772 + 1.7499999999999998e-5;
  const amrex::Real cse808 = cse785 + 1.34e-10;
  const amrex::Real cse809 = std::exp(-98.800000000000011*cse5);
  const amrex::Real cse810 = cse772*cse809;
  const amrex::Real cse811 = (1.0/3.0)*cse810;
  const amrex::Real cse812 = (1.0/5.0)*cse775;
  const amrex::Real cse813 = cse772*cse812;
  const amrex::Real cse814 = (1.0/15.0)*cse774;
  const amrex::Real cse815 = cse785*cse812;
  const amrex::Real cse816 = cse785*cse814 + cse799*cse813 + cse803*cse815;
  const amrex::Real cse817 = std::exp(-227.80000000000001*cse5);
  const amrex::Real cse818 = cse799*cse817;
  const amrex::Real cse819 = (3.0/5.0)*cse818;
  const amrex::Real cse820 = cse807*cse815 + cse807*cse819 + cse808*cse819;
  const amrex::Real cse821 = cse803*cse807 + cse803*cse808 + cse808*cse811 + cse816 + cse820;
  const amrex::Real cse822 = 1.0/cse821;
  const amrex::Real cse823 = 2.3872025442750958e-19*cse822;
  const amrex::Real cse824 = cse775*cse807;
  const amrex::Real cse825 = 2.1419999999999999e-10*cse788;
  const amrex::Real cse826 = cse808*cse817;
  const amrex::Real cse827 = cse807*cse817;
  const amrex::Real cse828 = 1.9139999999999998e-10*cse760*cse818 + 2.604e-10*cse764*cse818 + cse786*cse801 + cse805*cse824 + cse825*cse826 + cse825*cse827;
  const amrex::Real cse829 = 2.8023005140020001e-18*cse822;
  const amrex::Real cse830 = std::pow(cse821, -2);
  const amrex::Real cse831 = cse808*cse809;
  const amrex::Real cse832 = cse830*(-1.0633333333333333e-10*cse760*cse810 - 1.4466666666666666e-10*cse764*cse831 - cse765*cse803 - cse778*cse803 - cse793*cse807 - cse793*cse808 - cse806 - cse828);
  const amrex::Real cse833 = 2.3872025442750958e-19*cse816;
  const amrex::Real cse834 = 2.8023005140020001e-18*cse820;
  const amrex::Real cse835 = 0.045999999999999999*cse640;
  const amrex::Real cse836 = std::pow(cse639, cse835 + 0.22800000000000001);
  const amrex::Real cse837 = std::pow(cse639, cse835 + 0.23100000000000001);
  const amrex::Real cse838 = 2.6400000000000002e-10*cse837;
  const amrex::Real cse839 = cse838*state.xn[0];
  const amrex::Real cse840 = 0.031*cse640;
  const amrex::Real cse841 = 1.11e-10*std::pow(cse639, cse840 + 0.26300000000000001);
  const amrex::Real cse842 = 0.064000000000000001*cse640;
  const amrex::Real cse843 = 4.3749999999999998e-11*std::pow(cse639, cse842 + 0.071999999999999995);
  const amrex::Real cse844 = cse841 + cse843;
  const amrex::Real cse845 = 0.00235272*cse75;
  const amrex::Real cse846 = cse845 - 0.041816600000000002;
  const amrex::Real cse847 = cse75*cse846 + 0.35826400000000003;
  const amrex::Real cse848 = cse75*cse847 - 0.57443;
  const amrex::Real cse849 = 0.097857299999999994*cse75;
  const amrex::Real cse850 = cse849 - 3.1926800000000002;
  const amrex::Real cse851 = cse75*cse850 + 38.504899999999999;
  const amrex::Real cse852 = cse75*cse851 - 202.19300000000001;
  const amrex::Real cse853 = std::exp(((cse716) ? (
   cse75*cse848 - 7.4386999999999999
)
: (
   cse75*cse852 + 386.18599999999998
)));
  const amrex::Real cse854 = 1.7258000000000001e-6*cse638;
  const amrex::Real cse855 = cse853*cse854;
  const amrex::Real cse856 = cse839 + cse844*state.xn[3] + cse855;
  const amrex::Real cse857 = std::exp(-101.30000000000001*cse5);
  const amrex::Real cse858 = cse856*cse857;
  const amrex::Real cse859 = 8.9000000000000003e-11*cse836;
  const amrex::Real cse860 = cse859*state.xn[0];
  const amrex::Real cse861 = 0.037999999999999999*cse640;
  const amrex::Real cse862 = 5.1749999999999996e-11*std::pow(cse639, cse861 + 0.16900000000000001);
  const amrex::Real cse863 = 0.048000000000000001*cse640;
  const amrex::Real cse864 = 2.15e-11*std::pow(cse639, cse863 - 0.01);
  const amrex::Real cse865 = cse862 + cse864;
  const amrex::Real cse866 = 0.0070527699999999999*cse75;
  const amrex::Real cse867 = cse866 - 0.11133800000000001;
  const amrex::Real cse868 = cse75*cse867 + 0.69763799999999998;
  const amrex::Real cse869 = cse75*cse868 - 1.3074300000000001;
  const amrex::Real cse870 = 0.093813800000000003*cse75;
  const amrex::Real cse871 = cse870 - 3.0328300000000001;
  const amrex::Real cse872 = cse75*cse871 + 36.180300000000003;
  const amrex::Real cse873 = cse75*cse872 - 187.47399999999999;
  const amrex::Real cse874 = std::exp(((cse716) ? (
   cse75*cse869 - 7.6973500000000001
)
: (
   cse75*cse873 + 350.60899999999998
)));
  const amrex::Real cse875 = cse854*cse874;
  const amrex::Real cse876 = cse860 + cse865*state.xn[3] + cse875;
  const amrex::Real cse877 = cse857*cse876;
  const amrex::Real cse878 = 0.057000000000000002*cse640;
  const amrex::Real cse879 = std::pow(cse639, cse878 + 0.115);
  const amrex::Real cse880 = 1.26e-10*cse879;
  const amrex::Real cse881 = cse880*state.xn[0];
  const amrex::Real cse882 = 0.049000000000000002*cse640;
  const amrex::Real cse883 = 5.3249999999999997e-11*std::pow(cse639, cse882 - 0.0040000000000000001);
  const amrex::Real cse884 = 0.10199999999999999*cse640;
  const amrex::Real cse885 = 1.675e-11*std::pow(cse639, cse884 - 0.085000000000000006);
  const amrex::Real cse886 = cse883 + cse885;
  const amrex::Real cse887 = 0.00065632500000000003*cse75;
  const amrex::Real cse888 = -cse887 - 0.015089200000000001;
  const amrex::Real cse889 = cse75*cse888 + 0.36118400000000001;
  const amrex::Real cse890 = cse75*cse889 - 0.77378199999999997;
  const amrex::Real cse891 = 0.10508000000000001*cse75;
  const amrex::Real cse892 = cse891 - 3.4762;
  const amrex::Real cse893 = cse75*cse892 + 42.594999999999999;
  const amrex::Real cse894 = cse75*cse893 - 227.91300000000001;
  const amrex::Real cse895 = std::exp(((cse716) ? (
   cse75*cse890 - 9.2514099999999999
)
: (
   cse75*cse894 + 444.60000000000002
)));
  const amrex::Real cse896 = 2.8763333333333333e-6*cse895;
  const amrex::Real cse897 = cse638*cse896;
  const amrex::Real cse898 = cse881 + cse886*state.xn[3] + cse897;
  const amrex::Real cse899 = std::exp(-62.460000000000001*cse5);
  const amrex::Real cse900 = cse898*cse899;
  const amrex::Real cse901 = 1.32e-9*cse837;
  const amrex::Real cse902 = 6.3e-10*cse879;
  const amrex::Real cse903 = cse876*cse899;
  const amrex::Real cse904 = cse856*cse899;
  const amrex::Real cse905 = cse898 + 7.8800000000000004e-8;
  const amrex::Real cse906 = cse899*cse905;
  const amrex::Real cse907 = 4.4500000000000001e-10*cse836;
  const amrex::Real cse908 = 7.4166666666666679e-10*cse836*cse858 + 2.2000000000000003e-9*cse837*cse877 + cse900*cse901 + cse902*cse903 + cse902*cse904 + cse906*cse907;
  const amrex::Real cse909 = cse856 + 2.65e-7;
  const amrex::Real cse910 = cse876 + 1.81e-14;
  const amrex::Real cse911 = std::exp(-38.840000000000003*cse5);
  const amrex::Real cse912 = cse856*cse911;
  const amrex::Real cse913 = (5.0/3.0)*cse910;
  const amrex::Real cse914 = (25.0/3.0)*cse877;
  const amrex::Real cse915 = 5*cse899;
  const amrex::Real cse916 = cse856*cse915;
  const amrex::Real cse917 = cse876*cse915;
  const amrex::Real cse918 = cse856*cse914 + cse898*cse916 + cse905*cse917;
  const amrex::Real cse919 = std::exp(-23.619999999999997*cse5);
  const amrex::Real cse920 = cse898*cse919;
  const amrex::Real cse921 = 3*cse920;
  const amrex::Real cse922 = cse909*cse917 + cse909*cse921 + cse910*cse921;
  const amrex::Real cse923 = cse905*cse909 + cse905*cse910 + cse912*cse913 + cse918 + cse922;
  const amrex::Real cse924 = 1.0/cse923;
  const amrex::Real cse925 = 1.4210469458259591e-21*cse924;
  const amrex::Real cse926 = cse899*cse909;
  const amrex::Real cse927 = 3.7800000000000004e-10*cse879;
  const amrex::Real cse928 = cse910*cse919;
  const amrex::Real cse929 = cse909*cse919;
  const amrex::Real cse930 = 2.6700000000000001e-10*cse836*cse920 + 7.9200000000000006e-10*cse837*cse920 + cse901*cse903 + cse907*cse926 + cse927*cse928 + cse927*cse929;
  const amrex::Real cse931 = 2.5697412351439998e-22*cse924;
  const amrex::Real cse932 = std::pow(cse923, -2);
  const amrex::Real cse933 = cse910*cse911;
  const amrex::Real cse934 = cse932*(-1.4833333333333334e-10*cse836*cse912 - 4.4000000000000003e-10*cse837*cse933 - cse838*cse905 - cse859*cse905 - cse880*cse909 - cse880*cse910 - cse908 - cse930);
  const amrex::Real cse935 = 1.4210469458259591e-21*cse918;
  const amrex::Real cse936 = 2.5697412351439998e-22*cse922;
  const amrex::Real cse937 = state.T < 1000000.0;
  const amrex::Real cse938 = cse565*cse72*d2g;
  const amrex::Real cse939 = crate*cse669;
  const amrex::Real cse940 = 1.7623947000000002e-12*cse675*cse939*state.xn[5];
  const amrex::Real cse941 = 3.8323655572603624e-29*cse447;
  const amrex::Real cse942 = cse941*state.xn[3];
  const amrex::Real cse943 = cse676*(cse38*cse603*cse604 - cse680*cse682);
  const amrex::Real cse944 = 2.0187430200000004e-26*cse170;
  const amrex::Real cse945 = cse13*cse944;
  const amrex::Real cse946 = cse14*cse696*state.xn[3] + 26.399999999999999*cse656*cse658*cse69*cse693*state.xn[0] - cse688 - cse697;
  const amrex::Real cse947 = 1.602177e-12*cse92;
  const amrex::Real cse948 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse949 = 1.7623947000000002e-12*cse699;
  const amrex::Real cse950 = crate*cse949*state.xn[5];
  const amrex::Real cse951 = -7.1777529600000012e-12*cse123 + 3.0000000000000002e-15*cse170*cse608 - 7.1777529600000012e-12*cse500 + 1.7e-24*cse561*cse938 - cse588 - cse591 + cse89*cse940 + cse942*cse943 + cse943*cse945 + cse946*cse947 + cse946*cse948 + cse946*cse950;
  const amrex::Real cse952 = state.xn[2] + 1.0e-50;
  const amrex::Real cse953 = 1.0/cse952;
  const amrex::Real cse954 = cse4*cse72;
  const amrex::Real cse955 = 1.7*cse954;
  const amrex::Real cse956 = cse953*cse955 + 50;
  const amrex::Real cse957 = std::log(cse956);
  const amrex::Real cse958 = 0.45350000000000001 + 2.234/cse957;
  const amrex::Real cse959 = 1.4419999999999999 - 0.050889999999999998*cse957;
  const amrex::Real cse960 = d2g*std::pow(state.T, cse958)*std::exp(cse957*cse959);
  const amrex::Real cse961 = 1.8998126555754159e-29*cse960;
  const amrex::Real cse962 = cse566*cse568;
  const amrex::Real cse963 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse964 = std::pow(cse567, -2);
  const amrex::Real cse965 = std::pow(cse571, -2);
  const amrex::Real cse966 = 1.0/cse956;
  const amrex::Real cse967 = std::pow(cse952, -2);
  const amrex::Real cse968 = cse966*cse967;
  const amrex::Real cse969 = cse75/std::pow(cse957, 2);
  const amrex::Real cse970 = std::pow(cse564, -2);
  const amrex::Real cse971 = std::pow(cse562, 0.12299999999999994)*cse72;
  const amrex::Real cse972 = std::pow(cse562, -0.56899999999999995)*cse563*cse72;
  const amrex::Real cse973 = 1.3659996292638161e-10*cse295*cse962;
  const amrex::Real cse974 = cse574*cse973/std::pow(cse576, 2);
  const amrex::Real cse975 = 0.5*cse14*cse657 - cse689*cse691;
  const amrex::Real cse976 = cse961*state.xn[2];
  const amrex::Real cse977 = 9.0599999999999997e-6*cse289;
  const amrex::Real cse978 = 1.024e-10*cse791;
  const amrex::Real cse979 = 2.1600000000000004e-15*cse761;
  const amrex::Real cse980 = 9.7199999999999998e-11*cse776;
  const amrex::Real cse981 = 7.1999999999999997e-16*cse761*cse802 + 3.2399999999999999e-11*cse774*cse776 + cse786*cse978 + cse790*cse978 + cse800*cse979 + cse804*cse980;
  const amrex::Real cse982 = 3.0719999999999998e-10*cse791;
  const amrex::Real cse983 = 6.4799999999999996e-15*cse761*cse818 + 2.9159999999999995e-10*cse776*cse818 + cse786*cse979 + cse824*cse980 + cse826*cse982 + cse827*cse982;
  const amrex::Real cse984 = cse830*(-3.6000000000000001e-15*cse761*cse831 - cse762*cse803 - 1.6199999999999998e-10*cse776*cse810 - cse777*cse803 - cse792*cse807 - cse792*cse808 - cse981 - cse983);
  const amrex::Real cse985 = cse289*cse853;
  const amrex::Real cse986 = cse289*cse895;
  const amrex::Real cse987 = 1.4381666666666666e-5*cse986;
  const amrex::Real cse988 = cse289*cse874;
  const amrex::Real cse989 = 8.6289999999999999e-6*cse985;
  const amrex::Real cse990 = 8.6289999999999999e-6*cse988;
  const amrex::Real cse991 = 1.4381666666666668e-5*cse858*cse988 + 1.4381666666666668e-5*cse877*cse985 + cse900*cse989 + cse903*cse987 + cse904*cse987 + cse906*cse990;
  const amrex::Real cse992 = 5.1774000000000006e-6*cse920;
  const amrex::Real cse993 = 8.6289999999999999e-6*cse986;
  const amrex::Real cse994 = cse903*cse989 + cse926*cse990 + cse928*cse993 + cse929*cse993 + cse985*cse992 + cse988*cse992;
  const amrex::Real cse995 = cse289*cse896;
  const amrex::Real cse996 = 1.7258000000000001e-6*cse905;
  const amrex::Real cse997 = cse932*(-cse909*cse995 - cse910*cse995 - 2.8763333333333337e-6*cse912*cse988 - 2.8763333333333337e-6*cse933*cse985 - cse985*cse996 - cse988*cse996 - cse991 - cse994);
  const amrex::Real cse998 = cse4*cse587;
  const amrex::Real cse999 = 6.4000000000000004e-32*cse998;
  const amrex::Real cse1000 = cse590*d2g;
  const amrex::Real cse1001 = 200.0*cse1000;
  const amrex::Real cse1002 = 2.5634832e-11*cse607 + 6.4087080000000003e-13;
  const amrex::Real cse1003 = 5.7927337833400002e-20*cse653;
  const amrex::Real cse1004 = cse1003*cse654;
  const amrex::Real cse1005 = -cse646;
  const amrex::Real cse1006 = cse1003*cse685*state.xn[8];
  const amrex::Real cse1007 = cse176*cse596 + 2*cse679;
  const amrex::Real cse1008 = 2*cse677 - cse682*(cse1007 - cse597);
  const amrex::Real cse1009 = 2*cse685;
  const amrex::Real cse1010 = 2*cse688;
  const amrex::Real cse1011 = cse657*cse689;
  const amrex::Real cse1012 = cse690*cse692;
  const amrex::Real cse1013 = -1.0*cse1011 + 1.0*cse1012;
  const amrex::Real cse1014 = ((cse663) ? (
   0
)
: ((cse664) ? (
   3*cse695
)
: ((cse665) ? (
   (8.0/3.0)*cse695
)
: ((cse666) ? (
   (2.0/3.0)*cse695
)
: (
   0
)))));
  const amrex::Real cse1015 = cse176*cse667;
  const amrex::Real cse1016 = -cse1010 + cse1013*cse694 + cse1014*cse449 - cse1015 + cse668;
  const amrex::Real cse1017 = 1.0/cse756;
  const amrex::Real cse1018 = cse799*cse812;
  const amrex::Real cse1019 = (1.0/15.0)*cse802;
  const amrex::Real cse1020 = cse784*cse803;
  const amrex::Real cse1021 = cse1018*cse771 + cse1019*cse771 + cse1020*cse812 + cse784*cse814 + cse798*cse813 + cse798*cse815;
  const amrex::Real cse1022 = cse798*cse808;
  const amrex::Real cse1023 = (3.0/5.0)*cse817;
  const amrex::Real cse1024 = cse798*cse807;
  const amrex::Real cse1025 = cse807*cse812;
  const amrex::Real cse1026 = cse1022*cse1023 + cse1023*cse1024 + cse1025*cse784 + cse771*cse815 + cse771*cse819 + cse784*cse819;
  const amrex::Real cse1027 = (1.0/3.0)*cse831;
  const amrex::Real cse1028 = cse830*(-cse1020 - cse1021 - cse1022 - cse1024 - cse1026 - cse1027*cse771 - cse771*cse803 - cse784*cse811);
  const amrex::Real cse1029 = cse898*cse915;
  const amrex::Real cse1030 = (25.0/3.0)*cse858;
  const amrex::Real cse1031 = cse865*cse905;
  const amrex::Real cse1032 = cse1029*cse844 + cse1030*cse865 + cse1031*cse915 + cse844*cse914 + cse886*cse916 + cse886*cse917;
  const amrex::Real cse1033 = cse909*cse915;
  const amrex::Real cse1034 = cse886*cse910;
  const amrex::Real cse1035 = 3*cse919;
  const amrex::Real cse1036 = cse886*cse909;
  const amrex::Real cse1037 = cse1033*cse865 + cse1034*cse1035 + cse1035*cse1036 + cse844*cse917 + cse844*cse921 + cse865*cse921;
  const amrex::Real cse1038 = cse911*cse913;
  const amrex::Real cse1039 = (5.0/3.0)*cse912;
  const amrex::Real cse1040 = cse932*(-cse1031 - cse1032 - cse1034 - cse1036 - cse1037 - cse1038*cse844 - cse1039*cse865 - cse844*cse905);
  const amrex::Real cse1041 = cse561*cse938;
  const amrex::Real cse1042 = cse676*(-cse1007*cse682 + 2*cse38*cse603*cse604);
  const amrex::Real cse1043 = -cse1010 + 26.399999999999999*cse1013*cse656*cse658*cse69*state.xn[0] + cse1014*cse14*state.xn[3] - cse1015;
  const amrex::Real cse1044 = -state.xn[3]*((cse720) ? (
   0
)
: (
   cse736*cse758
));
  const amrex::Real cse1045 = std::pow(cse623, -2);
  const amrex::Real cse1046 = llte_co_interp2d_partial_1(state.T, cse622);
  const amrex::Real cse1047 = nhalf_co_interp2d_partial_1(state.T, cse622);
  const amrex::Real cse1048 = cse1047*cse621;
  const amrex::Real cse1049 = alpha_co_interp2d_partial_1(state.T, cse622);
  const amrex::Real cse1050 = std::log(cse629);
  const amrex::Real cse1051 = cse676*(3*cse38*cse603*cse604 - cse682*(3*cse678 + 3*cse679));
  const amrex::Real cse1052 = cse14*state.xn[3]*((cse663) ? (
   0
)
: ((cse664) ? (
   (9.0/2.0)*cse695
)
: ((cse665) ? (
   4*cse695
)
: ((cse666) ? (
   cse695
)
: (
   0
))))) + 26.399999999999999*cse656*cse658*cse69*state.xn[0]*(-1.5*cse1011 + 1.5*cse1012) - 3*cse688 - 3*cse697;
  const amrex::Real cse1053 = cse574*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1054 = cse295*cse568;
  const amrex::Real cse1055 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1056 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1057 = cse1055*cse569;
  const amrex::Real cse1058 = cse573*cse965*(2.5500935934394033e-12*cse1056*cse964*state.xn[2] - 6.6690584756579352e-10*cse1057);
  const amrex::Real cse1059 = cse273*cse72*cse953*cse966;
  const amrex::Real cse1060 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1061 = cse263*cse686;
  const amrex::Real cse1062 = cse5*cse638;
  const amrex::Real cse1063 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1064 = 100*cse5;
  const amrex::Real cse1065 = -2.2649999999999999e-6*cse1062 + cse643*(cse1064*(8.7000000000000001e-5*cse640 + 0.001281) + cse5*cse641) + state.xn[3]*(0.75*((cse644) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1063
)) + 0.25*((cse644) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1063
)));
  const amrex::Real cse1066 = std::pow(cse593, -3);
  const amrex::Real cse1067 = cse676*cse681*(-cse598*cse9*(0.0040460000000000001*cse1066 + 0.0036709999999999998*cse594) - cse601*cse9*(0.0047800000000000004*cse1066 - 5.5500000000000001e-5*cse594));
  const amrex::Real cse1068 = -cse1065;
  const amrex::Real cse1069 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1070 = std::pow(cse614, -2);
  const amrex::Real cse1071 = std::pow(cse619, -2);
  const amrex::Real cse1072 = (((5.7445531974402559e-18*cse4*cse618 - gradv < 0) ? (
   0
)
: ((5.7445531974402559e-18*cse4*cse618 - gradv == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1073 = cse1045*(2.872276598720128e-13*cse1046*cse1071*cse1072*cse273*cse618*state.xn[9] - llte_co_interp2d_partial_0(state.T, cse622));
  const amrex::Real cse1074 = 2.872276598720128e-13*cse1071*cse1072*cse273*cse618*state.xn[9];
  const amrex::Real cse1075 = -cse1047*cse1074 + nhalf_co_interp2d_partial_0(state.T, cse622);
  const amrex::Real cse1076 = ((cse701) ? (
   -cse263
)
: (
   0
));
  const amrex::Real cse1077 = -1.8719999999999998e-14*cse1076*cse747 - 3.9261999999999998e-15*cse1076*cse748 - 1.53e-21*cse1076*cse749 + 4.985670872372847e-33*cse750*cse753*((cse701) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse751*cse753*((cse701) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse751*((cse701) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse752, 2);
  const amrex::Real cse1078 = ((cse701) ? (
   1
)
: (
   0
));
  const amrex::Real cse1079 = cse1078*cse746;
  const amrex::Real cse1080 = cse1079*cse721;
  const amrex::Real cse1081 = cse1079*cse723;
  const amrex::Real cse1082 = cse1079*cse724;
  const amrex::Real cse1083 = cse1079*cse722;
  const amrex::Real cse1084 = cse1079*cse725;
  const amrex::Real cse1085 = cse1079*cse738;
  const amrex::Real cse1086 = cse1079*cse739;
  const amrex::Real cse1087 = 65.320000000000007*cse263;
  const amrex::Real cse1088 = cse5*cse767;
  const amrex::Real cse1089 = 0.00042999999999999999*cse640;
  const amrex::Real cse1090 = cse794*(cse1064*(0.0041900000000000001 - 3.0000000000000001e-5*cse640) - cse5*cse787) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse796*(cse1064*(0.00025000000000000001*cse640 + 0.00264) + cse5*cse795) + cse797*(cse1064*(cse1089 + 0.00296) + cse1088));
  const amrex::Real cse1091 = cse779*(cse1064*(0.0036900000000000001 - 6.0000000000000002e-5*cse640) - cse5*cse759) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse781*(cse1064*(0.00041000000000000005*cse640 + 0.0020300000000000001) + cse5*cse780) + cse783*(cse1064*(0.00058*cse640 + 0.0023700000000000001) + cse5*cse782));
  const amrex::Real cse1092 = cse766*(cse1064*(0.0075500000000000003 - 0.0016000000000000001*cse640) - cse5*cse763) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse768*(cse1064*(cse1089 + 0.0088900000000000003) + cse1088) + cse770*(cse1064*(0.0052500000000000003*cse640 + 0.011979999999999999) + cse5*cse769));
  const amrex::Real cse1093 = cse1091*cse803;
  const amrex::Real cse1094 = cse1018*cse1092 + cse1019*cse1092 + cse1087*cse772*cse800 + cse1087*cse785*cse804 + cse1090*cse813 + cse1090*cse815 + cse1091*cse814 + cse1093*cse812 + 28.360000000000003*cse263*cse774*cse785;
  const amrex::Real cse1095 = 136.68000000000001*cse263*cse818;
  const amrex::Real cse1096 = cse1090*cse808;
  const amrex::Real cse1097 = cse1090*cse807;
  const amrex::Real cse1098 = cse1023*cse1096 + cse1023*cse1097 + cse1025*cse1091 + cse1087*cse786*cse807 + cse1091*cse819 + cse1092*cse815 + cse1092*cse819 + cse1095*cse807 + cse1095*cse808;
  const amrex::Real cse1099 = cse830*(-cse1027*cse1092 - cse1091*cse811 - cse1092*cse803 - cse1093 - cse1094 - cse1096 - cse1097 - cse1098 - 32.933333333333337*cse263*cse808*cse810);
  const amrex::Real cse1100 = 312.30000000000001*cse263;
  const amrex::Real cse1101 = cse5*cse835;
  const amrex::Real cse1102 = 0.00046000000000000001*cse640;
  const amrex::Real cse1103 = 8.6290000000000003e-7*cse1062;
  const amrex::Real cse1104 = -cse1103*cse853 + cse839*(cse1064*(cse1102 + 0.00231) + cse1101) + cse855*((cse716) ? (
   cse5*cse848 + cse75*(cse5*cse847 + cse75*(cse5*cse845 + cse5*cse846))
)
: (
   cse5*cse852 + cse75*(cse5*cse851 + cse75*(cse5*cse849 + cse5*cse850))
)) + state.xn[3]*(cse841*(cse1064*(0.00031*cse640 + 0.00263) + cse5*cse840) + cse843*(cse1064*(0.00064000000000000005*cse640 + 0.00071999999999999994) + cse5*cse842));
  const amrex::Real cse1105 = -1.4381666666666666e-6*cse1062*cse895 + cse881*(cse1064*(0.00057000000000000009*cse640 + 0.00115) + cse5*cse878) + cse897*((cse716) ? (
   cse5*cse890 + cse75*(cse5*cse889 + cse75*(-cse5*cse887 + cse5*cse888))
)
: (
   cse5*cse894 + cse75*(cse5*cse893 + cse75*(cse5*cse891 + cse5*cse892))
)) + state.xn[3]*(cse883*(cse1064*(0.00048999999999999998*cse640 - 4.0000000000000003e-5) + cse5*cse882) + cse885*(cse1064*(0.0010199999999999999*cse640 - 0.00085000000000000006) + cse5*cse884));
  const amrex::Real cse1106 = -cse1103*cse874 + cse860*(cse1064*(cse1102 + 0.0022800000000000003) + cse1101) + cse875*((cse716) ? (
   cse5*cse869 + cse75*(cse5*cse868 + cse75*(cse5*cse866 + cse5*cse867))
)
: (
   cse5*cse873 + cse75*(cse5*cse872 + cse75*(cse5*cse870 + cse5*cse871))
)) + state.xn[3]*(cse862*(cse1064*(0.00038000000000000002*cse640 + 0.0016900000000000001) + cse5*cse861) + cse864*(cse1064*(0.00048000000000000001*cse640 - 0.0001) + cse5*cse863));
  const amrex::Real cse1107 = cse1106*cse905;
  const amrex::Real cse1108 = cse1029*cse1104 + cse1030*cse1106 + cse1100*cse856*cse900 + cse1100*cse876*cse906 + cse1104*cse914 + cse1105*cse916 + cse1105*cse917 + cse1107*cse915 + 844.16666666666686*cse263*cse856*cse877;
  const amrex::Real cse1109 = 70.859999999999985*cse263*cse920;
  const amrex::Real cse1110 = cse1105*cse910;
  const amrex::Real cse1111 = cse1105*cse909;
  const amrex::Real cse1112 = cse1033*cse1106 + cse1035*cse1110 + cse1035*cse1111 + cse1100*cse903*cse909 + cse1104*cse917 + cse1104*cse921 + cse1106*cse921 + cse1109*cse909 + cse1109*cse910;
  const amrex::Real cse1113 = cse932*(-cse1038*cse1104 - cse1039*cse1106 - cse1104*cse905 - cse1107 - cse1108 - cse1110 - cse1111 - cse1112 - 64.733333333333348*cse263*cse910*cse912);
  const amrex::Real cse1114 = -cse492 - 5.4061971382371991e-18*cse520 - cse544;

  jac(1, 1) = -cse105 - cse13*cse59 - cse3 + 2*cse32*state.xn[3] + 2*cse33*cse58 - cse60 + 2*cse63*state.xn[0]*state.xn[3];
  jac(1, 2) = cse118 + cse127;
  jac(1, 3) = cse11*cse130 + cse139*cse143 + cse150 + cse153 + cse161;
  jac(1, 4) = cse126*cse188 - cse143*cse192 + cse164 + cse169 + cse171 + 2*cse172 + 2*cse181 + 4*cse183 + cse184*state.xn[17] + cse185*state.rn[0] + cse191*state.xn[8] + cse196 + cse205;
  jac(1, 5) = cse126*cse210 + cse171 + 2*cse209 + cse213 + cse216;
  jac(1, 7) = cse222;
  jac(1, 8) = cse226;
  jac(1, 9) = cse191*state.xn[3];
  jac(1, 10) = -cse227;
  jac(1, 11) = cse127 + cse229;
  jac(1, 12) = cse232 + cse233 + cse236;
  jac(1, 15) = cse237 + cse238 + cse240 + cse241;
  jac(1, 16) = cse241 + cse245;
  jac(1, 17) = cse126*cse252 - 1.8000000000000002e-14*cse170 + 2*cse250 + 3*cse251 + cse259;
  jac(1, 18) = cse184*state.xn[3] - cse260 + cse262;
  jac(1, 19) = cse316*(cse11*cse277*state.xn[2] + cse126*cse275 - cse147*cse283 + 2*cse272 + 9.8669999999999991e-12*cse279 - 5.5769999999999994e-13*cse281 + cse285 + cse288 + cse293 + cse301 + cse309 + cse315);
  jac(1, 20) = cse185*state.xn[3];
  jac(2, 1) = cse105;
  jac(2, 2) = -cse118;
  jac(2, 3) = -cse160;
  jac(2, 4) = cse164 + cse202 + cse317;
  jac(2, 5) = cse212 + cse215 + cse317;
  jac(2, 7) = cse217;
  jac(2, 11) = cse318;
  jac(2, 12) = -cse235;
  jac(2, 15) = cse318;
  jac(2, 16) = cse318;
  jac(2, 17) = cse255 - cse256;
  jac(2, 18) = cse260;
  jac(2, 19) = cse316*(cse292 - cse301);
  jac(3, 1) = cse104 + cse320 - cse360 + cse68 - cse87 + cse93;
  jac(3, 2) = -cse117 - cse362;
  jac(3, 3) = -cse161 - cse372 - cse389 - cse394 - cse395 - cse409;
  jac(3, 4) = -cse205 - cse411 + cse412 + cse413 - cse418 - cse425 - cse430 - cse431;
  jac(3, 5) = -cse216 - cse411 - cse417 - cse424 - cse433;
  jac(3, 6) = cse434;
  jac(3, 7) = -cse436;
  jac(3, 8) = cse446 + cse451;
  jac(3, 9) = -cse454;
  jac(3, 10) = cse227;
  jac(3, 11) = -cse229 - cse362;
  jac(3, 12) = cse459;
  jac(3, 13) = cse461;
  jac(3, 14) = cse463;
  jac(3, 15) = cse464;
  jac(3, 16) = cse464;
  jac(3, 17) = -cse251 - cse259 - cse465 - cse466 - cse468 - cse469 - cse470 - cse471;
  jac(3, 18) = cse473;
  jac(3, 19) = cse316*(-cse286 - cse300 - cse305 - cse308 - cse315 - cse474 - cse475 - cse476 - cse477 - cse478 - cse482 - cse487);
  jac(3, 20) = cse492;
  jac(4, 1) = 3.0000000000000002e-15*cse13*d2g - cse493 - cse494 - cse495 - cse496 - cse498 - cse499;
  jac(4, 2) = cse502;
  jac(4, 3) = cse146 - cse148 - cse149 + cse408;
  jac(4, 4) = -cse12*cse182 - cse124 - cse172 - cse181 - cse420 - 1.4949836836302895e-18*cse447 - cse503 - cse505 - cse506 - cse507 - cse508;
  jac(4, 5) = -cse124 - cse209 - cse213 - cse432 - cse509;
  jac(4, 7) = -cse217 - cse510;
  jac(4, 8) = cse445;
  jac(4, 9) = -cse452;
  jac(4, 10) = cse511;
  jac(4, 11) = cse502;
  jac(4, 12) = cse458;
  jac(4, 15) = -cse240 - cse501;
  jac(4, 16) = cse502;
  jac(4, 17) = -cse250 + cse468 + cse512 - cse513 + cse515;
  jac(4, 18) = cse473;
  jac(4, 19) = cse316*(-cse272 - cse292 - cse309 - cse475 - cse481 - cse516 - cse519);
  jac(4, 20) = -cse521;
  jac(5, 1) = cse499;
  jac(5, 2) = cse361;
  jac(5, 4) = -cse169 + cse504 + cse506;
  jac(5, 5) = -cse211 - cse212 - cse433;
  jac(5, 7) = cse510;
  jac(5, 11) = cse361;
  jac(5, 15) = cse361;
  jac(5, 16) = cse361;
  jac(5, 17) = cse467;
  jac(5, 19) = -cse285*cse316;
  jac(6, 1) = cse332;
  jac(6, 2) = cse332;
  jac(6, 3) = cse372;
  jac(6, 4) = cse522;
  jac(6, 5) = cse411;
  jac(6, 6) = -cse434;
  jac(6, 7) = cse524;
  jac(6, 10) = cse525;
  jac(6, 11) = cse332;
  jac(6, 15) = cse332;
  jac(6, 16) = cse526;
  jac(6, 17) = cse469;
  jac(6, 19) = cse316*cse527;
  jac(7, 1) = cse528;
  jac(7, 2) = cse528;
  jac(7, 3) = -cse372;
  jac(7, 4) = -cse522;
  jac(7, 5) = -cse411;
  jac(7, 6) = cse434;
  jac(7, 7) = -cse524;
  jac(7, 10) = -cse525;
  jac(7, 11) = cse528;
  jac(7, 15) = cse528;
  jac(7, 16) = -cse526;
  jac(7, 17) = -cse469;
  jac(7, 19) = -cse316*cse527;
  jac(8, 1) = cse3 + cse532;
  jac(8, 2) = cse532;
  jac(8, 3) = cse389;
  jac(8, 4) = cse421 + cse533 + cse534*state.xn[8] + cse538;
  jac(8, 5) = cse538;
  jac(8, 8) = -cse226 - cse445 - cse451;
  jac(8, 9) = cse453 + cse534*state.xn[3];
  jac(8, 10) = cse540;
  jac(8, 11) = cse532;
  jac(8, 15) = cse532 + cse541;
  jac(8, 16) = -cse243 + cse532;
  jac(8, 17) = -cse466 + cse470 - cse542;
  jac(8, 19) = cse316*(2.2769999999999997e-12*cse279 - 1.287e-13*cse281 - cse287 + cse302 + cse480 + cse517);
  jac(8, 20) = -cse491 - cse545;
  jac(9, 1) = cse546;
  jac(9, 2) = cse546;
  jac(9, 3) = -cse389;
  jac(9, 4) = -cse425;
  jac(9, 5) = -cse424;
  jac(9, 7) = cse523;
  jac(9, 8) = cse451;
  jac(9, 9) = -cse454 - cse548;
  jac(9, 10) = cse525;
  jac(9, 11) = cse546;
  jac(9, 15) = cse546;
  jac(9, 16) = -cse346 - cse549;
  jac(9, 17) = -cse470;
  jac(9, 19) = cse316*(-cse482 - cse550);
  jac(9, 20) = cse491;
  jac(10, 1) = cse530;
  jac(10, 2) = cse530;
  jac(10, 3) = cse155;
  jac(10, 4) = cse551;
  jac(10, 5) = cse536;
  jac(10, 7) = -cse523;
  jac(10, 8) = cse226;
  jac(10, 10) = -cse552 - cse553;
  jac(10, 11) = cse228 + cse530;
  jac(10, 12) = cse234;
  jac(10, 15) = cse554;
  jac(10, 16) = cse243 + cse530;
  jac(10, 17) = -cse514 + cse542;
  jac(10, 19) = cse288*cse316;
  jac(10, 20) = cse545;
  jac(11, 3) = -cse155;
  jac(11, 9) = cse548;
  jac(11, 10) = cse552;
  jac(11, 11) = -cse228;
  jac(11, 16) = cse549;
  jac(11, 17) = cse514;
  jac(11, 19) = cse316*(-cse286 + cse550);
  jac(12, 1) = -cse530 - cse555;
  jac(12, 2) = -cse108 - cse530;
  jac(12, 3) = cse148 - cse404 + cse406;
  jac(12, 4) = cse194 - cse195 - cse429 - cse551;
  jac(12, 5) = cse537;
  jac(12, 7) = cse523;
  jac(12, 10) = cse553;
  jac(12, 11) = cse531;
  jac(12, 12) = -cse236 - cse556;
  jac(12, 15) = -cse554;
  jac(12, 16) = cse244 + cse531;
  jac(12, 17) = -cse465 - cse542;
  jac(12, 18) = cse260 + cse261;
  jac(12, 19) = cse316*(-cse487 - cse557);
  jac(12, 20) = cse544;
  jac(13, 1) = cse359;
  jac(13, 2) = cse359;
  jac(13, 3) = cse394;
  jac(13, 4) = cse418;
  jac(13, 5) = cse417;
  jac(13, 11) = cse359;
  jac(13, 13) = -cse461;
  jac(13, 14) = cse462;
  jac(13, 15) = cse359;
  jac(13, 16) = cse359;
  jac(13, 17) = cse471;
  jac(13, 19) = cse316*cse477;
  jac(13, 20) = -cse489;
  jac(14, 1) = cse558;
  jac(14, 2) = cse558;
  jac(14, 3) = -cse394;
  jac(14, 4) = -cse418;
  jac(14, 5) = -cse417;
  jac(14, 11) = cse558;
  jac(14, 13) = cse461;
  jac(14, 14) = cse463;
  jac(14, 15) = cse558;
  jac(14, 16) = cse558;
  jac(14, 17) = -cse471;
  jac(14, 19) = -cse316*cse477;
  jac(14, 20) = cse489;
  jac(15, 1) = cse496;
  jac(15, 4) = cse559*state.xn[8];
  jac(15, 8) = cse445;
  jac(15, 9) = cse559*state.xn[3];
  jac(15, 12) = -cse234;
  jac(15, 15) = -cse237 - cse541;
  jac(15, 17) = cse466;
  jac(15, 19) = cse316*(5.3129999999999997e-12*cse279 - 3.0029999999999999e-13*cse281 + cse303 + cse474);
  jac(16, 3) = cse150 + cse407;
  jac(16, 4) = cse430;
  jac(16, 7) = -cse221;
  jac(16, 8) = -cse226;
  jac(16, 9) = -cse548;
  jac(16, 12) = cse556;
  jac(16, 16) = -cse245 - cse549;
  jac(16, 17) = cse465;
  jac(16, 18) = cse262;
  jac(16, 19) = cse316*(-cse287 - cse518 - cse550 - cse560);
  jac(17, 3) = -cse409;
  jac(17, 4) = cse507;
  jac(17, 5) = cse211;
  jac(17, 8) = cse446;
  jac(17, 10) = -cse511;
  jac(17, 12) = cse459;
  jac(17, 17) = -cse251 - cse515;
  jac(17, 19) = cse316*(-cse476 + cse519);
  jac(18, 1) = cse555;
  jac(18, 2) = cse108;
  jac(18, 3) = -cse395;
  jac(18, 4) = -cse508;
  jac(18, 7) = cse221;
  jac(18, 12) = cse235;
  jac(18, 16) = cse242;
  jac(18, 18) = -cse260 - cse472;
  jac(18, 19) = cse316*(cse557 + cse560);
  jac(19, 1) = 3.8323655572603624e-29*state.c_hat*cse676*cse683*state.rn[0]*state.xn[3] + 1.7e-24*chi*cse561*cse565*cse71*d2g + 1.7623947000000002e-12*crate*cse669*cse675*cse90*state.xn[5] + 1.602177e-12*crate*cse669 + 1.7623947000000002e-12*crate*cse698*cse699*state.xn[5] + 1.602177e-12*crate*cse698*state.xn[0] + 3.2043540000000001e-12*crate*cse698*state.xn[3] - 2.1789607199999999e-11*cse104 + 3.0000000000000002e-15*cse13*cse608*d2g + 2.0187430200000004e-26*cse13*cse676*cse683*d2g*state.xn[0] - 7.1777529600000012e-12*cse494 - 7.1777529600000012e-12*cse495 - 7.1777529600000012e-12*cse498 - cse570*cse579 - cse588 - cse591 + 3.0000000000000002e-15*cse608*d2g*state.xn[0] - cse610*cse637 + 4.39089220777172e-29*cse642*cse647*cse649*cse653*cse670*state.xn[8] - 4.39089220777172e-29*cse642*cse655 - cse674*(-cse610*cse624 - cse610*cse672) - cse687*(-cse650*cse684 + cse684*cse685) - state.xn[11]*(cse806*cse823 + cse828*cse829 + cse832*cse833 + cse832*cse834) - state.xn[3]*((cse720) ? (
   0
)
: (
   cse731*cse758
)) - state.xn[7]*((cse937) ? (
   cse908*cse925 + cse930*cse931 + cse934*cse935 + cse934*cse936
)
: (
   0
));
  jac(19, 2) = cse951 - state.xn[3]*((cse720) ? (
   0
)
: (
   cse734*cse758
));
  jac(19, 3) = 1.7e-24*chi*cse13*cse561*cse71*cse970*d2g*(0.00025272365919999998*cse133*cse971 + 0.036605692000000002*cse133*cse972) + 4.6527220080000002e-11*crate*cse656*cse658*cse69*cse699*cse975*state.xn[0]*state.xn[5] + 4.22974728e-11*crate*cse656*cse658*cse69*cse975*std::pow(state.xn[0], 2) + 8.4594945600000001e-11*crate*cse656*cse658*cse69*cse975*state.xn[0]*state.xn[3] - 2.1789607199999999e-11*cse156 + 2.6241084038530201e-25*cse289*cse647*cse649*cse653*cse670*state.xn[8] - 2.6241084038530201e-25*cse289*cse655 + 6.0732876035249664e-19*cse573*cse577*cse963*cse964*cse965*state.xn[0]*state.xn[2] - cse579*cse962*state.xn[0] - cse616*cse637 - cse674*(-cse616*cse624 - cse616*cse672) - cse687*(-cse650*cse977 + cse685*cse977) - 7.2151085033443149e-29*cse954*cse960*cse968*cse969*state.xn[2] - cse961 - cse974*(5.9301787935659931e-17*cse573*cse963*cse964*cse965*state.xn[2] - cse575*cse962) - cse976*(0.086512999999999993*chi*cse4*cse71*cse957*cse966*cse967 - cse955*cse959*cse968) - state.xn[11]*(cse823*cse981 + cse829*cse983 + cse833*cse984 + cse834*cse984) - state.xn[3]*((cse720) ? (
   0
)
: (
   cse743*cse758
)) - state.xn[7]*((cse937) ? (
   cse925*cse991 + cse931*cse994 + cse935*cse997 + cse936*cse997
)
: (
   0
));
  jac(19, 4) = 1.4949836836302895e-18*state.c_hat*cse1002*state.rn[0] + 3.8323655572603624e-29*state.c_hat*cse1008*cse676*state.rn[0]*state.xn[3] + 3.3999999999999999e-24*chi*cse561*cse565*cse71*d2g + 1.7623947000000002e-12*crate*cse1016*cse699*state.xn[5] + 1.602177e-12*crate*cse1016*state.xn[0] + 3.2043540000000001e-12*crate*cse1016*state.xn[3] + 1.7623947000000002e-12*crate*cse200*cse669*cse675*state.xn[5] + 3.2043540000000001e-12*crate*cse669 - cse1001 - cse1004*cse646 - cse1005*cse1006 + 2.0187430200000004e-26*cse1008*cse13*cse676*d2g*state.xn[0] - 7.1777529600000012e-12*cse172 - 7.1777529600000012e-12*cse181 - 1.4355505920000002e-11*cse183 - 7.1777529600000012e-12*cse503 + 6.0000000000000005e-15*cse608*d2g*state.xn[0] - cse637 - cse674*(-cse624 - cse672) - cse687*(-cse1005*cse1009 - cse646*cse651) - cse999 - state.xn[11]*(cse1021*cse823 + cse1026*cse829 + cse1028*cse833 + cse1028*cse834) - state.xn[3]*((cse720) ? (
   0
)
: (
   cse732*cse758
)) - state.xn[7]*((cse937) ? (
   cse1032*cse925 + cse1037*cse931 + cse1040*cse935 + cse1040*cse936
)
: (
   0
)) - ((cse720) ? (
   0.0
)
: (
   cse1017*cse754
));
  jac(19, 5) = -cse1001 + 3.3999999999999999e-24*cse1041 + cse1042*cse942 + cse1042*cse945 + cse1043*cse947 + cse1043*cse948 + cse1043*cse950 + cse199*cse940 - 7.1777529600000012e-12*cse209 - 7.1777529600000012e-12*cse509 + cse60*cse608 - cse999;
  jac(19, 6) = cse1044 + cse939*cse949;
  jac(19, 7) = cse1044;
  jac(19, 8) = -((cse937) ? (
   cse918*cse925 + cse922*cse931
)
: (
   0.0
));
  jac(19, 9) = -cse1003*cse686;
  jac(19, 10) = -cse617*cse636 - cse674*(100000.0*cse1045*cse1046*cse617*cse620 - cse632*(100000.0*cse1045*cse1046*cse614*cse620*cse625 - cse1048*cse626) - cse633*(-cse1048*cse628*cse630 + 100000.0*cse1049*cse1050*cse620));
  jac(19, 11) = cse951;
  jac(19, 12) = -cse816*cse823 - cse820*cse829;
  jac(19, 15) = cse951;
  jac(19, 16) = cse951;
  jac(19, 17) = -300.0*cse1000 + 5.0999999999999995e-24*cse1041 + cse1051*cse942 + cse1051*cse945 + cse1052*cse947 + cse1052*cse948 + cse1052*cse950 - 7.1777529600000012e-12*cse250 + cse253*cse940 + cse512*cse608 - 7.1777529600000012e-12*cse513 - 9.6000000000000005e-32*cse998;
  jac(19, 19) = cse316*(1.7e-24*chi*cse13*cse561*cse71*cse970*d2g*(-0.00012636182959999999*cse282*cse971 - 0.018302846000000001*cse282*cse972) + 5.6227499999999995e-25*chi*cse13*cse565*cse71*d2g*std::pow(state.T, -0.85299999999999998) - cse1004*cse1065 - cse1006*cse1068 - 1.6173435610483583e-5*cse1053*cse1054*cse577 - 2.0489994438957239e-11*cse1054*cse1055*cse578 + 7.8348995947646122e-14*cse1056*cse572*cse573*cse577*cse964*state.xn[0]*state.xn[2] - cse1058*cse577*cse973 - cse1060*cse637 - 5.2835524837844135e-18*cse1061*cse653*state.xn[8] - cse1067*cse449*cse941 - cse1067*cse944 - 7.1777529600000012e-12*cse272 - 2.1789607199999999e-11*cse296 - 7.1777529600000012e-12*cse516 - cse582*cse589*psigd_coll_interp2d_partial_1(cse13, state.T) - cse586*(1.6000000000000001e-32*cse113*cse273*cse583 + cse584) - 1000*cse611*std::pow(cse612, 999)*cse635*cse673 - cse674*(-cse1060*cse624 + cse1069*cse1070*cse627*cse631 + cse1069*cse1070 - cse1073*cse617 - cse632*(-cse1069*cse624*cse625 - cse1073*cse614*cse625 - cse1075*cse626) - cse633*(cse1050*(-cse1049*cse1074 + alpha_co_interp2d_partial_0(state.T, cse622)) + cse625*cse671*(cse1060*cse628 - cse1075*cse617/std::pow(cse625, 2)))) - cse687*(-cse1009*cse1068 - 182.41999999999999*cse1061 - cse1065*cse651) - cse974*(-0.0015792330470357991*cse1053*cse569 + 7.6502807803182096e-12*cse1056*cse572*cse573*cse964*state.xn[2] - 2.0007175426973804e-9*cse1057*cse574 - 1.3338116951315871e-8*cse1058*cse570) - cse976*(-0.043256499999999996*cse1059*cse957 + 0.84999999999999998*cse1059*cse959) - cse976*(-1.8989*cse1059*cse969 + cse5*cse958) - state.xn[11]*(cse1094*cse823 + cse1098*cse829 + cse1099*cse833 + cse1099*cse834) - state.xn[3]*((cse720) ? (
   0
)
: (
   cse1017*cse1077 + cse754*cse757*(-cse1077*cse755 - cse745*cse754*(-cse733*(3.0182298984217999*cse1078*cse722*cse746 + 4.8223900769399082*cse1078*cse746 - 3.5529549287336835*cse1080 - 1.3735579540080118*cse1081 - 0.3872755400043702*cse1082) - cse735*(3.6184459289309552*cse1079 + 0.0708789387907936*cse1080 - 4.6974781513675152*cse1081 + 3.7035619079275177*cse1082 - 1.6316107607322889*cse1083) - cse737*(5.0409049417480247*cse1079 - 3.7541549062629067*cse1080 - 1.528565035159452*cse1081 + 2.2094886994529301*cse1082 + 2.0057552335975877*cse1083) - state.xn[0]*((cse727) ? (
      cse726*cse9*(37.383713*cse1079 + 116.29033200000001*cse1080 + 80.639324000000002*cse1081 + 19.239805*cse1082 + 145.968309*cse1083)
   )
   : ((cse729) ? (
      cse728*(8.2184944748967013*cse1078*cse746 - 52.189748993977005*cse1080 - 196.44057098336626*cse1081 - 48.951834264235494*cse1082 - 192.38155095558542*cse1083)
   )
   : (
      cse730*(10.695627721640689*cse1079 - 17.135767342440825*cse1080 - 50.756388852554174*cse1081 + 17.889115159724135*cse1082 + 41.010708268606813*cse1083)
   ))) - state.xn[2]*((cse644) ? (
      cse741*(38.719649225812766*cse1079 + 445.51869310442481*cse1080 + 6766.9891340104432*cse1081 + 11324.985706577943*cse1082 + 2370.6849681533818*cse1083 + 11077.448644792414*cse1084 + 5869.3152370465659*cse1085 + 1300.6686834345674*cse1086)
   )
   : (
      cse742*(3.8689780091986448*cse1079 + 4.297112944704045*cse1080 - 43.540996231705549*cse1081 - 101.40241318979159*cse1082 + 27.91190909651122*cse1083 + 123.18901581604325*cse1084 + 103.76942484394611*cse1085 - 117.34186576967058*cse1086)
   ))))
)) - state.xn[7]*((cse937) ? (
   cse1108*cse925 + cse1112*cse931 + cse1113*cse935 + cse1113*cse936
)
: (
   0
)));
  jac(19, 20) = cse1002*cse521;
  jac(20, 4) = -5.4061971382371991e-18*cse447;
  jac(20, 8) = -cse448;
  jac(20, 10) = -cse539;
  jac(20, 13) = -cse460;
  jac(20, 20) = cse1114;
  jac(21, 4) = -5.4061971382371991e-18*state.c_hat*state.rn[2*0+1];
  jac(21, 8) = -cse490*state.rn[2*0+1];
  jac(21, 10) = -cse543*state.rn[2*0+1];
  jac(21, 13) = -cse488*state.rn[2*0+1];
  jac(21, 21) = cse1114;
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
