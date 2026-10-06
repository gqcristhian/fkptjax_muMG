(* Standalone A(k), B(k) evaluation for HDKI/BZ_Mass, matching fkptjax's own
   MG_kernels.A_B_grid exactly: same model params, same xnow/xstop convention,
   same (r,x) slice used in examples/validation_against_sMGPT.ipynb's Step 3
   (r=0.3, x=0.3, output leg k swept over a 60-point log grid 0.001..1). No
   P(k) dependence -- solveABfull below is the pure growth-kernel ODE solve,
   sMGPT's own Ah/Bh machinery (mirrors 2_P22type.wl's solveABfull /
   this session's wolfram_bzmass_loopintegral_snippet.txt), independent of
   any input power spectrum. *)

om = 0.31519; lambdaA = 100.0; lambdaDS = 100.0; muKinf = 1.2;
z = 0.295; etaini = -3.912023; etaev = Log[1./(1. + z)];

mu[eta_, k_] := Module[{a, Dd, X},
   a = Exp[eta]; Dd = lambdaDS*a^(-3) + lambdaA;
   X = (k*lambdaA*lambdaDS/(Dd*a))^2; (1 + muKinf*X)/(1 + X)];

f1[eta_] := 2. - 3./(2.(1. + (1. - om)/om Exp[3. eta]));
f2[eta_] := 3./(2.(1. + (1. - om)/om Exp[3. eta]));

sourceA[kf_, k1_, k2_, eta_] := f2[eta]*(mu[eta, kf] +
    (mu[eta, kf] - mu[eta, k1])*(kf^2 - k1^2 - k2^2)/(2*k2^2) +
    (mu[eta, kf] - mu[eta, k2])*(kf^2 - k1^2 - k2^2)/(2*k1^2));
sourceb[kf_, k1_, k2_, eta_] := f2[eta]*(mu[eta, k1] + mu[eta, k2] - mu[eta, kf]);
D2plusi = 3. Exp[2. etaini]/7.; dD2plusi = 6. Exp[2. etaini]/7.;

solveABfull[kf_?NumericQ, k1_?NumericQ, k2_?NumericQ] := Module[
  {sol, Dpk1v, dDpk1v, Dpk2v, dDpk2v, Afv, dAfv, Bfv, dBfv, C, Cp},
  sol = NDSolve[{
     Dpk1''[eta] + f1[eta] Dpk1'[eta] - f2[eta] mu[eta, k1] Dpk1[eta] == 0,
     Dpk2''[eta] + f1[eta] Dpk2'[eta] - f2[eta] mu[eta, k2] Dpk2[eta] == 0,
     Af''[eta] + f1[eta] Af'[eta] - f2[eta] mu[eta, kf] Af[eta] == sourceA[kf, k1, k2, eta] Dpk1[eta] Dpk2[eta],
     Bf''[eta] + f1[eta] Bf'[eta] - f2[eta] mu[eta, kf] Bf[eta] == sourceb[kf, k1, k2, eta] Dpk1[eta] Dpk2[eta],
     Dpk1[etaini] == Exp[etaini], Dpk1'[etaini] == Exp[etaini],
     Dpk2[etaini] == Exp[etaini], Dpk2'[etaini] == Exp[etaini],
     Af[etaini] == D2plusi, Af'[etaini] == dD2plusi,
     Bf[etaini] == D2plusi, Bf'[etaini] == dD2plusi},
    {Dpk1, Dpk2, Af, Bf}, {eta, etaini, etaev}, MaxSteps -> Infinity, AccuracyGoal -> 10, PrecisionGoal -> 10];
  Dpk1v = Dpk1[etaev] /. sol[[1]]; dDpk1v = Dpk1'[etaev] /. sol[[1]];
  Dpk2v = Dpk2[etaev] /. sol[[1]]; dDpk2v = Dpk2'[etaev] /. sol[[1]];
  Afv = Af[etaev] /. sol[[1]]; dAfv = Af'[etaev] /. sol[[1]];
  Bfv = Bf[etaev] /. sol[[1]]; dBfv = Bf'[etaev] /. sol[[1]];
  C = (3./7.) Dpk1v Dpk2v; Cp = (3./7.) (dDpk1v Dpk2v + Dpk1v dDpk2v);
  {Afv/C, Bfv/C, dAfv/C - Afv Cp/C^2, dBfv/C - Bfv Cp/C^2}];

r = 0.3; x = 0.3;
nk = 60; kmin = 0.001; kmax = 1.0;
kGrid = Table[10^(Log10[kmin] + (ii - 1)*(Log10[kmax] - Log10[kmin])/(nk - 1)), {ii, 1, nk}];

Print["# k  A  B  Ap  Bp"];
Do[
  kf = kGrid[[ii]];
  q = r*kf;
  kminus = kf*Sqrt[1. + r^2 - 2.*r*x];
  {Av, Bv, Apv, Bpv} = solveABfull[kf, q, kminus];
  Print[NumberForm[kf, {16, 10}], " ", NumberForm[Av, {16, 10}], " ", NumberForm[Bv, {16, 10}], " ",
        NumberForm[Apv, {16, 10}], " ", NumberForm[Bpv, {16, 10}]],
  {ii, 1, nk}
];
Print["DONE"];
