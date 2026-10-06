(* Standalone Gamma2evR-type (D2) and C3Gamma3-type (D3) evaluation for
   HDKI/BZ_Mass, matching fkptjax's own MG_kernels.D2_fused_grid /
   D3_fused_grid exactly: same model params, same xnow/xstop convention.
   solveD2fusedFull / solveD3fused below are sMGPT's OWN functions, copied
   verbatim from this session's wolfram_bzmass_loopintegral_snippet.txt
   (already cross-checked at a single point there). No P(k) dependence --
   pure growth-kernel ODE solves, like A/B. *)

om = 0.31519; lambdaA = 100.0; lambdaDS = 100.0; muKinf = 1.2;
z = 0.295; etaini = -3.912023; etaev = Log[1./(1. + z)];
f0 = 0.6826138224244257; (* matches the official BZ_Mass run's f0 exactly *)

mu[eta_, k_] := Module[{a, Dd, X},
   a = Exp[eta]; Dd = lambdaDS*a^(-3) + lambdaA;
   X = (k*lambdaA*lambdaDS/(Dd*a))^2; (1 + muKinf*X)/(1 + X)];

f1[eta_] := 2. - 3./(2.(1. + (1. - om)/om Exp[3. eta]));
f2[eta_] := 3./(2.(1. + (1. - om)/om Exp[3. eta]));

kppf[x_, kk_, p_] := Sqrt[kk^2 + p^2 + 2. kk p x];
src2a[eta_, x_, kk_, p_] := f2[eta] mu[eta, kppf[x, kk, p]];
src2b[eta_, x_, kk_, p_] := f2[eta] (mu[eta, kk] + mu[eta, p] - mu[eta, kppf[x, kk, p]]);
src2FL[eta_, x_, kk_, p_] := f2[eta] (mu[eta, kppf[x, kk, p]]*(p/kk + kk/p)*x - kk x/p mu[eta, kk] - p x/kk mu[eta, p]);
srcD2[eta_, x_, kk_, p_] := src2a[eta, x, kk, p] - src2b[eta, x, kk, p] x^2 + src2FL[eta, x, kk, p];

solveD2fusedFull[x_?NumericQ, kk_?NumericQ, p_?NumericQ] := Module[
  {sol, Dpkv, dDpkv, Dppv, dDppv, D2v, dD2v, C, Cp, kplusp},
  kplusp = kppf[x, kk, p];
  sol = NDSolve[{
     Dpk''[eta] + f1[eta] Dpk'[eta] - f2[eta] mu[eta, kk] Dpk[eta] == 0,
     Dpp''[eta] + f1[eta] Dpp'[eta] - f2[eta] mu[eta, p] Dpp[eta] == 0,
     D2f''[eta] + f1[eta] D2f'[eta] - f2[eta] mu[eta, kplusp] D2f[eta] == srcD2[eta, x, kk, p] Dpk[eta] Dpp[eta],
     Dpk[etaini] == Exp[etaini], Dpk'[etaini] == Exp[etaini],
     Dpp[etaini] == Exp[etaini], Dpp'[etaini] == Exp[etaini],
     D2f[etaini] == 3. Exp[2. etaini]/7. (1 - x^2), D2f'[etaini] == 6. Exp[2. etaini]/7. (1 - x^2)},
    {Dpk, Dpp, D2f}, {eta, etaini, etaev}, MaxSteps -> Infinity, AccuracyGoal -> 10, PrecisionGoal -> 10];
  Dpkv = Dpk[etaev] /. sol[[1]]; dDpkv = Dpk'[etaev] /. sol[[1]];
  Dppv = Dpp[etaev] /. sol[[1]]; dDppv = Dpp'[etaev] /. sol[[1]];
  D2v = D2f[etaev] /. sol[[1]]; dD2v = D2f'[etaev] /. sol[[1]];
  C = (3./7.) Dpkv Dppv; Cp = (3./7.) (dDpkv Dppv + Dpkv dDppv);
  {D2v/C, dD2v/C - D2v Cp/C^2}];

src3I[eta_, x_, kk_, p_, D2fv_, D2mv_, Dpkv_, Dppv_] :=
  (f2[eta] (mu[eta, p] + mu[eta, kppf[x, kk, p]] - mu[eta, kk]) D2fv Dppv +
      srcD2[eta, x, kk, p] Dpkv Dppv Dppv) (1. - x^2)/(1. + (p/kk)^2 + 2 p/kk x) +
   (f2[eta] (mu[eta, p] + mu[eta, kppf[-x, kk, p]] - mu[eta, kk]) D2mv Dppv +
      srcD2[eta, -x, kk, p] Dpkv Dppv Dppv) (1. - x^2)/(1. + (p/kk)^2 - 2 p/kk x);

src3IIplus[eta_, x_, kk_, p_, D2fv_, Dpkv_, Dppv_] :=
  -f2[eta] (mu[eta, p] + mu[eta, kppf[x, kk, p]] - 2 mu[eta, kk]) Dppv (D2fv + Dpkv Dppv x^2) -
   f2[eta] (mu[eta, kppf[x, kk, p]] - mu[eta, kk]) Dpkv Dppv Dppv -
   f2[eta] ((mu[eta, kppf[x, kk, p]]*(p/kk + kk/p)*x - kk x/p mu[eta, kk] - p x/kk mu[eta, p])) Dpkv Dppv Dppv;
src3II[eta_, x_, kk_, p_, D2fv_, D2mv_, Dpkv_, Dppv_] :=
  src3IIplus[eta, x, kk, p, D2fv, Dpkv, Dppv] + src3IIplus[eta, -x, kk, p, D2mv, Dpkv, Dppv];

srcFL3Plus[eta_, x_, kk_, p_, D2fv_, Dpkv_, Dppv_] :=
  f2[eta] (((p^2 + p kk x)/(kk^2 + p^2 + 2. p kk x)) (mu[eta, p] - mu[eta, kk]) D2fv Dppv +
      ((p^2 + p kk x)/p^2) (mu[eta, kppf[x, kk, p]] - mu[eta, kk]) (D2fv Dppv + (1 + x^2) Dpkv Dppv Dppv) +
      (((p^2 + kk^2) x^2)/p^2 + ((p^2 + kk^2) x)/(p kk)) (mu[eta, kppf[x, kk, p]] - mu[eta, kk]) Dpkv Dppv Dppv);
srcFL3[eta_, x_, kk_, p_, D2fv_, D2mv_, Dpkv_, Dppv_] :=
  srcFL3Plus[eta, x, kk, p, D2fv, Dpkv, Dppv] + srcFL3Plus[eta, -x, kk, p, D2mv, Dpkv, Dppv];

solveD3fused[x_?NumericQ, kk_?NumericQ, p_?NumericQ] := Module[
  {sol, Dpkv, dDpkv, Dppv, dDppv, D3v, dD3v, C, xoneMx2},
  xoneMx2 = 1 - x^2;
  sol = NDSolve[{
     Dpk''[eta] + f1[eta] Dpk'[eta] - f2[eta] mu[eta, kk] Dpk[eta] == 0,
     Dpp''[eta] + f1[eta] Dpp'[eta] - f2[eta] mu[eta, p] Dpp[eta] == 0,
     D2f''[eta] + f1[eta] D2f'[eta] - f2[eta] mu[eta, kppf[x, kk, p]] D2f[eta] == srcD2[eta, x, kk, p] Dpk[eta] Dpp[eta],
     D2m''[eta] + f1[eta] D2m'[eta] - f2[eta] mu[eta, kppf[-x, kk, p]] D2m[eta] == srcD2[eta, -x, kk, p] Dpk[eta] Dpp[eta],
     D3s''[eta] + f1[eta] D3s'[eta] - f2[eta] mu[eta, kk] D3s[eta] ==
        src3I[eta, x, kk, p, D2f[eta], D2m[eta], Dpk[eta], Dpp[eta]] +
        src3II[eta, x, kk, p, D2f[eta], D2m[eta], Dpk[eta], Dpp[eta]] +
        srcFL3[eta, x, kk, p, D2f[eta], D2m[eta], Dpk[eta], Dpp[eta]],
     Dpk[etaini] == Exp[etaini], Dpk'[etaini] == Exp[etaini],
     Dpp[etaini] == Exp[etaini], Dpp'[etaini] == Exp[etaini],
     D2f[etaini] == 3. Exp[2. etaini]/7. xoneMx2, D2f'[etaini] == 6. Exp[2. etaini]/7. xoneMx2,
     D2m[etaini] == 3. Exp[2. etaini]/7. xoneMx2, D2m'[etaini] == 6. Exp[2. etaini]/7. xoneMx2,
     D3s[etaini] == 5./7. Exp[3. etaini]/9. (xoneMx2/(1. + (p/kk)^2 + 2 p/kk x) + xoneMx2/(1. + (p/kk)^2 - 2 p/kk x)) xoneMx2,
     D3s'[etaini] == 15./7. Exp[3. etaini]/9. (xoneMx2/(1. + (p/kk)^2 + 2 p/kk x) + xoneMx2/(1. + (p/kk)^2 - 2 p/kk x)) xoneMx2},
    {Dpk, Dpp, D2f, D2m, D3s}, {eta, etaini, etaev}, MaxSteps -> Infinity, AccuracyGoal -> 10, PrecisionGoal -> 10];
  Dpkv = Dpk[etaev] /. sol[[1]]; Dppv = Dpp[etaev] /. sol[[1]];
  D3v = D3s[etaev] /. sol[[1]]; dD3v = D3s'[etaev] /. sol[[1]];
  {D3v/(Dpkv Dppv Dppv), dD3v/(Dpkv Dppv Dppv)/(3 f0)}];

r = 0.3; x = 0.3;
nk = 40; kmin = 0.001; kmax = 1.0;
kGrid = Table[10^(Log10[kmin] + (ii - 1)*(Log10[kmax] - Log10[kmin])/(nk - 1)), {ii, 1, nk}];

Print["# k  D2  D2p  CFD3  CFD3p"];
Do[
  kk = kGrid[[ii]];
  p = r*kk;
  {D2v, D2pv} = solveD2fusedFull[x, kk, p];
  {C3g, C3gf} = solveD3fused[x, kk, p];
  Print[NumberForm[kk, {16, 10}], " ", NumberForm[D2v, {16, 10}], " ", NumberForm[D2pv, {16, 10}], " ",
        NumberForm[C3g, {16, 10}], " ", NumberForm[C3gf, {16, 10}]],
  {ii, 1, nk}
];
Print["DONE"];
