(* ::Package::

  Standalone, self-contained cross-check of eq. (2.20) of arXiv:2012.05077
  (Aviles et al., "Redshift space power spectrum beyond Einstein-de Sitter
  kernels"):

      F2[k1,k2] = 1/2 + 3/14 A + (1/2 - 3/14 B) x^2 + x/2 (k2/k1 + k1/k2)

  against the numbers printed by examples/check_bispectrum_Z2_singlepoint.py
  (folps.BispectrumCalculator_fk.Z2, isolated to its F2 piece via b1=1,
  b2=bs=mui=muj=0, for HDKI/BZ_Mass, mu_kinf=1.2, lambda_a=lambda_dS=1.0,
  Om=0.315192, z=0.295).

  This does NOT re-derive A(kf,k1,k2)/B(kf,k1,k2) themselves (that ODE
  machinery was already independently cross-checked in
  wolfram_I1udd1_singlepoint... style snippets for the power spectrum) --
  it takes those already-computed numeric A/B values as GIVEN inputs and
  checks purely that folps' own Z2 code correctly implements the eq.(2.20)
  algebra, which is the piece that's new to the bispectrum wiring.

  No external packages, no cloud connection needed -- run locally with:
      wolframscript -file wolfram_bispectrum_Z2_singlepoint_snippet.wl
*)

F2[k1_, k2_, x_, A_, B_] := 1/2 + 3/14 A + (1/2 - 3/14 B) x^2 + x/2 (k2/k1 + k1/k2);

(* --- B12 leg: kf = k3, (k1,k2) legs --- *)
k1B12 = 0.1; k2B12 = 0.15; xB12 = 0.3;
AB12 = 1.0051564551147658; BB12 = 1.0049192097891295;
f2B12 = F2[k1B12, k2B12, xB12, AB12, BB12];

(* --- B23 leg: kf = k1, (k2,k3) legs --- *)
k1B23 = 0.15; k2B23 = 0.20371548787463364; xB23 = -0.8835852486128686;
AB23 = 1.0050412908583937; BB23 = 1.005034292828503;
f2B23 = F2[k1B23, k2B23, xB23, AB23, BB23];

(* --- B31 leg: kf = k2, (k3,k1) legs --- *)
k1B31 = 0.20371548787463364; k2B31 = 0.1; xB31 = -0.7117770058270332;
AB31 = 1.0050491613760977; BB31 = 1.0049885674643002;
f2B31 = F2[k1B31, k2B31, xB31, AB31, BB31];

Print["Wolfram eq.(2.20) F2, independent evaluation:"];
Print["  F2(k1,k2) [B12 leg] = ", InputForm[f2B12]];
Print["  F2(k2,k3) [B23 leg] = ", InputForm[f2B23]];
Print["  F2(k3,k1) [B31 leg] = ", InputForm[f2B31]];

expected = {1.0660100841929452, 0.012286243457727303, -0.04012226041563957};
got = {f2B12, f2B23, f2B31};
Print["Reference (folps Z2-isolated F2, from check_bispectrum_Z2_singlepoint.py):"];
Print["  ", InputForm[expected]];
Print["Absolute differences: ", InputForm[Abs[got - expected]]];

(* ================================================================
   eq. (2.21): G2[k1,k2] = (3 A (f1+f2) + 3 Ap)/(14 f0)
                            + x/2 (f1/f0 k2/k1 + f2/f0 k1/k2)
                            + ( (f1+f2)/(2 f0) - (3 B (f1+f2) + 3 Bp)/(14 f0) ) x^2
   -- exercises Ap/Bp, which F2 never touches.
   ================================================================ *)
(* NOTE: kept on ONE line deliberately -- wolframscript -file was found (empirically,
   during this validation) to mis-parse a SetDelayed body split across multiple lines
   without per-line semicolons as separate top-level statements, silently keeping only
   the first line as the function body. Confirmed by isolating each term and by
   comparing single-line vs multi-line versions of this exact definition. *)
G2[k1_, k2_, x_, f1_, f2_, f0_, A_, Ap_, B_, Bp_] := (3 A (f1 + f2) + 3 Ap)/(14 f0) + (x/2) (f1/f0 k1/k2 + f2/f0 k2/k1) + ((f1 + f2)/(2 f0) - (3 B (f1 + f2) + 3 Bp)/(14 f0)) x^2;

f1B12 = 0.6826746563367813; f2B12v = 0.6827482022248875; f0B12 = f1B12;
ApB12 = 0.01136587765969721; BpB12 = 0.010755415971214877;
g2B12 = G2[k1B12, k2B12, xB12, f1B12, f2B12v, f0B12, AB12, ApB12, BB12, BpB12];

f1B23 = 0.6827482022248875; f2B23v = 0.6828596091444209; f0B23 = f1B23;
ApB23 = 0.011068128158207458; BpB23 = 0.011050163845714112;
g2B23 = G2[k1B23, k2B23, xB23, f1B23, f2B23v, f0B23, AB23, ApB23, BB23, BpB23];

f1B31 = 0.6828596091444209; f2B31v = 0.6826746563367813; f0B31 = f1B31;
ApB31 = 0.011088903632522795; BpB31 = 0.010932916976293505;
g2B31 = G2[k1B31, k2B31, xB31, f1B31, f2B31v, f0B31, AB31, ApB31, BB31, BpB31];

Print[""];
Print["Wolfram eq.(2.21) G2, independent evaluation:"];
Print["  G2(k1,k2) [B12 leg] = ", InputForm[g2B12]];
Print["  G2(k2,k3) [B23 leg] = ", InputForm[g2B23]];
Print["  G2(k3,k1) [B31 leg] = ", InputForm[g2B31]];

expectedG2 = {0.810334181053353, -0.04938667954409143, -0.17885417494882733};
gotG2 = {g2B12, g2B23, g2B31};
Print["Reference (folps Z2-extracted G2, from check_bispectrum_Z2_singlepoint.py):"];
Print["  ", InputForm[expectedG2]];
Print["Absolute differences: ", InputForm[Abs[gotG2 - expectedG2]]];
