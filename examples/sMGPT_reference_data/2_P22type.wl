(* ::Package:: *)

(* ::Title:: *)
(*RSD combined P22, A, B and C functions*)


(* ::Text:: *)
(*This notebook combines the repeated k-p-x integration routine from 2_P22, 4_A, and 5_BandC. The A kernel list is renamed to AKernelsT and the B/C kernel list is renamed to BCKernelsT so all kernels can coexist. The final export cell writes P22functions_*.dat, Afunctions_*.dat, and BandCfunctions_*.dat separately.*)


(* Make the file work both from a notebook and from Get["file.wl"]. *)
(* Resolve directories without changing the project working directory to src/. *)
If[! ValueQ[ProjectDirectory],
  ProjectDirectory = If[StringQ[$InputFileName] && StringLength[$InputFileName] > 0,
    DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]],
    If[ValueQ[NotebookDirectory] && StringQ[NotebookDirectory[]], NotebookDirectory[], Directory[]]
  ];
];
ProjectDirectory = ExpandFileName[ProjectDirectory];
If[! StringEndsQ[ProjectDirectory, $PathnameSeparator], ProjectDirectory = ProjectDirectory <> $PathnameSeparator];
If[! ValueQ[SourceDirectory], SourceDirectory = FileNameJoin[{ProjectDirectory, "src"}]];
SetDirectory[ProjectDirectory];
dir = ProjectDirectory;

Print["Running combined P22, A, B/C functions"];
(*<<0_params.m*)
(*<<0_commonfunctions.m*)
If[NumberQ[bQdef], None, Get[FileNameJoin[{SourceDirectory, "folders.m"}]]];


(* ::Section:: *)
(*General parameters*)


(*PSLT=Import[outputdir<>"/PSL_"<>suffix<>".dat"];
PSL=Interpolation[PSLT];

fkTh=Import[outputdir<>"/fk_"<>suffix<>".dat"];
f0=fkTh[[2,2]];
Print["f(k=0)=",f0];
(*If[bLCDM\[Equal]1,fk[k_]:=f0,fk=Interpolation[fkTh]];*)(* Assuming the MG model f[k=0]=f_LCDM. Does not work in DGP *)
fk=Interpolation[fkTh]*)

PSL=Interpolation[PSLT]
sizepk=Dimensions[PSLT][[1]];
Print["input PS points = ",sizepk,", kmin = ",PSLT[[1,1]],", kmax = ",PSLT[[sizepk,1]]];
fourpi2=4.Pi^2;






(* ::Section:: *)
(*General functions (modify here for other models)*)


(* ::Input:: *)
(**)


(* ::Chapter:: *)
(*Second order functions differential equations*)


etainI=-4;
etafI=0.0;
etaini=etainI;

Dplusi=Exp[etaini];
dDplusi=Exp[etaini];


D2plusi=3Exp[2etaini]/7;
dD2plusi=6Exp[2etaini]/7;
etafinal=etaev;
Clear[kf, k1, k2, Af, Bf, Dpk1, Dpk2, sysD2Values, d2Values, A, B, Aprime, Bprime]

(* Faster second-order ODE evaluation.
   The old version returned full InterpolatingFunctions and then used D[...] for Aprime/Bprime
   at every quadrature point.  Here we ask ParametricNDSolveValue only for the values needed
   at etaev: Af, Bf, their eta-derivatives, and the two growth factors and derivatives.
   d2Values memoizes those 8 numbers, so A, B, Aprime and Bprime do not trigger four
   independent ODE solves for the same {k12,k1,k2}. *)

sysD2Values = ParametricNDSolveValue[
  {
    Af''[eta] + f1[eta] Af'[eta] - f2[eta] mu[eta, kf] Af[eta] ==
      sourceA[kf, k1, k2, eta] Dpk1[eta] Dpk2[eta],
    Bf''[eta] + f1[eta] Bf'[eta] - f2[eta] mu[eta, kf] Bf[eta] ==
      sourceb[kf, k1, k2, eta] Dpk1[eta] Dpk2[eta],
    Dpk1''[eta] + f1[eta] Dpk1'[eta] - f2[eta] mu[eta, k1] Dpk1[eta] == 0,
    Dpk2''[eta] + f1[eta] Dpk2'[eta] - f2[eta] mu[eta, k2] Dpk2[eta] == 0,
    Af'[etaini] == dD2plusi, Af[etaini] == D2plusi,
    Bf'[etaini] == dD2plusi, Bf[etaini] == D2plusi,
    Dpk1'[etaini] == Exp[etaini], Dpk1[etaini] == Exp[etaini],
    Dpk2'[etaini] == Exp[etaini], Dpk2[etaini] == Exp[etaini]
  },
  {Af[etaev], Bf[etaev], Af'[etaev], Bf'[etaev],
   Dpk1[etaev], Dpk2[etaev], Dpk1'[etaev], Dpk2'[etaev]},
  {eta, etainI, etafinal},
  {kf, k1, k2}
];

ClearODECache[] := (Clear[d2Values];);

d2Values[k12_?NumericQ, kk1_?NumericQ, kk2_?NumericQ] :=
  d2Values[k12, kk1, kk2] = sysD2Values[k12, kk1, kk2];

Dk1[kk1_?NumericQ] := d2Values[0.001, kk1, 0.001][[5]];
Dk2[kk2_?NumericQ] := d2Values[0.001, 0.001, kk2][[6]];

A[k12_?NumericQ, kk1_?NumericQ, kk2_?NumericQ] := Module[{v, denom},
  v = d2Values[k12, kk1, kk2];
  denom = (3./7.) v[[5]] v[[6]];
  v[[1]]/denom
];

B[k12_?NumericQ, kk1_?NumericQ, kk2_?NumericQ] := Module[{v, denom},
  v = d2Values[k12, kk1, kk2];
  denom = (3./7.) v[[5]] v[[6]];
  v[[2]]/denom
];

Aprime[k12_?NumericQ, kk1_?NumericQ, kk2_?NumericQ] := Module[{v, d1, d2, d1p, d2p},
  v = d2Values[k12, kk1, kk2];
  {d1, d2, d1p, d2p} = v[[{5, 6, 7, 8}]];
  (v[[3]] d1 d2 - v[[1]] (d1p d2 + d1 d2p))/((3./7.) (d1 d2)^2)
];

Bprime[k12_?NumericQ, kk1_?NumericQ, kk2_?NumericQ] := Module[{v, d1, d2, d1p, d2p},
  v = d2Values[k12, kk1, kk2];
  {d1, d2, d1p, d2p} = v[[{5, 6, 7, 8}]];
  (v[[4]] d1 d2 - v[[2]] (d1p d2 + d1 d2p))/((3./7.) (d1 d2)^2)
];

AprimeLCDM = Aprime[0.0001, 0.0001, 0.0001]; (* Large-scales value. For f(R) this is LCDM. *)
ALCDM = A[0.00001, 0.00001, 0.00001];       (* Large-scales value. For f(R) this is LCDM. *)



(* ::Chapter:: *)
(*Tables of k and p points*)


ttft=True;
Get[FileNameJoin[{SourceDirectory, "tables.m"}]];
ttft=False;


(* ::Chapter:: *)
(*make Pdd Pud and Puu*)


(* ::Input:: *)
(**)


If[PTkernels==3||PTkernels==5,Clear[fk];fk[k_]:=f0];
Pdd[k_]:=PSL[k]
Pdu[k_]:=fk[k]/f0 PSL[k]
Puu[k_]:=(fk[k]/f0)^2PSL[k]


(* ::Chapter:: *)
(*Kernels and combined integration*)


(* ::Section:: *)
(*Define P22 kernels (P22kernelsT)*)


(*bLCDM=1;*)
AngleEvQ[r_,x_]:=(x-r )/Sqrt[1+r^2-2. r x];

S2evQ[r_,x_]:=AngleEvQ[r,x]AngleEvQ[r,x]-1/3;

F2evQ[k_,r_,x_,A_,B_]:=0.5+3./14. A +(0.5-3./14. B)AngleEvQ[r,x]^2 +AngleEvQ[r,x]/2 ( Sqrt[1+r^2-2. r x]/r + r/Sqrt[1+r^2-2. r x] );

G2evQ[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=3./14. A (fp+fkminusp)/f0 + 3/14.Aprime/f0 +( 1./2 (fp+fkminusp) -  3./14. B(fp+fkminusp)-3/14.Bprime)*AngleEvQ[r,x]^2/f0 +AngleEvQ[r,x]/(2f0) (fkminusp Sqrt[1+r^2-2. r x]/r + fp r/Sqrt[1+r^2-2. r x] );


KF2sq[k_,r_,x_,A_,B_]:=2r^2 (F2evQ[k,r,x,A,B])^2;
KG2sq[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=2r^2 (G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp])^2;
KF2G2[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=2r^2 F2evQ[k,r,x,A,B]G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp];



KPb2b1[k_,r_,x_,A_,B_]:=r^2F2evQ[k,r,x,A,B];
KPbs2b1[k_,r_,x_,A_,B_]:=r^2S2evQ[r,x]F2evQ[k,r,x,A,B];

KPb22[k_,r_,x_]:=1/2r^2 (1/2(1-PSL[k r]/PSL[k Sqrt[1+r^2-2 r x]])+1/2(1-PSL[k Sqrt[1+r^2-2 r x]]/PSL[k r]));

KPb2s2[k_,r_,x_]:=1/2r^2 (1/2(S2evQ[r,x]-2/3PSL[k r]/PSL[k Sqrt[1+r^2-2 r x]])+1/2(S2evQ[r,x]-2/3PSL[k Sqrt[1+r^2-2 r x]]/PSL[k r]));
KPbs22[k_,r_,x_]:=1/2r^2 (1/2(S2evQ[r,x]S2evQ[r,x]-4/9PSL[k r]/PSL[k Sqrt[1+r^2-2 r x]])+1/2(S2evQ[r,x]S2evQ[r,x]-4/9PSL[k Sqrt[1+r^2-2 r x]]/PSL[k r]));


KPb2theta[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=r^2G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp];
KPbs2theta[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=r^2S2evQ[r,x]G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp];


If[PTkernels==1||PTkernels==2,bLCDMkernels=0,bLCDMkernels=1];
If[PTkernels==3||PTkernels==5,Clear[fk];fk[k_]:=f0];
If[PTkernels==5||PTkernels==6,ALCDM=1;AprimeLCDM=0];

If[bLCDMkernels==0,Ah[k12_,k1_,k2_]:=A[k12,k1,k2],Ah[k12_,k1_,k2_]:=ALCDM]
If[bLCDMkernels==0,Bh[k12_,k1_,k2_]:=B[k12,k1,k2],Bh[k12_,k1_,k2_]:=ALCDM]
If[bLCDMkernels==0,Aprimeh[k12_,k1_,k2_]:=Aprime[k12,k1,k2],Aprimeh[k12_,k1_,k2_]:=AprimeLCDM]
If[bLCDMkernels==0,Bprimeh[k12_,k1_,k2_]:=Bprime[k12,k1,k2],Bprimeh[k12_,k1_,k2_]:=AprimeLCDM]





P22kernelsT[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:={KF2sq[k,r,x,A,B],KG2sq[k,r,x,A,B,Aprime,Bprime,fp,fkminusp],KF2G2[k,r,x,A,B,Aprime,Bprime,fp,fkminusp],KPb2b1[k,r,x,A,B],KPbs2b1[k,r,x,A,B],KPb22[k,r,x],KPb2s2[k,r,x],KPbs22[k,r,x],KPb2theta[k,r,x,A,B,Aprime,Bprime,fp,fkminusp],KPbs2theta[k,r,x,A,B,Aprime,Bprime,fp,fkminusp]};
P22kernelsT[1,1,0.3,1,1,0,0,1,1];
NumQs=Length@P22kernelsT[1,1,0.3,1,1,0,0,1,1];
Print["Integrating ", NumQs, " kernels"];




(* ::Section:: *)
(*Define A kernels (AAKernelsT)*)


(* ::Text:: *)
(*G2evN = Subscript[G, 2](-k,k-p)  the new form of evaluation, Neither R or Q*)


If[PTkernels==1||PTkernels==2,bLCDMkernels=0,bLCDMkernels=1];
If[PTkernels==3||PTkernels==5,Clear[fk];fk[k_]:=f0];
If[PTkernels==5||PTkernels==6,ALCDM=1;AprimeLCDM=0];

If[bLCDMkernels==0,Ah[k12_,k1_,k2_]:=A[k12,k1,k2],Ah[k12_,k1_,k2_]:=ALCDM]
If[bLCDMkernels==0,Bh[k12_,k1_,k2_]:=B[k12,k1,k2],Bh[k12_,k1_,k2_]:=ALCDM]
If[bLCDMkernels==0,Aprimeh[k12_,k1_,k2_]:=Aprime[k12,k1,k2],Aprimeh[k12_,k1_,k2_]:=AprimeLCDM]
If[bLCDMkernels==0,Bprimeh[k12_,k1_,k2_]:=Bprime[k12,k1,k2],Bprimeh[k12_,k1_,k2_]:=AprimeLCDM]

AngleEvQ[r_,x_]:=(x-r )/Sqrt[1+r^2-2. r x];
AngleEvR[r_,x_]:=-x;
AngleEvN[r_,x_]:=-((1-r x)/Sqrt[1+r^2-2. r x]);


F2evN[k_,r_,x_,A_,B_]:=0.5+3./14. A +(0.5-3./14. B)AngleEvN[r,x]^2 +AngleEvN[r,x]/2 ( Sqrt[1+r^2-2. r x]+ 1/Sqrt[1+r^2-2. r x] );

G2evN[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=3./14. A (fk+fkminusp)/f0 + 3/14.Aprime/f0 +( 1./2 (fk+fkminusp) -  3./14. B(fk+fkminusp)-3/14.Bprime)*AngleEvN[r,x]^2/f0 +AngleEvN[r,x]/(2f0) (fkminusp Sqrt[1+r^2-2. r x] + fk/Sqrt[1+r^2-2. r x] );



F2evQ[k_,r_,x_,A_,B_]:=0.5+3./14. A +(0.5-3./14. B)AngleEvQ[r,x]^2 +AngleEvQ[r,x]/2 ( Sqrt[1+r^2-2. r x]/r + r/Sqrt[1+r^2-2. r x] );

G2evQ[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=3./14. A (fp+fkminusp)/f0 + 3/14.Aprime/f0 +( 1./2 (fp+fkminusp) -  3./14. B(fp+fkminusp)-3/14.Bprime)*AngleEvQ[r,x]^2/f0 +AngleEvQ[r,x]/(2f0) (fkminusp Sqrt[1+r^2-2. r x]/r + fp r/Sqrt[1+r^2-2. r x] );

F2evR[k_,r_,x_,A_,B_]:=0.5+3./14. A +(0.5-3./14. B)AngleEvR[r,x]^2 +AngleEvR[r,x]/2. ( 1/r + r);

G2evR[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=3./14. A (fp+fk)/f0 + 3/14.Aprime/f0 +( 1./2 (fp+fk) -  3./14. B(fp+fk)-3/14.Bprime)*AngleEvR[r,x]^2/f0 +AngleEvR[r,x]/(2f0) (fk/r+ fp r);


(*Note functions Aij and tAij are defined a little different than in TNS paper: To bring them to TNS form one has to multiply them by (1+r^2-2r x)^2 *)

A11[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=2(G2evN[k,r,x,A,B,Aprime,Bprime,fk,fkminusp]x r+F2evN[k,r,x,A,B] (r^2(1-x r))/(1+r^2-2r x) fkminusp/f0);
tA11[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=2F2evQ[k,r,x,A,B](x r fp/f0+(r^2(1-x r))/(1+r^2-2r x) fkminusp/f0);
a11[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=2(F2evR[k,r,x,A,B]x r fp/f0+G2evR[k,r,x,A,B,Aprime,Bprime,fk,fp] (r^2(1-x r))/(1+r^2-2r x));


A12[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=-((r^2(1-x^2))/(1+r^2-2r x))G2evN[k,r,x,A,B,Aprime,Bprime,fk,fkminusp]fkminusp/f0;
tA12[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=-((r^2(1-x^2))/(1+r^2-2r x))F2evQ[k,r,x,A,B]fp fkminusp/f0^2;
a12[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=-((r^2(1-x^2))/(1+r^2-2r x))G2evR[k,r,x,A,B,Aprime,Bprime,fk,fp]fp/f0;



A22[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=((r^2(1-3x^2)+2x r)/(1+r^2-2r x) fkminusp/f0+ 2x r fk/f0)G2evN[k,r,x,A,B,Aprime,Bprime,fk,fkminusp]+(2r^2(1-r x))/(1+r^2-2r x) (fkminusp fk)/f0^2  F2evN[k,r,x,A,B];

tA22[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=((2r^2(1-r x))/(1+r^2-2r x) fkminusp/f0+ 2x r fp/f0)G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp]+(r^2(1-3x^2)+2x r)/(1+r^2-2r x) (fkminusp fp)/f0^2  F2evQ[k,r,x,A,B];


a22[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=((r^2(1-3x^2)+2x r)/(1+r^2-2r x) fp/f0+ (2r^2(1-r x))/(1+r^2-2r x) fk/f0)G2evR[k,r,x,A,B,Aprime,Bprime,fk,fp] +2 x r (fk fp)/f0^2  F2evR[k,r,x,A,B];




A23[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=(r^2(x^2-1))/(1+r^2-2r x) G2evN[k,r,x,A,B,Aprime,Bprime,fk,fkminusp] (fkminusp fk)/f0^2 ;


tA23[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=(r^2(x^2-1))/(1+r^2-2r x) G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp] (fkminusp fp)/f0^2 ;

a23[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=(r^2(x^2-1))/(1+r^2-2r x) G2evR[k,r,x,A,B,Aprime,Bprime,fk,fp] (fp fk)/f0^2 ;




A33[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fkminusp_]:=(r^2(1-3x^2)+2r x)/(1+r^2-2r x) G2evN[k,r,x,A,B,Aprime,Bprime,fk,fkminusp] (fkminusp fk)/f0^2 ;


tA33[k_,r_,x_,A_,B_,Aprime_,Bprime_,fp_,fkminusp_]:=(r^2(1-3x^2)+2r x)/(1+r^2-2r x) G2evQ[k,r,x,A,B,Aprime,Bprime,fp,fkminusp] (fkminusp fp)/f0^2 ;

a33[k_,r_,x_,A_,B_,Aprime_,Bprime_,fk_,fp_]:=(r^2(1-3x^2)+2r x)/(1+r^2-2r x) G2evR[k,r,x,A,B,Aprime,Bprime,fk,fp] (fp fk)/f0^2 ;




KernI1udd1[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:= A11[k,r,x,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fkminusp]*PSLk *PSLkminusp+ tA11[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,fp,fkminusp]*PSLp *PSLkminusp+a11[k,r,x,KAevR,KBevR,KAprimeEvR,KBprimeEvR,fk,fp]PSLk PSLp

KernI2uud1[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:= A12[k,r,x,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fkminusp]*PSLk *PSLkminusp+ tA12[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,fp,fkminusp]*PSLp *PSLkminusp+a12[k,r,x,KAevR,KBevR,KAprimeEvR,KBprimeEvR,fk,fp]PSLk PSLp

KernI2uud2[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:= A22[k,r,x,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fkminusp]*PSLk *PSLkminusp+ tA22[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,fp,fkminusp]*PSLp *PSLkminusp+a22[k,r,x,KAevR,KBevR,KAprimeEvR,KBprimeEvR,fk,fp]PSLk PSLp

KernI3uuu2[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:= A23[k,r,x,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fkminusp]*PSLk *PSLkminusp+ tA23[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,fp,fkminusp]*PSLp *PSLkminusp+a23[k,r,x,KAevR,KBevR,KAprimeEvR,KBprimeEvR,fk,fp]PSLk PSLp


KernI3uuu3[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:= A33[k,r,x,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fkminusp]*PSLk *PSLkminusp+ tA33[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,fp,fkminusp]*PSLp *PSLkminusp+a33[k,r,x,KAevR,KBevR,KAprimeEvR,KBprimeEvR,fk,fp]PSLk PSLp



AKernelsT[k_,r_,x_,KAevQ_,KBevQ_,KAprimeEvQ_,KBprimeEvQ_,KAevR_,KBevR_,KAprimeEvR_,KBprimeEvR_,KAevN_,KBevN_,KAprimeEvN_,KBprimeEvN_,fk_,fp_,fkminusp_,PSLk_,PSLp_,PSLkminusp_]:={KernI1udd1[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,KAevR,KBevR,KAprimeEvR,KBprimeEvR,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fp,fkminusp,PSLk,PSLp,PSLkminusp],KernI2uud1[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,KAevR,KBevR,KAprimeEvR,KBprimeEvR,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fp,fkminusp,PSLk,PSLp,PSLkminusp],KernI2uud2[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,KAevR,KBevR,KAprimeEvR,KBprimeEvR,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fp,fkminusp,PSLk,PSLp,PSLkminusp],KernI3uuu2[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,KAevR,KBevR,KAprimeEvR,KBprimeEvR,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fp,fkminusp,PSLk,PSLp,PSLkminusp],KernI3uuu3[k,r,x,KAevQ,KBevQ,KAprimeEvQ,KBprimeEvQ,KAevR,KBevR,KAprimeEvR,KBprimeEvR,KAevN,KBevN,KAprimeEvN,KBprimeEvN,fk,fp,fkminusp,PSLk,PSLp,PSLkminusp]}




test=AKernelsT[1,2,0.1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,100,100,100];
NumQs=Length@test;
Print["Integrating ",NumQs," kernels"];






(* ::Input:: *)
(**)


(* ::Section:: *)
(*Define B/C kernels (BCBCKernelsT)*)


(* ::Text:: *)
(*Bnab = \!\(\*SubsuperscriptBox[\(B\), \(ab\), \(n\)]\) are functions in TNS paper. a+b is the order of the moment to which it belongs and n shows the power of \[Mu]. *)
(*Hence, these functions become multiplied by \[Mu]^(2n) f^(a+b) . *)
(*Furthermore, to construct the integrand one has to multiply by     ((-1)^(a+b) Subscript[P, a2 ](k Sqrt[1+r^2-2r x])Subscript[P, b2 ](k r))/(1+r^2-2r x)^a where Subscript[P, 12 ]=Subscript[P, \[Delta]\[Theta] ] and  Subscript[P, 22 ]=Subscript[P, \[Theta]\[Theta] ]*)


(*Second moment Bs*)
B111[r_,x_]:=r^2 /2.(x^2-1.);
B211[r_,x_]:=r/2.(r+2.x-3.r x^2);

KI2uudd1B[k_,r_,x_]:=B111[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x);
KI2uudd2B[k_,r_,x_]:=B211[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x);

(*Third moment Bs*)
B112[r_,x_]:=3.r^2 /8.(x^2-1.)^2;
B121[r_,x_]:=3.r^4 /8.(x^2-1.)^2;
B212[r_,x_]:=-3./4 r (x^2-1.)(-r-2.x+ 5.r x^2);
B221[r_,x_]:=3.r^2/4.(x^2-1.)(-2+r^2+6.r x-5 r^2 x^2);
B312[r_,x_]:=r/8.( 4.x(3.-5.x^2)  +r(3.-30.x^2+35.x^4) );
B321[r_,x_]:=r/8.( -8.x +r(-12.+36.x^2   + 12.r x(3.-5.x^2)  +r^2(3.-30.x^2+35.x^4) )  );

KI3uuud1B[k_,r_,x_]:=-B112[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-B121[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2

KI3uuud2B[k_,r_,x_]:=-B212[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-B221[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2

KI3uuud3B[k_,r_,x_]:=-B312[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-B321[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2



(*Fourth moment Bs*)
B122[r_,x_]:=5.r^4 /16.(x^2-1.)^3;
B222[r_,x_]:=-3.r^2/16.(x^2-1.)^2(6.-30. r x-5.r^2+35.r^2x^2);
B322[r_,x_]:=3.r/16.(x^2-1.)( -8.x +r(-12.+60.x^2   +20.r x(3.-7.x^2)  +5.r^2(1.-14.x^2+21.x^4) )  );
B422[r_,x_]:=r/16.(   8.x (-3.+5.x^2)-6r(3.-30.x^2  +35x^4) + 6.r^2 x(15.-70.x^2+63.x^4 )   +   r^3(5.-  21.x^2(5.-15.x^2+11.x^4) )  );

KI4uuuu1B[k_,r_,x_]:=B122[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu2B[k_,r_,x_]:=B222[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu3B[k_,r_,x_]:=B322[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu4B[k_,r_,x_]:=B422[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;





(* ::Input:: *)
(**)


(*Second moment Cs: There are not a=b=1` Cmab because terms Pdu*Pdu does not exist for C. 
There is something similar, see below*)
C111[r_,x_]:=0;
C211[r_,x_]:=0;

KI2uudd1C[k_,r_,x_]:=1/4(1-x^2)(Pdd[k Sqrt[1+r^2-2 r x]]*Puu[k r]+r^4/(1+r^2-2 r x)^2 Puu[k Sqrt[1+r^2-2 r x]]*Pdd[k r]);
KI2uudd2C[k_,r_,x_]:=1/4(3x^2-1)Pdd[k Sqrt[1+r^2-2 r x]]*Puu[k r]+(r^2( 2-4r x+r^2(3x^2-1)))/(4(1+r^2-2 r x)^2) Puu[k Sqrt[1+r^2-2 r x]]*Pdd[k r];


(*Third moment Cs*)
C112[r_,x_]:=-B112[r,x];
C121[r_,x_]:=-B121[r,x];
C212[r_,x_]:=-1/4(1-x^2)(2-3 r^2-12 r x+15 r^2 x^2);
C221[r_,x_]:=r^2C212[r,x];
C312[r_,x_]:=-1/8(-4+3 r^2+24 r x+12 x^2-30 r^2 x^2-40 r x^3+35 r^2 x^4);
C321[r_,x_]:=r^2C312[r,x];

KI3uuud1C[k_,r_,x_]:=-C112[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-C121[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2;

KI3uuud2C[k_,r_,x_]:=-C212[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-C221[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2;

KI3uuud3C[k_,r_,x_]:=-C312[r,x]Pdu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)-C321[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Pdu[k r]/(1+r^2-2 r x)^2;


(*Fourth moment Cs*)
C122[r_,x_]:=-B122[r,x];
C222[r_,x_]:=3/16 r^2 (-1+x^2)^2 (7-5 r^2-30 r x+35 r^2 x^2);
C322[r_,x_]:=-(1/16) (-1+x^2) (4+3 r (-16 x+r (-14+70 x^2+20 r x (3-7 x^2)+5 r^2 (1-14 x^2+21 x^4))));
C422[r_,x_]:=1/16 (-4+12 x^2+16 r x (3-5 x^2)+7 r^2 (3-30 x^2+35 x^4)-6 r^3 x (15-70 x^2+63 x^4)+r^4 (-5+21 x^2 (5-15 x^2+11 x^4)));

KI4uuuu1C[k_,r_,x_]:=C122[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu2C[k_,r_,x_]:=C222[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu3C[k_,r_,x_]:=C322[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;
KI4uuuu4C[k_,r_,x_]:=C422[r,x]Puu[k Sqrt[1+r^2-2 r x]]*Puu[k r]/(1+r^2-2 r x)^2;



BCKernelsT[k_,r_,x_]:={KI2uudd1B[k,r,x],KI2uudd2B[k,r,x],KI3uuud1B[k,r,x],KI3uuud2B[k,r,x],KI3uuud3B[k,r,x],KI4uuuu1B[k,r,x],KI4uuuu2B[k,r,x],KI4uuuu3B[k,r,x],KI4uuuu4B[k,r,x],KI2uudd1C[k,r,x],KI2uudd2C[k,r,x],KI3uuud1C[k,r,x],KI3uuud2C[k,r,x],KI3uuud3C[k,r,x],KI4uuuu1C[k,r,x],KI4uuuu2C[k,r,x],KI4uuuu3C[k,r,x],KI4uuuu4C[k,r,x]};

BCKernelsT[0.1,0.1,0.2];
NumQs=Length@BCKernelsT[0.2,1,0.5];
Print["Integrating ",NumQs," kernels"];


(* ::Section:: *)
(*Combined integration*)



Needs["NumericalDifferentialEquationAnalysis`"]; (* GL angular integration *)

Clear[OutP22T, OutAT, OutBCT, OutP22TNW, OutATNW, OutBCTNW, computeOneKBothExact];

(* Keep a separate no-wiggle interpolation, but do not change the original kernel definitions.
   The wrappers below call the original P22kernelsT and BCKernelsT exactly, with PSL/Pdd/Pdu/Puu
   locally replaced by either the normal or no-wiggle spectrum.  This is intentionally a little
   less aggressive than the previous algebraic rewrite, but it should reproduce the old files. *)
PSLNW = Interpolation[PSLTNW];

Nx = 10;
kmax = PSLT[[sizepk, 1]];
kmin = PSLT[[2, 1]];
nKOut = Length @ kT;

numP22 = Length @ P22kernelsT[1, 1, 0.3, 1, 1, 0, 0, 1, 1];
numA = Length @ AKernelsT[
  1, 2, 0.1,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 1, 1,
  100, 100, 100
];
numBC = Length @ BCKernelsT[0.2, 1, 0.5];

Print["Integrating P22 kernels: ", numP22];
Print["Integrating A kernels: ", numA];
Print["Integrating B/C kernels: ", numBC];

glBase = GaussianQuadratureWeights[Nx, -1., 1.];
xGLBase = glBase[[All, 1]];
wGLBase = glBase[[All, 2]];

P22kernelsTWithPSL[psFun_, k_, r_, x_, AA_, BB_, AAp_, BBp_, fp_, fkm_] :=
  Block[{PSL = psFun},
    P22kernelsT[k, r, x, AA, BB, AAp, BBp, fp, fkm]
  ];

BCKernelsTWithPSL[psFun_, k_, r_, x_] :=
  Block[{Pdd, Pdu, Puu},
    Pdd[q_?NumericQ] := psFun[q];
    Pdu[q_?NumericQ] := fk[q]/f0 psFun[q];
    Puu[q_?NumericQ] := (fk[q]/f0)^2 psFun[q];
    BCKernelsT[k, r, x]
  ];

computeOneKBothExact[koutin_Integer] := Module[
  {ki, PSLkv, PSLkvNW, fkv, rmax, rmin,
   qP22, qA, qBC, qP22NW, qANW, qBCNW,
   qP22A, qAA, qBCA, qP22ANW, qAANW, qBCANW,
   PSLA, PSLANW, p, rr, PSLpv, PSLpvNW, fpv,
   mumin, mumax, halfRange, midPoint, xGL, wGL,
   qP22B, qAB, qBCB, qP22BNW, qABNW, qBCBNW,
   deltar, prefactor},

  ki = kT[[koutin]];
  PSLkv = PSL[ki];
  PSLkvNW = PSLNW[ki];
  fkv = fk[ki];

  rmax = kmax/ki;
  rmin = kmin/ki;

  qP22 = ConstantArray[0., numP22];
  qA   = ConstantArray[0., numA];
  qBC  = ConstantArray[0., numBC];

  qP22NW = ConstantArray[0., numP22];
  qANW   = ConstantArray[0., numA];
  qBCNW  = ConstantArray[0., numBC];

  qP22A = ConstantArray[0., numP22];
  qAA   = ConstantArray[0., numA];
  qBCA  = ConstantArray[0., numBC];

  qP22ANW = ConstantArray[0., numP22];
  qAANW   = ConstantArray[0., numA];
  qBCANW  = ConstantArray[0., numBC];

  PSLA = 0.;
  PSLANW = 0.;

  Do[
    p = pT[[pii]];
    rr = p/ki;

    PSLpv = PSL[p];
    PSLpvNW = PSLNW[p];
    fpv = fk[p];

    mumin = Max[-1.0, ((1. + rr^2 - rmax^2)/2.)/rr];
    mumax = Min[ 1.0, ((1. + rr^2 - rmin^2)/2.)/rr];
    If[rr >= 0.5, mumax = 0.5/rr];

    halfRange = (mumax - mumin)/2.;
    midPoint = (mumax + mumin)/2.;
    xGL = midPoint + halfRange xGLBase;
    wGL = halfRange wGLBase;

    {qP22B, qAB, qBCB, qP22BNW, qABNW, qBCBNW} = Total @ MapThread[
      Function[{xv, w},
        Module[{den, kminus, PSLkminuspv, PSLkminuspvNW, fkminuspv,
          KAevQ, KBevQ, KAprimeEvQ, KBprimeEvQ,
          KAevR, KBevR, KAprimeEvR, KBprimeEvR,
          KAevN, KBevN, KAprimeEvN, KBprimeEvN,
          p22kernels, p22kernelsNW, aKernels, aKernelsNW, bcKernels, bcKernelsNW},

          den = 1. + rr^2 - 2. rr xv;
          kminus = ki Sqrt[den];

          PSLkminuspv = PSL[kminus];
          PSLkminuspvNW = PSLNW[kminus];
          fkminuspv = fk[kminus];

          (* Q ordering *)
          KAevQ = Ah[ki, p, kminus];
          KBevQ = Bh[ki, p, kminus];
          KAprimeEvQ = Aprimeh[ki, p, kminus];
          KBprimeEvQ = Bprimeh[ki, p, kminus];

          (* R ordering *)
          KAevR = Ah[kminus, ki, p];
          KBevR = Bh[kminus, ki, p];
          KAprimeEvR = Aprimeh[kminus, ki, p];
          KBprimeEvR = Bprimeh[kminus, ki, p];

          (* N ordering *)
          KAevN = Ah[p, ki, kminus];
          KBevN = Bh[p, ki, kminus];
          KAprimeEvN = Aprimeh[p, ki, kminus];
          KBprimeEvN = Bprimeh[p, ki, kminus];

          p22kernels = P22kernelsTWithPSL[PSL, ki, rr, xv,
            KAevQ, KBevQ, KAprimeEvQ, KBprimeEvQ, fpv, fkminuspv];

          p22kernelsNW = P22kernelsTWithPSL[PSLNW, ki, rr, xv,
            KAevQ, KBevQ, KAprimeEvQ, KBprimeEvQ, fpv, fkminuspv];

          aKernels = AKernelsT[
            ki, rr, xv,
            KAevQ, KBevQ, KAprimeEvQ, KBprimeEvQ,
            KAevR, KBevR, KAprimeEvR, KBprimeEvR,
            KAevN, KBevN, KAprimeEvN, KBprimeEvN,
            fkv, fpv, fkminuspv,
            PSLkv, PSLpv, PSLkminuspv
          ];

          aKernelsNW = AKernelsT[
            ki, rr, xv,
            KAevQ, KBevQ, KAprimeEvQ, KBprimeEvQ,
            KAevR, KBevR, KAprimeEvR, KBprimeEvR,
            KAevN, KBevN, KAprimeEvN, KBprimeEvN,
            fkv, fpv, fkminuspv,
            PSLkvNW, PSLpvNW, PSLkminuspvNW
          ];

          bcKernels = BCKernelsTWithPSL[PSL, ki, rr, xv];
          bcKernelsNW = BCKernelsTWithPSL[PSLNW, ki, rr, xv];

          {
            w PSLkminuspv p22kernels,
            w aKernels,
            w bcKernels,
            w PSLkminuspvNW p22kernelsNW,
            w aKernelsNW,
            w bcKernelsNW
          }
        ]
      ],
      {xGL, wGL}
    ];

    deltar = (pT[[pii]] - pT[[pii - 1]])/ki;

    qP22 += ((qP22A PSLA + qP22B PSLpv)/2.) deltar;
    qA   += ((qAA   + qAB)/2.) deltar;
    qBC  += ((qBCA  + qBCB)/2.) deltar;

    qP22NW += ((qP22ANW PSLANW + qP22BNW PSLpvNW)/2.) deltar;
    qANW   += ((qAANW   + qABNW)/2.) deltar;
    qBCNW  += ((qBCANW  + qBCBNW)/2.) deltar;

    qP22A = qP22B;
    qAA   = qAB;
    qBCA  = qBCB;
    PSLA = PSLpv;

    qP22ANW = qP22BNW;
    qAANW   = qABNW;
    qBCANW  = qBCBNW;
    PSLANW = PSLpvNW;

    , {pii, 2, sizepT}
  ];

  prefactor = 2. ki^3/fourpi2;

  If[koutin/10 == Floor[koutin/10],
    Print[koutin, ": k=", ki, ",  time= ", AbsoluteTime[] - ta]
  ];

  {
    Transpose[{ConstantArray[ki, numP22], prefactor qP22}],
    Transpose[{ConstantArray[ki, numA],   prefactor qA}],
    Transpose[{ConstantArray[ki, numBC],  prefactor qBC}],
    Transpose[{ConstantArray[ki, numP22], prefactor qP22NW}],
    Transpose[{ConstantArray[ki, numA],   prefactor qANW}],
    Transpose[{ConstantArray[ki, numBC],  prefactor qBCNW}]
  }
];

ta = AbsoluteTime[];

integrationTiming = AbsoluteTiming[
  integrationResults = Table[computeOneKBothExact[koutin], {koutin, 1, nKOut}];
];

OutP22T = Transpose[integrationResults[[All, 1]], {2, 1, 3}];
OutAT   = Transpose[integrationResults[[All, 2]], {2, 1, 3}];
OutBCT  = Transpose[integrationResults[[All, 3]], {2, 1, 3}];
OutP22TNW = Transpose[integrationResults[[All, 4]], {2, 1, 3}];
OutATNW   = Transpose[integrationResults[[All, 5]], {2, 1, 3}];
OutBCTNW  = Transpose[integrationResults[[All, 6]], {2, 1, 3}];

Print["Total combined normal+NW P22/A/B/C integration time = ", integrationTiming[[1]], " seconds"];

P22MakeExportTable[out_] := Transpose[
  Join[
    {out[[1, All, 1]]},
    out[[All, All, 2]]
  ]
];

headerP22 =
  "# 1.k[h/Mpc] 2.Pdeltadelta22  3.Pthetatheta22  4.Pdeltatheta22  5.Pb2b1   6.Pbs2b1    7.Pb22   8.Pb2s2  9.Pbs22   10.Pb2theta   11.Pbs2theta";

headerA =
  "1.k   2.I1udd1  3.I2uud1  4.I2uud2  5.I3uuu2  6.I3uuu3";

headerBC =
  "# 1.k[h/Mpc] 2.I2uudd1B 3.I2uudd2B 4.I3uuud1B 5.I3uuud2B 6.I3uuud3B  7.I4uuuu1B  8.I4uuuu2B 9.I4uuuu3B 10.I4uuuu4B 11.I2uudd1C 12.I2uudd2C 13.I3uuud1C 14.I3uuud2C  15.I3uuud3C  16.I4uuuu1C  17.I4uuuu2C 18.I4uuuu3C 19.I4uuuu4C";

P22functionsT = Prepend[P22MakeExportTable[OutP22T], headerP22];
AfunctionsT = Prepend[P22MakeExportTable[OutAT], headerA];
BandCfunctionsT = Prepend[P22MakeExportTable[OutBCT], headerBC];

P22functionsTNW = Prepend[P22MakeExportTable[OutP22TNW], headerP22];
AfunctionsTNW = Prepend[P22MakeExportTable[OutATNW], headerA];
BandCfunctionsTNW = Prepend[P22MakeExportTable[OutBCTNW], headerBC];

If[TrueQ[ExportIntermediateTables],
  Export[outputdir <> "/P22functions_" <> suffix <> ".dat", P22functionsT];
  Export[outputdir <> "/Afunctions_" <> suffix <> ".dat", AfunctionsT];
  Export[outputdir <> "/BandCfunctions_" <> suffix <> ".dat", BandCfunctionsT];
  Export[outputdir <> "/P22functions_" <> suffix <> "_nw.dat", P22functionsTNW];
  Export[outputdir <> "/Afunctions_" <> suffix <> "_nw.dat", AfunctionsTNW];
  Export[outputdir <> "/BandCfunctions_" <> suffix <> "_nw.dat", BandCfunctionsTNW];
];

Print["P22/A/BandC normal and NW tables kept in memory."];
