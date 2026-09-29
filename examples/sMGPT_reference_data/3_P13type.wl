(* ::Package:: *)

(* ::Title:: *)
(* P13 functions *)


(* ::Text:: *)
(*Alejandro Aviles (avilescervantes@gmail.com)*)
(*Compute P13 functions for HS models and LCDM *)


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

Print["Running P13 functions"];
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
(*If[bLCDM\[Equal]1,fk[k_]:=f0,fk=Interpolation[fkTh]];*)(*Assuming the MG model f[k=0]=f_LCDM.Does not work in DGP*)
fk=Interpolation[fkTh]*)

PSL = Interpolation[PSLT];
PSLNW = Interpolation[PSLTNW];
sizepk = Dimensions[PSLT][[1]];
sizepkNW = Dimensions[PSLTNW][[1]];
kinMin = PSLT[[1, 1]];
kinMax = PSLT[[sizepk, 1]];
Print["normal input PS points = ", sizepk, ", kmin = ", PSLT[[1, 1]], ", kmax = ", PSLT[[sizepk, 1]]];
Print["NW input PS points = ", sizepkNW, ", kmin = ", PSLTNW[[1, 1]], ", kmax = ", PSLTNW[[sizepkNW, 1]]];
fourpi2 = 4. Pi^2;



(* ::Section:: *)
(*General functions (modify here for other models)*)


(* ::Input:: *)
(**)


(* ::Section:: *)
(*Equations*)


Clear[x,k,p]
kplusp[x_,k_,p_]:=Sqrt[k^2+p^2+2. k p x];
source2a[eta_,x_,k_,p_]:=f2[eta] mu[eta,kplusp[x,k,p]]
source2b[eta_,x_,k_,p_]:=f2[eta] (mu[eta,k]+mu[eta,p]-mu[eta,kplusp[x,k,p]])
source2FL[eta_,x_,k_,p_]:=f2[eta] (mu[eta,kplusp[x,k,p]]*(p/k+k/p)*x -k x/p mu[eta,k]-p x/k mu[eta,p])
source2dI[eta_,x_,k_,p_]:=Sc 1/6 ((OmM[eta]H[eta])/(Exp[eta]invH0))^2 (kplusp[x,k,p]^2 M2[eta])/(PiF[eta,kplusp[x,k,p]]PiF[eta,k]PiF[eta,p])

sourceD2[eta_,x_,k_,p_]:=source2a[eta,x,k,p]-source2b[eta,x,k,p]x^2+source2FL[eta,x,k,p]-Sc source2dI[eta,x,k,p]



source3I[eta_,x_,k_,p_]:=(f2[eta] (mu[eta,p]+ mu[eta,kplusp[x,k,p]]-mu[eta,k])D2f[eta]Dpp[eta]+sourceD2[eta,x,k,p]Dpk[eta]Dpp[eta]Dpp[eta]) (1.-x^2)/(1.+(p/k)^2+2 p/k x) +(f2[eta] (mu[eta,p]+ mu[eta,kplusp[-x,k,p]]-mu[eta,k])D2mf[eta]Dpp[eta]+sourceD2[eta,-x,k,p]Dpk[eta]Dpp[eta]Dpp[eta]) (1.-x^2)/(1.+(p/k)^2-2 p/k x)

source3IIplus[eta_,x_,k_,p_]:=-f2[eta](mu[eta,p]+mu[eta,kplusp[x,k,p]]-2mu[eta,k])Dpp[eta](D2f[eta]+Dpk[eta]Dpp[eta]x^2 )-f2[eta](mu[eta,kplusp[x,k,p]]-mu[eta,k])Dpk[eta]Dpp[eta]Dpp[eta]-f2[eta]((mu[eta,kplusp[x,k,p]]*(p/k+k/p)*x-k x/p mu[eta,k]-p x/k mu[eta,p]))Dpk[eta]Dpp[eta]Dpp[eta]


source3II[eta_,x_,k_,p_]:=source3IIplus[eta,x,k,p]+source3IIplus[eta,-x,k,p]




sourceFL3Plus[eta_,x_,k_,p_]:=f2[eta]( ((p^2+p k x)/(k^2+p^2+2. p k x))(mu[eta,p]-mu[eta,k])D2f[eta]Dpp[eta]+ ((p^2+p k x)/p^2)(mu[eta,kplusp[x,k,p]]-mu[eta,k])( D2f[eta]Dpp[eta]+(1+x^2)Dpk[eta]Dpp[eta]Dpp[eta] )+(((p^2+k^2) x^2)/p^2+((p^2+k^2) x)/(p k))(mu[eta,kplusp[x,k,p]]-mu[eta,k])Dpk[eta]Dpp[eta]Dpp[eta])


sourceFL3[eta_,x_,k_,p_]:=sourceFL3Plus[eta,x,k,p]+sourceFL3Plus[eta,-x,k,p]



sourceSc2plus[eta_,x_,k_,p_]:=(((OmM[eta] H[eta])/invH0)^2 (M2[eta]kplusp[x,k,p]^2 Exp[-2. eta])/(6PiF[eta,kplusp[x,k,p]]PiF[eta,k]PiF[eta,p]))Dpk[eta]Dpp[eta]Dpp[eta]+-f2[eta] M1[eta]/(3PiF[eta,k]) ((p^2+3 k p x+2 k^2 x^2)/kplusp[x,k,p]^2)(mu[eta,kplusp[x,k,p]]-1.)((2.A0[eta])/3 M2[eta]/(3PiF[eta,k]PiF[eta,p]))Dpk[eta]Dpp[eta]Dpp[eta];

sourceSc2[eta_,x_,k_,p_]:=sourceSc2plus[eta,x,k,p]+sourceSc2plus[eta,-x,k,p]


A0[eta_]:=1.5 (OmM[eta]H[eta]^2)/invH0^2;
KFL2[eta_,x_,k_,p_]:=2 x^2 (mu[eta,k]+mu[eta,p]-2.)+(p x)/k (mu[eta,k]-1.)+(k x)/p (mu[eta,p]-1.);
JFL[eta_,x_,k_,p_]:=9./(2.A0[eta]) KFL2[eta,x,k,p]PiF[eta,k]PiF[eta,p];

K3dI[eta_,x_,k_,p_]:=2((OmM[eta] H[eta])/invH0)^2 M2[eta]/(PiF[eta,k] PiF[eta,0] )+1/3. ((OmM[eta]^3 H[eta]^4)/invH0^4)(M3[eta]-(M2[eta](M2[eta]+JFL[eta,-1,p,p](3+2wBD)))/PiF[eta,0]) 1/(PiF[eta,p]^2 PiF[eta,k])+((OmM[eta] H[eta])/invH0)^2 M2[eta]/(PiF[eta,p] PiF[eta,kplusp[x,k,p]] ) (1+x^2+D2f[eta]/(Dpk[eta]Dpp[eta]))+1/3. ((OmM[eta]^3 H[eta]^4)/invH0^4)(M3[eta]-(M2[eta](M2[eta]+JFL[eta,x,k,p](3+2wBD)))/PiF[eta,kplusp[x,k,p]]) 1/(PiF[eta,p]^2 PiF[eta,k])+ ((OmM[eta] H[eta])/invH0)^2 M2[eta]/(PiF[eta,p] PiF[eta,kplusp[-x,k,p]] ) (1+x^2+D2mf[eta]/(Dpk[eta]Dpp[eta]))+1/3. ((OmM[eta]^3 H[eta]^4)/invH0^4)(M3[eta]-(M2[eta](M2[eta]+JFL[eta,-x,k,p](3+2wBD)))/PiF[eta,kplusp[-x,k,p]]) 1/(PiF[eta,p]^2 PiF[eta,k])


sourcedI3[eta_,x_,k_,p_]:=-Sc k^2/Exp[2.eta] 1./(6PiF[eta,k]) K3dI[eta,x,k,p]Dpk[eta]Dpp[eta]Dpp[eta];




(* ::Section:: *)
(*Diff equations*)


etainI=-4;
etafI=0.1;
etaini=etainI;
etafinal=0;

Dplusi=Exp[etaini];
dDplusi=Exp[etaini];

D2plusi=3.Exp[2etaini]/7.(1.-x^2);
dD2plusi=6.Exp[2etaini]/7.(1.-x^2);
D3symmi=5./7. Exp[3.etaini]/9.((1.-x^2)/(1.+(p/k)^2+2 p/k x)+(1.-x^2)/(1.+(p/k)^2-2 p/k x))(1.-x^2);
dD3symmi=15./7. Exp[3etaini]/9.((1.-x^2)/(1.+(p/k)^2+2 p/k x)+(1.-x^2)/(1.+(p/k)^2-2 p/k x))(1.-x^2);


eqD3symm=D3symmf''[eta] +f1[eta]D3symmf'[eta]-f2[eta] mu[eta,k]D3symmf[eta]==source3I[eta,x,k,p]+source3II[eta,x,k,p]+sourceFL3[eta,x,k,p]+ sourcedI3[eta,x,k,p]+ sourceSc2[eta,x,k,p];
eqD2p=D2f''[eta] +f1[eta]D2f'[eta]-f2[eta] mu[eta,kplusp[x,k,p]]D2f[eta]==sourceD2[eta,x,k,p]Dpk[eta]Dpp[eta];
eqD2m=D2mf''[eta] +f1[eta]D2mf'[eta]-f2[eta] mu[eta,kplusp[-x,k,p]]D2mf[eta]==sourceD2[eta,-x,k,p]Dpk[eta]Dpp[eta];
eqDpk=Dpk''[eta] +f1[eta]Dpk'[eta]-f2[eta] mu[eta,k]Dpk[eta]==0;
eqDpp=Dpp''[eta] +f1[eta]Dpp'[eta]-f2[eta] mu[eta,p]Dpp[eta]==0;

sysD3symm=ParametricNDSolve[{eqD3symm,
eqD2p,eqD2m,eqDpk,eqDpp,D3symmf'[etaini]==dD3symmi,D3symmf[etaini]==D3symmi,D2f'[etaini]==dD2plusi,D2f[etaini]==D2plusi,D2mf'[etaini]==dD2plusi,D2mf[etaini]==D2plusi,Dpk'[etaini]==Exp[etaini],Dpk[etaini]==Exp[etaini],Dpp'[etaini]==Exp[etaini],Dpp[etaini]==Exp[etaini]},{D3symmf,D2f,D2mf,Dpk,Dpp},{eta,etainI,etafinal},{x,k,p}]

D3symm:=Evaluate[D3symmf/.sysD3symm]
D2:=Evaluate[D2f/.sysD3symm]
D2m:=Evaluate[D2mf/.sysD3symm]
xpreDk:=Evaluate[Dpk/.sysD3symm]
xpreDp:=Evaluate[Dpp/.sysD3symm]
xDk[k_]:=xpreDk[0.001,k,0.001]
xDp[p_]:=xpreDp[0.001,0.001,p]

D3[x_,k_,p_]:=D3symm[x,k,p][etaev]
D3prime[x_,k_,p_]:=D[D3symm[x,k,p][eta],eta]/.eta->etaev

AminusBx2[x_,k_,p_]:=D2[x,k,p][etaev]/(3/7 xDk[k][etaev]xDp[p][etaev])
AminusBx2Prime[x_,k_,p_]:= D[D2[x,k,p][eta]/(3/7 xDk[k][eta]xDp[p][eta]),eta]/.eta->etaev

(*NOTE THIS gamma3 and gamma3f are defined differently than the other NOTEBOOKS,there is an additional 21/5*)C3gamma3[x_,k_,p_]:=D3symm[x,k,p][etaev]/(xDk[k][etaev] xDp[p][etaev] xDp[p][etaev]);
C3gamma3f[x_,k_,p_]:=D3prime[x,k,p]/(xDk[k][etaev] xDp[p][etaev] xDp[p][etaev]) 1/(3 f0);


lcdmCF=C3gamma3[0.,0.00001,0.00001] 21/5
lcdmCFprime=C3gamma3f[0.,0.00001,0.00001] 21/5

AprimeLCDM=AminusBx2Prime[0,0.0001,0.0001] (*Large scales value.For f(R) this is LCDM*)
ALCDM=AminusBx2[0,0.0001,0.0001]  (*Large scales value.For f(R) this is LCDM*)





(* ::Input:: *)
(**)


(* ::Chapter:: *)
(*Tables of k and p points*)


(* ::Input:: *)
(*Get[FileNameJoin[{SourceDirectory, "tables.m"}]];*)


(* ::Chapter:: *)
(*INTEGRATE*)


(* ::Input:: *)
(**)


(* ::Section:: *)
(*Define Kernels*)


If[PTkernels==1||PTkernels==2,bLCDMkernels=0,bLCDMkernels=1];
If[PTkernels==3||PTkernels==5,Clear[fk];fk[k_]:=f0];
If[PTkernels==5||PTkernels==6,ALCDM=1;AprimeLCDM=0];
If[PTkernels==5||PTkernels==6,lcdmCF =1;lcdmCFprime=1];

If[bLCDMkernels==0,AminusBx2h[x_,k_,p_]:=AminusBx2[x,k,p],AminusBx2h[x_,k_,p_]:=ALCDM (1-x^2)];
If[bLCDMkernels==0,AminusBx2Primeh[x_,k_,p_]:=AminusBx2Prime[x,k,p],AminusBx2Primeh[x_,k_,p_]:=AprimeLCDM (1.-x^2)];

kmpdk[x_,k_,p_]:=(1-x^2)^2/(1+(p/k)^2-2. p/k x);
(*C3gamma3LCDM[x_,k_,p_]  := 5/21lcdmCF             (kmpdk[x,k,p]+kmpdk[-x,k,p]) ;
C3gamma3fLCDM[x_,k_,p_]:= 5/21lcdmCFprime( kmpdk[x,k,p]+kmpdk[-x,k,p])  ;*) 
C3gamma3LCDM[x_,k_,p_]  := 2.*5/21lcdmCF             (kmpdk[x,k,p]) ;
C3gamma3fLCDM[x_,k_,p_]:= 2.*5/21lcdmCFprime( kmpdk[x,k,p])  ; 

If[bLCDMkernels==0,C3gamma3h[x_,k_,p_]:=C3gamma3[x,k,p],C3gamma3h[x_,k_,p_]:=C3gamma3LCDM[x,k,p]]
If[bLCDMkernels==0,C3gamma3fh[x_,k_,p_]:=C3gamma3f[x,k,p],C3gamma3fh[x_,k_,p_]:=C3gamma3fLCDM[x,k,p]]



C2=3/7;
F3K[k_,p_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:=1/6 C3gamma3+1/3 C2 ((k^2-k p x)k p x)/(p^2 (k^2+p^2-2p k x)) gamma2evR -1/6 k^2/p^2 x^2;

G3K[k_,p_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:=1/2 C3gamma3f+2/3 C2 (k x)/p gamma2fevR+1/3 C2 fk/f0 (k^2-k p x)/(k^2+p^2-2p k x) gamma2evR-1/6 k^2/p^2 x^2 fk/f0-1/3 C2(2gamma2fevR+gamma2evR fp/f0)(1-(k x-p)^2/( k^2+p^2-2p k x));



KP13dd[k_,r_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:=6 r^2F3K[k,k r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0];
KP13tt[k_,r_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:=6 r^2G3K[k,k r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0] fk/f0;
KP13dt[k_,r_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:=3 r^2G3K[k,k r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0]+3r^2F3K[k,k r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0] fk/f0;

sigma32PSLK[k_,p_,x_]:=105/16(  ((x-p/k )^2/(1+(p/k)^2-2. p/k  x)-1/3)(2/7 (x^2-1/3)-4/21 )+8/63);
(*sigma32PSLK[k_,p_,x_]:=15/8  (x^2-1/3)((x-p/k )^2/(1+(p/k)^2-2. p/k  x)-1 )+5/6;*)
Ksigma32PSL[k_,r_,x_]:=r^2sigma32PSLK[k,k r,x];



RkernelsT[k_,r_,x_,C3gamma3_,C3gamma3f_,gamma2evR_,gamma2fevR_,fk_,fp_,fkminusp_,f0_]:={KP13dd[k,r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0],KP13dt[k,r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0],KP13tt[k,r,x,C3gamma3,C3gamma3f,gamma2evR,gamma2fevR,fk,fp,fkminusp,f0],
Ksigma32PSL[k,r,x]};

khv=0.1;rhv=2;xhv=1;
C3gamma3hv=C3gamma3[xhv,khv,khv rhv];
C3gamma3fhv=C3gamma3f[xhv,khv,khv rhv];
gamma2evRhv=1;gamma2fevRhv=1;
fkhv=fk[khv];fphv=fk[khv rhv];fkminusphv=f0;


testT=RkernelsT[khv,rhv,xhv,C3gamma3hv,C3gamma3fhv,gamma2evRhv,gamma2fevRhv,fkhv,fphv,fkminusphv,f0]
NumRs=Dimensions[testT][[1]];
Print["Integrating ",NumRs," kernels"];



(* ::Section:: *)
(*Integrate*)


(* ::Input:: *)
(**)



np = 1;
Needs["NumericalDifferentialEquationAnalysis`"]; (*This is for GL angular integration*)
Nx = 10;
tGL = GaussianQuadratureWeights[Nx, -1, 1];
xGL = Table[tGL[[ii, 1]], {ii, 1, Nx}];
wGL = Table[tGL[[ii, 2]], {ii, 1, Nx}];

kmax = PSLT[[sizepk, 1]];
kmin = PSLT[[2, 1]];
f0v = f0;

ta = AbsoluteTime[];

Clear[computeP13OneK, makeP13Table];

computeP13OneK[koutin_Integer] := Module[
  {ki, fkv, Rp, RpNW, RfunctionsB, RfunctionsBNW, RfunctionsA, RfunctionsANW,
   p, rr, fpv, xv, w, KAminusBx2, KAminusBx2Prime, fkminuspv,
   gamma2evRv, gamma2fevRv, C3gamma3v, C3gamma3fv, RKT, psl, pslNW, deltar},

  ki = kT[[np*koutin]];
  fkv = fk[ki];

  Rp = ConstantArray[0., NumRs];
  RpNW = ConstantArray[0., NumRs];
  RfunctionsB = ConstantArray[0., NumRs];
  RfunctionsBNW = ConstantArray[0., NumRs];
  RfunctionsA = ConstantArray[0., NumRs];
  RfunctionsANW = ConstantArray[0., NumRs];

  Do[
    p = pT[[pii]];
    rr = p/ki;
    fpv = fk[ki rr];

    Do[
      xv = xGL[[xii]];
      w = wGL[[xii]];

      KAminusBx2 = AminusBx2h[-xv, ki, ki rr];
      KAminusBx2Prime = AminusBx2Primeh[-xv, ki, ki rr];

      fkminuspv = fk[ki Sqrt[1 + rr^2 - 2 rr xv]];

      gamma2evRv = KAminusBx2;
      gamma2fevRv = gamma2evRv (fkminuspv + fpv)/(2 f0) + 1/(2 f0) KAminusBx2Prime;

      C3gamma3v = C3gamma3h[xv, ki, ki rr];
      C3gamma3fv = C3gamma3fh[xv, ki, ki rr];

      RKT = RkernelsT[ki, rr, xv, C3gamma3v, C3gamma3fv, gamma2evRv, gamma2fevRv, fkv, fpv, fkminuspv, f0v];
      psl = PSL[ki rr];
      pslNW = PSLNW[ki rr];

      RfunctionsB += w psl RKT;
      RfunctionsBNW += w pslNW RKT;

      , {xii, 1, Nx}
    ];

    deltar = (pT[[pii]] - pT[[pii - 1]])/ki;

    Rp += (RfunctionsA + RfunctionsB)/2. deltar;
    RpNW += (RfunctionsANW + RfunctionsBNW)/2. deltar;

    RfunctionsA = RfunctionsB;
    RfunctionsANW = RfunctionsBNW;
    RfunctionsB = ConstantArray[0., NumRs];
    RfunctionsBNW = ConstantArray[0., NumRs];

    , {pii, 2, sizepT}
  ];

  Rp = ki^3 PSL[ki]/fourpi2 Rp;
  RpNW = ki^3 PSLNW[ki]/fourpi2 RpNW;

  If[koutin/10 == Floor[koutin/10], Print[koutin, ": k=", ki, ",  time= ", AbsoluteTime[] - ta]];

  {
    Transpose[{ConstantArray[ki, NumRs], Rp}],
    Transpose[{ConstantArray[ki, NumRs], RpNW}]
  }
];

p13Timing = AbsoluteTiming[
  p13Results = Table[computeP13OneK[koutin], {koutin, 1, sizekT/np}];
];

OutP13T = Transpose[p13Results[[All, 1]], {2, 1, 3}];
OutP13TNW = Transpose[p13Results[[All, 2]], {2, 1, 3}];

makeP13Table[out_] := Module[{ko, p13dd, p13dt, p13tt, sigma32pk, pre},
  ko = out[[1, All, 1]];
  p13dd = out[[1, All, 2]];
  p13dt = out[[2, All, 2]];
  p13tt = out[[3, All, 2]];
  sigma32pk = out[[4, All, 2]];
  pre = Table[{ko[[ii]], p13dd[[ii]], p13dt[[ii]], p13tt[[ii]], sigma32pk[[ii]]}, {ii, 1, Length@ko}];
  Prepend[pre, "# 1.k[h/Mpc]  2.P13dd  3.P13dt  4.P13tt  5.sigma32pk"]
];

P13functionsT = makeP13Table[OutP13T];
P13functionsTNW = makeP13Table[OutP13TNW];

If[TrueQ[ExportIntermediateTables],
  Export[outputdir <> "/P13functions_" <> suffix <> ".dat", P13functionsT];
  Export[outputdir <> "/P13functions_" <> suffix <> "_nw.dat", P13functionsTNW];
];

Print["Total P13 normal+NW integration time = ", p13Timing[[1]], " seconds"];
Print["P13 normal and NW tables kept in memory."];
Print[" "];
