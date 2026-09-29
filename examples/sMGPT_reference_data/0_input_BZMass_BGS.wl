(* ::Package:: *)

If[! ValueQ[step], step = 0];

(*PT Models*)
(* 1. PTkernels=1  (FullKernels) *)
(* 2. PTkernels=2  (MG no Screenings) *)
(* 3. PTkernels=3  (ALS kernels) *)
(* 4. PTkernels=4  (fk-kernels *)
(* 5. PTkernels=5  (EdS kernels) *)
(* 6. PTkernels=6  (EdS kernels but keeping f(k) *)

PTkernels=2;
tracer="BGS";

(* This model is fkptjax's own HDKI/BZ_Mass parameterization
   (arXiv:2208.10508), NOT one of sMGPT's built-in models.  Written here so
   sMGPT can generate an independent, screening-free (Sc=0) reference for a
   model fkptjax has already implemented and validated internally
   (src/fkptjax/mg_jax.py::_mu_bz_mass), giving a fair, apples-to-apples
   cross-check of fkptjax's fkpt_approximation=False (full, non-squeezed)
   kernels without needing to port sMGPT's own bespoke HDKIa model. *)
ModelName = "BZMass";
presuffix="BZMass" (*suffix for output files*)


h=0.6736;
om=0.31519; (* Omega_m*)



tracers={"BGS","LRG1","LRG2","LRG3","ELG","QSO"};
sigma8s={0.6907,0.6183,0.5613,0.5076,0.4289,0.3936};
zs={0.295,0.5096,0.7058,0.9228,1.3235,1.55};
b1p={1.12,1.2,1.124,1.128,0.625,0.990};

b1list=b1p/sigma8s;

index=Position[tracers,tracer][[1,1]];

zev=zs[[index]];
b1=b1list[[index]];
b2=8/21*(b1-1);
bs=-4/7*(b1-1);
b3=32/315*(b1-1);

suffix=tracer;

Print["tracer= ",tracer]
Print["z_ev= ",zev]
Print["b1 = ",b1]


inputpk="./input/Abacus_pklin_z0.dat"
InputpkIsLCDM = True;
zinputpk=0;


presuffix = tracer <>"_"<>presuffix;
Print["presuffix = ",presuffix]

(* --- HDKI/BZ_Mass, matching fkptjax's mg_jax._mu_bz_mass exactly ---
   mu(a,k) = (1 + mu_kinf * X) / (1 + X)
   X       = ( k * lambda_a * lambda_dS / (D(a) * a) )^2
   D(a)    = lambda_dS * a^-3 + lambda_a
   Sigma == 1 identically for this model (no lensing sector), and Sc=0
   (PTkernels=2) turns off all M2/M3 screening-source terms below, so mass[],
   M1[] are never actually used -- kept only to mirror sMGPT's expected
   input-deck shape. *)
mu_kinf = 1.2;
lambda_a = 100.0;
lambda_dS = 100.0;

(* eta = log(a)*)
a[eta_]      := Exp[eta];
Dfun[eta_]   := lambda_dS * a[eta]^(-3) + lambda_a;
mu[eta_,k_]  := (1 + mu_kinf * (k * lambda_a * lambda_dS / (Dfun[eta] a[eta]))^2) /
                (1 + (k * lambda_a * lambda_dS / (Dfun[eta] a[eta]))^2);
mass[eta_]   := 1.0;  (* unused: M2=M3=0 below, so mass/M1 never enter Sc=0 sources *)

(* Background evolution*)
OmM[eta_]:=1/(1+(1-om)/om Exp[3eta]);
H[eta_]:=Sqrt[om Exp[-3eta]+(1-om)]; (*H over H0*)
invH0 =2997.92458 ;(* H_0^-1 in Mpc/h units *)


(* Linear growth equation is :
   D''[eta] + f1[eta] D'[eta] - f2[eta] mu[eta, kf] D[eta] = 0 *)
f1[eta_]:= (2.-3./(2.(1.+(1-om)/om Exp[3.eta])));  (* 2 + H'[eta]/H[eta] *)
f2[eta_]:= 3./(2.(1.+(1-om)/om Exp[3.eta]));    (* 3/2 Omega_m*)


PiF[eta_,k_]:=k^2/Exp[2. eta]+ mass[eta]^2;
sourcedI[kf_,k1_,k2_,eta_]:=1/6 ((OmM[eta] H[eta])/(Exp[eta] invH0))^2 (kf^2 M2[eta])/( (kf^2 Exp[-2. eta]+mass[eta]^2) (k1^2 Exp[-2. eta]+mass[eta]^2) (k2^2 Exp[-2. eta]+mass[eta]^2) )
sourceA[kf_,k1_,k2_,eta_]:= f2[eta] (mu[eta,kf]+(mu[eta,kf]-mu[eta,k1]) (kf^2-k1^2-k2^2)/(2(k2^2) )+(mu[eta,kf]-mu[eta,k2]) (kf^2-k1^2-k2^2)/(2(k1^2) ) )-Sc*sourcedI[kf,k1,k2,eta];
sourceb[kf_,k1_,k2_,eta_]:=f2[eta] (mu[eta,k1]+mu[eta,k2]-mu[eta,kf]);



M1[eta_]:=3*mass[eta]^2;

M2[eta_]:=0;
M3[eta_]:=0;
(* Modify until here to change the model *)



wBD = 0;


(* User-facing run controls.  These are read by 0_RunAll.wl. *)
RunPKExtrapolation = True;
PKExtrapolateKMin = 0.00001;
PKExtrapolateKMax = 500;
PKExtrapolateN = 888;
RunPnwDecomposition = True;
RunIRResummation = True;
ExportIntermediateTables = False;

(* Output k grid and inner p grid. *)
OutputLogInk = 1;
koutmin = 0.001;
koutmax = 1;
pperdecadeout = 40;
pperdecade = 26;
(* If OutputLogInk = 0, these are used instead: *)
KMIN = 0.001;
KMAX = 0.3;
NK = 80;



  If[!ValueQ[IRb1], IRb1 = b1];
  If[!ValueQ[IRb2], IRb2 = b2];
  If[!ValueQ[IRbs2], IRbs2 = bs];
  If[!ValueQ[IRb3nl], IRb3nl = b3];
  If[!ValueQ[IRalpha0], IRalpha0 = 0];
  If[!ValueQ[IRalpha2], IRalpha2 = 0];
  If[!ValueQ[IRalpha4], IRalpha4 = 0];
  If[!ValueQ[IRctilde], IRctilde = 0];
  If[!ValueQ[IRalphashot0], IRalphashot0 = 1];
  If[!ValueQ[IRalphashot2], IRalphashot2 = 0];
  If[!ValueQ[IRPshotP], IRPshotP = 1/0.0002];



If[PTkernels==1,suffix=presuffix<>"_FullMG"];
If[PTkernels==2,suffix=presuffix<>"_NoScreen"];
If[PTkernels==3,suffix=presuffix<>"_LCDM"];
If[PTkernels==4,suffix=presuffix<>"_fkkernels"];
If[PTkernels==5,suffix=presuffix<>"_EdS"];
If[PTkernels==6,suffix=presuffix<>"_EdSfk"];

If[step==0 && PTkernels==1,Print["PTkernels=1: Using full MG kernels"]];
If[step==0 && PTkernels==2,Print["PTkernels=2: Using MG kernels without screenings"]];
If[step==0 && PTkernels==3,Print["PTkernels=3: Using ALS kernels"]];
If[step==0 && PTkernels==4,Print["PTkernels=4: Using fk-kernels"]];
If[step==0 && PTkernels==5,Print["PTkernels=5: Using EdS kernels"]];
If[step==0 && PTkernels==6,Print["PTkernels=6: Using EdS kernels but keeping f(k)"]];






etaev=Log[1/(1.+zev)];
If[step==0,Print["suffix = ", suffix]];
If[step==0,Print["OmegaM0 = ",om,", h = ", h]];
If[step==0,Print["redshift z = ",zev," , eta = ", etaev]];

(*with screening Sc= 1, no screening Sc= 0*)
If[PTkernels==1,Sc=1;If[step==0,Print["Sc=1: Screenings on"]]];
If[PTkernels==2||PTkernels==3||PTkernels==4||PTkernels==5||PTkernels==6,Sc=0;If[step==0,Print["Sc=0: Screenings off"]]];
