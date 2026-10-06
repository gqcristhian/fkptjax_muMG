(* ::Package:: *)

(* ::Title:: *)
(*Obtain linear theory for normal and no-wiggle spectra in one step*)

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


Print[" "];
Print["Running linear theory for normal and NW spectra"];
If[NumberQ[bQdef], None, Get[FileNameJoin[{SourceDirectory, "folders.m"}]]];

If[!ValueQ[inputpknw] || !StringQ[inputpknw] || !FileExistsQ[inputpknw],
  If[FileExistsQ[FileNameJoin[{SourceDirectory, "PnwDecomposition.wl"}]],
    Get[FileNameJoin[{SourceDirectory, "PnwDecomposition.wl"}]];
    resultnw = PnwDecomposition`PnwDecompose[inputpk, h, False];
    If[resultnw === $Failed, Print["ERROR: PnwDecompose failed."]; Abort[]];
    inputpknw = resultnw["NoWiggleFile"];,
    Print["ERROR: inputpknw is not defined and src/PnwDecomposition.wl was not found."];
    Abort[];
  ];
];

InputIsLCDM = InputpkIsLCDM;
zinput = zinputpk;
etainput = Log[1/(1. + zinput)];
etaini = -6;
Dplusi = Exp[etaini];
dDplusi = Exp[etaini];

Clear[computeGrowthTables, makeLinearPkTable, computeLinearPair];

computeGrowthTables[kGrid_List] := Module[{growthRows, Dplus, Dplusp, DplusInput, DpluspInput},
  growthRows = Table[
    Module[{kf = kGrid[[ii]], sysDplus, indoDplus, indoDplusp},
      sysDplus = NDSolve[
        {Df''[eta] + f1[eta] Df'[eta] - f2[eta] mu[eta, kf] Df[eta] == 0,
         Df'[etaini] == dDplusi, Df[etaini] == Dplusi},
        Df, {eta, etaini, etainput}
      ];
      indoDplus[et_] := Evaluate[Df[et] /. sysDplus][[1]];
      indoDplusp[et_] := Evaluate[Df'[et] /. sysDplus][[1]];
      {kf, indoDplus[etaev], indoDplusp[etaev], indoDplus[etainput], indoDplusp[etainput]}
    ],
    {ii, 1, Length@kGrid}
  ];

  Dplus = Interpolation[growthRows[[All, {1, 2}]]];
  Dplusp = Interpolation[growthRows[[All, {1, 3}]]];
  DplusInput = Interpolation[growthRows[[All, {1, 4}]]];
  DpluspInput = Interpolation[growthRows[[All, {1, 5}]]];

  <|
    "Rows" -> growthRows,
    "Dplus" -> Dplus,
    "Dplusp" -> Dplusp,
    "DplusInput" -> DplusInput,
    "DpluspInput" -> DpluspInput,
    "DplusT" -> growthRows[[All, {1, 2}]],
    "DpluspT" -> growthRows[[All, {1, 3}]],
    "DplusTInput" -> growthRows[[All, {1, 4}]],
    "DpluspTInput" -> growthRows[[All, {1, 5}]]
  |>
];

makeLinearPkTable[inPkT_List, growth_Association] := Module[
  {kGrid, inPk, Dplus, DplusInput, pkMG, pkLCDM},
  kGrid = inPkT[[All, 1]];
  inPk = Interpolation[inPkT];
  Dplus = growth["Dplus"];
  DplusInput = growth["DplusInput"];

  pkMG = Table[
    {kGrid[[ii]], (Dplus[kGrid[[ii]]]/DplusInput[kGrid[[ii]]])^2 inPk[kGrid[[ii]]]},
    {ii, 1, Length@kGrid}
  ];
  pkLCDM = Table[
    {kGrid[[ii]], (Dplus[kGrid[[ii]]]/DplusInput[kGrid[[2]]])^2 inPk[kGrid[[ii]]]},
    {ii, 1, Length@kGrid}
  ];

  If[InputIsLCDM, pkLCDM, pkMG]
];

computeLinearPair[inputFile_String, inputFileNW_String] := Module[
  {inPkT, inPkTNW, kGrid, kGridNW, sameGrid, growth, growthNW,
   prefkTout, f0out, f0Tout, fkTout, pkTout, pkToutNW},

  inPkT = Import[inputFile];
  inPkTNW = Import[inputFileNW];
  kGrid = inPkT[[All, 1]];
  kGridNW = inPkTNW[[All, 1]];
  sameGrid = (kGrid == kGridNW);

  If[InputIsLCDM,
    Print["Input file ", inputFile, " is a LCDM pk at zinput = ", zinput],
    Print["Input file ", inputFile, " is a MG pk at zinput = ", zinput]
  ];
  Print["Normal input k points = ", Length@inPkT, ", kmin = ", First@kGrid, ", kmax = ", Last@kGrid];
  Print["NW input k points = ", Length@inPkTNW, ", kmin = ", First@kGridNW, ", kmax = ", Last@kGridNW];

  growth = computeGrowthTables[kGrid];
  growthNW = If[sameGrid, growth, computeGrowthTables[kGridNW]];

  prefkTout = Table[{kGrid[[ii]], growth["Dplusp"][kGrid[[ii]]]/growth["Dplus"][kGrid[[ii]]]}, {ii, 1, Length@kGrid}];
  f0out = prefkTout[[2, 2]];
  f0Tout = Table[{kGrid[[ii]], f0out}, {ii, 1, Length@kGrid}];
  If[PTkernels == 3 || PTkernels == 5, fkTout = f0Tout, fkTout = prefkTout];

  pkTout = makeLinearPkTable[inPkT, growth];
  pkToutNW = makeLinearPkTable[inPkTNW, growthNW];

  <|
    "PkTout" -> pkTout,
    "PkToutNW" -> pkToutNW,
    "fkTout" -> fkTout,
    "DplusT" -> growth["DplusT"],
    "DpluspT" -> growth["DpluspT"],
    "DplusTInput" -> growth["DplusTInput"],
    "DpluspTInput" -> growth["DpluspTInput"]
  |>
];

linearResult = computeLinearPair[inputpk, inputpknw];

PkTout = linearResult["PkTout"];
PkToutNW = linearResult["PkToutNW"];
fkTout = linearResult["fkTout"];
DplusT = linearResult["DplusT"];
DpluspT = linearResult["DpluspT"];
DplusTInput = linearResult["DplusTInput"];
DpluspTInput = linearResult["DpluspTInput"];

PSLT = PkTout;
PSLTNW = PkToutNW;
fkTh = fkTout;
f0 = fkTh[[2, 2]];
fk = Interpolation[fkTh];

If[TrueQ[ExportIntermediateTables],
  outputdirh = If[StringQ[outputdir], outputdir, "."];
  Export[outputdirh <> "/PSL_" <> suffix <> ".dat", PkTout];
  Export[outputdirh <> "/PSL_" <> suffix <> "_nw.dat", PkToutNW];
  Export[outputdirh <> "/fk_" <> suffix <> ".dat", fkTout];
  Export[outputdirh <> "/Dplus_" <> suffix <> ".dat", DplusT];
];

Print["f(k=0)=", f0];
Print["Linear normal and NW PSL tables kept in memory."];
