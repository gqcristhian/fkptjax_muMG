(* Refresh the independent Wolfram reference without modifying archived outputs. *)
repo = DirectoryName[DirectoryName[ExpandFileName[$InputFileName]]];
src = DirectoryName[repo];
ref = FileNameJoin[{repo, "examples", "sMGPT_reference_data"}];
out = FileNameJoin[{repo, "validation_results", "bzmass_20260928"}];
Get[FileNameJoin[{src, "sMGPT", "src", "IREFTMultipoles.m"}]];
b1val = 1.12/0.6907;
pars = <|"b1" -> b1val, "b2" -> 8/21 (b1val-1), "bs2" -> -4/7 (b1val-1),
         "b3nl" -> 32/315 (b1val-1), "alpha0" -> 0., "alpha2" -> 0.,
         "alpha4" -> 0., "ctilde" -> 0., "alphashot0" -> 1.,
         "alphashot2" -> 0., "PshotP" -> 5000.|>;
result = IREFTMultipoles`IRResummedMultipoles[
  FileNameJoin[{ref, "AllFunctions_BGS_BZMass_NoScreen.dat"}],
  FileNameJoin[{ref, "AllFunctions_BGS_BZMass_NoScreen_nw.dat"}], pars,
  "Ells" -> {0,2,4}, "Nx" -> 16, "KOsc" -> 1/104., "KS" -> 0.4,
  "KMinIR" -> 0.0001];
If[result === $Failed, Exit[1]];
Export[FileNameJoin[{out, "multipoles_sMGPT_correct_columns.dat"}], result["Table"], "Table"];
Print["Exported fresh sMGPT multipoles with the actual 44-column layout."];
Get[FileNameJoin[{ref, "AB_kcompare.wl"}]];
ab = Table[With[{kf=kGrid[[ii]]},
  Prepend[solveABfull[kf,r*kf,kf*Sqrt[1+r^2-2*r*x]],kf]],{ii,Length[kGrid]}];
Export[FileNameJoin[{out,"AB_fresh.dat"}],ab,"Table"];
Print["Fresh BZ_Mass A/B reference complete."];
