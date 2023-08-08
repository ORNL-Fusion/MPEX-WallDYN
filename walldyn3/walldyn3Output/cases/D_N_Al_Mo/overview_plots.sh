#!/usr/bin/env bash

script="$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/GITR_Coupling/plotXYZTris.py"

if [ ! -f $script ]; then
  echo "Script: $script not found"
fi

trigen="../data_o_al/gitrGeometryPointPlane3d.cfg"
idxgrp="../data_o_al/surface/surface_inds_.txt"

$script --valfp "./results/ProtoEmpex_ppext_NetAdensChange_100.000.dat" --valcfg 1 1 --vallabel "Al-Net. ADens. Chg." --valuerange "-0.1 0.1" --trigen $trigen --idxgrp $idxgrp &
$script --valfp "./results/ProtoEmpex_ppext_NetAdensChange_100.000.dat" --valcfg 3 1 --vallabel "N-Net. ADens. Chg." --valuerange "-0.1 0.1" --trigen $trigen --idxgrp $idxgrp &

exit 0

$script --valfp "./results/ProtoEmpex_constantsGammaConst.dat" --valcfg 0 1 --vallabel "WallIDX" --norm "lin" --trigen $trigen --idxgrp $idxgrp &

$script --valfp "./results/ProtoEmpex_Conc_100.000.dat" --valcfg 3 1 --vallabel "N-Conc 100sec." --norm "lin" --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &
$script --valfp "./results/ProtoEmpex_Conc_0.000.dat" --valcfg 3 1 --vallabel "N-Conc 0sec." --norm "lin" --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &

$script --valfp "./results/ProtoEmpex_Conc_100.000.dat" --valcfg 1 1 --vallabel "Al-Conc 100sec." --norm "lin" --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &

$script --valfp "./results/ProtoEmpex_Gamma_100.000.dat" --valcfg 1 1 --vallabel "Al-Influx 100sec." --norm "log" --trigen $trigen --idxgrp $idxgrp &





$script --valfp "./results/ProtoEmpex_Conc_100.000.dat" --valcfg 1 1 --vallabel "Al-Conc 100sec." --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &
$script --valfp "./results/ProtoEmpex_Conc_0.000.dat" --valcfg 1 1 --vallabel "Al-Conc 0sec." --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &

$script --valfp "./results/ProtoEmpex_Conc_100.000.dat" --valcfg 2 1 --vallabel "Mo-Conc 100sec." --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &
$script --valfp "./results/ProtoEmpex_Conc_0.000.dat" --valcfg 2 1 --vallabel "Mo-Conc 0sec." --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &


$script --valfp "./results/ProtoEmpex_constantsGammaConst.dat" --valcfg 1 1 --vallabel "D-Flux" --norm "log" --trigen $trigen --idxgrp $idxgrp &
$script --valfp "./results/ProtoEmpex_constantswall_Te.dat" --valcfg 1 1 --vallabel "Te" --trigen $trigen --idxgrp $idxgrp &