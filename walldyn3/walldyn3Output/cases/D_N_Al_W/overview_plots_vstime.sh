#!/usr/bin/env bash

script="../../plotXYZTris_vs_time.py"

if [ ! -f $script ]; then
  echo "Script: $script not found"
fi

trigen="../data_o_al/gitrGeometryPointPlane3d.cfg"
idxgrp="../data_o_al/surface/surface_inds_.txt"
solvstime="./results/ProtoEmpex_sol_states.nc"
postvstime="./results/ProtoEmpex_post_states.nc"


# $script --statencfp $solvstime --valtoplot "Conc N" --vallabel "N-Concentration" --norm "lin" --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &
# $script --statencfp $solvstime --valtoplot "Conc Al" --vallabel "Al-Concentration" --norm "lin" --valuerange "0 1" --trigen $trigen --idxgrp $idxgrp &
$script --statencfp $solvstime --valtoplot "Gamma Al" --vallabel "Al-Influx" --norm "log"  --trigen $trigen --idxgrp $idxgrp &

# $script --postncfp $postvstime --valtoplot "TotalSource N" --vallabel "N-TotalSource" --norm "log" --trigen $trigen --idxgrp $idxgrp &
