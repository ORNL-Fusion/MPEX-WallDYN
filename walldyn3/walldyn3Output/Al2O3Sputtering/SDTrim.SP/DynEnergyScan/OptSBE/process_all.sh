#!/usr/bin/env bash

script="./totalyield.py"
destfile="sdtrim_res.dat"

if [[ -f $(readlink $script) ]]; then
  echo -e 'Energy\tYields\tCAl\tCO\tSBEAl\tSBEO' > $destfile
  for fd in Energy_*; do
    echo "Running in $fd"
    $script "$fd" $destfile
  done
else
  echo "script $script not found"
  exit 1
fi
