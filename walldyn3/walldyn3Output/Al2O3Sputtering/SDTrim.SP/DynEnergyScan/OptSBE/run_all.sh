#!/usr/bin/env bash

prog="$HOME/work/RPP/Programs/SDTrimSP_6.00/bin/linux_sdtrim_sp"
script="./totalyield.py"
destfile="sdtrim_res.dat"

# if [[  -f $(readlink $script) ]]; then
#   echo "$script found"
# else
#   echo "script $(readlink $script) not found"
# fi
#
#
# if [[  -f $prog  ]]; then
#   echo "$prog found"
# else
#   echo "Program $prog not found"
# fi
#
# exit 0

if [[ -f $prog && -f $(readlink $script) ]]; then
  echo -e 'Energy\tYields\tCAl\tCO\tSBEAl\tSBEO' > $destfile
  for fd in Energy_*; do
    bkp=$PWD
    cd $fd
    echo "Running in $PWD"
    $prog &
    cd $bkp
  done
wait
  echo "All jobs complete"
  for fd in Energy_*; do
    echo "Running in $fd"
    $script "$fd" $destfile
  done
else
  echo "Program $prog or script $script not found"
  exit 1
fi
