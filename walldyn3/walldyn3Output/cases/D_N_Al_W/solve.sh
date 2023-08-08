#!/usr/bin/env bash

npar=4
walldynSolverPath="/home/cloud/code/pyWallDYN/solver/release"
executable=${walldynSolverPath}/walldyn_solver
#executable="$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/solver/release/walldyn_solver"

# executable=echo

echo "npar = " $npar

case="ProtoEmpex"

control="_control.xml"
model="_cppMODEL.XML"
runtime="_cppRunTime.XML"

if [[ -f "$case$model"  && -f "$case$runtime" && -f "$case$control" ]]; then

export OMP_NUM_THREADS=$npar

  $executable \
  --model "$case$model" \
  --runtime "$case$runtime" \
  --control "$case$control" \
  --minfloat 1.0E-14 \
  --npar $npar \
  --verbostiy 0
else
  echo "$case$model, $case$runtime or $case$control do not exist"
fi
