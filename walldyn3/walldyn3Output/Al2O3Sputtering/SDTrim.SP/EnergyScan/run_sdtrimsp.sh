#!/bin/bash

echo "running SDTrim.SP in $PWD"
mpiexec -n 4 ~/work/RPP/Programs/SDTrimSP_6.00/bin/linux_sdtrim_sp
