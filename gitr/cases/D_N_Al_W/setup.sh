#!/usr/bin/env bash

walldynExecDirName="/home/cloud/code/pyWallDYN/setupMPI/release"

#time mpirun -np 4 

${walldynExecDirName}/walldyn_setup_MPI --input GITR_ProtoEmpex_W_N_Al.xml --wtd modelandruntime --minfloat 1.0E-14 --valjac 0 --verbostiy 0 --compress 0 --dolsys 0



# OLD
#walldyn_setup --input GITR_ProtoEmpex_W_N_Al.xml --wtd modelandruntime --minfloat 1.0E-14 --valjac 0 --verbostiy 0 --compress 0 --dolsys 0
