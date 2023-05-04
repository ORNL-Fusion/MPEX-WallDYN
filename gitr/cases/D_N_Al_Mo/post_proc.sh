#!/bin/bash

python=$(which python3)
script="$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/pyParseSolution.py"

# noplot="--noplot"
noplot=""

if [[ "$#" -ge 1 ]]; then
	case="./results/$1"
else
	echo "No case specified"
	exit 1
fi

if [[ "$#" -ge 2 ]]; then
	yldmodel="$2"
else
	yldmodel=""
	echo "No yldmodel defaulting to none"
fi

if [[ "$#" -ge 3 ]]; then
	tmax="$3"
else
	tmax="100.000000"
	echo "No tmax defaulting to $tmax"
fi

solvstime="Surfmod_toexport_vs_time.dat"
pprocvstime="surfmod_post_toexport_vs_time.dat"

if [ -f $script ] ; then

	cursoltateoft=$case"_sol_states.nc"
    curpostoft=$case"_post_states.nc"
    constants=$case"_constants.dat"


	if [ -f $cursoltateoft ] ; then
        echo "cursoltateoft: $cursoltateoft  exitst"
    else
        echo "cursoltateoft: $cursoltateoft  NOT FOUND"
    fi


    if [ -f $curpostoft ] ; then
        echo "curpostoft: $curpostoft  exitst"
    else
        echo "curpostoft: $curpostoft  NOT FOUND"
    fi


    if [ -f $constants ] ; then
        echo "constants: $constants  exitst"
    else
        echo "constants: $constants  NOT FOUND"
    fi

	if [ -f $solvstime ] ; then
        echo "solvstime: $solvstime  exitst"
    else
        echo "solvstime: $solvstime  NOT FOUND"
    fi

    if [[ -f $cursoltateoft && -f $curpostoft && -f $constants && -f $solvstime ]]; then
        $script --stateoft $cursoltateoft --postoft $curpostoft --constants $constants --tmax $tmax --solvstime $solvstime --noplot
    elif [[ -f $cursoltateoft && -f $constants &&-f $solvstime ]]; then
        $script --stateoft $cursoltateoft  --constants $constants --tmax $tmax --solvstime $solvstime --noplot
    else
        echo "--stateoft or --postoft or --solvstime not found"
    fi

	if [[ -f $curpostoft &&  -f $pprocvstime ]]; then
		$script --postoft $curpostoft --postvstime $pprocvstime --tmax $tmax
	else
        echo "--postoft or --postvstime not found"
    fi

	echo "noplot=|$noplot|"

	$python $script \
			--solstate $case"_0.000000.dat" \
	    --constants $case"_constants.dat" \
			--postproc $case"_ppext_0.000000.dat" $noplot

	$python $script \
		--solstate $case"_$tmax.dat" \
    --constants $case"_constants.dat" \
		--postproc $case"_ppext_$tmax.dat" $noplot

	$python $script \
		--stateoft $case"_sol_states.nc" \
		--postoft $case"_post_states.nc" \
		--constants $case"_constants.dat" \
		--tmax $tmax $noplot

	# Generate charge state resolved stuff
	$python $script \
		--constants $case"_constants.dat" \
		--solstate 	$case"_$tmax.dat" \
		--dochrgres
else
	echo "Script $script not found"
	cd $bkp
	exit 1
fi

script="$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/pyParseCustFuncTests.py"

if [ -f $script ] ; then

	$python $script \
		--input $case"_cust_func_tests.dat" \
		--out "./results/custfunctests_post/yieldtab$yldmodel"

	$python $script \
		--input $case"_cust_func_tests_final.dat" \
		--out "./results/custfunctests_post/final_yieldtab$yldmodel"
else
	echo "Script $script not found"
	cd $bkp
	exit 1
fi

cd $bkp
