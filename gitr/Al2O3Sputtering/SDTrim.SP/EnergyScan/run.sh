#!/usr/bin/env bash

energies=(100 1000 170 2000 250 300 4000 500)

echo 'Energy  Yields' > "sdtrim_res.dat"

for ee in "${energies[@]}"; do
  sd="$ee""eV"
  if [ ! -d $sd ]; then
    echo "creating run dir $sd"
    mkdir $sd
  else
    echo "rundir $sd already exists"
  fi

  sed "s/e0 = 100, 0, 0/e0 = $ee, 0, 0/g" tri_template.inp > "$sd/tri.inp"
  ln -sf "../mat_surfb.inp" "$sd/mat_surfb.inp"

  if [ ! -f "$sd/run-linux-6-00-MPI.sh" ]; then
    echo "Copying run file to $sd"
    cp "run_sdtrimsp.sh" "$sd/run-linux-6-00-MPI.sh"
  else
    echo "run file already exists in $sd"
  fi
  bkp=$PWD
  cd "$sd"
  echo "Running $ee"
  "./run-linux-6-00-MPI.sh" > out.txt
  cd $bkp
  "./totalyield.py" "$sd" "sdtrim_res.dat"
done
