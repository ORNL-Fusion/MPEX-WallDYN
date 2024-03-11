import numpy as np
import libconf
import io
from periodictable import deuterium, aluminum, nitrogen, tungsten
import subprocess as sp
import os
import shutil
from os.path import join

def run_gitr(base_path, output_path, location_index, impurity_symbol):
    """
    Launch GITR using subprocess.Popen and save output to the output folder.
    """
    positions_file = f"positions_{impurity_symbol}_loc_{location_index}.nc"
    gitr_path = join(base_path, "GITR")
    positions_file_path = join(output_path, positions_file)
    
    print("Running Simulation!")
    sp.Popen([gitr_path]).wait()
    
    shutil.copyfile(join(output_path, "positions.nc"), positions_file_path)
    os.remove(join(output_path, "positions.nc"))
    os.remove(join(output_path, "positions.m"))

def read_config(filename):
    """Reads a configuration from a given filename."""
    with io.open(filename, 'r') as f:
        return libconf.load(f)

def write_config(filename, config):
    """Writes a configuration to a given filename."""
    with io.open(filename, 'w') as f:
        libconf.dump(config, f)

def update_species_properties(filename, impurity_species, background_species, cell_index):
    """Updates species properties in the input file."""
    impurity_symbol = impurity_species.symbol
    ionization_file = f"ADAS_Rates_{impurity_symbol}.nc"
    particle_source_file = f"particle_source_{impurity_symbol}_{cell_index}.nc"
    
    try:
        config = read_config(filename)
        config['backgroundPlasmaProfiles'].update({'Z': background_species.number, 'amu': background_species.mass})
        config['impurityParticleSource']['initialConditions'].update({
            'impurity_Z': impurity_species.number, 'charge': 0.0, 'impurity_amu': impurity_species.mass
        })
        config['impurityParticleSource']['ionization']['fileString'] = ionization_file
        config['impurityParticleSource']['recombination']['fileString'] = ionization_file
        config['particleSource']['ncFileString'] = particle_source_file
        write_config(filename, config)
    except (KeyError, IOError) as e:
        print(f"Error updating species properties: {e}")

# Testing the function
BASE_PATH = "/Users/42d/GITR-1.3.0/build"
OUTPUT_PATH = "/Users/42d/MPEX-WallDYN/gitr/output"
FILENAME = "/Users/42d/MPEX-GITR-WallDYN/gitr/input/gitrInput.cfg"
MATERIALS = [deuterium, aluminum, nitrogen, tungsten]
MATERIALS = [aluminum]
for material in MATERIALS:
    for cell_index in range(1, 95):
        update_species_properties(FILENAME, material, deuterium, cell_index)
        run_gitr(BASE_PATH, OUTPUT_PATH, cell_index, material.symbol)
