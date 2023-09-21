import netCDF4 as nc
import numpy as np
from pathlib import Path

def load_indices_from_file(filename: Path) -> set:
    """Load indices from a file and return a set of integers."""
    with filename.open('r') as f:
        return {int(line.strip()) for line in f}

def get_new_surface_assignments_for_species(data, indices_set, species_val):
    hitwall = data['hitWall'][:]
    charge = data['charge'][:] #[hitwall > 0]
    surfaceHit = data['surfaceHit'][:] #[hitwall > 0]
    
    surfaceIndexHit = surfaceHit[charge == species_val]
    return np.array([1 if i in indices_set else 0 for i in surfaceIndexHit], dtype=int)



# Setting path to data
path2gitrData = Path('/Users/42d/MPEX-GITR-WallDYN/gitr/output')
path2SurfaceIndices = Path('/Users/42d/MPEX-GITR-WallDYN/walldyn3/data/surface')

# Setting up materials
materials = ['Al'] #['W', 'N', 'Al', 'D']

transfer_matrix = np.zeros((94, 94))  # Reset for every new combination of idx_from and idx_to

for material in materials:
    # Initialize an empty matrix for each material
    transfer_matrix = np.zeros((94, 94))
    
    for idx_from in range(94):
        for idx_to in range(94):
            data_path = path2gitrData / f"positions_{material}_loc_{idx_from + 1 }.nc"
            data = nc.Dataset(data_path)
            
            # Extract unique species from the data
            hitwall = data['hitWall'][:]
            charge = data['charge'][:]
            species = np.unique(charge)
            
            
            # Read indices from file
            indices_file = path2SurfaceIndices / f"surface_inds_{idx_to + 1}.txt"
            indices_set = load_indices_from_file(indices_file)
            
            for s in species:
                new_surface_assignments = get_new_surface_assignments_for_species(data, indices_set, s)

                # Store the sum in the matrix.
                transfer_matrix[idx_from, idx_to] += np.sum(new_surface_assignments)

                print(f"From index {idx_from}, To index {idx_to}, species {s}, surface {np.sum(new_surface_assignments)}")

    # Save the matrix
    print("Unique species", species)
    for s in species:
        filename = f"mat{material.capitalize()}{str(int(s))+'+'}.dat"
        np.savetxt(filename, transfer_matrix, fmt='%d', delimiter=',')


#EOF
