
import numpy as np
import netCDF4 as nc
from scipy.io import loadmat

def find_cell(x, y, z, ind_cell, z_edges, theta_edges, r_edges):
    # Convert x, y to polar coordinates
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Find the z segment
    z_segment = np.where((z >= z_edges[:, 0]) & (z < z_edges[:, 1]))[0]
    if len(z_segment) == 0:  # z is out of bounds
        return None
    z_segment = z_segment[0]
    
    # Find the theta segment
    theta_segment = np.where((theta >= theta_edges[:, 0]) & (theta < theta_edges[:, 1]))[0]
    if len(theta_segment) == 0:  # theta is out of bounds
        return None
    theta_segment = theta_segment[0]
    
    if z > 4.139999:  # this value seems to be from your MATLAB code
        r_segment = np.where((r >= r_edges[:, 0]) & (r < r_edges[:, 1]))[0]
        if len(r_segment) == 0:  # r is out of bounds
            return None
        r_segment = r_segment[0]
        return ind_cell[(theta_segment * len(z_edges)) + z_segment + r_segment]  # This is an approximation of the logic used in MATLAB
    
    index= ind_cell[(theta_segment * len(z_edges)) + z_segment]
    # print("index", index)
    return index


# Load the MAT file
data = loadmat('/Users/42d/MPEX-GITR-WallDYN/gitr/input/ind_cell.mat')

# Extract ind_cell from the loaded data
ind_cell = data['ind_cell']

print(ind_cell.shape)
# identiy cell 
# Inputs
nTheta = 6  # Number of segments in theta
nZ = 15 # Number of segments in z
nTarget_radii = 4 # Number of radial locations at target
max_radius = 0.0614

zlim0 = 1.55; # sets lower axial bound on geometry
zlim1 = 1.95; # sets upper axial bound on geometry
thetalim0 = -np.pi # sets lower axial bound on geometry
thetalim1 = np.pi # sets upper axial bound on geometry

z_edges = np.zeros((nZ, 2))
theta_edges = np.zeros((nTheta, 2))

dz= (zlim1-zlim0)/nZ
zgrid = np.linspace(zlim0, zlim1, nZ+1)

z_edges[0, :] = [-100, zgrid[1]]
z_edges[-1, :] = [zgrid[-2], 100]
z_edges[1:-1, 0] = zgrid[1:-2]
z_edges[1:-1, 1] = zgrid[2:-1]


dtheta = (thetalim1 - thetalim0) / nTheta
thetagrid = dtheta * np.arange(nTheta + 1) + thetalim0
theta_edges = np.zeros((nTheta, 2))
theta_edges[:, 0] = thetagrid[:-1]
theta_edges[:, 1] = thetagrid[1:]


dr = max_radius / nTarget_radii
rgrid = dr * np.arange(nTarget_radii + 1)
r_edges = np.zeros((nTarget_radii, 2))
r_edges[:, 0] = rgrid[:-1]
r_edges[:, 1] = rgrid[1:]



f = nc.Dataset('/Users/42d/MPEX-GITR-WallDYN/gitr/output/positions_Al_loc_1.nc', 'r')
hitwall =f.variables['hitWall'][...]

# get positions of particles that hit the wall
x = f.variables['x'][:][hitwall>0]
y = f.variables['y'][:][hitwall>0]
z = f.variables['z'][:][hitwall>0]
charge=f.variables['charge'][:][hitwall>0]


position = (x[0], y[0], z[0])
print(f"The position is: {position}")  #(-0.05297931006816852, -0.03182185850246151, 1.5713906902309873)
cell_index = find_cell(position[0], position[1], position[2], ind_cell, z_edges, theta_edges, r_edges)
print(f"The position is in cell index: {cell_index}")



exit()
# read the netcdf file positions.nc
MATERIALS = [deuterium, aluminum, nitrogen, tungsten]

f = nc.Dataset('positions_Al_loc_1.nc', 'r')
hitwall =f.variables['hitWall'][...]

# get positions of particles that hit the wall
x = f.variables['x'][:][hitwall>0]
y = f.variables['y'][:][hitwall>0]
z = f.variables['z'][:][hitwall>0]
charge=f.variables['charge'][:][hitwall>0]


# Number of particles
num_particles = 1000

# Number of surfaces
num_surfaces = 94

# Initialize the transfer matrix with zeros
M = np.zeros((num_surfaces, num_surfaces))

# Simulate the random transfer of particles
np.random.seed(0) # For reproducibility. You can comment this if you want truly random results each time.
for i in range(num_surfaces):
    # We assume an equal chance for a particle to land on any surface
    landed_particles = np.random.choice(num_surfaces, num_particles, replace=True)
    
    for j in landed_particles:
        M[i, j] += 1

# Define the column headers
header = []
for i in range(1, num_surfaces + 1):
    for j in range(1, num_surfaces + 1):
        header.append(f"{i}->{j}")

# Save the matrix to a CSV file
np.savetxt("transfer_matrix.csv", M, delimiter=",", header=",".join(header), comments="", fmt="%d")

