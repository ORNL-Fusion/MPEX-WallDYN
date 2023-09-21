import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits import mplot3d
import io, libconf
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


#######################--------------------#######################
#######################--------------------#######################

filename="../../gitr/input/gitrGeometryPointPlane3d.cfg"

with io.open(filename) as f:
    config = libconf.load(f)
    
data=config['geom']
coords=['x1','x2','x3','y1','y2','y3','z1','z2','z3']
[x1,x2,x3,y1,y2,y3,z1,z2,z3]=[data.get(var) for var in coords]
# get a,b,c,d for the plane
a = data.get('a')
b = data.get('b')
c = data.get('c')
d = data.get('d')
Zsurface = data.get('Z')
z1 = np.array(z1)
z2 = np.array(z2)
z3 = np.array(z3)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
Zsurface = np.array(Zsurface)

Zsurface = np.array(Zsurface)

Zsurface_indices = np.where(Zsurface)[0]
nSurfaces = len(a)
cap1_indices = np.where((Zsurface == 0) & (z1 < 0.50001) & (z2 < 0.50001) & (z3 < 0.50001))[0]
cap2_indices = np.where((Zsurface == 0) & (z1 > 4.139999) & (z2 > 4.139999) & (z3 > 4.139999))[0]
subset_indices = np.where(Zsurface > 0)[0]

# Centroid calculations
centroids = np.column_stack([(x1 + x2 + x3) / 3,
                            (y1 + y2 + y3) / 3,
                            (z1 + z2 + z3) / 3])

theta_centroids = np.arctan2(centroids[:, 1], centroids[:, 0])
r_centroids = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)

nTheta = 6
nZ = 15
nTarget_radii = 4
max_radius = 0.0614
zlim0 = 1.55
zlim1 = 1.95
thetalim0 = -np.pi
thetalim1 = np.pi

import numpy as np

# Initialize variables
zlim0 = 1.55  # sets lower axial bound on geometry
zlim1 = 1.95  # sets upper axial bound on geometry
thetalim0 = -np.pi
thetalim1 = np.pi

# Create z edges
dz = (zlim1 - zlim0) / nZ
zgrid = dz * np.arange(nZ + 1) + zlim0
z_edges = np.zeros((nZ, 2))
z_edges[0, :] = [-100, zgrid[1]]
z_edges[-1, :] = [zgrid[-2], 100]
z_edges[1:-1, 0] = zgrid[1:-2]
z_edges[1:-1, 1] = zgrid[2:-1]

# Create theta edges
dtheta = (thetalim1 - thetalim0) / nTheta
thetagrid = dtheta * np.arange(nTheta + 1) + thetalim0
theta_edges = np.column_stack([thetagrid[:-1], thetagrid[1:]])

# Create r edges
dr = max_radius / nTarget_radii
rgrid = dr * np.arange(nTarget_radii + 1)
r_edges = np.column_stack([rgrid[:-1], rgrid[1:]])

# Populate ind_cell list
ind_cell = []

for i in range(nTheta):
    for j in range(nZ):
        mask = (Zsurface > 0) & (centroids[:, 2] >= z_edges[j, 0]) & (centroids[:, 2] < z_edges[j, 1]) & \
              (theta_centroids >= theta_edges[i, 0]) & (theta_centroids < theta_edges[i, 1])
        ind_cell.append(np.where(mask)[0])

for i in range(nTarget_radii):
    mask = (Zsurface == 0) & (z1 > 4.139999) & (z2 > 4.139999) & (z3 > 4.139999) & \
           (r_centroids >= r_edges[i, 0]) & (r_centroids < r_edges[i, 1])
    ind_cell.append(np.where(mask)[0])


fig, ax = plt.subplots()

colors = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow']

for i in range(nTheta * nZ + nTarget_radii):
    subset = ind_cell[i]
    
    # Extract the X, Y values based on the subset
    X = np.column_stack([x1[subset], x2[subset], x3[subset]])
    Y = np.column_stack([y1[subset], y2[subset], y3[subset]])
    
    for j in subset:
        polygon = patches.Polygon([(x1[j], y1[j]), (x2[j], y2[j]), (x3[j], y3[j])], closed=True, facecolor=colors[i % 6], alpha=0.6)
        ax.add_patch(polygon)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
plt.axis('equal')  # This ensures the x and y axes are scaled the same
plt.show()



# Extract the magnitude of material 'w'
def get_w_magnitude(material_data_row):
    return abs(material_data_row[''])

# Load the material data and surface-cell mapping
path ='/Users/42d/Desktop/MPEX-WallDYN/walldyn3/walldyn3Output/cases/D_N_Al_W/results/'
material_data = np.genfromtxt(path + 'ProtoEmpex_ppext_NetAdensChange_100.000.dat', names=True)
path ='/Users/42d/Desktop/MPEX-WallDYN/walldyn3/walldyn3Output/cases/data_o_al/surface/'

fig, ax = plt.subplots()

# Find max and min values for normalization
w_values = [get_w_magnitude(row) for row in material_data]
max_w = max(w_values)
min_w = min(w_values)

# Using a colormap to represent magnitude
cmap = plt.cm.Reds
cmap = plt.cm.plasma

for surface_idx in range(1, 94, 1):
    cell_indices = np.genfromtxt(path + 'surface_inds_' + str(surface_idx) + '.txt', dtype=int)
    
    normalized_w = (get_w_magnitude(material_data[surface_idx]) - min_w) / (max_w - min_w)
    color = cmap(normalized_w)

    for cell_idx in cell_indices:
        corrected_idx = cell_idx - 1  # Convert to 0-based index
        polygon = patches.Polygon([(x1[corrected_idx], y1[corrected_idx]), 
                                   (x2[corrected_idx], y2[corrected_idx]), 
                                   (x3[corrected_idx], y3[corrected_idx])], 
                                  closed=True, facecolor=color, alpha=0.6)
        ax.add_patch(polygon)

# Create colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_w, vmax=max_w))
cbar = plt.colorbar(sm, ax=ax)


ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
plt.axis('equal')
plt.show()
