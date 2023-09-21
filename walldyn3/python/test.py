import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits import mplot3d
import io, libconf
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

def plot_geom(ax, subset, color_value, x1, x2, x3, y1, y2, y3, z1, z2, z3):
    subset = np.array(list(map(int, subset)))  # Convert subset to a numpy array
    X = np.array([x1[subset], x2[subset], x3[subset]]).T
    Y = np.array([y1[subset], y2[subset], y3[subset]]).T
    Z = np.array([z1[subset], z2[subset], z3[subset]]).T
    
    ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), color=color_value, shade=True, alpha=0.3)

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

# Read in surface index groupings
nSurfaces = 94
surf_ind_cell = []
for i in range(1, nSurfaces + 1):
    data = np.loadtxt(f'../../walldyn3/data/surface/surface_inds_{i}.txt').flatten()
    # surf_ind_cell.append(data.astype(int))
    surf_ind_cell.append(data.astype(int) - 1)

subset = np.array(np.arange(len(x1)))
print("Subset shape:", subset.shape)
print("Subset dtype:", subset.dtype)
print("x1 shape:", x1.shape, "dtype:", x1.dtype)
print("x2 shape:", x2.shape, "dtype:", x2.dtype)
print("x3 shape:", x3.shape, "dtype:", x3.dtype)
X = np.array([x1[subset], x2[subset], x3[subset]]).T


# For plotting different sections
colors = ['r', 'g', 'b', 'c', 'm', 'y']
# Plot the complete GITR geometry (all the same color)
subset = np.array(np.arange(len(x1)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(subset.dtype)
subset = subset.astype(int)
# for i in range(nSurfaces):
#     subset = np.array(surf_ind_cell[i], dtype=int)  # Convert to numpy and ensure integer type
#     plot_geom(ax, subset, colors[i % 6], x1, x2, x3, y1, y2, y3, z1, z2, z3)

# plt.show()


# # Plot each surface grouping (94 sections), and only the helicon and target
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(nSurfaces):
#     subset = surf_ind_cell[i]
#     plot_geom(ax, subset, colors[i % 6], x1, x2, x3, y1, y2, y3, z1, z2, z3)

# Read in areal density data
density_data = np.loadtxt('../../walldyn3/data/results/ProtoEmpex_Gamma_100.000.dat', skiprows=1, unpack=True)
density_data = np.loadtxt('../../walldyn3/data/results/ProtoEmpex_EroFlux_n_by_n_100.000.dat', skiprows=1, unpack=True)

al_dens = density_data[:, 1]  #* 1e18
n_dens = density_data[:, 2] #* 1e18
w_dens = density_data[:, 3] #* 1e18

# Create new figures for Al, N, and W densities
for idx, dens in enumerate([al_dens, n_dens, w_dens], start=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(nSurfaces):
        subset = surf_ind_cell[i]
        color_value = dens[i]
        plot_geom(ax, subset, color_value, x1, x2, x3, y1, y2, y3, z1, z2, z3)
    fig.colorbar(ax.collections[0], ax=ax)

plt.show()

