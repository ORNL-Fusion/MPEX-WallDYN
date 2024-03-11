# File to split the Proto GITR geometry into two part
# Adapted from Matlab code by T. Younkin
#
from matplotlib.patches import PathPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import netCDF4 as nc
import numpy as np
import os
import scipy.io
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from createParticleSource import createParticleSourceFile


# Read in 3D surface mesh geometry file
geometry_filename ="gitrGeometryPointPlane3d.cfg"
geometry = os.path.join(os.getcwd(), geometry_filename)
if not os.path.exists('x1'):
    with open(geometry, 'r') as fid:
        lines = fid.readlines()
        for line in lines[2:26]:
            exec(line.strip())
# Create a dictionary to store the geometry data
geometryData = {'x1': x1, 'x2': x2, 'x3': x3, 'y1': y1, 'y2': y2, 'y3': y3, 'z1': z1, 
'z2': z2, 'z3': z3, 'a': a, 'b': b, 'c': c, 'd': d, 'Z': Z, 'area': area, 'inDir': inDir,
'plane_norm': plane_norm}
particleSourcesFolder = "particleSources" #os.path.join(os.getcwd(), 'particleSources')
nP = 10
#atomic mass of each material Al and O
masses = [26.981539, 15.999]
materialsName = ['Al', 'O']
geometry_filename = 'gitrGeometryPointPlane3d.cfg'
# s =  createParticleSourceFile(geometryData, nP, masses, materialsName, particleSourcesFolder)
s = createParticleSourceFile(geometryData, nP, masses, materialsName, particleSourcesFolder,
    nTheta = 12, nZ = 30, nTarget_radii = 6, max_radius = 0.0614, 
    zlim0 = 1.55, zlim1 = 1.95, thetalim0 = -np.pi, thetalim1 = np.pi, 
    zmax = 4.139999, zmin =0.50001)




#if not os.path.exists('x1'):
#     with open(filename, 'r') as fid:
#         lines = fid.readlines()
#         for line in lines[2:26]:
#             exec(line.strip())
#
#     Zsurface = Z
#print(x1)
# abcd = np.column_stack((a, b, c, d))
# surface = np.where(Zsurface)
# nSurfaces = len(a)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = np.column_stack((x1, x2, x3)).T
# Y = np.column_stack((y1, y2, y3)).T
# Z = np.column_stack((z1, z2, z3)).T
# ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), triangles=np.arange(X.shape[1]).reshape(-1, 3), color='g', alpha=0.3, edgecolor=(0.1, 0.2, 0.5, 0.3))

# ax.set_title('Proto MPEX Geometry')
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')

# # plt.show()
# # Find end-cap indices
# z1 = np.array(z1)
# z2 = np.array(z2)
# z3 = np.array(z3)

# cap1 = np.where((Zsurface == 0) & (z1 < 0.50001) & (z2 < 0.50001) & (z3 < 0.50001))
# cap2 = np.where((Zsurface == 0) & (z1 > 4.139999) & (z2 > 4.139999) & (z3 > 4.139999))
# subset = cap2

# Zsurface = np.array(Zsurface)
# subset = np.where(Zsurface > 0)



# # Calculate some stuff about centroids
# # This script uses centroid locations to group surfaces
# x1  = np.array(x1)
# x2  = np.array(x2)
# x3  = np.array(x3)
# y1  = np.array(y1)
# y2  = np.array(y2)
# y3  = np.array(y3)
# centroid = np.column_stack((1/3 * (x1 + x2 + x3),
#                             1/3 * (y1 + y2 + y3),
#                             1/3 * (z1 + z2 + z3)))

# theta_centroid = np.arctan2(centroid[:, 1], centroid[:, 0])
# r_centroid = np.sqrt(centroid[:, 1] ** 2 + centroid[:, 0] ** 2)

# # Inputs
# nTheta = 12  # Number of segments in theta
# nZ = 30  # Number of axial segments
# nTarget_radii = 6  # Number of radial locations at target
# max_radius = 0.0614

# zlim0 = 1.55  # sets lower axial bound on geometry
# zlim1 = 1.95  # sets upper axial bound on geometry
# thetalim0 = -np.pi
# thetalim1 = np.pi

# z_edges = np.zeros((nZ, 2))
# theta_edges = np.zeros((nTheta, 2))

# dz = (zlim1 - zlim0) / nZ
# zgrid = dz * np.arange(nZ + 1) + zlim0

# z_edges[0, :] = [-100, zgrid[1]]
# z_edges[-1, :] = [zgrid[-2], 100]
# z_edges[1:-1, 0] = zgrid[1:-2]
# z_edges[1:-1, 1] = zgrid[2:-1]

# dtheta = (thetalim1 - thetalim0) / nTheta
# thetagrid = dtheta * np.arange(nTheta + 1) + thetalim0
# theta_edges[:, 0] = thetagrid[:-1]
# theta_edges[:, 1] = thetagrid[1:]

# dr = (max_radius) / nTarget_radii
# rgrid = dr * np.arange(nTarget_radii + 1)
# r_edges = np.zeros((nTarget_radii, 2))
# r_edges[:, 0] = rgrid[:-1]
# r_edges[:, 1] = rgrid[1:]

# ##

# # Assuming nTheta, nZ, z_edges, theta_edges, and nTarget_radii are defined

# # Assuming nTheta, nZ, z_edges, theta_edges, and nTarget_radii are defined
# ind_cell = []
# for i in range(nTheta):
#     for j in range(nZ):
#         indices = np.where((Zsurface > 0) & (centroid[:, 2] >= z_edges[j, 0]) & (centroid[:, 2] < z_edges[j, 1]) &
#                            (theta_centroid >= theta_edges[i, 0]) & (theta_centroid < theta_edges[i, 1]))
#         ind_cell.append(indices)

# for i in range(nTarget_radii):
#     indices = np.where((Zsurface == 0) & (z1 > 4.139999) & (z2 > 4.139999) & (z3 > 4.139999) &
#                        (r_centroid >= r_edges[i, 0]) & (r_centroid < r_edges[i, 1]))
#     ind_cell.append(indices)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colors = ['r', 'g', 'b', 'c', 'm', 'y']

# for i in range(nTheta * nZ + nTarget_radii):
#     subset = ind_cell[i][0]

#     X = np.column_stack((x1[subset], x2[subset], x3[subset])).T
#     Y = np.column_stack((y1[subset], y2[subset], y3[subset])).T
#     Z = np.column_stack((z1[subset], z2[subset], z3[subset])).T

#     for j in range(len(subset)):
#         verts = [list(zip(X[:, j], Y[:, j], Z[:, j]))]
#         poly = Poly3DCollection(verts, facecolor=colors[i % 6], alpha=1, edgecolor=(0, 0, 0, 0.1))
#         ax.add_collection3d(poly)

#     ax.set_title('Geometry')
#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_zlabel('Z [m]')

# plt.show()

# # Save ind_cell to a .mat file
# import scipy.io
# scipy.io.savemat('ind_cell.mat', {'ind_cell': ind_cell})


