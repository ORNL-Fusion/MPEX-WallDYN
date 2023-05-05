import os
import numpy as np
import netCDF4 as nc
import scipy.io
from scipy.spatial import ConvexHull


def split_proto_geometry(geometryData, nTheta = 12, nZ = 30, nTarget_radii = 6, max_radius = 0.0614, 
    zlim0 = 1.55, zlim1 = 1.95, thetalim0 = -np.pi, thetalim1 = np.pi, zmax = 4.139999, zmin =0.50001 ):
    """
    This function takes in a 3D surface mesh geometry file and splits it into cells
    based on the input parameters. It returns the indices of the cells and the
    geometry file.

    Inputs:
        geometryData - dictionary containing the geometry data: x1, x2, x3, y1, y2, y3, z1, z2, z3, a, b, c, d, Z
    """
    # Unpack geometry data
    Z = geometryData['Z']
    x1 = geometryData['x1']
    x2 = geometryData['x2']
    x3 = geometryData['x3']
    y1 = geometryData['y1']
    y2 = geometryData['y2']
    y3 = geometryData['y3']
    z1 = geometryData['z1']
    z2 = geometryData['z2']
    z3 = geometryData['z3']
    a = geometryData['a']
    b = geometryData['b']
    c = geometryData['c']
    d = geometryData['d']
    Z = geometryData['Z']
    inDir = geometryData['inDir']

    Zsurface = Z
    abcd = np.column_stack((a, b, c, d))
    surface = np.where(Zsurface)
    nSurfaces = len(a)

    # Find end-cap indices
    z1 = np.array(z1)
    z2 = np.array(z2)
    z3 = np.array(z3)

    cap1 = np.where((Zsurface == 0) & (z1 < zmin ) & (z2 < zmin ) & (z3 < zmin))
    cap2 = np.where((Zsurface == 0) & (z1 > zmax) & (z2 > zmax) & (z3 > zmax))
    subset = cap2

    Zsurface = np.array(Zsurface)
    subset = np.where(Zsurface > 0)

    # Calculate some stuff about centroids
    # This script uses centroid locations to group surfaces
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = np.array(y3)
    centroid = np.column_stack((1 / 3 * (x1 + x2 + x3),
                                 1 / 3 * (y1 + y2 + y3),
                                 1 / 3 * (z1 + z2 + z3)))

    theta_centroid = np.arctan2(centroid[:, 1], centroid[:, 0])
    r_centroid = np.sqrt(centroid[:, 1] ** 2 + centroid[:, 0] ** 2)

    z_edges = np.zeros((nZ, 2))
    theta_edges = np.zeros((nTheta, 2))

    dz = (zlim1 - zlim0) / nZ
    zgrid = dz * np.arange(nZ + 1) + zlim0

    z_edges[0, :] = [-100, zgrid[1]]
    z_edges[-1, :] = [zgrid[-2], 100]
    z_edges[1:-1, 0] = zgrid[1:-2]
    z_edges[1:-1, 1] = zgrid[2:-1]

    dtheta = (thetalim1 - thetalim0) / nTheta
    thetagrid = dtheta * np.arange(nTheta + 1) + thetalim0
    theta_edges[:, 0] = thetagrid[:-1]
    theta_edges[:, 1] = thetagrid[1:]

    dr = (max_radius) / nTarget_radii
    rgrid = dr * np.arange(nTarget_radii + 1)
    r_edges = np.zeros((nTarget_radii, 2))
    r_edges[:, 0] = rgrid[:-1]
    r_edges[:, 1] = rgrid[1:]

    ind_cell = []
    for i in range(nTheta):
        for j in range(nZ):
            indices = np.where((Zsurface > 0) & (centroid[:, 2] >= z_edges[j, 0]) & (centroid[:, 2] < z_edges[j, 1]) &
                               (theta_centroid >= theta_edges[i, 0]) & (theta_centroid < theta_edges[i, 1]))
            ind_cell.append(indices)

    for i in range(nTarget_radii):
        indices = np.where((Zsurface == 0) & (z1 > zmax) & (z2 > zmax) & (z3 > zmax) &
                           (r_centroid >= r_edges[i, 0]) & (r_centroid < r_edges[i, 1]))
        ind_cell.append(indices)

    # Save ind_cell to a .mat file
    scipy.io.savemat('ind_cell.mat', {'ind_cell': ind_cell})

    return ind_cell


if __name__ == '__main__':
    geometry_file = os.path.join(os.getcwd(), 'gitrGeometryPointPlane3d.cfg')
    if not os.path.exists('x1'):
        with open(geometry_file, 'r') as fid:
            lines = fid.readlines()
            for line in lines[2:26]:
                exec(line.strip())
    # Create a dictionary to store the geometry data
    geometryData = {'x1': x1, 'x2': x2, 'x3': x3, 'y1': y1, 'y2': y2, 'y3': y3, 'z1': z1, 'z2': z2, 'z3': z3, 'a': a, 'b': b, 'c': c, 'd': d, 'Z': Z}
    ind_cell_result = split_proto_geometry(geometryData)

