import numpy as np
import netCDF4 as nc
from scipy.constants import k, e, m_p 
from sample_triangle import sampleTriangle
from getCellIndices import split_proto_geometry
#
def createParticleSourceFile(geometryData, nP, masses, materialsName, particleSourcesFolder,
    nTheta = 12, nZ = 30, nTarget_radii = 6, max_radius = 0.0614, 
    zlim0 = 1.55, zlim1 = 1.95, thetalim0 = -np.pi, thetalim1 = np.pi, zmax = 4.139999, zmin =0.50001):
    """create particle source file given a number of particles, masses, and materials
    Args:
        geometryData dict: dictionary containing geometry data
        nP (int): number of particles
        masses (list): materials masses
        materialsName (string): list of material names/symbols
        particleSourcesFolder (string): folder to save particle source file
    """

    # Load geometry file
    data = split_proto_geometry(geometryData)
    # Get geometry data and cell indices
    ind_cell = data 
    ind_cell = np.array(ind_cell)
    # Unpack geometry data
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
    area = geometryData['area']
    inDir = geometryData['inDir']
    plane_norm = geometryData['plane_norm']

    abcd = np.array([a, b, c, d]).T
    area = np.array(area)

    for k in range(len(masses)):
        for ii in range(nTheta * nZ + nTarget_radii):
            areas = area[ind_cell[ii][0]]
            area_cdf = np.cumsum(areas)
            area_cdf = area_cdf / area_cdf[-1]
            area_cdf = np.concatenate(([0], area_cdf))

            x_sample = []
            y_sample = []
            z_sample = []
            vx_sample = []
            vy_sample = []
            vz_sample = []

            # Verify these hardcoded values
            m = 27
            nPoints = 200
            maxE = 20
            Eb = 3.39
            a = 5
            E = np.linspace(0, maxE, nPoints)
            dE = E[1] - E[0]
            thompson2 = a * (a - 1) * E * Eb**(a - 1) / (E + Eb)**(a + 1)

            m = masses[k]
            ecdf = np.cumsum(thompson2)
            ecdf = ecdf / ecdf[-1]
            rand1 = np.random.rand(nP)
            randTheta = 2 * np.pi * np.random.rand(nP)
            randPhi = 0.5 * np.pi * np.random.rand(nP)
            Esamp = np.interp(rand1, ecdf, E)
            v = np.sqrt(2 * Esamp * e/ m / m_p)
            vx = v * np.cos(randTheta) * np.sin(randPhi)
            vy = v * np.sin(randTheta) * np.sin(randPhi)
            vz = v * np.cos(randPhi)

            buffer = 1e-5
            planes = np.column_stack((x1, y1, z1, x2, y2, z2, x3, y3, z3))

            X = planes[:, [0, 3, 6]]
            Y = planes[:, [1, 4, 7]]
            Z = planes[:, [2, 5, 8]]

            nP0 = 0
            nTriangles = len(planes)

            r1 = np.random.rand(nP)
            this_triangle = np.floor(np.interp(r1, area_cdf, np.arange(len(areas) + 1))).astype(int)
            for j in range(nP):
                i = ind_cell[ii][0][this_triangle[j]]
                x_tri = X[i, :]
                y_tri = Y[i, :]
                z_tri = Z[i, :]
                parVec = np.array([x_tri[1] - x_tri[0], y_tri[1] - y_tri[0], z_tri[1] - z_tri[0]])
                parVec = parVec / np.linalg.norm(parVec)
                samples = sampleTriangle(x_tri, y_tri, z_tri, 1)
                # check if samples is not empty
                if samples.size !=0:
                    exit()
                
    

                normal = inDir[i] * (-abcd[i, :3] / plane_norm[i])

                v_inds = j
    
                x_sample.append(samples[0][0] + buffer * normal[0])
                y_sample.append(samples[1][0] + buffer * normal[1])
                z_sample.append(samples[2][0] + buffer * normal[2])

                parVec2 = np.cross(parVec, normal)
                newV = vx[v_inds] * parVec + vy[v_inds] * parVec2 + vz[v_inds] * normal
                vx_sample.append(newV[0])
                vy_sample.append(newV[1])
                vz_sample.append(newV[2])

                filename = f"{particleSourcesFolder}/particle_source_{materialsName[k]}_{ii}.nc"
                ncid = nc.Dataset(filename, mode='w')

                dimP = ncid.createDimension('nP', nP)

                xVar = ncid.createVariable('x', 'f8', ('nP',))
                yVar = ncid.createVariable('y', 'f8', ('nP',))
                zVar = ncid.createVariable('z', 'f8', ('nP',))
                vxVar = ncid.createVariable('vx', 'f8', ('nP',))
                vyVar = ncid.createVariable('vy', 'f8', ('nP',))
                vzVar = ncid.createVariable('vz', 'f8', ('nP',))

                xVar[:] = x_sample
                yVar[:] = y_sample
                zVar[:] = z_sample
                vxVar[:] = vx_sample
                vyVar[:] = vy_sample
                vzVar[:] = vz_sample

                ncid.close()


