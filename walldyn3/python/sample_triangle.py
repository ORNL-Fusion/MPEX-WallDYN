import numpy as np

def sampleTriangle(x, y, z, nP):
    """Sample a triangle defined by three points
    Args:
        x (list): x coordinates of triangle vertices
        y (list): y coordinates of triangle vertices
        z (list): z coordinates of triangle vertices
        nP (int): number of particles to sample
    Returns:
        samples (list): list of sampled particles
    """

    x_transform = x - x[0]
    y_transform = y - y[0]
    z_transform = z - z[0]
    
    v1 = np.array([x_transform[1], y_transform[1], z_transform[1]])
    v2 = np.array([x_transform[2], y_transform[2], z_transform[2]])
    v12 = v2 - v1
    normalVec = np.cross(v1, v2)

    a1 = np.random.rand(nP, 1)
    a2 = np.random.rand(nP, 1)

    samples = a1 * v1 + a2 * v2
    
    samples2x = samples[:, 0][0] - v2[0]
    samples2y = samples[:, 1][0]  - v2[1]
    samples2z = samples[:, 2][0]  - v2[2]
    samples12x = samples[:, 0][0]  - v1[0]
    samples12y = samples[:, 1][0]  - v1[1]
    samples12z = samples[:, 2][0]  - v1[2]


    v1Cross = np.column_stack([(v1[1]*samples[:, 2][0] - v1[2]*samples[:, 1][0]),
                                (v1[2]*samples[:, 0][0] - v1[0]*samples[:, 2][0]),
                                (v1[0]*samples[:, 1][0] - v1[1]*samples[:, 0][0])])

    v2 = -v2
    v2Cross = np.column_stack([(v2[1]*samples2z - v2[2]*samples2y),
                                (v2[2]*samples2x - v2[0]*samples2z),
                                (v2[0]*samples2y - v2[1]*samples2x)])

    v12Cross = np.column_stack([(v12[1]*samples12z - v12[2]*samples12y),
                                 (v12[2]*samples12x - v12[0]*samples12z),
                                 (v12[0]*samples12y - v12[1]*samples12x)])

    v1CD = normalVec[0]*v1Cross[:, 0] + normalVec[1]*v1Cross[:, 1] + normalVec[2]*v1Cross[:, 2]
    v2CD = normalVec[0]*v2Cross[:, 0] + normalVec[1]*v2Cross[:, 1] + normalVec[2]*v2Cross[:, 2]
    v12CD = normalVec[0]*v12Cross[:, 0] + normalVec[1]*v12Cross[:, 1] + normalVec[2]*v12Cross[:, 2]

    inside = np.abs(np.sign(v1CD) + np.sign(v2CD) + np.sign(v12CD))

    insideInd = np.where(inside == 3)
    notInsideInd = np.where(inside != 3)
    
    # check if notInsideInd is empty
    if notInsideInd[0].size != 0:
        samples[:, 0] = samples[:, 0] + x[0]
        samples[:, 1] = samples[:, 1] + y[0]
        samples[:, 2] = samples[:, 2] + z[0]
        
        v2 = -v2
        dAlongV1 = v1[0]*samples[notInsideInd[0], 0] + v1[1]*samples[notInsideInd[0], 1] + v1[2]*samples[notInsideInd[0], 2]
        dAlongV2 = v2[0]*samples[notInsideInd[0], 0] + v2[1]*samples[notInsideInd[0], 1] + v2[2]*samples[notInsideInd[0], 2]
        
        dV1 = np.linalg.norm(v1)
        dV2 = np.linalg.norm(v2)
        halfdV1 = 0.5 * dV1
        halfdV2 = 0.5 * dV2

        samples[notInsideInd[0], :] = [-(samples[notInsideInd[0], 0][0] - 0.5 * v1[0]) + 0.5 * v1[0],
                                    -(samples[notInsideInd[0], 1][0] - 0.5 * v1[1]) + 0.5 * v1[1],
                                    -(samples[notInsideInd[0], 2][0] - 0.5 * v1[2]) + 0.5 * v1[2]]

        samples[notInsideInd[0], :] = [(samples[notInsideInd[0], 0][0] + v2[0]),
                                    (samples[notInsideInd[0], 1][0] + v2[1]),
                                    (samples[notInsideInd[0], 2][0] + v2[2])]

        samples[:, 0] = samples[:, 0][0] + x[0]
        samples[:, 1] = samples[:, 1][0] + y[0]
        samples[:, 2] = samples[:, 2][0] + z[0]

        return samples

