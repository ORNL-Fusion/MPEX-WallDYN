import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits import mplot3d
import io, libconf
import os


#######################--------------------#######################
#######################--------------------#######################

filename="gitrGeometryPointPlane3d.cfg"

with io.open(filename) as f:
    config = libconf.load(f)
    
data=config['geom']
coords=['x1','x2','x3','y1','y2','y3','z1','z2','z3']
[x1,x2,x3,y1,y2,y3,z1,z2,z3]=[data.get(var) for var in coords]

ne = np.loadtxt("/Users/42d/MPEX-WallDYN/gitr/input/plasma/ne.txt", unpack=True)
Z = data.get('Z')
plt.plot(z1,ne, 'o')
plt.show()
print("shape Z", len(Z), len(x1), len(z1), len(ne))

exit()
# load particle source

figure = plt.figure(figsize = (10, 8))
from mpl_toolkits.mplot3d import Axes3D
ax = figure.add_subplot(111, projection='3d')

ax.autoscale(tight=False)

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.zaxis.set_tick_params(labelsize=20)

# Creating plot
#ax.scatter3D(x, y, z, color = "green")
ax.plot(np.append(x1,x1[0]), np.append(y1,y1[0]), np.append(z1,z1[0]) ,'bs')
ax.plot(np.append(x2,x2[0]), np.append(y2,y2[0]), np.append(z2,z2[0]), 'bs')

for i in range(1,367,1):
    particles = nc.Dataset("particle_sources/particle_source_Al_"+str(i)+".nc", "r")
    ax.plot(particles["x"][:], particles["y"][:], particles["z"][:], 'ro')

#ax.set_title('Final W Particle Positions')
ax.set_zlabel('Z[m]',fontsize=20, labelpad=30)
ax.set_xlabel('X[m]',fontsize=20, labelpad=20)
ax.set_ylabel('Y[m]',fontsize=20, labelpad=20)
ax.tick_params(axis='z', pad=22)
ax.grid(False)
plt.tight_layout()
plt.savefig('geometry.png')
plt.show()

