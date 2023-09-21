import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits import mplot3d
import io, libconf
import os


filename="gitrGeometryPointPlane3d.cfg"
filename="../../gitr/input/gitrGeometryPointPlane3d.cfg"

with io.open(filename) as f:
 config = libconf.load(f)

x1 = np.array(config.geom.x1)
x2 = np.array(config.geom.x2)
x3 = np.array(config.geom.x3)
y1 = np.array(config.geom.y1)
y2 = np.array(config.geom.y2)
y3 = np.array(config.geom.y3)
z1 = np.array(config.geom.z1)
z2 = np.array(config.geom.z2)
z3 = np.array(config.geom.z3)
area = np.array(config.geom.area)
surf = np.array(config.geom.surface)


figure = plt.figure(figsize = (10, 8))
kwds = {'projection':'3d'}
ax = figure.gca(**kwds)
ax.autoscale(tight=False)
        
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.zaxis.set_tick_params(labelsize=20)
        
# Creating plot
ax.scatter3D(x1, y1, z1, color = "green")
ax.set_title('Final W Particle Positions')
ax.set_zlabel('Z',fontsize=20, labelpad=30)
ax.set_xlabel('X',fontsize=20, labelpad=20)
ax.set_ylabel('Y',fontsize=20, labelpad=20)
ax.tick_params(axis='z', pad=22)
ax.grid(False)
plt.tight_layout()
plt.show()
