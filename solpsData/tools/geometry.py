import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from mpl_toolkits import mplot3d
import io, libconf
import os

#######################--------------------#######################
#######################--------------------#######################

filename="../../gitr/input/gitrGeometryPointPlane3d.cfg"
#filename="/Users/42d/Downloads/gitrWESTGeometry2D.cfg"


with io.open(filename) as f:
    config = libconf.load(f)
    
data=config['geom']
coords=['x1','x2','x3','y1','y2','y3','z1','z2','z3']
[x1,x2,x3,y1,y2,y3,z1,z2,z3]=[data.get(var) for var in coords]
    
Zbar =data.get('Z')
#print("data", data)
def get_color(Zbar_val):
    if Zbar_val == 0:
        return 'blue'
    elif Zbar_val == 74.0:
        return 'red'
    else:
        return 'green' # This is just a default color in case there's another unexpected value.

#
p= data.get('potential')
print("p", len(p))
print("Zbar", len(Zbar), "x1", len(x1)) #, "potential",len(data.get('potential')), 'slope', len(data.get('slope')),'inDir', len(data.get('inDir')))
exit()
#exit()

figure = plt.figure(figsize=(10, 8))
ax = figure.add_subplot(111, projection='3d')
ax.autoscale(tight=False)

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.zaxis.set_tick_params(labelsize=20)

# Plot the surfaces
for xi, yi, zi, zbari in zip([x1, x2, x3], [y1, y2, y3], [z1, z2, z3], Zbar):
    color = get_color(zbari)
    ax.plot(np.append(xi, xi[0]), np.append(yi, yi[0]), np.append(zi, zi[0]), color=color)

ax.set_zlabel('Z[m]',fontsize=20, labelpad=30)
ax.set_xlabel('X[m]',fontsize=20, labelpad=20)
ax.set_ylabel('Y[m]',fontsize=20, labelpad=20)
ax.tick_params(axis='z', pad=22)
ax.grid(True)
plt.tight_layout()
#plt.savefig('geometry.png')
plt.show()

#exit()
