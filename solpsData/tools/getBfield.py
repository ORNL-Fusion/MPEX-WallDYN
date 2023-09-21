import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

# load profiles.nc
# data = nc.Dataset("protoMPEXBfield.nc", "r")

data = nc.Dataset("profiles.nc", "r")

# Plot all profiles
ru = data.variables['gridR'][:]
zu = data.variables['gridZ'][:]

ne = data.variables['ne'][:]
te = data.variables['te'][:]

ni = data.variables['ni'][:]
ti = data.variables['ti'][:]

vr = data.variables['vr'][:]
vz = data.variables['vz'][:]

vt = data.variables['vt'][:]

br = data.variables['br'][:]
bt = data.variables['bt'][:]
bz = data.variables['bz'][:]

# ru=data.variables['gridR'][:]
# zu=data.variables['gridZ'][:]

# br = data.variables['br'][:]
# bt = data.variables['bt'][:]
# bz = data.variables['bz'][:]

# get shape
print ("shape", br.shape)
print ("shape", bt.shape)
print ("shape", bz.shape)
print ("shape", ru.shape)
print ("shape", zu.shape)

# exit()
r,z = np.meshgrid(ru, zu)
from matplotlib import pyplot as plt
fig, axs = plt.subplots(1,3, figsize=(10, 10))
fig.suptitle('SOLPS Data', fontsize=20)
cs = axs[0].contourf(r, z, te)
colorbar = fig.colorbar(cs, ax=axs[0])
axs[0].set_ylabel('Z [m]')
axs[0].set_xlabel('X [m]')
colorbar.set_label('br')

cs = axs[1].contourf(r, z, bt)
colorbar = fig.colorbar(cs, ax=axs[1])
axs[1].set_ylabel('Z [m]')
axs[1].set_xlabel('X [m]')
colorbar.set_label('br')


cs = axs[2].contourf(r, z, bz)
colorbar = fig.colorbar(cs, ax=axs[2])
axs[2].set_ylabel('Z [m]')
axs[2].set_xlabel('X [m]')
colorbar.set_label('br')


plt.show()


path ='/Users/42d/MPEX-GITR-WallDYN/solpsData/SOLPS_09092023/'
br = np.loadtxt(path + "Br.txt", unpack=True).T
bz = np.loadtxt(path +"Bz.txt", unpack=True).T
rs  = np.loadtxt(path +"r.txt")
zs = np.loadtxt(path +"Z.txt")

x= rs
z= zs
bt = bz
from matplotlib import pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('SOLPS Data', fontsize=20)
cs = axs[0,0].contourf(x, z, br)
colorbar = fig.colorbar(cs, ax=axs[0,0])
axs[0,0].set_ylabel('Z [m]')
axs[0,0].set_xlabel('X [m]')
colorbar.set_label('br')

cs =  axs[0,1].contourf(x, z, bz)
axs[0,1].set_ylabel('Z [m]')
axs[0,1].set_xlabel('X [m]')
colorbar = fig.colorbar(cs, ax=axs[0,1])
colorbar.set_label('bz')

cs = axs[1,0].contourf(x, z, bt)
colorbar = fig.colorbar(cs, ax=axs[1, 0])
axs[1,0].set_ylabel('Z [m]')
axs[1,0].set_xlabel('X [m]')
colorbar.set_label('bt')

plt.show()