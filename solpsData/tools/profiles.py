
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

# load profiles.nc
data = nc.Dataset("profiles.nc", "r")

# Plot all profiles
r = data.variables['gridR'][:]
z = data.variables['gridZ'][:]

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

# plot ne te ni ti vr vz br bz 8 subplots

import matplotlib.pyplot as plt

## Plotting SOLPS
from  solpsParser import solps_data
mat_file_path = '/Users/42d/MPEX-GITR-WallDYN/solpsData/SOLPS_09092023/matlab_SOLPS.mat'
parsed_data = solps_data(mat_file_path)

# read in data
z = parsed_data['Z'][0]
ne = parsed_data['ne']
te = parsed_data['te']
ni = parsed_data['ni'][0][:, :, 1]
R_ = parsed_data['R']
x= R_


fig, axs = plt.subplots(1,2, figsize=(20, 8))
fig.suptitle('solps data', fontsize=20)

cs = axs[0].contourf(x,z,ne)
colorbar = fig.colorbar(cs, ax=axs[0])
axs[0].set_ylabel('Z [m]')
axs[0].set_xlabel('R [m]')
colorbar.set_label('ne [$cm^{-3}$]')

cs = axs[1].contourf(x,z,te)
colorbar = fig.colorbar(cs, ax=axs[1])
axs[1].set_ylabel('Z [m]')
axs[1].set_xlabel('R [m]')
colorbar.set_label('Te [eV]')

#cs = axs[1,0].contourf(xg,zg, br.T)
#colorbar = fig.colorbar(cs, ax=axs[1, 0])
#axs[1,0].set_ylabel('Z [m]')
#axs[1,0].set_xlabel('R [m]')
#colorbar.set_label('Br [T]')
#
#
#cs = axs[1,1].contourf(xg,zg, bz.T)
#colorbar = fig.colorbar(cs, ax=axs[1, 1])
#axs[1,1].set_ylabel('Z [m]')
#axs[1,1].set_xlabel('R [m]')
#colorbar.set_label('Bz [T]')

plt.tight_layout()
plt.show()


# ## Plot te

# cs =  axs[0,1].contourf(R_new, Z_new, bz_new)
# axs[0,1].set_ylabel('Z [m]')
# axs[0,1].set_xlabel('R [m]')
# colorbar = fig.colorbar(cs, ax=axs[0,1])
# colorbar.set_label('bz')




# # Plot br
# cs = axs[1,0].contourf(r, z, br)
# colorbar = fig.colorbar(cs, ax=axs[1, 0])
# axs[1,0].set_ylabel('Z [m]')
# axs[1,0].set_xlabel('R [m]')
# colorbar.set_label('br')


# ## Plot te

# cs =  axs[1,1].contourf(r, z, bz)
# axs[1,1].set_ylabel('Z [m]')
# axs[1,1].set_xlabel('R [m]')
# colorbar = fig.colorbar(cs, ax=axs[1,1])
# colorbar.set_label('bz')


# # 
# import netCDF4 as nc
# data = nc.Dataset("/Users/42d/Downloads/profilesProtoMPEX_base.nc", "r")
# # dict_keys(['x', 'z', 'ne', 'ni', 'te', 'ti', 'vr', 'vt', 'vz', 'br', 'bt', 'bz'])
# print(data.variables.keys())
# r=data.variables['x'][:]
# z=data.variables['z'][:]
# br = data.variables['br'][:]
# bz = data.variables['bz'][:]

# print("bz", bz)
# exit()
# # Plot br and bz as function of r and z

# fig, axs = plt.subplots(1,2, figsize=(10, 5))
# fig.suptitle('SOLPS Data', fontsize=20)
# cs = axs[0].contourf(r, z, br)
# colorbar = fig.colorbar(cs, ax=axs[0])
# axs[0].set_ylabel('Z [m]')
# axs[0].set_xlabel('R [m]')
# colorbar.set_label('br')

# cs =  axs[1].contourf(r, z, bz)
# axs[1].set_ylabel('Z [m]')
# axs[1].set_xlabel('R [m]')
# colorbar = fig.colorbar(cs, ax=axs[1])
# colorbar.set_label('bz')


# plt.show()
