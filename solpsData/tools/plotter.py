
# Plot the SOLPS data from the MATLAB file
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from  solpsParser import solps_data
# Load the netCDF data

mat_file_path = '../SOLPS_09092023/matlab_SOLPS.mat'
parsed_data = solps_data(mat_file_path)

# read in data
z = parsed_data['Z'][0]
ne = parsed_data['ne']
te = parsed_data['te']
ti = parsed_data['ti']
ni = parsed_data['ni'][0][:, :, 1]
u_fluid_neutral = parsed_data['u_fluid_neutral']
u_deuterium_par = parsed_data['u_deuterium_par']
ni_neutral = parsed_data['ni_neutral']
ni_deuterium = parsed_data['ni_deuterium']
B = parsed_data['B'][0][:, :, -1]
R_ = parsed_data['R']
x= R_

br = B
bz = B

xu = np.unique(x)
zu = np.unique(z)
print("x bounds", min(xu), max(xu))
print("Z bounds", min(zu), max(zu))

exit()
# 2D plot
fig, axs = plt.subplots(2,3, figsize=(20, 8))
fig.suptitle('SOLPS Data', fontsize=20)

# Plot br
cs = axs[0, 0].contourf(z, x, parsed_data['B'][0][:, :, -1])
axs[0, 0].set_xlabel('Z [m]')
axs[0, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 0])
colorbar.set_label('$B$ [T]')

## Plot te
cs = axs[0, 1].contourf(z, x, te)
axs[0, 1].set_xlabel('Z [m]')
axs[0, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 1])
colorbar.set_label('$T_e$ [eV]')
#
# Plot ni
cs = axs[1, 0].contourf(z, x, ni)
axs[1, 0].set_xlabel('Z [m]')
axs[1, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 0])
colorbar.set_label('$n_i$ [$m^{-3}$]')

## Plot vr
cs = axs[1, 1].contourf(z, x, ne)
axs[1, 1].set_xlabel('Z [m]')
axs[1, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 1])
#colorbar.set_label('$u_{fn}$ [m/s]')
colorbar.set_label('$n_e$ [$m^{-3}$]')
## Plot vt
cs = axs[1, 2].contourf(z, x, u_deuterium_par)
axs[1, 2].set_xlabel('Z [m]')
axs[1, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 2])
colorbar.set_label('$u_{D\parallel}$ [m/s]')

### Plot vz
cs = axs[0, 2].contourf(z, x, ti)
axs[0, 2].set_xlabel('Z [m]')
axs[0, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 2])
colorbar.set_label('$T_i$ [eV]')

# Additional plots can be added to axs[2, 1] and axs[2, 2] as needed

plt.tight_layout()
plt.savefig('solpsData.png')
plt.show()
