# Plot the SOLPS data from the MATLAB file
import matplotlib.pyplot as plt
from  solpsParser import solps_data

# Load the netCDF data
mat_file_path = '../solpsData/SOLPS_09092023/matlab_SOLPS.mat'
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

# get 1 d br 
import numpy as np 
plt.plot(z, bz[:,-1], label='A')

Z= np.loadtxt("Z.txt", unpack=True)
BZ= np.loadtxt("B.txt", unpack=True)
plt.plot(Z, BZ, label='S')
plt.legend()
plt.xlim(0.,5)
plt.ylim(0,1.8)
plt.grid(alpha=0.3)
plt.show()
exit()
# 2D plot
fig, axs = plt.subplots(2,3, figsize=(20, 10))
fig.suptitle('Data from Atul Kumar', fontsize=20)

# Plot br
cs = axs[0, 0].contourf(z, x, br.T)
axs[0, 0].set_xlabel('Z [m]')
axs[0, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 0])
colorbar.set_label('$Br$ [T]')

# Plot bz
cs = axs[0, 1].contourf(z, x, bz.T)
axs[0, 1].set_xlabel('Z [m]')
axs[0, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 1])
colorbar.set_label('$Bz$ [T]')

# Plot te
cs = axs[0, 2].contourf(z, x, te.T)
axs[0, 2].set_xlabel('Z [m]')
axs[0, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 2])
colorbar.set_label('$T_e$ [eV]')

# Plot ni
cs = axs[1, 0].contourf(z, x, ni.T)
axs[1, 0].set_xlabel('Z [m]')
axs[1, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 0])
colorbar.set_label('$n_i$ [$m^{-3}$]')

# Plot vr
cs = axs[1, 1].contourf(z, x, vr.T)
axs[1, 1].set_xlabel('Z [m]')
axs[1, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 1])
colorbar.set_label('$v_r$ [m/s]')

# Plot vt
cs = axs[1, 2].contourf(z, x, vz.T)
axs[1, 2].set_xlabel('Z [m]')
axs[1, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 2])
colorbar.set_label('$v_z$ [m/s]')

# # Plot vz
# cs = axs[2, 0].contourf(z, x, vz.T)
# axs[2, 0].set_xlabel('Z [m]')
# axs[2, 0].set_ylabel('R [m]')
# colorbar = fig.colorbar(cs, ax=axs[2, 0])
# colorbar.set_label('$v_z$ [m/s]')

# Additional plots can be added to axs[2, 1] and axs[2, 2] as needed

plt.tight_layout()
plt.savefig('solpsDataBAtul.png')
plt.show()
