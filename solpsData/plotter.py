
# Plot the SOLPS data from the MATLAB file
#
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e
from solpsParser import solps_data
#
#
# Set path to MATLAB file
mat_file_path = '/Users/42d/MPEX-GITR-WallDYN/solpsData/matlab_SOLPS.mat'
# Load the MATLAB file
parsed_data = solps_data(mat_file_path)
# read in data
Z = parsed_data['Z']
ne = parsed_data['ne']
te = parsed_data['te']
ti = parsed_data['ti']
ni = parsed_data['ni']
u_fluid_neutral = parsed_data['u_fluid_neutral']
u_deuterium_par = parsed_data['u_deuterium_par']
ni_neutral = parsed_data['ni_neutral']
ni_deuterium = parsed_data['ni_deuterium']
B = parsed_data['B']
R_ = parsed_data['R']

# 2D plot
fig, axs = plt.subplots(2,3, figsize=(20, 10))

# Plot te
cs = axs[0, 0].contourf(Z[0], R_, te)
axs[0, 0].set_xlabel('Z [m]')
axs[0, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 0], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$T_e$ [eV]')

# t ti
cs = axs[0, 1].contourf(Z[0], R_, ti)
axs[0, 1].set_xlabel('Z [m]')
axs[0, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 1], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$T_i$ [eV]')

# Plot ne
cs = axs[1, 0].contourf(Z[0], R_, ne)
axs[1, 0].set_xlabel('Z [m]')
axs[1, 0].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 0], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$n_e$  [$ m^{-3}$]')

# Plot B
B= B[0][:, :, -1]
cs = axs[1, 1].contourf(Z[0], R_, B)
axs[1, 1].set_xlabel('Z [m]')
axs[1, 1].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 1], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$B$ [T]')

# Plot u_fluid_neutral u_deuterium_par
cs = axs[1, 2].contourf(Z[0], R_, u_deuterium_par)
axs[1, 2].set_xlabel('Z [m]')
axs[1, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[1, 2], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$u^{\parallel}_{D^{+}}$  [m/s]')

cs = axs[0, 2].contourf(Z[0], R_, ni_deuterium)
axs[0, 2].set_xlabel('Z [m]')
axs[0, 2].set_ylabel('R [m]')
colorbar = fig.colorbar(cs, ax=axs[0, 2], ticks=matplotlib.ticker.MaxNLocator(nbins=5))
colorbar.set_label('$n_{D^{+}}$ [$ m^{-3}$]')

plt.tight_layout()
plt.savefig('solpsData.png')
plt.show()
