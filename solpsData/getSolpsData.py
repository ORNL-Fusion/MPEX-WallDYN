import scipy.io as sio
from scipy.constants import e
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import interp2d

def interpolate_var(R_, Z, var, R_new, Z_new):
    """
    Interpolate a variable from a 2D grid to a new 2D grid.
    :param R_: 1D array of R coordinates
    :param Z: 1D array of Z coordinates
    :param var: 2D array of variable to interpolate
    :param R_new: 1D array of new R coordinates
    :param Z_new: 1D array of new Z coordinates
    :return: var_new: 2D array of interpolated variable
    """
    R_flat = R_.flatten()
    Z_flat = Z.flatten()
    var_flat = var.flatten()

    # Create a function for bilinear interpolation
    f = interp2d(R_flat, Z_flat, var_flat,kind='linear')

    # Perform interpolation
    var_new = f(R_new, Z_new)


    return var_new


# # Load the MATLAB file
atom_mat_contents = sio.loadmat('/Users/42d/MPEX-GITR-WallDYN/solpsData/dab2.mat')
mat_contents = sio.loadmat('/Users/42d/MPEX-GITR-WallDYN/solpsData/matlab_SOLPS.mat')

# Access the variables from the MATLAB file
Case = mat_contents['Case']
Geo = Case[0][0]['Geo'].item()
LcRight = Geo['LcRight']
R = Case[0][0]['Geo'].item()['r2d_cen']
R_ = R[0][0]

# Extract the variables
Z = LcRight[:, 0] + 0.
# Plasma parameters
State = Case[0][0]['State'].item()
ne = State['ne'][:, 0][0]
te = State['te'][:, 0][0]/e 
ti = State['ti'][:, 0][0]/e
ni = State['na'][:, 0]
ui = State['ua'][:, 0]
u_fluid_neutral = ui[0][:,:,0]
u_deuterium_par = ui[0][:,:,1]
ni_neutral = ni[0][:,:,0]
ni_deuterium = ni[0][:,:,1]
B = Geo['bb'][:, 0]
# Z=Z[0]


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
