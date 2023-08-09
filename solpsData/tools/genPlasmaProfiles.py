
"""
This files writes plasma profile for GITR
"""
import netCDF4 as nc
import numpy as np
from   typing import Tuple, Dict, Any
import os
from   scipy.interpolate import interp2d
from   solpsParser import solps_data
#
# Define the plasma data dictionary
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
#
def write_plasma_profile(profiles_filename: str, plasma_data: Dict[str, Any]) -> None:
    """Writes plasma profile data to a NetCDF file.

    Args:
        profiles_filename: The name of the NetCDF file to write.
        plasma_data: A dictionary containing the following keys: r, z, ne, te, ve, ni, ti, vi, br, bt, bz.
            The values associated with each key are 2D arrays of plasma parameters, where each row corresponds
            to a specific r value and each column corresponds to a specific z value.
    """
    if os.path.exists(profiles_filename):
        os.remove(profiles_filename)
    #
    rootgrp = nc.Dataset(profiles_filename, "w", format="NETCDF4")
    nR, nZ = plasma_data['ne'].shape
    nr = rootgrp.createDimension("nR", nR)
    nz = rootgrp.createDimension("nZ", nZ)
    r = rootgrp.createVariable("gridR", "f8", ("nR"))
    z = rootgrp.createVariable("gridZ", "f8", ("nZ"))
    ne = rootgrp.createVariable("ne", "f8", ("nZ", "nR"))
    te = rootgrp.createVariable("te", "f8", ("nZ", "nR"))
    ni = rootgrp.createVariable("ni", "f8", ("nZ", "nR"))
    ti = rootgrp.createVariable("ti", "f8", ("nZ", "nR"))
    vi = rootgrp.createVariable("vi", "f8", ("nZ", "nR"))
    vpar = rootgrp.createVariable("Vpara", "f8", ("nZ", "nR"))
    br = rootgrp.createVariable("br", "f8", ("nZ", "nR"))
    bt = rootgrp.createVariable("bt", "f8", ("nZ", "nR"))
    bz = rootgrp.createVariable("bz", "f8", ("nZ", "nR"))
    #
    r[:] = plasma_data['r']
    z[:] = plasma_data['z']
    ne[:] = plasma_data['ne'].T
    te[:] = plasma_data['te'].T
    ni[:] = plasma_data['ni'].T
    ti[:] = plasma_data['ti'].T
    vi[:] = plasma_data['vi'].T
    vpar[:] = 0.0
    br[:] = plasma_data['br'].T
    bt[:] = 0.0
    bz[:] = plasma_data['bz'].T
    rootgrp.close()
#
#
# Set path to MATLAB file
mat_file_path = 'matlab_SOLPS.mat'
# Load the MATLAB file
parsed_data = solps_data(mat_file_path)
# read in data
Z = parsed_data['Z'][0]
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

############## check shape of data ############################
# Check shape of data and print symbols and shapes for debugging
variables = [Z, ne, te, ti, ni, u_fluid_neutral, u_deuterium_par, ni_neutral, ni_deuterium, B, R_]
for symbol, variable in zip(['Z', 'ne', 'te', 'ti', 'ni', 'u_fluid_neutral', 'u_deuterium_par', 'ni_neutral', 'ni_deuterium', 'B', 'R_'], variables):
    print(symbol, variable.shape)

############## write plasma profile ############################
nZ,nR = R_.shape
ru = np.unique(R_)
zu = np.unique(Z)

r =  np.linspace(min(ru),max(ru),nR)
z =  np.linspace(min(zu),max(zu),nZ)


plasma_data = {}
profiles_filename = 'profiles.nc'
plasma_data['r'] = r
plasma_data['z'] = z
plasma_data['ne'] = ne.T 
plasma_data['te'] = te.T
plasma_data['ni'] = ni.T
plasma_data['ti'] = ti.T
plasma_data['vi'] = u_deuterium_par.T
plasma_data['br'] = B.T
plasma_data['bt'] = np.zeros_like(B.T)
plasma_data['bz'] = B.T
#
write_plasma_profile(profiles_filename, plasma_data)

