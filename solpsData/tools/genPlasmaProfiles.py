
"""
This files writes plasma profile for GITR
"""
import netCDF4 as nc
import numpy as np
from   typing import Tuple, Dict, Any
import os
from   scipy.interpolate import interp2d
from   solpsParser import solps_data
from scipy.interpolate import RegularGridInterpolator, griddata
#

def project_parallel_variable_xyz(v_parallel_total,br,bphi,bz,nR,nZ):
    b_total = np.sqrt(np.multiply(br,br) + np.multiply(bphi,bphi) + np.multiply(bz,bz))

    vr = np.divide(np.multiply(br,v_parallel_total),b_total)
    vt = np.divide(np.multiply(bphi,v_parallel_total),b_total)
    vz = np.divide(np.multiply(bz,v_parallel_total),b_total)

    return vr,vt,vz
#
#
def interpolate_var(R_, Z, var, R_new, Z_new):
    """
    Interpolate a variable from a 2D grid to a new 2D grid.
    """
    interpolating_function = RegularGridInterpolator((Z[:, 0], R_[0, :]), var)
    RZ_new = np.array(np.meshgrid(Z_new, R_new)).T.reshape(-1, 2)
    var_new = interpolating_function(RZ_new).reshape(len(Z_new), len(R_new))
    return var_new
#
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

    vr = rootgrp.createVariable("vr", "f8", ("nZ", "nR"))
    vz = rootgrp.createVariable("vz", "f8", ("nZ", "nR"))
    vt = rootgrp.createVariable("vt", "f8", ("nZ", "nR"))

    br = rootgrp.createVariable("br", "f8", ("nZ", "nR"))
    bt = rootgrp.createVariable("bt", "f8", ("nZ", "nR"))
    bz = rootgrp.createVariable("bz", "f8", ("nZ", "nR"))
    
    r[:] = plasma_data['r']
    z[:] = plasma_data['z']
    ne[:] = plasma_data['ne'].T
    te[:] = plasma_data['te'].T
    ni[:] = plasma_data['ni'].T
    ti[:] = plasma_data['ti'].T

    vi[:] = plasma_data['vi'].T

    vr[:] = plasma_data['vr'].T
    vz[:] = plasma_data['vz'].T
    vt[:] = plasma_data['vt'].T

    br[:] = plasma_data['br'].T
    bt[:] = plasma_data['bt'].T
    bz[:] = plasma_data['bz'].T
    rootgrp.close()
#
#
# Set path to MATLAB file
mat_file_path = '../SOLPS_09092023/matlab_SOLPS.mat'
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
R_ = parsed_data['R']

nZ,nR = R_.shape


# ############## check shape of data ############################
# # Check shape of data and print symbols and shapes for debugging
# variables = [Z, ne, te, ti, ni, u_fluid_neutral, u_deuterium_par, ni_neutral, ni_deuterium, R_]
# for symbol, variable in zip(['Z', 'ne', 'te', 'ti', 'ni', 'u_fluid_neutral', 'u_deuterium_par', 'ni_neutral', 'ni_deuterium', 'R_'], variables):
#     print(symbol, variable.shape)

# Get magnetic field and velocities

path ='/Users/42d/MPEX-GITR-WallDYN/solpsData/SOLPS_09092023/'
br = np.loadtxt(path + "Br.txt", unpack=True).T
bz = np.loadtxt(path +"Bz.txt", unpack=True).T
rs  = np.loadtxt(path +"r.txt")
zs = np.loadtxt(path +"Z.txt")

R_new = np.linspace(min(np.unique(rs)), max(np.unique(rs)), nR)
Z_new = np.linspace(min(np.unique(zs)), max(np.unique(zs)), nZ)

br_new = interpolate_var(rs, zs, br, R_new, Z_new) 
bz_new = interpolate_var(rs, zs, bz, R_new, Z_new)
bt_new = np.zeros_like(br_new)

vr,vt,vz = project_parallel_variable_xyz(u_deuterium_par,br_new, bt_new, bz_new,nR,nZ)

# ############## write data to dictionary and save to netcdf file ############################
plasma_data = {}
profiles_filename = 'profiles.nc'
plasma_data['r'] = R_new
plasma_data['z'] = Z_new
plasma_data['ne'] = ne.T 
plasma_data['te'] = te.T
plasma_data['ni'] = ni.T
plasma_data['ti'] = ti.T

plasma_data['vi'] = u_deuterium_par.T
plasma_data['vr'] = vr.T
plasma_data['vz'] = vz.T
plasma_data['vt'] = vt.T

plasma_data['br'] = br_new.T
plasma_data['bt'] = np.zeros_like(br_new.T)
plasma_data['bz'] = bz_new.T

write_plasma_profile(profiles_filename, plasma_data)

