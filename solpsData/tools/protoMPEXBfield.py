

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
def write_plasma_profile(profiles_filename: str, plasma_data: Dict[str, Any],  nZ, nR) -> None:
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
    # nZ, nR = plasma_data['br'].shape
    
    nr = rootgrp.createDimension("nR", nR)
    nz = rootgrp.createDimension("nZ", nZ)

    r = rootgrp.createVariable("gridR", "f8", ("nR"))
    z = rootgrp.createVariable("gridZ", "f8", ("nZ"))

    br = rootgrp.createVariable("br", "f8", ("nZ", "nR"))
    bt = rootgrp.createVariable("bt", "f8", ("nZ", "nR"))
    bz = rootgrp.createVariable("bz", "f8", ("nZ", "nR"))
    
    r[:] = plasma_data['r']
    z[:] = plasma_data['z']

    br[:] = plasma_data['br'].T
    bt[:] = plasma_data['bt'].T
    bz[:] = plasma_data['bz'].T
    rootgrp.close()

# Get magnetic field and velocities

path ='/Users/42d/MPEX-GITR-WallDYN/solpsData/SOLPS_09092023/'
br = np.loadtxt(path + "Br.txt", unpack=True).T
bz = np.loadtxt(path +"Bz.txt", unpack=True).T
rs  = np.loadtxt(path +"r.txt")
zs = np.loadtxt(path +"Z.txt")


# print("rs.shape", rs.shape)
# print(rs)
# exit()
ru = np.unique(rs)
zu = np.unique(zs)

Nz, Nr = br.shape
r = np.linspace(np.min(ru), np.max(ru), Nr)
z = np.linspace(np.min(zu), np.max(zu), Nz)

print(Nr, Nz)

print('br shape', br.shape)
print('bz shape', bz.shape)
print('rs shape', r.shape)
print('zs shape', z.shape)
# exit()

# ############## write data to dictionary and save to netcdf file ############################
plasma_data = {}
profiles_filename = 'protoMPEXBfield.nc'
plasma_data['r'] = r
plasma_data['z'] = z

# # plasma_data['br'] = br
plasma_data['br'] = br.T
plasma_data['bt'] = np.zeros_like(br.T)
plasma_data['bz'] = bz.T

write_plasma_profile(profiles_filename, plasma_data,  Nz, Nr)
