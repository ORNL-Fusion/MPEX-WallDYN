import scipy.io as sio
from scipy.constants import e

def solps_data(mat_file_path):
    """
    Parses SOLPS data from a MATLAB file.
    
    Args:
        mat_file_path (str): Path to the SOLPS data MATLAB file.

    Returns:
        dict: Parsed SOLPS data.
    """
    # Load the MATLAB file
    mat_contents = sio.loadmat(mat_file_path)
    
    # Access the variables from the MATLAB file
    case_data = mat_contents['Case']
    geo_data = case_data[0][0]['Geo'].item()
    lc_right = geo_data['LcRight']
    R = case_data[0][0]['Geo'].item()['r2d_cen'][0][0]

    # Extract the variables
    z_values = lc_right[:, 0]  # Z coordinates
    state_data = case_data[0][0]['State'].item()
    ne = state_data['ne'][:, 0][0]  # Electron density
    te = state_data['te'][:, 0][0] / e  # Electron temperature (eV)
    ti = state_data['ti'][:, 0][0] / e  # Ion temperature (eV)
    ni = state_data['na'][:, 0]  # Ion densities
    ui = state_data['ua'][:, 0]  # Ion flow velocities
    u_fluid_neutral = ui[0][:, :, 0]  # Fluid neutral flow velocity
    u_deuterium_par = ui[0][:, :, 1]  # Deuterium parallel flow velocity
    ni_neutral = ni[0][:, :, 0]  # Neutral densities
    ni_deuterium = ni[0][:, :, 1]  # Deuterium densities
    
# shifted_z_values = z_values + 0.5 ( see Sahinul SOLPS-ITER paper)
    z_values = z_values + 0.5
    data = {
        'R': R,
        'Z': z_values,
        'ne': ne,
        'te': te,
        'ti': ti,
        'ni': ni,
        'u_fluid_neutral': u_fluid_neutral,
        'u_deuterium_par': u_deuterium_par,
        'ni_neutral': ni_neutral,
        'ni_deuterium': ni_deuterium
    }
    
    return data
# end of file