import numpy as np
import matplotlib.pyplot as plt


# Extract the magnitude of material 'w'
def get_w_magnitude(material_data_row, material):
    return abs(material_data_row[str(material)])

def get_t_magnitude(material_data_row, material):
    return abs(material_data_row['Time'])

path ='/Users/42d/MPEX-WallDYN/gitr/cases/D_N_Al_W/results/'

material = ['N', 'W', 'Al']
material_data_N = np.genfromtxt(path + 'ProtoEmpex_sol_states_Conc_of_N_on_wall_w90_vs_time.dat', names=True)
material_data_Al = np.genfromtxt(path + 'ProtoEmpex_sol_states_Conc_of_Al_on_wall_w90_vs_time.dat', names=True)
material_data_W = np.genfromtxt(path + 'ProtoEmpex_sol_states_Conc_of_W_on_wall_w90_vs_time.dat', names=True)

conc_on_wall_w90 = [get_w_magnitude(row, 'W_90') for row in material_data_W]
conc_on_wall_N90 = [get_w_magnitude(row, 'N_90') for row in material_data_N]
conc_on_wall_Al90 = [get_w_magnitude(row, 'Al_90') for row in material_data_Al]

time = [get_t_magnitude(row, 'Al_90') for row in material_data_Al]

fig, ax = plt.subplots()

ax.plot(time, conc_on_wall_w90, label ='W')
ax.plot(time, conc_on_wall_N90, label ='N')
ax.plot(time, conc_on_wall_Al90, label ='Al')
plt.xlabel('time [s]')
plt.ylabel('concentration at wall index 90')
plt.legend()
plt.savefig('conc_wall_90_vs_time.png', dpi=90)
plt.show()
