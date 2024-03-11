import numpy as np 

import matplotlib.pyplot as plt


path2file = '/Users/42d/MPEX-WallDYN/gitr/cases/D_N_Al_W/results/'
data= np.loadtxt( path2file + 'ProtoEmpex_ppext_NetAdensChange_100.000.dat', skiprows=1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(data[:,0], data[:,1], 'r-', label='Al', linewidth=2)
plt.title('Net Adens Change')
plt.xlabel('wall index')
plt.ylabel('Net Adens Change')
plt.grid(alpha =0.3)
plt.legend()
plt.subplot(1,3,2)
plt.plot(data[:,0], data[:,2], 'b-', label='N', linewidth=2)
plt.title('Net Adens Change')
plt.xlabel('wall index')
plt.ylabel('Net Adens Change')
plt.grid(alpha =0.3)
plt.legend()

plt.subplot(1,3,3)
plt.plot(data[:,0], data[:,3], 'g-', label='W', linewidth=2)
plt.title('Net Adens Change')
plt.xlabel('wall index')
plt.grid(alpha =0.3)
plt.ylabel('Net Adens Change')
plt.legend()


plt.legend()
plt.tight_layout()
plt.savefig('NetAdensChange.png')
plt.show()