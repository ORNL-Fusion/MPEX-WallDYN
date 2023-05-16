import scipy.io as sio

# Load the MATLAB file
mat_contents = sio.loadmat('/Users/42d/MPEX-GITR-WallDYN/solpsData/matlab_SOLPS.mat')

# Access the variables from the MATLAB file
Case = mat_contents['Case']
Geo = Case[0][0]['Geo'].item()
LcRight = Geo['LcRight']
# Extract the variables
Z = LcRight[:, 0] + 0.5
print("Z = ", Z[0].shape)
B = Geo['bb'][:, 0]
print("B = ", B[0].shape)

# Plasma parameters
# ne = Case[0][0]['ne'].item()
State = Case[0][0]['State'].item()
ne = State['ne'][:, 0]
print("ne = ", ne[0].shape)
te = State['te'][:, 0]
print("te = ", te[0].shape)
ti = State['ti'][:, 0]
print("ti = ", ti[0].shape)
ui = State['ua'][:,0]
print("ui = ", ui[0].shape)

# Neutral data on EIRENE grid 
indAtom = 0
Trineuts = Case[0][0]['Trineuts'].item()
n_atom =   Trineuts['a'].item()['n'][:, indAtom]
print("n_atom = ", n_atom[0].shape)
n_mol = Trineuts['m'].item()['n'][:, indAtom]
print("n_mol = ", n_mol[0].shape)

# print("Z = ", Z[0][:,-1])
# import matplotlib.pyplot as plt
# plt.plot(Z[0][:,-1], ne[0][:,-1], label='ne')
# # # plt.plot(Z, te, label='te')
# plt.show()

# Neutral data on B2 grid
# % %%%Neutral data on B2 grid 
mat_contents_B2 = sio.loadmat('/Users/42d/MPEX-GITR-WallDYN/solpsData/dab2.mat')
# n_atom_B2 =load('/Users/42d/MPEX-GITR-WallDYN/dab2.mat'); %%uploaded on github
# % n_mol_B2 =load('/Users/42d/MPEX-GITR-WallDYN/dmb2.mat');
# % T_atom_B2 = load('/Users/42d/MPEX-GITR-WallDYN/tab2.mat');
# % T_mol_B2 = load('/Users/42d/MPEX-GITR-WallDYN/tmb2.mat');