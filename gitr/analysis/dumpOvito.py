import numpy as np
import netCDF4 as nc
import csv

#
#data=nc.Dataset("/Users/42d/Downloads/atul_impacts/output/history.nc")
data=nc.Dataset("output/history.nc")

print(data)

nP, nT = data['x'][:].shape
nPoints=nP
#print(nP, nT)
#freq=10000
#nT=nT

x=data['x'][...]
y=data['y'][...]
z=data['z'][...]
vx=data['vx'][...]
vy=data['vy'][...]
vz=data['vz'][...]
xu = np.unique(x)
yu = np.unique(y)
zu = np.unique(z)
xmax =50.1
xmin =-50.1
ymax =50.1
ymin =-50.1
zmax =50.1
zmin =-50.1

print("x", min(xu), max(xu))
print("y", min(yu), max(yu))
print("z", min(zu), max(zu))


#exit()
#print('x',x, 'y', y, 'z',z)
charge=data['charge'][:]
species = np.unique(charge)
#print('charge',species)

Production_timeFile = "dump.gitr"
#nspecies=np.arange(1,len(v_i)+1,1)
with open(Production_timeFile, 'w') as testfile:
        csv_writer = csv.writer(testfile,delimiter=' ')
#         csv_writer = csv.writer(testfile,delimiter=' ',quoting=csv.QUOTE_NONE,quotechar="'",doublequote=True) nP, nT
        for i in range(nT):
            csv_writer.writerow(['ITEM:', 'TIMESTEP'])
            csv_writer.writerow([i])
            csv_writer.writerow(['ITEM:', 'NUMBER', 'OF', 'ATOMS'])
            csv_writer.writerow([nPoints])
            csv_writer.writerow(['ITEM:', 'BOX', 'BOUNDS', 'p', 'p', 'p'])
            csv_writer.writerow([xmin, xmax])
            csv_writer.writerow([ymin, ymax])
            csv_writer.writerow([zmin, zmax])
            csv_writer.writerow(['ITEM:','ATOMS', 'id','type','x', 'y','z','vx', 'vy','vz'])
#            data=[x[j,i] for j in range(1,nPoints)]
#            for i in range(0,len(x_),100):
#        csv_writer.writerow([data[0][i],data[1][i],data[2][i]])
            for j in range(0,nPoints):
                    type=abs(charge[j,i])+1
                    print("type", type)
                    nbr= np.random.randint(1,1e10,1).tolist()
#                    print('charge', charge[j,i])
                    csv_writer.writerow([j+1+i,int(type),x[j,i],y[j,i],z[j,i], vx[j,i],vy[j,i],vz[j,i]])
###
#            for j in range(nPoints):
#                csv_writer.writerow([x[j,i]]])
##                csv_writer.writerow([j+1,type,x[i,j],y[i,j],z[i,j]])




#
#Production_timeFile = "/Users/42d/sparta/examples/ambi/test/data.txt"
#nspecies=np.arange(1,len(v_i)+1,1)
#with open(Production_timeFile, 'w') as testfile:
#        csv_writer = csv.writer(testfile,delimiter=' ')
##         csv_writer = csv.writer(testfile,delimiter=' ',quoting=csv.QUOTE_NONE,quotechar="'",doublequote=True)
#        csv_writer.writerow(['ITEM:', 'TIMESTEP'])
#        csv_writer.writerow([0])
#        csv_writer.writerow(['ITEM:', 'NUMBER', 'OF', 'ATOMS'])
#        csv_writer.writerow([len(v_i)])
#        csv_writer.writerow(['ITEM:', 'BOX', 'BOUNDS', 'pp', 'pp', 'pp'])
#        csv_writer.writerow([1.8, 3.3])
#        csv_writer.writerow([-1, 1])
#        csv_writer.writerow([-0.01, 0.01])
#        csv_writer.writerow(['ITEM:','ATOMS', 'id','type','x', 'y','z','vx', 'vy','vz'])
## Write the ions
#        eps=1.e-6
#        for i in range(len(v_i)):
#            csv_writer.writerow([i+1,1,rdak[i],zdak[i],0.0,vel_r[i],vel_z[i],vel_t[i]])
## # Write the electrons
##         for i in range(len(v_i)):
##             csv_writer.writerow([i+1+len(v_i),2,rdak[i]+eps,zdak[i]+eps,0.0,vel_re[i],vel_ze[i],vel_te[i]])
#
#x = np.random.randint(1,1000,(pts,shape[0])).tolist()

#
##
#data=nc.Dataset("output/history.nc")
#print(data)
#
#nP, nT = data['x'][:].shape
#nPoints=nP
##print(nP, nT)
##freq=10000
##nT=nT
#
#x=data['x'][...]
#y=data['y'][...]
#z=data['z'][...]
#
##print('x',x, 'y', y, 'z',z)
#charge=data['charge'][:]
#species = np.unique(charge)
##print('charge',species)
#
#Production_timeFile = "dump.gitr"
##nspecies=np.arange(1,len(v_i)+1,1)
#with open(Production_timeFile, 'w') as testfile:
#        csv_writer = csv.writer(testfile,delimiter=' ')
##         csv_writer = csv.writer(testfile,delimiter=' ',quoting=csv.QUOTE_NONE,quotechar="'",doublequote=True) nP, nT
#        for i in range(nT):
#            csv_writer.writerow(['ITEM:', 'TIMESTEP'])
#            csv_writer.writerow([i])
#            csv_writer.writerow(['ITEM:', 'NUMBER', 'OF', 'ATOMS'])
#            csv_writer.writerow([nPoints])
#            csv_writer.writerow(['ITEM:', 'BOX', 'BOUNDS', 'p', 'p', 'p'])
#            csv_writer.writerow([1.5, 3.3])
#            csv_writer.writerow([-0.3, 0.3])
#            csv_writer.writerow([-1, 1.])
#            csv_writer.writerow(['ITEM:','ATOMS', 'id','type','x', 'y','z'])
##            data=[x[j,i] for j in range(1,nPoints)]
##            for i in range(0,len(x_),100):
##        csv_writer.writerow([data[0][i],data[1][i],data[2][i]])
#            for j in range(0,nPoints):
#                    type=abs(int(charge[j,i]))+1
#                    nbr= np.random.randint(1,1e10,1).tolist()
##                    print('charge', charge[j,i])
#                    csv_writer.writerow([j+1+i,type,x[j,i],y[j,i],z[j,i]])
####
##            for j in range(nPoints):
##                csv_writer.writerow([x[j,i]]])
###                csv_writer.writerow([j+1,type,x[i,j],y[i,j],z[i,j]])
#



