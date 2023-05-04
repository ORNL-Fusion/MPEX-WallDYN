#!/usr/bin/env python3
"""
Generate the case xml input to be processed by walldyn_setup
"""

import sys
sys.path.append('/Users/42d/code/pyWallDYN/GITR_Coupling')

#import GITR_Coupling

# Now you can use the functions and classes in GITR_Coupling


#from GITR_Coupling import genWDInput as wdi
import genWDInput as wdi
from pyWallDYN.pyYieldModel4 import yielmodel4db

def main():
    """
    Main entry point
    :return:
    """

    wallbase = '../data_o_al/fields'
    wallpar = wdi.WallParams()
    wallpar.parse(Tefp=os.path.join(wallbase, 'te.txt'),
                  Tifp=os.path.join(wallbase, 'ti.txt'),
                  potfp=os.path.join(wallbase, 'potential.txt'),
                  anglefp=os.path.join(wallbase, 'angle.txt'),
                  )
    wallpar.add_const_flux('D', 1, os.path.join(wallbase, 'flux.txt'))

    trigen = wdi.TriangleGenerator()
    trigen.parse('../data_o_al/gitrGeometryPointPlane3d.cfg')

    indexgroups = wdi.IndexGroupGenerator()
    indexgroups.parse('../data_o_al/surface/surface_inds_.txt')

    matrices = wdi.MigrationMatrices(('Al', 'N', 'W'), (4, 4, 4))
    matrices.add_matrix('Al', 0, '../data_o_al/matrices/matAl0+.dat')
    matrices.add_matrix('Al', 1, '../data_o_al/matrices/matAl1+.dat')
    matrices.add_matrix('Al', 2, '../data_o_al/matrices/matAl2+.dat')
    matrices.add_matrix('Al', 3, '../data_o_al/matrices/matAl3+.dat')

    # HACK recylce Al matrices for W with limited charge state to +3
    matrices.add_matrix('W', 0, '../data_o_al/matrices/matAl0+.dat')
    matrices.add_matrix('W', 1, '../data_o_al/matrices/matAl1+.dat')
    matrices.add_matrix('W', 2, '../data_o_al/matrices/matAl2+.dat')
    matrices.add_matrix('W', 3, '../data_o_al/matrices/matAl3+.dat')

    # HACK recylce O matrices for N with limited charge state to +4
    matrices.add_matrix('N', 0, '../data_o_al/matrices/matO0+.dat')
    matrices.add_matrix('N', 1, '../data_o_al/matrices/matO1+.dat')
    matrices.add_matrix('N', 2, '../data_o_al/matrices/matO2+.dat')
    matrices.add_matrix('N', 3, '../data_o_al/matrices/matO3+.dat')

    matrixnorm = wdi.load_matrixnorm('../data_o_al/matrices/surfacegroup_nlaunch.dat')

    elements = wdi.ElementData()
    elements.add_element('D', 1, 2.0, 0.04E30, 0.0)
    elements.add_element('Al', 13, 27.0, 0.06E30, 1.0)  # last arg max concentration
    elements.add_element('N', 7, 14.0, 0.062E30, 0.52)
    elements.add_element('W', 74, 182, 0.062E30, 1.0)  # TODO: VERIFY NUMBER DENSITY

    perwall = wdi.PerWallData("perwalldata.ods")

    tolerances={"rel":1.0E-4, "absadens":1.0E-6, "absflux":1.0E-6, "absconc":1.0E-6}

     
    ylddb = os.path.expandvars("/Users/42d/code/pyWallDYN/yieldmodel_4/BlinSPLINEDB_D_Al_N_W_Al_N_W/BlinSPLINEDB_d_al_n_w_on_al_n_w.XML")
    yld4 = yielmodel4db()
    yld4.loadXML(ylddb)

    caseID = "ProtoMPEX"
    inp = wdi.Store_GITR_in_WDInput(caseID=caseID,
                                    elementdata=elements,
                                    triangledata=trigen,
                                    wallpars=wallpar,
                                    indexgroups=indexgroups,
                                    matrices=matrices,
                                    matrixnorm=matrixnorm,
                                    perwall=perwall,
                                    yld4=yld4,
                                    tolerances=tolerances)

    # Write input to XML
    outpath = "GITR_ProtoMPEX_W_N_Al.xml"
    compressXML = 0
    with open(outpath, 'wb') as fptr:
        inp.StoreAsXML(fptr, RedepNumeric=True, compression=compressXML)
    print('Stored case @ %s' % outpath)

    # Store the control XML
    outdir = "./"
    outtimes = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    wdi.Store_ControlFile(caseID, outdir, outtimes, storesteps = True, testjacfdd = False, outpath = './results')


if __name__ == '__main__':
    import sys, traceback, os

    try:
        main()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' % (repr(Ex), str(Ex)))
        print('Exception details:')
        traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script finished successfully')
