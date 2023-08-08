"""
Setup the SBE parameter scan
"""
import os
import datetime
import numpy as np
import pprint
from MyScripts.SDTRIMUtil import TRIINP, E031File, mat_surfb

pp = pprint.PrettyPrinter(indent=4)

def main():
    """
    Main entry point
    :return:
    """
    tstart = datetime.datetime.now()
    # Read the template
    triinp = TRIINP("tri_template.inp")
    matinp = mat_surfb()
    matinp.LoadMatrix('mat_surfb.inp')

    AlOnAlSBEs = np.arange(0.1, 6.0, 1)
    AlOnOSBEs = np.arange(0.1, 20.0, 2)
    OOnAlSBEs = np.arange(0.1, 20.0, 2)
    OonOSBes = np.arange(0.1, 6.0, 1)
    energies = sorted([100, 1000, 170, 2000, 250, 300, 500])

    print('AlOnAlSBEs:')
    pp.pprint(AlOnAlSBEs)
    print('AlOnOSBEs:')
    pp.pprint(AlOnOSBEs)
    print('OOnAlSBEs:')
    pp.pprint(OOnAlSBEs)
    print('OonOSBes:')
    pp.pprint(OonOSBes)
    print('energies:')
    pp.pprint(energies)

    def mkoutp(curoutpath, lbl, id):
        curoutpath = os.path.join(curoutpath, '%s_%d' % (lbl, id))
        if (not os.path.exists(curoutpath)):
            print('Creating %s in %s' %
                  (curoutpath, os.getcwd()))
            os.mkdir(curoutpath)
        return curoutpath

    rundirs = []
    cntr = 0
    for iAlOnAlSBE, AlOnAlSBE in enumerate(AlOnAlSBEs):
        for iAlOnOSBE, AlOnOSBE in enumerate(AlOnOSBEs):
            for iOOnAlSBE, OOnAlSBE in enumerate(OOnAlSBEs):
                for iOonOSBe, OonOSBe in enumerate(OonOSBes):
                    for iee, ee in enumerate(energies):
                        curoutpath = './'
                        curoutpath = mkoutp(curoutpath, "AlOnAl", iAlOnAlSBE)
                        curoutpath = mkoutp(curoutpath, "AlOnO", iAlOnOSBE)
                        curoutpath = mkoutp(curoutpath, "OOnAl", iOOnAlSBE)
                        curoutpath = mkoutp(curoutpath, "OonO", iOonOSBe)
                        curoutpath = mkoutp(curoutpath, "Energy", iee)

                        triinp.m_e0[0] = ee

                        matinp.SetMatrixByElement('Al','Al', AlOnAlSBE)
                        matinp.SetMatrixByElement('Al', 'O', AlOnOSBE)
                        matinp.SetMatrixByElement('O', 'Al', OOnAlSBE)
                        matinp.SetMatrixByElement('O', 'O', OonOSBe)

                        triinp.StoreFile(os.path.join(curoutpath, 'tri.inp'))
                        matinp.StoreMatrix(os.path.join(curoutpath, 'mat_surfb.inp'))

                        rundirs.append(
                            [triinp.m_qu, 'D', triinp.m_e0[0] , triinp.m_alpha0[0], os.path.join(os.getcwd(), curoutpath)])

                        cntr += 1

    tend = datetime.datetime.now()
    # Dump rundir info to disk
    fptr = open('ParamScan_rundirs.py', 'wt')
    fptr.write('rundirs = %s' % (repr(rundirs)))
    fptr.close()
    print('Rundir build complete (%6lf secs.)' % (tend - tstart).total_seconds())
    print('Created %d cases' % cntr)

if __name__ == '__main__':
    import traceback, sys
    try:
        main()
    except Exception as Ex:
        print('Caught exception: %s' % (Ex))
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
    except:
        print('General exception occurred')
    else:
        print('Script completet successfully')