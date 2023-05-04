"""
Processes the output
"""

import traceback
import sys
import MyScripts.SDTRIMUtil as STU
import MyScripts.Error as Err
import os
import glob
import multiprocessing as mp
import datetime
import numpy as np
try:
    from ParamScan_rundirs import rundirs
except:
    rundirs = []

# Number of CPU's
gNumCores = 12

# Base name for output generation
gBaseName = 'ScanSBE_'

class ProcessPathParallel(mp.Process):
    TO_DELETE = ['E0_34_moments.dat','energy_analyse.dat', 'layerinp.dat']
    def __init__(self, task_queue, data_queue, failed_queue):
        """
        Call base class and store queue to process
        """
        mp.Process.__init__(self)
        self.m_task_queue = task_queue
        self.m_data_queue = data_queue
        self.m_failed_queue = failed_queue
    def run(self):
        """
        Process the tasks
        """
        next_task = self.m_task_queue.get()
        while next_task != None:            
            currundir = next_task
            path = currundir[4]
            print('\t->@PID: %d processing %s' %(os.getpid(), path))
            curres = ProcessPath(currundir, ProcessPathParallel.TO_DELETE, verbose = 0)
            if curres != None:
                self.m_data_queue.put(curres)
                self.m_failed_queue.put(None)
            else:
                self.m_data_queue.put(None)
                self.m_failed_queue.put(path)
            next_task = self.m_task_queue.get()            
        return

def ProcessPath(currundir, todelete, verbose = 0):
    """
    Processes the contents of a single rundir
    and returns the result as a tuple
    (Proj, Angle, Energy, <concvec>, SputterYield, ReflectionYield)
    """
    curpath = currundir[4]
    e031path = os.path.join(curpath, 'E0_31_target.dat')
    if os.path.exists(e031path):
        e031 = STU.E031File()
        e031.LoadFile(e031path, verbose=0)
        inp = STU.TRIINP(os.path.join(curpath, 'tri.inp'), verbose=0)
        matinp = STU.mat_surfb()
        matinp.LoadMatrix(os.path.join(curpath, 'mat_surfb.inp'))

        AlOnAlSBE = matinp.GetMatrixByElement('Al', 'Al')
        AlOnOSBE = matinp.GetMatrixByElement('Al', 'O')
        OOnAlSBE = matinp.GetMatrixByElement('O', 'Al')
        OonOSBe = matinp.GetMatrixByElement('O', 'O')
        ee = inp.m_e0[0]

        lh = e031.m_histories[-1]
        blh = e031.m_histories[-2]
        nprj = lh.m_num_proj[0] - blh.m_num_proj[0]
        nsput = np.asarray(lh.m_num_back_sput) - np.asarray(blh.m_num_back_sput)


        res = [AlOnAlSBE, AlOnOSBE, OOnAlSBE, OonOSBe, ee]
        syield = nsput / nprj
        syieldinteg =  np.asarray(lh.m_num_back_sput) / lh.m_num_proj[0]
        for yld in syield:
            res.append(yld)
        for yld in syieldinteg:
            res.append(yld)
        for sfc in lh.m_sfc:
            res.append(sfc)

        if verbose > 5: print('Trying to delete: %s' % (repr(todelete)))
        for entry in todelete:
            curdelpath = os.path.join(curpath, entry)
            if os.path.exists(curdelpath):
                if verbose > 5: print('Deleting %s' % (curdelpath))
                os.remove(curdelpath)

        return res
    else:
        return None

def testing():
    """
    Test singe case processing
    :return:
    """
    rundir = "/home/korcslayer/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/GITR_Coupling/Al2O3Sputtering/SDTrim.SP/DynEnergyScan/1000eV"
    curres = ProcessPath((None, None, None, None, rundir), [], verbose=0)
    print(curres)

def main():
    """
    Main entry point
    :return:
    """
    start = datetime.datetime.now()
    task_queue = mp.Queue()
    print('Standby while queuing %d runs' % (len(rundirs)))
    numruns = 0
    for entry in rundirs:
        curpath = entry[4]
        if os.path.exists(curpath):
            task_queue.put(entry)
            numruns = numruns + 1
        if (numruns % 1000) == 0:
            print('\t%d' % (numruns), end=' ')
    # Start the processes
    # Result queues
    data_queue = mp.Queue()
    failed_queue = mp.Queue()
    for p in range(gNumCores):
        task_queue.put(None)  # Terminator
        p = ProcessPathParallel(task_queue, data_queue, failed_queue)
        p.start()
    # Grab results
    print('PARALLEL DONE->Grabbing results')
    data = []
    failed = []
    for i in range(numruns):
        datares = data_queue.get()
        if datares != None:
            data.append(datares)
        failedres = failed_queue.get()
        if failedres != None:
            failed.append(failedres)
    end = datetime.datetime.now()
    print('Time used to process the paramtere scan: %.6g seconds' % ((end - start).total_seconds()))
    print('Number of failed runs: %d' % (len(failed)))
    fptr = open(gBaseName + 'FailedRuns.py', 'wt')
    fptr.write('rundirs = %s\n' % repr(failed))
    fptr.close()

    with open('%sTOTAL_RES.dat' % gBaseName, 'wt') as fptr:
        fptr.write("AlOnAlSBE\tAlOnOSBE\tOOnAlSBE\tOonOSBe\tee\tYD\tYAl\tYO\tCD\tCAL\tCO\n")
        for AlOnAlSBE, AlOnOSBE, OOnAlSBE, OonOSBe, ee, YD, YAL, YO, TYD, TYAL, TYO, CD, CAL, CO in data:
            fptr.write("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g\n" %
                       (AlOnAlSBE, AlOnOSBE, OOnAlSBE, OonOSBe, ee,
                        TYD, TYAL, TYO, CD, CAL, CO)
                       )


if __name__ == '__main__':
    try:
        # testing()
        main()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' %(repr(Ex), str(Ex)))
        if __debug__:
            print('Exception details:')
            traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script completed successfully')    
