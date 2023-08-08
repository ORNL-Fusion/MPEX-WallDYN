"""
Runs the paramter scan previously setup
"""

import string
import os
import sys
import traceback
from subprocess import *
import shutil
import MyScripts.SDTRIMUtil
import MyScripts.Error
import time
import random
import multiprocessing as mp
from ParamScan_rundirs import rundirs


# Number of CPU's
gNumCores = 6

gRunScript='run-win-6-00.cmd'

class ProcessPathParallel(mp.Process):
    """
    Runs SDTRIM.SP in a rundir in parallel
    """
    def __init__(self, task_queue, res_queue):
        """
        Call base class and store queue to process
        """
        mp.Process.__init__(self)
        self.m_task_queue = task_queue        
        self.m_res_queue = res_queue        
    def run(self):
        """
        Process the tasks
        """
        next_task = self.m_task_queue.get()
        while next_task != None:            
            curpath = next_task
            print('\t->@PID: %d processing %s' %(os.getpid(), curpath))
            # Start SDTRIM.SP and wait for it to complete                      
            retcode = -42
            try:
                args =  gRunScript    
                PID = Popen(args, stdout=open(os.path.join(curpath, 'stdout.txt'), 'wt'), stderr=open(os.path.join(curpath, 'stderr.txt'), 'wt'), shell = True, cwd = curpath) 
                if PID != None:
                    print('%d>>>Waiting for PID %d' %(os.getpid(), PID.pid))
                    retcode = PID.wait()
                    print('%d>>>PID %d is done with code %d' %(os.getpid(), PID.pid, retcode))
            except OSError:                   
                print('%d>>>PID %d had allready completed' %(os.getpid(), PID.pid))                
            self.m_res_queue.put(retcode)
            next_task = self.m_task_queue.get()            
        return
    
if __name__ == '__main__':
    # Debug
    # rundirs = rundirs[:10]
    # os.chdir(r'C:\Users\skl\Documents\sdtrim\Model 3\\')        
    try:   
        # Copy runtime files
        tstart = time.process_time()
        numruns = 0
        # Copy runtime files & Run sdtrim in parallel on gNumCores cpu's
        # apply multi processing        
        print('Standby while queuing %d runs' %(len(rundirs)))
        task_queue = mp.Queue()        
        for entry in rundirs:
            src = os.path.join(os.getcwd(), gRunScript)
            dest = os.path.join(entry[4], gRunScript)
            shutil.copyfile(src, dest)
            numruns = numruns + 1 
            curpath = entry[4]                
            if os.path.exists(curpath):
                task_queue.put(curpath)  
            else:
                raise Exception('Encountered invalid run dir')
            if (numruns % 1000) == 0:
                print('\t%d' %(numruns), end=' ')
        # Start the processes    
        res_queue = mp.Queue() 
        for p in range(gNumCores):            
            task_queue.put(None) # Terminator
            p = ProcessPathParallel(task_queue, res_queue)
            p.start()
        # Wait for things to finish
        for i in range(numruns):
            res_queue.get()
        tend = time.process_time()
        print('Completed %d SDTRIM runs in (%6lf secs.)\n\t->DONE' %(numruns, tend-tstart))
    except Exception as Ex:
        print('Caught exception: %s' %(Ex))
        tb = sys.exc_info()[2]
        traceback.print_tb(tb)
    except:
        print('General exception occurred')
    else:
        print('Script completet successfully')
