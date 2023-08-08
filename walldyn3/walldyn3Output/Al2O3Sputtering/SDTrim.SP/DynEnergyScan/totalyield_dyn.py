#!/usr/bin/env python3
"""
Disect and output.dat and return total yield
"""
import os
import sys

import numpy as np
from MyScripts.SDTRIMUtil import TRIINP, E031File, ParticleHistory_505

def main():
    """
    Main entry point
    :return:
    """

    if os.path.isdir(sys.argv[1]) and os.path.exists(sys.argv[2]):
        srcdir = sys.argv[1]
        destfile = sys.argv[2]

        if os.path.exists(destfile):
            e031 = E031File()
            e031.LoadFile(os.path.join(srcdir, 'E0_31_target.dat'))
            inp = TRIINP(os.path.join(srcdir, 'tri.inp'))

            lh: ParticleHistory_505 = e031.m_histories[-1]
            blh: ParticleHistory_505 = e031.m_histories[1]
            nprj = lh.m_num_proj[0] - blh.m_num_proj[0]
            nsput = np.asarray(lh.m_num_back_sput) - np.asarray(blh.m_num_back_sput)
            lh.m_dp
            with open(destfile, 'at') as fptr:
                curline = '%.3f\t%.6g\t%.6g\t%.6g\t%.6g\t%.6g' % (inp.m_e0[0], nsput.sum()/nprj, lh.m_sfc[1], lh.m_sfc[2], lh.m_sbe[1], lh.m_sbe[2])
                print(curline)
                fptr.write(curline + '\n')
        else:
            raise Exception('Destination file %s does not exist' % destfile)
    else:
        raise Exception('Source directory %s does not exist' % srcdir)


if __name__ == '__main__':
    import traceback

    try:
        main()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' % (repr(Ex), str(Ex)))
        print('Exception details:')
        traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script finished successfully')
