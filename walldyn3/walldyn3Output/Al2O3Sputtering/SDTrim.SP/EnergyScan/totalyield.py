#!/usr/bin/env python3
"""
Disect and output.dat and return total yield
"""
import os
import sys
from MyScripts.SDTRIMUtil import SDTRIM_Output, TRIINP

def main():
    """
    Main entry point
    :return:
    """
    srcdir = sys.argv[1]
    if os.path.isdir(sys.argv[1]):
        destfile = sys.argv[2]

        if os.path.exists(destfile):
            outp = SDTRIM_Output(os.path.join(srcdir, 'output.dat'))
            inp = TRIINP(os.path.join(srcdir, 'tri.inp'))

            with open(destfile, 'at') as fptr:
                curline = '%.3f\t%.6g' % (inp.m_e0[0], (outp.m_backsput_i[1] + outp.m_backsput_i[2])/ outp.m_proj_i[0])
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
