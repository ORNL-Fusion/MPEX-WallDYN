#!/usr/bin/env python3

"""
Compare current optimized case to exp. data
"""

import sys
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from MyScripts.quick_MPL import DataTable, configmpl, PlotMethod
from MyScripts.SputterReflectionModels import Yield_Bohdansky
from MyScripts.SDTRIMUtil import mat_surfb
from MyScripts.SputterReflectionModels import Yield_Bohdansky

def main():
    """
    Main entry point
    :return:
    """
    configmpl()
    axes: Axes = None
    fig: Figure = None

    sbemat = mat_surfb()
    sbemat.LoadMatrix('mat_surfb.inp')
    oono = sbemat.GetMatrixByElement('o','o')
    oonal = sbemat.GetMatrixByElement('o','al')
    alono = sbemat.GetMatrixByElement('al','o')
    alonal = sbemat.GetMatrixByElement('al','al')
    title = 'oono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % (oono, oonal, alono, alonal)

    calc = DataTable(fpath='./sdtrim_res.dat')
    calc.sort(('Energy',))

    experiment = DataTable(fpath='../../../total_yield_data_Al2O3.dat')
    experiment.sort(('Energy',))

    M1 = 2.0
    M2 = 32 + 27
    Z1 = 1
    Z2 = (2. / 5.) * 13 + (3. / 5.) * 16
    Q = 0.141
    Eth = 66.
    E0 = np.arange(experiment.data('Energy').min(), experiment.data('Energy').max(), 10.0)
    yldbohd = Yield_Bohdansky(E0, M1, M2, Z1, Z2, Q, Eth)

    fig, axes = plt.subplots()

    experiment.Plot2D(axes, 'Energy', 'Yield', label='Experiment', method=PlotMethod.logxy, linewidth=0, marker='o',
                      markersize='16', color='r')
    calc.Plot2D(axes, 'Energy', 'Yields', label='SDTrim.SP', method=PlotMethod.logxy, linewidth=0, marker='x',
                      markersize='16', color='g')
    DataTable.DoPlot(axes, E0, yldbohd, label='Bohdansky', method=PlotMethod.logxy, linewidth=2, marker=None,
                      markersize='16', color='k')
    c_axes = axes.twinx()
    calc.Plot2D(c_axes, 'Energy', 'CAl', label='SDTrim.SP', method=PlotMethod.lin, linewidth=2, marker='<',
                      markersize='16', color='b')

    axes.legend(loc='upper left').set_draggable(True, use_blit=True)
    axes.set_xlabel('Energy (eV)')
    axes.set_ylabel('Sputter yield')
    axes.set_xlim(left=100, right=1E4)
    axes.set_ylim(bottom=1E-3, top=0.1)

    c_axes.legend(loc='lower right').set_draggable(True, use_blit=True)
    c_axes.set_ylabel('Al-SFC (at. frac.)')
    c_axes.set_ylim(bottom=0.4, top=1)

    fig, axes = plt.subplots()

    experiment.Plot2D(axes, 'Energy', 'Yield', label='Experiment', method=PlotMethod.logxy, linewidth=0, marker='o',
                      markersize='16', color='r')
    calc.Plot2D(axes, 'Energy', 'Yields', label='SDTrim.SP', method=PlotMethod.logxy, linewidth=0, marker='x',
                markersize='16', color='g')
    c_axes = axes.twinx()
    calc.Plot2D(c_axes, 'Energy', 'sbeal', label='SBE-Al', method=PlotMethod.lin, linewidth=2, marker='<',
                markersize='16', color='b')
    calc.Plot2D(c_axes, 'Energy', 'sbeo', label='SBE-O', method=PlotMethod.lin, linewidth=2, marker='>',
                markersize='16', color='m')

    axes.legend(loc='upper left').set_draggable(True, use_blit=True)
    axes.set_xlabel('Energy (eV)')
    axes.set_ylabel('Sputter yield')
    axes.set_xlim(left=100, right=1E4)
    axes.set_ylim(bottom=1E-3, top=0.1)

    c_axes.legend(loc='lower right').set_draggable(True, use_blit=True)
    c_axes.set_ylabel('SBE (eV)')
    c_axes.set_ylim(bottom=0, top=20)

    fig.suptitle(title)

    plt.show()

if __name__ == '__main__':
    import traceback
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