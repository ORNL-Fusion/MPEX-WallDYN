"""
Compare the result of the SBE scan with the experimental data
"""
import sys
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from MyScripts.quick_MPL import DataTable, configmpl, PlotMethod
from MyScripts.SputterReflectionModels import Yield_Bohdansky

def main():
    """
    Main entry point
    :return:
    """
    configmpl()
    axes: Axes = None
    fig: Figure = None

    sbescan = DataTable(fpath='ScanSBE__RES.zip')
    # sbescan = DataTable(fpath='ScanSBE_TOTAL_RES.zip')
    sbescan.sort(('ee', 'oonosbe',  'oonalsbe', 'alonosbe', 'alonalsbe'))

    experiment = DataTable(fpath='../../..//total_yield_data_Al2O3.dat')
    experiment.sort(('Energy',))

    calcenergies = sbescan.data('ee')
    calcunique_ee = np.unique(calcenergies)
    totyields = sbescan.data('yal') + sbescan.data('yo')
    expenergies = experiment.data('Energy')
    exptotylds = experiment.data('Yield')

    fig, axes = plt.subplots()

    for ee in expenergies:
        if ee in calcunique_ee:
            sel = calcenergies == ee
            curyields = totyields[sel]
            curenergies = calcenergies[sel]
            DataTable.DoPlot(axes, curenergies, curyields, linewidth=0, marker='x', markersize='8', color='b', label='%.1lf' % ee)

    experiment.Plot2D(axes, 'Energy', 'Yield', label='Experiment', linewidth=0, marker='o', markersize='16', color='r')

    axes.legend(loc='upper right').set_draggable(True, use_blit=True)
    axes.set_xlabel('Energy (eV)')
    axes.set_ylabel('Sputter yield')

    fig, axes = plt.subplots()

    calcenergies = sbescan.data('ee')
    totyields = sbescan.data('yal') + sbescan.data('yo')
    cal = sbescan.data('cal')

    oonosbe = sbescan.data('oonosbe')
    oonalsbe = sbescan.data('oonalsbe')
    alonosbe = sbescan.data('alonosbe')
    alonalsbe = sbescan.data('alonalsbe')

    u_oonosbe = np.unique(oonosbe)
    u_oonalsbe = np.unique(oonalsbe)
    u_alonosbe = np.unique(alonosbe)
    u_alonalsbe = np.unique(alonalsbe)

    M1 = 2.0
    M2 = 32 + 27
    Z1 = 1
    Z2 = (2. / 5.) * 13 + (3. / 5.) * 16
    Q = 0.141
    Eth = 66.
    E0 = np.arange(experiment.data('Energy').min(), experiment.data('Energy').max(), 10.0)
    yldbohd = Yield_Bohdansky(E0, M1, M2, Z1, Z2, Q, Eth)

    def floatarray_same(farry: np.ndarray, val: float, prec=1.0E-4):
        """
        Find array entries matching a value
        :param farry:
        :param val:
        :param prec:
        :return:
        """
        return np.abs(farry - val) < prec

    absprect = 1.0E-2
    relprec = 2.0
    emax = 10000

    def norm_comp(model, exp):
        """
        Compare using a weighted norm
        :param res:
        :param val:
        :return:
        """
        return abs(model-exp)/(relprec * exp + absprect)

    def compare_to_expyields(yields, energies):
        """
        Compares an SBE config to experimental values
        :param yields:
        :param energies:
        :return:
        """
        if len(energies) == len(np.unique(energies)):
            curylderr = 0.0
            for ee, yld in zip(expenergies, exptotylds):
                if ee < emax:
                    ie = np.argmin(np.abs(energies - ee))
                    curylderr += norm_comp(yields[ie], yld) # abs(yields[ie] - yld)/(relprec * yld + absprect)
            return curylderr
        else:
            return None

    def compare_to_Bohdansky(yields, energies):
        """
        Compares an SBE config to experimental values
        :param yields:
        :param energies:
        :return:
        """
        if len(energies) == len(np.unique(energies)):
            curylderr = 0.0
            for ee, yld in zip(E0, yldbohd):
                if ee < emax:
                    ie = np.argmin(np.abs(energies - ee))
                    curylderr += norm_comp(yields[ie], yld) # abs(yields[ie] - yld)/(relprec * yld + absprect)
            return curylderr
        else:
            return None

    mincase = None
    mincase_bohd = None
    for oono in u_oonosbe:
        for oonal in u_oonalsbe:
            for alono in u_alonosbe:
                for alonal in u_alonalsbe:
                    same_oono = floatarray_same(oonosbe, oono)
                    same_oonal = floatarray_same(oonalsbe, oonal)
                    same_alono = floatarray_same(alonosbe, alono)
                    same_alonal = floatarray_same(alonalsbe, alonal)

                    same = np.logical_and(np.logical_and(same_oono, same_oonal), np.logical_and(same_alono, same_alonal))

                    if np.any(same):
                        curenergies = calcenergies[same]
                        curyields = totyields[same]
                        curcal = cal[same]
                        err = compare_to_expyields(curyields, curenergies)

                        if err is not None:
                            comp = (err, oono, oonal, alono, alonal, curyields, curenergies, curcal)
                            if mincase is not None:
                                if mincase[0] > err:
                                    mincase = comp
                            else:
                                mincase = comp
                        else:
                            print('WARNING::No calculation results for:')
                            print('\toono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % oono, oonal, alono, alonal)

                        err = compare_to_Bohdansky(curyields, curenergies)
                        if err is not None:
                            comp = (err, oono, oonal, alono, alonal, curyields, curenergies, curcal)
                            if mincase_bohd is not None:
                                if mincase_bohd[0] > err:
                                    mincase_bohd = comp
                            else:
                                mincase_bohd = comp
                        else:
                            print('WARNING::No calculation results for:')
                            print('\toono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % oono, oonal, alono, alonal)

    experiment.Plot2D(axes, 'Energy', 'Yield', label='Experiment', method=PlotMethod.logxy, linewidth=0, marker='o', markersize='16', color='r')

    print('Best fit to experiment: %.3lf' % mincase[0])
    oono, oonal, alono, alonal, curyields, curenergies, curcal = mincase[1:]
    print('\toono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % (oono, oonal, alono, alonal))
    DataTable.DoPlot(axes, curenergies, curyields, label='Best-Fit-Exp.', method=PlotMethod.logxy, linewidth=2, marker='x', markersize='16', color='g')
    title = 'Fit-Exp: oono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % (oono, oonal, alono, alonal)

    print('Best fit to Bohdansky: %.3lf' % mincase_bohd[0])
    oono, oonal, alono, alonal, curyields, curenergies, curcal = mincase_bohd[1:]
    print('\toono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % (oono, oonal, alono, alonal))
    DataTable.DoPlot(axes, curenergies, curyields, label='Best-Fit-Bohdansky', method=PlotMethod.logxy, linewidth=2, marker='x',
                     markersize='16', color='k')
    title += '\nFit-Bohdansky: oono: %.3lf, oonal: %.3lf, alono: %.3lf, alonal: %.3lf' % (oono, oonal, alono, alonal)

    DataTable.DoPlot(axes, E0, yldbohd, label='Bohdansky', method=PlotMethod.logxy, linewidth=2, marker=None,
                     markersize='16', color='m')

    fig.suptitle(title)
    axes.legend(loc='upper left').set_draggable(True, use_blit=True)
    axes.set_xlabel('Energy (eV)')
    axes.set_ylabel('Sputter yield')
    axes.set_xlim(left=10, right=1E4)
    axes.set_ylim(bottom=1E-3, top=0.1)

    c_axes = axes.twinx()
    oono, oonal, alono, alonal, curyields, curenergies, curcal = mincase[1:]
    DataTable.DoPlot(c_axes, curenergies, curcal, label='Best-Fit-Exp.', method=PlotMethod.lin, linewidth=2,
                     marker='^', markersize='16', color='g')
    oono, oonal, alono, alonal, curyields, curenergies, curcal = mincase_bohd[1:]
    DataTable.DoPlot(c_axes, curenergies, curcal, label='Best-Fit-Bohdansky', method=PlotMethod.lin, linewidth=2,
                     marker='<',
                     markersize='16', color='k')
    c_axes.legend(loc='lower right').set_draggable(True, use_blit=True)

    c_axes.set_ylabel('Al-SFC (at. frac.)')

    c_axes.set_ylim(bottom=0.4, top=1)

    plt.show()

    return 0

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