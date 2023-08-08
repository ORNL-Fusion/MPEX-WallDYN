#!/usr/bin/env python3
"""
Compare N-erosion, Reflection influx and source flux to see where the oscillations come from
"""

import sys
sys.path.append("/Users/42d/code")
import matplotlib as mpl
import matplotlib.pyplot as plt
from MyScripts.quick_MPL import DataTable, configmpl, PlotMethod, Axes


def main():
    """
    Main entry plot
    :return:
    """
    ErosionFlux_of_N = DataTable(fpath='./results/ProtoEmpex_sol_states_ErosionFlux_of_N_on_wall_w90_vs_time.dat')
    Gamma_of_N = DataTable(fpath='./results/ProtoEmpex_sol_states_Gamma_of_N_on_wall_w90_vs_time.dat')
    ReflectedFlux_of_N = DataTable(fpath='./results/ProtoEmpex_sol_states_ReflectedFlux_of_N_on_wall_w90_vs_time.dat')
    RYield_of_n = DataTable(fpath='./results/ProtoEmpex_sol_states_RYield_of_n_1_on_wall_w90_vs_time.dat')
    SYieldBy_of_n_n = DataTable(fpath='./results/ProtoEmpex_sol_states_SYieldBy_of_n_n_1_on_wall_w90_vs_time.dat')
    Conc_of_N = DataTable(fpath='./results/ProtoEmpex_sol_states_Conc_of_N_on_wall_w90_vs_time.dat')
    TotalSource_of_N = DataTable(fpath='./results/ProtoEmpex_post_states_TotalSource_of_N_on_wall_w90_vs_time.dat')

    axes: Axes = None
    fig, axes = plt.subplots()

    # scalefunc = lambda data, name: 1.0/data.data(name).mean()
    scalefunc = lambda data, name: 1.0

    ErosionFlux_of_N.Plot2D(axes, 'Time', 'N_90', yscale=scalefunc(ErosionFlux_of_N, 'N_90'), label='Ero-Flux', linewidth=2.0)
    Gamma_of_N.Plot2D(axes, 'Time', 'N_90', yscale=scalefunc(Gamma_of_N, 'N_90'), label='Influx', linewidth=2.0)
    ReflectedFlux_of_N.Plot2D(axes, 'Time', 'N_90', yscale=scalefunc(ReflectedFlux_of_N, 'N_90'), label='Refl-Flux', linewidth=2.0)
    RYield_of_n.Plot2D(axes, 'Time', 'N_1_90', yscale=scalefunc(RYield_of_n, 'N_1_90'), yoff=None, label='Refl-Yield', linewidth=2.0)
    SYieldBy_of_n_n.Plot2D(axes, 'Time', 'n_n_1_90', yscale=scalefunc(SYieldBy_of_n_n, 'n_n_1_90'), label='Sputter-Yield', linewidth=2.0)
    Conc_of_N.Plot2D(axes, 'Time', 'N_90', yscale=scalefunc(Conc_of_N, 'N_90'), yoff=None, label='Conc.')
    TotalSource_of_N.Plot2D(axes, 'Time', 'N_90', yscale=scalefunc(TotalSource_of_N, 'N_90'), label='Tot. Src.', linewidth=2.0)

    axes.legend(loc='upper right').set_draggable(True, use_blit=True)
    # axes.set_ylim(bottom=0.9, top=1.2)
    # axes.set_xlim(left=22, right=24)

    plt.show()


if __name__ == '__main__':
    import traceback

    try:
        # testing()
        main()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' % (repr(Ex), str(Ex)))
        print('Exception details:')
        traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script finished successfully')
