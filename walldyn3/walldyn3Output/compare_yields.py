#!/usr/bin/env python3

"""
Compare yields as calculated at runtime with the ones from the SDTrim.SP parameter scan
"""
import os.path

import numpy as np
from MyScripts.quick_MPL import DataTable, PlotMethod, configmpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
import pyWallDYN.yieldmodel_3.load_model3db as load_model3db

class syieldby(object):
    """
    Parses  and SyieldBy post processing file
    """
    syieldby_header_entries = ('WallIDX', 'Tgt', 'Prj', 'Chrg', 'Value')
    def __init__(self, syieldyfp: str=None, Tefp: str=None, Tifp: str=None, Concfp: str=None):
        """
        Basic var def
        """
        self.m_WallIDX = None
        self.m_Tgt = None
        self.m_Prj = None
        self.m_Chrg = None
        self.m_Value = None
        self.m_Energy = None
        self.m_tgtelements: list = None
        self.m_Te = None
        self.m_Ti = None
        if syieldyfp is not None and Tefp is not None and Tifp is not None and Concfp is not None:
            self.parse(syieldyfp, Tefp, Tifp, Concfp)

    def parse(self, syieldyfp: str, Tefp: str, Tifp: str, Concfp: str):
        """
        Parse file
        :param fp:
        :return:
        """
        self.m_WallIDX = []
        self.m_Tgt = []
        self.m_Prj = []
        self.m_Chrg = []
        self.m_Value = []
        with open(syieldyfp, 'rt') as fptr:
            header = fptr.readline().strip().split('\t')
            colidx = [header.index(colname) for colname in syieldby.syieldby_header_entries]

            curline = fptr.readline()
            while len(curline) > 0:
                cursplit = curline.split('\t')
                self.m_WallIDX.append(int(cursplit[colidx[0]]))
                self.m_Tgt.append(str(cursplit[colidx[1]]))
                self.m_Prj.append(str(cursplit[colidx[2]]))
                self.m_Chrg.append(float(cursplit[colidx[3]]))
                self.m_Value.append(float(cursplit[colidx[4]]))
                curline = fptr.readline()
        self.m_WallIDX = np.asarray(self.m_WallIDX, dtype=int)
        self.m_Tgt = np.asarray(self.m_Tgt, dtype=str)
        self.m_Prj = np.asarray(self.m_Prj, dtype=str)
        self.m_Chrg = np.asarray(self.m_Chrg, dtype=float)
        self.m_Value = np.asarray(self.m_Value, dtype=float)

        self.m_tgtelements = np.unique(self.m_Tgt).tolist()

        self.m_conc = np.zeros_like(self.m_Value)
        with open(Concfp, 'rt') as fptr:
            header = fptr.readline().strip().split('\t')
            colidx = [header.index(elemName) for elemName in self.m_tgtelements]
            self.m_conc = np.loadtxt(fptr, usecols=colidx, dtype=float)[self.m_WallIDX]

        Te = np.loadtxt(Tefp, skiprows=1, usecols=(1,), dtype=float)
        Ti = np.loadtxt(Tifp, skiprows=1, usecols=(1,), dtype=float)

        self.m_Te = np.zeros_like(self.m_Value)
        self.m_Ti = np.zeros_like(self.m_Value)
        self.m_Te = Te[self.m_WallIDX]
        self.m_Ti = Ti[self.m_WallIDX]

        self.m_Energy = 3.0 * self.m_Chrg * self.m_Te + 2.0 * self.m_Ti

        # # Sort by prj, tgt, energy
        # table = [self.m_WallIDX, self.m_Tgt, self.m_Prj, self.m_Chrg, self.m_Value, self.m_Energy, self.m_conc]
        # isortby = [self.m_Prj, self.m_Tgt, self.m_Energy]
        # for db in isortby:
        #     isrt = db.argsort(kind='mergesort')
        #     for tosrt in table:
        #         tosrt = tosrt[isrt]

    def getyield_vs_energy(self, tgt, prj, concfrom=0.0, concto=1.0):
        """
        Get the yields vs energy for a particular tgt/prj pair
        :param tgt:
        :param prj:
        :param concfrom:
        :param concto:
        :return:
        """
        tgt_prj = np.logical_and(self.m_Tgt == tgt.lower(),  self.m_Prj == prj.lower())

        itgt = self.m_tgtelements.index(tgt.lower())
        tgtconc = self.m_conc[:, itgt]
        conc = np.logical_and(tgtconc >= concfrom,  tgtconc <= concto)

        sel = np.logical_and(tgt_prj, conc)

        selidx = self.m_Energy[sel].argsort(kind='mergesort')

        return self.m_Energy[sel][selidx], self.m_Value[sel][selidx], self.m_conc[sel][selidx][:, itgt]

def compareto(tgt, prj, angle, model3dbfp: str, yields: syieldby):
    """
    Com
    :return:
    """
    with open(model3dbfp, 'rt') as fptr:
        db, header = load_model3db.load_db(fptr)
    header = [h.lower() for h in header[1:]]
    axes: Axes = None
    fig, axes = plt.subplots()

    energy, yld, conc = yields.getyield_vs_energy(tgt, prj)
    DataTable.DoPlot(axes, energy, yld, yscale=None, xoff=None,
                     method=PlotMethod.lin, label="WallDYN %s->%s" % (prj, tgt),
                     marker=None, markersize=12, fillstyle='full', color='k',
                     drawstyle=None, linestyle=None, linewidth=1)

    anglecol = header.index('angle')
    conccol = header.index('c%s' % tgt.lower())
    yldcol = header.index('y%s' % tgt.lower())
    energycol = header.index('energy')

    selconc = np.logical_and(db[:, conccol] >= conc.min(), db[:, conccol] <= conc.max())


    dbenergies = db[selconc, energycol]
    dbeyields = db[selconc, yldcol]
    dbangles = db[selconc, anglecol]
    dbconc = db[selconc, conccol]
    uniqueangles = np.unique(dbangles)
    matchangle = uniqueangles[np.argmin(np.abs(uniqueangles - angle))]
    selangle = np.abs(dbangles - matchangle) < 1.0E-4
    dbenergies = dbenergies[selangle]
    dbeyields = dbeyields[selangle]
    esortidx = dbenergies.argsort(kind='mergesort')

    res = DataTable.DoScatter(axes, dbenergies[esortidx], dbeyields[esortidx], yscale=None, xoff=None,
                     method=PlotMethod.lin, label="SDTrim.SP %s->%s" % (prj, tgt),
                     marker='x', markersize=200, color=dbconc[esortidx], colornorm=Normalize(vmin=conc.min(), vmax=conc.max()))

    axes.legend(loc='upper right').set_draggable(True, use_blit=True)
    axes.set_xlabel("D-Energy (eV)")
    axes.set_ylabel("Sputter yield (Al+O)")
    fig.colorbar(res, label='Al-Conc (at. frac.)')

def compare_al2o3_sdtrim_to_lit(model3dbfp: str, litfp: str, alconc = 0.4, oconc = 0.6, angle = 0.0):
    """
    Compare total (O + Al) sputter yield with literature data
    :param model3dbfp:
    :param litfp:
    :return:
    """
    axes: Axes = None
    fig, axes = plt.subplots()

    with open(model3dbfp, 'rt') as fptr:
        db, header = load_model3db.load_db(fptr)
    header = [h.lower() for h in header[1:]]

    anglecol = header.index('angle')
    Oconccol = header.index('co')
    Alconccol = header.index('cal')
    Oyldcol = header.index('yo')
    Alyldcol = header.index('yal')
    energycol = header.index('energy')

    selconc = np.logical_and(np.abs(db[:, Alconccol] - alconc) < 1.0E-4, np.abs(db[:, Oconccol] - oconc) < 1.0E-4)
    selangle = np.abs(db[:, anglecol] - angle) < 1.0E-4
    sel = np.logical_and(selconc, selangle)

    dbenergies = db[sel, energycol]
    dbOyields = db[sel, Oyldcol]
    dbAlyields = db[sel, Alyldcol]
    dbeyields = dbOyields + dbAlyields
    esortidx = dbenergies.argsort(kind='mergesort')

    res = DataTable.DoScatter(axes, dbenergies[esortidx], dbeyields[esortidx], yscale=None, xoff=None,
                              method=PlotMethod.lin, label="SDTrim.SP",
                              marker='x', markersize=200, color='r')

    ltenergy, ltyield = np.loadtxt(litfp, skiprows=1).T
    res = DataTable.DoScatter(axes, ltenergy, ltyield, yscale=None, xoff=None,
                              method=PlotMethod.lin, label="Literature",
                              marker='o', markersize=200, color='b')

    

    for e,y in zip(dbenergies[esortidx], dbeyields[esortidx]):
        print("%.3f\t%.6g" % (e, y))

    print(60*"#")
    for e, y in zip(ltenergy, ltyield):
        print("%.3f\t%.6g" % (e, y))

    return axes

def append_recalc(axes: Axes, recalcfp: str):
    """
    Append data from ./Al2O3Sputtering/SDTrim.SP/EnergyScan/sdtrim_res.dat
    :param recalcfp:
    :return:
    """
    energy, yields = np.loadtxt(recalcfp, skiprows=1).T

    sortix = energy.argsort(kind='mergesort')

    res = DataTable.DoScatter(axes, energy[sortix], yields[sortix], yscale=None, xoff=None,
                              method=PlotMethod.lin, label="re-calc",
                              marker='^', markersize=200, color='g')
def main():
    """
    Main entry point
    :return:
    """
    syieldyfp = 'data_o_al/results/ProtoEmpex_SyieldBy_100.000.dat'
    Tefp = 'data_o_al/results/ProtoEmpex_constantswall_Te.dat'
    Tifp = 'data_o_al/results/ProtoEmpex_constantswall_Ti.dat'
    Concfp = 'data_o_al/results/ProtoEmpex_Conc_100.000.dat'

    yields = syieldby(syieldyfp, Tefp, Tifp, Concfp)

    configmpl()

    prj = 'd'
    model3dbfp = os.path.expandvars(
        '$HOME/work/W-Modelling/Simulations/SDTRIM/Analyth-EroDep-input-data/Model4_D_Al_O_on_Al_O/Model3ParamScan_D_exte0.dat')
    tgt = 'al'
    angle = 45.0
    compareto(tgt, prj, angle, model3dbfp, yields)

    litfp = os.path.expandvars(
        '$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/GITR_Coupling/Al2O3Sputtering/total_yield_data_Al2O3.dat')
    axes = compare_al2o3_sdtrim_to_lit(model3dbfp, litfp)

    recalcfp = os.path.expandvars(
        '$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/GITR_Coupling/Al2O3Sputtering/SDTrim.SP/EnergyScan/sdtrim_res.dat')
    append_recalc(axes, recalcfp)

    axes.legend(loc='upper right').set_draggable(True, use_blit=True)


    plt.show()



if __name__ == '__main__':
    import traceback, sys

    try:
        # testing()
        main()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' % (repr(Ex), str(Ex)))
        print('Exception details:')
        traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script finished successfully')
