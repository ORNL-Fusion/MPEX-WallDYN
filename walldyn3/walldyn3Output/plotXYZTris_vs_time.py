#!/usr/bin/env python3
"""
Plot 3D triangles grouped by indices displaying a time evolution of parameters as red from an nc file
we import everything from plotXYZTris and only adjust the main function
"""
import os
import sys
import argparse
import numpy as np
import vtk
from datetime import datetime as dt
import matplotlib.cm as mcm
from genWDInput import TriangleGenerator, IndexGroupGenerator, WallParams
from MyScripts.argparse_tools import two_float_args
import pprint
from plotXYZTris import plotTriXYZ
from pyWallDYN.pyParseSolution import SolStateOft, PostProcOft
from MyScripts.emc3_impdepo_plot import VTKRenderWindow

pp = pprint.PrettyPrinter()

class PlotParamAnimator():
    """
    Animates the surface color in a plotTriXYZ
    """
    def __init__(self, plotter: plotTriXYZ, valarray: np.ndarray, timearray: np.ndarray, curtimeidx = -1, timstepfrac = 0.01):
        """
        Basci var def
        """
        self.m_plotter: plotTriXYZ = plotter
        self.interactor.AddObserver("KeyPressEvent", self.KeyPressEvent)
        self.m_renderer = self.interactor.GetRenderWindow().GetRenderers().GetFirstRenderer()

        self.m_valarray: np.ndarray = valarray   # valarray[wk, time]
        self.m_time_array: np.ndarray = timearray
        if self.m_time_array.shape[0] >= self.m_valarray.shape[1]:
            self.m_num_time = self.m_valarray.shape[1]
            if 0 <= curtimeidx < self.m_valarray.shape[1]:
                self.m_timeidx = curtimeidx
            else:
                self.m_timeidx = self.m_valarray.shape[1] - 1 # Default to last time

            self.m_timstep = int(float(self.m_num_time) * timstepfrac)
        else:
            raise Exception('Time values do not match plot values in array size')

    def reset(self):
        """
        Reset to time = 0
        :return:
        """
        self.m_timeidx = 0
        return self.m_valarray[:, self.m_timeidx]

    def nextstep(self):
        """
        Return next per wk data array and advance time index
        :return:
        """

        if self.m_timeidx < self.m_num_time - self.m_timstep - 1:
            self.m_timeidx += self.m_timstep
        else:
            self.m_timeidx = 0

        return self.m_valarray[:, self.m_timeidx]

    def previousstep(self):
        """
        Return next per wk data array and advance time index
        :return:
        """

        if self.m_timeidx >= self.m_timstep :
            self.m_timeidx -= self.m_timstep
        else:
            self.m_timeidx = self.m_num_time - 1

        return self.m_valarray[:, self.m_timeidx]

    @property
    def interactor(self) -> vtk.vtkRenderWindowInteractor:
        """
        Extract the interactor
        :return:
        """
        return self.m_plotter.m_vtk_rw.Interactor()

    @property
    def interactorstyle(self) -> vtk.vtkInteractorStyle:
        """
        Extract the interactor style
        :return:
        """
        return self.m_plotter.m_vtk_rw.Interactor().GetInteractorStyle()

    def KeyPressEvent(self, obj, event):
        """
        Handle keypress
        :param obj:
        :param event:
        :return:
        """

        key = self.interactor.GetKeySym().lower()
        is_shift = self.interactor.GetShiftKey() > 0
        is_alt = self.interactor.GetAltKey() > 0
        is_ctrl = self.interactor.GetControlKey() > 0

        modkeys = {'is_shift': is_shift, 'is_alt': is_alt, 'is_ctrl': is_ctrl}

        ismod = is_shift or is_alt or is_ctrl

        data = None
        if key == 'right':
            # Advance in time
            data = self.nextstep()
        elif key == 'left':
            # Go back in time
            data = self.previousstep()
        elif key == 'down':
            self.m_timstep *= 0.5
            self.m_timstep = int(max(self.m_timstep, 1.0))
        elif key == 'up':
            self.m_timstep *= 2.0
            self.m_timstep = int(min(self.m_timstep, self.m_num_time - 1))
        else:
            print('PlotParamAnimator: %s->%s' % (key, repr(modkeys)))

        if data is not None:
            self.m_plotter.re_assign_values(data)
        print('\t@t = %.3g (%d of %d delta = %d)' % (self.m_time_array[self.m_timeidx], self.m_timeidx, self.m_num_time, self.m_timstep))

        # Pass on event
        self.interactorstyle.OnKeyPress()

def main():
    """
    Main entry point
    :return:
    """
    # parse command args
    print('Working dir:\n%s' % os.getcwd())
    print('Command arguments: ')
    pp.pprint(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigen', type=str,
                        help='Path to triangle geometry',
                        default='./data_o_al/gitrGeometryPointPlane3d.cfg')

    parser.add_argument('--idxgrp', type=str,
                        help='Path to triangle index groups',
                        default='./data_o_al/surface/surface_inds_.txt')

    parser.add_argument('--statencfp', type=str, help='Path to solstate nc file', default=None)
    parser.add_argument('--postncfp', type=str, help='Path to post nc file', default=None)
    parser.add_argument('--valtoplot', type=str,help='Parameter to plot from nc file',
                        default="")
    parser.add_argument('--valuerange', type=two_float_args, action='store', help='Plotrange (from to)',
                        default=[None, None])
    parser.add_argument('--lowColor', type=str, help='Below range color RGBA 0.0 to 1.0',
                        default=None)
    parser.add_argument('--highColor', type=str, help='Below range color RGBA 0.0 to 1.0',
                        default=None)
    parser.add_argument('--norm', type=str, choices=['lin', 'log'], default='lin')
    parser.add_argument('--vallabel', type=str, help='Value type name', default=None)

    args = parser.parse_args(sys.argv[1:])

    def parse_color(colstr):
        """
        parse color from args
        :param colstr:
        :return:
        """
        if colstr is not None:
            cursplit = colstr.replace(',', ' ').split()
            if len(cursplit) == 4:
                colstr = [float(e) for e in cursplit]
            else:
                raise Exception('Color must be a sequence of 4 floats like "1.0 1.0 1.0 0.5"')
            return colstr
        else:
            return None

    args.highColor = parse_color(args.highColor)
    args.lowColor = parse_color(args.lowColor)

    trigen = TriangleGenerator()
    trigen.parse(args.trigen)

    indexgroups = IndexGroupGenerator()
    indexgroups.parse(args.idxgrp)

    plotter = plotTriXYZ(trigen, indexgroups)

    if args.statencfp is not None:
        # Load a state variable
        vals = None
        soft = SolStateOft()
        soft.parsefile(args.statencfp)

        varname, element = args.valtoplot.strip().split()
        element = element.lower()

        keygen = lambda wk: (element, wk)

        if (varname.lower() == 'ErosionFlux'.lower()):
            toplot = soft.m_GammaEro
        elif (varname.lower() == 'ReflectedFlux'.lower()):
            toplot = soft.m_GammaRefl
        elif (varname.lower() == 'ErodedFlux'.lower()):
            toplot = soft.m_GammaEro
        elif (varname.lower() == 'Conc'.lower()):
            toplot = soft.m_Conc
        elif (varname.lower() == 'delta'.lower()):
            toplot = soft.m_delta
        elif (varname.lower() == 'Gamma'.lower()):
            toplot = soft.m_GammaTot[element]
            keygen = lambda wk: wk
        elif (varname.lower() == 'SYieldBy'.lower()):
            tgt, prj, qprj = element.split('_')
            qprj = int(qprj)
            toplot = soft.m_SYieldBy

            keygen = lambda wk: (tgt, prj, qprj, wk)
        else:
            raise Exception('Unsupported varname %s' % varname)

        timearray = soft.m_tvals
        valarray = np.zeros((soft.m_nwall, soft.m_tvals.shape[0]), dtype=float)
        for wk in range(soft.m_nwall):
            valarray[wk] = toplot[keygen(wk)]

    elif args.postncfp is not None:
        # Load a post proc variable
        vals = None
        post = PostProcOft()
        post.parsefile(args.postncfp)

        varname, element = args.valtoplot.strip().split()
        element = element.lower()

        if (varname.lower() == 'NetAdensChange'.lower()):
            toplot = post.m_NetAdensChange
        elif (varname.lower() == 'ReflectedFlux'.lower()):
            toplot = post.m_ReflectedFlux
        elif (varname.lower() == 'ADensResiduumODE'.lower()):
            toplot = post.m_ADensResiduumODE
        elif (varname.lower() == 'TotalSource'.lower()):
            toplot = post.m_TotalSource
        elif (varname.lower() == 'NeutSeed'.lower()):
            toplot = post.m_NeutSeed
        else:
            raise Exception('Unsupported varname %s' % varname)

        timearray = post.m_tvals
        valarray = np.zeros((post.m_nwall, post.m_tvals.shape[0]), dtype=float)
        for wk in range(post.m_nwall):
            valarray[wk] = toplot[(element, wk)]
    else:
        raise Exception('Neither --statencfp nor --postncfp specified -> Aborting')

    plotter.assign_bin_values(valarray[:, -1]) # Start with the final value

    # mcm.Spectral
    plotter.PlotTriangles(cmap=mcm.jet,culling=0, fromto=args.valuerange, normtype=args.norm, label=args.vallabel,
                          belowcolorRGBA=args.lowColor, abovecolorRGBA=args.highColor)

    animator = PlotParamAnimator(plotter, valarray, timearray)

    plotter.m_vtk_rw.Start()


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