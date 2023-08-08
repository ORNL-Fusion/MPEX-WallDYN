#!/usr/bin/env python3
"""
Plot 3D triangles grouped by indices
"""
import os
import sys
sys.path.append("/Users/42d/code")
import argparse
import numpy as np
import vtk
from datetime import datetime as dt
import matplotlib.cm as mcm
from MyScripts.emc3_impdepo_plot import VTKRenderWindow, Build_VTK_LUT
from genWDInput import TriangleGenerator, IndexGroupGenerator, WallParams

from MyScripts.argparse_tools import two_float_args
import pprint

pp = pprint.PrettyPrinter()


class CellPicker(vtk.vtkInteractorStyleTrackballCamera):
    """
    Performs picking of cells
    """

    def __init__(self, rwin: VTKRenderWindow, lnchoffset=1.0E-3, cell_chg_handler=None, poly_chg_handler=None,
                 key_handler=None, overridepixelratio=None):
        """
        Stores local refs and registers observers
        :param textActor:
        :param textMapper:
        """

        # For QT based VTK displays
        # This overrides the pixelratio determined by QVTKRenderWindowInteractor._getPixelRatio()
        # For mirrored dispays from a mac this returns fails and returns the retina pixel ratio of 2
        # This is a hack to undo the scaling
        self.m_pratio_override = overridepixelratio

        # Poly data
        self.m_plotbase_polydata = {}

        self.m_rwin = rwin

        self.AddObserver("LeftButtonReleaseEvent", self.LeftButtonReleaseEvent)
        self.AddObserver("KeyPressEvent", self.KeyPressEvent)

        self.m_locators = []

        # Current selection
        self.m_selectedMapper = vtk.vtkDataSetMapper()
        self.m_selectedActor = None

        self.m_sphere = vtk.vtkSphereSource()
        self.m_sphere.SetCenter(0.0, 0.0, 0.0)
        self.m_sphere.SetRadius(0.02)
        self.m_sphere.Update()
        self.m_sphere_actor = None

        # Selection change handler
        self.AddCellChgHandler(cell_chg_handler)

        # Create a line
        self.m_lineSource = vtk.vtkLineSource()
        self.m_lineSource.SetPoint1([0.0, 0.0, 0.0])
        self.m_lineSource.SetPoint2([1.0, 1.0, 1.0])

        self.m_lineMapper = vtk.vtkPolyDataMapper()
        self.m_lineMapper.SetInputConnection(self.m_lineSource.GetOutputPort())

        self.m_lineActor = vtk.vtkActor()
        self.m_lineActor.SetMapper(self.m_lineMapper)

        self.m_renderer = None

        # Keyboard handlers
        self.AddKeyHandler(key_handler)

    def setPratioOverride(self, value=None):
        """
        Set the pixel ratio override
        :param value:
        :return:
        """
        self.m_pratio_override = value
        if self.m_pratio_override is not None:
            print('INFO: pixel ratio override = %d' % self.m_pratio_override)
            print('\tIf picking fails this could be the reason')

    def AddCellChgHandler(self, cell_chg_handler):
        """
        Attach a m_sel_change_handler
        :param cell_chg_handler:
        :return:
        """

        if cell_chg_handler is not None:
            if not hasattr(cell_chg_handler, 'OnSelChange'):
                raise Exception('cell_chg_handler must support OnSelChange method')

        self.m_sel_cell_change_handler = cell_chg_handler

    def AddKeyHandler(self, key_handler):
        """
        Attach a m_key_handler
        :param key_handler:
        :return:
        """

        # Handler for keyboard input
        if key_handler is not None and hasattr(key_handler, 'OnKeyVTK'):
            self.m_key_handler = key_handler
        else:
            self.m_key_handler = None

    def AddPolyData(self, quads: vtk.vtkPolyData, polybase):
        """
        Add polygon data
        :param quads: The actual renderable poly data
        :param polybase: The source of the poly data for further computations
        :return:
        """

        # Add a locator and store a link between the quads and the polybase in a map
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(quads)
        #locator.LazyEvaluationOn()

        self.m_locators.append(locator)
        self.m_plotbase_polydata[quads] = polybase

    def KeyPressEvent(self, obj, event):
        """
        Handle keypress
        :param obj:
        :param event:
        :return:
        """

        if self.m_renderer is None:
            self.m_renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()

        key = self.GetInteractor().GetKeySym().lower()
        is_shift = self.GetInteractor().GetShiftKey() > 0
        is_alt = self.GetInteractor().GetAltKey() > 0
        is_ctrl = self.GetInteractor().GetControlKey() > 0

        modkeys = {'is_shift': is_shift, 'is_alt': is_alt, 'is_ctrl': is_ctrl}

        ismod = is_shift or is_alt or is_ctrl

        fpath = 'camera_view.npz'

        if key == 's' and not ismod:
            # Store current camera orientation
            cam = self.m_renderer.GetActiveCamera()
            pos = np.array(cam.GetPosition())
            focus = np.array(cam.GetFocalPoint())
            upvec = np.array(cam.GetViewUp())

            np.savez_compressed(fpath,
                                pos=pos,
                                focus=focus,
                                upvec=upvec)

            print('Stored current camera view @ %s' % os.path.abspath(fpath))
        elif key == 'l' and not ismod:

            if os.path.exists(fpath):
                loadres = np.load(fpath)
                pos = loadres['pos']
                focus = loadres['focus']
                upvec = loadres['upvec']

                cam = self.m_renderer.GetActiveCamera()
                cam.SetPosition(pos)
                cam.SetFocalPoint(focus)
                cam.SetViewUp(upvec)

                self.GetInteractor().Render()

                print('Camera view restored from @ %s' % os.path.abspath(fpath))
            else:
                print('No camera view @ %s\nPress "s" to store first' % os.path.abspath(fpath))
        elif key == 'd' and not ismod:
            dumppath = os.path.join(os.getcwd(), 'vtk_window_dump.png')
            print('Creating screen dump @ %s' % dumppath)
            screen = vtk.vtkWindowToImageFilter()
            screen.SetInput(self.GetInteractor().GetRenderWindow())
            # screen.SetScale(3,3)
            #  screen.FixBoundaryOff()
            screen.SetInputBufferTypeToRGBA()
            screen.ReadFrontBufferOff()
            screen.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(dumppath)
            writer.SetInputConnection(screen.GetOutputPort())
            writer.Write()
        else:
            if self.m_key_handler is not None:
                self.m_key_handler.OnKeyVTK(key, **modkeys)
            else:
                print('CellPicker->Pressed: %s' % str(key))

        # Pass on event
        self.OnKeyPress()

    def LeftButtonReleaseEvent(self, obj, event):
        """
        Handle left nouse release
        :param obj:
        :param event:
        :return:
        """
        clickPos = np.asarray(self.GetInteractor().GetEventPosition(), dtype=int)

        if self.m_pratio_override is not None:
            clickPos = np.divide(clickPos, self.m_pratio_override, casting='same_kind')
        print('clickPos: %s' % (repr(clickPos)))

        picker = vtk.vtkCellPicker()
        for loc in self.m_locators:
            picker.AddLocator(loc)
        picker.SetTolerance(0.001)

        # picker.Pick(clickPos[0], clickPos[1], -1.0, self.m_rwin.Renderer())
        picker.Pick(clickPos[0], clickPos[1], -1.0, self.m_rwin.Renderer())
        pickpos = picker.GetPickPosition()
        print('Picked Position\n\tX, Y, Z: %s' % str(pickpos))
        print('\tX, Y, Z: %s' % str(pickpos))

        textActor, textMapper = self.m_rwin.GetTextPainter()

        cellid = picker.GetCellId()

        if cellid < 0:
            point_picker = vtk.vtkPointPicker()
            point_picker.SetTolerance(0.001)

            point_picker.Pick(clickPos[0], clickPos[1], 0.0, self.m_rwin.Renderer())
            pickpos = point_picker.GetPickPosition()
            print('Point-Picked Position\n\tX, Y, Z: %s' % str(pickpos))
            print('\tX, Y, Z: %s' % str(pickpos))

            picker = point_picker
            cellid = picker.GetPointId()

        if cellid < 0:
            textActor.VisibilityOff()
            self.HideMarkerSphere()
            # print('Placing marker at %s' % str(pickpos))
            # self.PlaceMarkerSphere(pickpos, radius=0.5)
        else:

            if self.m_renderer is None:
                self.m_renderer = self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer()
                self.m_renderer.AddActor(self.m_lineActor)
                self.m_renderer.AddActor2D(textActor)

            selPt = picker.GetSelectionPoint()
            pickPos = picker.GetPickPosition()
            info = "cellid = %d @(%.6f, %.6f, %.6f)" % (cellid, pickPos[0], pickPos[1], pickPos[2])
            print(info)

            dataset = picker.GetDataSet()

            if dataset in self.m_plotbase_polydata:
                plotbase = self.m_plotbase_polydata[dataset]

                cen = plotbase.XYZfromCellIDX(cellid)
                NX, NY, NNZ = norm = plotbase.NormalfromCellIDX(cellid)
                val, binidx = plotbase.quadvalue(cellid)

                info = "cellid = %d(%d) (%.6g) @(X:%.6f, Y:%.6f, Z:%.6f\n\t" \
                       "(NX:%.6f, NY:%.6f, NZ:%.6f)" % (
                           cellid, binidx, val, cen[0], cen[1], cen[2], NX, NY, NNZ)
                print(info)

                textMapper.SetInput(info)
                textActor.SetPosition(selPt[:2])
                textActor.VisibilityOn()

                if (self.m_sel_cell_change_handler is not None):
                    if self.m_sel_cell_change_handler.OnSelChange(cellid, dataset):
                        self.MarkCell(cellid, dataset)
                else:
                    self.MarkCell(cellid, dataset)
            else:
                print('Picked something else...')
                self.PlaceMarkerSphere(pickPos)

        self.OnLeftButtonUp()
        return

    def MarkCell(self, cellid, dataset):
        """
        Marks the currently selected cell
        :param cellid: the id of the selected cell
        :return:
        """
        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        ids.InsertNextValue(cellid)

        selectionnode = vtk.vtkSelectionNode()
        selectionnode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionnode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionnode.SetSelectionList(ids)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionnode)

        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, dataset)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        selected = vtk.vtkUnstructuredGrid()
        selected.ShallowCopy(extractSelection.GetOutput())

        # Grow slightly
        shrink = vtk.vtkShrinkFilter()
        shrink.SetInputData(selected)
        shrink.SetShrinkFactor(1.1)
        self.m_selectedMapper.SetInputConnection(shrink.GetOutputPort())

        # self.m_selectedMapper.SetInputData(selected)

        if (self.m_selectedActor is not None):
            self.m_selectedActor.GetProperty().EdgeVisibilityOff()
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.m_selectedActor)
        else:
            self.m_selectedActor = vtk.vtkActor()

        self.m_selectedActor.SetMapper(self.m_selectedMapper)
        self.m_selectedActor.GetProperty().EdgeVisibilityOn()
        self.m_selectedActor.GetProperty().SetEdgeColor(1, 1, 1)
        self.m_selectedActor.GetProperty().SetLineWidth(10)

        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.m_selectedActor)

        if dataset in self.m_plotbase_polydata:
            plotbase = self.m_plotbase_polydata[dataset]
            cen = plotbase.XYZfromCellIDX(cellid)
            self.PlaceMarkerSphere(cen)

            NX, NY, NZ = plotbase.NormalfromCellIDX(cellid)
            normlen = np.linalg.norm([NX, NY, NZ])
            self.m_lineSource.SetPoint1(cen)
            normpoint = [NN + CC for NN, CC in zip(cen, [NX, NY, NZ])]
            self.m_lineSource.SetPoint2(normpoint)
            self.m_lineMapper.Update()
            self.m_lineActor.VisibilityOn()
        else:
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.m_sphere_actor)
            self.m_lineActor.VisibilityOff()

    def PlaceMarkerSphere(self, center, radius=0.02):
        """
        Just place the marker sphere
        :param center:
        :return:
        """
        self.m_sphere.SetRadius(radius)
        self.m_sphere.SetCenter(*center)

        self.m_sphere.Update()
        if self.m_sphere_actor is not None:
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.m_sphere_actor)
        else:
            self.m_sphere_actor = vtk.vtkActor()

        spheremapper = vtk.vtkPolyDataMapper()
        spheremapper.SetInputConnection(self.m_sphere.GetOutputPort())

        self.m_sphere_actor.SetMapper(spheremapper)
        bnds = self.m_sphere_actor.GetBounds()

        self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.m_sphere_actor)

    def HideMarkerSphere(self):
        """
        Remove the sphere actor from the scene
        :return:
        """
        if self.m_sphere_actor is not None:
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.m_sphere_actor)

class plotTriXYZ():
    """
    Plots triangles grouped by indices
    with values assigned to each index group
    """
    def __init__(self, trigen: TriangleGenerator, indexer: IndexGroupGenerator, renderwindow: VTKRenderWindow = None, ncolors = 16):
        """
        Basic var def
        :param renderwindow:
        """
        self.m_vtk_rw: VTKRenderWindow = renderwindow
        self.m_triangles = trigen
        self.m_wallbins = indexer
        self.m_cell_idx_hash = None

        self.m_lut = None
        self.m_scalebar = None
        self.m_geomactor = None

        self.m_cmap = mcm.jet
        self.m_normtype = 'lin'
        self.m_belowcolorRGBA = self.m_cmap(0.0)# (0.0, 0.0, 0.0, 0.5)
        self.m_abovecolorRGBA = self.m_cmap(1.0) # (1.0, 1.0, 1.0, 0.5)
        self.m_flipcolors = False

        self.m_values = None
        self.m_lut = None
        self.m_num_color_entries = ncolors

    def XYZfromCellIDX(self, cellidx):
        """
        Return center of triangle
        :param triidx:
        :return:
        """
        triidx = self.m_cell_idx_hash[cellidx]
        return self.m_triangles.XYZfromTriIDX(triidx)

    def NormalfromCellIDX(self, cellidx):
        """
        Return triangle normal
        :param triidx:
        :return:
        """
        triidx = self.m_cell_idx_hash[cellidx]
        return self.m_triangles.NormalfromTriIDX(triidx)

    def quadvalue(self, cellidx):
        """
        Get teh value from a trianlge index
        :param tridix:
        :return:
        """
        triidx = self.m_cell_idx_hash[cellidx]
        for ibin, bins in enumerate(self.m_wallbins.m_indices):
            if triidx in bins:
                return self.m_values[ibin], ibin
        return -42.9, -42

    def re_assign_values(self, values):
        """
        Assing new values to each bin defined by IndexGroupGenerator
        :param values:
        :return:
        """
        if len(values) == self.m_wallbins.numwall:
            self.m_values = np.asarray(values, dtype=float)
        else:
            raise Exception('Values array does not match bin definition')

        self.__assign_colors(fromto=self.m_fromto)

        self.m_vtk_rw.Window().Render()

    def assign_bin_values(self, values):
        """
        Assing a value to each bin defined by IndexGroupGenerator
        :param values:
        :return:
        """
        if len(values) == self.m_wallbins.numwall:
            self.m_values = np.asarray(values, dtype=float)
        else:
            raise Exception('Values array does not match bin definition')

    def __assign_colors(self, fromto=None, ncolors=None):
        """
        Assigns colors to quads
        :param values:
        :return:
        """
        if ncolors is not None:
            self.m_num_color_entries = ncolors

        self.m_lut = Build_VTK_LUT(self.m_values, fromto=fromto, normtype=self.m_normtype, cmap=self.m_cmap,
                                   num_col_vals=self.m_num_color_entries,
                                   flipcolors=self.m_flipcolors,
                                   belowcolorRGBA=self.m_belowcolorRGBA, abovecolorRGBA=self.m_abovecolorRGBA)

        # Backup fromo
        self.m_fromto = [val for val in fromto]

        colorData = vtk.vtkUnsignedCharArray()
        colorData.SetName('colors')  # Any name will work here.
        colorData.SetNumberOfComponents(4)

        t1 = dt.now()
        cntr = 0
        numquads = len( self.m_wallbins.m_indices)
        for val, triangles in zip(self.m_values, self.m_wallbins.m_indices):

            curcol = [0.0, 0.0, 0.0]
            self.m_lut.GetColor(val, curcol)
            opacity = self.m_lut.GetOpacity(val)

            ucrgb = list(map(int, [x * 255 for x in curcol]))

            for tri in triangles:
                colorData.InsertNextTuple4(ucrgb[0], ucrgb[1], ucrgb[2], int(opacity * 255))

            t2 = dt.now()
            cntr += 1
            if (t2 - t1).seconds > 1:
                print('Processed quad %d of %d' % (cntr, numquads))
                t1 = t2

        if self.m_geomactor is not None:
            mapper = self.m_geomactor.GetMapper()
            quads = mapper.GetInput()
            if mapper is not None and quads is not None:
                quads.GetCellData().SetScalars(colorData)
                mapper.SetScalarModeToUseCellData()
                mapper.Update()
            else:
                raise Exception('Failed to obtain mapper and/or mapper input from actor')
        else:
            raise Exception('No vtk.vtkActor() instance at self.m_geomactor')

    def PlotTriangles(self, cmap=mcm.jet, normtype='lin', fromto=None, label='Name and Units',
                         belowcolorRGBA=None, abovecolorRGBA=None, culling=0,
                         docolorbar=True, flipcolors=False):
        """
        Plot the triangle
        :param cmap:
        :param normtype:
        :param fromto:
        :param label:
        :param belowcolorRGBA:
        :param abovecolorRGBA:
        :param culling:
        :param docolorbar:
        :param flipcolors:
        :return:
        """

        # Copy
        self.m_cmap = cmap
        self.m_normtype = normtype
        if belowcolorRGBA is not None:
            self.m_belowcolorRGBA = belowcolorRGBA
        else:
            self.m_belowcolorRGBA = self.m_cmap(0.0)

        if abovecolorRGBA is not None:
            self.m_abovecolorRGBA = abovecolorRGBA
        else:
            self.m_abovecolorRGBA = self.m_cmap(1.0)

        self.m_flipcolors = flipcolors

        if (self.m_vtk_rw is None):
            self.m_vtk_rw = VTKRenderWindow()
        self.m_vtk_rw.AddAxis(scale=0.1)

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(self.m_triangles.m_xyz.shape[0])

        cells = vtk.vtkCellArray()
        cells.Allocate(self.m_triangles.numtri, 256)

        for ptid, XYZ in enumerate(self.m_triangles.m_xyz):
            pts.SetPoint(ptid, XYZ)

        self.m_cell_idx_hash = []
        for triangles in self.m_wallbins.m_indices:
            for i1, i2, i3 in self.m_triangles.m_triangles[triangles]:
                cells.InsertNextCell(3)
                cells.InsertCellPoint(i1)
                cells.InsertCellPoint(i2)
                cells.InsertCellPoint(i3)
            self.m_cell_idx_hash += triangles.tolist()

        # Build the polygon data
        tris = vtk.vtkPolyData()
        tris.SetPoints(pts)
        tris.SetPolys(cells)

        # Default to index group number
        if self.m_values is None:
            self.m_values = np.arange(self.m_wallbins.numwall)

        TriMapper = vtk.vtkPolyDataMapper()
        TriMapper.SetInputData(tris)

        # Setup quad actor
        self.m_vtk_rw.RemoveProp(self.m_geomactor)
        self.m_geomactor = vtk.vtkActor()
        self.m_geomactor.SetMapper(TriMapper)

        self.__assign_colors(fromto=fromto)

        if docolorbar:
            self.m_vtk_rw.RemoveProp(self.m_scalebar)
            self.m_scalebar = vtk.vtkScalarBarActor()
            self.m_scalebar.SetLookupTable(self.m_lut)
            self.m_scalebar.SetNumberOfLabels(self.m_lut.GetNumberOfTableValues())
            if label is not None:
                self.m_scalebar.SetTitle(label)
            self.m_vtk_rw.Renderer().AddActor2D(self.m_scalebar)
        else:
            self.m_vtk_rw.RemoveProp(self.m_scalebar)
            self.m_scalebar = None

        # Create trackball style manipulator
        # that supports picking on left click
        handler = self.m_vtk_rw.Interactor().GetInteractorStyle()

        if handler is None or not isinstance(handler, CellPicker):
            handler = CellPicker(self.m_vtk_rw)
            handler.SetDefaultRenderer(self.m_vtk_rw.Renderer())
            self.m_vtk_rw.Interactor().SetInteractorStyle(handler)

        handler.AddPolyData(tris, self)

        self.setCulling(culling, doinit=False)

        self.m_vtk_rw.Renderer().AddActor(self.m_geomactor)

        self.m_vtk_rw.Window().SetWindowName(label)

    def setCulling(self, culling, doinit=False):
        """
        Change culling
        :param culling:
        :return:
        """

        if self.m_geomactor is not None:
            if culling == 0:
                print('Culling off')
                self.m_geomactor.GetProperty().BackfaceCullingOff()
                self.m_geomactor.GetProperty().FrontfaceCullingOff()
            elif culling == 1:
                self.m_geomactor.GetProperty().BackfaceCullingOn()
                self.m_geomactor.GetProperty().FrontfaceCullingOff()
            elif culling == 2:
                self.m_geomactor.GetProperty().BackfaceCullingOff()
                self.m_geomactor.GetProperty().FrontfaceCullingOn()
            else:
                raise Exception(
                    'Unsupported culling mode %d (must be 0->Off, 1 Cull-Backfacing, 2 Cull-Frontfacing' % culling)

        if doinit:
            self.m_vtk_rw.Interactor().ReInitialize()



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

    parser.add_argument('--valfp', type=str, help='Path to per index group plot values', default=None)
    parser.add_argument('--valcfg', type=int, nargs=2, help='column to use, number of header lines to skip',
                        default=(1,1))
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

    if args.valfp is not None:
        vals = np.loadtxt(args.valfp, skiprows=args.valcfg[1], usecols=(args.valcfg[0]))
        plotter.assign_bin_values(vals)

    plotter.PlotTriangles(cmap=mcm.Spectral,culling=0, fromto=args.valuerange, normtype=args.norm, label=args.vallabel,
                          belowcolorRGBA=args.lowColor, abovecolorRGBA=args.highColor)

    plotter.m_vtk_rw.Start()


def testing():
    """
    Test basic plotting
    :return:
    """
    trigen = TriangleGenerator()
    trigen.parse('./data_o_al/gitrGeometryPointPlane3d.cfg')

    indexgroups = IndexGroupGenerator()
    indexgroups.parse('./data_o_al/surface/surface_inds_.txt')

    plotter = plotTriXYZ(trigen, indexgroups)
    plotter.PlotTriangles(culling=0, fromto=(89, 93))

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
