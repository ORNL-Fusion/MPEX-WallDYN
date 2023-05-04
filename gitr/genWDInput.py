#!/usr/bin/env python3
"""
Fill the GITR case into a WDInput object
similar to what is done in walldyncpp_input.py
"""

import sys, os, argparse, re, ast, glob, math
import pprint as pp
import numpy as np

# Appending path:

sys.path.append('/Users/42d/read-ods-with-odfpy')
sys.path.append('/Users/42d/code')
#from ODSReader.ODSReader import ODSReader
from ODSReader import ODSReader
from pyWallDYN.pyWDInput import WDInput
from pyWallDYN.pyWallModelLinear import WDWallModelLinear
from pyWallDYN.pyWDUnits import WDUnitConversion
from pyWallDYN.pyWDReDeposition import WDRedeposition1
from pyWallDYN.pyWDSurfaceModel import WDSurfaceModelRZBulk1
from pyWallDYN.pyWDEnergyModels import WDEnergyModelSheath
from pyWallDYN.pyYieldModel4 import yielmodel4db
from pyWallDYN.pyWDSputReflModel import WDRYieldModel4, WDSYieldModel4
from pyWallDYN.IDASolverConfig import IDASolverConfig


class MigrationMatrices():
    """
    Encapsulates access to migration matrices

    Hey Klaus,

    The source wall element index runs over rows. The destination runs over columns.

    In the matAl0+.dat file, the first 2x2 elements are:

    421 79
    65   481

    (This is how they appear in the text file, I know some readers may change the formatting).

    The diagonals are obvious. Then 79 particles originated at surface element 1 and were deposited at surface element 2. 65 particles originated at surface element 2 and were deposited at surface element 1.

    Let me know if that clears everything up!

    Thanks,
    Tim

    """
    def __init__(self, elements: list, chargestates: list):
        """
        Basic var def
        :param elements: list of elements
        :param chargestates: Number of charge states including 0
        """
        self.m_dim = None
        self.m_matrices: dict = {}  # Dictionarry for each element and list of matrices per charge state {...<ELEM>: [np.ndarray]}
        for ie, e in enumerate(elements):
            self.m_matrices[e.lower()] = [None for i in range(chargestates[ie])]

    @property
    def dim(self):
        """
        Matrix dimensions
        :return:
        """

        if self.m_dim is not None:
            return self.m_dim
        else:
            raise  Exception('Must first call checkmatrices() before accessing dim property')

    def add_matrix(self, elem: str, charge: int, fp:str):
        """
        Add a matrix
        :param elem:
        :param charge:
        :param fp:
        :return:
        """
        self.m_dim = None
        elem = elem.lower()
        if elem in self.m_matrices:
            if charge < len(self.m_matrices[elem]):
                if self.m_matrices[elem][charge] != None:
                    print('WARNING::Overrding existing matrix for %s charge state %d' %(elem, charge))
                self.m_matrices[elem][charge] = np.loadtxt(fp, delimiter=',')

    def is_dynamic(self, elem:str):
        """
        Check if a migration matrix exists for the element
        :param elem:
        :return:
        """
        return elem.lower() in self.m_matrices

    def max_charge(self, elem: str):
        """
        Maximum charge state
        :param elem:
        :return:
        """
        if self.is_dynamic(elem):
            return len(self.m_matrices[elem]) - 1
        else:
            return -1

    def redepmat(self, elements):
        """
        Create the list of redep matrices
        :param elements:
        :return:
        """
        redepmat = []
        norm = np.zeros((elements.nelem, self.dim))
        for ei, elem in enumerate(elements.names):
            curelemmat = None
            if self.is_dynamic(elem):
                curelemmat = []
                qmax = self.max_charge(elem)
                elemmat = self.m_matrices[elem]
                for qi in range(qmax+1):
                    curchrgmat = []
                    for wsrc in range(self.dim):
                        curwall = np.array(elemmat[qi][wsrc])
                        curchrgmat.append(curwall)
                        norm[ei, wsrc] += np.sum(curwall)
                    curelemmat.append(np.array(curchrgmat))
            redepmat.append(curelemmat)
        for ei, elem in enumerate(elements.names):
            if self.is_dynamic(elem):
                if norm[ei].sum() != self.dim:
                    print('Warning::Redistribution matrix for element %s is not normalized: %.6g' % (elem, norm[ei].sum() / self.dim))
        return redepmat

    def checkmatrices(self, nwall: int, normpersurfacegrp: np.ndarray = None):
        """
        Check matrix dimensions norm and compare to number of wall elements
        This also performs the matrix normalization
        :param nwall:
        :return:
        """
        dim = None
        for elem, chargemat in self.m_matrices.items():
            if dim is not None:
                summat = np.zeros((dim, dim))
            for qi, mat in enumerate(chargemat):
                if mat is not None:
                    if mat.shape[0] == mat.shape[1]:
                        if dim is not None:
                            if dim != mat.shape[0]:
                                raise Exception('Migration matrix for element %s at charge state %d has wrong shape' % (elem, qi))
                        else:
                            dim = mat.shape[0]
                            if dim == nwall:
                                summat = np.zeros((dim, dim))
                            else:
                                raise Exception('Matrix dimension %d does not match number of wall elements %d' % (dim, nwall))
                        summat += mat

            nrm = summat.sum(axis=1)

            if normpersurfacegrp is not None:
                if normpersurfacegrp.shape[0] == nrm.shape[0]:
                    nz = np.abs(nrm) > 0
                    meanerr = (nrm[nz]/normpersurfacegrp[nz]).mean()
                    print('INFO::Using external matrix norm mean normalization error = %.6g (1-nDepo/nLaunch)' % meanerr)
                    nrm = normpersurfacegrp.reshape((normpersurfacegrp.shape[0], 1))
                else:
                    raise Exception('Surface group normalization data does not match matrix dimension')
            else:
                print('INFO::Forcing mean norm = %.6g' % nrm.mean())
                nrm = nrm.reshape((nrm.shape[0], 1))

            summat /= nrm.reshape((nrm.shape[0], 1))

            for qi, mat in enumerate(chargemat):
                self.m_matrices[elem][qi] /= nrm

        self.m_dim = dim

        return True


class ElementData():
    """
    Encapsulates element data, core charge, mass etc.
    """

    def __init__(self):
        """
        Basic var def
        """
        self.m_elements: list = []
        self.m_core_Z: list = []
        self.m_mass_AMU: list = []
        self.m_pure_dens: list = []  # m^⁻3
        self.m_max_conc: list = []
        self.m_same_elements = {}

    def mark_same_element(self, elem, sameelem):
        """
        Mark an element as the same element but with a different name
        e.g.: D and DCX
        :param elem:
        :param sameelem:
        :return:
        """
        self.m_same_elements[elem] = sameelem

    @property
    def sameelemnts(self)->dict:
        """
        Create the sameelement list
        :return:
        """
        return self.m_same_elements

    def add_element(self, elem: str, core_Z: int, mass: float, puredens: float, maxconc: float):
        """
        Add a single element
        :param elem:
        :param core_Z:
        :param mass:
        :param puredens:
        :return:
        """
        self.m_elements.append(elem.lower())
        self.m_core_Z.append(core_Z)
        self.m_mass_AMU.append(mass)
        self.m_pure_dens.append(puredens)
        self.m_max_conc.append(maxconc)

    @property
    def names(self):
        """
        Element names
        :return:
        """
        return self.m_elements

    @property
    def  nelem(self):
        """
        # of elements
        :return:
        """
        return len(self.m_elements)

    @property
    def CorezMass(self):
        """
        Element names
        :return:
        """
        return [(Z, M) for Z, M in zip(self.m_core_Z, self.m_mass_AMU)]

    def CoreZ(self, ie):
        """
        Element names
        :return:
        """
        return self.m_core_Z[ie]

    @property
    def density(self):
        """
        Element names
        :return:
        """
        return self.m_pure_dens

    @property
    def maxconc(self):
        """
        Maximum allowed concentration
        :return:
        """
        return self.m_max_conc

    def accumflags(self, matrices: MigrationMatrices):
        """
        Generate accum flags
        :param matrices:
        :return:
        """
        accflags = []
        for elem in self.m_elements:
            if matrices.is_dynamic(elem):
                accflags.append(1)
            else:
                accflags.append(0)
        return accflags

    def charge(self, matrices: MigrationMatrices):
        """
        Get maximum charge state
        :param matrices:
        :return:
        """
        maxcharge = []
        for ie, elem in enumerate(self.m_elements):
            qmax = matrices.max_charge(elem)
            if qmax > 0:
                maxcharge.append(qmax)
            else:
                maxcharge.append(self.m_core_Z[ie])
        return maxcharge

    def numcharge(self, matrices: MigrationMatrices):
        """
        Get number of charge state = self.charge + 1
        :param matrices:
        :return:
        """
        nchrg = [qmax + 1 for qmax in self.charge(matrices)]
        return nchrg


class IndexGroupGenerator():
    """
    Generates index groups and exposes them as an iterator
    """
    def __init__(self):
        """
        Basic var def
        """
        self.m_indices: list = None  # List of np.ndarrays

    @property
    def wallbin(self):
        """
        Indexer
        :return:
        """
        for idx in self.m_indices:
            yield idx

    def parse(self, basename: str):
        """
        Search for basename*.ext and load all indices
        :param basename:
        :return:
        """
        fnbase, _ = os.path.splitext(os.path.split(basename)[1])
        fp, ext = os.path.splitext(basename)
        wildcard = '%s*%s' % (fp, ext)
        filelist = glob.glob(wildcard)
        nfiles = len(filelist)

        if nfiles > 0:
            print('Found %d files matchting %s' % (nfiles, wildcard))
            self.m_indices = [None for i in range(nfiles)]
            for fp in filelist:
                _, fn = os.path.split(fp)
                m = re.match(r'%s(\d+)%s' % (fnbase, ext), fn)
                if m is not None:
                    id = int(m.groups()[0])-1
                    if id < nfiles:
                        if self.m_indices[id] == None:
                            self.m_indices[id] = np.loadtxt(fp, dtype=int) - 1 # Convert to 0 based indexing
                        else:
                            raise Exception('Overwriting index group %d' % id)
                    else:
                        raise Exception('Index group %d out of range %d' % (id, nfiles))
        else:
            raise Exception('No files found for %s' % wildcard)

    @property
    def numwall(self):
        """
        Number of "wall-elements" made up from concatenated triangles
        :return:
        """
        return len(self.m_indices)

    def check_index_range(self, nsurf: int):
        """
        Make sure that the index ranges correctly match the number of surfaces
        :param nsurf:
        :return:
        """
        raw = np.concatenate(self.m_indices)
        nunique = np.unique(raw).shape[0]

        if raw.shape[0] == nunique:
            if np.max(raw) < nsurf:
                return True
            else:
                raise Exception('Surface index groups indices out of range')
        else:
            raise Exception('Surface index groups overlap')

class TriangleGenerator():
    """
    Generates triangles
    """
    def __init__(self):
        """
        Basic var def
        """
        # triangle coordinates
        self.m_xyz: np.ndarray = None  # shape (n-vtx, 3)
        self.m_triangles: np.ndarray = None # shape (n-tri, 3)

        # Surface areas
        self.m_area: np.ndarray = None # shape (n-tri,)

    @property
    def numtri(self):
        """
        Number of triangles
        :return:
        """
        return len(self.m_triangles)

    @property
    def area(self):
        """
        Number of triangles
        :return:
        """
        return self.m_area

    def XYZfromTriIDX(self, triidx):
        """
        Return center of triangle
        :param triidx:
        :return:
        """
        return self.m_xyz[self.m_triangles[triidx]].mean(axis=0)

    def NormalfromTriIDX(self, triidx):
        """
        Return triangle normal
        :param triidx:
        :return:
        """
        tricoords = self.m_xyz[self.m_triangles[triidx]]
        NXYZ = np.cross(tricoords[2] - tricoords[0], tricoords[1] - tricoords[0])
        diaglenscale = np.linalg.norm(tricoords[2] - tricoords[0]) / np.linalg.norm(NXYZ)
        diaglenscale = max(0.1, diaglenscale)
        return NXYZ * diaglenscale

    def walldyngeomproxy(self, indexing: IndexGroupGenerator):
        """
        Create a wall element coordinates (X, Y, Z of centers) wall element lengths and widths
        :param indexing:
        :return:
        """
        coords = []
        lengths = []
        widths = []
        for bin in indexing.wallbin:
            tridix = self.m_triangles[bin]
            tricoords = self.m_xyz[tridix]
            coords.append(tricoords.mean(axis=1).mean(axis=0))
            curarea = self.m_area[bin].sum()
            lengths.append(math.sqrt(curarea))
            widths.append(math.sqrt(curarea))

        return coords, lengths, widths, indexing.numwall

    def parse(self, fp: str, clearcache=False):
        """
        Parse from file
        :param fp:
        :return:
        """
        cachename = os.path.splitext(fp)[0] + '.npz'
        if not os.path.exists(cachename) or clearcache:
            with open(fp, 'r') as fptr:
                curline = fptr.readline().strip()
                while len(curline) > 0:
                    cursplit = curline.split('=')
                    if len(cursplit) == 2:
                        tok, val = [s.strip() for s in cursplit]
                        if len(val) > 0:
                            if tok == 'x1':
                                x1 = ast.literal_eval(val)
                            elif tok == 'x2':
                                x2 = ast.literal_eval(val)
                            elif tok == 'x3':
                                x3 = ast.literal_eval(val)
                            elif tok == 'y1':
                                y1 = ast.literal_eval(val)
                            elif tok == 'y2':
                                y2 = ast.literal_eval(val)
                            elif tok == 'y3':
                                y3 = ast.literal_eval(val)
                            elif tok == 'z1':
                                z1 = ast.literal_eval(val)
                            elif tok == 'z2':
                                z2 = ast.literal_eval(val)
                            elif tok == 'z3':
                                z3 = ast.literal_eval(val)
                            elif tok == 'area':
                                self.m_area = np.asarray(ast.literal_eval(val), dtype=float)
                    curline = fptr.readline().strip()

            xyz1 = np.vstack((x1, y1, z1)).T
            xyz2 = np.vstack((x2, y2, z2)).T
            xyz3 = np.vstack((x3, y3, z3)).T

            # Unique coordinates
            self.m_xyz = np.unique(np.vstack((xyz1, xyz2, xyz3)), axis=0)

            i1 = np.argmin(np.linalg.norm((self.m_xyz[:, np.newaxis] - xyz1), axis=2).T, axis=1)
            i2 = np.argmin(np.linalg.norm((self.m_xyz[:, np.newaxis] - xyz2), axis=2).T, axis=1)
            i3 = np.argmin(np.linalg.norm((self.m_xyz[:, np.newaxis] - xyz3), axis=2).T, axis=1)

            # Triangle buffer
            self.m_triangles = np.vstack((i1, i2, i3)).T

            # triidx = 12
            # print(self.m_xyz[self.m_triangles[triidx]])
            # print(xyz1[triidx])
            # print(xyz2[triidx])
            # print(xyz3[triidx])

            maxerrnorm = 1.0E-8
            for it, tri in enumerate(self.m_triangles):
                vb1, vb2, vb3 = self.m_xyz[tri]
                if np.linalg.norm(xyz1[it] - vb1) > maxerrnorm or np.linalg.norm(xyz2[it] - vb2) > maxerrnorm or np.linalg.norm(xyz3[it] - vb3) > maxerrnorm:
                    print('Missmatch @ it = %d:' % it)
                    print(((xyz1[it] - vb1), xyz2[it] - vb2, xyz3[it] - vb3))

            np.savez_compressed(cachename,
                                XYZ=self.m_xyz,
                                TRIS=self.m_triangles,
                                AREA=self.m_area
                                )
            print('Stored cached version @ %s' % cachename)
        else:
            cache = np.load(cachename)
            self.m_xyz = cache['XYZ']
            self.m_triangles = cache['TRIS']
            self.m_area = cache['AREA']
            print('Loaded from cache file %s' % cachename)


class WallParams():
    """
    Wall parameters like Te, Ti, flux etc
    """
    def __init__(self):
        """
        Basic var def
        """
        self.m_Te: np.ndarray = None
        self.m_Ti: np.ndarray = None
        self.m_pot: np.ndarray = None
        self.m_flux: dict = None
        self.m_angle: np.ndarray = None

    def parse(self, Tefp, Tifp, potfp, anglefp):
        """
        Parse the file
        :param Tefp:
        :param Tifp:
        :param potfp:
        :param fluxfp:
        :param anglefp:
        :return:
        """
        self.m_Te = np.loadtxt(Tefp)
        self.m_Ti = np.loadtxt(Tifp)
        self.m_pot = np.loadtxt(potfp)
        self.m_angle = np.loadtxt(anglefp)

        if not (len(self.m_Te) == len(self.m_Ti) == len(self.m_pot) == len(self.m_angle)):
            raise Exception('No all wall parameter arrays have same length')

    def add_const_flux(self, elem: str, qi: int, fp: str):
        """
        Add constant flux information
        :param elem:
        :param qi:
        :param fp:
        :return:
        """
        curflux = np.loadtxt(fp)

        if self.m_flux is None:
            self.m_flux = {}

        elem = elem.lower()
        if elem in self.m_flux:
            self.m_flux[elem][qi] = curflux
        else:
            self.m_flux[elem] = {qi: curflux}


    @property
    def dim(self):
        """
        Wall plasma data dimension
        :return:
        """
        return len(self.m_Te)

    def constfluxmatrix(self, matrices: MigrationMatrices, elements: ElementData, units: WDUnitConversion, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Generate a const flux matrix
        assigning m_flux to any non dynamic flux element
        :param matrices:
        :param elements:
        :param units:
        :param index:
        :param triangles:
        :return:
        """
        constflux = []
        binnedflux = []
        for ei, elem in enumerate(elements.names):
            curelem = None
            if not matrices.is_dynamic(elem):
                binnedflux.append(self.Flux(elem, index, triangles))
            else:
                binnedflux.append(None)

        for wk in range(matrices.dim):
            curwall = []
            for ei, elem in enumerate(elements.names):
                curelem = None
                if not matrices.is_dynamic(elem):
                    curelem = []
                    for qi in range(elements.CoreZ(ei)+1):
                        if qi in binnedflux[ei]:
                            flux = binnedflux[ei][qi][wk]
                        else:
                            flux = 0
                        if units is not None:
                            flux = units.ConvertFlux(flux)
                        curelem.append(flux)
                curwall.append(curelem)
            constflux.append(curwall)

        return constflux

    @staticmethod
    def area_weighted_average(value: np.ndarray, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Average per triangle over binned wall elements
        :param index:
        :return:
        """
        average = []
        for bin in index.wallbin:
            curarea = triangles.m_area[bin]
            curval = value[bin]
            binval = (curval * curarea).sum()/curarea.sum()
            average.append(binval)

        return  np.asarray(average)

    @staticmethod
    def total(value: np.ndarray, index: IndexGroupGenerator):
        """
        Average per triangle over binned wall elements
        :param index:
        :return:
        """
        total = []
        for bin in index.wallbin:
            curval = value[bin]
            binval = curval.sum()
            total.append(binval)

        return  np.asarray(total)

    def Flux(self, elem: str, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Average Flux over binned wall elements
        :param elem:
        :param qi:
        :param index:
        :param triangles:
        :return:
        """
        if elem in self.m_flux:
            res = {}
            for qi, flx in self.m_flux[elem].items():
                res[qi] = WallParams.total(flx * triangles.area, index) / WallParams.total(triangles.area, index)
            return res
        return None

    def Te(self, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Average Te over binned wall elements
        :param index:
        :return:
        """
        return WallParams.area_weighted_average(self.m_Te, index, triangles)

    def Ti(self, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Average Te over binned wall elements
        :param index:
        :return:
        """
        return WallParams.area_weighted_average(self.m_Ti, index, triangles)

    def Pot(self, index: IndexGroupGenerator, triangles: TriangleGenerator):
        """
        Average Pot over binned wall elements
        :param index:
        :return:
        """
        return WallParams.area_weighted_average(self.m_pot, index, triangles)

class PerWallData():
    """
    Parses per wall data from ODS file
    """
    def __init__(self, fpath: str):
        """
        Basic var def
        :param fp:
        """
        doc = ODSReader(fpath, clonespannedcolumns=True)

        def getsheet(name: str):
            """
            Get a sheet and report if it is missing
            :param name:
            :return:
            """
            if name in doc.SHEETS:
                table =  doc.getSheet(name)
                result = {}
                header = table[0]
                raw = np.asarray(table[1:], dtype=float).T
                for i, e in enumerate(header):
                    result[e.lower()] = raw[i]
                return result
            else:
                raise Exception('ODS file does not containt sheet %s' % name)

        self.m_bulkcomp = getsheet('bulkcomp')
        self.m_initRZ = getsheet('initRZ')
        self.m_temperature = getsheet('temperature')
        self.m_dx = getsheet('dx')

        try:
            self.m_impact_angle = getsheet('angle')
        except:
            self.m_impact_angle = {'angle': np.zeros_like(self.m_dx['dx'])}   # Default to perp impact

        dim = self.m_bulkcomp['wallidx'].shape[0]
        if dim == self.m_initRZ['wallidx'].shape[0] == self.m_temperature['wallidx'].shape[0] == self.m_dx['wallidx'].shape[0]:
            self.m_nwall = dim
        else:
            raise Exception('Per wall element data length missmatch')

    @property
    def dx(self) -> np.ndarray:
        """
        Reaction zone width
        :return:
        """
        return self.m_dx['dx']

    @property
    def nwall(self):
        """
        Numer of wall element data entries
        :return:
        """
        return self.m_nwall

    def initRZcompo(self, elem: str)->np.ndarray:
        """
        Initial concentration of element elem
        :param elem:
        :return:
        """
        return self.m_initRZ[elem.lower()]

    def bulkcompo(self, elem: str)->np.ndarray:
        """
        Initial concentration of element elem
        :param elem:
        :return:
        """
        return self.m_bulkcomp[elem.lower()]

    @property
    def temperature(self):
        return self.m_temperature['temperature']

    @property
    def impactangles(self):
        return self.m_impact_angle['angle']

def load_matrixnorm(fp, nrmcolname="norm"):
    """
    Load matrix normalization data for each index group
    :param fp:
    :return:
    """
    with open(fp, 'rt') as fptr:
        header = [e.lower() for e in fptr.readline().strip().split()]
        return np.loadtxt(fptr, usecols=(header.index(nrmcolname.lower()),))


def Store_GITR_in_WDInput(caseID: str,
                          elementdata: ElementData,
                          triangledata: TriangleGenerator,
                          wallpars: WallParams,
                          indexgroups: IndexGroupGenerator,
                          matrices: MigrationMatrices,
                          perwall: PerWallData,
                          yld4: yielmodel4db,
                          tolerances: dict,
                          matrixnorm: np.ndarray=None):
    """
    Store the case in a WDInput instance
    :param caseID:
    :param elementdata:
    :param triangledata:
    :param wallpars:
    :param indexgroups:
    :param matrices:
    :param perwall:
    :param yld4:
    :param tolerances:
    :param matrixnorm:
    :return:
    """

    if triangledata.numtri == wallpars.dim and indexgroups.check_index_range(triangledata.numtri):
        print('Basic data sanity check successfull, array dimensions and surface indices match')
    else:
        raise Exception('Geometry, WallPlasma parameters and index ranges missmatch')

    if matrices.checkmatrices(indexgroups.numwall, normpersurfacegrp=matrixnorm):
        print('Matrices seem to be ok')
    # No need to throw here checkmatrices throws if it fails

    # Fill everything into WDInput
    units = WDUnitConversion()
    inp = WDInput()
    inp.setCaseID(caseID)

    inp.SetElementData(elementdata.names,
                              elementdata.numcharge(matrices),
                              elementdata.CorezMass,
                              [units.ConvertDensty(roh) for roh in elementdata.density])

    AccumElemFlag = elementdata.accumflags(matrices)
    # For now a dynamic wall element is also a dynamic flux element
    inp.SetAccFluxFlags(AccumElemFlag, AccumElemFlag)

    # Configure the wall model
    coords, lengths, widths, nwall = triangledata.walldyngeomproxy(indexgroups)
    neutseed = [[0.0 for wj in range(nwall)] for ei in range(elementdata.nelem)]
    wallmodel = WDWallModelLinear(coords, lengths, perwall.temperature, neutseed, inp, widths, absorbingtiles = None)
    inp.SetWallModel(wallmodel)

    # Setup constant fluxes
    constflux = wallpars.constfluxmatrix(matrices, elementdata, units, indexgroups, triangledata)
    inp.SetConstFlux(constflux)

    # Set impact angle
    inp.SetIncidenAngle(perwall.impactangles)  # Use a fixed angle for now

    inp.setRedepMatrix(matrices.redepmat(elementdata))
    inp.SetRedepModel(WDRedeposition1(inp))

    # Set energy model
    # It seems there is an issue with the energy model because due to the RF nature of the
    # source the plasma potential is offset by it
    binTe = wallpars.Te(indexgroups, triangledata)
    binTi = wallpars.Te(indexgroups, triangledata)
    binPot = wallpars.Pot(indexgroups, triangledata)

    # Assume that pot is relative to the sheath edge then
    # Energy = q * (Pot + 3Te) + 2Ti
    # We simply implement this via WDEnergyModelSheath by adjusting TeFudge such that
    # 3 * TeFudge == (Pot + 3Te) --> TeFudge = (Pot/3 + Te)
    TeFudge = binPot/3.0 + binTe
    for ei, elem in enumerate(elementdata.names):
        inp.SetEnergyModel(ei, WDEnergyModelSheath(TeFudge,
                                                   binTi, inp))

    # Setup initial composition
    if nwall == perwall.nwall:
        initRZ = np.zeros((nwall, elementdata.nelem))
        initBulk = np.zeros((nwall, elementdata.nelem))

        for ei in inp.accumulating():
            initRZ[:, ei] = perwall.initRZcompo(elementdata.names[ei])
            initBulk[:, ei] = perwall.bulkcompo(elementdata.names[ei])

        surfacemodel = WDSurfaceModelRZBulk1(inp, initRZ, initBulk, perwall.dx)
        inp.SetSurfaceModel(surfacemodel)
    else:
        raise Exception('Perwall data # of wall elements missmatch')

    # Set maximum concentration
    inp.SetMaxConc(elementdata.maxconc)

    # Set yield models
    print('Generating a yielmodel4db')
    for elem, same in elementdata.sameelemnts:
        print('%s is identical to %s' % (same, elem))
        yld4.setSame(same, elem)

    syld = WDSYieldModel4(inp, yld4)
    ryld = WDRYieldModel4(inp, yld4)

    SYieldModel = []
    RYieldModel = []
    SublModel = None

    for ei in inp.elementidx():
        if inp.isaccumulating(ei):
            SYieldModel.append(syld)
            RYieldModel.append(ryld)
        else:
            SYieldModel.append(None)
            RYieldModel.append(None)

    SYieldModel = []
    RYieldModel = []

    for ei in inp.elementidx():
        if inp.isaccumulating(ei):
            SYieldModel.append(syld)
            RYieldModel.append(ryld)
        else:
            SYieldModel.append(None)
            RYieldModel.append(None)

    if len(SYieldModel) == inp.getNumElements():
        for ei, symod in enumerate(SYieldModel):
            inp.SetErolModel(ei, symod)
    else:
        raise Exception('Number of Sputter models must match number of elements')

    if len(RYieldModel) == inp.getNumElements():
        for ei, rymod in enumerate(RYieldModel):
            inp.SetRefllModel(ei, rymod)
    else:
        raise Exception('Number of Sputter models must match number of elements')

    if SublModel is not None:
        inp.SetSublModel(SublModel)
    else:
        inp.SetSublModel(None)

    # Set tolerances
    inp.SetTollerances(**tolerances)

    return inp

def Store_ControlFile(caseID: str, outdir: str, outtimes: list, storesteps=True, testjacfdd=False, outpath='results'):
    """
    Create the standard control file
    :return:
    """
    # Store the control XML
    solcfg = IDASolverConfig()

    solcfg.m_IDASolver = "IDASuperLUMT"

    print('Solver result outfilename base: %s' % outpath)
    solcfg.m_OutputAt_basefile = os.path.join(outpath, caseID + '.dat')
    print('Storing results at %s' % solcfg.m_OutputAt_basefile)
    solcfg.m_OutputAt_times = [float(t) for t in outtimes]
    print('at t = ')
    for t in solcfg.m_OutputAt_times:
        print('\t%.3g sec.' % t)

    solcfg.m_TMAX = max(solcfg.m_OutputAt_times)
    print('tmax = %.6g sec.' % solcfg.m_TMAX)

    solcfg.m_store_steps = storesteps
    if solcfg.m_store_steps:
        print('Turning on state storage after each step')
    else:
        print('State will only be stored at pre-selected time steps')

    solcfg.m_testjacfdd = testjacfdd
    if solcfg.m_store_steps:
        print('Testing jacobian using FDD')
    else:
        print('No testing of jacobian calculation')

    solcfg.m_IDASetErrFile = 'IDA_ERROR_LOG.log'

    cntrlpath = os.path.join(outdir, '%s_control.xml' % caseID)
    with open(cntrlpath, 'wb') as fptr:
        solcfg.store(fptr)

    print('Stored standart XML control file @\n\t%s' % cntrlpath)

def testcase():
    """
    Main entry point for testing
    :return:
    """

    wallbase = './data_o_al/fields'
    wallpar = WallParams()
    wallpar.parse(Tefp=os.path.join(wallbase, 'te.txt'),
                  Tifp=os.path.join(wallbase, 'ti.txt'),
                  potfp=os.path.join(wallbase, 'potential.txt'),
                  anglefp=os.path.join(wallbase, 'angle.txt'),
                  )
    wallpar.add_const_flux('D', 1, os.path.join(wallbase, 'flux.txt'))

    trigen = TriangleGenerator()
    trigen.parse('./data_o_al/gitrGeometryPointPlane3d.cfg')

    indexgroups = IndexGroupGenerator()
    indexgroups.parse('data_o_al/surface/surface_inds_.txt')

    matrices = MigrationMatrices(('Al', 'O'), (4, 4))
    matrices.add_matrix('Al', 0, 'data_o_al/matrices/matAl0+.dat')
    matrices.add_matrix('Al', 1, 'data_o_al/matrices/matAl1+.dat')
    matrices.add_matrix('Al', 2, 'data_o_al/matrices/matAl2+.dat')
    matrices.add_matrix('Al', 3, 'data_o_al/matrices/matAl3+.dat')

    matrices.add_matrix('O', 0, 'data_o_al/matrices/matO0+.dat')
    matrices.add_matrix('O', 1, 'data_o_al/matrices/matO1+.dat')
    matrices.add_matrix('O', 2, 'data_o_al/matrices/matO2+.dat')
    matrices.add_matrix('O', 3, 'data_o_al/matrices/matO3+.dat')

    matrixnorm = wdi.load_matrixnorm('../data_o_al/matrices/surfacegroup_nlaunch.dat')

    elements = ElementData()
    elements.add_element('D', 1, 2.0, 0.04E30, 0.0)
    elements.add_element('Al', 13, 27.0, 0.06E30, 1.0)
    elements.add_element('O', 16, 32.0, 0.06E30, 0.6)

    perwall = PerWallData("./data_o_al/perwalldata.ods")

    tolerances={"rel":1.0E-4, "absadens":1.0E-6, "absflux":1.0E-6, "absconc":1.0E-6}

    ylddb = os.path.expandvars("$HOME/work/W-Modelling/EMC3-eirene/walldyn/pyWallDYN/yieldmodel_4/BlinSPLINEDB_D_Al_O_on_Al_O/BlinSPLINEDB_d_al_o_on_al_o.XML")
    yld4 = yielmodel4db()
    yld4.loadXML(ylddb)

    caseID = "ProtoEmpex"
    inp = Store_GITR_in_WDInput(caseID, elements, trigen, wallpar, indexgroups, matrices, perwall, yld4, tolerances,
                                matrixnorm=matrixnorm)

    # Write input to XML
    outpath = "./data_o_al/GITR_ProtoEmpex_o_al.xml"
    compressXML = 0
    with open(outpath, 'wb') as fptr:
        inp.StoreAsXML(fptr, RedepNumeric=True, compression=compressXML)

    # Store the control XML
    outdir = "./data_o_al"
    outtimes = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    Store_ControlFile(caseID, outdir, outtimes, storesteps = True, testjacfdd = False, outpath = './results')


if __name__ == '__main__':
    import sys, traceback, os

    try:
        testcase()
    except Exception as Ex:
        print('Script failed due to: \n\t%s->%s' % (repr(Ex), str(Ex)))
        print('Exception details:')
        traceback.print_tb((sys.exc_info())[2])
    else:
        print('Script finished successfully')
