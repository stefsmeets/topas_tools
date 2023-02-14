from future.utils import raise_
import os, sys

import numpy as np
import pandas as pd

import math



from cctbx.array_family import flex

from cctbx import miller
from cctbx import xray
from cctbx import crystal

from cctbx.sgtbx import space_group_type
from cctbx.miller import index_generator
from cctbx import uctbx

from .cif import reader, CifParserError


centering_vectors = {
    'P': (['0.0', '0.0', '0.0'],),
    'A': (['0.0', '0.0', '0.0'], ['0.0', '0.5', '0.5']),
    'B': (['0.0', '0.0', '0.0'], ['0.5', '0.0', '0.5']),
    'C': (['0.0', '0.0', '0.0'], ['0.5', '0.5', '0.0']),
    'I': (['0.0', '0.0', '0.0'], ['0.5', '0.5', '0.5']),
    'R': (['0.0', '0.0', '0.0'], ['2/3', '1/3', '1/3'], ['1/3', '2/3', '2/3']),
    'S': (['0.0', '0.0', '0.0'], ['1/3', '2/3', '2/3'], ['2/3', '1/3', '1/3']),
    'T': (['0.0', '0.0', '0.0'], ['1/3', '2/3', '2/3'], ['2/3', '1/3', '2/3']),
    'H': (['0.0', '0.0', '0.0'], ['2/3', '1/3', '0.0'], ['1/3', '2/3', '0.0']),
    'K': (['0.0', '0.0', '0.0'], ['1/3', '0.0', '2/3'], ['2/3', '0.0', '1/3']),
    'L': (['0.0', '0.0', '0.0'], ['0.0', '2/3', '1/3'], ['0.0', '1/3', '2/3']),
    'F': (['0.0', '0.0', '0.0'], ['0.0', '0.5', '0.5'], ['0.5', '0.0', '0.5'], ['0.5', '0.5', '0.0'])
}


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, file):
            structures = reader(file_object=f).build_crystal_structures()
        elif isinstance(f, str):
            structures = reader(file_path=f).build_crystal_structures()
        else:
            raise TypeError(f'read_cif: Can not deal with type {type(f)}')
    except CifParserError as e:
        print(e)
        print("Error parsing cif file, check if the data tag does not contain any spaces.")
        sys.exit()
    for key, val in list(structures.items()):
        print("\nstructure:", key)
        val.show_summary().show_scatterers()
    return structures


def load_hkl(fin, labels=None, shelx=False, savenpy=False, verbose=True):
    """Read a file with filename 'fin', labels describe the data in the columns.
    The h,k,l columns are labeled by default and expected to be the first 3 columns.
    Returns a dataframe with hkl values as the indices

    e.g. load_hkl('red.hkl',labels=('F','sigmas')

    All columns should be labeled and columns labeled: 
        None,'None','none' or 'skip' will be ignored

    It is recommended to label all expected columns, so the algorithm will return an
    error rather than read the next column when columns are not delimited

    savenpy: save data as binary numpy format for faster loading on next use."""

    if shelx == True and labels == None:
        labels = ('F2', 'sigma')
    elif labels == None:
        raise TypeError('load_hkl() did not get a value for labels.')

    if isinstance(fin, file):
        fname = fin.name
    else:
        fname = fin

    labels = tuple(labels)
    skipcols = (None, 'None', 'none', 'skip') + \
        tuple(item for item in labels if item.startswith('#'))
    usecols = [0, 1, 2] + \
        [3+i for i, label in enumerate(labels) if label not in skipcols]

    root, ext = os.path.splitext(fname)
    changed = False

    try:
        inp = np.load(root+'.npy')
        assert len(inp.T) == len(
            usecols), 'npy data did not match to expected columns'
    except (OSError, AssertionError):
        changed = True
        if shelx == False:
            try:
                # if this fails, try shelx file
                inp = np.loadtxt(fin, usecols=usecols)
            except ValueError:
                inp = np.genfromtxt(
                    fin, usecols=usecols, delimiter=[4, 4, 4, 8, 8, 4])
        else:
            inp = np.genfromtxt(
                fin, usecols=usecols, delimiter=[4, 4, 4, 8, 8, 4])
    else:
        ext = '.npy'
        fname = root+'.npy'

    if savenpy:
        if ext != '.npy' or changed:
            print(f'Writing data as npy format to {fname}')
            np.save(root, inp)

    if verbose:
        print('')
        print(f'Loading data: {fname}')
        print(f'     usecols: {usecols}')
        print('      labels: {}'.format(' '.join(('h', 'k', 'l')+labels)))
        print(f'       shape: {inp.shape}')
    else:
        print(f'Loading data: {fname} => ({inp.shape[0]:5d}, {inp.shape[1]:2d})')

    h = list(map(int, inp[:, 0]))
    k = list(map(int, inp[:, 1]))
    l = list(map(int, inp[:, 2]))

    index = list(zip(h, k, l))

    labels = (label for label in labels if label not in skipcols)

    d = dict(list(zip(labels, inp[:, 3:].T)))

    df = pd.DataFrame(d, index=index)

    if not df.index.is_unique:
        print(f"\n** Warning: Duplicate indices detected in {fname} **\n")

        # useful, but very slow for large data sets!
        # index = list(df.index)
        # duplicates = set([x for x in index if index.count(x) > 1])
        # for x in duplicates: print x

    return df


def f_calc_dspacing(a, b, c, al, be, ga, h, k, l, kind='triclinic'):
    """
    Calculates d-spacing based on given parameters.
    a,b,c,al,be,ge are given as floats
    al,be,ga can be given as ndarrays or floats
    kind specifies the type of cell -> triclinic works for the general case, but is a bit slower
    although still fast enough

    Tested: orthorhombic cell on (orthorhombic, monoclinic, triclinic)
    Tested: triclinic cell with dvalues from topas
    """

    if kind == 'cubic':
        print('\n** Warning: cubic dspacing calculation unverified!! **\n')
        idsq = (h**2 + k**2 + l**2) / a**2

    elif kind == 'tetragonal':
        print('\n** Warning: tetragonal dspacing calculation unverified!! **\n')
        idsq = (h**2 + k**2) / a**2 + l**2 / c**2

    elif kind == 'orthorhombic':
        idsq = h**2 / a**2 + k**2 / b**2 + l**2 / c**2

    elif kind == 'hexagonal':
        print('\n** Warning: hexagonal dspacing calculation unverified!! **\n')
        idsq = (4/3) * (h**2 + h*k + k**2) * (1/a**2) + l**2 / c**2

    elif kind == 'monoclinic':
        print('\n** Warning: monoclinic dspacing calculation unverified!! **\n')
        be = math.radians(be)
        idsq = (1/math.sin(be)**2) * (h**2/a**2 + k**2 * math.sin(be)
                                      ** 2 / b**2 + l**2/c**2 - (2*h*l*math.cos(be)) / (a*c))

    elif kind == 'triclinic':
        V = calc_volume(a, b, c, al, be, ga)

        al = math.radians(al)
        be = math.radians(be)
        ga = math.radians(ga)

        idsq = (1/V**2) * (
            h**2 * b**2 * c**2 * math.sin(al)**2
            + k**2 * a**2 * c**2 * math.sin(be)**2
            + l**2 * a**2 * b**2 * math.sin(ga)**2
            + 2*h*k*a*b*c**2 * (math.cos(al) * math.cos(be) - math.cos(ga))
            + 2*k*l*b*c*a**2 * (math.cos(be) * math.cos(ga) - math.cos(al))
            + 2*h*l*c*a*b**2 * (math.cos(al) * math.cos(ga) - math.cos(be))
        )

    d = 1/idsq**0.5

    return d


def calc_dspacing(df, cell, col='d', kind='triclinic', inplace=True):
    """Calculate dspacing on df from indices"""
    a, b, c, al, be, ga = cell
    h, k, l = list(map(np.array, list(zip(*df.index))))
    d = f_calc_dspacing(a, b, c, al, be, ga, h, k, l, kind=kind)
    if inplace:
        df[col] = d
    else:
        return d


def calc_volume(a, b, c, al, be, ga):
    """Returns volume for the general case from cell parameters"""
    al = math.radians(al)
    be = math.radians(be)
    ga = math.radians(ga)
    V = a*b*c*((1+2*math.cos(al)*math.cos(be)*math.cos(ga) -
                math.cos(al)**2-math.cos(be)**2-math.cos(ga)**2)**.5)
    return V


def f_calc_structure_factors(structure, **kwargs):
    """Takes cctbx structure and returns f_calc miller array
    Takes an optional options dictionary with keys:
    input:
        **kwargs:
            'd_min': minimum d-spacing for structure factor calculation
            'algorithm': which algorithm to use ('direct', 'fft', 'automatic')
        structure: <cctbx.xray.structure.structure object>
    output:
        f_calc: <cctbx.miller.array object> with calculated structure factors
            in the f_calc.data() function

    TODO:
    - make this more general?
    - allow for specification of more parameters (like tables, ie. it1992 or wk1995)
    """

    dmin = kwargs.get('dmin', 1.0)
    algorithm = kwargs.get('algorithm', "automatic")
    anomalous = kwargs.get('anomalous', False)
    table = kwargs.get('scatfact_table', 'wk1995')
    return_as = kwargs.get('return_as', "series")

    if dmin <= 0.0:
        raise ValueError("d-spacing must be greater than zero.")

    if algorithm == "automatic":
        if structure.scatterers().size() <= 100:
            algorithm = "direct"
        else:
            algorithm = None

    structure.scattering_type_registry(table=table)

    f_calc_manager = structure.structure_factors(
        anomalous_flag=anomalous,
        d_min=dmin,
        algorithm=algorithm)
    f_calc = f_calc_manager.f_calc()

    print("\nScattering table:", structure.scattering_type_registry_params.table)
    structure.scattering_type_registry().show()
    print("Minimum d-spacing: %g" % f_calc.d_min())

    if return_as == "miller":
        return f_calc
    elif return_as == "series":
        fcalc = pd.Series(index=f_calc.indices(), data=np.abs(f_calc.data()))
        phase = pd.Series(index=f_calc.indices(), data=np.angle(f_calc.data()))
        return fcalc, phase
    elif return_as == "df":
        dffcal = pd.DataFrame(index=f_calc.index)
        dffcal['fcalc'] = np.abs(f_calc.data())
        dffcal['phase'] = np.angle(f_calc.data())
        return dffcal
    else:
        raise_(ValueError, f"Unknown argument for 'return_as':{return_as}")


def calc_structure_factors(cif, dmin=1.0, combine=None, table='xray', prefix='', **kwargs):
    """Wrapper around f_calc_structure_factors()
    Takes a cif file (str or file object)

    dmin can be a dataframe and it will take the minimum dspacing (as specified by col 'd') or a float
    if combine is specified, function will return a dataframe combined with the given one, otherwise a
    dictionary of dataframes

    prefix is a prefix for the default names fcalc/phases to identify different structures"""

    if isinstance(cif, str):
        f = open(cif)

    if isinstance(dmin, pd.DataFrame):
        dmin = min(dmin['d']) - 0.00000001

    structures = read_cif(f)

    if isinstance(structures, xray.structure):
        structures = {"fcalc": structures}

    col_phases = prefix+"phases"
    col_fcalc = prefix+"fcalc"

    for name, structure in list(structures.items()):
        fcalc, phase = f_calc_structure_factors(
            structure, dmin=dmin, scatfact_table=table, return_as="series", **kwargs)

        if len(structures) > 1:
            col_phases = prefix+"ph_"+name
            col_fcalc = prefix+"fc_"+name

        dffcal = pd.DataFrame({col_phases: phase, col_fcalc: fcalc})

        if combine is not None:
            combine = combine.combine_first(dffcal)
        else:
            structures[name] = dffcal

    if combine is not None:
        try:
            return combine.sort('d', ascending=False)
        except KeyError:
            return combine
    else:
        return structures


def merge_sym_equiv(m, output='ma', algorithm=None, verbose=True):
    """takes miller.array, returns hkl dictionary
    http://cci.lbl.gov/cctbx_sources/cctbx/miller/equivalent_reflection_merging.tex

    testing merging algorithms on different data sets:
    works = tested to give expected results
    fails = tested not to give expected results

                       gaussian        shelx
    unique             works           works
    unique+sigmas
    not unique                         works
    not unique+sigmas

    With no sigmas, shelx takes average of present refs in each group
    """
    n_unique = len(m.unique_under_symmetry().data())
    n_refs = len(m.data())

    # The number of merged reflections may be more than expected for powder data
    # this is because sometimes multiple indexes are read into the dataframe for some data sets
    # merging should then complete with R = 0, which is ok for powders

    if not algorithm:
        # indicates powder data set? => see also note above
        if n_refs == n_unique:
            # works with unique data, but gives strange results for single
            # crystal data
            algorithm = 'shelx'
        else:
            # gaussian returns 0 for unique data set, such as powders
            algorithm = 'shelx'

    # print "\ntotal/unique = {}/{} -> merging algorithm == {}".format(n_refs,n_unique,algorithm)
    # print "Using gaussian NEEDS accurate sigmas, shelx doesn't."

    merging = m.merge_equivalents(algorithm=algorithm)

    if verbose:
        print()
        merging.show_summary()
        print()

    m_out = merging.array()
    print('{} reflections merged/averaged to {}'.format(m.size(), m_out.size()))
    if output == 'dict':
        return miller_array_to_dict(m_out)
    elif output == 'ma':
        return m_out


def remove_sysabs(m, verbose=True):
    """Returns new miller.array with systematic absences removed"""
    sysabs = m.select_sys_absent().sort('data')

    if sysabs.size() > 0 and verbose:
        print(f'\nTop 10 systematic absences removed (total={sysabs.size()}):')

        if sysabs.sigmas() == None:
            for ((h, k, l), sf) in sysabs[0:10]:
                print(f'{h:4}{k:4}{l:4} {sf:8.2f}')
        else:
            for ((h, k, l), sf, sig) in sysabs[0:10]:
                print(f'{h:4}{k:4}{l:4} {sf:8.2f} {sig:8.2f}')

        print("Compared to largest 3 reflections:")
        if m.sigmas() == None:
            for ((h, k, l), sf) in m.sort('data')[0:3]:
                print(f'{h:4}{k:4}{l:4} {sf:8.2f}')
        else:
            for ((h, k, l), sf, sig) in m.sort('data')[0:3]:
                print(f'{h:4}{k:4}{l:4} {sf:8.2f} {sig:8.2f}')

        return m.remove_systematic_absences()
    elif sysabs.size() > 0:
        print(f"{sysabs.size()} systematic absences removed")
        return m.remove_systematic_absences()
    else:
        return m


def f_calc_multiplicities(df, cell, spgr):
    """Small function to calculate multiplicities for given dataframe"""
    m = df2m(df, cell, spgr)
    df['m'] = m.multiplicities().data()


def make_symmetry(cell, spgr):
    """takes cell parameters (a,b,c,A,B,C) and spacegroup (str, eg. 'cmcm'), returns cctbx
    crystal_symmetry class required for merging of reflections"""
    if not cell:
        cell = input("Please specify a cell:\n >> ")
        cell = list(map(float, cell.split()))
    if not spgr:
        spgr = input("Please specify space group:\n >> ")

    crystal_symmetry = crystal.symmetry(
        unit_cell=cell,
        space_group_symbol=spgr)
    return crystal_symmetry


def generate_indices(cell, spgr, dmin=1.0, ret='index'):
    """http://cci.lbl.gov/cctbx_sources/cctbx/miller/index_generator.h"""

    dmin = dmin-0.0000000000001  # because resolution_limit is < and not <=

    anomalous_flag = False
    symm = make_symmetry(cell, spgr)

    unit_cell = uctbx.unit_cell(cell)
    sg_type = space_group_type(spgr)
    # input hkl or resolution(d_min)
    mig = index_generator(
        unit_cell, sg_type, anomalous_flag=anomalous_flag, resolution_limit=dmin)
    indices = mig.to_array()
    if ret == 'index':  # suitable for df index
        return indices
    else:
        return miller.array(miller.set(crystal_symmetry=symm, indices=indices, anomalous_flag=anomalous_flag))


def m2df(m, data='data', sigmas='sigmas'):
    """Takes a miller.array object and returns the
    m: miller.array
    data and sigmas are the names for the columns in the resulting dataframe
    if no data/sigmas are present in the miller array, these are ignored.
    """
    df = pd.DataFrame(index=m.indices())
    if m.data():
        df[data] = m.data()
    if m.sigmas():
        df[sigmas] = m.sigmas()
    return df


def df2m(df, cell, spgr, data=None, sigmas=None):
    """Constructs a miller.array from the columns specified by data/sigmas in the dataframe,
    if both are None, returns just the indices.
    needs cell and spgr to generate a symmetry object."""
    anomalous_flag = False

    if isinstance(df, pd.DataFrame):
        if data:
            try:
                sel = df[data].notnull()  # select notnull data items for index
            except ValueError:
                index = df.index
            else:
                index = df.index[sel]
        else:
            index = df.index
    else:
        index = df

    indices = flex.miller_index(index)

    if data:
        data = flex.double(df[data][sel])
    if sigmas:
        sigmas = flex.double(df[sigmas][sel])

    symm = make_symmetry(cell, spgr)
    ms = miller.set(
        crystal_symmetry=symm, indices=indices, anomalous_flag=anomalous_flag)
    return miller.array(ms, data=data, sigmas=sigmas)


def reduce_all(df, cell, spgr, dmin=None, reindex=True, verbose=True):
    """Should be run after files2df. Takes care of some things that have to be done anyway.
    Once data has been loaded, this function reduces and merges all the data to a single 
    unique set, adds dspacings and multiplicities and orders columns and sorts by the dspacing.

    dmin:
    can be the name of a column and dmin is taken from that
    can be float
    can be None and the dmin is determined automatically asl the largest dmin of all columns read

    reindex: bool
    will reindex using all indices generated up to dmin

    Returns a dataframe object"""

    d = pd.Series(calc_dspacing(df, cell, inplace=False), index=df.index)
    dmins = [d[df[col].notnull()].min() for col in df if col not in ('m', 'd')]

    # little table with dmins
    cols = [col for col in df if col not in ('m', 'd')]
    ln = max([len(col) for col in cols])
    print('\n{:>{}} {:>6s}'.format('', ln, 'dmin'))
    for col, dval in zip(cols, dmins):
        print('{:>{}} {:6.3f}'.format(col, ln, dval))

    if not dmin:
        # find the largest dmin for all data sets
        dmin = max(dmins)
    elif isinstance(dmin, str):
        sel = df[dmin].notnull()
        dmin = min(d[sel])
    else:
        dmin = float(dmin)

    order = ['m', 'd']
    order = order + [col for col in df if col not in order]

    dfm = pd.DataFrame()

    for col in df:
        if col in ('m', 'd'):
            continue
        print(f'\n - Merging {col}: ')
        m = df2m(df, cell=cell, spgr=spgr, data=col)
        m = remove_sysabs(m, verbose=verbose)
        m = merge_sym_equiv(m, verbose=verbose)
        dfm = dfm.combine_first(m2df(m, data=col))

    if reindex:
        index = generate_indices(cell, spgr, dmin)
        index = pd.Index(index)
        dfm = dfm.reindex(index=index)
        dfm = dfm.reindex(columns=order)

    f_calc_multiplicities(dfm, cell, spgr)
    calc_dspacing(dfm, cell)

    print(f"\nReduced/merged data to dmin = {dmin}, {len(dfm)} refs")

    return dfm.sort_values(by='d', ascending=False)


def write_hkl(df, cols=None, out=None, no_hkl=False, pre=None, post=None, data_fmt=None, hkl_fmt=None):
    """Function for writing indices + selected columns to specified file/file object or terminal."""

    if isinstance(pre, list):
        if all('\n' in line for line in pre):
            pre = ''.join(pre)
        else:
            pre = '\n'.join(pre)
    elif isinstance(pre, str):
        pre = '\n'.join(pre.strip('\n'))

    if isinstance(post, list):
        post = ''.join(post)

    if not cols:
        cols = df.columns

    if isinstance(cols, str):
        cols = (cols,)

    if isinstance(out, str):
        out = open(out, 'w')

    cols = list(cols)

    if not hkl_fmt:
        if no_hkl:
            hkl_fmt = ''
        else:
            hkl_fmt = '{:4d}{:4d}{:4d}'

    if not data_fmt:
        ifmt = '{:4d}'
        dfmt = ' {:5d}'
        ffmt = ' {:9.3f}'
        bfmt = ' {:4}'

        n = len(cols)
        data_fmt = ''

        for item in cols[:]:
            if item == '*':
                cols.remove('*')
                data_fmt += '  *  '
                continue

            # tp = repr(type(df[item][0]))
            tp = repr(df[item].dtype)
            if 'int' in tp:
                data_fmt += dfmt
            elif 'float' in tp:
                data_fmt += ffmt
            elif 'bool' in tp:
                data_fmt += bfmt
            else:
                raise TypeError(f"No format associated with type {tp}")
    elif data_fmt == 'shelx':
        data_fmt = '{:8.3f}{:8.3f}'

    if pre:
        print(pre, file=out)

    print('>> Writing {} refs to file {}'.format(len(df), out.name if out else 'stdout'))

    for row in df.reindex(columns=cols).itertuples():

        # if (abs(row[1:][2] - row[1:][4]) < 0.0001 and
        #     abs(row[1:][2] - row[1:][5]) < 0.0001 and
        #     abs(row[1:][2] - row[1:][6]) < 0.0001):
        #   continue

        print(hkl_fmt.format(*row[0])+data_fmt.format(*row[1:]), file=out)

    if post:
        print(post, file=out)
