from cctbx import crystal


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


def make_special_position_settings(cell, space_group, min_dist_sym_equiv=0.000001):
    """Takes cell and space group, returns cctbx structure
    input:  cell: (a b c alpha beta gamma) as a tuple
            space_group: 'space_group' like a string
            min_dist_sym_equiv: float
    output: <cctbx.crystal.special_position_settings object>
                contains cell, space group and special positions
    """

    assert type(cell) == tuple, 'cell must be supplied as a tuple'
    assert len(cell) == 6, 'expected 6 cell parameters'

    special_position_settings = crystal.special_position_settings(
        crystal_symmetry=crystal.symmetry(
            unit_cell=cell,
            space_group_symbol=space_group),
        min_distance_sym_equiv=min_dist_sym_equiv)
    return special_position_settings


def print_superflip(fcalc, fout, fdiff_file=None):
    """Prints an inflip file that can directly be used with superflip for difference fourier maps

    - Tested and works fine with: EDI, SOD
    """
    print('title', 'superflip\n', file=fout)

    print('dimension 3', file=fout)
    print('voxel', end=' ', file=fout)
    for p in fcalc.unit_cell().parameters()[0:3]:
        print(int(((p*4) // 6 + 1) * 6), end=' ', file=fout)
    print(file=fout)
    print('cell', end=' ', file=fout)
    for p in fcalc.unit_cell().parameters():
        print(p, end=' ', file=fout)
    print('\n', file=fout)

    print('centers', file=fout)
    for cvec in centering_vectors[fcalc.space_group_info().type().group().conventional_centring_type_symbol()]:
        print(' '.join(cvec), file=fout)
    print('endcenters\n', file=fout)

    print('symmetry #', fcalc.space_group_info().symbol_and_number(), file=fout)
    print('# inverse no', file=fout)

    # number of unique symops, no inverses
    n_smx = fcalc.space_group_info().type().group().n_smx()
    # number of primitive symops, includes inverses
    order_p = fcalc.space_group_info().type().group().order_p()
    # total number of symops
    order_z = fcalc.space_group_info().type().group().order_z()

    # this should work going by the assumption that the unique primitive symops are stored first,
    # THEN the inverse symops and then all the symops due to centering.

    for n, symop in enumerate(fcalc.space_group_info().type().group()):
        if n == order_p:
            break
        elif n == n_smx:
            print('# inverse yes, please check!', file=fout)
        print(symop, file=fout)

        # Broken, because .inverse() doesn't work, but probably a better approach:
    # for symop in f.space_group_info().type().group().smx():
    #   print >> fout, symop
    # if f.space_group_info().type().group().is_centric():
    #   print >> fout, '# inverse yes'
    #   for symop in f.space_group_info().type().group().smx():
    #       print >> fout, symop.inverse() # inverse does not work?

    print('endsymmetry\n', file=fout)

    print('perform fourier', file=fout)
    print('terminal yes\n', file=fout)

    print('expandedlog yes', file=fout)
    print('outputfile superflip.xplor', file=fout)
    print('outputformat xplor\n', file=fout)

    print('dataformat amplitude phase', file=fout)

    if fdiff_file:
        print('fbegin fdiff.out\n', file=fout)
    else:
        print('fbegin', file=fout)
        print_simple(fcalc, fout, output_phases='cycles')

#       for i,(h,k,l) in enumerate(f.indices()):
#           # structurefactor = abs(f.data()[i])
#           # phase = phase(f.data()[i]
#           print >> fout, "%3d %3d %3d %10.6f %10.3f" % (
#               h,k,l, abs(f.data()[i]), phase(f.data()[i]) / (2*pi) )
        print('endf', file=fout)


def make_superflip(cell, spgr, wavelength, composition, datafile, dataformat, filename='sf.inflip'):
    sps = make_special_position_settings(cell, spgr)
    sg = sps.space_group()
    uc = sps.unit_cell()
    # sgi = sps.space_group_info()

    fout = open(filename, 'w')

    print('title', filename.split('.')[0], file=fout)
    print(file=fout)
    print('dimension 3', file=fout)
    print('voxel', end=' ', file=fout)
    for p in uc.parameters()[0:3]:
        print(int(((p*4) // 6 + 1) * 6), end=' ', file=fout)
    print(file=fout)
    print('cell', end=' ', file=fout)
    for p in uc.parameters():
        print(p, end=' ', file=fout)
    print(f'  # vol = {uc.volume():.4f} A3 \n', file=fout)

    print('centers', file=fout)
    for cvec in centering_vectors[sg.conventional_centring_type_symbol()]:
        print('  ', ' '.join(cvec), file=fout)
    print('endcenters\n', file=fout)

    print('symmetry #', sg.crystal_system(), sg.info(), file=fout)
    print('# +(0 0 0) Inversion-Flag = 0', file=fout)

    n_smx = sg.n_smx()
    order_p = sg.order_p()
    order_z = sg.order_z()

    for n, symop in enumerate(sg):
        if n == order_p:
            break
        elif n == n_smx:
            print('# +(0 0 0) Inversion-Flag = 1', file=fout)
        print('  ', symop, file=fout)
    print('endsymmetry\n', file=fout)
    print('derivesymmetry yes', file=fout)
    print('searchsymmetry average', file=fout)
    print(file=fout)
    print('delta AUTO', file=fout)
    print('weakratio 0.00', file=fout)
    print('biso 2.0', file=fout)
    print('randomseed AUTO', file=fout)
    print(file=fout)
    if composition:
        print(f'composition {composition}', file=fout)
        print('histogram composition', file=fout)
        print('hmparameters 10 5', file=fout)
    else:
        print('#composition #composition goes here', file=fout)
        print('#histogram composition', file=fout)
        print('#hmparameters 10 5', file=fout)
    print(file=fout)
    print('fwhmseparation 0.3', file=fout)
    print(f'lambda {wavelength}', file=fout)
    print(file=fout)
    print('maxcycles 200', file=fout)
    print('bestdensities 10', file=fout)
    print('repeatmode 100', file=fout)
    print(file=fout)
    print('polish yes', file=fout)
    print('convergencemode never', file=fout)
    print(file=fout)
    print('#referencefile filename.cif', file=fout)
    print('#modelfile filename.cif 0.2', file=fout)
    print(file=fout)
    print('terminal yes', file=fout)
    print('expandedlog yes', file=fout)
    outputfile = str(sg.info()).replace(' ', '').replace('/', 'o').lower()
    print(f'outputfile {outputfile}.xplor {outputfile}.ccp4', file=fout)
    print('outputformat xplor ccp4', file=fout)
    print(file=fout)
    print('dataformat', dataformat, file=fout)
    print(f'fbegin {datafile}\n', file=fout)


def main(filename='sf.inflip'):
    """Creates a basic superflip input file for structure solution by asking a few simple questions"""

    for x in range(3):
        cell = input("Enter cell parameters:\n >> ")

        cell = cell.split()
        if len(cell) != 6:
            print('Expecting 6 parameters: a b c alpha beta gamma')
            continue
        else:
            try:
                cell = tuple(map(float, cell))
            except ValueError as e:
                print('ValueError:', e)
                continue
            else:
                break

    for x in range(3):
        spgr = input('Enter space group:\n >> ')

        if not spgr.split():
            continue
        else:
            break

    wavelength = input('Enter wavelength\n >> [1.54056] ') or '1.54056'
    composition = input('Enter composition:\n >> [skip] ') or ''
    datafile = input('Enter datafile:\n >> [fobs.out] ') or 'fobs.out'

    for x in range(3):
        dataformat = input(
            'Enter dataformat:\n >> [intensity fwhm] ') or 'intensity fwhm'
        if not all(i in ('intensity', 'amplitude', 'amplitude difference', 'a', 'b', 'phase', 'group', 'dummy', 'fwhm', 'm91', 'm90', 'shelx')
                   for i in dataformat.split()):
            print('Unknown dataformat, please enter any of\n intensity/amplitude/amplitude difference/a/b/phase/group/dummy/fwhm/m91/m90/shelx\n')
            continue
        else:
            break

    make_superflip(
        cell, spgr, wavelength, composition, datafile, dataformat, filename=filename)


if __name__ == '__main__':
    main()
