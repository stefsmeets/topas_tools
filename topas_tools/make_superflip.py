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
    'F': (
        ['0.0', '0.0', '0.0'],
        ['0.0', '0.5', '0.5'],
        ['0.5', '0.0', '0.5'],
        ['0.5', '0.5', '0.0'],
    ),
}


def make_special_position_settings(cell, space_group, min_dist_sym_equiv=0.000001):
    """Takes cell and space group, returns cctbx structure
    input:  cell: (a b c alpha beta gamma) as a tuple
            space_group: 'space_group' like a string
            min_dist_sym_equiv: float
    output: <cctbx.crystal.special_position_settings object>
                contains cell, space group and special positions
    """

    assert isinstance(cell, tuple), 'cell must be supplied as a tuple'
    assert len(cell) == 6, 'expected 6 cell parameters'

    special_position_settings = crystal.special_position_settings(
        crystal_symmetry=crystal.symmetry(unit_cell=cell, space_group_symbol=space_group),
        min_distance_sym_equiv=min_dist_sym_equiv,
    )
    return special_position_settings


def make_superflip(
    cell,
    spgr,
    wavelength,
    composition,
    datafile,
    dataformat,
    filename='sf.inflip',
):
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
        print(int(((p * 4) // 6 + 1) * 6), end=' ', file=fout)
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
    """Create superflip input file for structure solution through a few simple questions"""
    spgr = None
    cell = None
    dataformat = None

    for x in range(3):
        cell = input('Enter cell parameters:\n >> ')

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
        dataformat = input('Enter dataformat:\n >> [intensity fwhm] ') or 'intensity fwhm'
        data_keys = (
            'intensity',
            'amplitude',
            'amplitude difference',
            'a',
            'b',
            'phase',
            'group',
            'dummy',
            'fwhm',
            'm91',
            'm90',
            'shelx',
        )
        if not all(i in data_keys for i in dataformat.split()):
            print(
                'Unknown dataformat, please enter any of\n'
                'intensity/amplitude/amplitude difference/\n'
                'a/b/phase/group/dummy/fwhm/m91/m90/shelx\n'
            )
            continue
        else:
            break

    assert dataformat, 'dataformat not defined'
    assert cell, 'cell not defined'
    assert spgr, 'cpace group not defined'

    make_superflip(cell, spgr, wavelength, composition, datafile, dataformat, filename=filename)


if __name__ == '__main__':
    main()
