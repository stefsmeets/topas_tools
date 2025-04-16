import argparse
import sys

import numpy as np

from .blender_mini import (
    calc_dspacing,
    calc_structure_factors,
    centering_vectors,
    load_hkl,
    read_cif,
    reduce_all,
    write_hkl,
)


def print_superflip(sgi, uc, fout, fdiff_file=None):
    """Print an inflip file to be used with superflip for difference fourier maps.

    - Tested and works fine with: EDI, SOD

    sgi: cctbx space_group_info()
    uc : cctbx unit_cell()
    """
    print('title', 'superflip\n', file=fout)

    print('dimension 3', file=fout)
    print('voxel', end=' ', file=fout)
    for p in uc.parameters()[0:3]:
        print(int(((p * 4) // 6 + 1) * 6), end=' ', file=fout)
    print(file=fout)
    print('cell', end=' ', file=fout)
    for p in uc.parameters():
        print(p, end=' ', file=fout)
    print('\n', file=fout)

    print('centers', file=fout)
    for cvec in centering_vectors[sgi.type().group().conventional_centring_type_symbol()]:
        print(' '.join(cvec), file=fout)
    print('endcenters\n', file=fout)

    print('symmetry #', sgi.symbol_and_number(), file=fout)
    print('# inverse no', file=fout)

    # number of unique symops, no inverses
    n_smx = sgi.type().group().n_smx()
    # number of primitive symops, includes inverses
    order_p = sgi.type().group().order_p()

    # this should work going by the assumption that
    # the unique primitive symops are stored first,
    # THEN the inverse symops and then all the symops due to centering.

    for n, symop in enumerate(sgi.type().group()):
        if n == order_p:
            break
        elif n == n_smx:
            print('# inverse yes, please check!', file=fout)
        print(symop, file=fout)

    print('endsymmetry\n', file=fout)

    print('perform fourier', file=fout)
    print('terminal yes\n', file=fout)

    print('expandedlog yes', file=fout)
    print('outputfile superflip.xplor', file=fout)
    print('outputformat xplor\n', file=fout)

    print('dataformat amplitude phase', file=fout)

    if not fdiff_file:
        raise ValueError('fdiff_file was not supplied')

    print('fbegin fdiff.out\n', file=fout)


def run_script(gui_options=None):
    description = """Notes:
    """

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('args', type=str, metavar='FILE', help='Path to input cif')

    parser.add_argument(
        '--diff',
        type=str,
        metavar='FILE',
        dest='diff',
        help=(
            'Path to file with observed amplitudes to diff with the input cif. '
            'Format: h k l F [phase]. This uses the indices for the observed reflections. '
            'Use the macro "Out_fobs(fobs.out)" in TOPAS for to output observed structure '
            'factors. Scale can be directly copied from topas file or generated automatically. '
            'Full command: sfc structure.cif --diff fobs.out'
        ),
    )

    parser.add_argument(
        '-r', '--dmin', type=float, metavar='d_min', dest='dmin', help='maximum resolution'
    )

    parser.add_argument(
        '-t',
        '--table',
        type=str,
        metavar='TABLE',
        dest='table',
        help='Choose scattering factor table [x-ray, neutron, electron], default=X-ray',
    )

    parser.set_defaults(
        dmin=None,
        diff=None,
        gui=False,
        superflip_path=None,
        run_superflip=False,
        scale=None,
        table='xray',
    )

    options = parser.parse_args()

    if gui_options:
        for k, v in list(gui_options.items()):
            setattr(options, k, v)

    cif = options.args
    topas_scale = options.scale
    fobs_file = options.diff
    dmin = options.dmin
    table = options.table

    if not cif or not fobs_file:
        print(
            'Error: Supply cif file and use `--diff fobs.out` '
            'to specify file with fobs (hkl + structure factors)'
        )
        sys.exit()

    s = list(read_cif(cif).values())[0]

    uc = s.unit_cell()
    cell = uc.parameters()
    a, b, c, al, be, ga = cell

    sgi = s.space_group().info()
    spgr = str(sgi)

    df = load_hkl(fobs_file, ('fobs',))

    # df = files2df(files)
    if not dmin:
        dmin = calc_dspacing(df, cell, inplace=False).min()

    fcalc = list(calc_structure_factors(cif, dmin=dmin, table=table).values())[0]

    df = df.combine_first(fcalc)

    df = reduce_all(df, cell, spgr)

    # remove NaN reflections (low angle typically)
    df = df[df['fobs'].notnull()]

    if not topas_scale:
        topas_scale = (
            input('Topas scale? >> [auto] ')
            .replace('`', '')
            .replace('@', '')
            .replace('scale', '')
            .strip()
        )

    if topas_scale:
        scale = float(topas_scale) ** 0.5
        print(f'Fobs scaled by {1 / scale} [=sqrt(1/{(float(topas_scale))})]')
    else:
        scale = df['fobs'].sum() / df['fcalc'].sum()
        print(f'No scale given, approximated as {scale} (sum(fobs) / sum(fcal))')

    df['fdiff'] = df['fobs'] / scale - df['fcalc']
    df['sfphase'] = df['phases'] / (2 * np.pi)

    sel = df['fdiff'] <= 0

    df.loc[sel, 'fdiff'] = abs(df.loc[sel, 'fdiff'])
    df.loc[sel, 'sfphase'] += 0.5

    cols = ('fdiff', 'sfphase')
    write_hkl(df, cols=cols, out='fdiff.out')

    print_superflip(sgi, uc, fout=open('sf.inflip', 'w'), fdiff_file='fdiff.out')

    if options.run_superflip:
        import subprocess as sp

        sp.call(f'{options.superflip_path} sf.inflip')


def main(options=None):
    if len(sys.argv) > 1 and sys.argv[1] == 'gui':
        from . import topasdiff_gui

        topasdiff_gui.run()
    else:
        run_script()


if __name__ == '__main__':
    main()
