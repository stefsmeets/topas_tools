#!/usr/bin/env python2.7

#    topas_tools - set of scripts to help using Topas
#    Copyright (C) 2015 Stef Smeets
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import argparse
from blender_mini import *
import sys

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"
__version__ = "2015-11-26"


def print_superflip(sgi, uc, fout, fdiff_file=None):
    """Prints an inflip file that can directly be used with superflip for difference fourier maps

    - Tested and works fine with: EDI, SOD

    sgi: cctbx space_group_info()
    uc : cctbx unit_cell()
    """
    print >> fout, 'title', 'superflip\n'

    print >> fout, 'dimension 3'
    print >> fout, 'voxel',
    for p in uc.parameters()[0:3]:
        print >> fout, int(((p*4) // 6 + 1) * 6),
    print >> fout
    print >> fout, 'cell',
    for p in uc.parameters():
        print >> fout, p,
    print >> fout, '\n'

    print >> fout, 'centers'
    for cvec in centering_vectors[sgi.type().group().conventional_centring_type_symbol()]:
        print >> fout, ' '.join(cvec)
    print >> fout, 'endcenters\n'

    print >> fout, 'symmetry #', sgi.symbol_and_number()
    print >> fout, '# inverse no'

    # number of unique symops, no inverses
    n_smx = sgi.type().group().n_smx()
    # number of primitive symops, includes inverses
    order_p = sgi.type().group().order_p()
    order_z = sgi.type().group().order_z()      # total number of symops

    # this should work going by the assumption that the unique primitive symops are stored first,
    # THEN the inverse symops and then all the symops due to centering.

    for n, symop in enumerate(sgi.type().group()):
        if n == order_p:
            break
        elif n == n_smx:
            print >> fout, '# inverse yes, please check!'
        print >> fout, symop

    # Broken, because .inverse() doesn't work, but probably a better approach:
    # for symop in f.space_group_info().type().group().smx():
    #   print >> fout, symop
    # if f.space_group_info().type().group().is_centric():
    #   print >> fout, '# inverse yes'
    #   for symop in f.space_group_info().type().group().smx():
    #       print >> fout, symop.inverse() # inverse does not work?

    print >> fout, 'endsymmetry\n'

    print >> fout, 'perform fourier'
    print >> fout, 'terminal yes\n'

    print >> fout, 'expandedlog yes'
    print >> fout, 'outputfile superflip.xplor'
    print >> fout, 'outputformat xplor\n'

    print >> fout, 'dataformat amplitude phase'

    if fdiff_file:
        print >> fout, 'fbegin fdiff.out\n'
    else:
        print >> fout, 'fbegin'
        print_simple(fcalc, fout, output_phases='cycles')

#       for i,(h,k,l) in enumerate(f.indices()):
#           # structurefactor = abs(f.data()[i])
#           # phase = phase(f.data()[i]
#           print >> fout, "%3d %3d %3d %10.6f %10.3f" % (
#               h,k,l, abs(f.data()[i]), phase(f.data()[i]) / (2*pi) )
        print >> fout, 'endf'


def run_script(gui_options=None):
    description = """Notes:
    """

    epilog = 'Updated: {}'.format(__version__)

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        version=__version__)

    parser.add_argument("args",
                        type=str, metavar="FILE",
                        help="Path to input cif")

    parser.add_argument("--diff",
                        type=str, metavar="FILE", dest="diff",
                        help="""Path to file with observed amplitudes to diff with the input cif. Format: h k l F [phase].
                            This uses the indices for the observed reflections. Use the macro 'Out_fobs(fobs.out)' in TOPAS for
                            to output observed structure factors. Scale can be directly copied from topas file or generated automatically.
                            Full command: sfc structure.cif --diff fobs.out""")

    parser.add_argument("-r", "--dmin",
                        type=float, metavar="d_min", dest="dmin",
                        help="maximum resolution")

    parser.set_defaults(
        dmin=None,
        diff=None,
        gui=False,
        superflip_path=None,
        run_superflip=False,
        scale=None
    )

    options = parser.parse_args()

    if gui_options:
        for k,v in gui_options.items():
            setattr(options, k, v)

    cif = options.args
    topas_scale = options.scale
    fobs_file = options.diff
    dmin = options.dmin

    if not cif or not fobs_file:
        print "Error: Supply cif file and use --diff fobs.out to specify file with fobs (hkl + structure factors)"
        sys.exit()

    s = read_cif(cif).values()[0]

    uc = s.unit_cell()
    cell = uc.parameters()
    a, b, c, al, be, ga = cell

    sgi = s.space_group().info()
    spgr = str(sgi)

    df = load_hkl(fobs_file, ('fobs',))

    # df = files2df(files)
    if not dmin:
        dmin = calc_dspacing(df, cell, inplace=False).min()

    fcalc = calc_structure_factors(cif, dmin=dmin).values()[0]
    df = df.combine_first(fcalc)

    df = reduce_all(df, cell, spgr)

    # remove NaN reflections (low angle typically)
    df = df[df['fobs'].notnull()]

    if not topas_scale:
        topas_scale = raw_input('Topas scale? >> [auto] ').replace(
            '`', '').replace('@', '').replace('scale', '').strip()

    if topas_scale:
        scale = float(topas_scale)**0.5
        print 'Fobs scaled by {} [=sqrt(1/{})]'.format(1/scale, (float(topas_scale)))
    else:
        scale = df['fobs'].sum() / df['fcalc'].sum()
        print "No scale given, approximated as {} (sum(fobs) / sum(fcal))".format(scale)

    df['fdiff'] = df['fobs']/scale - df['fcalc']
    df['sfphase'] = df['phases'] / (2*np.pi)

    sel = df['fdiff'] <= 0

    df.loc[sel, 'fdiff'] = abs(df.loc[sel, 'fdiff'])
    df.loc[sel, 'sfphase'] += 0.5

    cols = ('fdiff', 'sfphase')
    write_hkl(df, cols=cols, out='fdiff.out')

    print_superflip(
        sgi, uc, fout=open('sf.inflip', 'w'), fdiff_file='fdiff.out')

    if options.run_superflip:
        import subprocess as sp
        sp.call("{} sf.inflip".format(options.superflip_path))

def main(options=None):
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        import topasdiff_gui
        topasdiff_gui.run()
    else:
        run_script()

if __name__ == '__main__':
    main()
