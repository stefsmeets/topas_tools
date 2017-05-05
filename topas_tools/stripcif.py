#!/usr/bin/env python

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
import os

from cif import reader, CifParserError

# from IPython.terminal.embed import InteractiveShellEmbed
# InteractiveShellEmbed.confirm_exit = False
# ipshell = InteractiveShellEmbed(banner1='')

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"
__version__ = "11-03-2015"


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, file):
            r = reader(file_object=f)
        elif isinstance(f, str):
            r = reader(file_path=f)
        else:
            raise TypeError('read_cif: Can not deal with type {}'.format(type(f)))
    except CifParserError as e:
        print e
        print "Error parsing cif file, check if the data tag does not contain any spaces."
        sys.exit()
    structures = r.build_crystal_structures()
    for key, val in structures.items():
        print "\nstructure:", key
        val.show_summary().show_scatterers()
    return structures


def main():
    description = """Notes:
    """

    epilog = 'Updated: {}'.format(__version__)

    parser = argparse.ArgumentParser(  # usage=usage,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        version=__version__)

    parser.add_argument("args",
                        type=str, metavar="FILE",
                        help="Path to input cif")

    options = parser.parse_args()

    cif = options.args
    s = read_cif(cif).values()[0]

    # s = s.expand_to_p1()

    root, ext = os.path.splitext(cif)
    out = root + "_simple" + ext

    s.as_cif_simple(out=open(out, 'w'))
    print " >> Wrote file {}".format(out)

if __name__ == '__main__':
    main()
