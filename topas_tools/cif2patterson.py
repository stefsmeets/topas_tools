#!/usr/bin/env cctbx.python

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

from __future__ import division

import argparse

import os, sys

from cif import reader, CifParserError

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"
__version__ = "28-04-2015"


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, file):
            structures = reader(file_object=f).build_crystal_structures()
        elif isinstance(f, str):
            structures = reader(file_path=f).build_crystal_structures()
        else:
            raise TypeError('read_cif: Can not deal with type {}'.format(type(f)))
    except CifParserError as e:
        print e
        print "Error parsing cif file, check if the data tag does not contain any spaces."
        sys.exit()
    for key, val in structures.items():
        print "\nstructure:", key
        val.show_summary().show_scatterers()
    return structures


usage = """cif2patterson structure.cif"""

description = """Notes: Takes any cif file and generated patterson map
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


parser.set_defaults(
    spgr="P1"
)

options = parser.parse_args()

cif = options.args
s = read_cif(cif).values()[0]
s = s.expand_to_p1()
print "Expanded to P1 => {} atoms".format(s.scatterers().size())
print

root, ext = os.path.splitext(cif)

uc = s.unit_cell()

scatterers = s.scatterers()

fout1 = open("patterson_dists.txt", 'w')
fout2 = open("patterson_full.txt", 'w')

distances_all = []

verbose = False
for atom1 in scatterers:
    x1, y1, z1 = atom1.site

    if verbose:
        print
        atom1.show()
    distances = []
    for atom2 in scatterers:
        if verbose:
            atom2.show()

        x2, y2, z2 = atom2.site

        dx, dy, dz = x1-x2, y1-y2, z1-z2

        dx = dx % 1
        dy = dy % 1
        dz = dz % 1

        length = uc.length((dx, dy, dz))

        distances.append((atom2.label, dx, dy, dz, length))

        if verbose:
            print ' --> {:>4s} {:9.5f} {:9.5f} {:9.5f}  {:9.5f}'.format(atom2.label, dx, dy, dz, length)

        # print atom1.label, '-->', atom2.label, '=', uc.length((dx,dy,dz))

    distances_all.extend(distances)

    atom1.show(fout2)
    for label, dx, dy, dz, distance in sorted(distances, key=lambda x: x[-1]):
        print >> fout2, ' --> {:>4s} {:9.5f} {:9.5f} {:9.5f}  {:9.5f}'.format(
            label, dx, dy, dz, distance)
    print >> fout2

print 'Wrote file', fout2.name

for label, dx, dy, dz, distance in sorted(distances_all, key=lambda x: x[-1]):
    print >> fout1, '{:9.5f}'.format(distance)

print 'Wrote file', fout1.name

fout1.close()
fout2.close()
