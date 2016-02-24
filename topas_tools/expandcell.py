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

from __future__ import division

import argparse

from cctbx import xray
from cctbx import crystal
from cctbx.array_family import flex
import os

#from IPython.terminal.embed import InteractiveShellEmbed
#InteractiveShellEmbed.confirm_exit = False
#ipshell = InteractiveShellEmbed(banner1='')

from cif import reader, CifParserError

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"
__version__ = "11-03-2015"


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
        exit()
    for key, val in structures.items():
        print "\nstructure:", key
        val.show_summary().show_scatterers()
    return structures


usage = """"""

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

parser.add_argument("-x",
                    type=int, metavar="N", dest="expand_x",
                    help="""Expand N times along x direction""")

parser.add_argument("-y",
                    type=int, metavar="N", dest="expand_y",
                    help="""Expand N times along y direction""")

parser.add_argument("-z",
                    type=int, metavar="N", dest="expand_z",
                    help="""Expand N times along z direction""")

parser.add_argument("-e", "--exclude",
                    type=str, metavar="X", nargs="+", dest="exclude",
                    help="""Exclude these elements from final output""")

parser.add_argument("-f", "--shift",
                    type=float, metavar="dx dy dz", nargs=3, dest="shift",
                    help="""Applies shift to coordinates, this number comes directly from PLATON (Origin shifted to ... )""")

parser.add_argument("-s", "--spgr",
                    type=str, metavar="P1", dest="spgr",
                    help="""Apply this symmetry to the new structure (default = P1)""")


parser.set_defaults(
    expand_x=1,
    expand_y=1,
    expand_z=1,
    exclude=(),
    shift=[],
    spgr="P1"
)

options = parser.parse_args()

cif = options.args
s = read_cif(cif).values()[0]

excluded = []
if options.exclude:
    for atom in s.scatterers():
        if atom.element_symbol() in options.exclude:
            excluded.append(atom)

s = s.expand_to_p1()
print "Expanded to P1 => {} atoms".format(s.scatterers().size())
print

root, ext = os.path.splitext(cif)

expand_x = options.expand_x
expand_y = options.expand_y
expand_z = options.expand_z

spgr = options.spgr
shift = options.shift

assert expand_x > 0, "N must be bigger than 0"
assert expand_y > 0, "N must be bigger than 0"
assert expand_z > 0, "N must be bigger than 0"

out = root
for direction, number in enumerate((expand_x, expand_y, expand_z)):
    if number == 1:
        continue
    out += "_{}{}".format(number, 'xyz'[direction])
if expand_x * expand_y * expand_z == 1:
    if spgr == "P1":
        out += "_P1"
    else:
        out += "_new"
out += ext

cell = s.unit_cell().parameters()

new_cell = (cell[0] * expand_x, cell[1] * expand_y,
            cell[2] * expand_z, cell[3], cell[4], cell[5])

print "old cell:"
print cell
print
print "new cell:"
print new_cell
print


def expand_cell(scatterers, direction, number):
    for atom in scatterers:
        print atom.label, number
        site = list(atom.site)

        coord = site[direction]

        for n in range(number):
            n = float(n)
            new = (coord / number) + (n / number)
            site[direction] = new

            label = atom.label+chr(97+int(n))  # 97 => a
            yield xray.scatterer(label=label, site=site, u=atom.u_iso, occupancy=atom.occupancy)


if options.exclude:
    scatterers = excluded
    print ">> Excluding these atoms:", ", ".join(options.exclude)
    print
    for atom in s.scatterers():
        if atom.element_symbol() in options.exclude:
            continue
        else:
            scatterers.append(atom)
else:
    scatterers = s.scatterers()

print "Starting with {} scatterers".format(len(scatterers))
print

for direction, number in enumerate((expand_x, expand_y, expand_z)):
    if number == 1:
        continue
    print " >> Expanding cell along {} by {}".format('xyz'[direction], number)
    scatterers = [
        scatterer for scatterer in expand_cell(scatterers, direction, number)]
    print 'New number of scatterers:', len(scatterers)
    print


print " >> Applying spacegroup {}".format(spgr)
print

s = xray.structure(
    crystal_symmetry=crystal.symmetry(
        unit_cell=new_cell,
        space_group_symbol=spgr),
    scatterers=flex.xray_scatterer(scatterers))

if shift:
    # shift = [-1*number for number in shift]
    print ">> Applying shift {} to all atom sites".format(shift)
    print
    s = s.apply_shift(shift)

if spgr != "P1":
    # shift all atoms to inside unit cell, so asu can be applied
    s = s.sites_mod_positive()

    asu = s.space_group_info().brick().as_string()
    print "Asymmetric unit:"
    # print box_min, "=>", box_max
    print asu
    print
    asu_x, asu_y, asu_z = asu.split(';')
    fx = lambda x: eval(asu_x)
    fy = lambda y: eval(asu_y)
    fz = lambda z: eval(asu_z)

    if '-' in asu:
        print "Did not account for negative values in asymmetric unit. Duplicate atoms cannot be removed (use Kriber)"
        print ""
        print "Remove a,b,c etc from atom labels in cif, then run:"
        print " > cif2strudat", out
        print " > kriber"
        print " >> reacs global"
        print " >> wricif"
        print
    else:
        scatterers = []
        for atom in s.scatterers():
            x, y, z = atom.site

            if fx(x) and fy(y) and fz(z):
                scatterers.append(atom)

        s = xray.structure(
            crystal_symmetry=crystal.symmetry(
                unit_cell=new_cell,
                space_group_symbol=spgr),
            scatterers=flex.xray_scatterer(scatterers))

        print " >> Removing duplicate atoms, reduced number to {} atoms".format(s.scatterers().size())
        print

s.as_cif_simple(out=open(out, 'w'))
print " >> Wrote file {}".format(out)
print

if (not shift) and (spgr == "P1"):
    print "---"
    print "To find the right symmetry of the expanded unit cell:"
    print
    print "Run Platon, and then the Addsym routine"
    print "Select NoSubCell in sidebar, then run ADDSYMExact"
    print "Note the space group, and Origin shift"
    print
    print "Rerun expandcell with --shift X Y Z --spgr SPGR"
