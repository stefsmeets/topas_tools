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

import re
import sys

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mmk.su.se"


def main():
    pat = re.compile('[a-zA-Z]+')

    if len(sys.argv) == 1:
        print "Usage: fh2topas.py ~/path/to/output.fh [n]"
        exit()

    args = sys.argv[2:]

    moltotal = 1
    if args:
        moltotal = int(args[0])
    molnum = 1

    fin = open(sys.argv[1])

    lines = fin.readlines()

    molnum = 1
    while molnum <= moltotal:
        if moltotal == 1:
            molnum = ''

        number = None
        n = 0
        d = {}
        all_atoms = []

        print '      rigid'

        for line in lines:
            inp = line.split()

            if not inp:
                continue
            if not number:
                number = int(inp[0])
                continue
            n += 1

            print '         z_matrix ',

            if n > 0:
                tpe = inp[0]

                atom = '{}{}'.format(tpe, n)
                d[str(n)] = atom
                all_atoms.append(atom)

                scatterer = re.findall(pat, atom)[0]
                atom = atom.replace(scatterer, scatterer+str(molnum))
                print '{:6s}'.format(atom),

            if n > 1:
                bonded_with, bond = inp[1:3]
                bonded_with = d[bonded_with]
                scatterer = re.findall(pat, bonded_with)[0]
                bonded_with = bonded_with.replace(
                    scatterer, scatterer+str(molnum))
                print '{:6s} {:7s}'.format(bonded_with, bond),
            if n > 2:
                angle_with, angle = inp[3:5]
                angle_with = d[angle_with]
                scatterer = re.findall(pat, angle_with)[0]
                angle_with = angle_with.replace(
                    scatterer, scatterer+str(molnum))
                print '{:6s} {:7s}'.format(angle_with, angle),
            if n > 3:
                torsion_with, torsion = inp[5:7]
                torsion_with = d[torsion_with]
                scatterer = re.findall(pat, torsion_with)[0]
                torsion_with = torsion_with.replace(
                    scatterer, scatterer+str(molnum))
                print '{:6s} {:>7s}'.format(torsion_with, torsion),

            print

        print """
             Rotate_about_axies(@ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors)
             Translate(         @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors)
        """

        if molnum == '':
            break
        else:
            molnum += 1

    molnum = 1
    while molnum <= moltotal:
        if moltotal == 1:
            molstr = ''
        else:
            molstr = str(molnum)
        print
        print "      prm !occ{} 0.5 min 0.0 max  1.0".format(molnum)
        print "      prm !beq{} 3.0 min 1.0 max 10.0".format(molnum)
        print

        for atom in all_atoms:
            scatterer = re.findall(pat, atom)[0]
            atom = atom.replace(scatterer, scatterer+molstr)
            print '      site {:6s} x 0.0 y 0.0 z 0.0 occ {:2s} =occ{}; beq =beq{};'.format(atom, scatterer, molnum, molnum)
        molnum += 1


if __name__ == '__main__':
    main()
