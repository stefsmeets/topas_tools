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

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mat.ethz.ch"

from itertools import combinations

"""
x. Setting up restraints with topas_restraints.py
	- Go to folder topas_tools/
	- Copy the output from append_bond_lengths for all Si/O atoms to bonds.txt

	Note: The program uses this output to detect with bonds fall within the criteria for zeolites (i.e. bond lengths of 1.61 with some specified tolerance), and uses this connectivity to set up the restraints.

	- Run: >> python topas_restraints.py
		The restraints will be printed to the file restraints.out, these can be copy/pasted directly into the Topas input file.

	Note: There is no filtering of duplicate or symmetry related bonds, so these should be filtered out manually
"""

distdat = {
('T',  'O') : (1.610,    0.01),
('SI', 'O') : (1.610,    0.01),
('AL', 'O') : (1.740,    0.01),
('P',  'O') : (1.520,    0.01),
('GE', 'O') : (1.750,    0.01),
('GA', 'O') : (1.800,    0.01),
('B',  'O') : (1.460,    0.01),
'tot':(109.47 ,  2.0),
'oto':(145    ,  8.0)
}

# location of strings, assuming fixed width
atom1 = [0,10]
atom2 = [10,33]
bondl = [33,41]


organic = False

if not organic:
	# ideal distances/angles
	to_dist = 1.61
	tot_ang = 145.0
	oto_ang = 109.5
	
	# tolerance for T--O bond detection
	to_tol  = 0.2
if organic:
	# ideal distances/angles
	to_dist = 1.81
	tot_ang = 145.0
	oto_ang = 109.5

	# tolerance for T--O bond detection

def gen_section(f):
	part = []
	for line in f:
		a1 = line[atom1[0]:atom1[1]]
		a2 = line[atom2[0]:atom2[1]]
		ln = line[bondl[0]:bondl[1]]
		ln = float(ln)
		if a1.split():
			if len(part) > 0:
				part = [item.replace(':',' ') for item in part]
				print
				yield part
			part = [a1]
			print a1, '|',

		if to_dist-to_tol < ln < to_dist+to_tol:
			if not a1.split() and len(part) > 1:
				print '           |',
			part.append(a2)
			print a2, '|', ln
		else:
			pass
	if len(part) > 0:
		part = [item.replace(':',' ') for item in part]
		print
		yield part
	to_tol  = 0.8

def main():
	
	# sigmas
	to_s  = 0.01
	tot_s = 10.0
	oto_s = 0.8
	
	# weights
	to_w  = 1/(to_s**2)
	tot_w = 1/(tot_s**2)
	oto_w = 1/(oto_s**2)
	
	f = open('bonds.txt','r')
	fout = open('restraints.out','w')
	
	
	gs = gen_section(f)
	
	for part in gs:
		main = part[0]
		nbonds = len(part) - 1
	
		match = 0
	
		if 'Si' in main:
			match += 1
			if nbonds != 4:
				print '*** Warning: More/less than _4_ bonds detected for {}... bonds = {}\n'.format(main,nbonds)
	
			for ox in part[1:]:
				print >> fout, '      Distance_Restrain( {} {} , {}, 0.0, 0.0, {} )'.format(main,ox,to_dist,to_w)
	
			for ox1,ox2 in combinations(part[1:],2):
	
				print >> fout, '      Angle_Restrain( {} {} {} , {}, 0.0, 0.0, {} )'.format(ox1,main,ox2,oto_ang,oto_w)
	
		if 'O' in main:
			match += 1
			if nbonds != 2:
				print '*** Warning: More/less than _2_ bonds detected for {}... bonds = {}\n'.format(main,nbonds)
	
			for si1,si2 in combinations(part[1:],2):
	
				print >> fout, '      Angle_Restrain( {} {} {} , {}, 0.0, 0.0, {} )'.format(si1,main,si2,tot_ang,tot_w)
	
		if match == 0:
			print '*** Non-Si/O detected --> {}\n'.format(main)
	
			for ox in part[1:]:
				print >> fout, '      Distance_Restrain( {} {} , {}, 0.0, 0.0, {} )'.format(main,ox,to_dist,to_w)
	
			for ox1,ox2 in combinations(part[1:],2):
	
				print >> fout, '      Angle_Restrain( {} {} {} , {}, 0.0, 0.0, {} )'.format(ox1,main,ox2,oto_ang,oto_w)

if __name__ == '__main__':
	main()