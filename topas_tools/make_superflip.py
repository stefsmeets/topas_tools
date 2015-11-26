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

from sys import argv

from cctbx import crystal

# from cmath import phase, polar, pi

__author__ = "Stef Smeets"
__email__ = "stef.smeets@mat.ethz.ch"
__version__ = '2015-11-07'


centering_vectors = {
'P':(['0.0','0.0','0.0'],),
'A':(['0.0','0.0','0.0'],['0.0','0.5','0.5']),
'B':(['0.0','0.0','0.0'],['0.5','0.0','0.5']),
'C':(['0.0','0.0','0.0'],['0.5','0.5','0.0']),
'I':(['0.0','0.0','0.0'],['0.5','0.5','0.5']),
'R':(['0.0','0.0','0.0'],['2/3','1/3','1/3'],['1/3','2/3','2/3']),
'S':(['0.0','0.0','0.0'],['1/3','2/3','2/3'],['2/3','1/3','1/3']),
'T':(['0.0','0.0','0.0'],['1/3','2/3','2/3'],['2/3','1/3','2/3']),
'H':(['0.0','0.0','0.0'],['2/3','1/3','0.0'],['1/3','2/3','0.0']),
'K':(['0.0','0.0','0.0'],['1/3','0.0','2/3'],['2/3','0.0','1/3']),
'L':(['0.0','0.0','0.0'],['0.0','2/3','1/3'],['0.0','1/3','2/3']),
'F':(['0.0','0.0','0.0'],['0.0','0.5','0.5'],['0.5','0.0','0.5'],['0.5','0.5','0.0'])
}


def make_special_position_settings(cell,space_group,min_dist_sym_equiv=0.5):
	"""Takes cell and space group, returns cctbx structure
	input:	cell: (a b c alpha beta gamma) as a tuple
			space_group: 'space_group' like a string
			min_dist_sym_equiv: float
	output: <cctbx.crystal.special_position_settings object>
				contains cell, space group and special positions
	"""

	assert type(cell) == tuple, 'cell must be supplied as a tuple'
	assert len(cell) == 6, 'expected 6 cell parameters'

	special_position_settings = crystal.special_position_settings(
		crystal_symmetry = crystal.symmetry(
			unit_cell = cell,
			space_group_symbol = space_group),
		min_distance_sym_equiv = min_dist_sym_equiv)
	return special_position_settings


def print_superflip(fcalc,fout, fdiff_file = None):
	"""Prints an inflip file that can directly be used with superflip for difference fourier maps

	- Tested and works fine with: EDI, SOD
	"""
	print >> fout, 'title', 'superflip\n'

	print >> fout, 'dimension 3'
	print >> fout, 'voxel',
	for p in fcalc.unit_cell().parameters()[0:3]:
		print >> fout, int(((p*4) // 6 + 1) * 6),
	print >> fout
	print >> fout, 'cell',
	for p in fcalc.unit_cell().parameters():
		print >> fout, p,
	print >> fout, '\n'
	
	print >> fout, 'centers'
	for cvec in centering_vectors[fcalc.space_group_info().type().group().conventional_centring_type_symbol()]:
		print >> fout, ' '.join(cvec)
	print >> fout, 'endcenters\n'

	print >> fout, 'symmetry #', fcalc.space_group_info().symbol_and_number()
	print >> fout, '# inverse no'
	
	n_smx = fcalc.space_group_info().type().group().n_smx()			# number of unique symops, no inverses
	order_p = fcalc.space_group_info().type().group().order_p()		# number of primitive symops, includes inverses
	order_z = fcalc.space_group_info().type().group().order_z()		# total number of symops

	# this should work going by the assumption that the unique primitive symops are stored first,
	# THEN the inverse symops and then all the symops due to centering.

	for n,symop in enumerate(fcalc.space_group_info().type().group()):
		if n == order_p:
			break
		elif n == n_smx:
			print >> fout, '# inverse yes, please check!'
		print >> fout, symop

		### Broken, because .inverse() doesn't work, but probably a better approach:
	#for symop in f.space_group_info().type().group().smx():
	#	print >> fout, symop
	#if f.space_group_info().type().group().is_centric():
	#	print >> fout, '# inverse yes'
	#	for symop in f.space_group_info().type().group().smx():
	#		print >> fout, symop.inverse() # inverse does not work?
	
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
		print_simple(fcalc,fout,output_phases='cycles')

#		for i,(h,k,l) in enumerate(f.indices()):
#			# structurefactor = abs(f.data()[i])
#			# phase = phase(f.data()[i]
#			print >> fout, "%3d %3d %3d %10.6f %10.3f" % (
#				h,k,l, abs(f.data()[i]), phase(f.data()[i]) / (2*pi) )
		print >> fout, 'endf'

def make_superflip(cell, spgr, wavelength, composition, datafile, dataformat, filename='sf.inflip'):
	sps = make_special_position_settings(cell,spgr)
	sg = sps.space_group()
	uc = sps.unit_cell()
	#sgi = sps.space_group_info()

	fout = open(filename,'w')

	print >> fout, 'title', filename.split('.')[0]
	print >> fout
	print >> fout, 'dimension 3'
	print >> fout, 'voxel',
	for p in uc.parameters()[0:3]:
		print >> fout, int(((p*4) // 6 + 1) * 6), 
	print >> fout
	print >> fout, 'cell',
	for p in uc.parameters():
		print >> fout, p,
	print >> fout, '  # vol = {:.4f} A3 \n'.format(uc.volume())
	
	print >> fout, 'centers'
	for cvec in centering_vectors[sg.conventional_centring_type_symbol()]:
		print >> fout, '  ', ' '.join(cvec)
	print >> fout, 'endcenters\n'

	print >> fout, 'symmetry #', sg.crystal_system(), sg.info()
	print >> fout, '# +(0 0 0) Inversion-Flag = 0'
	
	n_smx = sg.n_smx()
	order_p = sg.order_p()
	order_z = sg.order_z()

	for n,symop in enumerate(sg):
		if n == order_p:
			break
		elif n == n_smx:
			print >> fout, '# +(0 0 0) Inversion-Flag = 1'
		print >> fout, '  ', symop
	print >> fout, 'endsymmetry\n'
	print >> fout, 'derivesymmetry yes'
	print >> fout, 'searchsymmetry average'
	print >> fout
	print >> fout, 'delta AUTO'
	print >> fout, 'weakratio 0.00'
	print >> fout, 'biso 2.0'
	print >> fout, 'randomseed AUTO'
	print >> fout
	if composition:
		print >> fout, 'composition {}'.format(composition)
		print >> fout, 'histogram composition'
		print >> fout, 'hmparameters 10 5'
	else:
		print >> fout, '#composition #composition goes here'
		print >> fout, '#histogram composition'
		print >> fout, '#hmparameters 10 5'
	print >> fout 
	print >> fout, 'fwhmseparation 0.3'
	print >> fout, 'lambda {}'.format(wavelength)
	print >> fout 
	print >> fout, 'maxcycles 200'
	print >> fout, 'bestdensities 10'
	print >> fout, 'repeatmode 100'
	print >> fout
	print >> fout, 'polish yes'
	print >> fout, 'convergencemode never'
	print >> fout
	print >> fout, '#referencefile filename.cif'
	print >> fout, '#modelfile filename.cif 0.2'
	print >> fout
	print >> fout, 'terminal yes'
	print >> fout, 'expandedlog yes'
	outputfile = str(sg.info()).replace(' ','').replace('/','o').lower()
	print >> fout, 'outputfile {}.xplor {}.ccp4'.format(outputfile,outputfile)
	print >> fout, 'outputformat xplor ccp4'
	print >> fout
	print >> fout, 'dataformat', dataformat
	print >> fout, 'fbegin {}\n'.format(datafile)


def main(filename='sf.inflip'):
	"""Creates a basic superflip input file for structure solution by asking a few simple questions"""

	for x in xrange(3):
		cell = raw_input("Enter cell parameters:\n >> ")

		cell = cell.split()
		if len(cell) != 6:
			print 'Expecting 6 parameters: a b c alpha beta gamma'
			continue
		else:
			try:
				cell = tuple(map(float,cell))
			except ValueError, e:
				print 'ValueError:', e
				continue
			else:
				break

	for x in xrange(3):
		spgr = raw_input('Enter space group:\n >> ')

		if not spgr.split():
			continue
		else:
			break		
	
	wavelength  = raw_input('Enter wavelength\n >> [1.54056] ') or '1.54056'
	composition = raw_input('Enter composition:\n >> [skip] ') or ''
	datafile    = raw_input('Enter datafile:\n >> [fobs.out] ') or 'fobs.out'

	for x in xrange(3):
		dataformat = raw_input('Enter dataformat:\n >> [intensity fwhm] ') or 'intensity fwhm'
		if not all(i in ('intensity','amplitude','amplitude difference','a','b','phase','group','dummy','fwhm','m91','m90','shelx')
					for i in dataformat.split()):
			print 'Unknown dataformat, please enter any of\n intensity/amplitude/amplitude difference/a/b/phase/group/dummy/fwhm/m91/m90/shelx\n'
			continue
		else:
			break

	make_superflip(cell, spgr, wavelength, composition, datafile, dataformat, filename=filename)


if __name__ == '__main__':
	main()
	