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

import argparse

from cctbx import xray
from cctbx import crystal
from cctbx.array_family import flex
import os

# from IPython.terminal.embed import InteractiveShellEmbed
# InteractiveShellEmbed.confirm_exit = False
# ipshell = InteractiveShellEmbed(banner1='')

__version__ = "11-03-2015"

def read_cif(f):
	"""opens cif and returns cctbx data object"""
	from iotbx.cif import reader, CifParserError
	try:
		if isinstance(f,file):
			structures = reader(file_object=f).build_crystal_structures()
		elif isinstance(f,str):
			structures = reader(file_path=f).build_crystal_structures()
		else:
			raise TypeError, 'read_cif: Can not deal with type {}'.format(type(f))
	except CifParserError as e:
		print e
		print "Error parsing cif file, check if the data tag does not contain any spaces."
		exit()
	for key,val in structures.items():
		print "\nstructure:", key
		val.show_summary().show_scatterers()
	return structures



usage = """"""

description = """Notes:
"""	
	
epilog = 'Updated: {}'.format(__version__)
	
parser = argparse.ArgumentParser(#usage=usage,
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

#s = s.expand_to_p1()

root,ext = os.path.splitext(cif)
out = root + "_simple" + ext

s.as_cif_simple(out=open(out,'w'))
print " >> Wrote file {}".format(out)
