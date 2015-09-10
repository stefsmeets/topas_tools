
import os
import sys

import subprocess as sp

def cif2patterson():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "cif2patterson.sh")
	
	if sys.platform == "darwin":
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def cif2topas():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "cif2topas.sh")
	
	if sys.platform == "darwin":
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def expandcell():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "expandcell.sh")
	
	if sys.platform == "darwin":
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def stripcif():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "stripcif.sh")
	
	if sys.platform == "darwin":
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def topasdiff():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "topasdiff.sh")
	
	if sys.platform == "darwin":
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"


if __name__ == '__main__':
	print "Running..."
	cif2patterson()
	print "Done!"