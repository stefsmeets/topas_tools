
import os
import sys

import subprocess as sp


def setpaths_osx():
	pass

def setpaths_linux():
	""" This doesn't work =( """

	BASE = os.path.join(os.path.expanduser("~"), "cctbx", "cctbx_build") # assuming ~/cctbx/cctbx_build
	
	os.environ["LIBTBX_BUILD"] = BASE
	
	for src in ["../cctbx_sources",
				"../cctbx_sources/clipper_adaptbx",
				"../cctbx_sources/docutils",
				"../cctbx_sources/boost_adaptbx",
				"../cctbx_sources/libtbx\pythonpath",
				"lib" ]:
		sys.path.insert(1, os.path.abspath(os.path.join(BASE, src)))
	
	clib1 = os.path.join(BASE, "lib") # ~/cctbx/cctbx_build/lib
	clib2 = "/usr/lib"

	LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
		
	if not LD_LIBRARY_PATH:
		LD_LIBRARY_PATH = clib1 + ":" + clib2
	else:
		LD_LIBRARY_PATH = clib1 + ":" + clib2 + ":" + LD_LIBRARY_PATH

	os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

def setpaths_windows():
	""" Only Windows allows for the environment variables to be set via python """

	BASE="C:\cctbx\cctbx_build"
	
	os.environ["LIBTBX_BUILD"] = BASE
	
	for src in ["..\cctbx_sources",
				"..\cctbx_sources\clipper_adaptbx",
				"..\cctbx_sources\docutils",
				"..\cctbx_sources\\boost_adaptbx",
				"..\cctbx_sources\libtbx\pythonpath",
				"lib" ]:
		sys.path.insert(1, os.path.join(BASE, src))
	
	cbin = os.path.join(BASE, "bin") # C:\cctbx\cctbx_build\bin
	clib = os.path.join(BASE, "lib") # C:\cctbx\cctbx_build\lib
	
	os.environ["PATH"] += ";"+cbin+";"+clib

def cif2patterson():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "cif2patterson.sh")
	
	platform = sys.platform

	if platform == "darwin":
		sp.call(path, *sys.argv[1:])
	elif platform == "win32":
		setpaths_windows()

		import cif2patterson
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "cif2patterson.sh")

		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def cif2topas():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		path = os.path.join(drc, "osx", "cif2topas.sh")
		
		sp.call(path, *sys.argv[1:])
	elif platform == "win32":
		setpaths_windows()

		import cif2topas
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "cif2topas.sh")

		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def expandcell():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		path = os.path.join(drc, "osx", "expandcell.sh")
		
		sp.call(path, *sys.argv[1:])
	elif platform == "win32":
		setpaths_windows()

		import expandcell
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "expandcell.sh")

		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def stripcif():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		path = os.path.join(drc, "osx", "stripcif.sh")
		
		sp.call(path, *sys.argv[1:])
	elif platform == "win32":
		setpaths_windows()

		import stripcif
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "stripcif.sh")

		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def topasdiff():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		path = os.path.join(drc, "osx", "topasdiff.sh")
	
		sp.call(path, *sys.argv[1:])
	elif platform == "win32":
		setpaths_windows()

		import topasdiff
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "topasdiff.sh")

		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"


if __name__ == '__main__':
	print "Running..."
	print
	topasdiff()
	print
	print "Done!"