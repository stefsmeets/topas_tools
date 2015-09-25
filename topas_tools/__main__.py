
import os
import sys

import subprocess as sp


def set_environment_variables_osx():
	"""This is annoying. It is not possible to change the environment variables for the running process, because they are evaluated at runtime.
	Some solutions are listed in http://stackoverflow.com/a/1186194
	From http://stackoverflow.com/questions/1178094/change-current-process-environment
	The best solution seems to be to set the environment variables, and then re-run the program
	The child process will then be able to read them"""

	BASE = os.environ.get('LIBTBX_BUILD', None)
	if not BASE:
		BASE = find_LIBTBX_BUILD()
		os.environ['LIBTBX_BUILD'] = BASE
	if not BASE:
		raise ImportError("Could not locate CCTBX, please ensure that LIBTBX_BUILD environment variable points at cctbx/cctbx_build")

	# cannot use sys.path here, because it is not persistent when calling child process
	PYTHONPATH = os.environ.get("PYTHONPATH", "")
	for src in ["../cctbx_sources",
				"../cctbx_sources/clipper_adaptbx",
				"../cctbx_sources/docutils",
				"../cctbx_sources/boost_adaptbx",
				"../cctbx_sources/libtbx/pythonpath",
				"lib" ]:
		# sys.path.insert(1, os.path.abspath(os.path.join(BASE, src)))
		PYTHONPATH = os.path.abspath(os.path.join(BASE, src)) + ":" + PYTHONPATH
	os.environ["PYTHONPATH"] = PYTHONPATH
	
	if not "DYLD_LIBRARY_PATH" in os.environ:
		os.environ['DYLD_LIBRARY_PATH'] = os.path.join(BASE, "lib") + ":" + os.path.join(BASE, "base", "lib")
	else:
		os.environ['DYLD_LIBRARY_PATH'] += ":" + os.path.join(BASE, "lib") + ":" + os.path.join(BASE, "base", "lib")


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
	# on Windows this seems to work though
	if os.environ.has_key('LIBTBX_BUILD'):
		BASE = os.environ['LIBTBX_BUILD']
	elif os.path.exists("C:\cctbx\cctbx_build"):
		BASE = "C:\cctbx\cctbx_build"
		os.environ["LIBTBX_BUILD"] = BASE
	else:
		raise ImportError("Could not locate CCTBX, please ensure that LIBTBX_BUILD environment variable points at /cctbx/cctbx_build, or CCTBX is installed in C:\cctbx\\")
	
	for src in ["..\cctbx_sources",
				"..\cctbx_sources\clipper_adaptbx",
				"..\cctbx_sources\docutils",
				"..\cctbx_sources\\boost_adaptbx",
				"..\cctbx_sources\libtbx\pythonpath",
				"lib" ]:
		sys.path.insert(1, os.path.join(BASE, src))
		
	cbin = os.path.join(BASE, "bin") # C:\cctbx\cctbx_build\bin
	clib = os.path.join(BASE, "lib") # C:\cctbx\cctbx_build\lib
		
	os.environ["PATH"] += ";" + cbin + ";" + clib

def cif2patterson():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'cif2patterson.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import cif2patterson
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "cif2patterson.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

def cif2topas():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'cif2topas.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import cif2topas
		cif2topas.main()
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "cif2topas.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

def expandcell():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'expandcell.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import expandcell
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "expandcell.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

def stripcif():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'stripcif.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import stripcif
		stripcif.main()
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "stripcif.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

def topasdiff():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'topasdiff.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import topasdiff
		topasdiff.main()
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "topasdiff.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

def make_superflip():
	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	
	platform = sys.platform

	if platform == "darwin":
		set_environment_variables_osx()

		import subprocess as sp
		sp.call([sys.executable, os.path.join(os.path.dirname(__file__), 'make_superflip.py')] + sys.argv[1:]) # call self
	elif platform == "win32":
		setpaths_windows()

		import make_superflip
		make_superflip.main()
	elif platform == "linux2":
		path = os.path.join(drc, "linux", "make_superflip.sh")

		sp.call([path] + sys.argv[1:])
	else:
		print "Operating system not supported!"

if __name__ == '__main__':
	print "Running..."
	print
	topasdiff()
	print
	print "Done!"
