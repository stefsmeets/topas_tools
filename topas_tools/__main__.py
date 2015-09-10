
import os
import sys

import subprocess as sp

def cif2patterson():
	from IPython.terminal.embed import InteractiveShellEmbed
	InteractiveShellEmbed.confirm_exit = False
	ipshell = InteractiveShellEmbed(banner1='')

	drc = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
	path = os.path.join(drc, "osx", "cif2patterson.sh")
	
	ipshell()	

	if sys.platform == "darwin":
		print path, os.path.abspath(path)
		print os.path.exists(path)
		print
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def cif2topas():
	if sys.platform == "darwin":
		path = "../osx/cif2topas.sh"
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def expandcell():
	if sys.platform == "darwin":
		path = "../osx/expandcell.sh"
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def stripcif():
	if sys.platform == "darwin":
		path = "../osx/stripcif.sh"
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"

def topasdiff():
	if sys.platform == "darwin":
		path = "../osx/topasdiff.sh"
		sp.call(path, *sys.argv[1:])
	else:
		print "Operating system not supported!"


if __name__ == '__main__':
	print "Running..."
	cif2patterson()
	print "Done!"