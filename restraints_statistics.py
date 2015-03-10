#!/usr/bin/env python2.7

import sys
import os
import re

def pbpaste():
	"""Mac only: paste text from clipboard"""
	outf = os.popen('pbpaste', 'r')
	content = outf.read()
	outf.close()
	return content

from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed(banner1='')

text = pbpaste()

oto =  109.5
tot =  145.0
to  =  1.61

oto_vals = []
tot_vals = []
to_vals  = []

n_lines = 0
n_oto = 0
n_tot = 0
n_to  = 0

def equals_about(val, compare_to):
	return compare_to * 0.9 <= val <= compare_to*1.1


for line in re.split('\r|\n',text):
	n_lines += 1

	line = line.strip()

	if line.startswith("'"):
		continue
	if not "Restrain" in line:
		continue

	stuff, restraint, measured, box, weight = line.split(',')

	restraint = float(restraint)
	measured  = float(measured.replace('`',''))

	if equals_about(restraint, oto):
		oto_vals.append(measured)
		n_oto += 1
	elif equals_about(restraint, tot):
		tot_vals.append(measured)
		n_tot += 1
	elif equals_about(restraint, to):
		to_vals.append(measured)
		n_to  += 1
	else:
		print line + ' -- FAIL'

# ipshell();exit()

if not tot_vals: tot_vals = [0]
if not  to_vals:  to_vals = [0]
if not oto_vals: oto_vals = [0]

print "Parsed {} lines".format(n_lines)
print "{} restraints - tot: {}, oto: {}, to: {}".format(n_tot+n_oto+n_to, n_tot, n_oto, n_to)
print ""
print '        {:>10s} {:>10s} {:>10s} {:>10s} '.format('restraint', 'min', 'max', 'avg')
print ' T-O-T  {:10.1f} {:10.3f} {:10.3f} {:10.3f} '.format(tot, min(tot_vals), max(tot_vals), sum(tot_vals)/len(tot_vals))
print ' O-T-O  {:10.1f} {:10.3f} {:10.3f} {:10.3f} '.format(oto, min(oto_vals), max(oto_vals), sum(oto_vals)/len(oto_vals))
print '   T-O  {:10.2f} {:10.3f} {:10.3f} {:10.3f} '.format( to, min( to_vals), max( to_vals), sum( to_vals)/len( to_vals))


