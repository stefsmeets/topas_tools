#!/usr/bin/env python2.7

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import math

import atominfo

from itertools import izip

import json

try:
	from cctbx.array_family import flex

	from cctbx import miller
	from cctbx import xray
	from cctbx import crystal
	
	from cctbx.sgtbx import space_group_type
	from cctbx.miller import index_generator
	from cctbx import uctbx
except ImportError:
	print 'CCTBX not available!!'
	print 'Please install cctbx (http://cctbx.sourceforge.net/)'

print 'numpy version:', np.version.full_version
print 'pandas version:', pd.version.version
if not pd.version.version.startswith('0.13.'):
	raise ImportError, 'Please use Pandas version 0.13.x'

__version__ = '05-06-2014'

# wider dataframes:
pd.set_option('display.line_width',160)

# Not used yet, use in combination with functools.partials to set partial functions
# this is useful to save time writing the names everytime
default_columns = {
	'single' 		: 'single',
	'powder' 		: 'powder',
	'overlap' 		: 'overlap',
	'fwhm' 			: 'fwhm',
	'dspacings' 	: 'd',
	'mult' 			: 'm',
	'twotheta' 		: '2th',
	'repartitioned' : 'repart'
}


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

# laue groups
crystal_system = {
	'triclinic':'-1',
	'monoclinic':'2/m',
	'orthorhombic':'mmm',
	'tetragonal':('4/m','4/mmm'),
	'hexagonal':('6/m','6/mmm'),
	'trigonal':('-3','-3m'),
	'cubic':('m3','m3m')
}


laue_symmetry = {
	'triclinic' : lambda h,k,l: h>=0,
	'monoclinic' : lambda h,k,l: h>=0 and k>=0,
	'monoclinicb' : lambda h,k,l: h>=0 and k>=0,
	'monoclinicc' : lambda h,k,l: h>=0 and l>=0,
	'orthorhombic' : lambda h,k,l: h>=0 and k>=0 and l>=0,
	'tetragonal' : lambda h,k,l: h>=0 and k<=l>=0,
	'hexagonal' : lambda h,k,l: h>=0 and l>=0 and h>=-1*k,
	'rhombohedral1' : lambda h,k,l: h>=0 and k>=0,
	'rhombohedral2' : lambda h,k,l: h>=k>=l>=0 and h>=-1*k,
	'cubic' : lambda h,k,l: h>= 0 and h>=k>=l>=0 and h>=-1*k
}


merge_laue_symmetry = {
	'triclinic' : lambda (h,k,l): (abs(h),k,l),
	'monoclinic' : lambda (h,k,l): (abs(h),abs(k),l),
	'monoclinicb' : lambda (h,k,l): (abs(h),abs(k),l),
	'monoclinicc' : lambda (h,k,l): (abs(h), l,abs(l)),
	'orthorhombic' : lambda (h,k,l): (abs(h),abs(k),abs(l))
	#'tetragonal' : lambda h,k,l: raise ValueError,
	#'hexagonal' : lambda h,k,l: raise ValueError,
	#'rhombohedral1' : lambda h,k,l: raise ValueError,
	#'rhombohedral2' : lambda h,k,l: raise ValueError,
	#'cubic' : lambda h,k,l: raise ValueError
}

""" Found on the internet, is this useful to add to merge_laue_symmetry???

point_groups

Pointgroup Laue group        Limits

 3 pg1     1bar       hkl:l>=0  hk0:h>=0  0k0:k>=0   1,2
   pg1bar
 4 pg2 (b) 2/m        hkl:k>=0, l>=0  hk0:h>=0       3/b,4/b....
   pgm pg2/m
 5 pg2 (c) 2/m        hkl:k>=0, l>=0  h0l:h>=0       1003,1004
 6 pg222   mmm        hkl:h>=0, k>=0, l>=0            16 ...
   pgmm2 pgmmm 
 7 pg4     4/m        hkl:h>=0, l>=0 with k>=0 if  h=0  and
   pg4bar pg4/m                            k>0 if h>0
 8 pg422   4/mmm       hkl:h>=0, k>=0, l>=0            89..
   pg4mm pg4bar2m pg4barm2 pg4/mmm
 9 pg3     3bar      hkl:h>=0, k>0  00l:l>0         143..
   pg3bar
10 pg312   3/m        hkl:h>=0, k>=0 with k<=h for all l.
   pg32 pg3m pg3m1 pg3barm1 if k = 0  l>=0
         Space group numbers :   149-151-153 157 159 162 163
11 pg321   3bar1m     hkl:h>=0, k>=0 with k<=h for all l.
   pg31m pg3bar1m      if h = k  l>=0
         Space group numbers :   150-152-154
12 pg6     6/m        hkl:h>=0, k>=0, l>=0 with k>=0 if  h=0
   pg6bar  6/m        and k> 0 if h>0
13 pg622   6/mmm       hkl:h>=0, k>=0, l>=0 with h>=k 177..
   pg6mm pg6barm2 pg6bar2m  pg 6/mmm
14 pg23    m3         hkl:h>=0, k>=0, l>=0 with l>=h,  k>=h
   pgm3bar 
15 pg432   m3m        hkl:h>=0, k>=0, l>=0  with  k>=l
   pg4bar3m pgm3barm
"""


try:
	from IPython.terminal.embed import InteractiveShellEmbed
	InteractiveShellEmbed.confirm_exit = False
	ipshell = InteractiveShellEmbed(banner1='')
except ImportError:
	ipshell = lambda x:x

def pprint(d):
	"""pretty printing using the json module"""
	print json.dumps(d,sort_keys=True,indent=4)

def pairwise(iterable):
	"s -> (s0,s1), (s1,s2), (s2, s3), ..."
	for i in range(1,len(iterable)):
		yield (iterable[i-1],iterable[i])

def make_special_position_settings(cell,spgr,min_dist_sym_equiv=0.5):
	"""Takes cell and space group, returns cctbx structure
	input:	cell: (a b c alpha beta gamma) as a tuple
			spgr: 'space_group' like a string
			min_dist_sym_equiv: float
	output: <cctbx.crystal.special_position_settings object>
				contains cell, space group and special positions
	"""
	special_position_settings = crystal.special_position_settings(
		crystal_symmetry = crystal.symmetry(
			unit_cell = cell,
			space_group_symbol = spgr),
		min_distance_sym_equiv = min_dist_sym_equiv)
	return special_position_settings

def make_symmetry(cell,spgr):
	"""takes cell parameters (a,b,c,A,B,C) and spacegroup (str, eg. 'cmcm'), returns cctbx
	crystal_symmetry class required for merging of reflections"""
	if not cell:
		cell = raw_input("Please specify a cell:\n >> ")
		cell = map(float,cell.split())
	if not spgr:
		spgr = raw_input("Please specify space group:\n >> ")

	crystal_symmetry = crystal.symmetry(
		unit_cell = cell,
		space_group_symbol = spgr)
	return crystal_symmetry

def generate_indices(cell,spgr,dmin=1.0,ret='index'):
	"""http://cci.lbl.gov/cctbx_sources/cctbx/miller/index_generator.h"""

	dmin = dmin-0.0000000000001 # because resolution_limit is < and not <=

	anomalous_flag = False
	symm = make_symmetry(cell,spgr)

	unit_cell = uctbx.unit_cell(cell)
	sg_type = space_group_type(spgr)
	mig = index_generator(unit_cell, sg_type, anomalous_flag=anomalous_flag, resolution_limit=dmin) # input hkl or resolution(d_min)
	indices = mig.to_array()
	if ret == 'index': # suitable for df index
		return indices
	else:
		return miller.array(miller.set(crystal_symmetry=symm,indices=indices,anomalous_flag=anomalous_flag))


def merge_sym_equiv(m,output='ma',algorithm=None, verbose=True):
	"""takes miller.array, returns hkl dictionary
	http://cci.lbl.gov/cctbx_sources/cctbx/miller/equivalent_reflection_merging.tex
	
	testing merging algorithms on different data sets:
	works = tested to give expected results
	fails = tested not to give expected results

	                   gaussian        shelx
	unique             works           works
	unique+sigmas
	not unique                         works
	not unique+sigmas

	With no sigmas, shelx takes average of present refs in each group
	"""
	n_unique = len(m.unique_under_symmetry().data())
	n_refs   = len(m.data())

	# The number of merged reflections may be more than expected for powder data
	# this is because sometimes multiple indexes are read into the dataframe for some data sets
	# merging should then complete with R = 0, which is ok for powders

	if not algorithm:
		if n_refs == n_unique: # indicates powder data set? => see also note above
			algorithm = 'shelx' # works with unique data, but gives strange results for single crystal data
		else:
			algorithm = 'shelx' # gaussian returns 0 for unique data set, such as powders

	#print "\ntotal/unique = {}/{} -> merging algorithm == {}".format(n_refs,n_unique,algorithm)
	#print "Using gaussian NEEDS accurate sigmas, shelx doesn't."

	merging = m.merge_equivalents(algorithm=algorithm)

	if verbose:
		print
		merging.show_summary()
		print 

	m_out = merging.array()
	print '%s reflections merged/averaged to %s' % (m.size(),m_out.size())
	if output == 'dict':
		return miller_array_to_dict(m_out)
	elif output == 'ma':
		return m_out

def merge(df,col,lauegr):
	"""Merge equivalents using pandas.groupby functionality

	R-linear = sum(abs(data - mean(data))) / sum(abs(data))
	R-square = sum((data - mean(data))**2) / sum(data**2)

	Returns pd.series"""
	key = merge_laue_symmetry[lauegr]
	s = df[col]
	s = s[s.notnull()]
	grouped = s.groupby(key)
	merged = grouped.mean()

	## WRONG	
	# sum_num = 0
	# sum_den = 0
	# sum_num_sq = 0
	# sum_den_sq = 0
	# for j,group in grouped:
	# 	if len(group) == 1:
	# 		continue
	# 	avg = group.mean()
	# 	# sum_den += avg
	# 	sum_num += (group - avg).abs().sum()
	# 	sum_den += (group).sum()

	# 	# sum_den_sq += avg**2
	# 	difsq = (group - avg)**2.sum()
	# 	sum_num_sq += difsq
	# 	sq = group**2.sum()
	# 	sum_den_sq += sq
	# print sum_num / sum_den
	# print sum_num_sq / sum_den_sq

	print '%s: %d reflections merged/averaged to %d using %s symmetry' % (col,len(s),len(merged),lauegr)
	return merged





def remove_sysabs(m, verbose=True):
	"""Returns new miller.array with systematic absences removed"""
	sysabs = m.select_sys_absent().sort('data')

	if sysabs.size() > 0 and verbose:
		print '\nTop 10 systematic absences removed (total={}):'.format(sysabs.size())
		
		if sysabs.sigmas() == None:
			for ((h,k,l),sf) in sysabs[0:10]:
				print '{:4}{:4}{:4} {:8.2f}'.format(h,k,l,sf)
		else:
			for ((h,k,l),sf,sig) in sysabs[0:10]:
				print '{:4}{:4}{:4} {:8.2f} {:8.2f}'.format(h,k,l,sf,sig)

		print "Compared to largest 3 reflections:"
		if m.sigmas() == None:
			for ((h,k,l),sf) in m.sort('data')[0:3]:
				print '{:4}{:4}{:4} {:8.2f}'.format(h,k,l,sf)
		else:
			for ((h,k,l),sf,sig) in m.sort('data')[0:3]:
				print '{:4}{:4}{:4} {:8.2f} {:8.2f}'.format(h,k,l,sf,sig)
			
		return m.remove_systematic_absences()
	elif sysabs.size() > 0:
		print "{} systematic absences removed".format(sysabs.size())
		return m.remove_systematic_absences()
	else:
		return m






def m2df(m,data='data',sigmas='sigmas'):
	"""Takes a miller.array object and returns the
	m: miller.array
	data and sigmas are the names for the columns in the resulting dataframe
	if no data/sigmas are present in the miller array, these are ignored.
	"""
	df = pd.DataFrame(index=m.indices())
	if m.data():
		df[data] = m.data()
	if m.sigmas():
		df[sigmas] = m.sigmas()
	return df

def df2m(df,cell,spgr,data=None,sigmas=None):
	"""Constructs a miller.array from the columns specified by data/sigmas in the dataframe,
	if both are None, returns just the indices.
	needs cell and spgr to generate a symmetry object."""
	anomalous_flag = False

	if isinstance(df,pd.DataFrame):
		try:
			sel = df[data].notnull() # select notnull data items for index
		except ValueError:
			index = df.index
		else:
			index = df.index[sel]
	else:
		index = df

	indices = flex.miller_index(index)

	if data:
		data = flex.double(df[data][sel])
	if sigmas:
		sigmas = flex.double(df[sigmas][sel])

	symm = make_symmetry(cell,spgr)
	ms = miller.set(crystal_symmetry=symm,indices=indices,anomalous_flag=anomalous_flag)
	return miller.array(ms,data=data,sigmas=sigmas)

def reduce_all(df,cell,spgr,dmin=None,reindex=True, verbose=True):
	"""Should be run after files2df. Takes care of some things that have to be done anyway.
	Once data has been loaded, this function reduces and merges all the data to a single 
	unique set, adds dspacings and multiplicities and orders columns and sorts by the dspacing.

	dmin: 
	can be the name of a column and dmin is taken from that
	can be float
	can be None and the dmin is determined automatically asl the largest dmin of all columns read

	reindex: bool
	will reindex using all indices generated up to dmin

	Returns a dataframe object"""

	d = pd.Series(calc_dspacing(df,cell,inplace=False), index=df.index)
	dmins = [d[df[col].notnull()].min() for col in df if col not in ('m','d')]
	
	## little table with dmins
	cols  = [col for col in df if col not in ('m','d')]
	ln = max([len(col) for col in cols])
	print '\n{:>{}} {:>6s}'.format('',ln,'dmin')
	for col,dval in zip(cols,dmins):
		print '{:>{}} {:6.3f}'.format(col,ln,dval)

	if not dmin:
		# find the largest dmin for all data sets
		dmin = max(dmins)
	elif isinstance(dmin,str):
		sel = df[dmin].notnull()
		dmin = min(d[sel])
	else:
		dmin = float(dmin)

	order = ['m','d']
	order = order + [col for col in df if col not in order]

	dfm = pd.DataFrame()

	for col in df:
		if col in ('m','d'):
			continue
		print '\n - Merging {}: '.format(col)
		m = df2m(df,cell=cell,spgr=spgr,data=col)
		m = remove_sysabs(m, verbose=verbose)
		m = merge_sym_equiv(m, verbose=verbose)
		dfm = dfm.combine_first(m2df(m,data=col))

	if reindex:
		index = generate_indices(cell,spgr,dmin)
		index = pd.Index(index)
		dfm = dfm.reindex(index=index)
		dfm = dfm.reindex(columns=order)

	f_calc_multiplicities(dfm,cell,spgr)
	calc_dspacing(dfm,cell)

	print "\nReduced/merged data to dmin = {}, {} refs".format(dmin,len(dfm))

	return dfm.sort_index(by='d',ascending=False)

def f_calc_multiplicities(df,cell,spgr):
	"""Small function to calculate multiplicities for given dataframe"""
	m = df2m(df,cell,spgr)
	df['m'] = m.multiplicities().data()


def calc_fwhm_caglioti(th2,U=0.0,V=0.0,W=0.05,th2_in_degrees=True):
	"""Return FWHM as function of U,V,w
	-- Cagliotti formula
		H^2 = U tan(th)^2 + V tan(th) + W

	For a Gaussian peak breadth H^2 == FHM^2

	"""

	if th2_in_degrees:
		th = np.radians(th2/2)
	else:
		th = th2/2

	H2 = U*np.tan(th)**2 + V*tan(th) + W
	fwhm = H2**0.5
	return fwhm

def calc_fwhm_topas(th2,ha=0.05,hb=0.0,hc=0.0,th2_in_degrees=True):
	"""Return FWHM as function of ha,hb and hc, a modification of the caglioti formula as used in TOPAS
	-- Modified Cagliotti formula (very similar shape)
		fwhm = ha + hb tan(th) + hc / cos(th) 
	"""

	if th2_in_degrees:
		th = np.radians(th2/2)
	else:
		th = th2/2

	fwhm = ha + hb*np.tan(th) + hc/np.cos(th)
	return fwhm

def write_box_file(df,of,out=None,fwhm='fwhm',key='powder',th2='2th',overlap='overlap',kind='mf2'):
	scale = 10
	offset = 50

	if isinstance(out,str):
		out = open(out,'w')

	for j in range(max(df[overlap] + 1)):
		ind = df[overlap] == j
		group = df[ind]

		xmin = group[th2][0]  - 0.5*of*group[fwhm][0]
		xmax = group[th2][-1] + 0.5*of*group[fwhm][-1]

		if kind == 'mf2':
			ymax = sum(group['m']*group[key]**2)*scale+offset
		else:
			raise NotImplementedError

		print >> out, "{} {} {}".format(xmin,xmax,ymax)



def print_template(name='input.py'):
	dirname, filename = os.path.split(os.path.realpath(__file__))
	print "Path to script:", dirname
	print "Name of script:", filename
	print "input template:", name

	root,ext = os.path.splitext(filename)

	out = open(name,'w')

	print >> out, "#!/usr/bin/env python2.7"
	print >> out
	print >> out, "# Template generated using {} version {}".format(__file__,__version__)
	print >> out
	print >> out, "import sys"
	print >> out, "sys.path.append('{}')".format(dirname)
	print >> out, "from {} import *".format(root)
	print >> out
	print >> out, "from IPython.terminal.embed import InteractiveShellEmbed"
	print >> out, "ipshell = InteractiveShellEmbed(banner1='')"
	print >> out
	print >> out, """cell = ''"""
	print >> out, """a,b,c,al,be,ga = cell = map(float,cell.split())
xwavelength = 1.000
composition ='Si100 O200'
spgr = 'Cmcm'

files = \"""
ZSM5b_fpm.fou  xf skip
#calcinated_cr02.hkl ef2 skip skip
no_calcinated_cr02.hkl ef2 skip skip
\""" 

df = files2df(files)
df = reduce_all(df,cell,spgr)

df['fwhm'] = 0.0300

df['ef'] = df['ef2']**0.5
del df['ef2']

completeness(df,target='ef')

of = 0.3

df['2th']           = calc_2th_from_d(df['d'],xwavelength)
df['overlap']       = assign_overlap_groups(df,overlap_factor=of)

rep = Repartition(df, single='ef', powder='xf', target='rep'        , overlap='overlap'       , kind="mf2", run=False)
# rep.classify()
rep.run()

pre  = open('zsm5_template.txt','r').readlines()
post = '\\nEnd'
cols = ('rep'             , '*', 'fwhm' )
write_hkl(df,out='focus_no_calc.inp',cols=cols,pre=pre,post=post)


"""

	print >> out, '\n\n\n\n\n\n\n\n\n'
	print >> out, "ipshell()"

def comp2dict(composition):
	"""Takes composition: Si20 O10, returns dict of atoms {'Si':20,'O':10}"""
	import re
	pat = re.compile('([A-z]+|[0-9]+)')
	m = re.findall(pat,composition)
	return dict(zip(m[::2],map(int,m[1::2])))

def whatis(f=None):
	"""Print all function names for current namespace"""
	ftype = type(whatis)
	if f:
		print f.__doc__
	else:
		flist = [globals()[f].func_name for f in globals() if isinstance(globals()[f],ftype)]
		for fname in sorted(flist):
			print fname

def linear_fit(x,a,b):
	return a + b*x

def wilson_plot(df,target='redf',kind='fobs',atoms=None,table='xray',interval=0.050,plot=True):
	"""Performs a wilson plot to approximate scale and biso values. Returns scale (k) ,biso"""
	from scipy.optimize import curve_fit
	
	sel = df[target].notnull()

	df['sitl'] = calc_sitl(df)

	mapper = df['sitl'].map(lambda x: x//interval)
	grouped = df[sel].groupby(mapper)

	rhs = grouped['sitl'].median()

	f0 = np.zeros_like(rhs)
	for atom,number in atoms.items():
		f0 += number * f_calc_scattering_factor(atom,rhs,table=table)**2

	rhs = rhs**2

	if kind == 'fobs':
		lhs = grouped[target].mean() / f0
		lhs = np.log(lhs)
	elif kind == 'iobs':
		lhs = (grouped[target]**2).mean() / f0
		lhs = np.log(lhs)
	else:
		raise KeyError, "Unknown kind {}, should be fobs / iobs".format(kind)

	guess = (lhs.iloc[0], (lhs.iloc[-1] - lhs.iloc[0]) / (rhs.iloc[-1] - rhs.iloc[1]) )
	# print guess

	popt,pcov = curve_fit(linear_fit,rhs,lhs, p0=guess)

	a,b = popt

	k = np.exp(a/2)
	biso = -b/2

	print
	print " >> WILSON PLOT (Dunitz 1979, p154)"
	print
	print 'linear fit: y = {:.6f} + {:.6f} * x'.format(a,b)
	print
	print 'Scale -> k * F_obs = F_calc'
	print
	print 'k     = {:.3f}   k^2   = {:.3f}'.format(k,k**2)
	print 'Biso  = {:.3f}'.format(biso)

	if plot:
		print
		yval = linear_fit(rhs,a,b)
		plt.plot(rhs,lhs,'r+',ms=10,label='wilson plot')
		plt.plot(rhs,yval,label='linear fit')
		# plt.legend()
		plt.show()
	else:
		print '(plot surpressed)'

	return (k,biso)

def read_file(path):
	"""opens file, returns file object for reading"""
	try:
		f = open(path,'r')
	except IOError:
		print 'cannot open', path
		exit()
	return f



def files2df(files,df=None,shelx=False,savenpy=False, verbose=True):
	"""Reads a multiline string or list of strings and parses the files and labels.
	Returns a pandas dataframe or updates an existing one.

	Passing a dataframe object to df will update the dataframe inplace with new values. 
	df's values are prioritized.

	filenames starting with $ are read as shelx file.
	column labels or lines starting with # are skipped 

	savenpy: save data as binary numpy format for faster loading on next use."""

	## for merging 2 data sets
	# combiner = lambda x, y: np.where(pd.notnull(x) & pd.notnull(y), (x+y)/2, np.where(pd.isnull(x),y,x))

	if isinstance(files,str):
		files = [f for f in files.splitlines() if not f.startswith("#")]

	df_out = pd.DataFrame()

	for item in files:
		if isinstance(item,str):
			item = item.split()
		if not item:
			continue
		

		fname,labels = item[0],item[1:]
		
		if fname.startswith('$'):
			fname = fname.replace('$','')
			data = load_hkl(fname,labels,shelx=True, savenpy=savenpy, verbose=verbose)
		else:
			data = load_hkl(fname,labels,shelx=shelx,savenpy=savenpy, verbose=verbose)

		if any(key in df_out for key in data):
			raise KeyError, "Duplicate column name detected for {}".format(fname)

			## don't use, doesn't merge correctly with more than 2 columns
			# df_out = df_out.combine(load_hkl(fname,labels,shelx=shelx),combiner)

		df_out = df_out.combine_first(data)

	if df is None:
		return df_out
	else:
		return df.combine_first(df_out)



def load_hkl(fin,labels=None,shelx=False,savenpy=False, verbose=True):
	"""Read a file with filename 'fin', labels describe the data in the columns.
	The h,k,l columns are labeled by default and expected to be the first 3 columns.
	Returns a dataframe with hkl values as the indices

	e.g. load_hkl('red.hkl',labels=('F','sigmas')

	All columns should be labeled and columns labeled: 
		None,'None','none' or 'skip' will be ignored

	It is recommended to label all expected columns, so the algorithm will return an
	error rather than read the next column when columns are not delimited

	savenpy: save data as binary numpy format for faster loading on next use."""


	if shelx == True and labels == None:
		labels = ('F2','sigma')
	elif labels == None:
		raise TypeError, 'load_hkl() did not get a value for labels.'

	if isinstance(fin,file):
		fname = fin.name
	else:
		fname = fin 

	labels = tuple(labels)
	skipcols = (None,'None','none','skip') + tuple(item for item in labels if item.startswith('#'))
	usecols = [0,1,2] + [3+i for i,label in enumerate(labels) if label not in skipcols]

	root,ext = os.path.splitext(fname)
	changed = False

	try:
		inp = np.load(root+'.npy')
		assert len(inp.T) == len(usecols), 'npy data did not match to expected columns'
	except (IOError, AssertionError):
		changed = True
		if shelx == False:
			try:
				inp = np.loadtxt(fin,usecols=usecols) # if this fails, try shelx file
			except ValueError:
				inp = np.genfromtxt(fin,usecols=usecols,delimiter=[4,4,4,8,8,4])
		else:
			inp = np.genfromtxt(fin,usecols=usecols,delimiter=[4,4,4,8,8,4])
	else:
		ext = '.npy'
		fname = root+'.npy'

	if savenpy:
		if ext != '.npy' or changed:
			print 'Writing data as npy format to {}'.format(fname)
			np.save(root,inp)

	if verbose:
		print ''
		print 'Loading data: {}'.format(fname)
		print '     usecols: {}'.format(usecols)
		print '      labels: {}'.format(' '.join(('h','k','l')+labels))
		print '       shape: {}'.format(inp.shape)
	else:
		print 'Loading data: {} => ({:5d}, {:2d})'.format(fname, inp.shape[0], inp.shape[1]) 

	h = map(int,inp[:,0])
	k = map(int,inp[:,1])
	l = map(int,inp[:,2])

	index = izip(h,k,l)

	labels = (label for label in labels if label not in skipcols)

	d = dict(izip(labels,inp[:,3:].T))

	df = pd.DataFrame(d,index=index)

	if not df.index.is_unique:
		print "\n** Warning: Duplicate indices detected in {} **\n".format(fname)

		## useful, but very slow for large data sets!
		# index = list(df.index)
		# duplicates = set([x for x in index if index.count(x) > 1])
		# for x in duplicates: print x

	return df


def add_indices(df):
	h,k,l = zip(*df.index)
	df['h'] = h
	df['k'] = k
	df['l'] = l


def scale(df,key1,key2,inplace=False):
	"""scales the values of 'key1' to be similar to 'key2'"""
	try:
		scale = df[key2].sum()/df[key1].sum()
	except ZeroDivisionError:
		print "Cannot calculate scale factor, denominator == 0"
		scale = 1.0		
	print "\nScaling: {} values multiplied by {}\n".format(key1,scale)
	if inplace:
		df[key1] = df[key1]*scale
	else:
		return df[key1]*scale



















def assign_overlap_groups(df,overlap_factor=0.3,key_fwhm='fwhm',key_2th='2th',splitlist=None):
	"""Assigns overlap group to groups that are found to be overlapping using formula:
		2*th2_2 - 2*th2_1 < (of/2) * (fwhm_1 + fwhm_2)

	df: dataframe object
	key_fwhm: which column to read fwhm values from
	key_2th: which column to read 2th values from

	splitlist: Optional list of 2th values where splits will be applied, regardless of overlap factor"""

	if not key_fwhm in df:
		raise IndexError, 'Missing FWHM values!!'
	
	if not key_2th in df:
		raise IndexError, 'Missing 2theta values!!' # this could be solved by taking wavelength and calculating it on the fly

	if not splitlist:
		splitlist = []

	splitlist.append(max(df[key_2th])+1)
	splitlist  = iter(splitlist)
	next_split = splitlist.next()

	def is_in_same_group(th2_1,th2_2,fwhm_1,fwhm_2,of):
		"""Returns True/False depending on whether peaks are overlapping"""
		return th2_2 - th2_1 < (of/2) * (fwhm_1 + fwhm_2)

	fwhm = df[key_fwhm]
	th2 = df[key_2th]

	#print '\n** ASSIGN OVERLAP GROUPS **\n'
	overlapped = 0
	nsplits = 0
	new = False

	group = 0
	group_list = [group]

	fwhm_1 = fwhm[0]
	th2_1 = th2[0]

	for i in xrange(1,len(df)):

		fwhm_2 = fwhm[i]
		th2_2 = th2[i]

		if th2_2 > next_split:
			while th2_2 > next_split: # if there are multiple splits between peaks
				next_split = splitlist.next()
			group += 1
			nsplits += 1
			group_list.append(group)
			new = True
		elif is_in_same_group(th2_1,th2_2,fwhm_1,fwhm_2,overlap_factor):
			if new:
				overlapped += 2
			else:
				overlapped += 1
			group_list.append(group)
			new = False
		else:
			group += 1
			group_list.append(group)
			new = True

		fwhm_1 = fwhm_2
		th2_1 = th2_2

	print '\n{:.2f}%% overlapped reflections with overlap_factor={}!!'.format((overlapped*100.0)/len(group_list),overlap_factor)
	print '{} overlap groups, ~{:.2f} refs/group'.format(max(group_list), float(len(group_list))/max(group_list))
	print 'Applied splits: ', nsplits

	return group_list


def set_custom_overlap_groups(df,overlap='overlap',rep_type='case',target='ov_new',single='single'):
	"""Function to set overlap groups manually. The idea is that 'good' repartitioned reflections should have
	overlap group 0, so they are not repartitioned in superflip. Poorly repartitioned groups, i.e., those with
	several reflections missing in the single case, could be repartitioned based on histogram matching to
	improve the structure solution procedure."""

	print max(df[overlap] + 1)

	all_cases = ('single','zero','normal','case0','case1','case2','case3','case4','case5','rest')

	good = ('single','normal','zero',)
	intermediate = ('case1','case2','case3','case4','case5')
	bad = tuple(case for case in all_cases if case not in  intermediate+good)

	print good
	print intermediate
	print bad

	ngood  = 0
	ngood_groups = 0
	ninter = 0
	ninter_groups = 0
	nbad   = 0
	nbad_groups   = 0

	if not target in df:
		df[target] = df[overlap]
	
	for j, group in df.groupby(overlap):
		case = group[rep_type][0]
		assert all(group[rep_type] == case)

		if case in good:
			index = group.index
			df[target][index] = 0

			ngood += len(group)
			ngood_groups += 1

		elif case in intermediate:
			if group[single].isnull().sum() == 1:
				index = group.index
				df[target][index] = 0

				ngood += len(group)
				ngood_groups += 1
			else:
				set_to_zero = group[single].isnull().astype(int) * j

				index = group.index
				df[target][index] = set_to_zero
				
				ninter += len(group)
				ninter_groups += 1

		else:
			nbad += len(group)
			nbad_groups += 1


	print 'ngood  = {} / {}'.format(ngood, ngood_groups)
	print 'ninter = {} / {}'.format(ninter, ninter_groups)
	print 'nbad   = {} / {}'.format(nbad, nbad_groups)


def overlap_groups_sf_style(df,overlap='overlap',inplace=False):
	"""Modifies formatting of overlap groups to mimic overlap groups in superflip
	0  : not overlapped
	1+ : overlapped"""

	if inplace:
		ov = df[overlap]
	else:
		ov = df[overlap].copy()

	def gencounter(n=1):
		while True:
			n += 1
			yield n
	
	counter = gencounter()

	for j, group in df.groupby(overlap):
		index = group.index
		
		if len(group) == 1:
			k = 0
		elif len(group) > 0:
			k = counter.next()
		else:
			raise IndexError, 'Empty group {}'.format(j)

		ov[index] = k

	print 'Overlap = {:.2f}%'.format(100*(ov > 0).sum() / float(len(ov)))

	if not inplace:
		return ov
	else:
		raise NotImplementedError, 'Set inplace=False to fix'




def digitize(df,single='red_f',powder='xrd_f',nbins=10):
	"""Digitizes/bins reflections for scaling purposes"""
	df_bins = pd.DataFrame(index=df.index)
	
	ser_single = df[single]
	ser_powder = df[powder]
	sel = ser_single.notnull()
	ser_single = ser_single[sel]
	
	min_single = ser_single.min()
	max_single = ser_single.max()
	min_powder = ser_powder.min()
	max_powder = ser_powder.max()
	
	bin_single = np.linspace(min_single, max_single, nbins)
	bin_powder = np.linspace(min_powder, max_powder, nbins)
	
	digi_single = pd.Series(data=np.digitize(ser_single,bin_single),index=ser_single.index)
	digi_powder = pd.Series(data=np.digitize(ser_powder,bin_powder),index=ser_powder.index)
	
	df_bins['bin'+powder] = digi_powder
	df_bins['bin'+single] = digi_single

	return df_bins, bin_single, bin_powder


class Repartition(object):
	"""Repartitions the data given in 'powder' using the data from 'single'. Uses the column 'overlap'
	for the overlapping reflections. Kind can be f, f2 or mf2, the latter was found to give the best results
	repartitioned intensities are added to column 'target' in the given dataframe.

	>> Breaks with pandas 10.1 <<

	single: single crystal like reflections from RED, ADT, etc.
	powder: overlapped powder diffraction data
	target: target column for repartitioned data
	overlap: column with the overlap groups
	kind: repartition based on 'f', 'f2', or 'mf2'
	program: "superflip" or "focus" => use different approach for treating groups (self.func_dict)

	run
		True/'run': run repartitioning
		False: setup the class, but prevent repartitioning, which can later be done by calling .run()
		'classify': Classify groups only
	"""
	def __init__(self, df, *args, **kwargs):
		super(Repartition, self).__init__()

		self.has_run = False


		self.dmin = kwargs.get('dmin',None)

		self.df = df

		self.kind        = kwargs.get('kind','mf2')
		self.single      = kwargs.get('single','single')
		self.powder      = kwargs.get('powder','powder')
		self.target      = kwargs.get('target','repart')
		self.overlap     = kwargs.get('overlap','overlap')

		program = kwargs.get("program","focus")
		run     = kwargs.get('run',True)

		self.treat_missing = "break"
		# break: raise ValueError
		# zero:  set missing to 0
		# estimate: estimate from electron data if available, else set to 0
		
		self.str_single  = 'single'
		self.str_zero    = 'zero'
		self.str_normal  = 'normal'
		self.str_case0   = 'case0'
		self.str_case1   = 'case1'
		self.str_case2   = 'case2'
		self.str_case3   = 'case3'
		self.str_case4   = 'case4'
		self.str_case5   = 'case5'
		self.str_rest    = 'rest'
		self.str_missing = 'missing'

		if program == "superflip":
			self.func_dict = {
 			'case0':   'repartition_copy',
 			'case1':   'repartition_with_nan_case_1',
 			'case2':   'repartition_with_nan_case_2',
 			'case3':   'repartition_copy',
 			'case4':   'repartition_copy',
 			'case5':   'repartition_with_nan',
 			'normal':  'repartition_normal',
 			'rest':    'repartition_copy',
 			'single':  'repartition_single',
 			'zero':    'repartition_empty',
 			'missing': 'handle_missing'}
		else:
			self.func_dict = {
			'single' : 'repartition_single',
			'zero'   : 'repartition_empty',
			'normal' : 'repartition_normal',
			'case0'  : 'repartition_copy',
			'case1'  : 'repartition_with_nan_case_1',
			'case2'  : 'repartition_with_nan_case_2',
			'case3'  : 'repartition_with_nan_case_3',
			'case4'  : 'repartition_with_nan_case_3',
			'case5'  : 'repartition_with_nan',
			'rest'   : 'repartition_copy',
 			'missing': 'handle_missing'}


		self.n_rep_missing = 0
		self.n_rep_missing_groups = 0
		self.n_rep_single = 0
		self.n_rep_single_groups = 0
		self.n_rep_zero = 0
		self.n_rep_zero_groups = 0
		self.n_rep_normal = 0
		self.n_rep_normal_groups = 0
		self.n_rep_case0 = 0
		self.n_rep_case0_groups = 0
		self.n_rep_case1 = 0
		self.n_rep_case1_groups = 0
		self.n_rep_case2 = 0
		self.n_rep_case2_groups = 0
		self.n_rep_case3 = 0
		self.n_rep_case3_groups = 0
		self.n_rep_case4 = 0
		self.n_rep_case4_groups = 0
		self.n_rep_case5 = 0
		self.n_rep_case5_groups = 0
		self.n_rep_rest = 0
		self.n_rep_rest_groups = 0

		self.setup_bins()

		if run == 'run' or run == True:
			self.run()
		elif run == 'classify':
			self.classify()
		else:
			pass

	def estimate_missing(self,group):
		"""Takes a NaN values in a group and estimates the reflection intensity based on the rank
		of the 'single' reflection. If both are NaN, xrd reflection is set to 0."""
		new_refs = group[self._powder].copy()
		kind = self.kind

		for index, row in group.iterrows():
			if pd.isnull(row[self._powder]): 
				rank = int(self.df_bins.ix[[index]]['bin'+self.single])
				val  = self.bin_powder[rank]
				if kind == 'f':
					val = val
				if kind == 'f2':
					val = val**2
				elif kind == 'mf2':
					m = self.df.ix[[index]]['m'][0] 
					val = m*val**2

				new_refs.ix[[index]] = val

		return new_refs

	def handle_missing(self,group):
		"""Handle missing reflections"""
		if self.treat_missing == 'break':
			raise ValueError, 'Unexpected NaN in group {} for {}'.format(group[self.overlap][0],self._powder)
		elif self.treat_missing == 'zero':
			group[self._powder] =  group[self._powder].fillna(0)
		elif self.treat_missing == 'estimate':
			group[self._powder] =  self.estimate_missing(group)
		else:
			raise ValueError, 'Unexpected argument {} for self.treat_missing'.format(self.treat_missing)

		j = group[self.overlap][0]

		case = self.get_case(group)
		func_name = self.func_dict[case]
		func = 	getattr(self,func_name)
		print "Overlap group {:>3d} restored missing as '{}' -> repartitioned as {}".format(j,self.treat_missing,case)
		return func(group)

	def repartition_single(self,group):
		"""Reflection was uniquely determined -> copy amplitude directly"""
		return group[self._powder]

	def repartition_empty(self,group):
		"""Reflection(s) can be set to zero"""
		new_refs = pd.Series(index=group.index,data=[0.0] * len(group)) 
		return new_refs

	def repartition_copy(self,group):
		"""General function for copying reflection"""
		return group[self._powder]

	def repartition_normal(self,group):
		"""All reflections in powder and single are available (no NaN),
		standard re-partitioning can be performed"""

		sum_powder = group[self._powder].sum()
		sum_single = group[self._single].sum()

		return [(group[self._single][i] * sum_powder/sum_single) for i in xrange(len(group))]

	def repartition_with_nan(self,group):
		"""single reflections have NaNs because they were not determined (ie. missing cone).
		Other reflections in the group can still be used to repartition some reflections"""

		new_refs = group[self._powder].copy()

		not_nan = group[self._single].notnull()

		sum_powder  = sum(group[self._powder][not_nan])
		sum_single = sum(group[self._single][not_nan])

		for i in xrange(len(group)):
			if np.isnan(group[self._single][i]):
				continue
			ref_el = group[self._single][i]
			new_refs[i] = ref_el * sum_powder/sum_single

		return new_refs

	def repartition_with_nan_case_1(self,group):
		"""Special case when dealing with NaNs if sum of single == 0
		0-reflections from single are used to set corresponding powder reflections to 0, and
		to redistribute reflection intensity over other powder reflections.
		Re-partitioning happens on basis of current partitioning of powder reflections."""

		new_refs = [np.NaN] * len(group)
		is_nan  = group[self._single].isnull()

		
		sum_powder     = sum(group[self._powder])
		sum_powder_nan = sum(group[self._powder][is_nan])
		
		for i in xrange(len(group)):
			if group[self._single][i] == 0:
				new_refs[i] = 0.0
			else:
				ref_x_old = group[self._powder][i]
				new_refs[i] = ref_x_old * sum_powder/sum_powder_nan

		return new_refs

	def repartition_with_nan_case_2(self,group):
		"""Special case of case_1, if xray reflections to partition with are also 0, 
		reflection intensity is instead divided by the number of reflections."""

		new_refs = [np.NaN] * len(group)   # group[self._powder].copy()

		sum_powder = sum(group[self._powder])
		
		n_nan = group[self._single].isnull().sum()
		
		for i in xrange(len(group)):
			if group[self._single][i] == 0:
				new_refs[i] = 0.0
			else:
				new_refs[i] = sum_powder / n_nan

		return new_refs

	def repartition_with_nan_case_3(self,group): # also case 4
		"""Special case when only a __single__ red value is present in group. The corresponding xrd reflection is
		based on the rank of the red reflection. The sum of the xrd group is corrected accordingly, and the
		remaining sum is equipartitioned over the other reflections."""

		new_refs = pd.Series(index=group.index,data=[np.NaN] * len(group))   # group[self._powder].copy()

		sum_powder = group[self._powder].sum()

		n_nan = group[self._single].isnull().sum()

		kind = self.kind

		for index, row in group.iterrows():
			if not pd.isnull(row[self._single]):

				rank = int(self.df_bins.ix[[index]]['bin'+self.single])
				val  = self.bin_powder[rank]

				if kind == 'f':
					val = val
				if kind == 'f2':
					val = val**2
				elif kind == 'mf2':
					# double brackets needed, because .ix expects a list
					# [0] will return int, otherwise series
					m = self.df.ix[[index]]['m'][0] 
					val = m*val**2

				#print 'rank = {:2d}  /  sum_powder =  {:7.2f}  /  val = {:7.2f}'.format(rank, sum_powder, val)
				val = min(sum_powder,val)
				new_refs.ix[[index]] = val
				sum_powder -= val
		for index, row in group.iterrows():
			if pd.isnull(row[self._single]):
				new_refs.ix[[index]] = sum_powder / n_nan

		return new_refs

	def statistics(self):
		"""Print detailed summary of repartitioning procedure"""

		print '\n ** Summary of repartitioning procedure'
		print 'Using {} to repartition {} based on equal {}, output: {}'.format(self.single,self.powder,self.overlap,self.kind,self.target)
		print '{:10s} {:>7s} {:>7s}'.format('case','nrefs','ngroups')
		print '-'*26
		
		for case,nrefs,ngroups in (
			(self.str_missing, self.n_rep_missing, self.n_rep_missing_groups),
			(self.str_single,  self.n_rep_single,  self.n_rep_single_groups),
		    (self.str_zero,    self.n_rep_zero,    self.n_rep_zero_groups),
		    (self.str_normal,  self.n_rep_normal,  self.n_rep_normal_groups),
		    (self.str_case0,   self.n_rep_case0,   self.n_rep_case0_groups),
		    (self.str_case1,   self.n_rep_case1,   self.n_rep_case1_groups),
		    (self.str_case2,   self.n_rep_case2,   self.n_rep_case2_groups),
		    (self.str_case3,   self.n_rep_case3,   self.n_rep_case3_groups),
		    (self.str_case4,   self.n_rep_case4,   self.n_rep_case4_groups),
		    (self.str_case5,   self.n_rep_case5,   self.n_rep_case5_groups),
		    (self.str_rest,    self.n_rep_rest,    self.n_rep_rest_groups)
		    ):
			func = self.func_dict[case]
			print "{:<10s} {:7d} {:7d}  --> {}".format(case,nrefs,ngroups,func)
		
		print '-'*26
		print '{:<10s} {:7d} {:7d}  --> {} NaNs'.format('Total',len(self.df), max(self.df[self.overlap]) + 1, self.df[self.single].isnull().sum() )

		sum_rep =  sum((self.n_rep_missing,self.n_rep_single,self.n_rep_zero,self.n_rep_normal,self.n_rep_case0,self.n_rep_case1,self.n_rep_case2,self.n_rep_case3,self.n_rep_case4,self.n_rep_case5,self.n_rep_rest))

		if self.dmin:
			sel = self.df['d'] > self.dmin
			total = sel.sum()
			mdso = max(self.df[sel][self.overlap])
		else:
			mdso = max(self.df[self.overlap])
			total   =  len(self.df)

		print '\nCheck if number of repartioned equals total number of reflections: {} == {}'.format(sum_rep,total),
		assert sum_rep == total, '... NOT OK\nNumber of repartitioned reflections does not match total. \nMaybe ran classify and run together if one is 2x the other?'
		print '...OK'

		assert sum((self.n_rep_missing_groups,
					self.n_rep_single_groups,
					self.n_rep_zero_groups,  
					self.n_rep_normal_groups,
					self.n_rep_case0_groups, 
					self.n_rep_case1_groups, 
					self.n_rep_case2_groups, 
					self.n_rep_case3_groups, 
					self.n_rep_case4_groups, 
					self.n_rep_case5_groups, 
					self.n_rep_rest_groups)) == mdso + 1, 'Number of repartitioned groups does not match total.'


	def get_case(self,group):
		"""Returns the case of the group, which determines hwo the group is dealt with
		Checks the following, in this order:

		When all reflections are present
		single - Group contains one reflection
		zero   - sum of powder refs in group == 0
		normal - Group has all powder and single refs present

		In case a group has missing reflections:
		case0  - Group contains now single refs
		case1  - Sum of single refs == 0 and sum of powder refs > 0
		case2  - Sum of single refs == 0 and sum of powder refs absent from single refs == 0
		case3  - Group contains one nonzero single ref and 2 refs total
		case4  - Group contains one nonzero single ref
		case5  - Group is not covered by the above. Has multiple single refs present.
		
		rest   - Everything else. Groups where sum of single refs == 0, but sum of powder refs > 0"""

		_powder = self._powder
		_single = self._single

		if np.any(group[_powder].isnull()):
			case = self.str_missing
			#raise ValueError, 'Unexpected NaN in group {} for {}'.format(list(group.index),_powder)
		
		elif len(group) == 1:
			case = self.str_single
		
		elif group[_powder].sum() == 0:
			case = self.str_zero

		elif np.all(group[_single].isnull()):
 			case = self.str_case0
		
		elif np.any(group[_single].isnull()):
			is_nan  = group[_single].isnull()
			
			sum_powder = group[_powder].sum()
			sum_single = group[_single].sum()
	
			if sum_single == 0:
				if group[_powder][is_nan].sum() > 0:
					case = self.str_case1
				else:
					case = self.str_case2

			elif len(group) == 2:
				case = self.str_case3
	
			elif group[_single].notnull().sum() == 1:
				case = self.str_case4

			else:
				case = self.str_case5
						
		elif group[_single].sum() > 0:
			case = self.str_normal
				
		else:
			case = self.str_rest

		return case

	def setup_bins(self,nbins=10,unitary=None,table=None,atoms=None):
		"""Sets up bins for the 'repartition_with_nan_case_3' function

		self.df_bins: dataframe containing ranks of single and powder data sets
		self.bin_single: bins for single data
		self.bin_powder: bins for powder data

		If unitary (column in df to calculate unitary structure factors for), atoms (dict of atoms+count)
		and type of scattering factor table (electron/xray) are present, ranking is instead calculated
		using unitary structure factors"""

		single = self.single
		powder = self.powder
		df = self.df

		df_bins, bin_single, bin_powder = digitize(df,single=single,powder=powder,nbins=nbins)
		self.bin_single = bin_single
		self.bin_powder = bin_powder
		self.df_bins = df_bins

		if unitary and atoms and table:
			prefix = 'e_sf_' if table == 'electron' else 'x_sf_'
			calc_scattering_factor(df,atoms,table=table)
			df[unitary] = calc_unitary_f(df,atoms,f_col=single,prefix=prefix)

			powder = powder
			df_bins_u, bin_single_u, bin_powder_u = digitize(df,single=unitary,powder=powder,nbins=nbins)
			df_bins['bin'+single] = df_bins_u['bin'+unitary]


		print '\nBins'
		print 'rank     single     powder'
		for i,(x,y) in enumerate(zip(bin_single,bin_powder)):
			print '  {:>2d} {:10.4f} {:10.4f}'.format(i,x,y)
		print
		

	def counter(self,case,n):
		"""Increments the number of reflections and groups handled

		case: name of the case
		n:    length of group"""
		self.__dict__['n_rep_' + case            ] += n
		self.__dict__['n_rep_' + case + '_groups'] += 1

	def run(self):
		"""Runs the repartitioning algorithm"""

		if self.has_run == True:
			raise ValueError, "I don't know what the effects are of run/classify twice, please create a new instance of Repartition class."
		self.has_run = True

		df      = self.df
		powder  = self.powder
		single  = self.single
		target  = self.target
		overlap = self.overlap
		kind    = self.kind

		df[target] = np.NaN

		if kind == 'f':
			_powder = powder
			_single = single
		elif kind == 'f2':
			_powder = powder+'_'+kind
			_single = single+'_'+kind
			df[_powder] = df[powder]**2
			df[_single] = df[single]**2
		elif kind == 'mf2':
			_powder = powder+'_'+kind
			_single = single+'_'+kind
			df[_powder] = df['m'] * df[powder]**2
			df[_single] = df['m'] * df[single]**2
		else:
			raise ValueError, "Unknown value '{}' for kind, should be in (f,f2,mf2)".format(kind)

		self._powder = _powder
		self._single = _single

		for j, group in df.groupby(overlap):

			if self.dmin:
				if group['d'].min() < self.dmin:
					print ' >> dmin limit reached ({}) ==> dmin of rep\'d refs = {:.4f}'.format(self.dmin,group['d'].max())
					self.dmin = group['d'].max()
					break

			case = self.get_case(group)

			func_name = self.func_dict[case]
			func = 	getattr(self,func_name)

			new_refs = func(group)

			assert len(new_refs) == len(group), "case: {}  /  group: {}".format(case,j)

			index = group.index
			df[target][index] = new_refs

			self.counter(case,len(group))

		if kind == 'f':
			pass
		elif kind == 'f2':
			df[target] = df[target]**0.5
			del df[_powder]
			del df[_single]
		elif kind == 'mf2':
			df[target] = (df[target] / df['m'])**0.5
			del df[_powder]
			del df[_single]
		else:
			raise ValueError, "Unknown value '{}' for kind, should be in (f,f2,mf2)".format(kind)

		self.statistics()

	def classify(self):
		"""Adds column to df with the cases for each reflection"""
		df      = self.df
		target  = self.target+'_case'
		overlap = self.overlap
		self._powder = self.powder
		self._single = self.single

		df[target] = ''

		for j, group in df.groupby(overlap):
			case = self.get_case(group)

			index = group.index
			df[target][index] = case

			self.counter(case,len(group))

		self.statistics()



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


def f_calc_structure_factors(structure,**kwargs):
	"""Takes cctbx structure and returns f_calc miller array
	Takes an optional options dictionary with keys:
	input:
		**kwargs:
			'd_min': minimum d-spacing for structure factor calculation
			'algorithm': which algorithm to use ('direct', 'fft', 'automatic')
		structure: <cctbx.xray.structure.structure object>
	output:
		f_calc: <cctbx.miller.array object> with calculated structure factors
			in the f_calc.data() function
	
	TODO:
	- make this more general?
	- allow for specification of more parameters (like tables, ie. it1992 or wk1995)
	"""

	dmin		= kwargs.get('dmin',1.0)
	algorithm 	= kwargs.get('algorithm',"automatic")
	anomalous 	= kwargs.get('anomalous',False)
	table 		= kwargs.get('scatfact_table','wk1995')
	return_as   = kwargs.get('return_as',"series")

	if dmin <= 0.0:
		raise ValueError, "d-spacing must be greater than zero."

	if algorithm == "automatic":
		if structure.scatterers().size() <= 100:
			algorithm = "direct"
		else:
			algorithm = None

	structure.scattering_type_registry(table=table)


	f_calc_manager = structure.structure_factors(
		anomalous_flag = anomalous,
		d_min = dmin,
		algorithm = algorithm)
	f_calc = f_calc_manager.f_calc()
	
	print "\nScattering table:", structure.scattering_type_registry_params.table
	structure.scattering_type_registry().show()
	print "Minimum d-spacing: %g" % f_calc.d_min()

	if return_as == "miller":
		return f_calc
	elif return_as == "series":
		fcalc = pd.Series(index=f_calc.indices(),data=np.abs(f_calc.data()))
		phase = pd.Series(index=f_calc.indices(),data=np.angle(f_calc.data()))
		return fcalc,phase
	elif return_as == "df":
		dffcal = pd.DataFrame(index=f_calc.index)
		dffcal['fcalc'] = np.abs(f_calc.data())
		dffcal['phase'] = np.angle(f_calc.data())
		return dffcal
	else:
		raise ValueError, "Unknown argument for 'return_as':{}".format(return_as)


def calc_structure_factors(cif,dmin=1.0,combine=None,table='xray',prefix='',**kwargs):
	"""Wrapper around f_calc_structure_factors()
	Takes a cif file (str or file object)

	dmin can be a dataframe and it will take the minimum dspacing (as specified by col 'd') or a float
	if combine is specified, function will return a dataframe combined with the given one, otherwise a
	dictionary of dataframes

	prefix is a prefix for the default names fcalc/phases to identify different structures"""

	if isinstance(cif,str):
		f = open(cif,'r')

	if isinstance(dmin,pd.DataFrame):
		dmin = min(dmin['d']) - 0.00000001

	structures = read_cif(f)

	if isinstance(structures,xray.structure):
		structures = {"fcalc":structures}

	col_phases = prefix+"phases"
	col_fcalc  = prefix+"fcalc" 

	for name,structure in structures.items():
		fcalc,phase = f_calc_structure_factors(structure,dmin=dmin,scatfact_table=table,return_as="series",**kwargs)
		
		if len(structures) > 1:
			col_phases = prefix+"ph_"+name
			col_fcalc  = prefix+"fc_"+name

		dffcal = pd.DataFrame({col_phases:phase,col_fcalc:fcalc})

		if combine is not None:
			combine = combine.combine_first(dffcal)
		else:
			structures[name] = dffcal

	if combine is not None:
		try:
			return combine.sort('d',ascending=False)
		except KeyError:
			return combine
	else:
		return structures





def calc_scattering_factor(df,atoms,table='xray'):
	"""Calculates scattering factors for atoms (dict)"""
	if table == 'xray':
		prefix = 'x_sf_'
	elif table == 'electron':
		prefix = 'e_sf_'

	sitl = 1/(2*df['d'])

	for atom in atoms:
		df[prefix + atom] = f_calc_scattering_factor(atom,sitl,table=table)


def calc_scattering_factor_w_biso(df,atoms,table='xray',biso=2.0):
	"""Calculates scattering factors for atoms (dict)"""

	raw_input("VERIFY ME!! -> calc_scattering_factor_w_biso")

	if table == 'xray':
		prefix = 'x_sf_'
	elif table == 'electron':
		prefix = 'e_sf_'

	sitl = 1/(2*df['d'])

	for atom in atoms:
		f0 = f_calc_scattering_factor(atom,sitl,table=table)
		df[prefix + atom] = f0*np.exp(-biso*sitl)


def f_calc_scattering_factor(atom,sitl,table='xray'):
	"""Calculate scattering factors"""
	if table == 'xray':
		coefficients = atominfo.wk1995[atom]
		return atominfo.calc_sf_xray(atom,coefficients,sitl)
	elif table == 'electron':
		coefficients = atominfo.it_table_4322[atom]
		return atominfo.calc_sf_electron(atom,coefficients,sitl)


def calc_unitary_f(df,atoms,f_col='xrd_f',prefix='x_sf_'):
	"""Use together with calc_scattering_factor"""

	sum_sf_name = prefix+'sum'

	sum_sf = np.zeros_like(df[f_col])
	# sum_sf.name = sum_sf_name

	total = sum(atoms.values())
	print total
	print atoms

	for atom,number in atoms.items():
		col = prefix+atom
		sum_sf += df[col]*number/total

	U = df[f_col] / sum_sf

	df[sum_sf_name] = sum_sf
	
	return U

def calc_normalized_f(df,atoms,f_col='xrd_f',prefix='x_sf_'):
	"""Use together with calc_scattering_factor

	Expected values intensity statistics, Dunitz 1797, p 155 direct methods / p 99 intensity statisitcs
			  centro  non-centro
	<|E^2|>   1.000        1.000
	<|E|>     0.798        0.886
	<|E^2|-1> 0.968        0.736
	"""

	raw_input("VERIFY ME!! -> calc_unitary_f")
	sum_sf_name = prefix+'norm'

	sum_sf = np.zeros_like(df[f_col])
	# sum_sf.name = sum_sf_name

	total = sum(atoms.values())
	print total
	print atoms

	for atom,number in atoms.items():
		col = prefix+atom
		sum_sf += (df[col]**2) * (number/total)

	e = 1 # for general position -> dunitz 1979, p 156
	E = df[f_col] / (e*sum_sf**0.5)

	df[sum_sf_name] = sum_sf**0.5
	
	return E

	
def calc_sitl(df):
	"""Calculate sin(th)/l from d"""
	return 1/(2*df['d'])



def calc_volume(a,b,c,al,be,ga):
	"""Returns volume for the general case from cell parameters"""
	al = math.radians(al)
	be = math.radians(be)
	ga = math.radians(ga)
	V = a*b*c*((1+2*math.cos(al)*math.cos(be)*math.cos(ga)-math.cos(al)**2-math.cos(be)**2-math.cos(ga)**2)**.5)
	return V

def calc_dspacing(df,cell,col='d',kind='triclinic',inplace=True):
	"""Calculate dspacing on df from indices"""
	a,b,c,al,be,ga = cell
	h,k,l = map(np.array,zip(*df.index))
	d = f_calc_dspacing(a,b,c,al,be,ga,h,k,l,kind=kind)
	if inplace:
		df[col] = d
	else:
		return d


def r_amplitude(df,fobs,fcal):
	"""Calculate agreement value for the structure factor amplitudes"""
	numerator   = abs(df[fobs]-df[fcal]).sum()
	denominator = abs(df[fobs]).sum()
	if denominator == 0:
		return np.inf
	else:
		return numerator/denominator

def r_bragg(df,fobs,fcal,m='m'):
	"""Calculate agreement value for the intensities (m F^2)"""
	numerator   = abs(df[m]*df[fobs]**2 - df[m]*df[fcal]**2).sum()
	denominator = abs(df[m]*df[fobs]**2).sum()
	if denominator == 0:
		return np.inf
	else:
		return numerator/denominator


def f_calc_dspacing(a,b,c,al,be,ga,h,k,l,kind='triclinic'):
	"""
	Calculates d-spacing based on given parameters.
	a,b,c,al,be,ge are given as floats
	al,be,ga can be given as ndarrays or floats
	kind specifies the type of cell -> triclinic works for the general case, but is a bit slower
	although still fast enough

	Tested: orthorhombic cell on (orthorhombic, monoclinic, triclinic)
	Tested: triclinic cell with dvalues from topas
	"""

	if kind == 'cubic':
		print '\n** Warning: cubic dspacing calculation unverified!! **\n'
		idsq = (h**2 + k**2 + l**2) / a**2

	elif kind == 'tetragonal':
		print '\n** Warning: tetragonal dspacing calculation unverified!! **\n'
		idsq = (h**2 + k**2) / a**2 + l**2 / c**2

	elif kind == 'orthorhombic':
		idsq = h**2 / a**2 + k**2 / b**2 + l**2 / c**2 

	elif kind == 'hexagonal':
		print '\n** Warning: hexagonal dspacing calculation unverified!! **\n'
		idsq = (4/3) * (h**2 + h*k + k**2) * (1/a**2) + l**2 / c**2

	elif kind == 'monoclinic':
		print '\n** Warning: monoclinic dspacing calculation unverified!! **\n'
		be = math.radians(be)
		idsq = (1/math.sin(be)**2) * (h**2/a**2 + k**2 * math.sin(be)**2 / b**2 + l**2/c**2 - (2*h*l*math.cos(be)) / (a*c))

	elif kind == 'triclinic':
		V = calc_volume(a,b,c,al,be,ga)
	
		al = math.radians(al)
		be = math.radians(be)
		ga = math.radians(ga)
	
		idsq = (1/V**2) * (
		  h**2 * b**2 * c**2 * math.sin(al)**2
		+ k**2 * a**2 * c**2 * math.sin(be)**2
		+ l**2 * a**2 * b**2 * math.sin(ga)**2
		+ 2*h*k*a*b*c**2 * (math.cos(al) * math.cos(be) - math.cos(ga))
		+ 2*k*l*b*c*a**2 * (math.cos(be) * math.cos(ga) - math.cos(al))
		+ 2*h*l*c*a*b**2 * (math.cos(al) * math.cos(ga) - math.cos(be))
		)

	d = 1/idsq**0.5

	return d


def calc_2th_from_d(d,wavelength):
	"""Calculates 2theta values for given dspacing -> takes float or ndarray"""
	return 2 * np.degrees(np.arcsin((wavelength) / (2*d))) 


def calc_d_from_2th(twotheta,wavelength):
	"""Calculates dspacings for given twotheta values -> takes float or ndarray"""
	theta = np.radians(twotheta / 2)
	return wavelength / (2*np.sin(theta))


def select_single(df,key='overlap'):
	"""Returns a boolean array corresponding to the overlap groups that contain more than 1 reflection
	(key == 'overlap')"""
	bool_list = []

	for j in xrange(max(df[key] + 1)):
		occ = (df[key] == j).sum()
		if occ == 1:
			bool_list.append(True)
		else:
			bool_list.extend([False]*occ)

	return np.array(bool_list)


def weak_reflection_elimination(df,single='RED_F',powder='XRD_F',ratio=0.5, kind='mf2',out=None):
	"""Convenience function around f_weak_reflection_elimination, which will do weak reflection
	elimination, repartition the data and optionally print a file with weak reflections filtered out.

	Repartitioning is done by adding 0.0001 to all strong reflections and converting all the NaN to 0,
	in order to trick the algorithm to treat all overlap groups in the proper way (redistributing 
	group intensity to strong reflections)"""
	f_weak_reflection_elimination(df,single=single,powder=powder,ratio=ratio)
	df['strong'] += 0.0001
	df['strong'] = df['strong'].fillna(0)
	repartition(df, single='strong',kind=kind)
	df['strong'][df['strong'] != 0.000] -= 0.0001

	if out:
		strong = df['WRE']
		write_hkl(df[strong],cols=('strong','FWHM'),out=out)


def f_weak_reflection_elimination(df,single='RED_F',powder='XRD_F',ratio=0.5):
	"""Applies weak reflection elimination procedure based on Xie (2010), PhD Thesis
	Adds 2 columsn to data frame, 'WRE', boolean array corresponding to 'not weak' reflections
	and 'strong', an array of all the 'not weak' reflections where missing ones are NaN"""

	sum_el = np.sum(df[single])
	n_not_nan  = df[single].notnull().sum()

	average = sum_el / n_not_nan
	threshold = ratio * average

	weak = df[single] < threshold

	print 'Threshold = {:.4f}'.format(threshold)
	print '{} out of {} reflection eliminated using {} single reflections'.format(weak.sum(),len(df),n_not_nan)

	df['WRE'] = weak == False
	df['strong'] = df[powder][weak==False]
	#df['strong'] = df['strong'].fillna(0)


def completeness(df,target='red_f',dmin=0.01, cuts=[], return_xy=False):
	print '\nCoverage statistics for {}'.format(target)
	print
	print ' Resolution             Refls. (shell - total)     Coverage    '
	print ' sin(th)/l       dmin   obs    calc   obs    calc  shell  total'    

	cuts = [1/(2.0*dval) for dval in cuts]

	sitl = 1/(2*df['d'])

	max_sitl = max(sitl)
	step = 0.050

	if 1/(2*dmin) < max_sitl:
		max_sitl = 1/(2*dmin)

	tot_obs = 0
	tot_cal = 0

	ret = []

	rng = sorted(np.arange(0.0,max_sitl,step).tolist() + [max_sitl] + cuts)

	for mn,mx in pairwise(rng):

		dmin = 1/(2*mx)
		ind = (mn < sitl) & (sitl <= mx)

		if ind.sum() == 0:
			continue # either index is broken (causes ZeroDivError below), or just no reflections in shell			

		shell_obs = df[target][ind].notnull().sum()
		shell_cal = len(df[target][ind])

		tot_obs += shell_obs
		tot_cal += shell_cal

		try:
			shell_cov = float(shell_obs)/float(shell_cal)
			tot_cov   = float(tot_obs)  /float(tot_cal)
		except ZeroDivisionError:
			print 'Invalid index. Please generate complete index before trying to calculate index'
			print 'Make completeness independent from current data set but calculate for specified dmin'
			print "Note: with 'if ind.sum() == 0:'' this shouldn't happen anymore"
			break

		print '{:6.3f} -{:6.3f} {:6.3f} {:6d} {:6d} {:6d} {:6d} {:5.1f}% {:5.1f}%'.format(
										mn,mx,dmin,
										shell_obs, shell_cal, tot_obs, tot_cal, 
										shell_cov*100, tot_cov*100)
		ret.append([mn,mx,dmin,shell_obs, shell_cal, tot_obs, tot_cal,shell_cov, tot_cov])

	if return_xy:
		return [item[1] for item in ret],[item[8] for item in ret]

	# return pd.DataFrame(ret,columns = ('min_sitl','max_sitl','dmin','shell_obs','shell_cal','tot_obs','tot_cal','shell_cov','tot_cov') )


def completeness_plot(df, target):
	"""Plots completeness of all items given in target (either list or str)
	Also runs completeness stats on all targets"""
	import matplotlib.pyplot as plt
	if isinstance(target, str):
		target = [target]

	for item in target:
		x,y = completeness(df,target=item, return_xy=True)
		plt.plot(x,y, label=item)

	plt.title('Completeness')
	plt.xlabel('sin(th/l)')
	plt.ylabel('%')
	plt.legend()
	plt.show()



def completeness_cctbx(df,cell,spgr,data='F',dmin=None):
	"""Similar to completeness(), but in a different way to cross-check results
	about 6x slower, but does not require dmin values"""

	try:
		sel = df[data].notnull() # select notnull data items for index
	except KeyError:
		idx_obs = df.index
	else:
		idx_obs = df.index[sel]

	if not dmin:
		try:
			dmin = df[sel]['d'].min()
		except KeyError:
			print 'dmin could not be found, please specify dmin:'
			dmin = float(raw_input(" dmin >> [0.5] ") or 0.5)

	idx_obs = set(idx_obs)

	max_sitl = 1/(2.0*dmin)

	i = 0

	sum_obs = 0
	sum_cal = 0

	print '\nCoverage statistics'
	print
	print ' Resolution             Refls. (shell - total)     Coverage    '
	print ' sin(th)/l       dmin   obs    calc   obs    calc  shell  total'     

	while i*0.050 < max_sitl:
		mn = i*0.050
		mx = (i+1)*0.050

		if mx >= max_sitl:
			mx = max_sitl

		dmin = 1/(2*mx)

		idx_cal = generate_indices(cell,spgr,dmin=dmin)
		idx_cal = set(idx_cal)
		

		tot_obs = len(idx_cal & idx_obs)
		tot_cal = len(idx_cal)

		shell_obs = tot_obs - sum_obs
		shell_cal = tot_cal - sum_cal

		sum_cal += shell_cal
		sum_obs += shell_obs

		tot_cov = float(tot_obs) / tot_cal
		try:
			shell_cov = float(shell_obs) / shell_cal
		except ZeroDivisionError:
			shell_cov = np.NaN

		print '{:6.3f} -{:6.3f} {:6.3f} {:6d} {:6d} {:6d} {:6d} {:5.1f}% {:5.1f}%'.format(
										mn,mx,dmin,
										shell_obs, shell_cal, tot_obs, tot_cal, 
										shell_cov*100, tot_cov*100)
		i+=1


def write_hkl(df,cols=None,out=None,no_hkl=False,pre=None,post=None,data_fmt=None,hkl_fmt=None):
	"""Function for writing indices + selected columns to specified file/file object or terminal."""

	if isinstance(pre,list):
		if all('\n' in line for line in pre):
			pre = ''.join(pre)
		else:
			pre = '\n'.join(pre)
	elif isinstance(pre,str):
		pre = '\n'.join(pre.strip('\n'))

	if isinstance(post,list):
		post = ''.join(post)

	if not cols:
		cols = df.columns

	if isinstance(cols,str):
		cols = (cols,)

	if isinstance(out,str):
		out = open(out,'w')

	cols = list(cols)


	if not hkl_fmt:
		if no_hkl:
			hkl_fmt = ''
		else:
			hkl_fmt = '{:4d}{:4d}{:4d}'

	if not data_fmt:
		ifmt = '{:4d}'
		dfmt = ' {:5d}'
		ffmt = ' {:9.3f}'
		bfmt = ' {:4}'
	
		n = len(cols)
		data_fmt = ''
	
		for item in cols[:]: 
			if item == '*':
				cols.remove('*')
				data_fmt += '  *  '
				continue
	
			#tp = repr(type(df[item][0]))
			tp = repr(df[item].dtype)
			if   'int'   in tp: data_fmt += dfmt
			elif 'float' in tp: data_fmt += ffmt
			elif 'bool'  in tp: data_fmt += bfmt
			else:
				raise TypeError, "No format associated with type {}".format(tp)
	elif data_fmt == 'shelx':
		data_fmt = '{:8.3f}{:8.3f}'



	if pre:
		print >> out, pre

	print '>> Writing {} refs to file {}'.format(len(df),out.name if out else 'stdout')


	last = 0
	for row in df.reindex(columns=cols).itertuples():

		# if (abs(row[1:][2] - row[1:][4]) < 0.0001 and 
		#     abs(row[1:][2] - row[1:][5]) < 0.0001 and 
		#     abs(row[1:][2] - row[1:][6]) < 0.0001):
		# 	continue

		print >> out, hkl_fmt.format(*row[0])+data_fmt.format(*row[1:])
	
	if post:
		print >> out, post




def write_superflip(cell,spgr,wavelength=1.000,composition=None,df=None,datafile='blender.hkl',filename='sf.inflip',**kwargs):
	"""Creates a basic superflip input file for structure solution by asking a few simple questions"""
	dataformat = "amplitude fwhm"

	sps = make_special_position_settings(cell,spgr)
	sg = sps.space_group()
	uc = sps.unit_cell()
	#sgi = sps.space_group_info()

	fout = open(filename,'w')

	if df:
		write_hkl(df,cols=('repart','fwhm'),out=datafile)

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
	print >> fout, '\n'
	
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
			print >> fout, '# +(0 0 0) Inversion-Flag = 1, please check!'
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
	print >> fout, 'outputfile {}.xplor'.format(str(sg.info()).replace(' ','').lower())
	print >> fout, 'outputformat xplor'
	print >> fout
	print >> fout, 'dataformat', dataformat
	print >> fout, 'fbegin {}\n'.format(datafile)

def transform_cell(cell,spgr,out='shelx.ins'):
	"""Semi-automatic routine for handling cell transformation of hkl data using platon"""
	print "Writing file {}, run it with PLATON->LePage".format(out)
	print "Needs hklf file with same basename to work."
	write_shelx_ins(cell,spgr,out=out)

	print "Enter 3 lines (a') (b') (c') >> \n"
	line1 = raw_input("(a'): ")
	line2 = raw_input("(b'): ")
	line3 = raw_input("(c'): ")

	tr_mat = (
	line1[8:14],
	line1[14:20],
	line1[20:26],
	line2[8:14],
	line2[14:20],
	line2[20:26],
	line3[8:14],
	line3[14:20],
	line3[20:26] ) 

	# print tr_mat

	print "\nWriting file {}, run it again with PLATON->HKL-Transf".format(out)
	write_shelx_ins(cell,spgr,out=out,tr_mat=tr_mat)


def write_shelx_ins(cell,spgr,wavelength=1.0000,out='shelx.ins',tr_mat = None):
	"""Simple function that writes a basic shelx input file
	
	cell: cell parameters
	spgr: space group"""
	from iotbx.shelx.write_ins import LATT_SYMM
	symm = make_symmetry(cell,spgr)

	if isinstance(out,str):
		out = open(out,'w')
	
	cell = symm.unit_cell().parameters()
	Z = symm.space_group().order_z()
	sgi  = symm.space_group().info()

	print >> out, "TITL shelx ins file from blender"
	print >> out, "CELL {} {} {} {} {} {} {}".format(wavelength,*cell)
	print >> out, "ZERR {}  0  0  0  0  0  0".format(Z)
	LATT_SYMM(out,sgi.group())
	print >> out
	if tr_mat:
		print >> out, 'TRMX {} {} {} {} {} {} {} {} {} 0 0 0'.format(*tr_mat)
	print >> out
	print >> out, 'HKLF 4'
	print >> out, 'END'

def write_focus():
	print 'does nothing'
	pass

def write_ticks_file(cell,spgr,wavelength=1.000):
	"""Quick function to generate tick marks for given cell/spgr/wavelength"""
	index = generate_indices(cell,spgr)
	index = pd.Index(index)
	df = pd.DataFrame(index=index)
	calc_dspacing(df,cell)
	df['th2'] = calc_2th_from_d(df['d'],wavelength)
	df = df.sort('th2',ascending=True)
	write_hkl(df,cols=('th2'),out='ticks.out',no_hkl=True)
	return df

def quickplot(df,col1,col2,fmt='o'):
	"""Convenience function for plotting col1 and col2 against eachother"""
	import matplotlib.pyplot as plt
	x = df[col1]
	y = df[col2]
	plt.plot(x,y,fmt)
	plt.xlabel(col1)
	plt.ylabel(col2)
	plt.show()



def main(options,args):
	if options.template:
		print_template()
		exit()

	if options.cell:
		cell = uctbx.unit_cell(options.cell).parameters()
	spgr = options.spgr
	wavelength = options.wavelength

	if options.merge:
		for fin in args:
			### write file with indices for full set, using dmin/cell/spgr of last read file
			if fin == 'indices':
				with open('indices.hkl','w') as iout:
					for h,k,l in index:
						print >> iout, '{:4d}{:4d}{:4d} 0 0'.format(h,k,l)
				iout.close()
				continue
			###

			root,ext = os.path.splitext(fin)
			out = root+'_merged'+ext
			df = load_hkl(fin,shelx=True)
			completeness_cctbx(df,cell=cell,spgr=spgr,data='F2',dmin=0.5)
			m = df2m(df,cell,spgr,data='F2',sigmas='sigma')
			m = remove_sysabs(m)
			m = merge_sym_equiv(m)
			dmin = m.d_min()
			df = m2df(m,data='F2',sigmas='sigma')
			if df['sigma'].isnull().sum() > 0:
				print 'Sigmas have {} nans, setting sigmas as sqrt(F2)'.format(df['sigma'].isnull().sum())
				df['sigma'] = df['F2']**0.5
			index = generate_indices(cell,spgr,dmin=dmin)

			write_hkl(df,cols=('F2','sigma'),out=out,data_fmt='shelx')

	if options.write_ticks_file:
		write_ticks_file(cell,spgr,wavelength)


	if options.ipython:
		from IPython.terminal.embed import InteractiveShellEmbed
		ipshell = InteractiveShellEmbed(banner1='')
		ipshell()







if __name__ == '__main__':
	
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
						type=str, metavar="FILE",nargs='*',
						help="Paths to input files.")



		
	parser.add_argument("--ipython",
						action="store_true", dest="ipython",
						help="Starts interactive ipython shell on program finish.")

	parser.add_argument("--template",
						action="store_true", dest="template",
						help="Print template input file and exit")

	parser.add_argument("--merge",
						action="store_true", dest="merge",
						help="Merges the given data set and strips systematic absences. Expects h,k,l")

	parser.add_argument("-s","--spgr",
						action="store", type=str, dest="spgr",
						help="Space group (default P1)")

	parser.add_argument("-c","--cell",
						action="store", type=float, nargs="*", dest="cell",
						help="Unit cell, can be minimal representation")

	parser.add_argument("-w","--wavelength",
						action="store", type=float, dest="wavelength",
						help="Wavelength in ANGSTROM (default = 1.000)")



	parser.add_argument("--ticks",
						action="store_true", dest="write_ticks_file",
						help="Writes tick file for given unit cell/spgr and optionally wavelength")

	
#
#	parser.add_argument("-f", "--files", metavar='FILE',
#						action="store", type=str, nargs='+', dest="files",
#						help="Sflog files to open. This should be the last argument. (default: all sflog files in current directory)")


	
	parser.set_defaults(ipython=False,
						template=False,
						merge=False,
						cell=None,
						spgr="P1",
						wavelength=1.0000,
						write_ticks_file=False)
	
	options = parser.parse_args()
	args = options.args


	df = main(options,args)

