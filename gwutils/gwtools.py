## General tools

from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


import os
import re
import time
import numpy as np
import copy
import math
import cmath
import gc, h5py
from math import pi, factorial
from numpy import array, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors


# Note: syntax to load in another .py program, without making this a package
#import imp
#utils = imp.load_source('utils', '/Users/marsat/Mathematica/Programmes/eLISA/utils.py')
#from utils import *

# Cubic spline interpolation
# WARNING: Note that the defaults ext=0 allows extrapolation
# Use ext=2 to forbid any extrapolation (including  when out-of-bounds by machine-precision errors...)
# def spline(x, y, k=3, ext=0):
#     return ip.InterpolatedUnivariateSpline(x, y, k=k, ext=ext)

# Numerical values - taken from LAL for physical quantities, but pure numbers like pi are from numpy
msols = 4.925491025543575903411922162094833998e-6
msol = 1.988546954961461467461011951140572744e30
G = 6.67384e-11
pc = 3.085677581491367278913937957796471611e16
c = 2.99792458e8
yr = 31557600e0 # Julian year, 365.25*86400
# Additional numerical values as copied from the C code for LISA
au = 1.4959787066e11
yrsid = 3.15581497635e7 # Sideral year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html
Omega = 1.99098659277e-7 # Orbital pulsation: 2pi/year - use sidereal year as found on http://hpiers.obspm.fr/eop-pc/models/constants.html
# L_SI = 5.0e9 # Arm length of the detector (in m): (Standard LISA)
L = 2.5e9 # Arm length of the detector (in m): (L3 reference LISA)
R = 1.4959787066e11 # Radius of the orbit around the sun: 1AU


# Loading QNM
# Source : https://centra.tecnico.ulisboa.pt/network/grit/files/ringdown/
resources_dir = os.path.join(os.path.dirname(__file__), 'data/')
QNMlmndata = {}
QNMlmndata[(2,2,0)] = np.loadtxt(resources_dir+'QNM/QNMl2/n1l2m2.dat')
QNMlmndata[(2,1,0)] = np.loadtxt(resources_dir+'QNM/QNMl2/n1l2m1.dat')
QNMlmndata[(2,0,0)] = np.loadtxt(resources_dir+'QNM/QNMl2/n1l2m0.dat')
QNMomegalmnInt = {}
QNMsigmalmnInt = {}
QNMomegalmnInt[(2,2,0)] = spline(QNMlmndata[(2,2,0)][:,0], QNMlmndata[(2,2,0)][:,1])
QNMomegalmnInt[(2,1,0)] = spline(QNMlmndata[(2,1,0)][:,0], QNMlmndata[(2,1,0)][:,1])
QNMomegalmnInt[(2,0,0)] = spline(QNMlmndata[(2,0,0)][:,0], QNMlmndata[(2,0,0)][:,1])
QNMsigmalmnInt[(2,2,0)] = spline(QNMlmndata[(2,2,0)][:,0], -QNMlmndata[(2,2,0)][:,2])
QNMsigmalmnInt[(2,1,0)] = spline(QNMlmndata[(2,1,0)][:,0], -QNMlmndata[(2,1,0)][:,2])
QNMsigmalmnInt[(2,0,0)] = spline(QNMlmndata[(2,0,0)][:,0], -QNMlmndata[(2,0,0)][:,2])

# Function to close all open hdf5 files
def closeHDF5files():
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed

# Used for direct conversion of factorial to float, to avoid having huge integers when multiplying factorials together in sYlm or Wigner coeffs
def ffactorial(n):
    return float(factorial(n))

# Conversion from mass ratio to symmetric mass ration and back
def qofeta(eta):
    return (1.0 + np.sqrt(1.0 - 4.0*eta) - 2.0*eta) / (2.0*eta)
def etaofq(q):
    return q/(1.0 + q)**2
def Mchirpofm1m2(m1, m2):
    return (m1*m2)**(3./5) / (m1+m2)**(1./5)
def etaofm1m2(m1, m2):
    return (m1*m2)/(m1+m2)**2.
def m1ofMchirpeta(Mchirp, eta):
    M = Mchirp * np.power(eta, -3./5)
    delta = np.sqrt(1 - 4*eta)
    return M * (1+delta)/2.
def m2ofMchirpeta(Mchirp, eta):
    M = Mchirp * np.power(eta, -3./5)
    delta = np.sqrt(1 - 4*eta)
    return M * (1-delta)/2.

# Newtonian estimate of the relation Mf(deltat/M) (for the 22 mode) - gives the starting geometric frequency for a given mass ratio and a given geometric duration of the observations
def funcNewtonianfoftGeom(q, deltat):
    nu = q/(1. + q)/(1. + q)
    return 1./pi*(256*nu/5.*deltat)**(-3./8)
# Newtonian estimate of the relation f(deltat) (for the 22 mode) - gives the starting geometric frequency for a given mass ratio and a given geometric duration of the observations-output in Hz -- m1,m2,t in solar masses and s
def funcNewtonianfoft(m1, m2, deltat):
    mtot = m1 + m2
    q = m1/m2
    return funcNewtonianfoftGeom(q, deltat/(mtot*msols))/(mtot*msols)
# Newtonian estimate of the relation deltat/M(Mf) (for the 22 mode) - gives the time to merger from a given starting geometric frequency for a given mass ratio
def funcNewtoniantoffGeom(q, f):
    nu = q/(1. + q)/(1. + q)
    return 5./256/nu*(pi*f)**(-8./3)
# Newtonian estimate of the relation deltat(f) (for the 22 mode) - gives the time to merger from a given starting  frequency for a given mass ratio - output in s -- m1,m2,f in solar masses and Hz
def funcNewtoniantoff(m1, m2, f):
    mtot = m1 + m2
    q = m1/m2
    return funcNewtoniantoffGeom(q, (mtot*msols)*f)*(mtot*msols)

# Schwarzschild ISCO for total mass M in solar masses
def funcfISCO(M):
    return 1./(M*msols) * 1./pi * (1./6)**(3./2)

# List of (l,m) modes (tuples) for a given value of l
def listmodesl(l):
    return [(l, l-i) for i in range(2*l+1)]

# Loading binary data
def load_binary_data(filename, ncol):
    data = np.fromfile(filename)
    n = len(data)
    if not n % ncol == 0:
        raise ValueError('Error in loadbinary: ncol does not divide data length.')
    else:
        return data.reshape((n//ncol, ncol))

# Mod 2pi - shifted to return a result in [-pi, pi[
def mod2pi(x):
    rem = np.remainder(x, 2*pi)
    if isinstance(x, np.ndarray) and x.shape is not ():
        mask = (rem>pi)
        rem[mask] -= 2*pi
    else:
        if rem>pi:
            rem -= 2*pi
    return rem
# Mod pi
def modpi(x):
    return np.remainder(x, pi)

# Unwrapping phase data with a modulus other than 2pi
def unwrap_mod(p, mod=2*np.pi, axis=-1):
    p = np.asarray(p)
    nd = len(p.shape)
    dd = np.diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    ddmod = np.mod(dd + mod/2., mod) - mod/2.
    np.core.numeric.copyto(ddmod, mod/2., where=(ddmod == -mod/2.) & (dd > 0))
    ph_correct = ddmod - dd
    #_nx.copyto(ph_correct, 0, where=abs(dd) < discont)
    up = np.array(p, copy=True, dtype='d')
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up

################################################################################

# Restrict data to an interval according to first column
# If data[:,0]=x and interval=[a,b], selects data such that a<=x<=b
# Typically results in data being entirely inside interval
def restrict_data(data, interval):
    if not interval: # test if interval==[]
        return data
    else:
        # Make an initial guess based on global length - then adjust starting and ending indices
        x = data[:,0]
        n = len(data)
        deltax = (x[-1] - x[0]) / n
        if interval[0] < x[0]:
            ibeg = 0
        else:
            ibeg = min(int((interval[0]-x[0]) / deltax), n-1)
            while ibeg < n and x[ibeg] < interval[0]:
                ibeg += 1
            while ibeg > 0 and x[ibeg-1] > interval[0]:
                ibeg -= 1
        if interval[-1] > x[-1]:
            iend = n-1
        else:
            iend = n-1 - min(int((x[-1] - interval[-1]) / deltax), n-1)
            while iend > 0 and x[iend] > interval[-1]:
                iend -= 1
            while iend < n-1 and x[iend+1] < interval[-1]:
                iend += 1
        return data[ibeg:iend+1]
# Restrict data to an interval according to first column
# If data[:,0]=x and interval=[a,b], selects data from the last point such that x<=a to the first point such that b<=x
# Typically results in interval being entirely covered by data
def restrict_data_soft(data, interval):
    if not interval: # test if interval==[]
        return data
    else:
        # Make an initial guess based on global length - then adjust starting and ending indices
        x = data[:,0]
        n = len(data)
        deltax = (x[-1] - x[0]) / n
        if interval[0] < x[0]:
            ibeg = 0
        else:
            ibeg = min(int((interval[0]-x[0]) / deltax), n-1)
            while ibeg < n-1 and x[ibeg+1] <= interval[0]:
                ibeg += 1
            while ibeg > 0 and x[ibeg] > interval[0]:
                ibeg -= 1
        if interval[-1] > x[-1]:
            iend = n-1
        else:
            iend = n-1 - min(int((x[-1] - interval[-1]) / deltax), n-1)
            while iend > 0 and x[iend-1] >= interval[-1]:
                iend -= 1
            while iend < n-1 and x[iend+1] < interval[-1]:
                iend += 1
        return data[ibeg:iend+1]
# Trim zeros from a waveform according to chosen columns
def trim_zeros_bycol(data, cols, ifallzero_returnvoid=False):
    ilastnonzero = len(data)-1
    while (ilastnonzero>=0) and all([data[ilastnonzero, i]==0. for i in cols]):
        ilastnonzero -= 1
    ifirstnonzero = 0
    while (ifirstnonzero<=len(data)-1) and all([data[ifirstnonzero, i]==0. for i in cols]):
        ifirstnonzero += 1
    if ifirstnonzero>ilastnonzero and not ifallzero_returnvoid: # if everything is zero, do nothing
        return data
    else:
        return data[ifirstnonzero:ilastnonzero+1]

# Integration with the discrete trapeze rule
def integrate_trapeze(data):
    return np.sum(np.diff(data[:,0]) * (data[:-1,1] + data[1:,1])/2)
# Normalize data series with integral computed from discrete trapeze rule (e.g., normalize pdf)
def normalize_trapeze(data):
    res = np.copy(data)
    res[:,1] = res[:,1] / integrate_trapeze(data)
    return res
# Cumulative integral, computed with the discrete trapeze rule, of data series (e.g., cumulative of pdf)
def cumulative_trapeze(data):
    res = np.copy(data)
    res[0,1] = 0
    res[1:,1] = np.cumsum(np.diff(data[:,0]) * (data[:-1,1] + data[1:,1])/2)
    res[:,1] = res[:,1] / res[-1,1]
    return res

# Logarithmic sampling
def logspace(start, stop, nb):
    ratio = (stop/start)**(1./(nb-1))
    return start * np.power(ratio, np.arange(nb))

# Resample waveform (either Re/Im or Amp/Phase) on input samples
def resample(wf, samples):
    int1 = spline(wf[:,0], wf[:,1])
    int2 = spline(wf[:,0], wf[:,2])
    return np.array([samples, int1(samples), int2(samples)]).T

# Random values on an interval - uniformly sampled
def random_values(xmin, xmax, n):
    return xmin + np.sort(np.random.rand(n)) * (xmax-xmin)
# Random values on a hypercube, returns flat array - uniformly sampled in all directions
def random_values_multidim(intervals, n):
    d = len(intervals)
    arr = np.random.rand(n, d)
    for i in range(d):
        arr[:,i] = intervals[i][0] + arr[:,i] * (intervals[i][1] - intervals[i][0])
    return arr

# Quadratic Legendre interpolation polynomial
def quad_legendre(x, y):
    res = np.zeros(3, dtype=y.dtype)
    if (not len(x)==3) or (not len(y)==3):
        raise ValueError('Only allows an input length of 3 for x and y.')
    c0 = y[0] / ((x[0]-x[1]) * (x[0]-x[2]))
    c1 = y[1] / ((x[1]-x[0]) * (x[1]-x[2]))
    c2 = y[2] / ((x[2]-x[0]) * (x[2]-x[1]))
    res[0] = c0*x[1]*x[2] + c1*x[2]*x[0] + c2*x[0]*x[1]
    res[1] = -c0*(x[1]+x[2]) - c1*(x[2]+x[0]) - c2*(x[0]+x[1])
    res[2] = c0 + c1 + c2
    return res

# Normalize vector
def norm(vec):
    return np.linalg.norm(vec)
def normalize(vec):
    return vec / np.linalg.norm(vec)

# Step function from 0 to 1 - increase steepness of transition at 0.5 by increasing degree
def funcpolystep(x, degree):
    if 0<=x<=0.5:
        return 0.5 * (2*x)**degree
    elif x<=1:
        return 1 - 0.5 * (2*(1-x))**degree
def funcpolyleft(x, degree):
    return x**degree
def funcpolyright(x, degree):
    return 1. - (1.-x)**degree
# Functions generating values denser at either end or both - increase degree to increase concentration
def xvaluesdenserend(xmin, xmax, nbpt, degree=2):
    return np.array(list(map(lambda x: xmin + (xmax-xmin) * funcpolystep((x-xmin)/(xmax-xmin), degree), np.linspace(xmin, xmax, nbpt))))
def xvaluesdenserend_left(xmin, xmax, nbpt, degree=2):
    return np.array(list(map(lambda x: xmin + (xmax-xmin) * funcpolyleft((x-xmin)/(xmax-xmin), degree), np.linspace(xmin, xmax, nbpt))))
def xvaluesdenserend_right(xmin, xmax, nbpt, degree=2):
    return np.array(list(map(lambda x: xmin + (xmax-xmin) * funcpolyright((x-xmin)/(xmax-xmin), degree), np.linspace(xmin, xmax, nbpt))))

# Function to find closest value in series
def find_closest_value(data, value):
    n = len(data)
    if n==1:
        return data[0]
    if data[n/2]==value:
        return value
    elif data[n/2]<value:
        return find_closest_value(data[n/2:], value)
    else:
        return find_closest_value(data[:n/2], value)
def find_closest_index(data, value):
    valuearray = find_closest_value(data, value)
    return np.where(data == valuearray)[0][0]

# Function to write all possible combinations (tensor product) of elements of 1d arrays - returns a 2-dim array
# argslist is a list of numpy arrays
def kronecker_join_flat(argslist):
    l = len(argslist)
    if l==1:
        return argslist[0]
    else:
        sdlast = argslist[-2]
        last = argslist[-1]
        if last.shape==(len(last),):
            last = last.reshape(last.shape + (1,))
        p = last.shape[0]
        q = last.shape[1]
        n = len(sdlast)
        lastkron = np.zeros((n*p, q+1))
        for i in xrange(n):
            for j in xrange(p):
                lastkron[i*p+j] = np.insert(last[j], 0, sdlast[i])
        if l==2:
            return lastkron
        else:
            return kronecker_join_flat(argslist[:-2] + [lastkron])
# argslist is a list of list - convert first to np arrays with dtype=object
# WARNING: found issues, see TSEOBv4_sanity.ipynb
# listrangesq = [[1.0, 2.0], [1.0, 20.0], [1.0, 1.0]]
# listrangeschi = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
# kronecker_join_list_flat(listrangesq, listrangeschi) -- fails
def kronecker_join_list_flat(argslist):
    argslista = list(map(lambda x: np.array(x, dtype=object), argslist))
    l = len(argslista)
    if l==1:
        res = argslist[0]
    else:
        sdlast = argslista[-2]
        last = argslista[-1]
        if last.shape==(len(last),):
            last = last.reshape(last.shape + (1,))
        p = last.shape[0]
        q = last.shape[1]
        n = len(sdlast)
        lastkron = np.zeros((n*p, q+1), dtype=object)
        for i in xrange(n):
            for j in xrange(p):
                lastkron[i*p+j] = np.insert(last[j], 0, sdlast[i])
        if l==2:
            res = list(map(list, lastkron))
        else:
            res = kronecker_join_list_flat(argslist[:-2] + [list(lastkron)])
    return res
# From Mathematica - hence unconventional indices in the internals
def combine_lists_flat_index(lengths, index):
    n = len(lengths)
    temp = index+1 # go to Mathematica convention
    res = []
    for i in range(1, n+1):
        res = res + [1 + (temp-1) // int(np.prod(np.array(lengths[i:])))]
        temp = (temp-1) % int(np.prod(np.array(lengths[i:]))) + 1
    res = list(map(lambda x: x-1, res)) # go back to the python convention
    return res
def combine_lists_flat(*args):
    if len(args)==1:
        return list(map(lambda x: [x], args[0]))
    else:
        argslist = list(args)
        nargs = len(argslist)
        lengths = list(map(len, argslist))
        n = int(np.prod(np.array(lengths)))
        indices = [combine_lists_flat_index(lengths, i) for i in range(0, n)] # changed the ranges from Mathematica
        res = [ [argslist[k][indices[i][k]] for k in range(0, nargs)] for i in range(0, n)] # changed the ranges from Mathematica
        return res

#Definitions for the windowing function
#In order to avoid overflows in the exponentials, we set the boundaries (di, df) so that anything below 10^-20 is considered zero
def window_planck(x, xi, xf, deltaxi, deltaxf):
    di = deltaxi/(20*np.log(10))
    df = deltaxf/(20*np.log(10))
    if x <= xi + di:
        return 0
    elif xi +di < x < xi + deltaxi - di:
        return 1./(1 + np.exp(deltaxi/(x - xi) + deltaxi/(x - (xi + deltaxi))))
    elif xi + deltaxi - di <= x <= xf - deltaxf + df:
        return 1
    elif xf - deltaxf + df < x < xf - df:
        return 1./(1 + np.exp(-(deltaxf/(x - (xf - deltaxf))) - deltaxf/(x - xf)))
    else:
        return 0
def window_planck_left(x, xf, deltaxf):
    df = deltaxf/(20*np.log(10))
    if x <= xf - deltaxf + df:
        return 1
    elif xf - deltaxf + df < x < xf - df:
        return 1./(1 + np.exp(-deltaxf/(x - (xf - deltaxf)) - deltaxf/(x - xf)))
    else:
        return 0
def window_planck_right(x, xi, deltaxi):
    di = deltaxi/(20*np.log(10))
    if x <= xi + di:
        return 0
    elif xi + di < x < xi + deltaxi - di:
        return 1./(1 + np.exp(deltaxi/(x - xi) + deltaxi/(x - (xi + deltaxi))))
    else:
        return 1

# The FFT function - accepts real or complex (real+real) input, discards negative frequencies
def fft_positivef(timeseries):
    n = len(timeseries)
    ncol = timeseries.shape[1] - 1
    deltat = timeseries[1,0] - timeseries[0,0]
    deltaf = 1./(n*deltat)
    #Fast Fourier Transform
    frequencies = deltaf*np.arange(n)
    # Input values for the fft - accomodate for the real and complex cases
    if ncol==1:
        vals = timeseries[:,1]
    elif ncol==2:
        vals = timeseries[:,1] + 1j*timeseries[:,2]
    else:
        raise Exception('Incorrect number of columns in array.')
    #BEWARE: due to the different convention for the sign of Fourier frequencies, we have to reverse the FFT output
    #Beware also that the FFT treats effectively the initial time as 0
    #BEWARE: in the reversion of the numpy-convention FFT output, we have to set aside the 0-frequency term
    numpyfftvalues = deltat * np.fft.fft(vals)
    fouriervals = np.concatenate((numpyfftvalues[:1], numpyfftvalues[1:][::-1]))
    #Discarding information on negative frequencies - if real timeseries in input, no information loss
    fouriervals = fouriervals[:n//2]
    frequencies = frequencies[:n//2]
    #Coming back to the initial times
    tshift = timeseries[0,0]
    factorsshifttime = np.exp(1j*2*np.pi*frequencies*tshift)
    fouriervals = np.multiply(fouriervals, factorsshifttime)
    fourierseries = array([frequencies, np.real(fouriervals), np.imag(fouriervals)]).T
    return fourierseries

# Function for zero-padding a real array at the end
# Assumes a constant x-spacing - works with any number of columns
def zeropad(series, extend=0):
    n = len(series)
    ncol = series.shape[1]
    deltax = series[1,0] - series[0,0]
    xf = series[-1,0]
    nzeros = pow(2, extend + int(math.ceil(np.log(n)/np.log(2)))) - n
    res = np.zeros((n+nzeros, ncol), dtype=float)
    xextension = xf + deltax*np.arange(1, nzeros+1)
    res[:n, :] = series
    res[n:, 0] = xextension
    return res

#Function for Windowing, given the windows as arrays
# def taper_array(timeseries, window1, window2):
#     N = len(timeseries)
#     nbpointsw1 = len(window1)
#     nbpointsw2 = len(window2)
#     ones = np.empty(N - nbpointsw1 - nbpointsw2)
#     ones.fill(1.)
#     window = np.concatenate((window1, ones, window2))
#     return np.array([timeseries[:,0], np.multiply(timeseries[:,1], window)]).T

# Returns largest index i such that arr[i]<=xval, starting from a guess
def func_simplefinder_below(arr, xval, i0):
    n = len(arr)
    i = i0
    if xval>arr[n-1] or xval<arr[0]:
        raise Exception('Value outside of range')
    if i<n-1 and xval>=arr[i+1]:
        while i<n-1 and xval>=arr[i+1]:
            i += 1
    elif i>0 and xval<arr[i]:
        while i>0 and xval<arr[i]:
            i -= 1
    return i
# Returns smallest index i such that xval<=arr[i], starting from a guess
def func_simplefinder_above(arr, xval, i0):
    n = len(arr)
    i = i0
    if xval>arr[n-1] or xval<arr[0]:
        raise Exception('Value outside of range')
    if i<n-1 and xval>=arr[i+1]:
        while i<n-1 and xval>=arr[i]:
            i += 1
    elif i>0 and xval<arr[i-1]:
        while i>0 and xval<arr[i-1]:
            i -= 1
    return i
# Function for tapering a time/frequency series given starting and ending tapering lengths
# Admits any number of columns (2 for real time series in array form, 3 for complex series with real/imag)
# Does not assume equal sampling in time but uses it as a first guess to find at what index to taper
def taper_array(series, windowbeg1, windowbeg2, windowend1, windowend2):
    lbounds = [windowbeg1, windowbeg2, windowend1, windowend2]
    if not sorted(lbounds)==lbounds:
        raise Exception('Incompatible tapering bounds.')
    n = len(series)
    ncol = series.shape[1] - 1
    x = series[:,0]
    deltax = x[1] - x[0] # used to estimate indices - not required to be constant
    # Initial guesses
    iwindowbeg1 = min(max(0, np.int((windowbeg1 - x[0])/deltax)), n-1)
    iwindowbeg2 = min(max(0, np.int((windowbeg2 - x[0])/deltax)), n-1)
    iwindowend1 = min(max(0, np.int((windowend1 - x[0])/deltax)), n-1)
    iwindowend2 = min(max(0, np.int((windowend2 - x[0])/deltax)), n-1)
    # Adjustments
    iwindowbeg1 = func_simplefinder_below(x, windowbeg1, iwindowbeg1)
    iwindowbeg2 = func_simplefinder_below(x, windowbeg2, iwindowbeg2)
    iwindowend1 = func_simplefinder_above(x, windowend1, iwindowend1)
    iwindowend2 = func_simplefinder_above(x, windowend2, iwindowend2)
    wbeg = np.array(list(map(lambda x: window_planck_right(x, series[iwindowbeg1, 0], series[iwindowbeg2,0] - series[iwindowbeg1,0]), series[iwindowbeg1:iwindowbeg2, 0])))
    wend = np.array(list(map(lambda x: window_planck_left(x, series[iwindowend2, 0], series[iwindowend2,0] - series[iwindowend1,0]), series[iwindowend1:iwindowend2, 0])))

    # Compute windowed array
    res = np.zeros(series.shape, dtype=float)
    res[:, 0] = x
    for col in range(1, ncol+1):
        res[0:iwindowbeg1, col] = 0
        res[iwindowbeg1:iwindowbeg2, col] = wbeg * series[iwindowbeg1:iwindowbeg2, col]
        res[iwindowbeg2:iwindowend1, col] = series[iwindowbeg2:iwindowend1, col]
        res[iwindowend1:iwindowend2, col] = wend * series[iwindowend1:iwindowend2, col]
        res[iwindowend2:n, col] = 0
    return res

#Function for putting a Re/Im waveform in an Amp/Phase form
def amp_phase(wavedata):
    x = wavedata[:,0]
    h = wavedata[:,1] + 1j*wavedata[:,2]
    amplitudes = np.absolute(h)
    phases = np.unwrap(np.angle(h))
    return array([x, amplitudes, phases]).T
# Function to extract amplitude and phase of complex series
def ampseries(cseries):
    return np.absolute(cseries)
def phaseseries(cseries):
    return np.unwrap(np.angle(cseries))

# Find the time of peak using the sum of square of mode amplitudes - dict of interpolated amplitudes as input
def find_tpeak(tmin, tmax, AlmInt, listmodes):
    scalefactor = tmax-tmin
    def ftominimize(x):
        if (x*scalefactor > tmax) or (x*scalefactor < tmin):
            return +1e99
        else:
            return -1 * sum([AlmInt[lm](x*scalefactor)**2 for lm in listmodes])
    res = op.minimize(ftominimize, (tmin+tmax)/2./scalefactor, method='Nelder-Mead', options={'disp':False})
    return scalefactor * (res.x)

# Compute J from SEOB dynamics
def J_from_dynamics(dyn, index):
    # Format for dynamics: np array with 15 rows for t, x, p, S1, S2, phiDMod, phiMod
    # Use index=0 for Ji, index=-1 for Jf
    x = dyn[1:4][:,index]
    p = dyn[4:7][:,index]
    L = np.cross(x, p)
    S1 = dyn[7:10][:,index]
    S2 = dyn[10:13][:,index]
    return L + S1 + S2

# Fold a list : get n elements by cycling through the list
def fold_list(xlist, n):
    l = len(xlist)
    return (xlist * (n//l + 1))[:n]

################################################################################
# Functions to plot arrays
# Input format :
# ax axes object (for subplot use)
# arg data sequence [data1, [colx1, coly1]], ...
# kwargs options : domain (figsize now at the level of subplot)
# allow for two data formats, distinguished by wether the 2nd arg in the list is a numpy array or not :
# i) [nparray d, cols [i,j]] : datax = d[:,i], datay = d[:,j]
# ii) [nparray datax, nparray datay] directly
# typical usage for continuous color map: [cm, cmb] = ['inferno', [0.,0.9]] lplot(...,  colormap=cm, colormapbounds=cmb)
# Color palette stolen from seaborn package (deep, with reshuffled ordering)
# SEABORN_PALETTES = dict(
#     deep=["#4C72B0", "#55A868", "#C44E52",
#           "#8172B2", "#CCB974", "#64B5CD"],
#     muted=["#4878CF", "#6ACC65", "#D65F5F",
#            "#B47CC7", "#C4AD66", "#77BEDB"],
#     pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
#             "#D0BBFF", "#FFFEA3", "#B0E0E6"],
#     bright=["#003FFF", "#03ED3A", "#E8000B",
#             "#8A2BE2", "#FFC400", "#00D7FF"],
#     dark=["#001C7F", "#017517", "#8C0900",
#           "#7600A1", "#B8860B", "#006374"],
#     colorblind=["#0072B2", "#009E73", "#D55E00",
#                 "#CC79A7", "#F0E442", "#56B4E9"]
#)
rc_params = {'backend': 'ps',
            'font.family': 'Times New Roman',
            'font.sans-serif': ['Bitstream Vera Sans'],
            'axes.unicode_minus':False,
            'text.usetex':True,
            'grid.linestyle':':',
            'grid.linewidth':1.,
            'axes.labelsize':16,
            'axes.titlesize':16,
            'xtick.labelsize':16,
            'ytick.labelsize':16,
            'legend.fontsize':16,
            'figure.dpi':300}

plt.rcParams.update(rc_params)

plotpalette = ["#4C72B0", "#C44E52", "#CCB974", "#55A868", "#8172B2", "#64B5CD"]
def lplot(ax, *args, **kwargs):
    rangex = kwargs.pop('rangex', [])
    rangey = kwargs.pop('rangey', [])
    ds = kwargs.pop('downsample', 1)
    size = kwargs.pop('figsize', (8, 4))
    grid = kwargs.pop('grid', True)
    colormap = kwargs.pop('colormap', None)
    colormapbounds = kwargs.pop('colormapbounds', [0.,1.])
    colors = kwargs.pop('colors', None)
    linestyles = kwargs.pop('linestyles', None)
    linewidths = kwargs.pop('linewidths', None)
    log_xscale = kwargs.pop('log_xscale', False)
    log_yscale = kwargs.pop('log_yscale', False)
    n = len(args)
    if colors is None: # colors option supersedes colormap
        if colormap is not None:
            colorm = cm.get_cmap(colormap)
            colors = [colorm(x) for x in np.linspace(colormapbounds[0], colormapbounds[1], n)]
        else:
            #defaultcolorlist = plt.rcParams['axes.prop_cycle'].by_key()['color']
            defaultcolorlist = plotpalette
            colors = fold_list(defaultcolorlist, n)
    if linestyles is None:
        linestyles = ['-' for i in range(n)]
    if linewidths is None:
        linewidths = [1 for i in range(n)]
    f = plt.figure(0, figsize=size)
    minxvals = np.zeros(n)
    maxxvals = np.zeros(n)
    minyvals = np.zeros(n)
    maxyvals = np.zeros(n)
    avyvals = np.zeros(n)
    for i, x in enumerate(args):
        if type(x[1]) is np.ndarray:
            data = restrict_data_soft(np.array([x[0][::ds], x[1][::ds]]).T, rangex)
            col1, col2 = [0, 1]
        else:
            data = restrict_data_soft(x[0][::ds], rangex)
            col1, col2 = x[1]
        if not (log_xscale and log_yscale):
            minxvals[i] = data[0, col1]
        else: # Restrict to the first non-zero value of y - convenient for log-x plots (also always exclude x=0)
            datax = data[:, col1]
            datay = data[:, col2]
            if datax[0]==0.:
                datax = datax[1:]
                datay = datay[1:]
            minxvals[i] = datax[(datay > 0)][0]
        maxxvals[i] = data[-1, col1]
        minyvals[i] = min(data[:, col2])
        maxyvals[i] = max(data[:, col2])
        avyvals[i] = np.average(data[:, col2])
        ax.plot(data[:,col1], data[:,col2], color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i], **kwargs)
    if rangex:
        ax.set_xlim(rangex[0], rangex[1])
    else:
        ax.set_xlim(min(minxvals), max(maxxvals))
    if rangey:
        ax.set_ylim(rangey[0], rangey[1])
    else:
        if log_yscale:
            minyvalplot = max(min(minyvals), 1e-8*np.average(avyvals))
            ax.set_ylim(1./2*minyvalplot, 2*max(maxyvals))
        else:
            if max(maxyvals)==min(minyvals): # Collapsed case: plot a constant, scale is arbitrary, just plot +-1.
                ax.set_ylim(min(minyvals) - 1., max(maxyvals) + 1.)
            else:
                margin = 0.1 * (max(maxyvals) - np.average(avyvals))
                ax.set_ylim(min(minyvals) - margin, max(maxyvals) + margin)
    if log_xscale:
        ax.set_xscale('log')
    if log_yscale:
        ax.set_yscale('log')
    if grid:
        ax.grid('on', linestyle=':')
def llogplot(ax, *arg, **kwargs):
    args = (ax,) + arg
    return lplot(*args, log_yscale=True, **kwargs)
def lloglinearplot(ax, *arg, **kwargs):
    args = (ax,) + arg
    return lplot(*args, log_xscale=True, **kwargs)
def lloglogplot(ax, *arg, **kwargs):
    args = (ax,) + arg
    return lplot(*args, log_xscale=True, log_yscale=True, **kwargs)

################################################################################

# Sort array by column
# - for instance, sort posterior samples according to LogLikelihood value
def sortby(arr, col):
     return arr[np.argsort(-arr[:, col])]

# Antenna pattern functions for LIGO-type detectors
# cf (57) in the Living Review by Sathyaprakash & Schutz
# http://relativity.livingreviews.org/Articles/lrr-2009-2/download/lrr-2009-2Color.pdf
def Fplus(theta, phi, psi):
    return 1./2*(1+cos(theta)**2)*cos(2*phi)*cos(2*psi)-cos(theta)*sin(2*phi)*sin(2*psi)
def Fcross(theta, phi, psi):
    return 1./2*(1+cos(theta)**2)*cos(2*phi)*sin(2*psi)+cos(theta)*sin(2*phi)*cos(2*psi)
# Same, where input is cos(theta) and not theta -  used to improve performance when averaging over the sphere
def Fplus_costheta(c, phi, psi):
    return 1./2*(1+c**2)*cos(2*phi)*cos(2*psi)-c*sin(2*phi)*sin(2*psi)
def Fcross_costheta(c, phi, psi):
    return 1./2*(1+c**2)*cos(2*phi)*sin(2*psi)+c*sin(2*phi)*cos(2*psi)
# Mixing angle on the orthonormal basis (e1,e2), returns its [cos, sin]
# See appendix B of Babak&al_1607 https://arxiv.org/abs/1607.05661
def pattern_mixing_angle_cos_sin(hpnorm, hcnorm, hphcnorm, Fplus, Fcross):
    inversehnorm = 1./sqrt(Fplus**2*hpnorm**2 + 2*Fplus*Fcross*hpnorm*hcnorm + Fcross**2*hcnorm**2)
    return [inversehnorm * (Fplus*hpnorm + Fcross*hcnorm*hphcnorm), inversehnorm * Fcross*hcrossnorm*sqrt(1-hphcnorm**2)]

# Function to produce all modes (l,m) for l<=lmax in a flat list
def list_all_modes(lmax):
    return [(l,m) for l in range(2, lmax+1) for m in range(-l, l+1)]

# Spin-weighted spherical harmonics
# Only for spin weight s=-2
def sYlm(l, m, theta, phi):
    c = cos(0.5*theta); s = sin(0.5*theta);
    if s==0:
        if m==2:
            sdlm = 1
        else:
            sdlm = 0
    elif c==0:
        if m==-2:
            sdlm = (-1)**l
        else:
            sdlm = 0
    else:
        values = sqrt(ffactorial(l+m)*ffactorial(l-m)*ffactorial(l+2)*ffactorial(l-2)) * np.array([(-1)**k/ffactorial(k) / (ffactorial(k-m+2)*ffactorial(l+m-k)*ffactorial(l-k-2)) * c**(2*l+m-2*k-2) * s**(2*k-m+2) for k in np.arange(max(0, m-2), min(l+m, l-2)+1)]) # NOTE: float conversion necessary to avoid type problems with long when multiplying large factorials
        sdlm = np.sum(values)
    return sqrt((2*l+1) / (4*pi)) * sdlm * exp(1j * m * phi)
