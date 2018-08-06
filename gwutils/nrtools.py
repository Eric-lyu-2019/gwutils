## Tools to load and process NR waveforms in the SXS format

import sys
import re
import time
import numpy as np
import copy
import math
import cmath
from math import pi, factorial
from numpy import array, conjugate, dot, sqrt, cos, sin, exp
import scipy
import scipy.interpolate as ip
import scipy.optimize as op
import matplotlib.pyplot as plt

import h5py
import glob
from natsort import realsorted

import GWFrames
import Quaternions

import gwtools
import seobtools
import gwplots


# Functions to load and parse metadata files
def metadata_extract_value(path, name, filename='metadata.txt'):
    pattern = re.compile(name)
    with open(path + filename, 'r') as f:
        i = 0
        for line in f:
            if re.search("#", line) is None and pattern.search(line):
                i += 1
                valuestr = line.split('=')[1]
                try:
                    res = float(valuestr.replace("<","")) # It can happen that eccentricity is given in the metadatafile as an upper limit, e.g. = <1e-4
                except ValueError:
                    res = np.nan # It can happen that eccentricity rhs is "[simulation too short]"
    if i==0:
        print 'Name ' + name + ' not found.'
    elif i>1:
        print 'Name ' + name + ' found more than once.'
    else:
        return res
def metadata_extract_vector(path, name, filename='metadata.txt'):
    pattern = re.compile(name)
    with open(path + filename, 'r') as f:
        i = 0
        for line in f:
            if re.search("#", line) is None and pattern.search(line):
                i += 1
                res = np.array( map( float, (line.split('=')[1]).split(',')))
    if i==0:
        print 'Name ' + name + ' not found.'
    elif i>1:
        print 'Name ' + name + ' found more than once.'
    else:
        return res
def metadata_extract_keywords(path, name, filename='metadata.txt'):
    pattern = re.compile(name)
    with open(path + filename, 'r') as f:
        i = 0
        for line in f:
            if re.search("#", line) is None and pattern.search(line):
                i += 1
                res = map( lambda x: x.strip(),(line.split('=')[1]).split(','))
    if i==0:
        print 'Name ' + name + ' not found.'
    elif i>1:
        print 'Name ' + name + ' found more than once.'
    else:
        return res
def metadata_extract(path, filename='metadata.txt'):
    metadata = {}
    for name in ['initial-separation', 'initial-orbital-frequency', 'initial-adot', 'initial-ADM-energy', 'initial-mass1', 'initial-mass2', 'relaxed-measurement-time', 'relaxed-mass1', 'relaxed-mass2', 'relaxed-eccentricity', 'common-horizon-time', 'number-of-orbits', 'remnant-mass']:
        metadata[name] = metadata_extract_value(path, name, filename)
    for name in ['initial-ADM-linear-momentum', 'initial-ADM-angular-momentum', 'initial-dimensionless-spin1', 'initial-dimensionless-spin2', 'initial-position1', 'initial-position2', 'relaxed-dimensionless-spin1', 'relaxed-dimensionless-spin2', 'relaxed-position1', 'relaxed-position2', 'relaxed-orbital-frequency', 'remnant-dimensionless-spin', 'remnant-velocity']:
        metadata[name] = metadata_extract_vector(path, name, filename)
    metadata['keywords'] = metadata_extract_keywords(path, 'keywords', filename)
    # Derived and renamed parameters - note that there are some redundancies
    m1 = metadata['relaxed-mass1']
    m2 = metadata['relaxed-mass2']
    mf = metadata['remnant-mass']
    chi1 = metadata['relaxed-dimensionless-spin1']
    chi2 = metadata['relaxed-dimensionless-spin2']
    chif = metadata['remnant-dimensionless-spin']
    q = m1 / m2
    eta = gwtools.etaofq(q)
    # chi1 = S1 / m1**2
    # chi2 = S2 / m2**2
    # chif = np.linalg.norm(Sf) / mf**2
    metadata['m1'] = m1
    metadata['m2'] = m2
    metadata['mf'] = mf
    metadata['q'] = q
    metadata['eta'] = eta
    metadata['chi1'] = chi1
    metadata['chi2'] = chi2
    metadata['chif'] = chif
    return metadata
def metadata_extract_dimfullspin(path, filename='metadata.txt'):
    metadata = {}
    for name in ['initial-separation', 'initial-orbital-frequency', 'initial-adot', 'initial-ADM-energy', 'initial-mass1', 'initial-mass2', 'relaxed-measurement-time', 'relaxed-mass1', 'relaxed-mass2', 'relaxed-eccentricity', 'common-horizon-time', 'number-of-orbits', 'remnant-mass']:
        metadata[name] = metadata_extract_value(path, name, filename)
    for name in ['initial-ADM-linear-momentum', 'initial-ADM-angular-momentum', 'initial-spin1', 'initial-spin2', 'initial-position1', 'initial-position2', 'relaxed-spin1', 'relaxed-spin2', 'relaxed-position1', 'relaxed-position2', 'relaxed-orbital-frequency', 'remnant-spin', 'remnant-velocity']:
        metadata[name] = metadata_extract_vector(path, name, filename)
    metadata['keywords'] = metadata_extract_keywords(path, 'keywords', filename)
    # Derived and renamed parameters - note that there are some redundancies
    m1 = metadata['relaxed-mass1']
    m2 = metadata['relaxed-mass2']
    mf = metadata['remnant-mass']
    S1 = metadata['relaxed-spin1']
    S2 = metadata['relaxed-spin2']
    Sf = metadata['remnant-spin']
    q = m1 / m2
    eta = gwtools.etaofq(q)
    chi1 = S1 / m1**2
    chi2 = S2 / m2**2
    chif = Sf / mf**2
    metadata['m1'] = m1
    metadata['m2'] = m2
    metadata['mf'] = mf
    metadata['q'] = q
    metadata['eta'] = eta
    metadata['chi1'] = chi1
    metadata['chi2'] = chi2
    metadata['chif'] = chif
    return metadata

# Function to load raw NR waveform from file
def load_NR_raw_waveform(path, filename='rhOverM_Asymptotic_GeometricUnits.h5', listmodes=[(2,2), (2,1), (2,0), (2,-1), (2,-2)], extrapolation='N4', metadata=False, metadata_dimfullspin=False, metadataname='metadata.txt', dynamics=False, dynamicsname='Horizons.h5'):
    wf = {}
    wf['listmodes'] = listmodes
    wf['extrapolation'] = extrapolation
    wf['hlm'] = {}
    with h5py.File(path + filename, 'r') as f:
        for lm in listmodes:
            wf['hlm'][lm] = np.array(f['Extrapolated_' + extrapolation + '.dir/Y_l' + str(lm[0]) + '_m' + str(lm[1]) + '.dat'])
    if metadata:
        if metadata_dimfullspin:
            wf['metadata'] = metadata_extract_dimfullspin(path, metadataname)
        else:
            wf['metadata'] = metadata_extract(path, metadataname)
    if dynamics:
        wf['dynamics'] = {}
        with h5py.File(path + dynamicsname, 'r') as f:
            wf['dynamics']['posA'] = np.array(f['/AhA.dir']['CoordCenterInertial.dat'][:])
            wf['dynamics']['posB'] = np.array(f['/AhB.dir']['CoordCenterInertial.dat'][:])
            wf['dynamics']['spinA'] = np.array(f['/AhA.dir']['DimensionfulInertialSpin.dat'][:])
            wf['dynamics']['spinB'] = np.array(f['/AhB.dir']['DimensionfulInertialSpin.dat'][:])
    return wf

# Function to cut junk, resample and shift times to have tpeak=0
# If tcutrelaxed, cut the waveform at relaxed time from metadat - otherwise use tcutbeg
# The dynamics is resampled as well, on the subset of times where it is defined
def resample_NR_waveform(wfI, listmodes=None, tcutrelaxed=True, tcutbeg=150., tcutend=0., deltat=0.2, set_phi22_at_t=False, phi22_at_t=None, t_def_phi22=None):

    # Get list of modes
    if listmodes==None:
        listmodes = wfI['listmodes']

    # Interpolating function for amplitudes and phases
    AIrawInt = {}; phiIrawInt = {}
    for lm in listmodes:
        AIrawInt[lm] = ip.InterpolatedUnivariateSpline(wfI['hlm'][lm][:,0], gwtools.ampseries(wfI['hlm'][lm][:,1] + 1j*wfI['hlm'][lm][:,2]), k=3)
        phiIrawInt[lm] = ip.InterpolatedUnivariateSpline(wfI['hlm'][lm][:,0], gwtools.phaseseries(wfI['hlm'][lm][:,1] + 1j*wfI['hlm'][lm][:,2]), k=3)

    # Find time of peak - use all the modes given, default window of 2000M at the end of the waveform
    tpeak = gwtools.find_tpeak(wfI['hlm'][(2,2)][-1,0]-2000, wfI['hlm'][(2,2)][-1,0], AIrawInt, listmodes)

    # Common time grid - with junk cutted
    # If tcutrelaxed, cut the waveform at relaxed time from metadata - otherwise use tcutbeg
    # 150M should be enough to cut junk at the beginning - can be changed as an option
    # NOTE: assumes all modes (and interpolating functions) have been defined on the same domain, up to tend
    # NOTE: 'relaxed-measurement-time' is absolute, not relative to the beginning of the waveform (the waveform typically starts at t=-100M, and t=0 is as close as possible to t=0 in the dynamics time (mapping non-obvious))
    if tcutrelaxed:
        tbeg = wfI['metadata']['relaxed-measurement-time']
    else:
        tbeg = wfI['hlm'][(2,2)][0,0] + tcutbeg
    tend = wfI['hlm'][(2,2)][-1,0] - tcutend
    tI = np.arange(tbeg, tend, deltat)

    # Reinterpolate modes
    AI = {}; phiI = {}; hI = {};
    for lm in listmodes:
        AI[lm] = AIrawInt[lm](tI)
        phiI[lm] = phiIrawInt[lm](tI)
        hI[lm] = AI[lm] * np.array(map(lambda x: exp(1j * x), phiI[lm]))

    # Shift time axis to put the peak at t=0
    tIshifted = tI - tpeak

    # Azimuthal shift in phase (in the I-frame), based on the inertial-frame 22 mode - mainly used for spin-aligned waveform
    phi22_shift = 0.
    if set_phi22_at_t:
        if (t_def_phi22 is None) or (phi22_at_t is None):
            raise ValueError('Must specify t_def_phi22 and phi22_at_t.')
        if t_def_phi22=='tstart':
            phi22_shift = phi22_at_t - phiIrawInt[(2,2)](tI[0])
        elif t_def_phi22=='tpeak':
            phi22_shift = phi22_at_t - phiIrawInt[(2,2)](tpeak)
        elif isinstance(t_def_phi22, float):
            if (tIshifted[0]<=t_def_phi22 and t_def_phi22<=tIshifted[-1]):
                phi22_shift = phi22_at_t - phiIrawInt[(2,2)](tpeak + t_def_phi22)
        else:
            raise ValueError('t_def_phi22 %s not recognized.' % t_def_phi22)
        for lm in listmodes:
            m = lm[1]
            phiI[lm] = phiI[lm] + m/2. * phi22_shift

    # Interpolating and resampling trajectory
    if 'dynamics' in wfI:
        posAInt = {}; posBInt = {}; spinAInt = {}; spinBInt = {};
        for i in [1,2,3]:
            posAInt[i] = ip.InterpolatedUnivariateSpline(wfI['dynamics']['posA'][:,0], wfI['dynamics']['posA'][:,i], k=3)
            posBInt[i] = ip.InterpolatedUnivariateSpline(wfI['dynamics']['posB'][:,0], wfI['dynamics']['posB'][:,i], k=3)
            spinAInt[i] = ip.InterpolatedUnivariateSpline(wfI['dynamics']['spinA'][:,0], wfI['dynamics']['spinA'][:,i], k=3)
            spinBInt[i] = ip.InterpolatedUnivariateSpline(wfI['dynamics']['spinB'][:,0], wfI['dynamics']['spinB'][:,i], k=3)
        tbegdyn = wfI['dynamics']['posA'][0,0]
        tenddyn = wfI['dynamics']['posA'][-1,0]
        tdyn = tI[np.logical_and((tI>=tbegdyn), (tI<=tenddyn))]
        posA = np.array([posAInt[i](tdyn) for i in [1,2,3]]).T
        posB = np.array([posBInt[i](tdyn) for i in [1,2,3]]).T
        velA = np.array([posAInt[i](tdyn, 1) for i in [1,2,3]]).T
        velB = np.array([posBInt[i](tdyn, 1) for i in [1,2,3]]).T
        spinA = np.array([spinAInt[i](tdyn) for i in [1,2,3]]).T
        spinB = np.array([spinBInt[i](tdyn) for i in [1,2,3]]).T
        tdynshift = tdyn - tpeak

    # Output waveform
    wf = {}
    wf['listmodes'] = listmodes
    wf['extrapolation'] = wfI['extrapolation']
    wf['tI'] = tIshifted
    wf['AI'] = AI
    wf['phiI'] = phiI
    wf['hI'] = hI
    if 'metadata' in wfI:
        wf['metadata'] = wfI['metadata']
    if 'dynamics' in wfI:
        wf['dynamics'] = {}
        wf['dynamics']['tdyn'] = tdynshift
        wf['dynamics']['posA'] = posA
        wf['dynamics']['posB'] = posB
        wf['dynamics']['velA'] = velA
        wf['dynamics']['velB'] = velB
        wf['dynamics']['spinA'] = spinA
        wf['dynamics']['spinB'] = spinB
    return wf

# Function to rotate waveform and dynamics to the frame (x,y,z)=(n,LNhat*n,LNhat) as read from the dynamics at the specified time
def initialrotation_NR_waveform(wfI, listmodes=None, trotate=None):

    # Get list of modes
    if listmodes==None:
        listmodes = wfI['listmodes']

    # Check dynamics has been loaded
    if not 'dynamics' in wfI:
        raise ValueError('Input waveform data does not have a dynamics key.')
    tdyn = wfI['dynamics']['tdyn']

    # time to set the (x,y,z)
    if trotate is None:
        trotate = wfI['tI'][0]

    # Check trotate is covered by dynamics data
    if ((trotate < wfI['dynamics']['tdyn'][0]) or (trotate > wfI['dynamics']['tdyn'][-1])):
        raise ValueError('Time for initial rotation %g is not covered by dynamics data on [%g, %g].' % (trotate, wfI['dynamics']['tdyn'][0], wfI['dynamics']['tdyn'][-1]))

    # Compute x, v, LN in the original frame
    posAInt = {}; posBInt = {}; velAInt = {}; velBInt = {}; spinAInt = {}; spinBInt = {};
    for i in [1,2,3]:
        posAInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['posA'][:,i-1], k=3)
        posBInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['posB'][:,i-1], k=3)
        velAInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['velA'][:,i-1], k=3)
        velBInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['velB'][:,i-1], k=3)
        spinAInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['spinA'][:,i-1], k=3)
        spinBInt[i] = ip.InterpolatedUnivariateSpline(tdyn, wfI['dynamics']['spinB'][:,i-1], k=3)
    x = np.array([posAInt[i](trotate) - posBInt[i](trotate) for i in [1,2,3]])
    v = np.array([velAInt[i](trotate) - velBInt[i](trotate) for i in [1,2,3]])
    LN = np.cross(x, v)
    n = gwtools.normalize(x)
    LNhat = gwtools.normalize(LN)
    lambd = gwtools.normalize(np.cross(LNhat, n)) # normalization here normally redundant, eliminates round-off

    # Rotation matrix and associated Euler angles
    # R active rotation matrix from the original frame to the new frame
    R = np.array([n, lambd, LNhat]).T
    alpha, beta, gamma = gwtools.euler_from_rotmatrix(R)

    # Rotate waveform modes, other fields are deepcopied
    wfrot = gwtools.rotate_wf_euler(wfI, alpha, beta, gamma)

    # Rotate dynamics
    def rotate(R, dyndata):
        Rinv = R.T
        res = np.zeros(dyndata.shape)
        for i in range(len(res)):
            res[i,:] = np.dot(Rinv, dyndata[i,:])
        return res
    posA = rotate(R, wfI['dynamics']['posA'])
    posB = rotate(R, wfI['dynamics']['posB'])
    velA = rotate(R, wfI['dynamics']['velA'])
    velB = rotate(R, wfI['dynamics']['velB'])
    spinA = rotate(R, wfI['dynamics']['spinA'])
    spinB = rotate(R, wfI['dynamics']['spinB'])

    # Rotate chi1, chi2, chif to be expressed in the relaxed frame
    chi1 = np.dot(R.T, wfI['metadata']['chi1'])
    chi2 = np.dot(R.T, wfI['metadata']['chi2'])
    chif = np.dot(R.T, wfI['metadata']['chif'])
    mf = wfI['metadata']['mf']

    # Output
    wf = {}
    wf['listmodes'] = listmodes
    wf['extrapolation'] = wfI['extrapolation']
    wf['tI'] = wfrot['tI']
    wf['AI'] = wfrot['AI']
    wf['phiI'] = wfrot['phiI']
    wf['hI'] = wfrot['hI']
    # Note: all vector quantities in metadata are still decomposed in the initial frame
    if 'metadata' in wfI:
        wf['metadata'] = wfI['metadata']
    wf['mf'] = mf
    wf['chi1'] = chi1 # decomposed in the relaxed frame
    wf['chi2'] = chi2 # decomposed in the relaxed frame
    wf['chif'] = chif # decomposed in the relaxed frame
    wf['dynamics'] = {}
    wf['dynamics']['tdyn'] = tdyn
    wf['dynamics']['posA'] = posA
    wf['dynamics']['posB'] = posB
    wf['dynamics']['velA'] = velA
    wf['dynamics']['velB'] = velB
    wf['dynamics']['spinA'] = spinA
    wf['dynamics']['spinB'] = spinB

    return wf

# Function to extract the precessing frame  and the P-frame modes - uses GWFrame
# Two choices for the precessing frame vector : DominantEigenvector for the dominant eigenvector of the LL matrix, 'AngularVelocityVector' for the frame angular velocity vector, omega = - LL . Ldt
# metadata used to read final J
def process_NR_waveform(wfI, listmodes=None, Pframe='DominantEigenvector', rotateJ=True, Jframechoice='e1JalongJcrossLN'):

    # List of modes and metadata
    if listmodes==None:
        listmodes = wfI['listmodes']
    metadata = wfI['metadata']

    # Rotate in place the waveform to the final-J-frame
    # Beware: metadata vectors are in the initial frame, not the relaxed frame, chif has been rotated to the relaxed frame
    if rotateJ:
        finalJhat = gwtools.normalize(wfI['chif'])
        wfI = rotate_NR_waveform_Jframe(wfI, finalJhat, listmodes=listmodes, Jframechoice=Jframechoice)

    # Build precessing-frame waveform - uses GWFrames
    T_data = wfI['tI']
    LM_data = array(map(list, listmodes), dtype=np.int32)
    mode_data = np.array([wfI['hI'][lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1)
    W_v3.SetDataType(1)
    if Pframe=='DominantEigenvector':
        W_v3.TransformToCoprecessingFrame()
    elif Pframe=='AngularVelocityVector':
        W_v3.TransformToAngularVelocityFrame()
    else:
        raise 'P-frame choice not recognized.'

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    logquat = np.array(map(lambda q: q.log(), quat))
    quatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), quat))
    logquatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), logquat))
    Vfseries = np.array(map(gwtools.Vf_from_quat, quat))
    eulerVfseries = np.array(map(gwtools.euler_from_quat, quat))
    alphaVfseries = np.unwrap(eulerVfseries[:,0])
    betaVfseries = eulerVfseries[:,1]
    gammaVfseries = np.unwrap(eulerVfseries[:,2])
    eulerVfseries = np.array([alphaVfseries, betaVfseries, gammaVfseries]).T

    # Mode-by-mode output for precessing-frame waveform - here tP is the same as tI
    iGWF = {}; AP = {}; phiP = {};
    for lm in listmodes:
        iGWF[lm] = W_v3.FindModeIndex(lm[0], lm[1])
        AP[lm] = W_v3.Abs(iGWF[lm])
        phiP[lm] = W_v3.ArgUnwrapped(iGWF[lm])
    tP = wfI['tI']

    # Output waveform
    wf = {}
    wf['metadata'] = metadata
    wf['listmodes'] = listmodes
    wf['extrapolation'] = wfI['extrapolation']
    wf['tI'] = wfI['tI']
    wf['AI'] = wfI['AI']
    wf['phiI'] = wfI['phiI']
    wf['hI'] = wfI['hI']
    wf['tP'] = tP
    wf['AP'] = AP
    wf['phiP'] = phiP
    wf['Vf'] = Vfseries
    wf['alpha'] = eulerVfseries[:,0]
    wf['beta'] = eulerVfseries[:,1]
    wf['gamma'] = eulerVfseries[:,2]
    wf['quat'] = quatseries
    wf['logquat'] = logquatseries
    if 'dynamics' in wfI:
        wf['dynamics'] = copy.deepcopy(wfI['dynamics'])
    return wf

# Note : we do not modify the input waveform but copy the dictionary that represents it
# Note : convention for the J-frame is e1J along J * z
def rotate_NR_waveform_Jframe(wf, Jhat, listmodes=None, Jframechoice='e1JalongJcrossLN'):
    # List of modes
    if listmodes==None:
        listmodes = wf['listmodes']
    # Frame vectors
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    e3J = Jhat
    if Jframechoice=='e1JalongJcrossLN':
        e1J = gwtools.normalize(np.cross(Jhat, e3)) # Convention e1J along J * z
        e2J = np.cross(e3J, e1J)
    elif Jframechoice=='ninplanee1Je3J':
        e1J = gwtools.normalize(e1 - np.dot(Jhat, e1)*Jhat) # Convention x in plane (e1J, e3J)
        e2J = np.cross(e3J, e1J)
    # print e1J
    # print e2J
    # print e3J
    # Euler angles for the active rotation from the J frame to the xyz frame
    alpha = np.arctan2(np.dot(e2J, e3), np.dot(e1J, e3))
    beta = np.arccos(np.dot(e3J, e3))
    gamma = np.arctan2(np.dot(e3J, e2), -np.dot(e3J, e1))
    # Store conversion factors
    conversionfactorItoJlmmp = {}
    for lm in listmodes:
        conversionfactorItoJlmmp[lm] = np.array([gwtools.WignerDMatrixstar(lm[0], mp, lm[1], alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    hJ = {}
    for lm in listmodes:
        hJ[lm] = np.sum(np.array([conversionfactorItoJlmmp[lm][lm[0]+mp]*wf['hI'][(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    wfres = copy.deepcopy(wf)
    for lm in listmodes:
        wfres['hI'][lm] = hJ[lm]
    for lm in listmodes:
        wfres['AI'][lm] = gwtools.ampseries(wfres['hI'][lm])
        wfres['phiI'][lm] = gwtools.phaseseries(wfres['hI'][lm])
    return wfres
