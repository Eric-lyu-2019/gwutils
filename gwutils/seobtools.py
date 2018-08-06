## OUT OF DATE
##############

## Tools to generate and process SEOB waveforms
## Depends on LAL branch SEOBNRv3ROMdevel

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

import lal
import lalsimulation as lalsim

import gwtools
import nrtools

# Generate SEOBNRv3 waveform - syntax depends on branch SEOBNRv3ROMdevel
# Called with M in solar masses and f_min, fs in Hz - spin parameters give chi
def SEOBNRv3_All(M, eta, s1x, s1y, s1z, s2x, s2y, s2z, f_min=20, iota=0, fs=4096, fOut=0, flag_radiationframe=0, flag_outputframe=0):
    q = gwtools.qofeta(eta)
    m1 = M * q/(1.0+q)
    m2 = M * 1.0/(1.0+q)
    phiC = 0
    r = 1e6*lal.PC_SI
    deltaT = 1./fs

    #
    print phiC, deltaT, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_min, r, iota, s1x, s1y, s1z, s2x, s2y, s2z, fOut, flag_radiationframe, flag_outputframe

    return lalsim.SimIMRSpinEOBWaveformAll(phiC, deltaT, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_min, r, iota,
                            s1x, s1y, s1z, s2x, s2y, s2z, fOut, flag_radiationframe, flag_outputframe)

# Function to generate SEOBNRv3 waveform from LAL
# Outputs P-frame waveform, Euler angles and log quaternion
# Note - f_min, fs are Mfmin, Mfs
def genSEOBNRv3_Pframe(eta, s1x, s1y, s1z, s2x, s2y, s2z, Mfmin, Mfsample, flag_radiationframe=1, flag_outputframe=1, t0choice='tpeak', lambdabeg=0, adjustoutputframe=True):
    q = gwtools.qofeta(eta)

    # List of modes
    listmodes = [(2,2),(2,1),(2,0),(2,-1),(2,-2)]

    # Generate the SEOBNRv3 modes and Euler angles
    # Use default 20 Msol for M - Use 0 inclination for generation - will use the modes
    M = 20.
    f_min = Mfmin/(M*gwtools.msols)
    fs = Mfsample/(M*gwtools.msols)
    hp, hc, dynHi, hP, hPHi, hJHi, hI, AttachPars, dynamics, alphaDyn, betaDyn, gammaDyn, tE, Omega, tO, tA = SEOBNRv3_All(M, eta, s1x, s1y, s1z, s2x, s2y, s2z, flag_radiationframe=flag_radiationframe, flag_outputframe=flag_outputframe, f_min=f_min, iota=0., fs=fs)

    # Mode-by-mode output for the full waveform in Inertial Frame
    tI = np.arange(hI.mode.data.length) * hI.mode.deltaT/(M*lal.MTSUN_SI)
    hIlm = {}
    hIlm[(2,2)] = hI.next.next.next.next.mode.data.data
    hIlm[(2,1)] = hI.next.next.next.mode.data.data
    hIlm[(2,0)] = hI.next.next.mode.data.data
    hIlm[(2,-1)] = hI.next.mode.data.data
    hIlm[(2,-2)] = hI.mode.data.data

    # Rotating I-frame modes (that is, rotate the final-J frame) around e3J to enforce
    # the condition n(t=0)=x in the plane (e1J,e3J)
    # Final J in cartesian coordinates
    if adjustoutputframe:
        xf = dynamics[-1][0:3]
        pf = dynamics[-1][3:6]
        Lf = np.cross(xf, pf)
        S1f = dynamics[-1][6:9]
        S2f = dynamics[-1][9:12]
        Jf = Lf + S1f + S2f
        e3J = Jf / np.linalg.norm(Jf)
        xhat = np.array([1,0,0]); yhat = np.array([0,1,0]); zhat = np.array([0,0,1])
        deltaphi = np.arctan2(np.dot(e3J, yhat), np.dot(e3J, xhat) * np.dot(e3J, zhat))
        for lm in listmodes:
            hIlm[lm] = np.exp(-lm[1]*1j*deltaphi) * hIlm[lm]

    # Extract amplitudes and phases of the I-frame modes
    AI = {}; phiI = {};
    for lm in listmodes:
        AI[lm] = gwtools.ampseries(hIlm[lm])
        phiI[lm] = gwtools.phaseseries(hIlm[lm])

    # Build precessing-frame waveform - uses GWFrames
    T_data = tI
    LM_data = array(map(list, listmodes), dtype=np.int32)
    mode_data = np.array([hIlm[lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    W_v3.TransformToCoprecessingFrame();

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    logquat = np.array(map(lambda q: q.log(), quat))
    quatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), quat))
    logquatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), logquat))
    Vfseries = np.array(map(gwtools.Vf_from_quat, quat))
    eulerseries = np.array(map(gwtools.euler_from_quat, quat))

    # Mode-by-mode output for precessing-frame waveform
    iGWF = {}; APlm = {}; phiPlm = {};
    for lm in listmodes:
        iGWF[lm] = W_v3.FindModeIndex(lm[0], lm[1])
        APlm[lm] = W_v3.Abs(iGWF[lm])
        phiPlm[lm] = W_v3.ArgUnwrapped(iGWF[lm])
    tP = tI

    # Imposing t=0 either at tA or at tpeak
    t0 = 0
    if t0choice=='tpeak':
        APlmInt = {}
        for lm in listmodes:
            APlmInt[lm] = ip.InterpolatedUnivariateSpline(tP, APlm[lm], k=3)
        t0 = gwtools.find_tpeak(tI[-1]-1000, tI[-1], APlmInt, listmodes)
    elif t0choice=='tA':
        t0 = tA
    tI = tI - t0
    tP = tP - t0
    tE = tE - t0

    # Generate metadata for output
    metadata = {}
    metadata['M'] = M
    metadata['eta'] = eta
    metadata['q'] = q
    metadata['m1'] = q/(1.+q) * M
    metadata['m2'] = 1./(1.+q) * M
    metadata['chi1'] = np.array([s1x, s1y, s1z])
    metadata['chi2'] = np.array([s2x, s2y, s2z])
    metadata['fmin'] = f_min

    # Output
    wf = {}
    wf['metadata'] = metadata
    wf['listmodes'] = listmodes
    wf['tP'] = tP
    wf['APlm'] = APlm
    wf['phiPlm'] = phiPlm
    wf['euler'] = eulerseries
    wf['logquat'] = logquatseries
    return wf

# Function to generate SEOBNRv3 waveform from LAL
# Outputs I-frame and P-frame waveform, Euler angles and log quaternion, and dynamics
# Note: dynamics given in the initial L-frame - f_min, fs are Mfmin, Mfs
def genSEOBNRv3_all(eta, s1x, s1y, s1z, s2x, s2y, s2z, Mfmin, Mfsample, flag_radiationframe=1, flag_outputframe=1, t0choice='tpeak', lambdabeg=0, adjustoutputframe=True):
    q = gwtools.qofeta(eta)

    # List of modes
    listmodes = [(2,2),(2,1),(2,0),(2,-1),(2,-2)]

    # Generate the SEOBNRv3 modes and Euler angles
    # Use 0 inclination for generation - will use the modes
    M = 20.
    f_min = Mfmin/(M*gwtools.msols)
    fs = Mfsample/(M*gwtools.msols)
    hp, hc, dynHi, hP, hPHi, hJHi, hI, AttachPars, dynamics, alphaDyn, betaDyn, gammaDyn, tE, Omega, tO, tA = SEOBNRv3_All(M, eta, s1x, s1y, s1z, s2x, s2y, s2z, flag_radiationframe=flag_radiationframe, flag_outputframe=flag_outputframe, f_min=f_min, iota=0., fs=fs)

    # Mode-by-mode output for the full waveform in Inertial Frame
    tI = np.arange(hI.mode.data.length) * hI.mode.deltaT/(M*lal.MTSUN_SI)
    hIlm = {}
    hIlm[(2,2)] = hI.next.next.next.next.mode.data.data
    hIlm[(2,1)] = hI.next.next.next.mode.data.data
    hIlm[(2,0)] = hI.next.next.mode.data.data
    hIlm[(2,-1)] = hI.next.mode.data.data
    hIlm[(2,-2)] = hI.mode.data.data

    # Euler angles and dynamics - only extend to attachment point
    tE = (tE.data) # Note: already t/M
    alphaDyn = alphaDyn.data
    betaDyn = betaDyn.data
    gammaDyn = gammaDyn.data

    # Dynamics - convert to an array
    # Corresponding time vector is tE
    dynamics = dynamics.data
    npt = len(dynamics)/15
    dynamics = np.reshape(dynamics, (15, npt)).T

    # Rotating I-frame modes (that is, rotate the final-J frame) around e3J to enforce
    # the condition n(t=0)=x in the plane (e1J,e3J)
    # Final J in cartesian coordinates
    xf = dynamics[-1][0:3]
    pf = dynamics[-1][3:6]
    Lf = np.cross(xf, pf)
    S1f = dynamics[-1][6:9]
    S2f = dynamics[-1][9:12]
    Jf = Lf + S1f + S2f
    if adjustoutputframe:
        e3J = Jf / np.linalg.norm(Jf)
        xhat = np.array([1,0,0]); yhat = np.array([0,1,0]); zhat = np.array([0,0,1])
        deltaphi = np.arctan2(np.dot(e3J, yhat), np.dot(e3J, xhat) * np.dot(e3J, zhat))
        for lm in listmodes:
            hIlm[lm] = np.exp(-lm[1]*1j*deltaphi) * hIlm[lm]

    # Extract amplitudes and phases of the I-frame modes
    AIlm = {}; phiIlm = {};
    for lm in listmodes:
        AIlm[lm] = gwtools.ampseries(hIlm[lm])
        phiIlm[lm] = gwtools.phaseseries(hIlm[lm])

    # Build precessing-frame waveform - uses GWFrames
    T_data = tI
    LM_data = array(map(list, listmodes), dtype=np.int32)
    mode_data = np.array([hIlm[lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    W_v3.TransformToCoprecessingFrame();

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    logquat = np.array(map(lambda q: q.log(), quat))
    quatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), quat))
    logquatseries = np.array(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), logquat))
    Vfseries = np.array(map(gwtools.Vf_from_quat, quat))
    eulerseries = np.array(map(gwtools.euler_from_quat, quat))

    # Mode-by-mode output for precessing-frame waveform
    iGWF = {}; APlm = {}; phiPlm = {};
    for lm in listmodes:
        iGWF[lm] = W_v3.FindModeIndex(lm[0], lm[1])
        APlm[lm] = W_v3.Abs(iGWF[lm])
        phiPlm[lm] = W_v3.ArgUnwrapped(iGWF[lm])
    tP = tI

    # Imposing t=0 either at tA or at tpeak
    t0 = 0
    if t0choice=='tpeak':
        APlmInt = {}
        for lm in listmodes:
            APlmInt[lm] = ip.InterpolatedUnivariateSpline(tP, APlm[lm], k=3)
        t0 = gwtools.find_tpeak(tI[-1]-1000, tI[-1], APlmInt, listmodes)
    elif t0choice=='tA':
        t0 = tA
    tI = tI - t0
    tP = tP - t0
    tE = tE - t0

    # Generate metadata for output
    metadata = {}
    metadata['M'] = M
    metadata['eta'] = eta
    metadata['q'] = q
    metadata['m1'] = q/(1.+q) * M
    metadata['m2'] = 1./(1.+q) * M
    metadata['chi1'] = np.array([s1x, s1y, s1z])
    metadata['chi2'] = np.array([s2x, s2y, s2z])
    metadata['fmin'] = f_min
    metadata['chif'] = gwtools.norm(Jf) / M**2
    print metadata['chif']

    # Output
    wf = {}
    wf['metadata'] = metadata
    wf['listmodes'] = listmodes
    wf['tI'] = tI
    wf['AIlm'] = AIlm
    wf['phiIlm'] = phiIlm
    wf['tP'] = tP
    wf['APlm'] = APlm
    wf['phiPlm'] = phiPlm
    wf['euler'] = eulerseries
    wf['logquat'] = logquatseries
    wf['tD'] = tE # tE renamed tD - times vector for the dynamics
    wf['dynamics'] = dynamics # Note: dynamics output in (x,y,z) coordinates tied to the initial-L frame, which differs from the I-frame used for the waveform (if given )
    return wf

# Function to compare NR waveform to a SEOB waveform - parameters read from metadata
# Can take initial Omega for SEOB, to be determined externally by adjusting time to merger - if not given, just using the same as the NR value
# NOTE: by construction assumes that the scale M is 1 in NR -- but relaxed masses 1+2 is not exactly 1 ?
def genSEOBNRv3_compare_NR(wfnr, Omega0=0., listmodes=None, rotateJ=True, flag_radiationframe=1, flag_outputframe=1, verbose=True):
    # Use all l=2 modes by default
    if listmodes==None:
        listmodes = [(2,2),(2,1),(2,0),(2,-1),(2,-2)]
    # Get parameters
    metadata = wfnr['metadata']
    #M = metadata['initial-ADM-energy']
    M = metadata['m1'] + metadata['m2']
    eta = metadata['eta']
    q = gwtools.qofeta(eta)
    [s1x, s1y, s1z] = metadata['chi1']
    [s2x, s2y, s2z] = metadata['chi2']
    # Read sampling frequency from NR waveform
    deltatnr = wfnr['tI'][1] - wfnr['tI'][0]
    Mfsample = 1./deltatnr
    # If initial frequency fseob not given, take the frequency of NR at relaxed time
    if Omega0==0.:
        Omega0 = metadata['relaxed-orbital-frequency']
    Mfmin = Omega0 / pi
    # Generate and process SEOB waveform
    # test
    #return [M, eta, s1x, s1y, s1z, s2x, s2y, s2z, fs, fstart]
    wfseob = genSEOBNRv3_all(eta, s1x, s1y, s1z, s2x, s2y, s2z, Mfmin, Mfsample, flag_radiationframe=flag_radiationframe, flag_outputframe=flag_outputframe, t0choice='tpeak', lambdabeg=0, adjustoutputframe=True)
    # Show time to peak - both waveforms come with tpeak=0 by default
    tpeakseob = -wfseob['tP'][0]
    tpeaknr = -wfnr['tP'][0]
    if verbose:
        print("Time to peak (t/M): NR " + "%.6f" % tpeaknr + " | SEOB  " + "%.6f" % tpeakseob)
    return wfseob
