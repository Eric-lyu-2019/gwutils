# Note: requires patches on top of SEOBNRv4P branch of lalsuite_ossokine:
#0001-SEOBNRv3-expand-AttachParams-return-both-dynamics.patch
#0002-SEOB-aligned-return-both-dynamics.patch

from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


import numpy as np
import copy
import math
import cmath
from math import pi, factorial
from numpy import array, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.optimize as op

import lal
import lalsimulation as lalsim

import GWFrames
import Quaternions
import gwutils.gwtools as gwtools


def get_PrecBBHid_integer(wfid):
    listdigits = [s for s in list(wfid) if s.isdigit()]
    id_integer = int(''.join(listdigits))
    return id_integer
def get_SXSid_integer(wfid):
    listdigits = [s for s in list(wfid) if s.isdigit()]
    id_integer = int(''.join(listdigits))
    return id_integer

def initialphase_mod2pi(phase):
    shift = gwtools.mod2pi(phase[0]) - phase[0]
    return phase + shift
def funcchieff(q, chi1, chi2): #Note: corresponds to the conserved chieff
    eta = gwtools.etafun(q)
    return q/(1+q)*chi1 + 1/(1+q)*chi2
def funcchiP(q, chi1perp, chi2perp): #Note: corresponds to chiP from the LAL code
    eta = gwtools.etafun(q)
    A1 = 2+3./(2*q)
    A2 = 2+3.*q/2
    m1 = q/(1.+q)
    m2 = 1/(1.+q)
    return max(A1*m1**2*chi1perp, A2*m2**2*chi2perp)/(A1*m1**2)

#-------------------------------------------------------------------------------
# Functions to extract orbital frequency
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Extract omega orbital from SEOB dynamics
# Note: we do not use the fields of the extended dynamics as this is to be applied to dynamics returned by both v3 and v4P
def func_omegaorb_from_dyn_seob(dyn):
    t = dyn[0]
    x, y, z = dyn[1], dyn[2], dyn[3]
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    vx = spline(t, x)(t, 1)
    vy = spline(t, y)(t, 1)
    vz = spline(t, z)(t, 1)
    pos = np.array([x, y, z]).T
    vel = np.array([vx, vy, vz]).T
    n = len(t)
    omega_orb = np.zeros(n)
    for i in xrange(n):
        omega_orb[i] = np.linalg.norm(np.cross(pos[i], vel[i])) / (r[i]**2)
    return omega_orb
# Extract omega orbital from SEOB dynamics in the spin-alignedspin-aligned case
# Dynamics format: t, r, phi, pr, pphi
def func_omegaorb_from_dyn_seob_aligned(dyn_aligned):
    t = dyn_aligned[0]
    phi = dyn_aligned[2]
    omega_orb = spline(t, phi)(t, 1)
    return omega_orb

#-------------------------------------------------------------------------------
# Extract omega orbital from NR dynamics
def func_omegaorb_from_dyn_nr(dyn_nr):
    t = dyn_nr['tdyn']
    posAB = dyn_nr['posA'] - dyn_nr['posB']
    velAB = dyn_nr['velA'] - dyn_nr['velB']
    r = np.sqrt(np.power(posAB[:,0], 2) + np.power(posAB[:,1], 2) + np.power(posAB[:,2], 2))
    n = len(t)
    omega_orb = np.zeros(n)
    for i in xrange(n):
        omega_orb[i] = np.linalg.norm(np.cross(posAB[i], velAB[i])) / (r[i]**2)
    return omega_orb

#-------------------------------------------------------------------------------
# Find the peak in omega
def find_omegapeak(t, omega, tmin=None, tmax=None):
    if tmin is None:
        tmin = t[0]
    if tmax is None:
        tmax = t[-1]
    omega_Int = spline(t, omega)
    scalefactor = tmax-tmin
    def ftominimize(x):
        if (x*scalefactor > tmax) or (x*scalefactor < tmin):
            return +1e99
        else:
            return -1 * omega_Int(x*scalefactor)
    res = op.minimize(ftominimize, (tmin+tmax)/2./scalefactor, method='Nelder-Mead', options={'disp':False})
    return [scalefactor * (res.x[0]), -res.fun]

#-------------------------------------------------------------------------------
# Utilities for lists of modes (l,m)
def func_lvalues(listmodes):
    lvalues = []
    for lm in listmodes:
        if not lm[0] in lvalues:
            lvalues += [lm[0]]
    return lvalues
def func_allmodes_lvalues(lvalues):
    modes = []
    for l in lvalues:
        modes += [(l, l-m) for m in range(2*l+1)]
    return modes

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Waveform generation wrappers
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Note: default Mf_sampling corresponds to 4096Hz, 50Msol
default_Mf_sampling = 50 * 4096. * lal.MTSUN_SI
default_Mf_sampling

#-------------------------------------------------------------------------------
# Spin-aligned 22-mode SEOB waveforms
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# SEOBNRv2
# Note: at the moment, unable to return the dynamics
def gen_SEOBNRv2(q, chi1z, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling):

    # Arbitrary mass, distance
    M = 50. # in solar masses
    dist = 1. # in Mpc

    # Masses in solar masses
    m1 = M * q/(1.0+q)
    m2 = M * 1.0/(1.0+q)

    # SI units
    distSI = dist * 1e6 * lal.PC_SI
    m1SI = m1 * lal.MSUN_SI
    m2SI = m2 * lal.MSUN_SI
    f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
    deltaT = 1./f_sampling
    f_min = Mf_min / (M*lal.MTSUN_SI)

    # Prefactors to adimension h and t
    prefactor_h = distSI/(M*lal.MRSUN_SI) # the polarizations hp, hc are in physical strain units
    prefactor_t = 1./(M*lal.MTSUN_SI)

    # Pointers to REAL8Vectors required, will be returned emtpy
    # TODO: how to get the dynamics to be output ?
    t_dyn = lal.CreateREAL8Vector(0)
    r_dyn = lal.CreateREAL8Vector(0)
    phi_dyn = lal.CreateREAL8Vector(0)
    pr_dyn = lal.CreateREAL8Vector(0)
    pphi_dyn = lal.CreateREAL8Vector(0)

    # Generate SEOBNR, collect data pieces
    SEOBversion = 2
    hp, hc, dyn, dynHi = lalsim.SimIMRSpinAlignedEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, 0., chi1z, chi2z, SEOBversion, 0., 0., 0., 0., 0., 0., 0., 0., None, 0)

    # Output dictionary - same set of keys for v2, v4
    wf = {}
    wf['hp'] = hp
    wf['hc'] = hc
    wf['dynamics'] = dyn # requires patch
    wf['dynamicsHi'] = dynHi # requires patch
    # We save also the prefactors for adimensioned time and strain
    wf['prefact_t'] = prefactor_t
    wf['prefact_h'] = prefactor_h

    return wf

#-------------------------------------------------------------------------------
# SEOBNRv4
# Note: at the moment, unable to return the dynamics
def gen_SEOBNRv4(q, chi1z, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling):

    # Arbitrary mass, distance
    M = 50. # in solar masses
    dist = 1. # in Mpc

    # Masses in solar masses
    m1 = M * q/(1.0+q)
    m2 = M * 1.0/(1.0+q)

    # SI units
    distSI = dist * 1e6 * lal.PC_SI
    m1SI = m1 * lal.MSUN_SI
    m2SI = m2 * lal.MSUN_SI
    f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
    deltaT = 1./f_sampling
    f_min = Mf_min / (M*lal.MTSUN_SI)

    # Prefactors to adimension h and t
    prefactor_h = distSI/(M*lal.MRSUN_SI) # the polarizations hp, hc are in physical strain units
    prefactor_t = 1./(M*lal.MTSUN_SI)

    # Pointers to REAL8Vectors required, will be returned emtpy
    # TODO: how to get the dynamics to be output ?
    # t_dyn = lal.CreateREAL8Vector(0)
    # r_dyn = lal.CreateREAL8Vector(0)
    # phi_dyn = lal.CreateREAL8Vector(0)
    # pr_dyn = lal.CreateREAL8Vector(0)
    # pphi_dyn = lal.CreateREAL8Vector(0)

    # Generate SEOBNR, collect data pieces
    SEOBversion = 4
    hp, hc, dyn, dynHi = lalsim.SimIMRSpinAlignedEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, 0., chi1z, chi2z, SEOBversion, 0., 0., 0., 0., 0., 0., 0., 0., None, 0)

    # Output dictionary - same set of keys for v2, v4
    wf = {}
    wf['hp'] = hp
    wf['hc'] = hc
    wf['dynamics'] = dyn # requires patch
    wf['dynamicsHi'] = dynHi # requires patch
    # We save also the prefactors for adimensioned time and strain
    wf['prefact_t'] = prefactor_t
    wf['prefact_h'] = prefactor_h

    return wf

#-------------------------------------------------------------------------------
# Precessing SEOB waveforms
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# SEOBNRv3
def gen_SEOBNRv3(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling):

    # Arbitrary mass, distance
    M = 50. # in solar masses
    dist = 1. # in Mpc

    # Masses in solar masses
    m1 = M * q/(1.0+q)
    m2 = M * 1.0/(1.0+q)

    # SI units
    distSI = dist * 1e6 * lal.PC_SI
    m1SI = m1 * lal.MSUN_SI
    m2SI = m2 * lal.MSUN_SI
    f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
    deltaT = 1./f_sampling
    f_min = Mf_min / (M*lal.MTSUN_SI)

    # Prefactors to adimension h and t
    prefactor_h = 1. # the modes hIlm are adimensioned
    prefactor_t = 1./(M*lal.MTSUN_SI)

    # Generate SEOBNRv3, collect data pieces
    hp, hc, dynamicsHi, dynamics, hPinsp, hPinspHi, hJinspHi, hIIMR, AttachPars = lalsim.SimIMRSpinEOBWaveformAll(phi, deltaT, m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_min, distSI, 0., chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, 3)

    # Output dictionary - the set of keys will differ for v3, v4P_old, v4P
    res = {}
    res['hp'] = hp
    res['hc'] = hc
    res['dynamics'] = dynamics # requires patch
    res['dynamicsHi'] = dynamicsHi # requires patch
    res['hPinsp'] = hPinsp
    res['hPinspHi'] = hPinspHi
    res['hJinspHi'] = hJinspHi
    res['hJinspHi'] = hJinspHi
    res['hIIMR'] = hIIMR
    res['AttachPars'] = AttachPars
    # We save also the prefactors for adimensioned time and strain
    res['prefact_t'] = prefactor_t
    res['prefact_h'] = prefactor_h

    return res

#-------------------------------------------------------------------------------
# SEOBNRv4 pre-code rewrite
# Note: default Mf_sampling corresponds to 4096Hz, 50Msol
# def gen_SEOBNRv4Pold(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling):
#
#     # Arbitrary mass, distance
#     M = 50. # in solar masses
#     dist = 1. # in Mpc
#
#     # Masses in solar masses
#     m1 = M * q/(1.0+q)
#     m2 = M * 1.0/(1.0+q)
#
#     # SI units
#     distSI = dist * 1e6 * lal.PC_SI
#     m1SI = m1 * lal.MSUN_SI
#     m2SI = m2 * lal.MSUN_SI
#     f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
#     deltaT = 1./f_sampling
#     f_min = Mf_min / (M*lal.MTSUN_SI)
#
#     # Prefactors to adimension h and t
#     prefactor_h = 1. # the modes hIlm are adimensioned
#     prefactor_t = 1./(M*lal.MTSUN_SI)
#
#     # For old syntax, even though we are using v4
#     PrecEOBversion = 3
#
#     # Set the Euler extension to the simple-precession extension
#     flagEulerextension = 1
#
#     # Generate the older version of SEOBNRv4P, collect data pieces
#     hplus, hcross, dynamics, hPinsp, hPinspHi, hJinspHi, hIIMR, AttachPars = lalsim.SimIMRSpinPrecEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, PrecEOBversion, flagEulerextension)
#
#     # Output dictionary - the set of keys will differ for v3, v4P_old, v4P
#     res = {}
#     res['hp'] = hp
#     res['hc'] = hc
#     res['dynamics'] = dynamics
#     res['hPinsp'] = hPinsp
#     res['hPinspHi'] = hPinspHi
#     res['hJinspHi'] = hJinspHi
#     res['hJinspHi'] = hJinspHi
#     res['hIIMR'] = hIIMR
#     res['AttachPars'] = AttachPars
#     # We save also the prefactors for adimensioned time and strain
#     res['prefact_t'] = prefactor_t
#     res['prefact_h'] = prefactor_h
#
#     return res

#-------------------------------------------------------------------------------
# SEOBNRv4P
# Note: default Mf_sampling corresponds to 4096Hz, 50Msol
def gen_SEOBNRv4P(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling, flags_v4P=None):
    return  gen_SEOBNRv4PHM(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling, flags_v4P=None, PframeModes=[(2,2), (2,1)])

    # # Arbitrary mass, distance
    # M = 50. # in solar masses
    # dist = 1. # in Mpc
    #
    # # Masses in solar masses
    # m1 = M * q/(1.0+q)
    # m2 = M * 1.0/(1.0+q)
    #
    # # SI units
    # distSI = dist * 1e6 * lal.PC_SI
    # m1SI = m1 * lal.MSUN_SI
    # m2SI = m2 * lal.MSUN_SI
    # f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
    # deltaT = 1./f_sampling
    # f_min = Mf_min / (M*lal.MTSUN_SI)
    #
    # # Prefactors to adimension h and t - modes hIlm already adimensioned in this case
    # prefactor_h = 1.
    # prefactor_t = 1.
    #
    # # Flags
    # if flags_v4P is None:
    #     flagHamiltonianDerivative = 0
    #     flagEulerextension = 1
    #     flagZframe = 0
    #     flagSamplingInspiral = 0
    # else:
    #     #print flags_v4P
    #     if not include_flag_v4P_constSampling:
    #         flagHamiltonianDerivative, flagEulerextension, flagZframe, flagSamplingInspiral = flags_v4P
    #     else:
    #         flagHamiltonianDerivative, flagEulerextension, flagZframe = flags_v4P
    #
    # # Version of the Hamiltonian, flux/waveform, NQC to be used
    # flag_version_spinaligned = 0
    # if version_spinaligned=='v4':
    #     flag_version_spinaligned = 4
    # elif version_spinaligned=='v2':
    #     flag_version_spinaligned = 2
    # else:
    #     raise ValueError('version_spinaligned %s not recognized.' % version_spinaligned)
    #
    # #return (phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, flagHamiltonianDerivative, flagEulerextension, flagZframe)
    #
    # # Generate SEOBNRv4P, collect data pieces
    # # We allow for two interfaces, with and without flagSamplingInspiral - to be cleaned up
    # if not include_flag_v4P_constSampling:
    #     hplus, hcross, hIlm, hJlm, seobdynamicsAdaSVector, seobdynamicsHiSVector, seobdynamicsAdaSHiSVector, tVecPmodes, hP22_amp, hP22_phase, hP21_amp, hP21_phase, alphaJ2P, betaJ2P, gammaJ2P, mergerParams = lalsim.SimIMRSpinPrecEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, flag_version_spinaligned, flagHamiltonianDerivative, flagEulerextension, flagZframe)
    # else:
    #     hplus, hcross, hIlm, hJlm, seobdynamicsAdaSVector, seobdynamicsHiSVector, seobdynamicsAdaSHiSVector, tVecPmodes, hP22_amp, hP22_phase, hP21_amp, hP21_phase, alphaJ2P, betaJ2P, gammaJ2P, mergerParams = lalsim.SimIMRSpinPrecEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, flag_version_spinaligned, flagHamiltonianDerivative, flagEulerextension, flagZframe, flagSamplingInspiral)
    #
    # # Output dictionary - the set of keys will differ for v3, v4P_old, v4P
    # res = {}
    # res['hplus'] = hplus
    # res['hcross'] = hcross
    # res['hIIMR'] = hIlm
    # res['hJIMR'] = hJlm
    # res['dynamicsAdaS'] = seobdynamicsAdaSVector
    # res['dynamicsHiS'] = seobdynamicsHiSVector
    # res['dynamicsAdaSHiS'] = seobdynamicsAdaSHiSVector
    # res['tP'] = tVecPmodes
    # res['hP22_amp'] = hP22_amp
    # res['hP22_phase'] = hP22_phase
    # res['hP21_amp'] = hP21_amp
    # res['hP21_phase'] = hP21_phase
    # res['alphaJ2P'] = alphaJ2P
    # res['betaJ2P'] = betaJ2P
    # res['gammaJ2P'] = gammaJ2P
    # res['mergerParams'] = mergerParams
    # # We save also the prefactors for adimensioned time and strain
    # res['prefact_t'] = prefactor_t
    # res['prefact_h'] = prefactor_h
    #
    # return res

#-------------------------------------------------------------------------------
# SEOBNRv4PHM
# Note: default Mf_sampling corresponds to 4096Hz, 50Msol
def gen_SEOBNRv4PHM(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling, flags_v4P=None, PframeModes='All'):

    # Set of modes to be generated in the P-frame
    if PframeModes=='All':
        Pmodes = [(2,2), (2,1), (3,3), (4,4), (5,5)]
    else:
        Pmodes = PframeModes
    modearray = lalsim.SimInspiralCreateModeArray();
    for lm in Pmodes:
        lalsim.SimInspiralModeArrayActivateMode(modearray, lm[0], lm[1]);

    # Arbitrary mass, distance
    M = 50. # in solar masses
    dist = 1. # in Mpc

    # Masses in solar masses
    m1 = M * q/(1.0+q)
    m2 = M * 1.0/(1.0+q)

    # SI units
    distSI = dist * 1e6 * lal.PC_SI
    m1SI = m1 * lal.MSUN_SI
    m2SI = m2 * lal.MSUN_SI
    f_sampling = Mf_sampling / (M*lal.MTSUN_SI)
    deltaT = 1./f_sampling
    f_min = Mf_min / (M*lal.MTSUN_SI)

    # Prefactors to adimension h and t - modes hIlm already adimensioned in this case
    prefactor_h = 1.
    prefactor_t = 1.

    # Flags
    seobflags = lal.CreateDict()
    # NOTE:
    # DictInsertINT4Value chokes on those flag names, apparently because they are too long -- only when calling it through SWIG wrappings !
    # Deactivated for now -- defaults will be used in SimIMRSpinPrecEOBWaveformAll
    if flags_v4P is not None:
        print 'WARNING: flags_v4P temporarily ignored, problem with the length of the flag names.'
    # if flags_v4P is None:
    #     # Use v4 as underlying model
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_SpinAlignedEOBversion", 4);
    #     # Generate P-frame modes m<0 with the symmetry hP_l-m ~ (-1)^l hP_lm* */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_SymmetrizehPlminusm", 1);
    #     # Use numerical derivatives of the Hamiltonian */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_HamiltonianDerivative", lalsim.FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL);
    #     # Extension of Euler angles post-merger: simple precession around final J at a rate set by QNMs */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_euler_extension", lalsim.FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION);
    #     # Z-axis of the radiation frame L */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_Zframe", lalsim.FLAG_SEOBNRv4P_ZFRAME_L);
    #     # No debug output
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_debug", 0);
    # else:
    #     # Use v4 as underlying model
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_SpinAlignedEOBversion", flags_v4P.get('flagSEOBNRv4P_SpinAlignedEOBversion', default=4));
    #     # Generate P-frame modes m<0 with the symmetry hP_l-m ~ (-1)^l hP_lm* */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_SymmetrizehPlminusm", flags_v4P.get('flagSEOBNRv4P_SymmetrizehPlminusm', default=1));
    #     # Use numerical derivatives of the Hamiltonian */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_HamiltonianDerivative", flags_v4P.get('flagSEOBNRv4P_HamiltonianDerivative', default=lalsim.FLAG_SEOBNRv4P_HAMILTONIAN_DERIVATIVE_NUMERICAL));
    #     # Extension of Euler angles post-merger: simple precession around final J at a rate set by QNMs */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_euler_extension", flags_v4P.get('flagSEOBNRv4P_euler_extension', default=lalsim.FLAG_SEOBNRv4P_EULEREXT_QNM_SIMPLE_PRECESSION));
    #     # Z-axis of the radiation frame L */
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_Zframe", flags_v4P.get('flagSEOBNRv4P_Zframe', default=lalsim.FLAG_SEOBNRv4P_ZFRAME_L));
    #     # No debug output
    #     lal.DictInsertINT4Value(seobflags, "flagSEOBNRv4P_debug", flags_v4P.get('flagSEOBNRv4P_debug', default=0));

    # Generate SEOBNRv4P, collect data pieces
    #TEST
    # print phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, modearray, seobflags
    hplus, hcross, hIlm, hJlm, seobdynamicsAdaSVector, seobdynamicsHiSVector, seobdynamicsAdaSHiSVector, tVecPmodes, hP22_amp, hP22_phase, hP21_amp, hP21_phase, hP33_amp, hP33_phase, hP44_amp, hP44_phase, hP55_amp, hP55_phase, alphaJ2P, betaJ2P, gammaJ2P, mergerParams = lalsim.SimIMRSpinPrecEOBWaveformAll(phi, deltaT, m1SI, m2SI, f_min, distSI, inc, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, modearray, seobflags)

    # Output dictionary - the set of keys will differ for v3, v4P_old, v4P
    res = {}
    res['hplus'] = hplus
    res['hcross'] = hcross
    res['hIIMR'] = hIlm
    res['hJIMR'] = hJlm
    res['dynamicsAdaS'] = seobdynamicsAdaSVector
    res['dynamicsHiS'] = seobdynamicsHiSVector
    res['dynamicsAdaSHiS'] = seobdynamicsAdaSHiSVector
    res['tP'] = tVecPmodes
    res['hP22_amp'] = hP22_amp
    res['hP22_phase'] = hP22_phase
    res['hP21_amp'] = hP21_amp
    res['hP21_phase'] = hP21_phase
    res['hP33_amp'] = hP33_amp
    res['hP33_phase'] = hP33_phase
    res['hP44_amp'] = hP44_amp
    res['hP44_phase'] = hP44_phase
    res['hP55_amp'] = hP55_amp
    res['hP55_phase'] = hP55_phase
    res['alphaJ2P'] = alphaJ2P
    res['betaJ2P'] = betaJ2P
    res['gammaJ2P'] = gammaJ2P
    res['mergerParams'] = mergerParams
    # We save also the prefactors for adimensioned time and strain
    res['prefact_t'] = prefactor_t
    res['prefact_h'] = prefactor_h

    return res

#-------------------------------------------------------------------------------
# Generic wrapper functions for SEOB waveforms
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Generate SEOB waveform for spin-aligned non-precessing waveforms with 22 mode only, extract 22-mode
# To be generalized to HM as well
def gen_SEOB_aligned(q, chi1z, chi2z, Mf_min, phi=0., Mf_sampling=default_Mf_sampling, t0choice='tpeak', t_def_phi='tstart', version='v4', amp_cut_threshold=1e-12, drop=0, PQconvention='NR'):

    # Dictionary for output waveform
    wf = {}

    # List of modes
    modes = [(2,2)]
    wf['listmodes'] = modes

    # Generate SEOBNR waveform according to version - here dictionary returned is the same in different cases
    # We generate with 0 inclination in order to extract the 22 mode
    # If t_def_phi is not None, we generate with 0 phase, it is going to be set at t_def_phi
    if t_def_phi is not None:
        phi_gen = 0.
    else:
        phi_gen = phi
    if version=='v2':
        wfdict = gen_SEOBNRv2(q, chi1z, chi2z, Mf_min, inc=0., phi=phi_gen, Mf_sampling=Mf_sampling)
    elif version=='v4':
        wfdict = gen_SEOBNRv4(q, chi1z, chi2z, Mf_min, inc=0., phi=phi_gen, Mf_sampling=Mf_sampling)
    else:
        raise ValueError('Version %s not recognized.' % version)

    # Factors for adimensioned data
    prefact_t = wfdict['prefact_t']
    prefact_h = wfdict['prefact_h']

    # Global sign convention due to choice of polarization vectors (P,Q)
    if PQconvention=='NR':
        factorPQ = -1
    elif PQconvention=='EOB':
        factorPQ = 1
    else:
        raise ValueError('PQconvention %s not recognized.' % PQconvention)

    # Extract 22 mode from 0-inclination hp, hc
    nI = wfdict['hp'].data.length
    deltaTI = wfdict['hp'].deltaT
    tI = prefact_t * np.arange(nI) * deltaTI
    hI = {}
    hI[(2,2)] = factorPQ * prefact_h * 2*np.sqrt(pi/5.) * (wfdict['hp'].data.data - 1j*wfdict['hc'].data.data)

    # Cut samples where amplitude is too low, noisy for the phases - also allow to throw away some more samples at the end (drop parameter)
    # If no threshold is given, drop only the zeros
    combined_amp = np.sqrt(np.sum(np.real(np.array([hI[lm]*np.conj(hI[lm]) for lm in modes])), axis=0))
    max_combined_amp = np.max(combined_amp)
    if amp_cut_threshold is None:
        mask = (combined_amp > 0.)
    else:
        mask = (combined_amp > amp_cut_threshold * max_combined_amp)
    tI = (tI[mask])
    for lm in modes:
        hI[lm] = hI[lm][mask]
    if drop > 0:
        tI = tI[:-drop]
        for lm in modes:
            hI[lm] = hI[lm][:-drop]

    # Dynamics - convert to an array
    if version=='v4' or version=='v2': # requires patch
        dynamics = wfdict['dynamics'].data
        npt = len(dynamics)/5
        dyn = np.reshape(dynamics, (5, npt))
        dynamicsHi = wfdict['dynamicsHi'].data
        nptHi = len(dynamicsHi)/5
        dynHi = np.reshape(dynamicsHi, (5, nptHi))
    else:
        raise ValueError('Version %s not recognized.' % version)
    wf['dyn'] = dyn
    wf['dynHi'] = dynHi

    # No final J yet
    # No attachment parameters yet

    # Extract amplitudes and phases of the I-frame modes
    AI = {}; phiI = {};
    for lm in modes:
        AI[lm] = gwtools.ampseries(hI[lm])
        phiI[lm] = gwtools.phaseseries(hI[lm])

    # Imposing t=0 either at tA or at tpeak - if tstart, ignore (no shift in time is done)
    AIInt = {}
    for lm in modes:
        AIInt[lm] = ip.InterpolatedUnivariateSpline(tI, AI[lm], k=3)
    tpeak = gwtools.find_tpeak(max(tI[0], tI[-1]-1000), tI[-1], AIInt, modes)
    t0 = 0.
    if t0choice=='tpeak':
        t0 = tpeak
    elif t0choice=='tstart':
        t0 = tI[0]
    else:
        raise ValueError('t0choice %s not recognized.' % t0choice)
    tI = tI - t0
    tpeak = tpeak - t0
    # Also apply time shift in the dynamics
    if wf['dyn'] is not None:
        wf['dyn'][0] = wf['dyn'][0] - t0
    if wf['dynHi'] is not None:
        wf['dynHi'][0] = wf['dynHi'][0] - t0

    # Set the 22-phase to the specified value at the specified time
    phi22_shift = 0.
    if t_def_phi is not None:
        phi22_int = spline(tI, phiI[(2,2)])
        if t_def_phi=='tstart':
            phi22_shift = phi - phi22_int(tI[0])
        elif t_def_phi=='tpeak':
            phi22_shift = phi - phi22_int(tpeak)
        elif isinstance(t_def_phi, float):
            if (tI[0]<=t_def_phi and t_def_phi<=tI[-1]):
                phi22_shift = phi - phi22_int(t_def_phi)
            else:
                raise ValueError('Time of definition of the phase t_def_phi %.12f not in the range of times covered by the waveform, [%.12f, %.12f].' % (t_def_phi, tI[0], tI[-1]))
    for lm in modes:
        m = lm[1]
        phiI[lm] = phiI[lm] + m/2. * phi22_shift

    # Output
    wf['tI'] = tI
    wf['hI'] = hI
    wf['AI'] = AI
    wf['phiI'] = phiI
    wf['tpeak'] = tpeak
    return wf

#-------------------------------------------------------------------------------
# Generate SEOB waveform for precessing waveforms, extract P-frame angles and modes
# Two choices for the precessing frame vector : DominantEigenvector for the dominant eigenvector of the LL matrix, 'AngularVelocityVector' for the frame angular velocity vector, omega = - LL . Ldt
# Note: default Mf_sampling corresponds to 4096Hz, 50Msol
# Generate waveform and extract P-frame angles and modes
# Note: version_spinaligned and flags_v4P are only taken into account in v4P
def gen_SEOB(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=0., phi=0., Mf_sampling=default_Mf_sampling, t0choice='tpeak', Pframe='DominantEigenvector', Iframe='initialLN', Iframe_e1e2='e1inplanee1pe3p', version='v4P', flags_v4P=None, amp_cut_threshold=1e-12, drop=0, PQconvention='NR', PframeModes='All'):

    # TEST
    # print "(%.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e, %.16e)" % (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min)

    # Dictionary for output waveform
    wf = {}

    # List of modes
    if PframeModes=='All':
        #if version in ['v3', 'v4Pold', 'v4P']:
        if version in ['v3', 'v4P']:
            Pmodes = [(2,2), (2,1)]
        elif version=='v4PHM':
            Pmodes = [(2,2), (2,1), (3,3), (4,4), (5,5)]
        else:
            raise ValueError('version %s not recognized.' % version)
    else:
        Pmodes = PframeModes
    lvalues = func_lvalues(Pmodes)
    modes = func_allmodes_lvalues(lvalues)
    modes_l2 = [(2, 2-m) for m in range(5)] # Will be used for the amplitude
    wf['listmodes'] = modes

    # eta
    eta = gwtools.etaofq(q)

    # Generate SEOBNR waveform according to version - dictionary returned is different in different cases
    if version=='v3':
        wfdict = gen_SEOBNRv3(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=inc, phi=phi, Mf_sampling=Mf_sampling)
    # elif version=='v4Pold':
    #     wfdict = gen_SEOBNRv4Pold(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=inc, phi=phi, Mf_sampling=Mf_sampling)
    elif version=='v4P' or version=='v4PHM':
        #TEST
        # print 'before gen_SEOBNRv4PHM'
        wfdict = gen_SEOBNRv4PHM(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, inc=inc, phi=phi, Mf_sampling=Mf_sampling, flags_v4P=flags_v4P, PframeModes=Pmodes)
    else:
        raise ValueError('Version %s not recognized.' % version)

    # Factors for adimensioned data
    prefact_t = wfdict['prefact_t']
    prefact_h = wfdict['prefact_h']

    # Global sign convention due to choice of polarization vectors (P,Q)
    if PQconvention=='NR':
        factorPQ = -1
    elif PQconvention=='EOB':
        factorPQ = 1
    else:
        raise ValueError('PQconvention %s not recognized.' % PQconvention)

    # Mode-by-mode output for the full waveform in Inertial Frame
    hI_SphHarmTS = wfdict['hIIMR']
    nI = lalsim.SphHarmTimeSeriesGetMode(hI_SphHarmTS, 2, 2).data.length
    deltaTI = lalsim.SphHarmTimeSeriesGetMode(hI_SphHarmTS, 2, 2).deltaT # in physical units
    tI = prefact_t * np.arange(nI) * deltaTI
    hI = {}
    for m in range(-2,3):
        hI[(2,m)] = factorPQ * prefact_h * lalsim.SphHarmTimeSeriesGetMode(hI_SphHarmTS, 2, m).data.data
    wf['hI'] = hI

    # Cut samples where amplitude is too low, noisy for the phases - also allow to throw away some more samples at the end (drop parameter)
    # If no threshold is given, drop only the zeros
    combined_amp = np.sqrt(np.sum(np.real(np.array([hI[lm]*np.conj(hI[lm]) for lm in modes])), axis=0))
    max_combined_amp = np.max(combined_amp)
    if amp_cut_threshold is None:
        mask = (combined_amp > 0.)
    else:
        mask = (combined_amp > amp_cut_threshold * max_combined_amp)
    tI = (tI[mask])
    for lm in modes:
        hI[lm] = (wf['hI'][lm][mask])
    if drop > 0:
        tI = tI[:-drop]
        for lm in modes:
            hI[lm] = hI[lm][:-drop]
    for lm in modes:
        wf['hI'][lm] = hI[lm]

    # Dynamics - convert to an array - differs in different cases
    if version=='v3': # after patch, inspiral dynamics is returned
        dynamics = wfdict['dynamics'].data
        npt = len(dynamics)/15
        dyn = np.reshape(dynamics, (15, npt))
        dynamicsHi = wfdict['dynamicsHi'].data
        nptHi = len(dynamicsHi)/15
        dynHi = np.reshape(dynamicsHi, (15, nptHi))
    # elif version=='v4Pold': # dynLowS was returned in this case, high-sampling dynamics not supported
    #     dynamics = wfdict['dynamics'].data
    #     npt = len(dynamics)/15
    #     dyn = np.reshape(dynamics, (15, npt))
    #     dynHi = None
    elif version=='v4P' or version=='v4PHM': # 26-dimensional extended dynamics here
        dynamics = wfdict['dynamicsAdaSHiS'].data
        npt = len(dynamics)/26
        dyn_all = np.reshape(dynamics, (26, npt))
        dyn = dyn_all[:15] # we only keep the first 15 fields, coming back to the same format as v3 and v4Pold
        dynamicsHi = wfdict['dynamicsHiS'].data
        nptHi = len(dynamicsHi)/26
        dynHi_all = np.reshape(dynamicsHi, (26, nptHi))
        dynHi = dynHi_all[:15] # we only keep the first 15 fields, coming back to the same format as v3 and v4Pold
    else:
        raise ValueError('Version %s not recognized.' % version)
    # The SEOB dynamics returned by the C code has p/eta instead of p -- here restore p
    dyn[4:7] *= eta
    dynHi[4:7] *= eta
    wf['dyn'] = dyn
    wf['dynHi'] = dynHi
    # Also keep the extended dynamics if using v4P
    if version=='v4P' or version=='v4PHM':
        wf['dyn_all'] = dyn_all
        wf['dynHi_all'] = dynHi_all
    else:
        wf['dyn_all'] = None
        wf['dynHi_all'] = None

    # Initial and final J
    Jidyn = gwtools.J_from_dynamics(dyn, 0)
    Jidynhat = gwtools.normalize(Jidyn)
    Jfdyn = gwtools.J_from_dynamics(dyn, -1)
    Jfdynhat = gwtools.normalize(Jfdyn)
    wf['Jidyn'] = Jidyn
    wf['Jfdyn'] = Jfdyn

    # Read final spin used in the EOB ringdown - dictionary returned is different in different cases
    # For debugging purposes, also keep mergerParams of v4P, AttachPars of v3 under the name mergerParams
    if version=='v3': # after patch, extended AttachPars output
        finalMass = (wfdict['AttachPars'].data)[5]
        finalSpin = (wfdict['AttachPars'].data)[6]
        finalJ = (wfdict['AttachPars'].data)[7:10]
        chifvec = gwtools.normalize(finalJ) * finalSpin
        chif = finalSpin
        wf['chifvec'] = chifvec
        wf['chif'] = chif
        wf['Mf'] = None
        wf['mergerParams'] = wfdict['AttachPars'].data
    # elif version=='v4Pold':
    #     chifvec = (wfdict['AttachPars'].data)[6:9]
    #     chif = gwtools.norm(chifvec)
    #     wf['chifvec'] = chifvec
    #     wf['chif'] = chif
    #     wf['Mf'] = None
    #     wf['mergerParams'] = None
    elif version=='v4P' or version=='v4PHM': # we set the direction of chif along final J, with chif=finalSpin
        Jfvec = (wfdict['mergerParams'].data)[3:6]
        finalMass = (wfdict['mergerParams'].data)[6]
        finalSpin = (wfdict['mergerParams'].data)[7]
        wf['chifvec'] = gwtools.normalize(Jfvec) * finalSpin
        wf['chif'] = finalSpin
        wf['Mf'] = finalMass
        wf['mergerParams'] = wfdict['mergerParams'].data
    else:
        raise ValueError('Version %s not recognized.' % version)

    # Rotate waveform to the desired output inertial frame
    # Three choices : initial inertial frame (x,y,z), initial-J frame, final-J frame
    # For J-frame, allow two different prescription for fixing (e1p,e2p) by rotating around J
    # Note: these functions return a deepcopy of wf dictionary with modified modes - other keys are copied
    if Iframe=='initialLN':
        pass
    elif Iframe=='initialJdyn':
        wf = gwtools.rotate_wf_toframe(wf, Jidynhat, e1e2framechoice=Iframe_e1e2)
    elif Iframe=='finalJdyn':
        wf = gwtools.rotate_wf_toframe(wf, Jfdynhat, e1e2framechoice=Iframe_e1e2)
    elif Iframe=='finalSpin':
        wf = gwtools.rotate_wf_toframe(wf, gwtools.normalize(wf['chifvec']), e1e2framechoice=Iframe_e1e2)
    else:
        raise ValueError('Unrecognized value for Iframe.')

    # Extract amplitudes and phases of the I-frame modes
    AI = {}; phiI = {};
    for lm in modes:
        AI[lm] = gwtools.ampseries(hI[lm])
        phiI[lm] = gwtools.phaseseries(hI[lm])
    wf['AI'] = AI
    wf['phiI'] = phiI

    # Build precessing-frame waveform - uses GWFrames
    #TEST
    # print 'before GWFrame'
    T_data = tI
    LM_data = np.array(list(map(list, modes)), dtype=np.int32)
    mode_data = np.array([wf['hI'][lm] for lm in modes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    if Pframe=='DominantEigenvector':
        W_v3.TransformToCoprecessingFrame()
    elif Pframe=='AngularVelocityVector':
        W_v3.TransformToAngularVelocityFrame()
    else:
        raise 'P-frame choice not recognized.'
    #TEST
    # print 'after GWFrame'

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    euler = np.array(list(map(gwtools.euler_from_quat, quat)))
    alpha = np.unwrap(euler[:,0])
    beta = euler[:,1]
    gamma = np.unwrap(euler[:,2])
    wf['euler'] = {}
    wf['euler']['alpha'] = alpha
    wf['euler']['beta'] = beta
    wf['euler']['gamma'] = gamma

    # Mode-by-mode output for precessing-frame waveform
    iGWF = {}; AP = {}; phiP = {};
    for lm in modes:
        iGWF[lm] = W_v3.FindModeIndex(lm[0], lm[1])
        AP[lm] = W_v3.Abs(iGWF[lm])
        phiP[lm] = W_v3.ArgUnwrapped(iGWF[lm])
    tP = tI
    wf['AP'] = AP
    wf['phiP'] = phiP

    # Locate the time of peak for the combined amplitude
    APInt = {}
    for lm in modes:
        APInt[lm] = ip.InterpolatedUnivariateSpline(tP, AP[lm], k=3)
    tpeak = gwtools.find_tpeak(tI[-1]-1000, tI[-1], APInt, modes)

    # Imposing t=0 either at tA, at tpeak or at the start
    t0 = 0.
    if t0choice=='tpeak':
        t0 = tpeak
    elif t0choice=='tstart':
        t0 = tP[0]
    elif t0choice=='tA':
        t0 = tA
    else:
        raise ValueError('t0choice %s not recognized.' % t0choice)
    tI = tI - t0
    tP = tP - t0
    tpeak = tpeak - t0
    # Also apply time shift in the dynamics
    if wf['dyn'] is not None:
        wf['dyn'][0] = wf['dyn'][0] - t0
    if wf['dynHi'] is not None:
        wf['dynHi'][0] = wf['dynHi'][0] - t0

    # Output times
    wf['tI'] = tI
    wf['tP'] = tP
    wf['tpeak'] = tpeak

    # Output
    return wf

#-------------------------------------------------------------------------------
# Obtain SEOB initial frequency for comparison with NR
#-------------------------------------------------------------------------------

def funcSEOBtimetomerger(Mf_min, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_sampling=default_Mf_sampling, version='v4P'):
    wf = gen_SEOB(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, Mf_sampling=Mf_sampling, t0choice='tpeak', Pframe='DominantEigenvector', Iframe='initialLN', version=version)
    return -wf['tI'][0]

def funcgetSEOBinitialomega(omegaNR, DeltatNR, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4P'):
    def func0(omega):
        deltat = funcSEOBtimetomerger(omega/pi, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_sampling=Mf_sampling, version=version) - DeltatNR
        return deltat
    return scipy.optimize.brentq(func0, omegaNR - omegawidth, omegaNR + omegawidth, xtol=1e-8, rtol=1e-6, maxiter=30, full_output=False, disp=True)

# Root-finding to get the time to get to a certain value of orbital omega, as measured from the start
def func_timetoomega(t, omega_orb, omega_orb_target):
    omega_orb_int = spline(t, omega_orb)
    def func0(t):
        return omega_orb_int(t) - omega_orb_target
    t_omega = scipy.optimize.brentq(func0, t[0], t[-1], xtol=1e-8, rtol=1e-6, maxiter=30, full_output=False, disp=False)
    return t_omega - t[0]

def funcSEOBtimetoomega(Mf_min, omega_end, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_sampling=default_Mf_sampling, version='v4P'):
    wf = gen_SEOB(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_min, Mf_sampling=Mf_sampling, t0choice='tpeak', Pframe='DominantEigenvector', Iframe='initialLN', version=version)
    dyn = wf['dyn']
    t = dyn[0]
    omega_orb = func_omegaorb_from_dyn_seob(dyn)
    return func_timetoomega(t, omega_orb, omega_end)

def funcgetSEOBinitialomega_timetoomega(omegaNR_start, omegaNR_end, DeltatNR, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4P'):
    def func0(omega):
        Deltat_diff = funcSEOBtimetoomega(omega/pi, omegaNR_end, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, Mf_sampling=Mf_sampling, version=version) - DeltatNR
        return Deltat_diff
    return scipy.optimize.brentq(func0, omegaNR_start - omegawidth, omegaNR_start + omegawidth, xtol=1e-8, rtol=1e-6, maxiter=30, full_output=False, disp=True)

def funcSEOBAlignedtimetomerger(Mf_min, q, chi1z, chi2z, Mf_sampling=default_Mf_sampling, version='v4'):
    wf = gen_SEOB_aligned(q, chi1z, chi2z, Mf_min, Mf_sampling=Mf_sampling, t0choice='tpeak', version=version)
    return -wf['tI'][0]

def funcgetSEOBAlignedinitialomega_optimize(omegaNR, DeltatNR, q, chi1z, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4'):
    def func0(omega):
        deltat = funcSEOBAlignedtimetomerger(omega/pi, q, chi1z, chi2z, Mf_sampling=Mf_sampling, version=version) - DeltatNR
        return deltat
    return scipy.optimize.brentq(func0, omegaNR - omegawidth, omegaNR + omegawidth, xtol=1e-8, rtol=1e-6, maxiter=30, full_output=False, disp=True)

def funcgetSEOBAlignedinitialomega(omegaNR, DeltatNR, q, chi1z, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4'):
    # Generate a longer waveform - spins are constants
    Mf_min_NR = omegaNR / pi
    wf = gen_SEOB_aligned(q, chi1z, chi2z, 0.8*Mf_min_NR, Mf_sampling=Mf_sampling, t0choice='tpeak', version=version)
    dyn = wf['dyn']
    t = dyn[0]
    omega_of_t = np.array([t, func_omegaorb_from_dyn_seob_aligned(dyn)]).T
    omega_of_t = gwtools.restrict_data(omega_of_t, [omega_of_t[0,0], wf['tpeak']])
    omega_of_t_int = spline(omega_of_t[:,0], omega_of_t[:,1])
    return omega_of_t_int(wf['tpeak'] - DeltatNR)

###

def funcgetSEOBAlignedinitialomega_timetoomega(omegaNR_start, omegaNR_end, DeltatNR, q, chi1z, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4'):
    # Generate a longer waveform - spins are constants
    Mf_min_NR = omegaNR_start / pi
    wf = gen_SEOB_aligned(q, chi1z, chi2z, 0.8*Mf_min_NR, Mf_sampling=Mf_sampling, t0choice='tpeak', version=version)
    dyn = wf['dyn']
    t = dyn[0]
    omega_of_t = np.array([t, func_omegaorb_from_dyn_seob_aligned(dyn)]).T
    omega_of_t = gwtools.restrict_data(omega_of_t, [omega_of_t[0,0], wf['tpeak']])
    omega_of_t_int = spline(omega_of_t[:,0], omega_of_t[:,1])
    t_of_omega = np.array([omega_of_t[:,1], omega_of_t[:,0]]).T
    t_of_omega_int = spline(t_of_omega[:,0], t_of_omega[:,1])
    if omegaNR_start < omega_of_t[0,1] or omegaNR_end < omega_of_t[0,1]:
        raise ValueError("SEOB waveform not long enough.")
    return omega_of_t_int(t_of_omega_int(omegaNR_end) - DeltatNR)

def funcgetSEOBAlignedinitialomega_timetoomega_optimize(omegaNR_start, omegaNR_end, DeltatNR, q, chi1z, chi2z, omegawidth=1e-3, Mf_sampling=default_Mf_sampling, version='v4'):
    def func0(omega):
        Deltat_diff = funcSEOBAlignedtimetoomega(omega/pi, omegaNR_end, q, chi1z, chi2z, Mf_sampling=Mf_sampling, version=version) - DeltatNR
        return Deltat_diff
    return scipy.optimize.brentq(func0, omegaNR_start - omegawidth, omegaNR_start + omegawidth, xtol=1e-8, rtol=1e-6, maxiter=30, full_output=False, disp=True)

def funcSEOBAlignedtimetoomega(Mf_min, omega_end, q, chi1z, chi2z, Mf_sampling=default_Mf_sampling, version='v4'):
    wf = gen_SEOB_aligned(q, chi1z, chi2z, Mf_min, Mf_sampling=Mf_sampling, t0choice='tpeak', version=version)
    dyn = wf['dyn']
    t = dyn[0]
    omega_orb = func_omegaorb_from_dyn_seob_aligned(dyn)
    return func_timetoomega(t, omega_orb, omega_end)
