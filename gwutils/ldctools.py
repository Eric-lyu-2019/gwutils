from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


import numpy as np
from numpy import pi, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import pyFDresponse
import LISAConstants as LC

import gwutils.gwtools as gwtools

from astropy.cosmology import Planck15 as cosmo

################################################################################
# Noise functions
################################################################################

# Copied from flare LISAnoise.c, converted to python by hand
# NOTE: on master 71ff1ad01f0, there is a bug in LISAnoise.c:
# SnAXYZNoRescaling was using SpmLISA2017, SopLISA2017, disregarding variant -- but the default is LISAproposal

#/* Proof mass and optical noises - f in Hz */
#/* L3 Reference Mission, from Petiteau LISA-CST-TN-0001 */
def SpmLISA2017(f):
    invf2 = 1./(f*f);
    #//invf4=invf2*invf2;
    #//invf8=invf4*invf4;
    #//invf10=invf8*invf2;
    twopi2=4.0*np.pi*np.pi;
    ddtsq=twopi2/invf2; #//time derivative factor
    C2=1.0*gwtools.c*gwtools.c; #//veloc to doppler
    Daccel_white=3.0e-15; #//acceleration noise in m/s^2/sqrt(Hz)
    Daccel_white2=Daccel_white*Daccel_white;
    Dloc=1.7e-12; #//local IFO noise in m/sqrt(Hz)
    Dloc2=Dloc*Dloc;
    Saccel_white=Daccel_white2/ddtsq; #//PM vel noise PSD (white accel part)
    #//Saccel_red=Saccel_white*(1.0 + 2.12576e-44*invf10 + 3.6e-7*invf2); #//reddening factor from Petiteau Eq 1
    Saccel_red=Saccel_white*(1.0 + 36.0*(np.power(3e-5/f,10) + 1e-8*invf2)); #//reddening factor from Petiteau Eq 1
    #//Saccel_red*=4.0;#//Hack to decrease low-freq sens by fac of 2.
    Sloc=Dloc2*ddtsq/4.0;#//Factor of 1/4.0 is in Petiteau eq 2
    S4yrWDWD=5.16e-27*exp(-np.power(f,1.2)*2.9e3)*np.power(f,(-7./3.))*0.5*(1.0 + np.tanh(-(f-2.0e-3)*1.9e3))*ddtsq;#//Stas' fit for 4yr noise (converted from Sens curve to position noise by multipyling by 3*L^2/80) which looks comparable to my fit), then converted to velocity noise
    Spm_vel = ( Saccel_red + Sloc + S4yrWDWD );
    return Spm_vel / C2;#//finally convert from velocity noise to fractional-frequency doppler noise.
def SopLISA2017(f):
    #//invf2 = 1./(f*f);
    twopi2=4.0*np.pi*np.pi;
    ddtsq=twopi2*f*f; #//time derivative factor
    C2=gwtools.c*gwtools.c; #//veloc to doppler
    Dloc=1.7e-12; #//local IFO noise in m/sqrt(Hz)
    Dsci=8.9e-12; #//science IFO noise in m/sqrt(Hz)
    Dmisc=2.0e-12; #//misc. optical path noise in m/sqrt(Hz)
    Dop2=Dsci*Dsci+Dloc*Dloc+Dmisc*Dmisc;
    Sop=Dop2*ddtsq/C2; #//f^2 coeff for OP frac-freq noise PSD.  Yields 1.76e-37 for Dop=2e-11.
    return Sop;

#/* Proof mass and optic noises - f in Hz */
#/* Taken from (4) in McWilliams&al_0911 */
def SpmLISA2010(f):
    invf2 = 1./(f*f);
    #//return 2.5e-48 * invf2 * sqrt(1. + 1e-8*invf2);
    #//const double Daccel=3.0e-15; //acceleration noise in m/s^2/sqrt(Hz)
    Daccel=3.0e-15; #//scaled off L3LISA-v1 to for equal-SNR PE experiment
    SaccelFF=Daccel*Daccel/4.0/np.pi/np.pi/gwtools.c/gwtools.c; #//f^-2 coeff for fractional-freq noise PSD from accel noise; yields 2.54e-48 from 3e-15;
    invf8=invf2*invf2*invf2*invf2;
    #//Here we add an eyeball approximation based on 4yrs integration with L3LISAReferenceMission looking at a private comm from Neil Cornish 2016.11.12
    WDWDnoise=5000.0/sqrt(1e-21*invf8 + invf2 + 3e28/invf8)*SaccelFF*invf2;
    return SaccelFF * invf2 * sqrt(1. + 1e-8*invf2) + WDWDnoise;
def SopLISA2010(f):
    Dop=2.0e-11; #//Optical path noise in m/rtHz (Standard LISA)
    SopFF=Dop*Dop*4.0*np.pi*np.pi/gwtools.c/gwtools.c; #//f^2 coeff for OP frac-freq noise PSD.  Yields 1.76e-37 for Dop=2e-11.
    return SopFF * f * f;

#/* Proof mass and optical noises - f in Hz */
#/* LISA Proposal, copied from the LISA Data Challenge pipeline */
def SpmLISAProposal(f):
    #/* Acceleration noise */
    noise_Sa_a = 9.e-30; #/* m^2/sec^4 /Hz */
    #/* In acceleration */
    Sa_a = noise_Sa_a * (1.0 + np.power(0.4e-3/f, 2)) * (1.0 + np.power((f/8e-3), 4));
    #/* In displacement */
    Sa_d = Sa_a * pow(2.*np.pi*f, -4);
    #/* In relative frequency unit */
    Sa_nu = Sa_d * pow(2.*np.pi*f/gwtools.c, 2);
    Spm = Sa_nu;
    return Spm;
def SopLISAProposal(f):
    #/* Optical Metrology System noise */
    noise_Soms_d = np.power((10e-12), 2); #/* m^2/Hz */
    #/* In displacement */
    Soms_d = noise_Soms_d * (1. + np.power(2.e-3/f, 4));
    #/* In relative frequency unit */
    Soms_nu = Soms_d * np.power(2.*np.pi*f/gwtools.c, 2);
    Sop = Soms_nu;
    return Sop;

# Proof mass and optical noises - f in Hz
# LISA Science Requirement Document, copied from the LISA Data Challenge pipeline -- tdi.py and LISAParameters.py
# 'SciRDv1': Science Requirement Document: ESA-L3-EST-SCI-RS-001 14/05/2018 (https://atrium.in2p3.fr/f5a78d3e-9e19-47a5-aa11-51c81d370f5f)
def SpmLISASciRDv1(f):
    ### Acceleration noise
    Sa_a = (3.e-15)**2 # see LISAParameters.py
    ## In acceleration
    Sa_a = Sa_a * (1.0 +(0.4e-3/f)**2)*(1.0+(f/8e-3)**4)
    ## In displacement
    Sa_d = Sa_a*(2.*np.pi*f)**(-4.)
    ## In relative frequency unit
    Sa_nu = Sa_d*(2.0*np.pi*f/gwtools.c)**2
    Spm = Sa_nu
    return Spm
def SopLISASciRDv1(f):
    ### Optical Metrology System
    Soms_d = (15.e-12)**2 # see LISAParameters.py
    ## In displacement
    Soms_d = Soms_d * (1. + (2.e-3/f)**4)
    ## In relative frequency unit
    Soms_nu = Soms_d*(2.0*np.pi*f/gwtools.c)**2
    Sop = Soms_nu
    return Sop

def SpmLISA(f, variant='LISAproposal'):
    if variant=='LISAproposal':
        return SpmLISAProposal(f)
    elif variant=='LISA2017':
        return SpmLISA2017(f)
    elif variant=='LISA2010':
        return SpmLISA2010(f)
    elif variant=='LISASciRDv1':
        return SpmLISASciRDv1(f)
    else:
        raise ValueError('Unrecognized variant %s' % variant)
def SopLISA(f, variant='LISAproposal'):
    if variant=='LISAproposal':
        return SopLISAProposal(f)
    elif variant=='LISA2017':
        return SopLISA2017(f)
    elif variant=='LISA2010':
        return SopLISA2010(f)
    elif variant=='LISASciRDv1':
        return SopLISASciRDv1(f)
    else:
        raise ValueError('Unrecognized variant %s' % variant)

# NOTE: here variant does not include L like in the C code
#/* Rescaled by 2*sin2pifL^2 */
def SnAXYZ(f, L=2.5e9, variant='LISAproposal'):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    c2 = np.cos(twopifL);
    c4 = np.cos(2*twopifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 2*(3. + 2*c2 + c4)*Spm + (2 + c2)*Sop;
#/* Rescaled by 2*sin2pifL^2 */
def SnEXYZ(f, L=2.5e9, variant='LISAproposal'):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    c2 = np.cos(twopifL);
    c4 = np.cos(2*twopifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 2*(3. + 2*c2 + c4)*Spm + (2 + c2)*Sop;
#/* Rescaled by 8*sin2pifL^2*sinpifL^2 */
def SnTXYZ(f, L=2.5e9, variant='LISAproposal'):
    pifL = np.pi*L/gwtools.c*f;
    s1 = np.sin(pifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 4*s1*s1*Spm + Sop;
#/* Noise functions for AET(XYZ) without rescaling */
#/* Scaling by 2*np.sin2pifL^2 put back */
def SnAXYZNoRescaling(f, L=2.5e9, variant='LISAproposal'):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    c2 = np.cos(twopifL);
    c4 = np.cos(2*twopifL);
    s2 = np.sin(twopifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 2*s2*s2 * (2*(3. + 2*c2 + c4)*Spm + (2 + c2)*Sop);
#/* Scaling by 2*np.sin2pifL^2 put back */
def SnEXYZNoRescaling(f, L=2.5e9, variant='LISAproposal'):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    c2 = np.cos(twopifL);
    c4 = np.cos(2*twopifL);
    s2 = np.sin(twopifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 2*s2*s2 * (2*(3. + 2*c2 + c4)*Spm + (2 + c2)*Sop);
#/* Scaling by 8*np.sin2pifL^2*np.sinpifL^2 put back*/
def SnTXYZNoRescaling(f, L=2.5e9, variant='LISAproposal'):
    pifL = np.pi*L/gwtools.c*f;
    s1 = np.sin(pifL);
    s2 = np.sin(2*pifL);
    Spm = SpmLISA(f, variant=variant);
    Sop = SopLISA(f, variant=variant);
    return 8*s1*s1*s2*s2 * (4*s1*s1*Spm + Sop);

# Not from C, written here to access the rescaling conveniently
def RescalingSnAXYZ(f, L=2.5e9):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    s2 = np.sin(twopifL);
    return 2*s2*s2;
def RescalingSnEXYZ(f, L=2.5e9):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    s2 = np.sin(twopifL);
    return 2*s2*s2;
def RescalingSnTXYZ(f, L=2.5e9):
    pifL = np.pi*L/gwtools.c*f;
    s1 = np.sin(pifL);
    s2 = np.sin(2*pifL);
    return 8*s1*s1*s2*s2;

# Not from C, written here to access the rescaling conveniently
#  Quote from C:
#     /* First-generation rescaled TDI aet from X,Y,Z */
#     /* With x=pifL, factors scaled out: A,E I*sqrt2*sin2x*e2ix - T 2*sqrt2*sin2x*sinx*e3ix */
def RescalingTDIAXYZ(f, L=2.5e9):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    s2 = np.sin(twopifL);
    e2 = np.exp(1j*twopifL);
    return 1j*np.sqrt(2)*s2*e2;
def RescalingTDIEXYZ(f, L=2.5e9):
    twopifL = 2.*np.pi*L/gwtools.c*f;
    s2 = np.sin(twopifL);
    e2 = np.exp(1j*twopifL);
    return 1j*np.sqrt(2)*s2*e2;
def RescalingTDITXYZ(f, L=2.5e9):
    pifL = np.pi*L/gwtools.c*f;
    s1 = np.sin(pifL);
    s2 = np.sin(2*pifL);
    e3 = np.exp(1j*3*pifL);
    return 2*np.sqrt(2)*s2*s1*e3;

################################################################################
# SNR functions
################################################################################

################################################################################
# SNR

# BEWARE: For now, assume input TDI wf has 3 channels AET
# Returns the array of the total SNR, followed by those of the three channels A,E,T
# If cumul is True, returns arrays of cumulative SNR instead of total value
# NOTE: to avoid 0/0 situations, wfTDI has to be rescaled, and we use the rescaled version of the noises
# NOTE: LDC convention for A,E,T assumed -- (A,E,T)_LDC = 2 * (A,E,T)_flare
def LISAComputeSNR_TDIAET(wfTDI, df=None, cumul=False, npts=10000, variant='LISAproposal'):
    if not wfTDI['TDItag']=='TDIAET':
        raise ValueError('TDItag not supported, must be TDIAET.')
    # If df is not specified, determine it from the Nyquist time, with duration of the signal estimated from tf at the beginning and end of the waveform
#     if df is None:
#         phasespline = gwtools.spline(wfTDI['freq'], wfTDI['phase'])
#         tfstart = 1./(2*np.pi) * phasespline(wfTDI['freq'][0], 1)
#         tfend = 1./(2*np.pi) * phasespline(wfTDI['freq'][-1], 1)
#         dfval = 0.8 * 1./(2*np.abs(tfend - tfstart)) # Factor 0.8 arbitrary, safety margin in the Nyquist criterion
#     else:
#         dfval = df
#     if downsample is not None:
#         dfval = downsample*dfval
#     freqs = np.arange(wfTDI['freq'][0], wfTDI['freq'][-1] + dfval, step=dfval)
    freqs = gwtools.logspace(wfTDI['freq'][0], wfTDI['freq'][-1], npts)
    # Compute rescaled noise, flare convention
    Sn_AE_flare = SnAXYZ(freqs, L=2.5e9, variant=variant)
    Sn_T_flare = SnTXYZ(freqs, L=2.5e9, variant=variant)
    # factor of 4 larger than in flare conventions as (A,E,T)_LDC = 2 * (A,E,T)_flare
    Sn_AE = 4*Sn_AE_flare
    Sn_T = 4*Sn_T_flare
    # Evaluate TDI channels on the array of frequencies
    tdiA_vals = pyFDresponse.LISAEvaluateTDIFreqGrid(freqs, wfTDI, chan=1)
    tdiE_vals = pyFDresponse.LISAEvaluateTDIFreqGrid(freqs, wfTDI, chan=2)
    tdiT_vals = pyFDresponse.LISAEvaluateTDIFreqGrid(freqs, wfTDI, chan=3)
    if not cumul:
        # Compute total SNR^2 for each channel
        SNR2_A = 4.* np.sum(np.diff(freqs) * (np.abs(tdiA_vals)**2/Sn_AE)[:-1])
        SNR2_E = 4.* np.sum(np.diff(freqs) * (np.abs(tdiE_vals)**2/Sn_AE)[:-1])
        SNR2_T = 4.* np.sum(np.diff(freqs) * (np.abs(tdiT_vals)**2/Sn_T)[:-1])
    else:
        # Compute total SNR^2 for each channel
        SNR2_A_cum = 4.* np.cumsum(np.diff(freqs) * (np.abs(tdiA_vals)**2/Sn_AE)[:-1])
        SNR2_E_cum = 4.* np.cumsum(np.diff(freqs) * (np.abs(tdiE_vals)**2/Sn_AE)[:-1])
        SNR2_T_cum = 4.* np.cumsum(np.diff(freqs) * (np.abs(tdiT_vals)**2/Sn_T)[:-1])
        # Downsample on the original frequency array
        SNR2_A = gwtools.spline(freqs[:-1], SNR2_A_cum)(wfTDI['freq'])
        SNR2_E = gwtools.spline(freqs[:-1], SNR2_E_cum)(wfTDI['freq'])
        SNR2_T = gwtools.spline(freqs[:-1], SNR2_T_cum)(wfTDI['freq'])

    # Result
    return np.array([np.sqrt(SNR2_A + SNR2_E + SNR2_T), np.sqrt(SNR2_A), np.sqrt(SNR2_E), np.sqrt(SNR2_T)])

def LISASNR_AET(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal'):
    zval = z
    Mval = M
    qval = q

    # Masses
    m1 = Mval*qval/(1.+qval)
    m2 = Mval*1/(1.+qval)

    dist = cosmo.luminosity_distance(zval).value

    wftdi_rescaled = pyFDresponse.LISAGenerateTDI(phi, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0, settRefAtfRef=False, tRef=0., TDItag='TDIAET', order_fresnel_stencil=0, nptmin=100, rescaled=True)

    tf = 1./(2*np.pi)*gwtools.spline(wftdi_rescaled['freq'], wftdi_rescaled['phase'])(wftdi_rescaled['freq'], 1)

    return LISAComputeSNR_TDIAET(wftdi_rescaled, df=None, cumul=False, npts=npts, variant=variant)

def LISASNR(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal'):
    return LISASNR_AET(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0, fRef=fRef, npts=npts, variant=variant)[0]

def draw_random_angles():
    phi = np.random.uniform(low=-np.pi, high=np.pi)
    inc = np.arccos(np.random.uniform(low=-1., high=1.))
    lambd = np.random.uniform(low=-np.pi, high=np.pi)
    beta = np.arcsin(np.random.uniform(low=-1., high=1.))
    psi = np.random.uniform(low=0., high=np.pi)
    return np.array([phi, inc, lambd, beta, psi])

def LISASNR_AET_average_angles(M, q, chi1, chi2, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False):
    SNR_AET_arr = np.zeros((N,4))
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        SNR_AET_arr[i] = LISASNR_AET(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    SNR_av = np.mean(SNR_AET_arr[:,0])
    SNR_A_av = np.mean(SNR_AET_arr[:,1])
    SNR_E_av = np.mean(SNR_AET_arr[:,2])
    SNR_T_av = np.mean(SNR_AET_arr[:,3])
    SNR_std = np.std(SNR_AET_arr[:,0])
    SNR_A_std = np.std(SNR_AET_arr[:,1])
    SNR_E_std = np.std(SNR_AET_arr[:,2])
    SNR_T_std = np.std(SNR_AET_arr[:,3])
    if not return_std:
        return np.array([SNR_av, SNR_A_av, SNR_E_av, SNR_T_av])
    else:
        return np.array([SNR_av, SNR_A_av, SNR_E_av, SNR_T_av]), np.array([SNR_std, SNR_A_std, SNR_E_std, SNR_T_std])

def LISASNR_average_angles(M, q, chi1, chi2, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False):
    SNR_arr = np.zeros(N)
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        SNR_arr[i] = LISASNR(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    SNR_av = np.mean(SNR_arr)
    SNR_std = np.std(SNR_arr)
    if not return_std:
        return SNR_av
    else:
        return SNR_av, SNR_std

def LISASNR_AET_average_angles_spin(M, q, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False):
    SNR_AET_arr = np.zeros((N,4))
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        chi1 = np.random.uniform(low=-1., high=1.)
        chi2 = np.random.uniform(low=-1., high=1.)
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        SNR_AET_arr[i] = LISASNR_AET(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    SNR_av = np.mean(SNR_AET_arr[:,0])
    SNR_A_av = np.mean(SNR_AET_arr[:,1])
    SNR_E_av = np.mean(SNR_AET_arr[:,2])
    SNR_T_av = np.mean(SNR_AET_arr[:,3])
    SNR_std = np.std(SNR_AET_arr[:,0])
    SNR_A_std = np.std(SNR_AET_arr[:,1])
    SNR_E_std = np.std(SNR_AET_arr[:,2])
    SNR_T_std = np.std(SNR_AET_arr[:,3])
    if not return_std:
        return np.array([SNR_av, SNR_A_av, SNR_E_av, SNR_T_av])
    else:
        return np.array([SNR_av, SNR_A_av, SNR_E_av, SNR_T_av]), np.array([SNR_std, SNR_A_std, SNR_E_std, SNR_T_std])

def LISASNR_average_angles_spin(M, q, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False):
    SNR_arr = np.zeros(N)
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        chi1 = np.random.uniform(low=-1., high=1.)
        chi2 = np.random.uniform(low=-1., high=1.)
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        SNR_arr[i] = LISASNR(M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    SNR_av = np.mean(SNR_arr)
    SNR_std = np.std(SNR_arr)
    if not return_std:
        return SNR_av
    else:
        return SNR_av, SNR_std

################################################################################
# Time to merger

def LISAtimetomergerofSNR(SNR, M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=5., minf=1e-5, maxf=1., t0=0.,  fRef=0., npts=4000, variant='LISAproposal'):
    zval = z
    Mval = M
    qval = q

    # Masses
    m1 = Mval*qval/(1.+qval)
    m2 = Mval*1/(1.+qval)

    dist = cosmo.luminosity_distance(zval).value

    wftdi_rescaled = pyFDresponse.LISAGenerateTDI(phi, fRef, m1, m2, chi1, chi2, dist, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0, settRefAtfRef=False, tRef=0., TDItag='TDIAET', order_fresnel_stencil=0, nptmin=100, rescaled=True)

    tf = 1./(2*np.pi)*gwtools.spline(wftdi_rescaled['freq'], wftdi_rescaled['phase'])(wftdi_rescaled['freq'], 1)

    cumul_SNR = LISAComputeSNR_TDIAET(wftdi_rescaled, df=None, cumul=True, npts=npts, variant=variant)[0]

    # Cut freq at first max of tf
    if not np.any(np.diff(tf) <= 0):
        ilast_tf = len(tf) - 1
    else:
        ilast_tf = np.where(np.logical_not(np.diff(tf) > 0))[0][0]
    last_tf = tf[ilast_tf]
    margin = 1. # Margin for representing ln(tflast - tf + margin)

    # Detectability threshold
    if not np.any(cumul_SNR > SNR):
        #print('Warning: cumul_SNR does not reach threshold.')
        tthreshold = np.nan
    else:
        if not np.any(cumul_SNR < SNR):
            #print('Warning: cumul_SNR exceeds threshold at first point ?')
            ithreshold = 0

        else:
            ithreshold = np.where(cumul_SNR < SNR)[0][-1]
        tthreshold = last_tf - tf[ithreshold] + margin

    return tthreshold

def LISAtimetomergerofSNR_average_angles(SNR, M, q, chi1, chi2, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False, ignore_nan=False):
    tSNR_arr = np.zeros(N)
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        tSNR_arr[i] = LISAtimetomergerofSNR(SNR, M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    if ignore_nan:
        mask = np.logical_not(np.isnan(tSNR_arr))
        tSNR_arr = tSNR_arr[mask]
        if len(tSNR_arr)==0:
            return np.nan, np.nan
    tSNR_av = np.mean(tSNR_arr)
    tSNR_std = np.std(tSNR_arr)
    if not return_std:
        return tSNR_av
    else:
        return tSNR_av, tSNR_std

def LISAtimetomergerofSNR_average_angles_spin(SNR, M, q, z, N=1000, tobs=5., minf=1e-5, maxf=1., t0=0., fRef=0., npts=10000, variant='LISAproposal', return_std=False, ignore_nan=False):
    tSNR_arr = np.zeros(N)
    for i in range(N):
        phi, inc, lambd, beta, psi = draw_random_angles()
        chi1 = np.random.uniform(low=-1., high=1.)
        chi2 = np.random.uniform(low=-1., high=1.)
        if t0=='av':
            t0val = np.random.uniform(low=0., high=1.)
        else:
            t0val = t0
        tSNR_arr[i] = LISAtimetomergerofSNR(SNR, M, q, chi1, chi2, z, phi, inc, lambd, beta, psi, tobs=tobs, minf=minf, maxf=maxf, t0=t0val, fRef=fRef, npts=npts, variant=variant)
    if ignore_nan:
        mask = np.logical_not(np.isnan(tSNR_arr))
        tSNR_arr = tSNR_arr[mask]
        if len(tSNR_arr)==0:
            return np.nan, np.nan
    tSNR_av = np.mean(tSNR_arr)
    tSNR_std = np.std(tSNR_arr)
    if not return_std:
        return tSNR_av
    else:
        return tSNR_av, tSNR_std
