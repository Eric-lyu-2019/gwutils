from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


import numpy as np
from numpy import pi, conjugate, dot, sqrt, cos, sin, tan, exp, real, imag, arccos, arcsin, arctan, arctan2
import scipy
import scipy.interpolate as ip
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import gwutils.gwtools as gwtools
from gwutils.gwtools import R, c, Omega, msols


################################################################################
# Functions for the L-frame conversion for LISA
################################################################################

# Conversions between SSB-frame and L-frame parameters (for initial position alpha=0)
def functLfromtSSB(tSSB, lambd, beta):
    return tSSB - R/c*cos(beta)*cos(Omega*tSSB - lambd)
def functSSBfromtL(tL, lambd, beta):
    return tL + R/c*cos(beta)*cos(Omega*tL - lambd) - 1./2*Omega*(R/c*cos(beta))**2*sin(2.*(Omega*tL - lambd))
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure observer phase shift, equivalent to having SetphiRefSSBAtfRef=False AND flipping the sign
# So phiL = -phiSSB(SetphiRefSSBAtfRef=False)
# We set phiL' = -phiSSB(SetphiRefSSBAtfRef=True) + pi*tSSB*fRef to remove the tRef-dependence
# NOTE: phiL != phiL'
# the two still differ by fRef-dependent terms that require access to the waveform to evaluate
# NOTE: phiL' only supports fRef set at its default, the maximum 22 frequency covered by the ROM
def funcphiL(m1, m2, tSSB, phiSSB, SetphiRefSSBAtfRef=False): # note: mod to [0,2pi]
    if not SetphiRefSSBAtfRef:
        return -phiSSB
    else:
        MfROMmax22 = 0.14
        fRef = MfROMmax22/((m1 + m2)*msols)
        return gwtools.mod2pi(-phiSSB + pi*tSSB*fRef)
# Inverse transformation of the phase
# NOTE: we take tSSB as an argument, not tL - because computing tSSB requires the sky position as well
def funcphiSSB(m1, m2, tSSB, phiL, SetphiRefSSBAtfRef=False): # note: mod to [0,2pi]
    if not SetphiRefSSBAtfRef:
        return -phiL
    else:
        MfROMmax22 = 0.14
        fRef = MfROMmax22/((m1 + m2)*msols)
        return gwtools.mod2pi(-phiL + pi*tSSB*fRef)
# NOTE: simple relation between L-frame definitions
# lambdaL_old = lambdaL_paper - pi/2
def funclambdaL(lambd, beta, defLframe='paper'):
    if defLframe=='paper':
        return arctan2(cos(beta)*sin(lambd), cos(beta)*cos(lambd)*cos(pi/3) + sin(beta)*sin(pi/3))
    elif defLframe=='old':
        return -arctan2(cos(beta)*cos(lambd)*cos(pi/3) + sin(beta)*sin(pi/3), cos(beta)*sin(lambd))
    else:
        raise ValueError('Value %s for defLframe is not recognized.' % defLframe)
def funcbetaL(lambd, beta):
    return -arcsin(cos(beta)*cos(lambd)*sin(pi/3) - sin(beta)*cos(pi/3))
# NOTE: old equivalent writing
# modpi(arctan2(cos(pi/3)*cos(beta)*sin(psi) - sin(pi/3)*(sin(lambd)*cos(psi) - cos(lambd)*sin(beta)*sin(psi)), cos(pi/3)*cos(beta)*cos(psi) + sin(pi/3)*(sin(lambd)*sin(psi) + cos(lambd)*sin(beta)*cos(psi))))
def funcpsiL(lambd, beta, psi): # note: mod to [0,pi]
    return gwtools.modpi(psi + arctan2(-sin(pi/3)*sin(lambd), cos(pi/3)*cos(beta) + sin(pi/3)*cos(lambd)*sin(beta)))

# Copy of C functions for translation between frames
# We modify constellation variant, keeping only the initial constellation phase : Omega is fixed, constellation_ini_phase replaces variant->ConstPhi0
# NOTE: C function for time duplicate python functions functLfromtSSB and functSSBfromtL
# Compute Solar System Barycenter time tSSB from retarded time at the center of the LISA constellation tL
# NOTE: depends on the sky position given in SSB parameters
def tSSBfromLframe(tL, lambdaSSB, betaSSB, constellation_ini_phase=0.):
    phase = Omega*tL + constellation_ini_phase - lambdaSSB
    RoC = R/c
    return tL + RoC*cos(betaSSB)*cos(phase) - 1./2*Omega*pow(RoC*cos(betaSSB), 2)*sin(2.*phase);
# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, constellation_ini_phase=0.):
    phase = Omega*tSSB + constellation_ini_phase - lambdaSSB
    RoC = R/c
    return tSSB - RoC*cos(betaSSB)*cos(phase)
# Convert L-frame params to SSB-frame params
# NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
def ConvertLframeParamsToSSBframe(
    tL,
    lambdaL,
    betaL,
    psiL,
    constellation_ini_phase=0.):

    alpha = 0.; cosalpha = 0; sinalpha = 0.; coslambdaL = 0; sinlambdaL = 0.; cosbetaL = 0.; sinbetaL = 0.; cospsiL = 0.; sinpsiL = 0.;
    coszeta = cos(pi/3.)
    sinzeta = sin(pi/3.)
    coslambdaL = cos(lambdaL)
    sinlambdaL = sin(lambdaL)
    cosbetaL = cos(betaL)
    sinbetaL = sin(betaL)
    cospsiL = cos(psiL)
    sinpsiL = sin(psiL)
    lambdaSSB_approx = 0.
    betaSSB_approx = 0.
    # Initially, approximate alpha using tL instead of tSSB - then iterate
    tSSB_approx = tL
    for k in range(3):
        alpha = Omega * (tSSB_approx) + constellation_ini_phase
        cosalpha = cos(alpha)
        sinalpha = sin(alpha)
        lambdaSSB_approx = arctan2(cosalpha*cosalpha*cosbetaL*sinlambdaL -sinalpha*sinbetaL*sinzeta + cosbetaL*coszeta*sinalpha*sinalpha*sinlambdaL -cosalpha*cosbetaL*coslambdaL*sinalpha + cosalpha*cosbetaL*coszeta*coslambdaL*sinalpha, cosbetaL*coslambdaL*sinalpha*sinalpha -cosalpha*sinbetaL*sinzeta + cosalpha*cosalpha*cosbetaL*coszeta*coslambdaL -cosalpha*cosbetaL*sinalpha*sinlambdaL + cosalpha*cosbetaL*coszeta*sinalpha*sinlambdaL)
        betaSSB_approx = arcsin(coszeta*sinbetaL + cosalpha*cosbetaL*coslambdaL*sinzeta + cosbetaL*sinalpha*sinzeta*sinlambdaL)
        tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, constellation_ini_phase=constellation_ini_phase)
    tSSB = tSSB_approx
    lambdaSSB = lambdaSSB_approx
    betaSSB = betaSSB_approx
    # Polarization
    psiSSB = gwtools.modpi(psiL + arctan2(cosalpha*sinzeta*sinlambdaL -coslambdaL*sinalpha*sinzeta, cosbetaL*coszeta -cosalpha*coslambdaL*sinbetaL*sinzeta -sinalpha*sinbetaL*sinzeta*sinlambdaL))
    return [tSSB, lambdaSSB, betaSSB, psiSSB]
# Convert SSB-frame params to L-frame params
# NOTE: no transformation of the phase -- approximant-dependence with e.g. EOBNRv2HMROM setting phiRef at fRef, and freedom in definition
def ConvertSSBframeParamsToLframe(
    tSSB,
    lambdaSSB,
    betaSSB,
    psiSSB,
    constellation_ini_phase=0.):

    alpha = 0.; cosalpha = 0; sinalpha = 0.; coslambda = 0; sinlambda = 0.; cosbeta = 0.; sinbeta = 0.; cospsi = 0.; sinpsi = 0.;
    coszeta = cos(pi/3.)
    sinzeta = sin(pi/3.)
    coslambda = cos(lambdaSSB)
    sinlambda = sin(lambdaSSB)
    cosbeta = cos(betaSSB)
    sinbeta = sin(betaSSB)
    cospsi = cos(psiSSB)
    sinpsi = sin(psiSSB)
    alpha = Omega * tSSB + constellation_ini_phase
    cosalpha = cos(alpha)
    sinalpha = sin(alpha)
    tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, constellation_ini_phase=constellation_ini_phase)
    lambdaL = arctan2(cosalpha*cosalpha*cosbeta*sinlambda + sinalpha*sinbeta*sinzeta + cosbeta*coszeta*sinalpha*sinalpha*sinlambda -cosalpha*cosbeta*coslambda*sinalpha + cosalpha*cosbeta*coszeta*coslambda*sinalpha, cosalpha*sinbeta*sinzeta + cosbeta*coslambda*sinalpha*sinalpha + cosalpha*cosalpha*cosbeta*coszeta*coslambda -cosalpha*cosbeta*sinalpha*sinlambda + cosalpha*cosbeta*coszeta*sinalpha*sinlambda)
    betaL = arcsin(coszeta*sinbeta -cosalpha*cosbeta*coslambda*sinzeta -cosbeta*sinalpha*sinzeta*sinlambda)
    psiL = gwtools.modpi(psiSSB + arctan2(coslambda*sinalpha*sinzeta -cosalpha*sinzeta*sinlambda, cosbeta*coszeta + cosalpha*coslambda*sinbeta*sinzeta + sinalpha*sinbeta*sinzeta*sinlambda))
    return [tL, lambdaL, betaL, psiL]

################################################################################
# Derivatives and Jacobian in the SSB-frame L-frame parameter map
################################################################################

# Derivatives of L-frame parameters with respect to SSB-frame parameters
# tL
def funcdtLdtSSB(tSSB, lambd, beta):
    return 1 + R/c * cos(beta) * Omega*sin(Omega*tSSB - lambd)
def funcdtLdlambda(tSSB, lambd, beta):
    return -R/c * cos(beta) * sin(Omega*tSSB - lambd)
def funcdtLdbeta(tSSB, lambd, beta):
    return R/c * sin(beta) * cos(Omega*tSSB - lambd)
# phiL
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure observer phase phase shift, equivalent to having SetphiRefSSBAtfRef=False AND flipping the sign
# So phiL = -phiSSB(SetphiRefSSBAtfRef=False)
def funcdphiLdm1(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
    if SetphiRefSSBAtfRef:
        MfROMmax22 = 0.14
        dfRefdm1 = -MfROMmax22/((m1 + m2)**2 * msols)
        return pi*t*dfRefdm1
    else:
        return 0.
def funcdphiLdm2(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
    if SetphiRefSSBAtfRef:
        MfROMmax22 = 0.14
        dfRefdm2 = -MfROMmax22/((m1 + m2)**2 * msols)
        return pi*t*dfRefdm2
    else:
        return 0.
def funcdphiLdt(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
    if SetphiRefSSBAtfRef:
        MfROMmax22 = 0.14
        fRef = MfROMmax22/((m1 + m2)*msols)
        return pi*fRef
    else:
        return 0.
def funcdphiLdphi(m1, m2, t, phi, SetphiRefSSBAtfRef=False):
    return -1
# lambdaL
# NOTE: derivative identical for different L-frame conventions that differ by a constant shift in lambdaL
def funcdlambdaLdlambda(lambd, beta):
    return (cos(beta)*(cos(beta)*cos(pi/3) + cos(lambd)*sin(beta)*sin(pi/3))) / ((cos(beta)*cos(pi/3)*cos(lambd) + sin(beta)*sin(pi/3))**2 +
 cos(beta)**2 * sin(lambd)**2)
def funcdlambdaLdbeta(lambd, beta):
    return -((sin(pi/3)*sin(lambd))/((cos(beta)*cos(pi/3)*cos(lambd) + sin(beta)*sin(pi/3))**2 +
  cos(beta)**2*sin(lambd)**2))
# betaL
def funcdbetaLdlambda(lambd, beta):
    return cos(beta)*sin(pi/3)*sin(lambd) / np.sqrt(1 - (cos(pi/3)*sin(beta) - cos(beta)*cos(lambd)*sin(pi/3))**2)
def funcdbetaLdbeta(lambd, beta):
    return (cos(beta)*cos(pi/3) + cos(lambd)*sin(beta)*sin(pi/3)) / np.sqrt(1 - (cos(pi/3)*sin(beta) - cos(beta)*cos(lambd)*sin(pi/3))**2)
# psiL
def funcdpsiLdlambda(lambd, beta, psi):
    return -((sin(pi/3)*(cos(beta)*cos(pi/3)*cos(lambd) + sin(beta)*sin(pi/3))) / ((cos(beta)*cos(pi/3) + cos(lambd)*sin(beta)*sin(pi/3))**2 + sin(pi/3)**2 * sin(lambd)**2))
def funcdpsiLdbeta(lambd, beta, psi):
    return (sin(pi/3)*(-cos(pi/3)*sin(beta) + cos(beta)*cos(lambd)*sin(pi/3))*sin(lambd)) / ((cos(beta)*cos(pi/3) + cos(lambd)*sin(beta)*sin(pi/3))**2 +
 sin(pi/3)**2 * sin(lambd)**2)
def funcdpsiLdpsi(lambd, beta, psi):
    return 1.
# Jacobian matrix of the SSB-Lframe transformation
# Required to convert Fisher matrices computed with SSB params to L-frame params
# Parameters order : m1 m2 t D phi inc lambda beta psi
# Matrix Jij = \partial xLi / \partial xj
# Option SetphiRefSSBAtfRef indicates the meaning of phase in the input SSB frame
# L-frame phase is always a pure phase shift, like having SetphiRefSSBAtfRef=False
def funcJacobianSSBtoLframe(params, SetphiRefSSBAtfRef=False):
    m1, m2, t, D, phi, inc, lambd, beta, psi = params
    J = np.zeros((9,9), dtype=float)
    # m1
    J[0,0] = 1.
    # m2
    J[1,1] = 1.
    # tL
    J[2,2] = funcdtLdtSSB(t, lambd, beta)
    J[2,6] = funcdtLdlambda(t, lambd, beta)
    J[2,7] = funcdtLdbeta(t, lambd, beta)
    # D
    J[3,3] = 1.
    # phi - uses
    J[4,0] = funcdphiLdm1(m1, m2, t, phi, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    J[4,1] = funcdphiLdm2(m1, m2, t, phi, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    J[4,2] = funcdphiLdt(m1, m2, t, phi, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    J[4,4] = funcdphiLdphi(m1, m2, t, phi, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    # inc
    J[5,5] = 1.
    # lambdaL
    J[6,6] = funcdlambdaLdlambda(lambd, beta)
    J[6,7] = funcdlambdaLdbeta(lambd, beta)
    # betaL
    J[7,6] = funcdbetaLdlambda(lambd, beta)
    J[7,7] = funcdbetaLdbeta(lambd, beta)
    # psiL
    J[8,6] = funcdpsiLdlambda(lambd, beta, psi)
    J[8,7] = funcdpsiLdbeta(lambd, beta, psi)
    J[8,8] = funcdpsiLdpsi(lambd, beta, psi)
    return J

################################################################################
# Simplified likelihood
################################################################################

# From simple_likelihood.py, modified to follow notations of the paper
# Format : pars numpy array in the form [d, phiL, inc, lambdaL, betaL, psiL]
# Here d is DL/DLinj, distance scale relative to the injection
# The angles lambdaL, betaL, psiL have a LISA-frame meaning here - and phiL is the quasi-orbital phase shift defined for the L-frame as well (that is, assuming tL is common to all waveforms, not tSSB)
# factor = 4 int((pi f L/c)^(2) |h22|^2) - to be given as input, and gives the SNR scale
# sa, se will be precomputed for the injection
def func_Faplus(lambd, beta):
    return 1./2 * (1 + sin(beta)**2) * cos(2*lambd - pi/3)
def func_Facross(lambd, beta):
    return sin(beta) * sin(2*lambd - pi/3)
def func_Feplus(lambd, beta):
    return 1./2 * (1 + sin(beta)**2) * cos(2*lambd + pi/6)
def func_Fecross(lambd, beta):
    return sin(beta) * sin(2*lambd + pi/6)
def func_sa(params):
    [d, phi, inc, lambd, beta, psi] = params
    Faplus = func_Faplus(lambd, beta)
    Facross = func_Facross(lambd, beta)
    a22 = 1./4/d * sqrt(5/pi) * cos(inc/2)**4 * exp(2.*1j*(phi-psi)) * 1./2*(Faplus + 1j*Facross)
    a2m2 = 1./4/d * sqrt(5/pi) * sin(inc/2)**4 * exp(2.*1j*(phi+psi)) * 1./2*(Faplus - 1j*Facross)
    return a22 + a2m2
def func_se(params):
    [d, phi, inc, lambd, beta, psi] = params
    Feplus = func_Feplus(lambd, beta)
    Fecross = func_Fecross(lambd, beta)
    e22 = 1./4/d * sqrt(5/pi) * cos(inc/2)**4 * exp(2.*1j*(phi-psi)) * 1./2*(Feplus + 1j*Fecross)
    e2m2 = 1./4/d * sqrt(5/pi) * sin(inc/2)**4 * exp(2.*1j*(phi+psi)) * 1./2*(Feplus - 1j*Fecross)
    return e22 + e2m2
def simple_likelihood_22(pars, factor, sainj, seinj):
    return -1./2 * factor * (abs(func_sa(pars) - sainj)**2 + abs(func_se(pars) - seinj)**2)

# Degenerate sky positions - see PE paper
# Because we take the ratio sigma_+/sigma_-, d and phi are silent
def func_degen_sky(inc, lambd, beta, psi):
    phi_dummy = 0.
    d_dummy = 1.
    params = [d_dummy, phi_dummy, inc, lambd, beta, psi]
    sa = func_sa(params)
    se = func_se(params)
    sigma_plus = sa + 1j*se
    sigma_minus = sa - 1j*se
    r = sigma_plus / sigma_minus
    lambdaL_star = pi/6. - 1./4 * np.angle(r)
    betaL_star = pi/2. - 2*np.arctan((np.abs(r))**(1./4))
    return (lambdaL_star, betaL_star)
# Degenerate parameters - see PE paper
# Looks for building degenerate point at 0 inclination and 0 phase
# Gives value of all [d, phi, inc, lambd, beta, psi]
def func_degen_params_0inc_0phase(d, phi, inc, lambd, beta, psi):
    params = [d, phi, inc, lambd, beta, psi]
    sa = func_sa(params)
    se = func_se(params)
    sigma_plus = sa + 1j*se
    sigma_minus = sa - 1j*se
    r = sigma_plus / sigma_minus
    # Inclination, phase -- by convention
    inc_star = 0. # we choose to look for face-on point
    phi_star = 0. # we choose to look for zero-phase (exact degen with psi)
    # Sky
    lambdaL_star = gwtools.mod2pi(pi/6. - 1./4 * np.angle(r))
    betaL_star = pi/2. - 2*np.arctan((np.abs(r))**(1./4))
    # Distance
    thetaL_star = np.pi/2. - betaL_star
    d_star = 1./4*np.sqrt(5./np.pi) * 1./(1. + np.tan(thetaL_star/2.)**2)**2 / np.abs(sigma_minus)
    # Polarization, computed here for phi_star = 0
    psiL_star = gwtools.modpi(-1./2 * np.angle(sigma_minus) + lambdaL_star + np.pi/6 + phi_star) # defined mod pi
    return (d_star, phi_star, inc_star, lambdaL_star, betaL_star, psiL_star)
