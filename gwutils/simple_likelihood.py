import sys
import re
import time
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
import matplotlib.pyplot as plt
from matplotlib import cm

# Solar mass in s
msols = 4.925491025543575903411922162094833998e-6
# Max geometric frequency for the 22 mode of the ROM EOBNRv2HM
MfROMmax22 = 0.14

# Mod 2pi - returns a result in ]-pi, pi]
def mod2pi(x):
    rem = np.remainder(x, 2*pi)
    if rem<=pi:
        return rem
    else:
        return rem - 2*pi

# Functions to convert to LISA-frame quantities - plus phiL, which has a special definition
# fullparams in the format of posterior files : [m1, m2, tRef, DL, phiRef, inc, lambda, beta, pol, loglike]
# Impose ranges phiL in [-pi, pi], psiL in [0, pi], lambdaL in [-pi, pi]
def funcphiL_pars(postparams):
    m1, m2, t, phi = postparams[[0,1,2,4]]
    return gwtools.funcphiL(m1, m2, t, phi)
def funclambdaL_pars(postparams):
    lambd, beta = postparams[[6,7]]
    return gwtools.funclambdaL(lambd, beta)
def funcbetaL_pars(postparams):
    lambd, beta = postparams[[6,7]]
    return gwtools.funcbetaL(lambd, beta)
def funcpsiL_pars(postparams):
    lambd, beta, psi = postparams[[6,7,8]]
    return gwtools.funcpsiL(lambd, beta, psi)
# NOTE: this python file was written with old conventions, different from the paper conventions
def funcconvertparamsL(postparams, injpostparams):
    phiL = funcphiL_pars(postparams)
    lambdL = funclambdaL_pars(postparams, defLframe='old')
    betaL = funcbetaL_pars(postparams)
    psiL = funcpsiL_pars(postparams)
    inc = postparams[5]
    d = postparams[3] / injpostparams[3]
    return array([d, phiL, inc, lambdL, betaL, psiL])
def funcconvertallparamsL(postparams):
    phiL = funcphiL_pars(postparams)
    lambdL = funclambdaL_pars(postparams, defLframe='old')
    betaL = funcbetaL_pars(postparams)
    psiL = funcpsiL_pars(postparams)
    [m1, m2, tRef, DL] = postparams[0:4]
    inc = postparams[5]
    logL = postparams[9]
    return array([m1, m2, tRef, DL, phiL, inc, lambdL, betaL, psiL, logL])

# Format : pars numpy array in the form [d, phiL, inc, lambdaL, betaL, psiL]
# Here d is DL/DLinj, distance scale relative to the injection
# The angles lambdaL, betaL, psiL have a LISA-frame meaning here - and phiL is the quasi-orbital phase shift defined for the L-frame as well (that is, assuming tL is common to all waveforms, not tSSB)
# factor = 4 int((pi f L/c)^(2) |h22|^2) - to be given as input, and gives the SNR scale
# sa, se will be precomputed for the injection
def funcDaplus(lambd, beta):
    return 1j*3./4 * (3 - cos(2*beta)) * cos(2*lambd - pi/3)
def funcDacross(lambd, beta):
    return 1j*3*sin(beta) * sin(2*lambd - pi/3)
def funcDeplus(lambd, beta):
    return -1j*3./4 * (3 - cos(2*beta)) * sin(2*lambd - pi/3)
def funcDecross(lambd, beta):
    return 1j*3*sin(beta) * cos(2*lambd - pi/3)
def funcsa(params):
    [d, phi, inc, lambd, beta, psi] = params
    Daplus = funcDaplus(lambd, beta)
    Dacross = funcDacross(lambd, beta)
    a22 = 1./d*1./2 * sqrt(5/pi) * cos(inc/2)**4 * exp(2.*1j*(-phi-psi)) * 1./2*(Daplus + 1j*Dacross)
    a2m2 = 1./d*1./2 * sqrt(5/pi) * sin(inc/2)**4 * exp(2.*1j*(-phi+psi)) * 1./2*(Daplus - 1j*Dacross)
    return a22 + a2m2
def funcse(params):
    [d, phi, inc, lambd, beta, psi] = params
    Deplus = funcDeplus(lambd, beta)
    Decross = funcDecross(lambd, beta)
    e22 = 1./d*1./2 * sqrt(5/pi) * cos(inc/2)**4 * exp(2.*1j*(-phi-psi)) * 1./2*(Deplus + 1j*Decross)
    e2m2 = 1./d*1./2 * sqrt(5/pi) * sin(inc/2)**4 * exp(2.*1j*(-phi+psi)) * 1./2*(Deplus - 1j*Decross)
    return e22 + e2m2
def simple_likelihood_22(pars, factor, sainj, seinj):
    return -1./2 * factor * (abs(funcsa(pars) - sainj)**2 + abs(funcse(pars) - seinj)**2)

# Value computed in Mathematica for runcan at snr 200
factor = 216147.866077

# Load posterior, sort and extract injection params
# post = np.loadtxt('bambi_22_runcan_tLpinmtfrznlf_post_equal_weights.dat')
# post = post[post[:,-1].argsort()]
# injparams = post[-1]
# injparamsL = funcconvertparamsL(injparams, injparams)
# sainj, seinj = funcsa(injparamsL), funcse(injparamsL)
# postL = array(map(lambda x: funcconvertparamsL(x, injparams), post))
# likelihoodvals = map(lambda x: simple_likelihood_22(x, factor, sainj, seinj), postL)
#
# simplelikepost = np.loadtxt('bambi_22_runcan_simplelike_post_equal_weights.dat')
# simplelikepostallL = array(map(lambda x: funcconvertallparamsL(x), simplelikepost))
# np.savetxt('bambi_22_runcan_simplelikeL_post_equal_weights.dat', simplelikepostallL)
