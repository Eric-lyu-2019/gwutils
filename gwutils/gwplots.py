## Tools for plotting

from __future__ import absolute_import, division, print_function
import sys
if sys.version_info[0] == 2:
    from future_builtins import map, filter


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
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import gwutils.gwtools as gwtools
import gwutils.lisatools as lisatools
import gwutils.corner_covar as corner_covar


####################################
## Functions to plot single waveform
####################################

# NOTE: for phases and omega, the convention in the code is philm=Arg(hlm) - for the plots, we change the convention to philm=-Arg(hlm) to make the connection with hlm=Alm e^(-i philm) with increasing philm

# Plot Inertial-frame amplitudes
def plot_wf_AIlm(wf, interval=[-400,150], exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a4 = plt.subplot2grid((4,3), (0,0))
    a5 = plt.subplot2grid((4,3), (0,1))
    a6 = plt.subplot2grid((4,3), (0,2))
    a1 = plt.subplot2grid((4,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((4,3), (2,0), colspan=3)
    a3 = plt.subplot2grid((4,3), (3,0), colspan=3)
    [ti, tf] = [wf['tI'][0], wf['tI'][-1]]
    # AIlm for the full waveform
    a1.plot(wf['tI'], wf['AIlm'][(2,2)], 'b', label=r'$A^{\rm I}_{22}$')
    a1.plot(wf['tI'], wf['AIlm'][(2,-2)], 'r', label=r'$A^{\rm I}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tI'], wf['AIlm'][(2,1)], 'b', label=r'$A^{\rm I}_{21}$')
    a2.plot(wf['tI'], wf['AIlm'][(2,-1)], 'r', label=r'$A^{\rm I}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    a3.plot(wf['tI'], wf['AIlm'][(2,0)], 'b', label=r'$A^{\rm I}_{20}$')
    a3.set_xlim([ti, tf])
    a3.axvline(x=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Zooming at the end
    modes = [(2,2), (2,1), (2,0), (2,-1), (2,-2)]
    AIlmrestr = {}
    for lm in modes:
        AIlmrestr[lm] = gwtools.restrict_data(np.array([wf['tI'], wf['AIlm'][lm]]).T, interval)[:,1]
    tIrestr = gwtools.restrict_data(np.array([wf['tI'], wf['AIlm'][(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tIrestr[0], tIrestr[-1]]
    a4.plot(tIrestr, AIlmrestr[(2,2)], 'b', label=r'$A^{\rm I}_{22}$')
    a4.plot(tIrestr, AIlmrestr[(2,-2)], 'r', label=r'$A^{\rm I}_{2-2}$')
    a4.set_xlim([tir, tfr])
    a4.axvline(x=0, color='k')
    a4.set_xlabel(r'$t/M$')
    a5.plot(tIrestr, AIlmrestr[(2,1)], 'b', label=r'$A^{\rm I}_{21}$')
    a5.plot(tIrestr, AIlmrestr[(2,-1)], 'r', label=r'$A^{\rm I}_{2-1}$')
    a5.set_xlim([tir, tfr])
    a5.axvline(x=0, color='k')
    a5.set_xlabel(r'$t/M$')
    a6.plot(tIrestr, AIlmrestr[(2,0)], 'b', label=r'$A^{\rm I}_{20}$')
    a6.set_xlim([tir, tfr])
    a6.axvline(x=0, color='k')
    a6.set_xlabel(r'$t/M$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a4.grid()
    a5.grid()
    a6.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    a4.legend(loc='upper left')
    a5.legend(loc='upper left')
    a6.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot Inertial-frame phases
def plot_wf_phiIlm(wf, interval=[-400,150], exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a4 = plt.subplot2grid((4,3), (0,0))
    a5 = plt.subplot2grid((4,3), (0,1))
    a6 = plt.subplot2grid((4,3), (0,2))
    a1 = plt.subplot2grid((4,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((4,3), (2,0), colspan=3)
    a3 = plt.subplot2grid((4,3), (3,0), colspan=3)
    [ti, tf] = [wf['tI'][0], wf['tI'][-1]]
    # phiIlm for the full waveform
    a1.plot(wf['tI'], -wf['phiIlm'][(2,2)], 'b', label=r'$\phi^{\rm I}_{22}$')
    a1.plot(wf['tI'], wf['phiIlm'][(2,-2)], 'r', label=r'$-\phi^{\rm I}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tI'], -wf['phiIlm'][(2,1)], 'b', label=r'$\phi^{\rm I}_{21}$')
    a2.plot(wf['tI'], wf['phiIlm'][(2,-1)], 'r', label=r'$-\phi^{\rm I}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    a3.plot(wf['tI'], -wf['phiIlm'][(2,0)], 'b', label=r'$\phi^{\rm I}_{20}$')
    a3.set_xlim([ti, tf])
    a3.axvline(x=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Zooming at the end
    modes = [(2,2), (2,1), (2,0), (2,-1), (2,-2)]
    phiIlmrestr = {}
    for lm in modes:
        phiIlmrestr[lm] = gwtools.restrict_data(np.array([wf['tI'], wf['phiIlm'][lm]]).T, interval)[:,1]
    tIrestr = gwtools.restrict_data(np.array([wf['tI'], wf['phiIlm'][(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tIrestr[0], tIrestr[-1]]
    a4.plot(tIrestr, -phiIlmrestr[(2,2)], 'b', label=r'$\phi^{\rm I}_{22}$')
    a4.plot(tIrestr, phiIlmrestr[(2,-2)], 'r', label=r'$-\phi^{\rm I}_{2-2}$')
    a4.set_xlim([tir, tfr])
    a4.axvline(x=0, color='k')
    a4.set_xlabel(r'$t/M$')
    a5.plot(tIrestr, -phiIlmrestr[(2,1)], 'b', label=r'$\phi^{\rm I}_{21}$')
    a5.plot(tIrestr, phiIlmrestr[(2,-1)], 'r', label=r'$-\phi^{\rm I}_{2-1}$')
    a5.set_xlim([tir, tfr])
    a5.axvline(x=0, color='k')
    a5.set_xlabel(r'$t/M$')
    a6.plot(tIrestr, -phiIlmrestr[(2,0)], 'b', label=r'$\phi^{\rm I}_{20}$')
    a6.set_xlim([tir, tfr])
    a6.axvline(x=0, color='k')
    a6.set_xlabel(r'$t/M$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a4.grid()
    a5.grid()
    a6.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    a4.legend(loc='upper left')
    a5.legend(loc='upper left')
    a6.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot Precessing-frame amplitudes
def plot_wf_APlm(wf, interval=[-400,150], exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a4 = plt.subplot2grid((4,3), (0,0))
    a5 = plt.subplot2grid((4,3), (0,1))
    a6 = plt.subplot2grid((4,3), (0,2))
    a1 = plt.subplot2grid((4,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((4,3), (2,0), colspan=3)
    a3 = plt.subplot2grid((4,3), (3,0), colspan=3)
    [ti, tf] = [wf['tP'][0], wf['tP'][-1]]
    # APlm for the full waveform
    a1.plot(wf['tP'], wf['APlm'][(2,2)], 'b', label=r'$A^{\rm P}_{22}$')
    a1.plot(wf['tP'], wf['APlm'][(2,-2)], 'r', label=r'$A^{\rm P}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tP'], wf['APlm'][(2,1)], 'b', label=r'$A^{\rm P}_{21}$')
    a2.plot(wf['tP'], wf['APlm'][(2,-1)], 'r', label=r'$A^{\rm P}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    a3.plot(wf['tP'], wf['APlm'][(2,0)], 'b', label=r'$A^{\rm P}_{20}$')
    a3.set_xlim([ti, tf])
    a3.axvline(x=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Zooming at the end
    modes = [(2,2), (2,1), (2,0), (2,-1), (2,-2)]
    APlmrestr = {}
    for lm in modes:
        APlmrestr[lm] = gwtools.restrict_data(np.array([wf['tP'], wf['APlm'][lm]]).T, interval)[:,1]
    tPrestr = gwtools.restrict_data(np.array([wf['tP'], wf['APlm'][(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tPrestr[0], tPrestr[-1]]
    a4.plot(tPrestr, APlmrestr[(2,2)], 'b', label=r'$A^{\rm P}_{22}$')
    a4.plot(tPrestr, APlmrestr[(2,-2)], 'r', label=r'$A^{\rm P}_{2-2}$')
    a4.set_xlim([tir, tfr])
    a4.axvline(x=0, color='k')
    a4.set_xlabel(r'$t/M$')
    a5.plot(tPrestr, APlmrestr[(2,1)], 'b', label=r'$A^{\rm P}_{21}$')
    a5.plot(tPrestr, APlmrestr[(2,-1)], 'r', label=r'$A^{\rm P}_{2-1}$')
    a5.set_xlim([tir, tfr])
    a5.axvline(x=0, color='k')
    a5.set_xlabel(r'$t/M$')
    a6.plot(tPrestr, APlmrestr[(2,0)], 'b', label=r'$A^{\rm P}_{20}$')
    a6.set_xlim([tir, tfr])
    a6.axvline(x=0, color='k')
    a6.set_xlabel(r'$t/M$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a4.grid()
    a5.grid()
    a6.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    a4.legend(loc='upper left')
    a5.legend(loc='upper left')
    a6.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot Precessing-frame phases
def plot_wf_phiPlm(wf, interval=[-400,150], exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a4 = plt.subplot2grid((4,3), (0,0))
    a5 = plt.subplot2grid((4,3), (0,1))
    a6 = plt.subplot2grid((4,3), (0,2))
    a1 = plt.subplot2grid((4,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((4,3), (2,0), colspan=3)
    a3 = plt.subplot2grid((4,3), (3,0), colspan=3)
    [ti, tf] = [wf['tP'][0], wf['tP'][-1]]
    # phiPlm for the full waveform
    a1.plot(wf['tP'], -wf['phiPlm'][(2,2)], 'b', label=r'$\phi^{\rm P}_{22}$')
    a1.plot(wf['tP'], wf['phiPlm'][(2,-2)], 'r', label=r'$-\phi^{\rm P}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tP'], -wf['phiPlm'][(2,1)], 'b', label=r'$\phi^{\rm P}_{21}$')
    a2.plot(wf['tP'], wf['phiPlm'][(2,-1)], 'r', label=r'$-\phi^{\rm P}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    a3.plot(wf['tP'], -wf['phiPlm'][(2,0)], 'b', label=r'$\phi^{\rm P}_{20}$')
    a3.set_xlim([ti, tf])
    a3.axvline(x=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Zooming at the end
    modes = [(2,2), (2,1), (2,0), (2,-1), (2,-2)]
    phiPlmrestr = {}
    for lm in modes:
        phiPlmrestr[lm] = gwtools.restrict_data(np.array([wf['tP'], wf['phiPlm'][lm]]).T, interval)[:,1]
    tPrestr = gwtools.restrict_data(np.array([wf['tP'], wf['phiPlm'][(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tPrestr[0], tPrestr[-1]]
    a4.plot(tPrestr, -phiPlmrestr[(2,2)], 'b', label=r'$\phi^{\rm P}_{22}$')
    a4.plot(tPrestr, phiPlmrestr[(2,-2)], 'r', label=r'$-\phi^{\rm P}_{2-2}$')
    a4.set_xlim([tir, tfr])
    a4.axvline(x=0, color='k')
    a4.set_xlabel(r'$t/M$')
    a5.plot(tPrestr, -phiPlmrestr[(2,1)], 'b', label=r'$\phi^{\rm P}_{21}$')
    a5.plot(tPrestr, phiPlmrestr[(2,-1)], 'r', label=r'$-\phi^{\rm P}_{2-1}$')
    a5.set_xlim([tir, tfr])
    a5.axvline(x=0, color='k')
    a5.set_xlabel(r'$t/M$')
    a6.plot(tPrestr, -phiPlmrestr[(2,0)], 'b', label=r'$\phi^{\rm P}_{20}$')
    a6.set_xlim([tir, tfr])
    a6.axvline(x=0, color='k')
    a6.set_xlabel(r'$t/M$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a4.grid()
    a5.grid()
    a6.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    a4.legend(loc='upper left')
    a5.legend(loc='upper left')
    a6.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot precessing frame Euler angles
# Note: beta is very noisy at the very end of ringdown - we impose a plotting range based on the value shortly before merger
def plot_wf_euler(wf, interval=[-400,150], showQNM=True, exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a3 = plt.subplot2grid((3,2), (0,0))
    a4 = plt.subplot2grid((3,2), (0,1))
    a1 = plt.subplot2grid((3,2), (1,0), colspan=2)
    a2 = plt.subplot2grid((3,2), (2,0), colspan=2)
    [ti, tf] = [wf['tP'][0], wf['tP'][-1]]
    # Values of alpha, beta close to merger to set plot range
    iclosemerger = gwtools.find_closest_index(wf['tP'], -50)
    betaclosemerger = wf['euler'][iclosemerger, 1]
    talphaclosemerger = 10
    iclosemergeralpha = gwtools.find_closest_index(wf['tP'], talphaclosemerger)
    alphaclosemerger = wf['euler'][iclosemergeralpha, 0]
    # Plotting Euler angles for the full waveform
    a1.plot(wf['tP'], wf['euler'][:,0], 'b', label=r'$\alpha$')
    a1.plot(wf['tP'], -wf['euler'][:,2], 'r', label=r'$-\gamma$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.axhline(y=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tP'], wf['euler'][:,1], 'y', label=r'$\beta$')
    a2.set_xlim([ti, tf])
    a2.set_ylim([0, 2*(betaclosemerger)])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    # Zooming at the end
    tPeulerrestr = gwtools.restrict_data(np.array([wf['tP'], wf['euler'][:,0], wf['euler'][:,1], wf['euler'][:,2]]).T, interval)
    tPrestr = tPeulerrestr[:,0]
    [tir, tfr] = [tPrestr[0], tPrestr[-1]]
    eulerrestr = tPeulerrestr[:,1:]
    # Show the GaTech model for the post-merger rotation of the GW frame around final J
    chif = gwtools.norm(wf['metadata']['chif'])
    if showQNM:
        tpostmerger = np.array(list(filter(lambda x: x>=0, tPrestr)))
        a3.plot(tpostmerger, alphaclosemerger + (gwtools.QNMomegalmnInt[(2,2,0)](chif)-gwtools.QNMomegalmnInt[(2,1,0)](chif))*(tpostmerger-talphaclosemerger), 'k:', label=r'$\omega^{\rm QNM}_{220} - \omega^{\rm QNM}_{210}$')
    minalpharestr = np.min(gwtools.restrict_data(tPeulerrestr, [interval[0], 0])[:,1]) # min value of alpha premerger on the restricted data
    maxalphaestimate = alphaclosemerger + (gwtools.QNMomegalmnInt[(2,2,0)](chif)-gwtools.QNMomegalmnInt[(2,1,0)](chif))*(interval[1]-talphaclosemerger) # estimate max value of alpha to plot from GaTech model
    a3.plot(tPrestr, eulerrestr[:,0], 'b', label=r'$\alpha$')
    a3.plot(tPrestr, -eulerrestr[:,2], 'r', label=r'$-\gamma$')
    a3.set_xlim([tir, tfr])
    a3.set_ylim([minalpharestr-1., maxalphaestimate+1.])
    a3.axvline(x=0, color='k')
    a3.axhline(y=0, color='k')
    a3.set_xlabel(r'$t/M$')
    a4.plot(tPrestr, eulerrestr[:,1], 'y', label=r'$\beta$')
    a4.set_xlim([tir, tfr])
    a4.set_ylim([0, 2*(betaclosemerger)])
    a4.axvline(x=0, color='k')
    a4.set_xlabel(r'$t/M$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a4.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    a4.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot mode-based frequencies omega for the Inertial-frame waveform
def plot_wf_omegaIlm(wf, interval=[-400,150], showQNM=True, exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a3 = plt.subplot2grid((3,3), (0,0), colspan=3)
    a1 = plt.subplot2grid((3,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((3,3), (2,0), colspan=3)
    [ti, tf] = [wf['tI'][0], wf['tI'][-1]]
    # Get QNM frequencies
    if showQNM:
        chif = gwtools.norm(wf['metadata']['chif'])
        omegaQNM22 = gwtools.QNMomegalmnInt[(2,2,0)](chif)
        omegaQNM21 = gwtools.QNMomegalmnInt[(2,1,0)](chif)
    else: # still defining default for omegaQNM22 to set ylimits on the plots
        omegaQNM22 = 0.5 # only a guess
    # Compute omegaIlm
    modes = [(2,2), (2,1), (2,-1), (2,-2)]
    phiIlmint = {}
    omegaIlmint = {}
    omegaIlm = {}
    for lm in modes:
        phiIlmint[lm] = ip.InterpolatedUnivariateSpline(wf['tI'], -wf['phiIlm'][lm], k=3)
        omegaIlmint[lm] = (phiIlmint[lm]).derivative()
        omegaIlm[lm] = omegaIlmint[lm](wf['tI'])
    # omegaIlm for the full waveform
    a1.plot(wf['tI'], omegaIlm[(2,2)], 'b', label=r'$\omega^{\rm I}_{22}$')
    a1.plot(wf['tI'], -omegaIlm[(2,-2)], 'r', label=r'$-\omega^{\rm I}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.set_ylim([0, 1.5*omegaQNM22])
    a1.axhline(y=0, color='k')
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tI'], omegaIlm[(2,1)], 'b', label=r'$\omega^{\rm I}_{21}$')
    a2.plot(wf['tI'], -omegaIlm[(2,-1)], 'r', label=r'$-\omega^{\rm I}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.set_ylim([0, 1.5*omegaQNM21])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    # omegaPlm zooming at the end
    omegaIlmrestr = {}
    for lm in modes:
        omegaIlmrestr[lm] = gwtools.restrict_data(np.array([wf['tI'], omegaIlm[lm]]).T, interval)[:,1]
    tIrestr = gwtools.restrict_data(np.array([wf['tI'], omegaIlm[(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tIrestr[0], tIrestr[-1]]
    a3.plot(tIrestr, omegaIlmrestr[(2,2)], 'b', label=r'$\omega^{\rm I}_{22}$')
    a3.plot(tIrestr, -omegaIlmrestr[(2,-2)], 'r', label=r'$-\omega^{\rm I}_{2-2}$')
    a3.plot(tIrestr, omegaIlmrestr[(2,1)], 'b:', label=r'$\omega^{\rm I}_{21}$')
    a3.plot(tIrestr, -omegaIlmrestr[(2,-1)], 'r:', label=r'$-\omega^{\rm I}_{2-1}$')
    a3.set_xlim([tir, tfr])
    a3.set_ylim([0, 1.5*omegaQNM22])
    a3.axvline(x=0, color='k')
    a3.axhline(y=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Add horizontal lines for the omegaQNM representing the asymptotic behaviour
    if showQNM:
        a1.plot([0, tf], [omegaQNM22, omegaQNM22], 'k:', label=r'$\omega^{\rm QNM}_{220}$')
        a2.plot([0, tf], [omegaQNM21, omegaQNM21], 'k:', label=r'$\omega^{\rm QNM}_{210}$')
        a3.plot([0, tIrestr[-1]], [omegaQNM22, omegaQNM22], 'k-', linewidth=1.5, label=r'$\omega^{\rm QNM}_{220}$')
        a3.plot([0, tIrestr[-1]], [omegaQNM21, omegaQNM21], 'k:', linewidth=1.5, label=r'$\omega^{\rm QNM}_{210}$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Plot mode-based frequencies omega for the Precessing-frame waveform
def plot_wf_omegaPlm(wf, interval=[-400,150], showQNM=True, exportpdf=False, pdffile='', showparams=True):
    f = plt.figure(0, figsize=(20,20))
    a3 = plt.subplot2grid((3,3), (0,0), colspan=3)
    a1 = plt.subplot2grid((3,3), (1,0), colspan=3)
    a2 = plt.subplot2grid((3,3), (2,0), colspan=3)
    [ti, tf] = [wf['tI'][0], wf['tI'][-1]]
    # Get QNM frequencies
    if showQNM:
        chif = gwtools.norm(wf['metadata']['chif'])
        omegaQNM22 = gwtools.QNMomegalmnInt[(2,2,0)](chif)
        omegaQNM21 = gwtools.QNMomegalmnInt[(2,1,0)](chif)
    else: # still defining default for omegaQNM22 to set ylimits on the plots
        omegaQNM22 = 0.5 # only a guess
    # Compute omegaPlm
    modes = [(2,2), (2,1), (2,-1), (2,-2)]
    phiPlmint = {}
    omegaPlmint = {}
    omegaPlm = {}
    for lm in modes:
        phiPlmint[lm] = ip.InterpolatedUnivariateSpline(wf['tI'], -wf['phiPlm'][lm], k=3)
        omegaPlmint[lm] = (phiPlmint[lm]).derivative()
        omegaPlm[lm] = omegaPlmint[lm](wf['tI'])
    # omegaPlm for the full waveform
    a1.plot(wf['tI'], omegaPlm[(2,2)], 'b', label=r'$\omega^{\rm P}_{22}$')
    a1.plot(wf['tI'], -omegaPlm[(2,-2)], 'r', label=r'$-\omega^{\rm P}_{2-2}$')
    a1.set_xlim([ti, tf])
    a1.set_ylim([0, 1.5*omegaQNM22])
    a1.axhline(y=0, color='k')
    a1.axvline(x=0, color='k')
    a1.set_xlabel(r'$t/M$')
    a2.plot(wf['tI'], omegaPlm[(2,1)], 'b', label=r'$\omega^{\rm P}_{21}$')
    a2.plot(wf['tI'], -omegaPlm[(2,-1)], 'r', label=r'$-\omega^{\rm P}_{2-1}$')
    a2.set_xlim([ti, tf])
    a2.set_ylim([0, 1.5*omegaQNM21])
    a2.axvline(x=0, color='k')
    a2.set_xlabel(r'$t/M$')
    # omegaPlm zooming at the end
    omegaPlmrestr = {}
    for lm in modes:
        omegaPlmrestr[lm] = gwtools.restrict_data(np.array([wf['tI'], omegaPlm[lm]]).T, interval)[:,1]
    tIrestr = gwtools.restrict_data(np.array([wf['tI'], omegaPlm[(2,2)]]).T, interval)[:,0]
    [tir, tfr] = [tIrestr[0], tIrestr[-1]]
    a3.plot(tIrestr, omegaPlmrestr[(2,2)], 'b', label=r'$\omega^{\rm P}_{22}$')
    a3.plot(tIrestr, -omegaPlmrestr[(2,-2)], 'r', label=r'$-\omega^{\rm P}_{2-2}$')
    a3.plot(tIrestr, omegaPlmrestr[(2,1)], 'b:', label=r'$\omega^{\rm P}_{21}$')
    a3.plot(tIrestr, -omegaPlmrestr[(2,-1)], 'r:', label=r'$-\omega^{\rm P}_{2-1}$')
    a3.set_xlim([tir, tfr])
    a3.set_ylim([0, 1.5*omegaQNM22])
    a3.axvline(x=0, color='k')
    a3.axhline(y=0, color='k')
    a3.set_xlabel(r'$t/M$')
    # Add horizontal lines for the omegaQNM representing the asymptotic behaviour
    if showQNM:
        a1.plot([0, tf], [omegaQNM22, omegaQNM22], 'k:', label=r'$\omega^{\rm QNM}_{220}$')
        a2.plot([0, tf], [omegaQNM21, omegaQNM21], 'k:', label=r'$\omega^{\rm QNM}_{210}$')
        a3.plot([0, tIrestr[-1]], [omegaQNM22, omegaQNM22], 'k-', linewidth=1.5, label=r'$\omega^{\rm QNM}_{220}$')
        a3.plot([0, tIrestr[-1]], [omegaQNM21, omegaQNM21], 'k:', linewidth=1.5, label=r'$\omega^{\rm QNM}_{210}$')
    # Display grids and legends
    a1.grid()
    a2.grid()
    a3.grid()
    a1.legend(loc='upper left')
    a2.legend(loc='upper left')
    a3.legend(loc='upper left')
    # Show parameters as title
    if showparams:
        plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Note: beta is very noisy at the very end of ringdown - we impose a plotting range based on the value shortly before merger
def plot_wf_frame_compare_extrapolation(wfN4, wfN3, interval=[-400,150], showQNM=False, exportpdf=False, pdffile=''):
    f = plt.figure(0, figsize=(20,30))
    a1 = plt.subplot2grid((6,2), (0,0), colspan=2)
    a2 = plt.subplot2grid((6,2), (1,0), colspan=2)
    a3 = plt.subplot2grid((6,2), (2,0), colspan=2)
    a4 = plt.subplot2grid((6,2), (3,0), colspan=2)
    a5 = plt.subplot2grid((6,2), (4,0))
    a6 = plt.subplot2grid((6,2), (4,1))
    a7 = plt.subplot2grid((6,2), (5,0))
    a8 = plt.subplot2grid((6,2), (5,1))
    # Values of Vfz, beta close to merger to set plot range
    iclosestmerger = gwtools.find_closest_index(wfN4['tP'], -50)
    Vfzclosestmerger = wfN4['Vf'][iclosestmerger, 2]
    betaclosestmerger = wfN4['eulerVf'][iclosestmerger, 1]
    alphaclosestmerger = wfN4['eulerVf'][iclosestmerger, 0]
    # Check extrapolation order of inputs
    if not (wfN4['extrapolation']=='N4' and wfN3['extrapolation']=='N3'):
        raise ValueError("Error in plot_wf_frame_compareextrapolation: extrapolation order tags do not match.")
    # Plotting Vf Cartesian components, Euler angles for the full waveform - plotted for N4
    wf = wfN4
    a1.plot(wf['tP'], wf['Vf'][:,0], 'b', label=r'$V_f^x$')
    a1.plot(wf['tP'], wf['Vf'][:,1], 'r', label=r'$V_f^y$')
    a1.axvline(x=0, color='k')
    a1.axhline(y=0, color='k')
    a1.legend()
    a2.plot(wf['tP'], 1-wf['Vf'][:,2], 'y', label=r'$1-V_f^z$')
    a2.set_ylim([0, 2*(1-Vfzclosestmerger)])
    a2.axvline(x=0, color='k')
    a2.legend()
    a3.plot(wf['tP'], wf['eulerVf'][:,0], 'b', label=r'$\alpha$')
    a3.plot(wf['tP'], -wf['eulerVf'][:,2], 'r', label=r'$-\gamma$')
    a3.axvline(x=0, color='k')
    a3.axhline(y=0, color='k')
    a3.legend()
    a4.plot(wf['tP'], wf['eulerVf'][:,1], 'y', label=r'$\beta$')
    a4.set_ylim([0, 2*(betaclosestmerger)])
    a4.axvline(x=0, color='k')
    a4.legend()
    # Zooming at the end - showing both N4 and N3
    tPrestr = gwtools.restrict_data(np.array([wf['tP'], wf['Vf'][:,0], wf['Vf'][:,1], wf['Vf'][:,2]]).T, interval)[:,0]
    Vfrestr = gwtools.restrict_data(np.array([wf['tP'], wf['Vf'][:,0], wf['Vf'][:,1], wf['Vf'][:,2]]).T, interval)[:,1:]
    eulerVfrestr = gwtools.restrict_data(np.array([wf['tP'], wf['eulerVf'][:,0], wf['eulerVf'][:,1], wf['eulerVf'][:,2]]).T, interval)[:,1:]
    tPrestrN3 = gwtools.restrict_data(np.array([wfN3['tP'], wfN3['Vf'][:,0], wfN3['Vf'][:,1], wfN3['Vf'][:,2]]).T, interval)[:,0]
    VfrestrN3 = gwtools.restrict_data(np.array([wfN3['tP'], wfN3['Vf'][:,0], wfN3['Vf'][:,1], wfN3['Vf'][:,2]]).T, interval)[:,1:]
    eulerVfrestrN3 = gwtools.restrict_data(np.array([wfN3['tP'], wfN3['eulerVf'][:,0], wfN3['eulerVf'][:,1], wfN3['eulerVf'][:,2]]).T, interval)[:,1:]
    # Show the GaTech model for the post-merger rotation of the GW frame around final J
    if showQNM:
        chif = gwtools.norm(wfN4['metadata']['chif'])
        tpostmerger = np.array(list(filter(lambda x: x>=0, tPrestr)))
        a7.plot(tpostmerger, alphaclosestmerger + (QNMomegalmnInt[(2,2,0)](chif)-QNMomegalmnInt[(2,1,0)](chif))*tpostmerger, 'k:', label=r'$\omega^{\rm QNM}_{220} - \omega^{\rm QNM}_{210}$')
    a5.plot(tPrestr, Vfrestr[:,0], 'b', label=r'$V_f^x$ N4')
    a5.plot(tPrestr, Vfrestr[:,1], 'r', label=r'$V_f^y$ N3')
    a5.plot(tPrestrN3, VfrestrN3[:,0], 'b--', label=r'$V_f^x$ N4')
    a5.plot(tPrestrN3, VfrestrN3[:,1], 'r--', label=r'$V_f^y$ N3')
    a5.axvline(x=0, color='k')
    a5.axhline(y=0, color='k')
    a5.legend(loc='upper left')
    a6.plot(tPrestr, 1-Vfrestr[:,2], 'y', label=r'$1-V_f^z$ N4')
    a6.plot(tPrestrN3, 1-VfrestrN3[:,2], 'y--', label=r'$1-V_f^z$ N3')
    a6.set_ylim([0, 2*(1-Vfzclosestmerger)])
    a6.axvline(x=0, color='k')
    a6.legend(loc='upper left')
    a7.plot(tPrestr, eulerVfrestr[:,0], 'b', label=r'$\alpha$ N4')
    a7.plot(tPrestr, -eulerVfrestr[:,2], 'r', label=r'$-\gamma$ N3')
    a7.plot(tPrestrN3, eulerVfrestrN3[:,0], 'b--', label=r'$\alpha$ N4')
    a7.plot(tPrestrN3, -eulerVfrestrN3[:,2], 'r--', label=r'$-\gamma$ N3')
    a7.axvline(x=0, color='k')
    a7.axhline(y=0, color='k')
    a7.legend(loc='upper left')
    a8.plot(tPrestr, eulerVfrestr[:,1], 'y', label=r'$\beta$ N4')
    a8.plot(tPrestrN3, eulerVfrestrN3[:,1], 'y--', label=r'$\beta$ N3')
    a8.set_ylim([0, 2*(betaclosestmerger)])
    a8.axvline(x=0, color='k')
    a8.legend(loc='upper left')
    # Show parameters as title
    plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

# Note: beta is very noisy at the very end of ringdown - we impose a plotting range based on the value shortly before merger
def plot_wf_Vf(wf, interval=[-400,150], showQNM=False, exportpdf=False, pdffile=''):
    f = plt.figure(0, figsize=(20,20))
    a1 = plt.subplot2grid((3,2), (0,0), colspan=2)
    a2 = plt.subplot2grid((3,2), (1,0), colspan=2)
    a3 = plt.subplot2grid((3,2), (2,0))
    a4 = plt.subplot2grid((3,2), (2,1))
    [ti, tf] = [wf['tP'][0], wf['tP'][-1]]
    # Values of beta close to merger to set plot range
    iclosemerger = gwtools.find_closest_index(wf['tP'], -50)
    Vfzclosemerger = wf['Vf'][iclosemerger, 2]
    # Plotting Euler angles for the full waveform
    a1.plot(wf['tP'], wf['Vf'][:,0], 'b', label=r'$V_f^x$')
    a1.plot(wf['tP'], wf['Vf'][:,1], 'r', label=r'$V_f^y$')
    a1.set_xlim([ti, tf])
    a1.axvline(x=0, color='k')
    a1.axhline(y=0, color='k')
    a1.legend()
    a2.plot(wf['tP'], 1-wf['Vf'][:,2], 'y', label=r'$1-V_f^z$')
    a2.set_xlim([ti, tf])
    a2.set_ylim([0, 2*(1-Vfzclosemerger)])
    a2.axvline(x=0, color='k')
    a2.legend()
    # Zooming at the end
    tPVfrestr = gwtools.restrict_data(np.array([wf['tP'], wf['Vf'][:,0], wf['Vf'][:,1], wf['Vf'][:,2]]).T, interval)
    tPrestr = tPVfrestr[:,0]
    [tir, tfr] = [tPrestr[0], tPrestr[-1]]
    Vfrestr = tPVfrestr[:,1:]
    a3.plot(tPrestr, Vfrestr[:,0], 'b', label=r'$V_f^x$')
    a3.plot(tPrestr, Vfrestr[:,1], 'r', label=r'$V_f^y$')
    a3.set_xlim([tir, tfr])
    a3.axvline(x=0, color='k')
    a3.axhline(y=0, color='k')
    a3.legend(loc='upper left')
    a4.plot(tPrestr, 1-Vfrestr[:,2], 'y', label=r'$1-V_f^z$')
    a4.set_xlim([tir, tfr])
    a4.set_ylim([0, 2*(1-Vfzclosemerger)])
    a4.axvline(x=0, color='k')
    a4.legend(loc='upper left')
    # Show parameters as title
    plt.figtext(0.5, 0.915, r'$q=%.3f \quad \chi_1=(%.3f, %.3f, %.3f) \quad \chi_2=(%.3f, %.3f, %.3f)$'%(wf['metadata']['q'], wf['metadata']['chi1'][0], wf['metadata']['chi1'][1], wf['metadata']['chi1'][2], wf['metadata']['chi2'][0], wf['metadata']['chi2'][1], wf['metadata']['chi2'][2]), fontsize=20, horizontalalignment='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.3'))

    # Show plot or export pdf to file
    if not exportpdf:
        plt.show()
    else:
        if pdffile=='':
            raise ValueError("Error in plot_wf_frame_compare_extrapolation: empty file name for export pdf.")
        else:
            f.savefig(pdffile, bbox_inches='tight')
            plt.close(f)

########################################################################
## Functions to parse and manipulate parameter files and posterior files
########################################################################

# Functions to convert ptmcmc-formatted params to the L-frame
def convert_ptmcmc_params_Lframe(x, defLframe='paper', SetphiRefSSBAtfRef=False):
    xc = x.copy()
    tL = lisatools.functLfromtSSB(x[4], x[8], x[9])
    phiL = lisatools.funcphiL(x[2], x[3], x[4], x[6], SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    lambdaL = lisatools.funclambdaL(x[8], x[9], defLframe=defLframe)
    betaL = lisatools.funcbetaL(x[8], x[9])
    psiL = lisatools.funcpsiL(x[8], x[9], x[10])
    xc[4] = tL
    xc[6] = phiL
    xc[8] = lambdaL
    xc[9] = betaL
    xc[10] = psiL
    return xc
def convert_ptmcmc_post_Lframe(posterior, defLframe='paper', SetphiRefSSBAtfRef=False):
    return np.array(list(map(lambda x: convert_ptmcmc_params_Lframe(x, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef), posterior)))
# Function to convert ptmcmc-formatted posterior to bambi/multinest-formatted posterior
def convert_ptmcmc_post_to_bambi(posterior):
    likes = posterior[:,1]
    otherparams = posterior[:,2:]
    return np.concatenate((otherparams, np.array([likes]).T), axis=1)
# Takes a posterior array with m1 and m2 as first columns, and appends columns for M, q, Mchirp, eta
def compute_massparams_post(post):
    M = post[:,0] + post[:,1]
    q = post[:,0] / post[:,1]
    Mchirp = np.array(list(map(lambda x: gwtools.Mchirpofm1m2(x[0], x[1]), post)))
    eta = post[:,0] * post[:,1] / (M * M)
    ext = np.array([M, q, Mchirp, eta]).T
    return np.concatenate((post, ext), axis=1)
# Takes a posterior wiht bambi/multinest format (columns : params + like) and put it in the plotting format, with other mass params appended and no likelihood
def convert_post_to_plotformat(post):
    res = post[:,:-1]
    res = compute_massparams_post(res)
    return res

# def convertposttSSBtotL(posterior):
#     post = posterior.copy()
#     tLvals = np.array(map(lambda x: functLfromtSSB(x[2], x[6], x[7]), post))
#     post[:,2] = tLvals
#     return post
# def convertposttLtotSSB(posterior):
#     post = posterior.copy()
#     tSSBvals = np.array(map(lambda x: functSSBfromtL(x[2], x[6], x[7]), post))
#     post[:,2] = tSSBvals
#     return post
def convert_params_Lframe(params, defLframe='paper', SetphiRefSSBAtfRef=False, flipphase=True, frozenLISA=False, tfrozenLISA=None):
    tSSB, lambdaSSB, betaSSB, psiSSB = params[[2, 6, 7, 8]]
    tL, lambdaL, betaL, psiL = lisatools.ConvertSSBframeParamsToLframe(
        tSSB,
        lambdaSSB,
        betaSSB,
        psiSSB,
        constellation_ini_phase=0.,
        frozenLISA=frozenLISA,
        tfrozenLISA=tfrozenLISA)
    m1, m2, phiSSB = params[[0,1,4]]
    phiL = lisatools.funcphiL(m1, m2, tSSB, phiSSB, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef, flipphase=flipphase)
    params_L = np.copy(params)
    params_L[[2, 4, 6, 7, 8]] = tL, phiL, lambdaL, betaL, psiL
    return params_L
def convert_post_Lframe(posterior, defLframe='paper', SetphiRefSSBAtfRef=False, flipphase=True, frozenLISA=False, tfrozenLISA=None):
    return np.array(list(map(lambda x: convert_params_Lframe(x, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef, flipphase=flipphase, frozenLISA=frozenLISA, tfrozenLISA=tfrozenLISA), posterior)))
# Function to convert L-frame params to SSB-frame params
def convert_params_SSBframe(params, SetphiRefSSBAtfRef=False, flipphase=True, frozenLISA=False, tfrozenLISA=None):
    tL, lambdaL, betaL, psiL = params[[2, 6, 7, 8]]
    tSSB, lambdaSSB, betaSSB, psiSSB = lisatools.ConvertLframeParamsToSSBframe(
        tL,
        lambdaL,
        betaL,
        psiL,
        constellation_ini_phase=0.,
        frozenLISA=frozenLISA,
        tfrozenLISA=tfrozenLISA)
    m1, m2, phiL = params[[0,1,4]]
    phiSSB = lisatools.funcphiSSB(m1, m2, tSSB, phiL, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef, flipphase=flipphase)
    params_SSB = np.copy(params)
    params_SSB[[2, 4, 6, 7, 8]] = tSSB, phiSSB, lambdaSSB, betaSSB, psiSSB
    return params_SSB
def convert_post_SSBframe(posterior, defLframe='paper', SetphiRefSSBAtfRef=False, flipphase=True, frozenLISA=False, tfrozenLISA=None):
    return np.array(list(map(lambda x: convert_params_SSBframe(x, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef, flipphase=flipphase, frozenLISA=frozenLISA, tfrozenLISA=tfrozenLISA), posterior)))

# Functions to parse a bambi .out file
# Extract evolution of evidence and acceptance rate with the number of evaluations
# def parse_bambi_outputfile():
#     pass

# Functions to parse a _params.txt file, extracting injected parameters
def linematchingsymbol(stringarr, symbol):
    stringarrmatch = list(filter(lambda x: re.match(r"^ *" + re.escape(symbol) + r" *:", x) is not None, stringarr))
    if len(stringarrmatch) == 0:
        return ""
    elif len(stringarrmatch) > 1:
        return ""
    else:
        return stringarrmatch[0]

def valueinline(line):
    if line == '':
        return 'nan'
    else:
        return re.search(" (?! )(.*)$", line).group(1)

def parse_params_file(file, detector='LISA'):
    with open(file) as f:
        content = f.readlines()
    f.close
    res = []
    m1 = float(valueinline(linematchingsymbol(content, 'm1')))
    m2 = float(valueinline(linematchingsymbol(content, 'm2')))
    tRef = float(valueinline(linematchingsymbol(content, 'tRef')))
    phiRef = float(valueinline(linematchingsymbol(content, 'phiRef')))
    inc = float(valueinline(linematchingsymbol(content, 'inclination')))
    pol = float(valueinline(linematchingsymbol(content, 'polarization')))
    M = m1 + m2
    q = m1 / m2
    Mchirp = gwtools.Mchirpofm1m2(m1, m2)
    eta = m1 * m2 / (M * M)
    # Read SNR - not always present in param file
    if not linematchingsymbol(content, 'SNR') == '':
        SNR = float(valueinline(linematchingsymbol(content, 'SNR')))
    # Special treatment for the distance
    if not linematchingsymbol(content, 'dist_resc') == '':
        dist = float(valueinline(linematchingsymbol(content, 'dist_resc')))
    else:
        dist = float(valueinline(linematchingsymbol(content, 'distance')))
    if detector=='LISA':
        lambd = float(valueinline(linematchingsymbol(content, 'lambda')))
        beta = float(valueinline(linematchingsymbol(content, 'beta')))
        res =  [m1, m2, tRef, dist, phiRef, inc, lambd, beta, pol, M, q, Mchirp, eta]
    elif detector=='LLV':
        ra = float(valueinline(linematchingsymbol(content, 'ra')))
        dec = float(valueinline(linematchingsymbol(content, 'dec')))
        res = [m1, m2, tRef, dist, phiRef, inc, ra, dec, pol, M, q, Mchirp, eta]
    else:
        raise ValueError('Unrecognized value for detector.')
    return np.array(res)

# Function to load and convert _post_separate.dat files (with modes separated by empty lines)
def load_post_separate_data(file):
    f = open(file, 'r')
    lines = [line.rstrip('\n') for line in f]
    f.close()
    i = 0
    indices = []
    wasempty = re.match("^ *$", lines[0]) is not None
    while(i<len(lines)):
        emptyline = re.match("^ *$", lines[i]) is not None
        if(emptyline and wasempty):
            pass
        elif(emptyline and not wasempty):
            end = i-1
            indices.append([start, end])
            wasempty = True
        elif(not emptyline and not wasempty):
            pass
        elif(not emptyline and wasempty):
            start = i
            wasempty = False
        i += 1
    if(not wasempty and i==len(lines)):
        end = i-1
        indices.append([start, end])
    arrfromfile = np.loadtxt(file)
    reslist = [(arrfromfile[ij[0]:ij[1]+1])[:,2:] for ij in indices] #Role unknown for the first two columns, not copied - the 9 others seem to correspond to parameters
    return list(map(lambda x: compute_massparams_post(x), reslist))

def load_post_data(file):
    #last column is logLikelihood, not copied
    return compute_massparams_post((np.loadtxt(file))[:,:9])

# Function taken from J.Baker's corner-fisher.py
def read_covariance(file):
    pars=[]
    done=False
    trycount=0
    with open(file,'r') as f:
        line="#"
        while("#" in line): line=f.readline() #Skip comment
        for val in line.split():
            pars.append(float(val))
        Npar=len(pars)
        while(not "#Covariance" in line):line=f.readline() #Skip until the good stuff
        covar=np.zeros((Npar,Npar))
        i=0
        for par in pars:
            line=f.readline()
            covar[i]=np.array(line.split())
            i+=1
    return covar
def read_fisher(file):
    pars=[]
    done=False
    trycount=0
    with open(file,'r') as f:
        line="#"
        while("#" in line): line=f.readline() #Skip comment
        for val in line.split():
            pars.append(float(val))
        Npar=len(pars)
        while(not "#Fisher" in line):line=f.readline() #Skip until the good stuff
        covar=np.zeros((Npar,Npar))
        i=0
        for par in pars:
            line=f.readline()
            covar[i]=np.array(line.split())
            i+=1
    return covar
# Convert SSB-frame covariance to L-frame
# Fisher matrix : if J_{a',b} = \partial xL^a' / \partial x^b Jacobian matrix
# F' = tJ^-1 . F . J^-1
# C' = (F^-1)' = J . C . tJ
def convert_covariance_Lframe(params, cov, SetphiRefSSBAtfRef=False):
    J = lisatools.funcJacobianSSBtoLframe(params, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
    return np.dot(J, np.dot(cov, J.T))

# Compute posterior values for individual posterior samples by multiplying with the prior
# Assumes a np array for the posterior samples, the last column being the likelihood values
# Assumes flat prior in time, phase, polarization, sphere prior in inclination and sky position
# Normalization is arbitrary
# Parameter order assumed : m1, m2, tRef, dist, phiRef, inc, lambda, beta, pol, loglikelihood
def compute_posterior(posterior, flatdistprior=False, logflatmassprior=False):
    # sort posterior samples, highest loglikelihood first
    posterior = (posterior[posterior[:,9].argsort()])[::-1]
    if flatdistprior and logflatmassprior:
        def prior(x):
            return 1./x[0] * 1./x[1] * np.sin(x[5]) * np.cos(x[7])
    elif flatdistprior and not logflatmassprior:
        def prior(x):
            return np.sin(x[5]) * np.cos(x[7])
    elif not flatdistprior and logflatmassprior:
        def prior(x):
            return 1./x[0] * 1./x[1] * x[3]**2 * np.sin(x[5]) * np.cos(x[7])
    elif not flatdistprior and not logflatmassprior:
        def prior(x):
            return  x[3]**2 * np.sin(x[5]) * np.cos(x[7])
    priorvalues = np.array(list(map(prior, posterior)))
    # normalize (arbitrarily) to the prior value of highest likelihood (injection)
    priorvalues = priorvalues / priorvalues[0]
    posteriorvalues = np.log(priorvalues) + posterior[:,9]
    return np.concatenate((posterior, np.array([posteriorvalues]).T), axis=1)

#########################################################################
## Corner plot function, based on corner_covar, modification of corner.py
#########################################################################

# Order of parameters in the output of loadpostdata
# Follows ordering in the C code, with M, q, Mchirp, eta appended
# Deltat, phi, lambda, beta, pol can be either SSB-frame or L-frame
def paramsmap(detector):
    paramsmap = {}
    if detector=='LISA':
        paramsmap = {'m1':0, 'm2':1, 'Deltat':2, 'D':3, 'phi':4, 'inc':5, 'lambda':6, 'beta':7, 'pol':8, 'M':9, 'q':10, 'Mchirp':11, 'eta':12}
    elif detector=='LLV':
        paramsmap = {'m1':0, 'm2':1, 'Deltat':2, 'D':3, 'phi':4, 'inc':5, 'ra':6, 'dec':7, 'pol':8, 'M':9, 'q':10, 'Mchirp':11, 'eta':12}
    return paramsmap

unit_M_dict = {
    '1e8':r'$(10^{8}\mathrm{M}_{\odot})$',
    '1e7':r'$(10^{7}\mathrm{M}_{\odot})$',
    '1e6':r'$(10^{6}\mathrm{M}_{\odot})$',
    '1e5':r'$(10^{5}\mathrm{M}_{\odot})$',
    '1e4':r'$(10^{4}\mathrm{M}_{\odot})$',
    '1e3':r'$(10^{3}\mathrm{M}_{\odot})$',
    '1e2':r'$(\mathrm{M}_{\odot})$',
    '1e1':r'$(\mathrm{M}_{\odot})$',
    '1':r'$(\mathrm{M}_{\odot})$'}
unit_D_dict = {
    'Gpc':r'$(\mathrm{Gpc})$',
    'Mpc':r'$(\mathrm{Mpc})$'}
unit_t_dict = {
    's':r'$(\mathrm{s})$',
    'ms':r'$(\mathrm{ms})$'}
scale_dict_M = {'1e8':1e8, '1e7':1e7, '1e6':1e6, '1e5':1e5, '1e4':1e4, '1e3':1e3, '1e2':1e2, '1e1':10., '1':1.}
scale_dict_D = {'Gpc':1e3, 'Mpc':1.}
scale_dict_t = {'s':1., 'ms':1e-3}

# Display of parameter names
def param_label_dict(detector, scales, Lframe=False):
    unit_str_M = unit_M_dict[scales[0]]
    unit_str_D = unit_D_dict[scales[1]]
    unit_str_t = unit_t_dict[scales[2]]
    res = {}
    if detector=='LISA':
        if not Lframe:
            res = {
                'm1'     : r'$m_1 \;$' + unit_str_M,
                'm2'     : r'$m_2 \;$' + unit_str_M,
                'Deltat' : r'$\Delta t \;$' + unit_str_t,
                'D'      : r'$D_L \;$' + unit_str_D,
                'phi'    : r'$\phi \; (\mathrm{rad})$',
                'inc'    : r'$\iota \; (\mathrm{rad})$',
                'lambda' : r'$\lambda \; (\mathrm{rad})$',
                'beta'   : r'$\beta \; (\mathrm{rad})$',
                'pol'    : r'$\psi \; (\mathrm{rad})$',
                'M'      : r'$M \;$' + unit_str_M,
                'q'      : r'$q$',
                'Mchirp' : r'$\mathcal{M}_c \;$' + unit_str_M,
                'eta'    : r'$\eta$'
            }
        else:
            res = {
                'm1'     : r'$m_1 \;$' + unit_str_M,
                'm2'     : r'$m_2 \;$' + unit_str_M,
                'Deltat' : r'$\Delta t_{L} \;$' + unit_str_t,
                'D'      : r'$D_L \;$' + unit_str_D,
                'phi'    : r'$\phi_{L} \; (\mathrm{rad})$',
                'inc'    : r'$\iota \; (\mathrm{rad})$',
                'lambda' : r'$\lambda_{L} \; (\mathrm{rad})$',
                'beta'   : r'$\beta_{L} \; (\mathrm{rad})$',
                'pol'    : r'$\psi_{L} \; (\mathrm{rad})$',
                'M'      : r'$M \;$' + unit_str_M,
                'q'      : r'$q$',
                'Mchirp' : r'$\mathcal{M}_c \;$' + unit_str_M,
                'eta'    : r'$\eta$'
            }
    elif detector=='LLV':
        res = {
            'm1'     : r'$m_1 \;$' + unit_str_M,
            'm2'     : r'$m_2 \;$' + unit_str_M,
            'Deltat' : r'$\Delta t \;$' + unit_str_t,
            'D'      : r'$D_L \;$' + unit_str_D,
            'phi'    : r'$\phi \; (\mathrm{rad})$',
            'inc'    : r'$\iota \; (\mathrm{rad})$',
            'ra'     : r'$\mathrm{ra} \; (\mathrm{rad})$',
            'dec'    : r'$\mathrm{dec} \; (\mathrm{rad})$',
            'pol'    : r'$\psi \; (\mathrm{rad})$',
            'M'      : r'$M \;$' + unit_str_M,
            'q'      : r'$q$',
            'Mchirp' : r'$\mathcal{M}_c \;$' + unit_str_M,
            'eta'    : r'$\eta$'
        }
    return res

def automatic_scales(injparams, posterior):
    M = injparams[9]
    scale_M_int = min(8, max(int(np.floor(np.log(M)/np.log(10.))), 0))
    if scale_M_int<=2:
        scale_M = '1'
    else:
        scale_M = '1e' + str(scale_M_int)
    DL = injparams[3]
    if DL>=1e3:
        scale_D = 'Gpc'
    else:
        scale_D = 'Mpc'
    Deltat_mean = np.abs(np.mean(posterior[:,2]))
    if Deltat_mean>=0.1:
        scale_t = 's'
    else:
        scale_t = 'ms'
    return [scale_M, scale_D, scale_t]
# Assumed format for posterior: m1 m2 Deltat D phi inc lambda beta pol M q Mchirp eta
def scale_posterior(posterior, scales):
    [sM, sD, st] = scales
    scaling = np.diag(np.array([1./sM, 1./sM, 1./st, 1./sD, 1., 1., 1., 1., 1., 1./sM, 1., 1./sM, 1.]))
    return np.dot(posterior, scaling)
# Assumed format for posterior: m1 m2 Deltat D phi inc lambda beta pol M q Mchirp eta
def scale_injparams(injparams, scales):
    [sM, sD, st] = scales
    scaling = np.array([1./sM, 1./sM, 1./st, 1./sD, 1., 1., 1., 1., 1., 1./sM, 1., 1./sM, 1.])
    return np.multiply(injparams, scaling)
# Assumed format for covariance: m1 m2 Deltat D phi inc lambda beta pol
def scale_covariance(cov, scales, params):
    [sM, sD, st] = scales
    scale_dict = {'m1':sM, 'm2':sM, 'Mchirp':sM, 'M':sM, 'q':1., 'eta':1., 'D':sD, 'Deltat':st, 'phi':1., 'inc':1., 'lambda':1., 'beta':1., 'pol':1.}
    scaling = np.diag(np.array([1./scale_dict[p] for p in params]))
    return np.dot(scaling, np.dot(cov, scaling))

# Default levels for contours in 2d histogram - TODO: check the correspondence with 1,2,3sigma
default_levels = 1.0 - np.exp(-0.5 * np.linspace(1.0, 3.0, num=3) ** 2)

def corner_plot(injparams, posterior, add_posteriors=None, output=False, output_dir=None, output_file=None, histograms=True, fisher=False, cov=None, detector='LISA', params=['m1', 'm2', 'Deltat', 'D', 'phi', 'inc', 'lambda', 'beta', 'pol'], params_range=None, Lframe=False, scales=None, add_truths=None, color="k", add_colors=None, cov_color=None, show_truths=True, truth_color="#4682b4", add_truth_colors=None, bins=50, show_histograms=True, quantiles=[0.159, 0.5, 0.841], show_quantiles=True, levels=default_levels, plot_density=True, plot_contours=True, plot_datapoints=True, smooth=None, smooth1d=None, label_kwargs={"fontsize": 16}, show_titles=True, defLframe='paper', SetphiRefSSBAtfRef=False, no_reorder_fisher=False, plot_density_add_post=False):

    # If required, transform to parameters in the L-frame
    if Lframe:
        injparams = convert_params_Lframe(injparams, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
        posterior = convert_post_Lframe(posterior, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef)
        if add_posteriors is not None:
            add_posteriors = list(map(lambda x: convert_post_Lframe(x, defLframe=defLframe, SetphiRefSSBAtfRef=SetphiRefSSBAtfRef), add_posteriors))

    # If not provided, determine automatically scales [M, D, t]
    if not scales:
        scales_val = automatic_scales(injparams, posterior)
    else:
        scales_val = scales
    [scale_M, scale_D, scale_t] = scales_val
    scalefactors = [scale_dict_M[scale_M], scale_dict_D[scale_D], scale_dict_t[scale_t]]
    injparams_scaled = scale_injparams(injparams, scalefactors)
    posterior_scaled = scale_posterior(posterior, scalefactors)
    if add_posteriors is not None:
        add_posteriors = list(map(lambda x: scale_posterior(x, scalefactors), add_posteriors))

    # Posterior and injection parameters for the ordered set of params required
    parmap = paramsmap(detector)
    ordered_cols = list(map(parmap.get, params))
    ordered_posterior = posterior_scaled[:,ordered_cols]
    ordered_injparams = injparams_scaled[ordered_cols]
    if add_posteriors is not None:
        ordered_add_posteriors = list(map(lambda x: x[:,ordered_cols], add_posteriors))
    if add_truths is not None:
        ordered_add_truths = list(map(lambda x: x[ordered_cols], add_truths))
    else:
        ordered_add_truths = None
    if show_truths:
        truths = ordered_injparams
    else:
        truths = None

    # If required, load Fisher matrix
    # Use no_reorder_fisher e.g. if the input Fisher matrix is given in Mchirp-eta
    if fisher:
        if cov is None:
            raise ValueError('Covariance matrix not specified.')
        # For now, Fisher matrix is only available from J.B.'s code with the original parameters, no derived mass params
        if not no_reorder_fisher and any([p in params for p in ['M', 'q', 'Mchirp', 'eta']]):
            raise ValueError('If not ignore_param_reordering_fisher, only m1, m2 are supported as mass parameters for the Fisher matrix.')
        #cov = read_covariance(cov_file)
        ordered_cols_cartesian = np.ix_(ordered_cols, ordered_cols)
        if not no_reorder_fisher:
            cov = cov[ordered_cols_cartesian]
        cov = scale_covariance(cov, scalefactors, params)
    else:
        cov = None

    # Main call to corner function
    label_dict = param_label_dict(detector, scales_val, Lframe)
    labels = list(map(lambda x: label_dict[x], params))
    fig, axes = corner_covar.corner(ordered_posterior, cov=cov, bins=bins, params_range=params_range, levels=levels,
                                 labels=labels, label_kwargs=label_kwargs, color=color,
                                 truths=truths, add_truths=ordered_add_truths, truth_color=truth_color, add_truth_colors=add_truth_colors, plot_datapoints=plot_datapoints,
                                 smooth=smooth, smooth1d=smooth1d, show_histograms=show_histograms,
                                 quantiles=quantiles, show_quantiles=show_quantiles, plot_contours=plot_contours,
                                 show_titles=show_titles, plot_density=plot_density)

    # Overlay other posterior
    if add_posteriors is not None:
        if add_colors is None:
            add_colors = [gwtools.plotpalette[i % len(gwtools.plotpalette)] for i in range(len(add_posteriors))]
        for i, add_post in enumerate(ordered_add_posteriors):
            corner_covar.corner(add_post, figaxes=(fig, axes), cov=None, bins=bins, params_range=params_range, levels=levels,
                                     labels=labels, label_kwargs=label_kwargs, color=add_colors[i],
                                     truths=None, plot_datapoints=plot_datapoints,
                                     smooth=smooth, smooth1d=smooth1d, show_histograms=show_histograms,
                                     quantiles=quantiles, show_quantiles=show_quantiles, plot_contours=plot_contours,
                                     show_titles=False,
                                     plot_density=plot_density_add_post, no_fill_contours=True)

    # Output
    if output:
        if not output_dir:
            raise ValueError('output_dir not defined.')
        if not output_file:
            raise ValueError('output_file not defined.')
        fig.savefig(output_dir + output_file, bbox_inches='tight')
    else:
        return fig
