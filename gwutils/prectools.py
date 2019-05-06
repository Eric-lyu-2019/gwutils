## Tools for precessing waveforms
## NOTE: dependency on GWFrames and Quaternions

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

import GWFrames
import Quaternions

from gwtools import ffactorial


# Mathematica-generated LL matrix elements
LLmatrix = {}
LLmatrix[(2,-2,-2)] = np.array([[1., 0. - 1. * 1j, 0.], [0. + 1. * 1j, 1., 0.], [0., 0., 4.]])
LLmatrix[(2,-2,-1)] = np.array([[0., 0., -2.], [0., 0., 0. + 2. * 1j], [-1., 0. + 1. * 1j, 0.]])
LLmatrix[(2,-2,0)] = np.array([[1.224744871391589, 0. - 1.224744871391589 * 1j, 0.], [0. - 1.224744871391589 * 1j, -1.224744871391589, 0.], [0., 0., 0.]])
LLmatrix[(2,-2,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,-2,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,-1,-2)] = np.array([[0., 0., -1.], [0., 0., 0. - 1. * 1j], [-2., 0. - 2. * 1j, 0.]])
LLmatrix[(2,-1,-1)] = np.array([[2.5, 0. - 0.5 * 1j, 0.], [0. + 0.5 * 1j, 2.5, 0.], [0., 0., 1.]])
LLmatrix[(2,-1,0)] = np.array([[0., 0., -1.224744871391589], [0., 0., 0. + 1.224744871391589 * 1j], [0., 0., 0.]])
LLmatrix[(2,-1,1)] = np.array([[1.5, 0. - 1.5 * 1j, 0.], [0. - 1.5 * 1j, -1.5, 0.], [0., 0., 0.]])
LLmatrix[(2,-1,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,0,-2)] = np.array([[1.224744871391589, 0. + 1.224744871391589 * 1j, 0.], [0. + 1.224744871391589 * 1j, -1.224744871391589, 0.], [0., 0., 0.]])
LLmatrix[(2,0,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [-1.224744871391589, 0. - 1.224744871391589 * 1j, 0.]])
LLmatrix[(2,0,0)] = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 0.]])
LLmatrix[(2,0,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [1.224744871391589, 0. - 1.224744871391589 * 1j, 0.]])
LLmatrix[(2,0,2)] = np.array([[1.224744871391589, 0. - 1.224744871391589 * 1j, 0.], [0. - 1.224744871391589 * 1j, -1.224744871391589, 0.], [0., 0., 0.]])
LLmatrix[(2,1,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,1,-1)] = np.array([[1.5, 0. + 1.5 * 1j, 0.], [0. + 1.5 * 1j, -1.5, 0.], [0., 0., 0.]])
LLmatrix[(2,1,0)] = np.array([[0., 0., 1.224744871391589], [0., 0., 0. + 1.224744871391589 * 1j], [0., 0., 0.]])
LLmatrix[(2,1,1)] = np.array([[2.5, 0. + 0.5 * 1j, 0.], [0. - 0.5 * 1j, 2.5, 0.], [0., 0., 1.]])
LLmatrix[(2,1,2)] = np.array([[0., 0., 1.], [0., 0., 0. - 1. * 1j], [2., 0. - 2. * 1j, 0.]])
LLmatrix[(2,2,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,2,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(2,2,0)] = np.array([[1.224744871391589, 0. + 1.224744871391589 * 1j, 0.], [0. + 1.224744871391589 * 1j, -1.224744871391589, 0.], [0., 0., 0.]])
LLmatrix[(2,2,1)] = np.array([[0., 0., 2.], [0., 0., 0. + 2. * 1j], [1., 0. + 1. * 1j, 0.]])
LLmatrix[(2,2,2)] = np.array([[1., 0. + 1. * 1j, 0.], [0. - 1. * 1j, 1., 0.], [0., 0., 4.]])
LLmatrix[(3,-3,-3)] = np.array([[1.5, 0. - 1.5 * 1j, 0.], [0. + 1.5 * 1j, 1.5, 0.], [0., 0., 9.]])
LLmatrix[(3,-3,-2)] = np.array([[0., 0., -3.674234614174767], [0., 0., 0. + 3.674234614174767 * 1j], [-2.449489742783178, 0. + 2.449489742783178 * 1j, 0.]])
LLmatrix[(3,-3,-1)] = np.array([[1.936491673103708, 0. - 1.936491673103708 * 1j, 0.], [0. - 1.936491673103708 * 1j, -1.936491673103708, 0.], [0., 0., 0.]])
LLmatrix[(3,-3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-3,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-3,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-3,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-2,-3)] = np.array([[0., 0., -2.449489742783178], [0., 0., 0. - 2.449489742783178 * 1j], [-3.674234614174767, 0. - 3.674234614174767 * 1j, 0.]])
LLmatrix[(3,-2,-2)] = np.array([[4., 0. - 1. * 1j, 0.], [0. + 1. * 1j, 4., 0.], [0., 0., 4.]])
LLmatrix[(3,-2,-1)] = np.array([[0., 0., -3.16227766016838], [0., 0., 0. + 3.16227766016838 * 1j], [-1.58113883008419, 0. + 1.58113883008419 * 1j, 0.]])
LLmatrix[(3,-2,0)] = np.array([[2.738612787525831, 0. - 2.738612787525831 * 1j, 0.], [0. - 2.738612787525831 * 1j, -2.738612787525831, 0.], [0., 0., 0.]])
LLmatrix[(3,-2,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-2,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-2,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-1,-3)] = np.array([[1.936491673103708, 0. + 1.936491673103708 * 1j, 0.], [0. + 1.936491673103708 * 1j, -1.936491673103708, 0.], [0., 0., 0.]])
LLmatrix[(3,-1,-2)] = np.array([[0., 0., -1.58113883008419], [0., 0., 0. - 1.58113883008419 * 1j], [-3.16227766016838, 0. - 3.16227766016838 * 1j, 0.]])
LLmatrix[(3,-1,-1)] = np.array([[5.5, 0. - 0.5 * 1j, 0.], [0. + 0.5 * 1j, 5.5, 0.], [0., 0., 1.]])
LLmatrix[(3,-1,0)] = np.array([[0., 0., -1.732050807568877], [0., 0., 0. + 1.732050807568877 * 1j], [0., 0., 0.]])
LLmatrix[(3,-1,1)] = np.array([[3., 0. - 3. * 1j, 0.], [0. - 3. * 1j, -3., 0.], [0., 0., 0.]])
LLmatrix[(3,-1,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,-1,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,0,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,0,-2)] = np.array([[2.738612787525831, 0. + 2.738612787525831 * 1j, 0.], [0. + 2.738612787525831 * 1j, -2.738612787525831, 0.], [0., 0., 0.]])
LLmatrix[(3,0,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [-1.732050807568877, 0. - 1.732050807568877 * 1j, 0.]])
LLmatrix[(3,0,0)] = np.array([[6., 0., 0.], [0., 6., 0.], [0., 0., 0.]])
LLmatrix[(3,0,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [1.732050807568877, 0. - 1.732050807568877 * 1j, 0.]])
LLmatrix[(3,0,2)] = np.array([[2.738612787525831, 0. - 2.738612787525831 * 1j, 0.], [0. - 2.738612787525831 * 1j, -2.738612787525831, 0.], [0., 0., 0.]])
LLmatrix[(3,0,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,1,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,1,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,1,-1)] = np.array([[3., 0. + 3. * 1j, 0.], [0. + 3. * 1j, -3., 0.], [0., 0., 0.]])
LLmatrix[(3,1,0)] = np.array([[0., 0., 1.732050807568877], [0., 0., 0. + 1.732050807568877 * 1j], [0., 0., 0.]])
LLmatrix[(3,1,1)] = np.array([[5.5, 0. + 0.5 * 1j, 0.], [0. - 0.5 * 1j, 5.5, 0.], [0., 0., 1.]])
LLmatrix[(3,1,2)] = np.array([[0., 0., 1.58113883008419], [0., 0., 0. - 1.58113883008419 * 1j], [3.16227766016838, 0. - 3.16227766016838 * 1j, 0.]])
LLmatrix[(3,1,3)] = np.array([[1.936491673103708, 0. - 1.936491673103708 * 1j, 0.], [0. - 1.936491673103708 * 1j, -1.936491673103708, 0.], [0., 0., 0.]])
LLmatrix[(3,2,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,2,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,2,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,2,0)] = np.array([[2.738612787525831, 0. + 2.738612787525831 * 1j, 0.], [0. + 2.738612787525831 * 1j, -2.738612787525831, 0.], [0., 0., 0.]])
LLmatrix[(3,2,1)] = np.array([[0., 0., 3.16227766016838], [0., 0., 0. + 3.16227766016838 * 1j], [1.58113883008419, 0. + 1.58113883008419 * 1j, 0.]])
LLmatrix[(3,2,2)] = np.array([[4., 0. + 1. * 1j, 0.], [0. - 1. * 1j, 4., 0.], [0., 0., 4.]])
LLmatrix[(3,2,3)] = np.array([[0., 0., 2.449489742783178], [0., 0., 0. - 2.449489742783178 * 1j], [3.674234614174767, 0. - 3.674234614174767 * 1j, 0.]])
LLmatrix[(3,3,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,3,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,3,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(3,3,1)] = np.array([[1.936491673103708, 0. + 1.936491673103708 * 1j, 0.], [0. + 1.936491673103708 * 1j, -1.936491673103708, 0.], [0., 0., 0.]])
LLmatrix[(3,3,2)] = np.array([[0., 0., 3.674234614174767], [0., 0., 0. + 3.674234614174767 * 1j], [2.449489742783178, 0. + 2.449489742783178 * 1j, 0.]])
LLmatrix[(3,3,3)] = np.array([[1.5, 0. + 1.5 * 1j, 0.], [0. - 1.5 * 1j, 1.5, 0.], [0., 0., 9.]])
LLmatrix[(4,-4,-4)] = np.array([[2., 0. - 2. * 1j, 0.], [0. + 2. * 1j, 2., 0.], [0., 0., 16.]])
LLmatrix[(4,-4,-3)] = np.array([[0., 0., -5.656854249492381], [0., 0., 0. + 5.656854249492381 * 1j], [-4.242640687119286, 0. + 4.242640687119286 * 1j, 0.]])
LLmatrix[(4,-4,-2)] = np.array([[2.645751311064591, 0. - 2.645751311064591 * 1j, 0.], [0. - 2.645751311064591 * 1j, -2.645751311064591, 0.], [0., 0., 0.]])
LLmatrix[(4,-4,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-4,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-4,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-4,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-4,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-4,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-3,-4)] = np.array([[0., 0., -4.242640687119286], [0., 0., 0. - 4.242640687119286 * 1j], [-5.656854249492381, 0. - 5.656854249492381 * 1j, 0.]])
LLmatrix[(4,-3,-3)] = np.array([[5.5, 0. - 1.5 * 1j, 0.], [0. + 1.5 * 1j, 5.5, 0.], [0., 0., 9.]])
LLmatrix[(4,-3,-2)] = np.array([[0., 0., -5.612486080160912], [0., 0., 0. + 5.612486080160912 * 1j], [-3.741657386773941, 0. + 3.741657386773941 * 1j, 0.]])
LLmatrix[(4,-3,-1)] = np.array([[3.968626966596886, 0. - 3.968626966596886 * 1j, 0.], [0. - 3.968626966596886 * 1j, -3.968626966596886, 0.], [0., 0., 0.]])
LLmatrix[(4,-3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-3,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-3,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-3,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-3,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-2,-4)] = np.array([[2.645751311064591, 0. + 2.645751311064591 * 1j, 0.], [0. + 2.645751311064591 * 1j, -2.645751311064591, 0.], [0., 0., 0.]])
LLmatrix[(4,-2,-3)] = np.array([[0., 0., -3.741657386773941], [0., 0., 0. - 3.741657386773941 * 1j], [-5.612486080160912, 0. - 5.612486080160912 * 1j, 0.]])
LLmatrix[(4,-2,-2)] = np.array([[8., 0. - 1. * 1j, 0.], [0. + 1. * 1j, 8., 0.], [0., 0., 4.]])
LLmatrix[(4,-2,-1)] = np.array([[0., 0., -4.242640687119286], [0., 0., 0. + 4.242640687119286 * 1j], [-2.121320343559642, 0. + 2.121320343559642 * 1j, 0.]])
LLmatrix[(4,-2,0)] = np.array([[4.743416490252569, 0. - 4.743416490252569 * 1j, 0.], [0. - 4.743416490252569 * 1j, -4.743416490252569, 0.], [0., 0., 0.]])
LLmatrix[(4,-2,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-2,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-2,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-2,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-1,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-1,-3)] = np.array([[3.968626966596886, 0. + 3.968626966596886 * 1j, 0.], [0. + 3.968626966596886 * 1j, -3.968626966596886, 0.], [0., 0., 0.]])
LLmatrix[(4,-1,-2)] = np.array([[0., 0., -2.121320343559642], [0., 0., 0. - 2.121320343559642 * 1j], [-4.242640687119286, 0. - 4.242640687119286 * 1j, 0.]])
LLmatrix[(4,-1,-1)] = np.array([[9.5, 0. - 0.5 * 1j, 0.], [0. + 0.5 * 1j, 9.5, 0.], [0., 0., 1.]])
LLmatrix[(4,-1,0)] = np.array([[0., 0., -2.23606797749979], [0., 0., 0. + 2.23606797749979 * 1j], [0., 0., 0.]])
LLmatrix[(4,-1,1)] = np.array([[5., 0. - 5. * 1j, 0.], [0. - 5. * 1j, -5., 0.], [0., 0., 0.]])
LLmatrix[(4,-1,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-1,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,-1,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,0,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,0,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,0,-2)] = np.array([[4.743416490252569, 0. + 4.743416490252569 * 1j, 0.], [0. + 4.743416490252569 * 1j, -4.743416490252569, 0.], [0., 0., 0.]])
LLmatrix[(4,0,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [-2.23606797749979, 0. - 2.23606797749979 * 1j, 0.]])
LLmatrix[(4,0,0)] = np.array([[10., 0., 0.], [0., 10., 0.], [0., 0., 0.]])
LLmatrix[(4,0,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [2.23606797749979, 0. - 2.23606797749979 * 1j, 0.]])
LLmatrix[(4,0,2)] = np.array([[4.743416490252569, 0. - 4.743416490252569 * 1j, 0.], [0. - 4.743416490252569 * 1j, -4.743416490252569, 0.], [0., 0., 0.]])
LLmatrix[(4,0,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,0,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,1,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,1,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,1,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,1,-1)] = np.array([[5., 0. + 5. * 1j, 0.], [0. + 5. * 1j, -5., 0.], [0., 0., 0.]])
LLmatrix[(4,1,0)] = np.array([[0., 0., 2.23606797749979], [0., 0., 0. + 2.23606797749979 * 1j], [0., 0., 0.]])
LLmatrix[(4,1,1)] = np.array([[9.5, 0. + 0.5 * 1j, 0.], [0. - 0.5 * 1j, 9.5, 0.], [0., 0., 1.]])
LLmatrix[(4,1,2)] = np.array([[0., 0., 2.121320343559642], [0., 0., 0. - 2.121320343559642 * 1j], [4.242640687119286, 0. - 4.242640687119286 * 1j, 0.]])
LLmatrix[(4,1,3)] = np.array([[3.968626966596886, 0. - 3.968626966596886 * 1j, 0.], [0. - 3.968626966596886 * 1j, -3.968626966596886, 0.], [0., 0., 0.]])
LLmatrix[(4,1,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,2,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,2,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,2,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,2,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,2,0)] = np.array([[4.743416490252569, 0. + 4.743416490252569 * 1j, 0.], [0. + 4.743416490252569 * 1j, -4.743416490252569, 0.], [0., 0., 0.]])
LLmatrix[(4,2,1)] = np.array([[0., 0., 4.242640687119286], [0., 0., 0. + 4.242640687119286 * 1j], [2.121320343559642, 0. + 2.121320343559642 * 1j, 0.]])
LLmatrix[(4,2,2)] = np.array([[8., 0. + 1. * 1j, 0.], [0. - 1. * 1j, 8., 0.], [0., 0., 4.]])
LLmatrix[(4,2,3)] = np.array([[0., 0., 3.741657386773941], [0., 0., 0. - 3.741657386773941 * 1j], [5.612486080160912, 0. - 5.612486080160912 * 1j, 0.]])
LLmatrix[(4,2,4)] = np.array([[2.645751311064591, 0. - 2.645751311064591 * 1j, 0.], [0. - 2.645751311064591 * 1j, -2.645751311064591, 0.], [0., 0., 0.]])
LLmatrix[(4,3,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,3,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,3,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,3,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,3,1)] = np.array([[3.968626966596886, 0. + 3.968626966596886 * 1j, 0.], [0. + 3.968626966596886 * 1j, -3.968626966596886, 0.], [0., 0., 0.]])
LLmatrix[(4,3,2)] = np.array([[0., 0., 5.612486080160912], [0., 0., 0. + 5.612486080160912 * 1j], [3.741657386773941, 0. + 3.741657386773941 * 1j, 0.]])
LLmatrix[(4,3,3)] = np.array([[5.5, 0. + 1.5 * 1j, 0.], [0. - 1.5 * 1j, 5.5, 0.], [0., 0., 9.]])
LLmatrix[(4,3,4)] = np.array([[0., 0., 4.242640687119286], [0., 0., 0. - 4.242640687119286 * 1j], [5.656854249492381, 0. - 5.656854249492381 * 1j, 0.]])
LLmatrix[(4,4,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(4,4,2)] = np.array([[2.645751311064591, 0. + 2.645751311064591 * 1j, 0.], [0. + 2.645751311064591 * 1j, -2.645751311064591, 0.], [0., 0., 0.]])
LLmatrix[(4,4,3)] = np.array([[0., 0., 5.656854249492381], [0., 0., 0. + 5.656854249492381 * 1j], [4.242640687119286, 0. + 4.242640687119286 * 1j, 0.]])
LLmatrix[(4,4,4)] = np.array([[2., 0. + 2. * 1j, 0.], [0. - 2. * 1j, 2., 0.], [0., 0., 16.]])
LLmatrix[(5,-5,-5)] = np.array([[2.5, 0. - 2.5 * 1j, 0.], [0. + 2.5 * 1j, 2.5, 0.], [0., 0., 25.]])
LLmatrix[(5,-5,-4)] = np.array([[0., 0., -7.905694150420949], [0., 0., 0. + 7.905694150420949 * 1j], [-6.324555320336759, 0. + 6.324555320336759 * 1j, 0.]])
LLmatrix[(5,-5,-3)] = np.array([[3.354101966249685, 0. - 3.354101966249685 * 1j, 0.], [0. - 3.354101966249685 * 1j, -3.354101966249685, 0.], [0., 0., 0.]])
LLmatrix[(5,-5,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-5,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,-5)] = np.array([[0., 0., -6.324555320336759], [0., 0., 0. - 6.324555320336759 * 1j], [-7.905694150420949, 0. - 7.905694150420949 * 1j, 0.]])
LLmatrix[(5,-4,-4)] = np.array([[7., 0. - 2. * 1j, 0.], [0. + 2. * 1j, 7., 0.], [0., 0., 16.]])
LLmatrix[(5,-4,-3)] = np.array([[0., 0., -8.48528137423857], [0., 0., 0. + 8.48528137423857 * 1j], [-6.363961030678928, 0. + 6.363961030678928 * 1j, 0.]])
LLmatrix[(5,-4,-2)] = np.array([[5.196152422706632, 0. - 5.196152422706632 * 1j, 0.], [0. - 5.196152422706632 * 1j, -5.196152422706632, 0.], [0., 0., 0.]])
LLmatrix[(5,-4,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-4,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,-5)] = np.array([[3.354101966249685, 0. + 3.354101966249685 * 1j, 0.], [0. + 3.354101966249685 * 1j, -3.354101966249685, 0.], [0., 0., 0.]])
LLmatrix[(5,-3,-4)] = np.array([[0., 0., -6.363961030678928], [0., 0., 0. - 6.363961030678928 * 1j], [-8.48528137423857, 0. - 8.48528137423857 * 1j, 0.]])
LLmatrix[(5,-3,-3)] = np.array([[10.5, 0. - 1.5 * 1j, 0.], [0. + 1.5 * 1j, 10.5, 0.], [0., 0., 9.]])
LLmatrix[(5,-3,-2)] = np.array([[0., 0., -7.348469228349534], [0., 0., 0. + 7.348469228349534 * 1j], [-4.898979485566356, 0. + 4.898979485566356 * 1j, 0.]])
LLmatrix[(5,-3,-1)] = np.array([[6.48074069840786, 0. - 6.48074069840786 * 1j, 0.], [0. - 6.48074069840786 * 1j, -6.48074069840786, 0.], [0., 0., 0.]])
LLmatrix[(5,-3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-3,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,-4)] = np.array([[5.196152422706632, 0. + 5.196152422706632 * 1j, 0.], [0. + 5.196152422706632 * 1j, -5.196152422706632, 0.], [0., 0., 0.]])
LLmatrix[(5,-2,-3)] = np.array([[0., 0., -4.898979485566356], [0., 0., 0. - 4.898979485566356 * 1j], [-7.348469228349534, 0. - 7.348469228349534 * 1j, 0.]])
LLmatrix[(5,-2,-2)] = np.array([[13., 0. - 1. * 1j, 0.], [0. + 1. * 1j, 13., 0.], [0., 0., 4.]])
LLmatrix[(5,-2,-1)] = np.array([[0., 0., -5.291502622129181], [0., 0., 0. + 5.291502622129181 * 1j], [-2.645751311064591, 0. + 2.645751311064591 * 1j, 0.]])
LLmatrix[(5,-2,0)] = np.array([[7.245688373094719, 0. - 7.245688373094719 * 1j, 0.], [0. - 7.245688373094719 * 1j, -7.245688373094719, 0.], [0., 0., 0.]])
LLmatrix[(5,-2,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-2,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,-3)] = np.array([[6.48074069840786, 0. + 6.48074069840786 * 1j, 0.], [0. + 6.48074069840786 * 1j, -6.48074069840786, 0.], [0., 0., 0.]])
LLmatrix[(5,-1,-2)] = np.array([[0., 0., -2.645751311064591], [0., 0., 0. - 2.645751311064591 * 1j], [-5.291502622129181, 0. - 5.291502622129181 * 1j, 0.]])
LLmatrix[(5,-1,-1)] = np.array([[14.5, 0. - 0.5 * 1j, 0.], [0. + 0.5 * 1j, 14.5, 0.], [0., 0., 1.]])
LLmatrix[(5,-1,0)] = np.array([[0., 0., -2.738612787525831], [0., 0., 0. + 2.738612787525831 * 1j], [0., 0., 0.]])
LLmatrix[(5,-1,1)] = np.array([[7.5, 0. - 7.5 * 1j, 0.], [0. - 7.5 * 1j, -7.5, 0.], [0., 0., 0.]])
LLmatrix[(5,-1,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,-1,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,-2)] = np.array([[7.245688373094719, 0. + 7.245688373094719 * 1j, 0.], [0. + 7.245688373094719 * 1j, -7.245688373094719, 0.], [0., 0., 0.]])
LLmatrix[(5,0,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [-2.738612787525831, 0. - 2.738612787525831 * 1j, 0.]])
LLmatrix[(5,0,0)] = np.array([[15., 0., 0.], [0., 15., 0.], [0., 0., 0.]])
LLmatrix[(5,0,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [2.738612787525831, 0. - 2.738612787525831 * 1j, 0.]])
LLmatrix[(5,0,2)] = np.array([[7.245688373094719, 0. - 7.245688373094719 * 1j, 0.], [0. - 7.245688373094719 * 1j, -7.245688373094719, 0.], [0., 0., 0.]])
LLmatrix[(5,0,3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,0,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,-1)] = np.array([[7.5, 0. + 7.5 * 1j, 0.], [0. + 7.5 * 1j, -7.5, 0.], [0., 0., 0.]])
LLmatrix[(5,1,0)] = np.array([[0., 0., 2.738612787525831], [0., 0., 0. + 2.738612787525831 * 1j], [0., 0., 0.]])
LLmatrix[(5,1,1)] = np.array([[14.5, 0. + 0.5 * 1j, 0.], [0. - 0.5 * 1j, 14.5, 0.], [0., 0., 1.]])
LLmatrix[(5,1,2)] = np.array([[0., 0., 2.645751311064591], [0., 0., 0. - 2.645751311064591 * 1j], [5.291502622129181, 0. - 5.291502622129181 * 1j, 0.]])
LLmatrix[(5,1,3)] = np.array([[6.48074069840786, 0. - 6.48074069840786 * 1j, 0.], [0. - 6.48074069840786 * 1j, -6.48074069840786, 0.], [0., 0., 0.]])
LLmatrix[(5,1,4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,1,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,2,0)] = np.array([[7.245688373094719, 0. + 7.245688373094719 * 1j, 0.], [0. + 7.245688373094719 * 1j, -7.245688373094719, 0.], [0., 0., 0.]])
LLmatrix[(5,2,1)] = np.array([[0., 0., 5.291502622129181], [0., 0., 0. + 5.291502622129181 * 1j], [2.645751311064591, 0. + 2.645751311064591 * 1j, 0.]])
LLmatrix[(5,2,2)] = np.array([[13., 0. + 1. * 1j, 0.], [0. - 1. * 1j, 13., 0.], [0., 0., 4.]])
LLmatrix[(5,2,3)] = np.array([[0., 0., 4.898979485566356], [0., 0., 0. - 4.898979485566356 * 1j], [7.348469228349534, 0. - 7.348469228349534 * 1j, 0.]])
LLmatrix[(5,2,4)] = np.array([[5.196152422706632, 0. - 5.196152422706632 * 1j, 0.], [0. - 5.196152422706632 * 1j, -5.196152422706632, 0.], [0., 0., 0.]])
LLmatrix[(5,2,5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,3,1)] = np.array([[6.48074069840786, 0. + 6.48074069840786 * 1j, 0.], [0. + 6.48074069840786 * 1j, -6.48074069840786, 0.], [0., 0., 0.]])
LLmatrix[(5,3,2)] = np.array([[0., 0., 7.348469228349534], [0., 0., 0. + 7.348469228349534 * 1j], [4.898979485566356, 0. + 4.898979485566356 * 1j, 0.]])
LLmatrix[(5,3,3)] = np.array([[10.5, 0. + 1.5 * 1j, 0.], [0. - 1.5 * 1j, 10.5, 0.], [0., 0., 9.]])
LLmatrix[(5,3,4)] = np.array([[0., 0., 6.363961030678928], [0., 0., 0. - 6.363961030678928 * 1j], [8.48528137423857, 0. - 8.48528137423857 * 1j, 0.]])
LLmatrix[(5,3,5)] = np.array([[3.354101966249685, 0. - 3.354101966249685 * 1j, 0.], [0. - 3.354101966249685 * 1j, -3.354101966249685, 0.], [0., 0., 0.]])
LLmatrix[(5,4,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,4,2)] = np.array([[5.196152422706632, 0. + 5.196152422706632 * 1j, 0.], [0. + 5.196152422706632 * 1j, -5.196152422706632, 0.], [0., 0., 0.]])
LLmatrix[(5,4,3)] = np.array([[0., 0., 8.48528137423857], [0., 0., 0. + 8.48528137423857 * 1j], [6.363961030678928, 0. + 6.363961030678928 * 1j, 0.]])
LLmatrix[(5,4,4)] = np.array([[7., 0. + 2. * 1j, 0.], [0. - 2. * 1j, 7., 0.], [0., 0., 16.]])
LLmatrix[(5,4,5)] = np.array([[0., 0., 6.324555320336759], [0., 0., 0. - 6.324555320336759 * 1j], [7.905694150420949, 0. - 7.905694150420949 * 1j, 0.]])
LLmatrix[(5,5,-5)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,-4)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,-3)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,-2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,-1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,0)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,1)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,2)] = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
LLmatrix[(5,5,3)] = np.array([[3.354101966249685, 0. + 3.354101966249685 * 1j, 0.], [0. + 3.354101966249685 * 1j, -3.354101966249685, 0.], [0., 0., 0.]])
LLmatrix[(5,5,4)] = np.array([[0., 0., 7.905694150420949], [0., 0., 0. + 7.905694150420949 * 1j], [6.324555320336759, 0. + 6.324555320336759 * 1j, 0.]])
LLmatrix[(5,5,5)] = np.array([[2.5, 0. + 2.5 * 1j, 0.], [0. - 2.5 * 1j, 2.5, 0.], [0., 0., 25.]])

# Function to compute LL matrix from the value of the modes hIlm at a given time
def func_LLab(hlm, list_l=[2]):
    LLab = np.zeros((3,3), dtype=complex)
    for l in list_l:
        mvals = [m in range(-l,l+1) if (l,m) in hlm.keys()]
        for m in mvals:
            for mp in mvals:
                for i in range(3):
                    for j in range(3):
                        LLab[i,j] += np.conj(hlm[(l, mp)]) * LLmatrix[(l, m, mp)] * hlm[(l, m)])
    return np.real(LLab)
def func_LLabNorm(hlm):
    LLabnorm = np.zeros((3,3), dtype=complex)
    for l in list_l:
        mvals = [m in range(-l,l+1) if (l,m) in hlm.keys()]
        for m in mvals:
            for mp in mvals:
                for i in range(3):
                    for j in range(3):
                        LLabnorm[i,j] += np.conj(hlm[(l, mp)]) * LLmatrix[(l, m, mp)] * hlm[(l, m)])
    norm = 0.
    for lm in hlm.keys():
        norm += hlm[lm] * np.conj(hlm[lm])
    return np.real(LLabnorm / norm)
# Function to remove by hand, in-place, flips in Z-frame direction
def func_rectify_Zframe(Zframe):
    n = len(Zframe)
    for i in xrange(2, n):
        if np.dot(Zframe[i], Zframe[i-1]) < 0:
            Zframe[i] = -Zframe[i]
    return
# Function to extract Z axis of wave frame -- NOTE: assumes same t-values for all modes
def func_extract_Zframe(hlm):
    modes = hlm.keys()
    lengths = [len(hlm[lm]) for lm in modes]
    if not all(length==lengths[0] for length in lengths):
        raise ValueError('Not all modes have the same length.')
    length = lengths[0]
    Zframe = np.array((length, 3), dtype=real)
    for i in xrange(length):
        hlmval = {}
        for lm in modes:
            hlmval[lm] = hlm[lm][i]
        LLabnorm = func_LLabNorm(hlmval)
        lambd, v = np.linalg.eig(LLabnorm)
        kmax = np.argmax(lambd)
        Zframe[i] = v[:,kmax]
    func_rectify_Zframe(Zframe)
    return Zframe
# Function to break by hand sign degeneracy in Z axis of wave frame

# Functions for Wigner matrices - ABFO convention, differs from Boyle's convention by a transposition
def WignerdMatrix(l, mp, m, beta):
    kmin = max(0, m-mp)
    kmax = min(l+m, l-mp)
    c = np.cos(0.5*beta)
    s = np.sin(0.5*beta)
    return np.sum(sqrt(ffactorial(l-mp)*ffactorial(l+mp)*ffactorial(l-m)*ffactorial(l+m)) * np.array([(-1)**k/ffactorial(k) / (ffactorial(k-m+mp)*ffactorial(l+m-k)*ffactorial(l-mp-k)) * np.power(c, 2*l-mp+m-2*k)*np.power(s, 2*k+mp-m) for k in range(kmin, kmax+1)])) # NOTE: float conversion necessary to avoid type problems with long when multiplying large factorials
WignerdMatrix_vec = np.vectorize(WignerdMatrix, excluded=['l', 'mp', 'm'])
def WignerDMatrix(l, mp, m, alpha, beta, gamma):
    return exp(1j*mp*gamma) * WignerdMatrix(l, mp, m, beta) * exp(1j*m*alpha)
def WignerDMatrixstar(l, mp, m, alpha, beta, gamma):
    return exp(-1j*mp*gamma) * WignerdMatrix(l, mp, m, beta) * exp(-1j*m*alpha)
def WignerDMatrix_vec(l, mp, m, alpha, beta, gamma):
    return exp(1j*mp*gamma) * WignerdMatrix_vec(l, mp, m, beta) * exp(1j*m*alpha)
def WignerDMatrixstar_vec(l, mp, m, alpha, beta, gamma):
    return exp(-1j*mp*gamma) * WignerdMatrix_vec(l, mp, m, beta) * exp(-1j*m*alpha)

# Rotate waveform from frame 1 to frame 2 given Euler angles series for the active rotation 1->2
# Typical use : given Euler angles I->P frame, takes hIlm as input and returns hPlm
def rotate_euler_wf(h1lm, euler, listmodes):
    # Euler angles for the active rotation from the J frame to the xyz frame
    alpha = euler[:,0]
    beta = euler[:,1]
    gamma = euler[:,2]
    # Store conversion factors
    conversionfactor1to2lmmp = {}
    for lm in listmodes:
        conversionfactor1to2lmmp[lm] = np.array([WignerDMatrix_vec(lm[0], lm[1], mp, alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    h2lm = {}
    for lm in listmodes:
        h2lm[lm] = np.sum(np.array([conversionfactor1to2lmmp[lm][lm[0]+mp] * h1lm[(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    return h2lm
# Rotate waveform from frame 2 back to frame 1 given Euler angles series for the active rotation 1->2
# Typical use : given Euler angles I->P frame, takes hPlm as input and returns back hIlm
def rotate_euler_wf_inverse(h2lm, euler, listmodes):
    # Euler angles for the active rotation from the J frame to the xyz frame
    alpha = euler[:,0]
    beta = euler[:,1]
    gamma = euler[:,2]
    # Store conversion factors
    conversionfactor2to1lmmp = {}
    for lm in listmodes:
        conversionfactor2to1lmmp[lm] = np.array([WignerDMatrixstar_vec(lm[0], mp, lm[1], alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    h1lm = {}
    for lm in listmodes:
        h1lm[lm] = np.sum(np.array([conversionfactor2to1lmmp[lm][lm[0]+mp] * h2lm[(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    return h1lm

# Rotate waveform to the J-frame
# Note: we do not modify the input waveform but copy the dictionary that represents it
# NOTE: wf here full waveform (with listmode in particular), not just dictionary hIlm
def rotate_wf_Jframe(wf, Jhat, listmodes=None):
    # List of modes
    if listmodes==None:
        listmodes = wf['listmodes']
    # Frame vectors
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    e3J = Jhat
    e1J = np.cross(Jhat, e1) # Convention e1J along J * x
    e2J = np.cross(Jhat, e1J)
    # Euler angles for the active rotation from the J frame to the xyz frame
    alpha = np.arctan2(np.dot(e2J, e3), np.dot(e1J, e3))
    beta = np.arccos(np.dot(e3J, e3))
    gamma = np.arctan2(np.dot(e3J, e2), -np.dot(e3J, e1))
    # Store conversion factors
    conversionfactorItoJlmmp = {}
    for lm in listmodes:
        conversionfactorItoJlmmp[lm] = np.array([WignerDMatrixstar(lm[0], mp, lm[1], alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    hJlm = {}
    for lm in listmodes:
        hJlm[lm] = np.sum(np.array([conversionfactorItoJlmmp[lm][lm[0]+mp]*wf['hIlm'][(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    wfres = copy.deepcopy(wf)
    for lm in listmodes:
        wfres['hIlm'][lm] = hJlm[lm]
    for lm in listmodes:
        wfres['AIlm'][lm] = ampseries(wfres['hIlm'][lm])
        wfres['phiIlm'][lm] = phaseseries(wfres['hIlm'][lm])
    return wfres

# Function to rotate a waveform from an inertial frame (e1,e2,e3) to a new inertial frame (e1p,e2p,e3p) (constant rotation)
# Input specifies e3p and how to compute e1p, e2p according to various prescriptions
def rotate_wf_toframe(wf, e3p, e1e2framechoice='e1palonge3pcrosse3'):
    # Normalize e3p for forgetful users
    e3phat = normalize(e3p)
    # List of modes
    listmodes = wf['listmodes']
    # Frame vectors of the starting frame
    # All vector components are expressed in this frame
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    if e1e2framechoice=='e1palonge3pcrosse3': # Convention where e1p is along e3p \times e3
        e1p = normalize(np.cross(e3phat, e3))
        e2p = np.cross(e3phat, e1p)
    elif e1e2framechoice=='e1inplanee1pe3p':
        e1p = normalize(e1 - np.dot(e3phat, e1)*e3phat) # Convention where e1 is in the plane (e1p, e3p)
        e2p = np.cross(e3phat, e1p)
    # Euler angles for the active rotation from the (e1p,e2p,e3p) frame to the (e1,e2,e3) frame
    alpha = np.arctan2(np.dot(e2p, e3), np.dot(e1p, e3))
    beta = np.arccos(np.dot(e3phat, e3))
    gamma = np.arctan2(np.dot(e3phat, e2), -np.dot(e3phat, e1))
    # Store conversion factors
    conversionfactoretoeplmmp = {}
    for lm in listmodes:
        conversionfactoretoeplmmp[lm] = np.array([WignerDMatrixstar(lm[0], mp, lm[1], alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    # hprime : modes in the new inertial frame
    hprime = {}
    for lm in listmodes:
        hprime[lm] = np.sum(np.array([conversionfactoretoeplmmp[lm][lm[0]+mp]*wf['hI'][(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    wfprime = copy.deepcopy(wf)
    for lm in listmodes:
        wfprime['hI'][lm] = hprime[lm]
    wfprime['AI'] = {}
    wfprime['phiI'] = {}
    for lm in listmodes:
        wfprime['AI'][lm] = ampseries(wfprime['hI'][lm])
        wfprime['phiI'][lm] = phaseseries(wfprime['hI'][lm])
    return wfprime
# Function to rotate a waveform from an inertial frame to a new inertial frame
# Euler angles alpha, beta, gamma parametrize the active rotation from the old to the new frame in the (z,y,z) convention
def rotate_wf_euler(wf, alpha, beta, gamma):
    # List of modes
    listmodes = wf['listmodes']
    # Store conversion factors
    conversionfactorlmmp = {}
    for lm in listmodes:
        conversionfactorlmmp[lm] = np.array([WignerDMatrix(lm[0], lm[1], mp, alpha, beta, gamma) for mp in range(-lm[0], lm[0]+1)])
    # hprime : modes in the new inertial frame
    hprime = {}
    for lm in listmodes:
        hprime[lm] = np.sum(np.array([conversionfactorlmmp[lm][lm[0]+mp]*wf['hI'][(lm[0],mp)] for mp in range(-lm[0], lm[0]+1)]), axis=0)
    wfprime = copy.deepcopy(wf)
    for lm in listmodes:
        wfprime['hI'][lm] = hprime[lm]
    wfprime['AI'] = {}
    wfprime['phiI'] = {}
    for lm in listmodes:
        wfprime['AI'][lm] = ampseries(wfprime['hI'][lm])
        wfprime['phiI'][lm] = phaseseries(wfprime['hI'][lm])
    return wfprime

# Functions to extract Vf and the Euler angles from a quaternion
def Vf_from_quat(q):
    qvf = q * Quaternions.Quaternion(0,0,0,1) * q.conjugate()
    return np.array([qvf[1],qvf[2],qvf[3]])
def euler_from_quat(q):
    qxprime = q * Quaternions.Quaternion(0,1,0,0) * q.conjugate()
    qyprime = q * Quaternions.Quaternion(0,0,1,0) * q.conjugate()
    qzprime = q * Quaternions.Quaternion(0,0,0,1) * q.conjugate()
    x = np.array([1,0,0]); y = np.array([0,1,0]); z = np.array([0,0,1]);
    xprime =  np.array([qxprime[1],qxprime[2],qxprime[3]])
    yprime =  np.array([qyprime[1],qyprime[2],qyprime[3]])
    zprime =  np.array([qzprime[1],qzprime[2],qzprime[3]])
    alpha = np.arctan2(np.dot(zprime, y), np.dot(zprime, x)) #beware the reversal of arguments
    if np.dot(zprime, z) > 1: # protection against numerical noise that can lead to values off-range by a tiny amount
        beta = 0.
    elif np.dot(zprime, z) < -1:
        beta = pi
    else:
        beta = np.arccos(np.dot(zprime, z))
    gamma = np.arctan2(np.dot(yprime, z), -np.dot(xprime, z)) #beware the reversal of arguments
    return np.array([alpha, beta, gamma])
def euler_from_rotmatrix(R):
    alpha = np.arctan2(R[1, 2], R[0, 2]) #beware the reversal of arguments
    # Here, protection against numerical noise that can lead to values off-range by a tiny amount
    if R[2,2] > 1:
        beta = 0.
    elif R[2,2] < -1:
        beta = pi
    else:
        beta = np.arccos(R[2,2])
    gamma = np.arctan2(R[2,1], -R[2,0]) #beware the reversal of arguments
    return np.array([alpha, beta, gamma])

# Function to extract P-frame waveform and series for the Euler angles from an inertial-frame waveform - uses GWFrames
def extract_Pframe_wf_euler(tI, hIlm, listmodes):
    # Build precessing-frame waveform - uses GWFrames
    T_data = tI
    LM_data = np.array(list(map(list, listmodes)), dtype=np.int32)
    mode_data = np.array([hIlm[lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    W_v3.TransformToCoprecessingFrame();

    iGWF = {}; APlm = {}; phiPlm = {}; hPlm = {};
    for lm in listmodes:
        iGWF[lm] = W_v3.FindModeIndex(lm[0], lm[1])
        APlm[lm] = W_v3.Abs(iGWF[lm])
        phiPlm[lm] = W_v3.ArgUnwrapped(iGWF[lm])
        hPlm[lm] = APlm[lm] * exp(1j*phiPlm[lm])

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    eulerVfseries = np.array(list(map(euler_from_quat, quat)))
    alphaVfseries = np.unwrap(eulerVfseries[:,0])
    betaVfseries = eulerVfseries[:,1]
    gammaVfseries = np.unwrap(eulerVfseries[:,2])
    eulerVfseries = np.array([alphaVfseries, betaVfseries, gammaVfseries]).T
    return [hPlm, eulerVfseries]
# Function to extract series for the Euler angles from an inertial-frame waveform - uses GWFrames
def extract_Pframe_euler(tI, hIlm, listmodes):
    # Build precessing-frame waveform - uses GWFrames
    T_data = tI
    LM_data = np.array(list(map(list, listmodes)), dtype=np.int32)
    mode_data = np.array([hIlm[lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    W_v3.TransformToCoprecessingFrame();

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    eulerVfseries = np.array(list(map(euler_from_quat, quat)))
    alphaVfseries = np.unwrap(eulerVfseries[:,0])
    betaVfseries = eulerVfseries[:,1]
    gammaVfseries = np.unwrap(eulerVfseries[:,2])
    eulerVfseries = np.array([alphaVfseries, betaVfseries, gammaVfseries]).T
    return eulerVfseries
# Function to extract series for the Euler angles, quaternion, log-quaternion, and Vf from an inertial-frame waveform - uses GWFrames
def extract_Pframe_all(tI, hIlm, listmodes):
    # Build precessing-frame waveform - uses GWFrames
    T_data = tI
    LM_data = np.array(list(map(list, listmodes)), dtype=np.int32)
    mode_data = np.array([hIlm[lm] for lm in listmodes])
    W_v3 = GWFrames.Waveform(T_data, LM_data, mode_data)
    W_v3.SetFrameType(1);
    W_v3.SetDataType(1);
    W_v3.TransformToCoprecessingFrame();

    # Time series for the dominant radiation vector
    quat = W_v3.Frame()
    logquat = np.array(list(map(lambda q: q.log(), quat)))
    quatseries = np.array(list(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), quat)))
    logquatseries = np.array(list(map(lambda q: np.array([q[0], q[1], q[2], q[3]]), logquat)))
    Vfseries = np.array(list(map(Vf_from_quat, quat)))
    eulerVfseries = np.array(list(map(euler_from_quat, quat)))
    alphaVfseries = np.unwrap(eulerVfseries[:,0])
    betaVfseries = eulerVfseries[:,1]
    gammaVfseries = np.unwrap(eulerVfseries[:,2])
    eulerVfseries = np.array([alphaVfseries, betaVfseries, gammaVfseries]).T
    return [eulerVfseries, quatseries, logquatseries, Vfseries]
