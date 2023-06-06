# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
import os
import warnings
import shapely.geometry as shp
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import json
import copy

import windPlotting as wplt
import windCAD

from shapely.ops import voronoi_diagram
from typing import List,Literal,Dict,Tuple,Any,Union,Set
from scipy import signal
from scipy.stats import skew,kurtosis
from scipy.interpolate import interp1d


#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================

fps2mps = 0.3048

mm2m = 0.001

DEFAULT_SI_UNITS = {
                    'L':'m',
                    'T':'s',
                    'V':'mps',
                    'P':'Pa',
                    'M':'kg'
                }

VALID_CP_TH_STAT_FIELDS = ['mean','std','peak','peakMin','peakMax','skewness','kurtosis']

VALID_VELOCITY_STAT_FIELDS = ['U','V','W','Iu','Iv','Iw','xLu','xLv','xLw','uv','uw','vw']

SCALABLE_CP_STATS = ['mean','std','peakMin','peakMax']

DEFAULT_PEAK_SPECS = {
                        'method':'gumbel',
                        'fit_method':'BLUE',
                        'Num_seg':16, 
                        'Duration':16, 
                        'prob_non_excd':0.5704,
                    }

DEFAULT_VELOCITY_STAT_FIELDS = ['U','Iu','Iv','Iw','xLu','xLv','xLw','uw']

PATH_SRC = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RF = np.logspace(-5,3,400)

with open(PATH_SRC+r'/refData/bluecoeff.json', 'r') as f:
    BLUE_COEFFS = json.load(f)

#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

#------------------------------- WIND FIELD ------------------------------------
def integTimeScale(x,dt,rho_tol=0.0001,showPlots=False,removeNaNs=True):
    """
    Gets the integral time scale of a signal by integrating the right side of 
    its auto-correlation coefficient function up to the first zero crossing.

    Parameters
    ----------
    x : 1-D float vector
        Input sequence.
    dt : float
        Time step of x.
    rho_tol : float, optional
        The cut-off auto-correlation coefficient beyond which it is assumed 
        to be decorrelated. The default is 0.0001.
    showPlots : bool, optional
        Whether or not to show plots for debugging and whatnot. The default is False.

    Returns
    -------
    I : float
        One-sided integral time scale of x.
    tau_0 : float
        The time lag at which the auto-correlation coefficient first crosses 
        zero.
    tau : 1-D float vector
        The time lag vector from 0 to tau_0.
    rho : 1-D float vector
        The auto-correlation coefficient from zero-lag (i.e., 1) upto tau_0.

    """

    if removeNaNs and not all(~np.isnan(x)):
        x = x[np.argwhere(~np.isnan(x))]
    if len(x) == 0:
        warnings.warn("The input value has no valid number of elements.")
        return np.nan, np.nan, [], []

    x = x - np.mean(x)
    rhoAll = signal.correlate(x,x)
    iMiddle = int(np.floor(len(rhoAll)/2))
    rhoAll = rhoAll/rhoAll[iMiddle]
    rho = rhoAll[iMiddle:]
    
    if all(rho < rho_tol) or not all(~np.isnan(rho)):
        return np.nan, np.nan, [], []
            
    i_0 = np.where(rho < rho_tol)[0][0]
    rho = rho[0:i_0]
    tau = np.linspace(0, (len(rho)-1)*dt, len(rho))
    tau_0 = len(rho)*dt
    I = np.trapz(rho,dx=dt)
    
    if showPlots:
        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        ax = fig1.add_subplot(1,1,1)
        ax.plot(tau,rho,'-k',label='auto-correlation coef.',ms=1)
        plt.xlabel('tau')
        plt.ylabel('rho')
        plt.hlines(rho_tol,0,tau[-1],color='grey')
        plt.hlines(0,0,tau[-1],color='grey')
        ax.legend()
        plt.show()
    
    return I, tau_0, tau, rho

def integLengthScale(x,dt,meanU=None,rho_tol=0.0001,showPlots=False):
    """
    Computes the longitudianl integral length scale of a velocity time history 
    where Taylor's frozen turbulence hypothesis applies (Simiu and Yeo (2019)). 
    Returns xLu, xLv, and xLw components.

    Parameters
    ----------
    x : Velocity time history. If it is non-mean-removed longitudinal component,
        the meanU can automatically be calculated. For v and w components, it 
        has to be given separately, otherwise it would assume it as u component.
    dt : float
        The sampling time step of the velocity time history.
    meanU : float, optional
        The mean longitudinal velocity. If the input vel is not contain all 
        three components or the u component is mean-removed, this value is used. 
        The default is None.
    rho_tol : float, optional
        The cut-off auto-correlation coefficient beyond which it is assumed 
        to be decorrelated. The default is 0.0001.
    showPlots : bool, optional
        Whether or not to show plots for debugging and whatnot. The default is False.

    Returns
    -------
    L : float scalar or 1-D float vector
        The x-integral length scale of each component in vel.

    References:
    -------
    Simiu, E., Yeo, D., 2019. Wind Effects on Structures: Fundamentals and Appilications to Design, 4th edn. ed. John Wiley & Sons Ltd., Oxford, UK.
        
    """
    
    if meanU is None:
        meanU = np.mean(x)
    
    L = meanU * integTimeScale(x-meanU, dt, rho_tol=rho_tol, showPlots=showPlots)[0]
     
    return L

def psd(x,fs,nAvg=8):
    x = x - np.mean(x)
    N = len(x)
    nblock = int(np.floor(N/nAvg))
    overlap = int(np.ceil(nblock/2))
    win = signal.hanning(nblock, True)
    N = nblock*nAvg
    f, Pxxf = signal.welch(x[0:N], fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True)
    return f, Pxxf

def lowpass(x, fs, fc, axis=-1, order = 4, resample=False):
    Wn = fc / (fs / 2)
    b, a = signal.butter(order, Wn, 'low')
    y = signal.filtfilt(b, a, x, axis=axis)
    if resample:
        y = signal.resample(y, int(np.shape(y)[axis] * fc / fs), axis=axis)
    return y

def smooth(x, window_len=11, window:Literal['flat', 'hanning', 'hamming', 'bartlett', 'blackman']='hanning', 
           mode:Literal['same', 'valid', 'full', 'wrap', 'constant']='valid',
           usePadding=True, paddingFactor=0.5, paddingMode='edge'):
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    half_window = np.floor(window_len*paddingFactor).astype(int)
    if usePadding:
        x = np.pad(x, (half_window,half_window), mode=paddingMode)

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode=mode)
    y = y[int(window_len/2-1):-int(window_len/2)]
    if usePadding:
        return y[half_window:-half_window]
    else:
        return y

def vonKarmanSuu(n,U,Iu,xLu):
    return np.divide((4*xLu*(Iu**2)*U), np.power((1 + 70.8*np.power(n*xLu/U,2)),5/6))

def vonKarmanSvv(n,U,Iv,xLv):
    return np.divide( np.multiply( (4*xLv*(Iv**2)*U), (1 + 755.2*np.power(n*xLv/U,2) ) ), 
                                       np.power((1 + 283.1*np.power(n*xLv/U,2)),11/6))

def vonKarmanSww(n,U,Iw,xLw):
    return np.divide( np.multiply( (4*xLw*(Iw**2)*U), (1 + 755.2*np.power(n*xLw/U,2) ) ), 
                                       np.power((1 + 283.1*np.power(n*xLw/U,2)),11/6))

def vonKarmanSpectra(n,U,Iu=None,Iv=None,Iw=None,xLu=None,xLv=None,xLw=None):
    Suu = Svv = Sww = None
    if Iu is not None and xLu is not None:
        Suu = vonKarmanSuu(n,U,Iu,xLu)
    if Iv is not None and xLv is not None:
        Svv = vonKarmanSuu(n,U,Iv,xLv)
    if Iw is not None and xLw is not None:
        Sww = vonKarmanSuu(n,U,Iw,xLw)
    return Suu, Svv, Sww

def fitESDUgivenIuRef(
                    Zref,
                    IuRef,
                    z0i=[1e-7,1e1],
                    Uref=10.0,
                    phi=30,
                    ESDUversion='ESDU85',
                    tolerance=0.0001,
                    ):
    if ESDUversion == 'ESDU85':
        es = ESDU85(z0=z0i[0],Zref=Zref,Uref=Uref,phi=phi)
    elif ESDUversion == 'ESDU74':
        es = ESDU74(z0=z0i[0],Zref=Zref,Uref=Uref,phi=phi)
    else:
        raise Exception("Unknown ESDU version: "+ESDUversion)

    z0_0 = z0i[0]
    z0_1 = z0i[1]

    es.z0 = z0_0
    Iu_0 = es.Iu(Z=Zref)
    es.z0 = z0_1
    Iu_1 = es.Iu(Z=Zref)

    err_0 = IuRef - Iu_0
    err_1 = IuRef - Iu_1

    if err_0*err_1 > 0:
        raise Exception("The initial estimates for z0 do not encompass the root. Change the values and try again.")

    z0 = (z0_0 + z0_1)/2
    es.z0 = z0
    Iu = es.Iu(Z=Zref)
    err = IuRef - Iu

    while abs(err) > tolerance:
        # print("z0 = "+str(z0))
        # print("Iu = "+str(Iu))
        z0 = (z0_0 + z0_1)/2
        es.z0 = z0
        Iu = es.Iu(Z=Zref)
        err = IuRef - Iu
        if err*err_0 < 0:
            z0_1 = z0
            err_1 = err
        else:
            z0_0 = z0
            err_0 = err
    z0 = (z0_0 + z0_1)/2
    return z0, es

def fitVelToLogProfile(Z, U, Zref=None, Uref=None, uStar=None, d=0.0):
    if Zref is None and Uref is None and uStar is None:
        raise Exception("Either 'uStar' or 'Zref' and 'Uref' are required.")
    raise NotImplementedError()

def fitVelToPowerLawProfile(Z, U, Zg=None):
    raise NotImplementedError()

#---------------------------- SURFACE PRESSURE ---------------------------------
def peak_gumbel(x, axis:int=0, 
                specs: dict=DEFAULT_PEAK_SPECS,
                detailedOutput=False, debugMode=False):
    x = np.array(x, dtype=float)
    ndim = x.ndim
    xShp = x.shape
    if xShp[axis] < specs['Num_seg']:
        raise Exception(f"The time dimension of {xShp[axis]} along axis {axis} is less than number of segments, N of {specs['Num_seg']}. The input shape you gave is: {xShp}")
    if debugMode:
        print(f"x: {np.shape(x)}")

    # Extract N min/max values
    if ndim == 1:
        x = np.array_split(x, specs['Num_seg'])
    else:
        x = np.split(x, specs['Num_seg'], axis=axis)
    x = np.moveaxis(x,0,-1)
    if debugMode:
        print(f"x: {np.shape(x)}")

    x_min = np.min(x,axis=axis)
    x_max = np.max(x,axis=axis)

    del x # to save RAM 

    # Sort
    x_max = np.sort(x_max, axis=axis)
    x_min = np.flip(np.sort(x_min, axis=axis),axis=axis)
    if debugMode:
        print(f"x_max: {np.shape(x_max)}")
        print(f"x_min: {np.shape(x_min)}")

    # Get Gumbel coefficients
    if specs['fit_method'] == 'BLUE':
        if specs['Num_seg'] < 4 or specs['Num_seg'] > 100:
            raise NotImplemented()
        ai, bi = np.array(BLUE_COEFFS['ai'][str(specs['Num_seg'])],dtype=float), np.array(BLUE_COEFFS['bi'][str(specs['Num_seg'])],dtype=float)
    else: # This can be extended to other fitting methods (e.g., Gringorton, Leastsquare, etc.) to get ai and bi
        raise NotImplemented()
    ext_dim = [np.newaxis]*x_max.ndim
    ext_dim[axis] = slice(None)
    ai, bi = ai[tuple(ext_dim)], bi[tuple(ext_dim)]
    if debugMode:
        print(f"ai: {np.shape(ai)},  bi: {np.shape(bi)}")
        print(f"axis: {axis}")
        print(f"np.multiply(ai,x_max):  {(np.multiply(ai,x_max))}")
        print(f"np.sum(np.multiply(ai,x_max),axis=axis):  {np.shape(np.sum(np.multiply(ai,x_max),axis=axis))}")

    mu_max, sig_max = np.sum(np.multiply(ai,x_max),axis=axis), np.sum(np.multiply(bi,x_max),axis=axis)
    mu_min, sig_min = np.sum(np.multiply(ai,x_min),axis=axis), np.sum(np.multiply(bi,x_min),axis=axis)

    if debugMode:
        print(f"mu_max: {np.shape(mu_max)},  sig_max: {np.shape(sig_max)}")
        print(f"mu_min: {np.shape(mu_min)},  sig_min: {np.shape(sig_min)}")

    pkMax = mu_max - sig_max*np.log(-np.log(specs['prob_non_excd']))
    pkMin = mu_min - sig_min*np.log(-np.log(specs['prob_non_excd']))

    peakMax = pkMax + sig_max*np.log(specs['Duration'])
    peakMin = pkMin + sig_min*np.log(specs['Duration'])

    details = {
        'ai': ai,
        'bi': bi,
        'x_max':x_max,
        'x_min':x_min,
        'mu_max':mu_max,
        'mu_min':mu_min,
        'sig_max':sig_max,
        'sig_min':sig_min,
    }

    if detailedOutput:
        return peakMin, peakMax, details
    else:
        return peakMin, peakMax

def peak(x,axis=0,
        specs: dict=DEFAULT_PEAK_SPECS,
        debugMode=False, detailedOutput=False,
        ):
    if specs['method'] == 'gumbel':
        return peak_gumbel(x,axis=axis,specs=specs,debugMode=debugMode, detailedOutput=detailedOutput)
    elif specs['method'] == 'minmax':
        return np.amin(x,axis=axis), np.amax(x,axis=axis)
    else:
        raise NotImplemented()

def get_CpTH_stats(TH,axis=0,
                fields: Literal['mean','std','peak','peakMin','peakMax','skewness','kurtosis'] = ['mean','std','peak'],
                peakSpecs: dict=DEFAULT_PEAK_SPECS,
                ) -> dict:
    if not all(x in VALID_CP_TH_STAT_FIELDS for x in fields):
        msg = "Not all elements given as fields are valid. Choose from: "+str(VALID_CP_TH_STAT_FIELDS)
        raise Exception(msg)
    stats = {}
    if 'mean' in fields:
        stats['mean'] = np.mean(TH,axis=axis)
    if 'std' in fields:
        stats['std'] = np.std(TH,axis=axis)
    if 'peak' in fields:
        stats['peakMin'], stats['peakMax'] = peak(TH,axis=axis,specs=peakSpecs)
    if 'peakMin' in fields:
        stats['peakMin'], _ = peak(TH,axis=axis,specs=peakSpecs)
    if 'peakMax' in fields:
        _, stats['peakMax'] = peak(TH,axis=axis,specs=peakSpecs)
    if 'skewness' in fields:
        stats['skewness'] = skew(TH, axis=axis)
    if 'kurtosis' in fields:
        stats['kurtosis'] = kurtosis(TH, axis=axis)
    return stats

def get_velTH_stats(UofT: np.ndarray=None, 
                    VofT: np.ndarray=None, 
                    WofT: np.ndarray=None,
                    timeAxis=1, dt=None,
                    fields: List[Literal['U','V','W','Iu','Iv','Iw','xLu','xLv','xLw','uv','uw','vw']]=DEFAULT_VELOCITY_STAT_FIELDS):
    if not all(x in VALID_VELOCITY_STAT_FIELDS for x in fields):
        msg = "Not all elements given as fields are valid. Choose from: "+str(VALID_VELOCITY_STAT_FIELDS)
        raise Exception(msg)
    uIsHere = UofT is not None
    vIsHere = VofT is not None
    wIsHere = WofT is not None
    if timeAxis == 0:
        UofT = np.transpose(UofT)
        if vIsHere:
            VofT = np.transpose(VofT)
        if wIsHere:
            WofT = np.transpose(WofT)
        timeAxis = 1
    if len(np.shape(UofT)) > 2:
        raise Exception("The velocity time history matrices must be 2D.")
    if vIsHere and np.shape(UofT) != np.shape(VofT):
        raise Exception("The shapes of UofT and VofT must match for profile stat calculation.")
    if wIsHere and np.shape(UofT) != np.shape(WofT):
        raise Exception("The shapes of UofT and WofT must match for profile stat calculation.")
    if dt is None:
        if any(x in fields for x in ['xLu','xLv','xLw']):
            warnings.warn("Time step 'dt' of the time histories is not provided while requesting integral length scales. Skipping those stats.")
            fields = [x for x in fields if x not in ['xLu','xLv','xLw']]
    
    stats = {}
    if uIsHere:
        U = np.mean(UofT,axis=timeAxis)
        u_ofT = UofT - U[:, np.newaxis]
        if 'U' in fields:
            stats['U'] = U
        if 'Iu' in fields:
            stats['Iu'] = np.std(UofT, axis=timeAxis) / U
        if 'xLu' in fields:
            L = np.zeros_like(U)
            for i,u in enumerate(U):
                L[i] = integLengthScale(UofT[i,:], dt, meanU=u)
            stats['xLu'] = L

    if vIsHere:
        V = np.mean(VofT, axis=timeAxis)
        v_ofT = VofT - V[:, np.newaxis]
        if 'V' in fields:
            stats['V'] = V
        if 'Iv' in fields:
            stats['Iv'] = np.std(VofT, axis=timeAxis) / U
        if 'xLv' in fields:
            L = np.zeros_like(U)
            for i,u in enumerate(U):
                L[i] = integLengthScale(VofT[i,:], dt, meanU=u)
            stats['xLv'] = L
        if 'uv' in fields:
            stats['uv'] = np.mean(u_ofT*v_ofT, axis=timeAxis)
    
    if wIsHere:
        W = np.mean(WofT, axis=timeAxis)
        w_ofT = WofT - W[:, np.newaxis]
        if 'W' in fields:
            stats['W'] = W
        if 'Iw' in fields:
            stats['Iw'] = np.std(WofT, axis=timeAxis) / U
        if 'xLw' in fields:
            L = np.zeros_like(U)
            for i,u in enumerate(U):
                L[i] = integLengthScale(WofT[i,:], dt, meanU=u)
            stats['xLw'] = L
        if 'uw' in fields:
            stats['uw'] = np.mean(u_ofT*w_ofT, axis=timeAxis)
        if 'vw' in fields and vIsHere:
            stats['vw'] = np.mean(v_ofT*w_ofT, axis=timeAxis)
    
    return stats

#-------------------------------- PLOTTING -------------------------------------
def formatAxis(ax, gridMajor=True, gridMinor=False, tickLabelSize=10, labelSize=12,
               tickTop=True, tickRight=True, gridColor_mjr=[0.8,0.8,0.8], gridColor_mnr=[0.85,0.85,0.85], 
            #    kwargs_grid_major={'linestyle':'-', 'linewidth':1, 'color':'k'}, 
            #    kwargs_grid_minor={},
               ):
    ax.tick_params(axis='both', which='major', labelsize=tickLabelSize, direction='in', top=tickTop, right=tickRight)
    ax.tick_params(axis='both', which='minor', labelsize=tickLabelSize, direction='in', top=tickTop, right=tickRight)
    ax.xaxis.label.set_size(labelSize)
    ax.yaxis.label.set_size(labelSize)
    if gridMajor:
        ax.grid(True, which='major', linestyle='-', linewidth=1, color=gridColor_mjr)
    if gridMinor:
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color=gridColor_mnr)
    return ax

#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

#------------------------------- WIND FIELD ------------------------------------
class spectra:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self, name=None, UofT=None, VofT=None, WofT=None, samplingFreq=None, 
                 n=None, Suu=None, Svv=None, Sww=None, nSpectAvg=8,
                 Z=None, U=None, Iu=None, Iv=None, Iw=None,
                 xLu=None, xLv=None, xLw=None):
        
        self.name = name
        self.samplingFreq = samplingFreq
        self.n = n
        self.Suu = Suu
        self.Svv = Svv
        self.Sww = Sww
        self.nSpectAvg = nSpectAvg
        
        self.Z = Z
        self.U = U
        self.Iu = Iu
        self.Iv = Iv
        self.Iw = Iw
        self.xLu = xLu
        self.xLv = xLv
        self.xLw = xLw
        
        self.UofT = UofT
        self.VofT = VofT
        self.WofT = WofT

        self.Update()

    def __calculateSpectra(self):
        if self.UofT is not None:
            self.n, self.Suu = psd(self.UofT, self.samplingFreq, nAvg=self.nSpectAvg)
            self.U = np.mean(self.UofT)
            self.Iu = np.std(self.UofT)/self.U
            self.xLu = integLengthScale(self.UofT, 1/self.samplingFreq)

        if self.VofT is not None:
            n, self.Svv = psd(self.VofT, self.samplingFreq, nAvg=self.nSpectAvg)
            self.Iv = np.std(self.VofT)/self.U
            self.xLv = integLengthScale(self.VofT, 1/self.samplingFreq,meanU=self.U)
            if self.n is None:
                self.n = n
            elif not len(n) == len(self.n):
                raise Exception("UofT and VofT have different no. of time steps. The number of time steps in all UofT, VofT, and WofT must match for spectra calculation.")
                
        if self.WofT is not None:
            n, self.Sww = psd(self.WofT, self.samplingFreq, nAvg=self.nSpectAvg)
            self.Iw = np.std(self.WofT)/self.U
            self.xLw = integLengthScale(self.WofT, 1/self.samplingFreq,meanU=self.U)
            if self.n is None:
                self.n = n
            elif not len(n) == len(self.n):
                raise Exception("WofT has different no. of time steps from UofT and VofT. All three must have the same number of time steps for spectra calculation.")
        
    def __str__(self):
        return str(self.name)
    
    """--------------------------------- Normalizers ----------------------------------"""
    def rf(self,n='auto',normZ:Literal['Z','xLi']='Z'):
        if n == 'auto':
            n = self.n
        if normZ == 'Z':
            normZ = self.Z
        elif normZ == 'xLi':
            normZ = self.xLu
        elif not type(normZ) == int or float:
            raise Exception("Unknown normalization height type. Choose from {'Z', 'xLi'} or specify a number.")
        if n is None or self.U is None or normZ is None:
            return None
        else:
            return n * normZ/self.U

    def rSuu(self,normU:Literal['U','sigUi']='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iu * self.U
        if self.n is None or self.Suu is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Suu)/(normU**2)

    def rSvv(self,normU:Literal['U','sigUi']='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iv * self.U
        if self.n is None or self.Svv is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Svv)/(normU**2)

    def rSww(self,normU:Literal['U','sigUi']='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iw * self.U
        if self.n is None or self.Sww is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Sww)/(normU**2)
    
    """-------------------------------- Data handlers ---------------------------------"""
    def Update(self):
        self.__calculateSpectra()

    def writeSpecsToFile(self):
        pass

    def writeDataToFile(self):
        pass

    def writeToJSON(self, filename=None):
        if filename is None:
            filename = self.name+'.json'
        if os.path.exists(filename):
            print("File already exists. Overwrite? (y/n)")
            if input() == 'y':
                os.remove(filename)
            else:
                return
        out = {}
        out['name'] = self.name
        out['samplingFreq'] = self.samplingFreq
        out['n'] = self.n
        out['Suu'] = self.Suu
        out['Svv'] = self.Svv
        out['Sww'] = self.Sww
        out['nSpectAvg'] = self.nSpectAvg
        out['Z'] = self.Z
        out['U'] = self.U
        out['Iu'] = self.Iu
        out['Iv'] = self.Iv
        out['Iw'] = self.Iw
        out['xLu'] = self.xLu
        out['xLv'] = self.xLv
        out['xLw'] = self.xLw
        out['UofT'] = self.UofT
        out['VofT'] = self.VofT
        out['WofT'] = self.WofT
        with open(self.name+'.json', 'w') as outfile:
            json.dump(out, outfile)
    
    def readFromJSON(self, filename=None):
        if filename is None:
            filename = self.name+'.json'
        with open(filename) as json_file:
            data = json.load(json_file)
        self.name = data['name']
        self.samplingFreq = data['samplingFreq']
        self.n = data['n']
        self.Suu = data['Suu']
        self.Svv = data['Svv']
        self.Sww = data['Sww']
        self.nSpectAvg = data['nSpectAvg']
        self.Z = data['Z']
        self.U = data['U']
        self.Iu = data['Iu']
        self.Iv = data['Iv']
        self.Iw = data['Iw']
        self.xLu = data['xLu']
        self.xLv = data['xLv']
        self.xLw = data['xLw']
        self.UofT = data['UofT']
        self.VofT = data['VofT']
        self.WofT = data['WofT']

    """--------------------------------- Fittings -------------------------------------"""
    def Suu_vonK(self,n=None,normalized=False,normU:Literal['U','sigUi']='U'):
        if n is None:
            n = self.n
        Suu = vonKarmanSuu(n=n,U=self.U,Iu=self.Iu,xLu=self.xLu)
        if normalized:
            if normU == 'U':
                normU = self.U
            elif normU == 'sigUi':
                normU = self.Iu * self.U
            return np.multiply(n,Suu)/(normU**2)
        else:
            return Suu

    def Svv_vonK(self,n=None,normalized=False,normU:Literal['U','sigUi']='U'):
        if n is None:
            n = self.n
        Svv = vonKarmanSvv(n=n,U=self.U,Iv=self.Iv,xLv=self.xLv)
        if normalized:
            if normU == 'U':
                normU = self.U
            elif normU == 'sigUi':
                normU = self.Iv * self.U            
            return np.multiply(n,Svv)/(normU**2)
        else:
            return Svv

    def Sww_vonK(self,n=None,normalized=False,normU:Literal['U','sigUi']='U'):
        if n is None:
            n = self.n
        Sww = vonKarmanSww(n=n,U=self.U,Iw=self.Iw,xLw=self.xLw)
        if normalized:
            if normU == 'U':
                normU = self.U
            elif normU == 'sigUi':
                normU = self.Iw * self.U            
            return np.multiply(n,Sww)/(normU**2)
        else:
            return Sww

    def Suu_ESDU74(self):
        pass

    def Svv_ESDU74(self):
        pass

    def Sww_ESDU74(self):
        pass

    def Suu_ESDU85(self):
        pass

    def Svv_ESDU85(self):
        pass

    def Sww_ESDU85(self):
        pass

    """--------------------------------- Plotters -------------------------------------"""
    def plotSpect_any(self, f, S, ax=None, fig=None, figSize=[15,5], label=None, xLabel=None, yLabel=None, xLimits=None, yLimits=None, 
                      plotType: Literal['loglog', 'semilogx', 'semilogy']='loglog', 
                      kwargs_plot={'color': 'k', 'linestyle': '-'}, kwargs_legend={'loc': 'lower left', 'fontsize': 12}, kwargs_ax={}):
        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            ax = fig.add_subplot()

        if plotType == 'loglog':
            ax.loglog(f,S, label=label, **kwargs_plot)
        elif plotType == 'semilogx':
            ax.semilogx(f,S, label=label, **kwargs_plot)
        elif plotType == 'semilogy':
            ax.semilogy(f,S, label=label, **kwargs_plot)
        
        if xLabel is not None:
            ax.set_xlabel(xLabel)
        if yLabel is not None:
            ax.set_ylabel(yLabel)
        if xLimits is not None:
            ax.set_xlim(xLimits)
        if yLimits is not None:
            ax.set_ylim(yLimits)
        
        if newFig:
            fig.legend(**kwargs_legend)
            formatAxis(ax, **kwargs_ax)
            plt.show()
        return fig, ax

    def plot(self, fig=None, ax_Suu=None, ax_Svv=None, ax_Sww=None, figSize=[15,4], label=None, plotSuu=True, plotSvv=True, plotSww=True,
                    xLabel=None, yLabel_Suu=None, yLabel_Svv=None, yLabel_Sww=None,
                    xLimits=None, yLimits=None, 
                    normalize=True, normZ:Literal['Z','xLi']='Z', normU:Literal['U','sigUi']='U',         
                    plotType: Literal['loglog', 'semilogx', 'semilogy']='loglog', avoidZeroFreq=True, 
                    kwargs_Suu={'color': 'k', 'linestyle': '-'}, kwargs_Svv={'color': 'k', 'linestyle': '-'}, kwargs_Sww={'color': 'k', 'linestyle': '-'}, 
                    showLegend=True,  kwargs_legend={'loc': 'lower left', 'fontsize': 12}, 
                    kwargs_ax={}):
        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            axs = fig.subplots(1,3)
            ax_Suu = axs[0]
            ax_Svv = axs[1]
            ax_Sww = axs[2]
        
        label = self.name if label is None else label
        if normalize:
            n = self.rf(normZ=normZ)
            Suu = self.rSuu(normU=normU)
            Svv = self.rSvv(normU=normU)
            Sww = self.rSww(normU=normU)
            
            if xLabel is None:
                xLabel = r'$nH/Uh$' if normZ == 'Z' else r'$^xLi/Uh$'
            if yLabel_Suu is None:
                yLabel_Suu = r'$nS_{uu}/\sigma_u^2$' if normU == 'sigUi' else r'$nS_{uu}/Uh^2$'
            if yLabel_Svv is None:
                yLabel_Svv = r'$nS_{vv}/\sigma_v^2$' if normU == 'sigUi' else r'$nS_{vv}/Uh^2$'
            if yLabel_Sww is None:
                yLabel_Sww = r'$nS_{ww}/\sigma_w^2$' if normU == 'sigUi' else r'$nS_{ww}/Uh^2$'
        else:
            n = self.n
            Suu = self.Suu
            Svv = self.Svv
            Sww = self.Sww

            if xLabel is None:
                xLabel = r'$n$'
            if yLabel_Suu is None:
                yLabel_Suu = r'$S_{uu}$'
            if yLabel_Svv is None:
                yLabel_Svv = r'$S_{vv}$'
            if yLabel_Sww is None:
                yLabel_Sww = r'$S_{ww}$'

        if avoidZeroFreq and n[0] == 0:
            n = n[1:]
            Suu = Suu[1:]
            Svv = Svv[1:]
            Sww = Sww[1:]
        
        if plotSuu and ax_Suu is not None:
            self.plotSpect_any(n, Suu, ax=ax_Suu, fig=fig, figSize=figSize, label=label, xLabel=xLabel, yLabel=yLabel_Suu, xLimits=xLimits, yLimits=yLimits, 
                            kwargs_plot=kwargs_Suu, plotType=plotType)
        if plotSvv and ax_Svv is not None:
            self.plotSpect_any(n, Svv, ax=ax_Svv, fig=fig, figSize=figSize, label=label, xLabel=xLabel, yLabel=yLabel_Svv, xLimits=xLimits, yLimits=yLimits,
                                kwargs_plot=kwargs_Svv, plotType=plotType)
        if plotSww and ax_Sww is not None:
            self.plotSpect_any(n, Sww, ax=ax_Sww, fig=fig, figSize=figSize, label=label, xLabel=xLabel, yLabel=yLabel_Sww, xLimits=xLimits, yLimits=yLimits,
                                kwargs_plot=kwargs_Sww, plotType=plotType)

        if showLegend:
            ax_Suu.legend(**kwargs_legend)

        if newFig:
            formatAxis(ax_Suu, **kwargs_ax)
            formatAxis(ax_Svv, **kwargs_ax)
            formatAxis(ax_Sww, **kwargs_ax)
            plt.show()
        pass

class profile:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                name="profile", 
                profType: Union[Literal["continuous","discrete","scatter"], None] =None, 
                Z=None, H=None, dt=None, t=None,
                UofT: np.ndarray=None, 
                VofT: np.ndarray=None,
                WofT: np.ndarray=None,
                fields=VALID_VELOCITY_STAT_FIELDS,
                stats: Dict ={},
                SpectH: spectra =None, nSpectAvg=8,
                fileName=None,
                keepTH=True,
                interpolateToH=False, units=DEFAULT_SI_UNITS):
        self.name = name
        self.profType = profType
        self.Z: Union[np.ndarray, None] = Z  # [N_pts]
        self.stats: Dict = stats
        self.fields = fields

        self.H = H
        self.dt = dt
        self.samplingFreq = None if dt is None else 1/dt
        self.t = t
        self.UofT: np.ndarray = UofT  # [N_pts x nTime]
        self.VofT: np.ndarray = VofT  # [N_pts x nTime]
        self.WofT: np.ndarray = WofT  # [N_pts x nTime]
        
        self.SpectH : spectra = SpectH
        self.nSpectAvg = nSpectAvg
        
        self.origFileName = fileName
        self.interpolateToH = interpolateToH
        self.units = units

        self.Update()
        if not keepTH:
            self.UofT = None
            self.VofT = None
            self.WofT = None
    
    def __verifyData(self):
        pass

    def __computeVelStats(self):
        if all([self.UofT is None, self.VofT is None, self.WofT is None]):
            return
        N_T = np.shape(self.UofT)[1]
        
        if self.t is None and self.dt is not None:
            self.t = np.linspace(0,(N_T-1)*self.dt,num=N_T)
        if self.dt is None and self.t is not None:
            self.dt = np.mean(np.diff(self.t))

        self.stats = get_velTH_stats(UofT=self.UofT, VofT=self.VofT, WofT=self.WofT, dt=self.dt,
                                     fields=self.fields)
        self.__computeSpectra()
        
    def __computeSpectra(self):
        if self.SpectH is not None:
            return
        uOfT = vOfT = wOfT = None
        if self.UofT is not None:
            uOfT = self.UofT[self.H_idx,:]
        if self.VofT is not None:
            vOfT = self.VofT[self.H_idx,:]
        if self.WofT is not None:
            wOfT = self.WofT[self.H_idx,:]
        
        self.SpectH = spectra(name=self.name, UofT=uOfT, VofT=vOfT, WofT=wOfT, samplingFreq=self.samplingFreq, Z=self.H, nSpectAvg=self.nSpectAvg)

    def __str__(self):
        return self.name

    """----------------------------------- Properties ---------------------------------"""
    @property
    def T(self):
        dur = None
        if self.t is not None and self.dt is not None:
            dur = self.dt * self.t[-1]
        return dur

    @property
    def Tstar(self):
        dur = self.T
        durStar = None
        if dur is not None and self.H is not None and self.Uh is not None:
            durStar = dur * self.Uh / self.H
        return durStar
    
    @property
    def N_pts(self):
        if self.Z is None:
            return 0
        else:
            return len(self.Z)

    @property
    def H_idx(self):
        if self.H is None or self.Z is None:
            return None
        else:
            return np.argmin(np.abs(self.Z-self.H))
    
    @property
    def Uh(self):
        idx = self.H_idx
        if idx is None or self.U is None:
            return None
        else:
            if self.interpolateToH:
                return np.interp(self.H,self.Z,self.U)
            else:
                return self.U[idx]
    
    @property
    def U(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'U' not in self.stats:
            return None
        return self.stats['U']
    
    @property
    def Iu(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'Iu' not in self.stats:
            return None
        return self.stats['Iu']
    
    @property
    def Iv(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'Iv' not in self.stats:
            return None
        return self.stats['Iv']
    
    @property
    def Iw(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'Iw' not in self.stats:
            return None
        return self.stats['Iw']
    
    @property
    def xLu(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'xLu' not in self.stats:
            return None
        return self.stats['xLu']
    
    @property
    def xLv(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'xLv' not in self.stats:
            return None
        return self.stats['xLv']
    
    @property
    def xLw(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'xLw' not in self.stats:
            return None
        return self.stats['xLw']
    
    @property
    def uw(self) -> Union[np.ndarray, None]:
        if self.stats is None or 'uw' not in self.stats:
            return None
        return self.stats['uw']

    """---------------------------------- Normalizers ---------------------------------"""
    @property
    def ZbyH(self) -> Union[np.ndarray, None]:
        if self.H is None or self.Z is None:
            return None
        return self.Z/self.H

    @property
    def UbyUh(self) -> Union[np.ndarray, None]:
        if self.Uh is None or self.U is None:
            return None
        return self.U/self.Uh

    @property
    def xLuByH(self) -> Union[np.ndarray, None]:
        if self.H is None or self.Z is None or self.xLu is None:
            return None
        else:
            return self.xLu/self.H

    @property
    def xLvByH(self) -> Union[np.ndarray, None]:
        if self.H is None or self.Z is None or self.xLv is None:
            return None
        else:
            return self.xLv/self.H

    @property
    def xLwByH(self) -> Union[np.ndarray, None]:
        if self.H is None or self.Z is None or self.xLw is None:
            return None
        else:
            return self.xLw/self.H
        
    @property
    def uwByUh(self) -> Union[np.ndarray, None]:
        if self.Uh is None or self.uw is None:
            return None
        else:
            return self.uw/(self.Uh**2)

    def stat_at_H(self, field):
        if self.H is None or self.stats is None:
            return None
        else:
            return self.stats[field][self.H_idx]

    def stat_norm(self, field):
        if self.stats is None:
            return None, ''
        else:
            if field == 'U':
                return self.UbyUh, 'U/Uh'
            elif field in ['Iu','Iv','Iw']:
                return self.stats[field], field
            elif field in ['xLu','xLv','xLw']:
                if self.H is None:
                    return None, ''
                return self.stats[field]/self.H, field+'/H'
            elif field in ['uv','uw','vw']:
                if self.Uh is None:
                    return None, ''
                return self.stats[field]/(self.Uh**2), field+'/Uh^2'
            else:
                raise NotImplementedError("Normalization for field '{}' not implemented".format(field))
            
    """-------------------------------- Data handlers ---------------------------------"""
    def Update(self):
        self.__verifyData()
        self.__computeVelStats()
        if self.origFileName is not None:
            self.readFromFile(self.origFileName)

    def writeToFile(self,outDir,
                    nameSuffix='',
                    writeTH=False, writeTimeWithTH=False, writeZwithTH=False,
                    writeProfiles=True,writeSpectra=False):
        if writeTH:
            fileName = outDir + "/" + self.name + "_" + nameSuffix + "_U-TH"
            np.save(fileName,self.UofT)
            fileName = outDir + "/" + self.name + "_" + nameSuffix + "_V-TH"
            np.save(fileName,self.VofT)
            fileName = outDir + "/" + self.name + "_" + nameSuffix + "_W-TH"
            np.save(fileName,self.WofT)
            
        if writeProfiles:
            fileName = outDir + "/" + self.name + "_" + nameSuffix + "_profiles.csv"
            M = np.reshape(self.Z,[-1,1])
            header = "Z"
            flds = ["U","Iu","Iv","Iw","xLu","xLv","xLw"]
            for fld in flds:
                ffld = getattr(self,fld)
                if (ffld is not None) and not (ffld.size == 0):
                    M = np.concatenate((M,np.reshape(ffld,[-1,1])), axis=1)
                    header += ", "+fld
            np.savetxt(fileName, M, 
               delimiter=',',header=header,comments='')
            
        if writeSpectra:
            pass
    
    def write(self,outDir):

        pass

    def readFromFile(self,fileName,getHfromU_Uh=False):
        # data = pd.read_csv(fileName)
        # self.Z = data.Z
        # self.U = data.U
        # self.Iu = data.Iu
        # self.Iv = data.Iv
        # self.Iw = data.Iw
        # if 'U_Uh' in data.columns and self.H is None:
        #     idx = (np.abs(data.U_Uh - 1)).argmin()
        #     self.H = self.Z[idx]

        # self.N_pts = len(self.Z)
        # # self.__updateUh()
        pass

    def copy(self):
        return copy.deepcopy(self)

    """--------------------------------- Plotters -------------------------------------"""
    def plotProfile_any(self, fld, ax=None, label=None, normalize=True, overlay_H=True, xLabel=None, yLabel=None, xLimits=None, yLimits=None, 
                        kwargs={'color': 'k', 'linestyle': '-'}, kwargs_ax={}, kwargs_Hline={'color': 'k', 'linestyle': '--', 'linewidth': 0.5}):
        newFig = False
        if ax is None:
            newFig = True
            plt.figure()
            ax = plt.subplot()
        
        label = self.name if label is None else label
        if normalize:
            Z = self.ZbyH
            H = 1.0
            F, name = self.stat_norm(fld)
            xLabel = name if xLabel is None else xLabel
            yLabel = 'Z/H' if yLabel is None else yLabel
        else:
            Z = self.Z
            H = self.H
            F = self.stats[fld]
            xLabel = fld if xLabel is None else xLabel
            yLabel = 'Z' if yLabel is None else yLabel

        ax.plot(F, Z, label=label, **kwargs)
        ax.axvline(x=0, label='_', c='k', ls='-', lw=0.5)
        if overlay_H:
            ax.axhline(y=H, label='_', **kwargs_Hline)
        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        if xLimits is not None:
            ax.set_xlim(xLimits)
        if yLimits is not None:
            ax.set_ylim(yLimits)

        if newFig:
            ax = formatAxis(ax, **kwargs_ax)
            plt.show()
        return ax

    def plotProfile_basic1(self, fig=None, ax_U=None, ax_Iu=None, figsize=[8,6], U_lgnd_lbl=None, Iu_lgnd_lbl=None, normalize=True, overlay_H=True,
                           xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, yLimits=None, showLegend=True,
                           kwargs_legend_U={'fontsize': 12, 'loc': 'upper center', 'ncol': 1}, kwargs_legend_Iu={'fontsize': 12, 'loc': 'center', 'ncol': 1},
                           kwargs_U={'color': 'k', 'linestyle': '-'}, kwargs_Iu={'color': 'k', 'linestyle': '--'}, kwargs_Hline={'color': 'k', 'linestyle': '--'},
                           kwargs_ax={}, kwargs_ax2={}):
        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figsize)
        if ax_U is None:
            newFig = True
            ax_U = plt.subplot()
        if ax_Iu is None:
            ax_Iu = ax_U.twiny()
        
        if normalize:
            U_lgnd_lbl = "U/Uh ("+self.name+")" if U_lgnd_lbl is None else U_lgnd_lbl
        else:
            U_lgnd_lbl = "U ("+self.name+")" if U_lgnd_lbl is None else U_lgnd_lbl
        Iu_lgnd_lbl = "Iu ("+self.name+")" if Iu_lgnd_lbl is None else Iu_lgnd_lbl
        if zLabel is None:
            zLabel = 'Z/H' if normalize else 'Z'
        
        self.plotProfile_any('U', ax=ax_U, label=U_lgnd_lbl, normalize=normalize, overlay_H=overlay_H, kwargs_Hline=kwargs_Hline,
                            xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_U, yLimits=yLimits, kwargs=kwargs_U)
        self.plotProfile_any('Iu', ax=ax_Iu, label=Iu_lgnd_lbl, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iu, yLimits=yLimits, kwargs=kwargs_Iu)
        
        if showLegend:
            ax_U.legend(**kwargs_legend_U)
            ax_Iu.legend(**kwargs_legend_Iu)

        if newFig:
            ax_U = formatAxis(ax_U, **kwargs_ax)
            ax_Iu = formatAxis(ax_Iu, tickRight=False, gridMajor=False, gridMinor=False, **kwargs_ax2)
            plt.show()
        return fig, ax_U, ax_Iu

    def plotProfile_basic2(self, fig=None, axs=None, figsize=[12,10], label=None, normalize=True,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None,
                            yLimits=None, lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt={'color': 'k', 'linestyle': '-'}, kwargs_ax={}):
        newFig = False
        if fig is None:
            newFig = True
            fig, axs = plt.subplots(3,3)
            fig.set_size_inches(figsize)
        
        label = self.name if label is None else label
        if zLabel is None:
            zLabel = 'Z/H' if normalize else 'Z'
        
        if 'U' in self.stats.keys():
            self.plotProfile_any('U', ax=axs[0,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_U, yLimits=yLimits, kwargs=kwargs_plt)
        if 'uw' in self.stats.keys():
            self.plotProfile_any('uw', ax=axs[0,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_uw, yLimits=yLimits, kwargs=kwargs_plt)
        if 'Iu' in self.stats.keys():
            self.plotProfile_any('Iu', ax=axs[1,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iu, yLimits=yLimits, kwargs=kwargs_plt)
        if 'Iv' in self.stats.keys():
            self.plotProfile_any('Iv', ax=axs[1,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iv, yLimits=yLimits, kwargs=kwargs_plt)
        if 'Iw' in self.stats.keys():
            self.plotProfile_any('Iw', ax=axs[1,2], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iw, yLimits=yLimits, kwargs=kwargs_plt)
        if 'xLu' in self.stats.keys():
            self.plotProfile_any('xLu', ax=axs[2,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLu, yLimits=yLimits, kwargs=kwargs_plt)
        if 'xLv' in self.stats.keys():
            self.plotProfile_any('xLv', ax=axs[2,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLv, yLimits=yLimits, kwargs=kwargs_plt)
        if 'xLw' in self.stats.keys():
            self.plotProfile_any('xLw', ax=axs[2,2], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLw, yLimits=yLimits, kwargs=kwargs_plt)
        
        if newFig:
            axs[0,2].axis('off')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fig.legend(handles=axs[0,0].get_lines(), bbox_transform=axs[0,2].transAxes, **lgnd_kwargs)
            for ax in axs.flatten():
                formatAxis(ax, **kwargs_ax)
            plt.show()
        return fig, axs
    
    def plot(self, fig=None, ax_U=None, ax_Iu=None, ax_Spect=None, figsize=None, landscape=True, 
             kwargs_profile={}, 
             kwargs_spect={}):
        newFig = False
        if fig is None:
            newFig = True
            if landscape:
                figsize = [12,5] if figsize is None else figsize
                fig, axs = plt.subplots(1,2)
            else:
                figsize = [5,12] if figsize is None else figsize
                fig, axs = plt.subplots(2,1)
            ax_U = axs[0]
            ax_Iu = ax_U.twiny()
            ax_Spect = axs[1]
            fig.set_size_inches(figsize)

        self.plotProfile_basic1(fig=fig, ax_U=ax_U, ax_Iu=ax_Iu, **kwargs_profile)
        self.SpectH.plot(fig=fig, ax_Suu=ax_Spect, **kwargs_spect)

        if newFig:
            formatAxis(ax_U, tickTop=False, gridMajor=False, gridMinor=False)
            formatAxis(ax_Iu, tickRight=False, gridMajor=False, gridMinor=False)
            formatAxis(ax_Spect, gridMajor=False, gridMinor=False)
            plt.show()
        return fig, ax_U, ax_Iu, ax_Spect
                
    def plotTimeHistory(self,
                    figFile=None,
                    normalizeTime=False,
                    normalizeVel=False,
                    dataLabels='auto', # automatically taken from the profile object
                    xLabel='auto',
                    yLabels=("U(H,t)","V(H,t)","W(H,t)"), 
                    xLimits='auto', # [tMin,tMax]
                    yLimits='auto', # ([Umin, Umax], [Vmin, Vmax], [Wmin, Wmax])
                    figSize=[15, 5],
                    alwaysShowFig=False
                    ):
        if all((self.UofT is None, self.VofT is None, self.WofT is None)):
            raise Exception("At least one of UofT, VofT, or WofT has to be provided to plot time history.")
        if all((self.dt is None, self.t is None)):
            raise Exception("Either dt or t has to be provided to plot time history.")
        
        if self.UofT is not None:
            N_T = np.shape(self.UofT)[1]
            U = self.UofT[self.H_idx,:]/self.Uh if normalizeVel else self.UofT[self.H_idx,:]
        if self.VofT is not None:
            N_T = np.shape(self.VofT)[1]
            V = self.VofT[self.H_idx,:]/self.Uh if normalizeVel else self.VofT[self.H_idx,:] 
        if self.WofT is not None:
            N_T = np.shape(self.WofT)[1]
            W = self.WofT[self.H_idx,:]/self.Uh if normalizeVel else self.WofT[self.H_idx,:] 
        
        if self.t is None:
            self.t = np.linspace(0,(N_T-1)*self.dt,num=N_T)
        if self.dt is None:
            self.dt = np.mean(np.diff(self.t))
        t = self.t * self.Uh / self.H if normalizeTime else self.t

        if xLabel == 'auto':
            xLabel = r't^*' if normalizeTime else 't [s]'
        if yLabels == 'auto':
            yLabels = ("U(t)/Uh","V(t)/Uh","W(t)/Uh") if normalizeVel else ("U(t)","V(t)","W(t)")
        name = self.name if dataLabels == 'auto' else dataLabels
        
        wplt.plotVelTimeHistories(
                                T=(t,),
                                U=(U,),
                                V=(V,),
                                W=(W,),
                                dataLabels=(name,),
                                pltFile=figFile,
                                xLabel=xLabel,
                                yLabels=yLabels,
                                xLimits=xLimits,
                                yLimits=yLimits,
                                figSize=figSize,
                                alwaysShowFig=alwaysShowFig
                                )   


class Profiles:
    def __init__(self, profiles=[]):
        self._currentIndex = 0
        self.profiles:List[profile] = profiles

    def __numOfProfiles(self):
        if self.profiles is None:
            return 0
        return len(self.profiles)

    def __iter__(self):
        return self

    def __next__(self):
        if self._currentIndex < self.__numOfProfiles():
            member = self.profiles[self._currentIndex]
            self._currentIndex += 1
            return member
        self._currentIndex = 0
        raise StopIteration

    @property
    def N(self):
        return len(self.profiles)

    def copy(self):
        return copy.deepcopy(self)

    def plot(self, figsize=None, landscape=True, 
             kwargs_profile={}, 
             kwargs_spect={}):
        if landscape:
            figsize = [12,5] if figsize is None else figsize
            fig, axs = plt.subplots(1,2)
        else:
            figsize = [5,12] if figsize is None else figsize
            fig, axs = plt.subplots(2,1)
        ax_U = axs[0]
        ax_Iu = ax_U.twiny()
        ax_Spect = axs[1]
        fig.set_size_inches(figsize)

        for i, prof in enumerate(self.profiles):
            if i == 0:
                kwargs_profile['overlay_H'] = True
            else:
                kwargs_profile['overlay_H'] = False
            prof.plot(fig=fig, ax_U=ax_U, ax_Iu=ax_Iu, ax_Spect=ax_Spect, kwargs_profile=kwargs_profile, kwargs_spect=kwargs_spect)

        formatAxis(ax_U, tickTop=False, gridMajor=False, gridMinor=False)
        formatAxis(ax_Iu, tickRight=False, gridMajor=False, gridMinor=False)
        formatAxis(ax_Spect, gridMajor=False, gridMinor=False)

    def plot__(self, fig=None, prof_ax=None, spect_ax=None, zLim=None, col=None, 
             linestyle_U=None, linestyle_Iu=None, marker_U=None, marker_Iu=None, mfc_U=None, mfc_Iu=None, linestyle_Spect=None, marker_Spect=None, alpha_Spect=None,
             Iu_factr=100.0, IuLim=[0,100], Ulim=None, IuLgndLoc='upper left', UlgndLoc='upper right',
             fontSz_axNum=10, fontSz_axLbl=12, fontSz_lgnd=12,
             freqLim=None, rSuuLim=None):

        N = len(self.profiles)
        c = plt.cm.Dark2(np.linspace(0,1,N))
        ls = ['solid', 'dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),
              'solid', 'dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),
              'solid', 'dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),
              'solid', 'dashed', 'dashdot', 'dotted', (0, (1, 1)), (0, (5, 10)), (0, (5, 5)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5, 1, 5)),
              ]
        mrkr = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_',
                '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_',
                '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_',
                ]

        fig=plt.figure(figsize=[8, 6]) if fig is None else fig
        ax = plt.subplot(1,2,1) if prof_ax is None else prof_ax
        ax2 = ax.twiny()
        for i,prof in enumerate(self.profiles):
            col_i = c[i] if col is None else col[i]
            ls_U_i = ls[i] if linestyle_U is None else linestyle_U[i]
            ls_Iu_i = ls[i+N] if linestyle_Iu is None else linestyle_Iu[i]
            mrkr_U_i = mrkr[i] if marker_U is None else marker_U[i]
            mrkr_Iu_i = mrkr[i+N] if marker_Iu is None else marker_Iu[i]
            mfc_U_i = col_i if mfc_U is None else mfc_U[i]
            mfc_Iu_i = 'w' if mfc_Iu is None else mfc_Iu[i]
            alpha_S_i = 1.0 if alpha_Spect is None else alpha_Spect[i]

            ax.plot(prof.UbyUh, prof.ZbyH, 
                    ls=ls_U_i, marker=mrkr_U_i, markerfacecolor=mfc_U_i, color=col_i, label=r"$U/U_H$ "+prof.name)
            ax2.plot(prof.Iu*Iu_factr, prof.ZbyH, 
                    ls=ls_Iu_i, marker=mrkr_Iu_i, markerfacecolor=mfc_Iu_i, color=col_i, label=r"$I_u$ "+prof.name)

        ax2.set_xlabel(r"$I_u$",fontsize=fontSz_axLbl)

        if zLim is not None:
            ax.set_ylim(zLim)
            ax2.set_ylim(zLim)
        if IuLim is not None:
            ax2.set_xlim(IuLim)
        if Ulim is not None:
            ax.set_xlim(Ulim)
        ax2.legend(loc=IuLgndLoc,fontsize=fontSz_lgnd)
        ax.set_xlabel(r"$U/U_H$",fontsize=fontSz_axLbl)
        ax.set_ylabel(r"$Z/H$",fontsize=fontSz_axLbl)
        ax.legend(loc=UlgndLoc,fontsize=fontSz_lgnd)
        ax.axhline(y=1.0, color='k',linestyle='--',linewidth=0.5)
        ax.tick_params(axis='both',direction='in',which='both',top=False,right=True)
        ax2.tick_params(axis='both',direction='in',which='both',top=True,right=False)
        ax.tick_params(axis='both', which='major', labelsize=fontSz_axNum)
        ax2.tick_params(axis='both', which='major', labelsize=fontSz_axNum)


        ax = plt.subplot(1,2,2) if spect_ax is None else spect_ax
        for i,prof in enumerate(self.profiles):
            col_i = c[i] if col is None else col[i]
            ls_Spect_i = ls[i] if linestyle_Spect is None else linestyle_Spect[i]
            mrkr_Spect_i = mrkr[i] if marker_Spect is None else marker_Spect[i]
            
            S = prof.SpectH
            if S is None:
                continue
            if S.Suu is None:
                continue
            ax.loglog(S.rf(), S.rSuu(normU='sigUi'),
                       ls=ls_Spect_i, marker=mrkr_Spect_i, color=col_i, label=prof.SpectH.name, alpha=alpha_S_i)
        ax.set_xlabel(r"$nH/U_H$",fontsize=fontSz_axLbl)
        ax.set_ylabel(r"$nS_{uu}/\sigma_u^2$",fontsize=fontSz_axLbl)
        ax.legend(fontsize=fontSz_lgnd)
        if freqLim is not None:
            ax.set_xlim(freqLim)
        if rSuuLim is not None:
            ax.set_ylim(rSuuLim)
        ax.tick_params(axis='both',direction='in',which='both',top=True,right=True)
        ax.tick_params(axis='both', which='major', labelsize=fontSz_axNum)
        return fig
    
    def plotProfiles(self,figFile=None,xLimits='auto',zLimits='auto',figSize=[14,6],normalize=True):
        Z = ()
        val = ()
        names = ()
        # maxU = 0
        if normalize:
            for i in range(self.N):
                Z += (self.profiles[i].Z/self.profiles[i].H,)
                val += (np.transpose(np.stack((self.profiles[i].U/self.profiles[i].Uh, self.profiles[i].Iu, self.profiles[i].Iv, self.profiles[i].Iw))),)
                names += (self.profiles[i].name,)
                # maxU = max(maxU, max(val[i][:,0]))
            xlabels = (r"$U/U_{ref}$",r"$I_u$",r"$I_v$",r"$I_w$")
            zlabel = r"$Z/Z_{ref}$"
        else:
            for i in range(self.N):
                Z += (self.profiles[i].Z,)
                val += (np.transpose(np.stack((self.profiles[i].U, self.profiles[i].Iu, self.profiles[i].Iv, self.profiles[i].Iw))),)
                names += (self.profiles[i].name,)
                # maxU = max(maxU, max(val[i][:,0]))
            xlabels = (r"$U$",r"$I_u$",r"$I_v$",r"$I_w$")
            zlabel = r"$Z$"

        if xLimits is None:
            xLimits = [[0, 2],[0,0.4],[0,0.4],[0,0.3]]

        wplt.plotProfiles(
                        Z, # ([n1,], [n2,], ... [nN,])
                        val, # ([n1,M], [n2,M], ... [nN,M])
                        dataLabels=names, # ("str1", "str2", ... "strN")
                        pltFile=figFile, # "/path/to/plot/file.pdf"
                        xLabels=xlabels, # ("str1", "str2", ... "str_m")
                        yLabel=zlabel,
                        xLimits=xLimits, # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                        yLimits=zLimits, # [zMin, zMax]
                        figSize=figSize,
                        nCols=4
                        )
    
    def plotProfile_basic2(self, figsize=[12,10], label=None, normalize=True,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None,
                            yLimits=None, lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt=None, kwargs_ax={}):
        fig, axs = plt.subplots(3,3)
        fig.set_size_inches(figsize)

        kwargs_plt = [{} if kwargs_plt is None else kwargs_plt[i] for i in range(self.N)]

        for i, prof in enumerate(self.profiles):
            prof.plotProfile_basic2(fig=fig, axs=axs, label=prof.name, normalize=normalize, xLabel=xLabel, zLabel=zLabel, xLimits_U=xLimits_U, xLimits_Iu=xLimits_Iu, 
                                    xLimits_Iv=xLimits_Iv, xLimits_Iw=xLimits_Iw, xLimits_xLu=xLimits_xLu, xLimits_xLv=xLimits_xLv, xLimits_xLw=xLimits_xLw, xLimits_uw=xLimits_uw, 
                                    yLimits=yLimits, kwargs_plt=kwargs_plt[i], kwargs_ax=kwargs_ax)
        
        axs[0,2].axis('off')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fig.legend(handles=axs[0,0].get_lines(), bbox_transform=axs[0,2].transAxes, **lgnd_kwargs)
        for ax in axs.flatten():
            formatAxis(ax, **kwargs_ax)
        plt.show()

    def plotTimeHistory(self,
                    figFile=None,
                    xLabel='t [s]',
                    yLabels=("U(t)","V(t)","W(t)"), 
                    xLimits='auto', # [tMin,tMax]
                    yLimits='auto', # ([Umin, Umax], [Vmin, Vmax], [Wmin, Wmax])
                    figSize=[15, 5],
                    overlay=False,
                    ):
        
        T = U = V = W = names = ()
        for i in range(self.N):
            t = self.profiles[i].t
            dt = self.profiles[i].dt
            iH = self.profiles[i].iH
            UofT = self.profiles[i].UofT
            VofT = self.profiles[i].VofT
            WofT = self.profiles[i].WofT

            if all((UofT is None, VofT is None, WofT is None)):
                raise Exception("At least one of UofT, VofT, or WofT has to be provided to plot time history.")
            if all((dt is None, t is None)):
                raise Exception("Either dt or t has to be provided to plot time history.")
            if UofT is not None:
                N_T = np.shape(UofT)[1]
                U += (UofT[iH,:],)
            if VofT is not None:
                N_T = np.shape(VofT)[1]
                V += (VofT[iH,:],)
            if WofT is not None:
                N_T = np.shape(WofT)[1]
                W += (WofT[iH,:],)
            
            if t is None:
                t = np.linspace(0,(N_T-1)*dt,num=N_T)
            if dt is None:
                dt = np.mean(np.diff(t))
            T += (t,)
            names += (self.profiles[i].name,)
        
        wplt.plotVelTimeHistories(
                                T=T,
                                U=U,
                                V=V,
                                W=W,
                                dataLabels=names,
                                pltFile=figFile,
                                xLabel=xLabel,
                                yLabels=yLabels,
                                xLimits=xLimits,
                                yLimits=yLimits,
                                figSize=figSize
                                )   

    def plotSpectra(self, 
                    figFile=None, 
                    figSize=[15,5], 
                    normalize=True,
                    normZ:Literal['Z','xLi']='Z',
                    normU:Literal['U','sigUi']='U',
                    plotType='loglog',
                    xLimits='auto', # [nMin,nMax]
                    yLimits='auto', # ([SuuMin, SuuMax], [SvvMin, SvvMax], [SwwMin, SwwMax])
                    overlayVonK=False, # Either one entry or array equal to N
                    ):
        overlayVonK = (overlayVonK,)*self.N if isinstance(overlayVonK,bool) else overlayVonK

        n = Suu = Svv = Sww = names = ()
        if normalize:
            for i in range(self.N):
                if self.profiles[i].SpectH is None:
                    continue
                n += (self.profiles[i].SpectH.rf(normZ=normZ),)
                Suu += (self.profiles[i].SpectH.rSuu(normU=normU),)
                Svv += (self.profiles[i].SpectH.rSvv(normU=normU),)
                Sww += (self.profiles[i].SpectH.rSww(normU=normU),)
                names += (self.profiles[i].SpectH.name,)

            if normU == 'U':
                ylabels = (r"$nS_{uu}/U_{ref}^2$",r"$nS_{vv}/U_{ref}^2$",r"$nS_{ww}/U_{ref}^2$")
            elif normU == 'sigUi':
                ylabels = (r"$nS_{uu}/\sigma_u^2$",r"$nS_{vv}/\sigma_v^2$",r"$nS_{ww}/\sigma_w^2$")
            if normZ == 'Z':
                xlabel = r"$n Z_{ref}/U$"
            elif normZ == 'xLi':
                xlabel = r"$n ^xL_u/U$"
            drawXlineAt_rf1 = True
        else:
            for i in range(self.N):
                if self.profiles[i].SpectH is None:
                    continue
                n += (self.profiles[i].SpectH.n,)
                Suu += (self.profiles[i].SpectH.Suu,)
                Svv += (self.profiles[i].SpectH.Svv,)
                Sww += (self.profiles[i].SpectH.Sww,)
                names += (self.profiles[i].SpectH.name,)
            ylabels = (r"$S_{uu}$",r"$S_{vv}$",r"$S_{ww}$")
            xlabel = r"$n [Hz]$"
            drawXlineAt_rf1 = False
        for i in range(self.N):
            if overlayVonK[i]:
                n += (self.profiles[i].SpectH.rf(normZ=normZ),) if normalize else (self.profiles[i].SpectH.n,)
                Suu += (self.profiles[i].SpectH.Suu_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                Svv += (self.profiles[i].SpectH.Svv_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                Sww += (self.profiles[i].SpectH.Sww_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                names += (self.profiles[i].SpectH.name+'-vonK',)

        wplt.plotSpectra(
                        freq=n, # ([n1,], [n2,], ... [nN,])
                        Suu=Suu, # ([n1,], [n2,], ... [nN,])
                        Svv=Svv, # ([n1,], [n2,], ... [nN,])
                        Sww=Sww, # ([n1,], [n2,], ... [nN,])
                        dataLabels=names, # ("str1", "str2", ... "strN")
                        pltFile=figFile, # "/path/to/plot/file.pdf"
                        xLabel=xlabel,
                        yLabels=ylabels, # ("str1", "str2", ... "str_m")
                        xLimits=xLimits, # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                        yLimits=yLimits, # ([SuuMin, SuuMax], [SvvMin, SvvMax], [SwwMin, SwwMax])
                        figSize=figSize,
                        plotType=plotType,
                        drawXlineAt_rf1=drawXlineAt_rf1
                        )

class ESDU74:
    __Omega = 72.7e-6            # Angular rate of rotation of the earth in [rad/s].  ESDU 74031 sect. A.1
    __k = 0.4   # von Karman constant
    def __init__(self,
                    phi=30,
                    z0=0.03,
                    Z=np.sort(np.append(np.logspace(np.log10(0.5),np.log10(300.0),99),10)), 
                    Zref=10.0,
                    Uref=10.0,
                    ):
        self.phi = phi # latitude in degrees
        self.z0 = z0
        self.Z = Z
        self.Zref = Zref
        self.Uref = Uref
        if not self.Zref in self.Z:
            self.Z = np.sort(np.append(self.Z, self.Zref))

    def __str__(self):
        return 'ESDU-74 (z0 = '+str(self.z0)+'m)'

    def d(self):
        return 0.0  # The consideration of zero-displacement is not implemnted. See Section 3.10 of ESDU 72026 to implement it.

    def Zd(self,Z=None):
        Z = self.Z if Z is None else Z
        return Z - self.d()

    def f(self):
        # Coriolis parameter            ESDU 72026, section A.1
        return 2*self.__Omega*np.sin(np.radians(self.phi))

    def Zg(self):
        return 1000*np.power(self.z0,0.18)        # ESDU 74031, eq. A.14

    def Ug(self):
        pass

    def uStar(self):
        if self.z0 is None or self.Uref is None or self.Zref is None or self.phi is None:
            return None
        else:
            return np.divide(0.4*self.Uref, 2.303*np.log10(self.Zref/self.z0) + self.C()*self.f()*self.Zref)  # ESDU 72026, eq. A.2 (only for the lowest 200m)

    def C(self):
        return -20.5858*np.log10(self.f()*self.z0) - 70.4644  # ESDU 72026, Figure A.2

    def checkZ(self,Z):
        if (Z > 200).any():
            raise Warning("The provided Z vector contains values higher than 200m which is beyond the range provided in ESDU 72026, eq. A.2.")

    def U(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return np.multiply(self.uStar()/self.__k, 2.303*np.log10(Z/self.z0) + self.C()*self.f()*Z)    # ESDU 72026, eq. A.2 (only for the lowest 200m)

    def lambda_(self):
        # ESDU 74031, eq. A.4b
        if self.z0 <= 0.02:
            return 1.0
        elif self.z0 <= 1.0:
            return 0.76/(self.z0**0.07)
        else:
            return 0.76 

    def Iu(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        Fu = np.multiply(0.867 + 0.556*np.log10(Z) - 0.246*np.power(np.log10(Z),2), self.lambda_())     # ESDU 74031, eq. A.4a
        return np.divide(Fu, 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def Iv(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        Fv = 0.655 + 0.201*np.log10(Z) - 0.095*np.power(np.log10(Z),2)     # ESDU 74031, eq. A.5
        return np.divide(Fv, 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def Iw(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        Fv = 0.381 + 0.172*np.log10(Z) - 0.062*np.power(np.log10(Z),2)     # ESDU 74031, eq. A.5
        return np.divide(Fv, 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def uPwP(self,Z=None):
        pass         # ESDU 74031, eq. A.10

    def xLu(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return 25*np.divide(np.power(Z,0.35),
                            np.power(self.z0,0.063))     # ESDU 74031, eq. A.16

    def xLv(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return 5.1*np.divide(np.power(Z,0.48),
                            np.power(self.z0,0.086))     # ESDU 74031, eq. A.17

    def xLw(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return 0.35*Z     # ESDU 74031, eq. A.18

    def __rSuu(self,rf=DEFAULT_RF):
        return np.divide(4*rf,
                         np.power(1 + 70.8*np.power(rf,2),5/6) )   # ESDU 74031, eq. 8

    def __rSii(self,rf=DEFAULT_RF):
        return np.divide(np.multiply(4*rf, (1 + 755.2*np.power(rf,2))),
                         np.power(1 + 283.2*np.power(rf,2), 11/6) ) # ESDU 74031, eq. 9 and 10

    def rSuu(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSuu = self.__rSuu(rf)
        std_u = self.Iu(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSuu = _rSuu * (std_u/u)**2
        elif normU == 'sigUi':
            rSuu = _rSuu
        return rSuu

    def rSvv(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSvv = self.__rSii(rf)
        std_v = self.Iv(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSvv = _rSvv * (std_v/u)**2
        elif normU == 'sigUi':
            rSvv = _rSvv
        return rSvv

    def rSww(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSww = self.__rSii(rf)
        std_w = self.Iw(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSww = _rSww * (std_w/u)**2
        elif normU == 'sigUi':
            rSww = _rSww
        return rSww

    def Suu(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLu(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLu(Z), self.U(Z)))
        _rSuu = self.__rSii(rf)
        std_u = self.Iu(Z) * self.U(Z)
        Suu = np.divide(np.multiply(_rSuu,std_u**2), n)
        return Suu
    
    def Svv(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLv(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLv(Z), self.U(Z)))
        _rSvv = self.__rSii(rf)
        std_v = self.Iv(Z) * self.U(Z)
        Svv = np.divide(np.multiply(_rSvv,std_v**2), n)
        return Svv
    
    def Sww(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLw(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLw(Z), self.U(Z)))
        _rSww = self.__rSii(rf)
        std_w = self.Iw(Z) * self.U(Z)
        Sww = np.divide(np.multiply(_rSww,std_w**2), n)
        return Sww

    def rf(self,n,Z=None,normZ:Literal['Z','xLi']='Z'):
        Z = self.Zref if Z is None else Z
        if normZ == 'Z':
            normZ = Z
        elif normZ == 'xLi':
            normZ = self.xLu(Z=Z)
        elif not type(normZ) == int or float:
            raise Exception("Unknown normalization height type. Choose from {'Z', 'xLi'} or specify a number.")
        Uref = self.U(Z=Z)
        return n * normZ/Uref

    def fitToIuRef(self,
                    IuRef,
                    z0i=[1e-7,1e1],
                    tolerance=0.0001,
                    ):
        self.z0 = fitESDUgivenIuRef(
            Zref=self.Zref,
            IuRef=IuRef,
            z0i=z0i,
            Uref=self.Uref,
            phi=self.phi,
            ESDUversion='ESDU74',
            tolerance=tolerance
        )[0]

    def toProfileObj(self,name=None,n=None) -> profile:
        if name is None:
            name = 'ESDU74 (z0='+str(self.z0)+'m)'

        if n is None:
            n = np.multiply(DEFAULT_RF, np.divide(self.Uref, self.Zref))

        Zref = self.Zref
        Suu = self.Suu(n=n, Z=Zref)
        Svv = self.Svv(n=n, Z=Zref)
        Sww = self.Sww(n=n, Z=Zref)
        spect = spectra(name=name, 
                        n=n, Suu=Suu, Svv=Svv, Sww=Sww, 
                        Z=Zref, U=self.U(Zref), Iu=self.Iu(Zref), Iv=self.Iv(Zref), Iw=self.Iw(Zref), 
                        xLu=self.xLu(Zref), xLv=self.xLv(Zref), xLw=self.xLw(Zref) )
        
        stats = {}
        stats['U'] = self.U()
        stats['Iu'] = self.Iu()
        stats['Iv'] = self.Iv()
        stats['Iw'] = self.Iw()
        stats['xLu'] = self.xLu()
        stats['xLv'] = self.xLv()
        stats['xLw'] = self.xLw()

        prof = profile(name=name, 
                profType="continuous", 
                Z=self.Z, H=self.Zref,
                stats=stats,
                SpectH=spect )
        return prof
    
class ESDU85:
    __Omega = 72.9e-6            # Angular rate of rotation of the earth [rad/s] ESDU 82026 ssA1

    def __init__(self,
                    phi=30,
                    z0=0.03,
                    Z=np.sort(np.append(np.logspace(np.log10(0.5),np.log10(300),99),10)), 
                    Zref=10,
                    Uref=10,
                    ):
        self.phi = phi # latitude in degrees
        self.z0 = z0
        self.d = 0.0  # zero-plane displacement not implemented
        self.Z = Z
        self.Zref = Zref
        self.Uref = Uref
        
        if not self.Zref in self.Z:
            self.Z = np.sort(np.append(self.Z, self.Zref))

    def __str__(self):
        return 'ESDU-85 (z0 = '+str(self.z0)+'m)'

    def f(self):
        # Coriolis parameter            ESDU 82026, section A1
        return 2*self.__Omega*np.sin(np.radians(self.phi))

    def uStar(self):
        # Shear velocity.               ESDU 82026, eq. A1.8. (Works for the lowest 300m)
        if self.z0 is None or self.Uref is None or self.Zref is None or self.phi is None:
            return None
        else:
            return (0.4*self.Uref - 34.5*self.f()*self.Zref)/(np.log(self.Zref/self.z0))
    
    def h(self):
        # Boundary layer height         ESDU 82026, eq. A1.6
        return self.uStar()/(6*self.f())

    def U(self,Z=None):
        Z = self.Z if Z is None else Z
        # if (Z > 300).any():
        #     raise Warning("The provided Z vector contains values higher than 300m which is beyond the range provided in ESDU 82026, eq. A1.8.")
        if Z is None or self.z0 is None or self.Uref is None or self.Zref is None:
            return None
        else:
            return np.multiply(self.uStar()/0.4, (np.log(Z/self.z0) + 34.5*self.f()*Z/self.uStar())) # ESDU 82026, eq. A1.8. (Works for the lowest 300m)
    
    def sigUbyUstar(self,Z=None):
        Z = self.Z if Z is None else Z
        eta = 1 - np.divide(6*self.f()*Z, 
                            self.uStar())
        p = eta**16
        return np.divide(7.5*eta * np.power(0.538 + 0.09*np.log(Z/self.z0), p),
                                1 + 0.156*np.log(self.uStar()/(self.f()*self.z0)))      # ESDU 85020, eq. 4.2 

    def Iu(self,Z=None):
        Z = self.Z if Z is None else Z
        return np.multiply( self.sigUbyUstar(Z), np.divide(self.uStar(),
                                                 self.U(Z))   )      # ESDU 85020, eq. 4.1

    def sigVbySigU(self,Z=None):
        Z = self.Z if Z is None else Z
        return 1 - 0.22 * np.power(np.cos(0.5*np.pi*np.divide(Z,self.h())) ,4)   # ESDU 85020, eq. 4.4

    def Iv(self,Z=None):
        Z = self.Z if Z is None else Z
        return np.multiply(self.sigVbySigU(Z),self.Iu(Z))

    def sigWbySigU(self,Z=None):
        Z = self.Z if Z is None else Z
        return 1 - 0.45 * np.power(np.cos(0.5*np.pi*np.divide(Z,self.h())) ,4)   # ESDU 85020, eq. 4.5

    def Iw(self,Z=None):
        Z = self.Z if Z is None else Z
        return np.multiply(self.sigWbySigU(Z),self.Iu(Z))
    
    def uPwP(self,Z=None):
        Z = self.Zref if Z is None else Z
        temp1 = -1*np.multiply(np.multiply(self.Iu(Z),self.Iv(Z)), np.power(self.U(Z),2) )
        temp2 = np.divide(1 - 2*Z/self.h(),
                          np.multiply(np.power(self.sigUbyUstar(Z),2), self.sigWbySigU(Z) ) )
        return np.multiply(temp1,temp2)     # ESDU 85020, eq. 4.7
    
    def A(self,Z=None):
        Z = self.Z if Z is None else Z
        return 0.115 * np.power(1 + 0.315*np.power(1-Z/self.h(),6),2/3)   # ESDU 85020, eq. A2.15

    def xLu(self,Z=None):
        Z = self.Z if Z is None else Z
        Ro = self.uStar()/(self.f()*self.z0) # Rosby number
        N = 1.24 * Ro**0.008    # ESDU 85020, eq. A2.21
        B = 24 * Ro**0.155      # ESDU 85020, eq. A2.20
        K0 = 0.39/(Ro**0.11)    # ESDU 85020, eq. A2.19
        Kz = 0.19 - (0.19-K0)*np.exp(-1*B*np.power(Z/self.h(),N))    # ESDU 85020, eq. A2.18
        temp1 = np.multiply(np.power(self.A(Z),3/2), np.multiply(np.power(self.sigUbyUstar(Z),3), Z))
        temp2 = np.multiply(2.5*np.power(Kz,3/2), np.multiply(np.power(1-Z/self.h(), 2), 1+5.75*Z/self.h() ))
        return np.divide(temp1, temp2)      # ESDU 85020, eq. A2.14

    def xLv(self,Z=None):
        Z = self.Z if Z is None else Z
        return np.multiply(self.xLu(Z), 0.5*np.power(self.sigVbySigU(Z),3))   # ESDU 85020, eq. 7.1

    def xLw(self,Z=None):
        Z = self.Z if Z is None else Z
        return np.multiply(self.xLu(Z), 0.5*np.power(self.sigWbySigU(Z),3))   # ESDU 85020, eq. 7.2

    def alpha(self,Z=None):
        Z = self.Zref if Z is None else Z
        return 0.535 + 2.76*np.power(0.138-self.A(Z), 0.68)   # ESDU 85020, eq. B5.10

    def beta1(self,Z=None):
        Z = self.Zref if Z is None else Z
        return 2.357*self.alpha(Z) - 0.761    # ESDU 85020, eq. B5.8

    def beta2(self,Z=None):
        Z = self.Zref if Z is None else Z
        return 1 - self.beta1(Z)     # ESDU 85020, eq. B5.7
        
    def F2(self,ni,Z=None):
        Z = self.Zref if Z is None else Z
        return 1 + 2.88*np.exp(-0.218*np.divide(ni,np.power(self.alpha(Z),-0.9)))   # ESDU 85020, eq. B4.4

    def __rSuu(self,nu=DEFAULT_RF,Z=None):
        Z = self.Zref if Z is None else Z
        F1 = 1 + 0.455*np.exp(-0.76*np.divide(nu,np.power(self.alpha(Z),-0.8)))   # ESDU 85020, eq. B4.3
        term1 = np.multiply(self.beta1(Z), np.divide(2.987*np.divide(nu,self.alpha(Z)),
                                            np.power(1 + np.power(np.divide(2*np.pi*nu,self.alpha(Z)) ,2) ,5/6) ) 
                                            )
        term2 = np.multiply(self.beta2(Z), np.multiply(np.divide(np.divide(1.294*nu, self.alpha(Z)),
                                                         np.power(1 + np.power(np.divide(np.pi*nu, self.alpha(Z)),2) ,5/6) )  ,F1) )
        rSuu = term1 + term2  # ESDU 85020, eq. B4.1
        return rSuu

    def __rSii(self,ni,Z=None):
        Z = self.Zref if Z is None else Z
        term1 = np.multiply(self.beta1(Z), np.divide(2.987* np.multiply(1 + (8/3)*np.power(np.divide(4*np.pi*ni,self.alpha(Z)),2), ni/self.alpha(Z)) ,
                                            np.power(1 + np.power(np.divide(4*np.pi*ni,self.alpha(Z)) ,2), 11/6) ) 
                                            )
        term2 = np.multiply(self.beta2(Z), np.multiply(np.divide(np.divide(1.294*ni, self.alpha(Z)),
                                                         np.power(1 + np.power(np.divide(2*np.pi*ni, self.alpha(Z)),2) ,5/6) )  ,self.F2(ni,Z)) )
        return term1 + term2  # ESDU 85020, eq. B4.1

    def rSuu(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSuu = self.__rSuu(rf)
        std_u = self.Iu(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSuu = _rSuu * (std_u/u)**2
        elif normU == 'sigUi':
            rSuu = _rSuu
        return rSuu

    def rSvv(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSvv = self.__rSii(rf)
        std_v = self.Iv(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSvv = _rSvv * (std_v/u)**2
        elif normU == 'sigUi':
            rSvv = _rSvv
        return rSvv

    def rSww(self,rf=DEFAULT_RF,Z=None,normU:Literal['U','sigUi']='U'):
        Z = self.Zref if Z is None else Z
        _rSww = self.__rSii(rf)
        std_w = self.Iw(Z) * self.U(Z)
        if normU == 'U':
            u = self.U(Z)
            rSww = _rSww * (std_w/u)**2
        elif normU == 'sigUi':
            rSww = _rSww
        return rSww

    def Suu(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLu(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLu(Z), self.U(Z)))
        _rSuu = self.__rSii(rf)
        std_u = self.Iu(Z) * self.U(Z)
        Suu = np.divide(np.multiply(_rSuu,std_u**2), n)
        return Suu
    
    def Svv(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLv(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLv(Z), self.U(Z)))
        _rSvv = self.__rSii(rf)
        std_v = self.Iv(Z) * self.U(Z)
        Svv = np.divide(np.multiply(_rSvv,std_v**2), n)
        return Svv
    
    def Sww(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            rf = DEFAULT_RF
            n = np.multiply(rf, np.divide(self.U(Z), self.xLw(Z)))
        else:
            rf = np.multiply(n, np.divide(self.xLw(Z), self.U(Z)))
        _rSww = self.__rSii(rf)
        std_w = self.Iw(Z) * self.U(Z)
        Sww = np.divide(np.multiply(_rSww,std_w**2), n)
        return Sww

    def rf(self,n,Z=None,normZ:Literal['Z','xLi']='Z'):
        Z = self.Zref if Z is None else Z
        if normZ == 'Z':
            normZ = Z
        elif normZ == 'xLi':
            normZ = self.xLu(Z=Z)
        elif not type(normZ) == int or float:
            raise Exception("Unknown normalization height type. Choose from {'Z', 'xLi'} or specify a number.")
        Uref = self.U(Z=Z)
        return n * normZ/Uref

    def fitToIuRef(self,
                    IuRef,
                    z0i=[1e-7,1e1],
                    tolerance=0.0001,
                    ):
        self.z0 = fitESDUgivenIuRef(
            Zref=self.Zref,
            IuRef=IuRef,
            z0i=z0i,
            Uref=self.Uref,
            phi=self.phi,
            ESDUversion='ESDU85',
            tolerance=tolerance
        )[0]

    def toProfileObj(self,name=None,n=None) -> profile:
        if name is None:
            name = 'ESDU85 (z0='+str(self.z0)+'m)'

        if n is None:
            n = np.multiply(DEFAULT_RF, np.divide(self.Uref, self.Zref))

        Zref = self.Zref
        Suu = self.Suu(n=n, Z=Zref)
        Svv = self.Svv(n=n, Z=Zref)
        Sww = self.Sww(n=n, Z=Zref)
        spect = spectra(name=name, 
                        n=n, Suu=Suu, Svv=Svv, Sww=Sww, 
                        Z=Zref, U=self.U(Zref), Iu=self.Iu(Zref), Iv=self.Iv(Zref), Iw=self.Iw(Zref), 
                        xLu=self.xLu(Zref), xLv=self.xLv(Zref), xLw=self.xLw(Zref) )
        
        stats = {}
        stats['U'] = self.U()
        stats['Iu'] = self.Iu()
        stats['Iv'] = self.Iv()
        stats['Iw'] = self.Iw()
        stats['xLu'] = self.xLu()
        stats['xLv'] = self.xLv()
        stats['xLw'] = self.xLw()

        prof = profile(name=name, 
                profType="continuous", 
                Z=self.Z, H=self.Zref,
                stats=stats,
                SpectH=spect )
        return prof

#---------------------------- SURFACE PRESSURE ---------------------------------
class faceCp(windCAD.face):
    def __init__(self, 
                 ID=None, name=None, note=None, origin=None, basisVectors=None, origin_plt=None, basisVectors_plt=None, vertices=None, tapNo: List[int] = None, tapIdx: List[int] = None, tapName: List[str] = None, badTaps=None, allBldgTaps=None, tapCoord=None, zoneDict=None, nominalPanelAreas=None, numOfNominalPanelAreas=5, file_basic=None, file_derived=None):
        super().__init__(ID, name, note, origin, basisVectors, origin_plt, basisVectors_plt, vertices, tapNo, tapIdx, tapName, badTaps, allBldgTaps, tapCoord, zoneDict, nominalPanelAreas, numOfNominalPanelAreas, file_basic, file_derived)

    @property
    def Cf_N(self):
        pass

class bldgCp(windCAD.building):
    def __init__(self, 
                # Inputs for the base class
                bldgName=None, H=None, He=None, Hr=None, B=None, D=None, roofSlope=None, lScl=1, valuesAreScaled=True, 
                faces: List[windCAD.face] = [], faces_file_basic=None, faces_file_derived=None,
                # Inputs for the derived class
                caseName=None,
                notes_Cp=" ",
                AoA_zero_deg_basisVector=None,
                AoA_rotation_direction:Literal['CW','CCW']=None,
                refProfile:profile=None,
                Zref_input=None,  # for the Cp TH being input below
                Uref_input=None,  # for the Cp TH being input below
                Uref_FS=None,     # Full-scale reference velocity for scaling purposes
                samplingFreq=None,
                airDensity=1.125,
                AoA=None,
                CpOfT=None,  # Cp TH referenced to Uref at Zref
                badTaps=None, # tap numbers to remove
                reReferenceCpToH=True, # whether or not to re-reference Cp to building height velocity
                pOfT=None,
                p0ofT=None,
                CpStats=None,
                peakSpecs=DEFAULT_PEAK_SPECS,
                keepTH=True,
                ):
        super().__init__(name=bldgName, H=H, He=He, Hr=Hr, B=B, D=D, roofSlope=roofSlope, lScl=lScl, valuesAreScaled=valuesAreScaled, faces=faces,
                        faces_file_basic=faces_file_basic, faces_file_derived=faces_file_derived)

        self.name = caseName
        self.notes_Cp = notes_Cp
        self.refProfile:profile = refProfile
        self.samplingFreq = samplingFreq
        self.airDensity = airDensity

        self.Zref = Zref_input
        self.Uref = [Uref_input,] if np.isscalar(Uref_input) else Uref_input
        self.Uref_FS = Uref_FS
        self.badTaps = badTaps
        self.AoA = [AoA,] if np.isscalar(AoA) else AoA          # [N_AoA]
        self.CpOfT = CpOfT      # [N_AoA,Ntaps,Ntime]
        self.pOfT = pOfT        # [N_AoA,Ntaps,Ntime]
        self.p0ofT = p0ofT      # [N_AoA,Ntime]
        self.CpStats = CpStats          # dict{nFlds:[N_AoA,Ntaps]}
        self.peakSpecs = peakSpecs
        
        self.CpStatsAreaAvg = None      # dict{nFlds:[Nzones][N_AoA,Npanels]}
        self.velRatio = None

        self.__handleBadTaps()
        if reReferenceCpToH:
            self.__reReferenceCp()
        if self.Uref_FS is not None and self.Uref is not None and self.lScl is not None:
            self.vScl = np.mean(self.Uref)/self.Uref_FS
            self.tScl = self.lScl/self.vScl
        else:
            self.vScl = self.tScl = None
        self.Update()
        if not keepTH:
            self.CpOfT = self.pOfT = self.p0ofT = None

    def __verifyData(self):
        pass

    def __handleBadTaps(self):
        if self.badTaps is None:
            return
        tapNo = np.array(self.tapNo_all)
        badIdx = np.where(np.in1d(tapNo, self.badTaps))[0]

        if self.CpOfT is not None:
            self.CpOfT = np.delete(self.CpOfT, badIdx, axis=1)
        if self.pOfT is not None:
            self.pOfT = np.delete(self.pOfT, badIdx, axis=1)
        if self.CpStats is not None:
            for fld in self.CpStats:
                self.CpStats[fld] = np.delete(self.CpStats[fld], badIdx, axis=1)

    def __computeCpTHfrom_p_TH(self):
        if self.CpOfT is not None:
            return
        if self.pOfT is None and self.p0ofT is None:
            return
        p0ofT = 0.0 if self.p0ofT is None else self.p0ofT
        if not np.isscalar(p0ofT) and not np.shape(p0ofT)[-1] == np.shape(self.pOfT)[-1]:
            raise Exception(f"The p and p0 time series for Cp calculation do not match in time steps. Shapes of p0ofT : {np.shape(p0ofT)}, pOfT : {np.shape(self.pOfT)}")
        pOfT = np.empty(np.shape(self.pOfT))
        self.CpOfT = np.divide(np.subtract(self.pOfT,p0ofT),
                                0.5*self.airDensity*self.Uref**2)

    def __computeAreaAveragedCp_____depricated(self):
        if self.NumPanels == 0 or self.CpOfT is None:
            return
        
        axT = len(np.shape(self.CpOfT))-1
        nT = np.shape(self.CpOfT)[-1]
        nAoA = self.NumAoA
        self.CpStatsAreaAvg = [] # [Nzones][N_AoA,Npanels]

        for z,(wght_z,idx_z) in enumerate(zip(self.tapWeightsPerPanel,self.tapIdxsPerPanel)):
            for p,(wght,idx) in enumerate(zip(wght_z,idx_z)):
                cpTemp = np.multiply(np.reshape(wght,(-1,1)), self.CpOfT[:,idx,:])
                cpTemp = np.reshape(np.sum(cpTemp,axis=1), [nAoA,1,nT])
                if p == 0:
                    avgCp = get_CpTH_stats(cpTemp,axis=axT,peakSpecs=self.peakSpecs)
                else:
                    temp = get_CpTH_stats(cpTemp,axis=axT,peakSpecs=self.peakSpecs)
                    for fld in temp:
                        avgCp[fld] = np.concatenate((avgCp[fld],temp[fld]),axis=1)

            self.CpStatsAreaAvg.append(avgCp)

    def __computeAreaAveragedCp(self):
        if self.NumPanels == 0 or self.CpOfT is None:
            return
        
        axT = len(np.shape(self.CpOfT))-1
        nT = np.shape(self.CpOfT)[-1]
        nAoA = self.NumAoA
        self.CpStatsAreaAvg = [] # [Nfaces][Nzones][Narea][N_AoA,Npanels]

        for _, fc in enumerate(self.faces):
            avgCp_f = []
            for _,(wght_z,idx_z) in enumerate(zip(fc.tapWghtPerPanel,fc.tapIdxPerPanel)):
                avgCp_z = []
                for _, (wght_a,idx_a) in enumerate(zip(wght_z,idx_z)):
                    for p,(wght,idx) in enumerate(zip(wght_a,idx_a)):
                        cpTemp = np.multiply(np.reshape(wght,(-1,1)), self.CpOfT[:,idx,:])
                        cpTemp = np.reshape(np.sum(cpTemp,axis=1), [nAoA,1,nT])
                        if p == 0:
                            avgCp = get_CpTH_stats(cpTemp,axis=axT,peakSpecs=self.peakSpecs)
                        else:
                            temp = get_CpTH_stats(cpTemp,axis=axT,peakSpecs=self.peakSpecs)
                            for fld in temp:
                                avgCp[fld] = np.concatenate((avgCp[fld],temp[fld]),axis=1)
                    avgCp_z.append(avgCp)
                avgCp_f.append(avgCp_z)
            self.CpStatsAreaAvg.append(avgCp_f)

    def __reReferenceCp(self):
        if self.refProfile is None or self.Zref is None or self.Uref is None or self.AoA is None:
            return
        if self.CpOfT is None and self.CpStats is None:
            return
        
        vel = self.refProfile
        UofZ = interp1d(vel.Z, vel.U)
        self.velRatio = UofZ(self.Zref) / vel.Uh
        factor = (self.velRatio)**2
        self.Zref = vel.H
        self.Uref = self.Uref * (1/self.velRatio)
        if self.CpOfT is not None:
            self.CpOfT = self.CpOfT*factor
        if self.CpStats is not None:
            for fld in self.CpStats:
                if fld in SCALABLE_CP_STATS:
                    self.CpStats[fld] = self.CpStats[fld]*factor

    def __str__(self):
        return self.name

    @property
    def NumAoA(self) -> int:
        if self.AoA is None:
            return 0
        if np.isscalar(self.AoA):
            return 1
        return len(self.AoA)

    @property
    def CpStatEnvlp(self) -> dict:
        if self.CpStats is None:
            return None
        stats = self.CpStats.copy()
        for s in stats:
            arr = self.CpStats[s]
            abs_max = np.amax(np.abs(arr), axis=0, keepdims=True)
            stats[s] = np.squeeze(abs_max * np.sign(arr))
        return stats
    
    @property
    def CpStatEnvlp_high(self) -> dict:
        if self.CpStats is None:
            return None
        stats = self.CpStats.copy()
        for s in stats:
            stats[s] = np.max(self.CpStats[s], axis=0)
        return stats

    @property
    def CpStatEnvlp_low(self) -> dict:
        if self.CpStats is None:
            return None
        stats = self.CpStats.copy()
        for s in stats:
            stats[s] = np.min(self.CpStats[s], axis=0)
        return stats
    
    def CpStatsAreaAvgCollected(self, mixNominalAreasFromAllZonesAndFaces=False, 
                        envelope:Literal['max','min','none']='none', 
                        extremesPerNominalArea:Literal['max','min','none']='none'):
        # [Nfaces][Nzones][Narea][Nflds][N_AoA,Npanels]
        zNames = []
        for z, zn in enumerate(self.zoneDict):
            zNames.append(self.zoneDict[zn][0]+'_'+self.zoneDict[zn][1])
        CpAavg = self.zoneDict
        for zm, zone_m in enumerate(CpAavg):
            if mixNominalAreasFromAllZonesAndFaces:
                CpAavg[zone_m][2] = {}
                for _, fld in enumerate(self.CpStatsAreaAvg[0][zm][0]):
                    CpAavg[zone_m][2][fld] = None                
            else:
                CpAavg[zone_m][2] = []
                for a, _ in enumerate(self.faces[0].nominalPanelAreas):
                    CpAavg[zone_m][2].append({})
                    for _, fld in enumerate(self.CpStatsAreaAvg[0][zm][0]):
                        CpAavg[zone_m][2][a][fld] = None

        def envelopeFld(fld): # assumes all fields have the AoA as the first index
            if envelope == 'max':
                return np.max(fld,axis=0,keepdims=True) 
            elif envelope == 'min':
                return np.min(fld,axis=0,keepdims=True)
            elif envelope == 'none':
                return fld

        def extremePerArea(fld):
            if extremesPerNominalArea == 'max':
                return np.max(fld,axis=1,keepdims=True) 
            elif extremesPerNominalArea == 'min':
                return np.min(fld,axis=1,keepdims=True)
            elif extremesPerNominalArea == 'none':
                return fld

        for f,fc in enumerate(self.faces):
            for z,zone in enumerate(fc.zoneDict):
                zIdx = zNames.index(fc.zoneDict[zone][0]+'_'+fc.zoneDict[zone][1])
                for a,_ in enumerate(fc.nominalPanelAreas):
                    for _, fld in enumerate(self.CpStatsAreaAvg[f][z][a]):
                        if mixNominalAreasFromAllZonesAndFaces:
                            if CpAavg[zIdx][2][fld] is None:
                                CpAavg[zIdx][2][fld] = {}
                                CpAavg[zIdx][2][fld] = envelopeFld(self.CpStatsAreaAvg[f][z][a][fld])
                            else:
                                CpAavg[zIdx][2][fld] = np.concatenate((CpAavg[zIdx][2][fld], envelopeFld(self.CpStatsAreaAvg[f][z][a][fld])), axis=1)
                        else:
                            if CpAavg[zIdx][2][a][fld] is None:
                                CpAavg[zIdx][2][a][fld] = {}
                                CpAavg[zIdx][2][a][fld] = envelopeFld(self.CpStatsAreaAvg[f][z][a][fld])
                            else:
                                CpAavg[zIdx][2][a][fld] = np.concatenate((CpAavg[zIdx][2][a][fld], envelopeFld(self.CpStatsAreaAvg[f][z][a][fld]) ), axis=1)
        if not extremesPerNominalArea == 'none' and not mixNominalAreasFromAllZonesAndFaces:
            __CpAavg = CpAavg
            CpAavg = self.zoneDict
            for zm, zone_m in enumerate(CpAavg):
                CpAavg[zone_m][2] = {}
                for _, fld in enumerate(self.CpStatsAreaAvg[0][zm][0]):
                    CpAavg[zone_m][2][fld] = np.zeros_like(self.faces[0].nominalPanelAreas)
                    for a, _ in enumerate(self.faces[0].nominalPanelAreas):
                        CpAavg[zone_m][2][fld][a] = np.squeeze(extremePerArea(__CpAavg[zone_m][2][a][fld]))
        return CpAavg

    def Update(self):
        self.__verifyData()
        self.__computeCpTHfrom_p_TH()
        if self.CpOfT is not None:
            self.CpStats = get_CpTH_stats(self.CpOfT,axis=len(np.shape(self.CpOfT))-1,peakSpecs=self.peakSpecs)
        self.__computeAreaAveragedCp()
    
    def write(self):
        pass

    def checkStatField(self,fld):
        if self.CpStats is None:
            raise Exception(f"CpStats is not defined.")
        if fld in self.CpStats:
            return True
        else:
            raise Exception(f"The field {fld} is not a part of the available stat fields. Available stat fields: {list(self.CpStats.keys())}")
    
    """--------------------------------- Plotters -------------------------------------"""
    def plotTapCpStatsPerAoA(self, figs=None, all_axes=None,
                            fields=['peakMin','mean','peakMax',],fldRange=[-15,10], tapsToPlot=None, includeTapName=True,
                            nCols=7, nRows=10, cols = ['r','k','b','g','m','r','k','b','g','m'],mrkrs = ['v','o','^','s','p','d','.','*','<','>','h'], 
                            ls=['-','-','-','-','-','-','-','-','-','-',],
                            xticks=None, mrkrSize=2,
                            legend_bbox_to_anchor=(0.5, 0.905), pageNo_xy=(0.5,0.1), figsize=[15,20], sharex=True, sharey=True,
                            overlayThis=None, overlay_AoA=None, overlayLabel=None, kwargs_overlay={}):
        tapIdxs = self.tapIdx if tapsToPlot is None else self.idxOfTapNum(tapsToPlot)
        
        for fld in fields:
            self.checkStatField(fld)
        nPltPerPage = nCols * nRows
        nPltTotal = len(tapIdxs)
        nPages = int(np.ceil(nPltTotal/nPltPerPage))

        tapPltCount = 0
        tapIdx = tapIdxs[tapPltCount]
        
        newFig = False
        if figs is None:
            newFig = True
            figs = []
            all_axes = []
        for p in range(nPages):
            # fig = plt.figure(figsize=figsize)
            if newFig:
                fig, axs = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex, sharey=sharey)
                all_axes.append(axs)
                figs.append(fig)
            else:
                axs = all_axes[p]
                # fig = figs[p]
                
            for i in range(nPltPerPage):
                if tapPltCount >= nPltTotal:
                    break
                ax = axs[i//nCols,i%nCols]
                # ax = plt.subplot(nRows,nCols,i+1)
                for f,fld in enumerate(fields):
                    ax.plot(self.AoA, self.CpStats[fld][:,tapIdx], label=fld,
                            marker=mrkrs[f], color=cols[f], ls=ls[f],mfc=cols[f], ms=mrkrSize)
                    if overlayThis is not None:
                        ax.plot(overlay_AoA, overlayThis[fld][:,tapPltCount], label=overlayLabel, color=cols[f],**kwargs_overlay)
                ax.hlines([-1,0,1],0,360,colors=['k','k','k'],linestyles=['--','-','--'],lw=0.7)
                if includeTapName:
                    if self.tapName is not None and self.tapName[tapIdx] != '':
                        tapName = '('+self.tapName[tapIdx]+')'
                    else:
                        tapName = ''
                else:
                    tapName = ''
                tag = str(self.tapNo[tapIdx]) + tapName
                ax.annotate(tag, xy=(0,0), xycoords='axes fraction',xytext=(0.05, 0.05), textcoords='axes fraction',
                            fontsize=12, ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
                ax.set_ylim(fldRange)
                if xticks is not None:
                    ax.xaxis.set_ticks(xticks)
                ax.tick_params(axis=u'both', which=u'both',direction='in')
                
                ax.grid(which='both')
                
                if i//nCols == nRows-1:
                    ax.set_xlabel(r'AoA')
                if i%nCols == 0:
                    ax.set_ylabel(r'$C_p$')
                # if i == 0:
                #     ax.set_xlabel(r'AoA')
                #     ax.set_ylabel(r'$C_p$')
                #     ax.xaxis.tick_top()
                #     ax.xaxis.set_label_position('top')
                # else:
                #     ax.xaxis.set_ticklabels([])
                #     ax.yaxis.set_ticklabels([])
                tapPltCount += 1
                if tapPltCount < len(tapIdxs):
                    tapIdx = tapIdxs[tapPltCount]

            handles, labels = ax.get_legend_handles_labels()
            if newFig:
                fig.legend(handles, fields, loc='upper center',ncol=len(fields), bbox_to_anchor=legend_bbox_to_anchor, bbox_transform=fig.transFigure)
                plt.annotate(f"Page {p+1} of {nPages}", xy=pageNo_xy, xycoords='figure fraction', ha='right', va='bottom')
                plt.annotate(self.name, xy=pageNo_xy, xycoords='figure fraction', ha='left', va='bottom')
                plt.show()
        if newFig:
            return figs, all_axes
        else:
            return 

    def plotTapCpStatContour(self, fieldName, dxnIdx=None, envelopeType:Literal['high','low','both']='both', figSize=[15,10], ax=None, 
                             fldRange=None, nLvl=100, cmap='RdBu', extend='both', title=None, colBarOrientation='horizontal'):
        self.checkStatField(fieldName)
        if dxnIdx is None:
            if envelopeType == 'both':
                data = self.CpStatEnvlp[fieldName]
            elif envelopeType == 'high':
                data = self.CpStatEnvlp_high[fieldName]
            elif envelopeType == 'low':
                data = self.CpStatEnvlp_low[fieldName]
            else:
                msg = f"Unknown envelope type {envelopeType}"
                raise Exception(msg)
        else:
            data = self.CpStats[fieldName][dxnIdx,:]
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            ax = fig.add_subplot()
        cObj = self.plotTapField(ax=ax, field=data, fldRange=fldRange, nLvl=nLvl, cmap=cmap, extend=extend)
        if newFig:
            self.plotEdges(ax=ax)

            # from matplotlib.colorbar import make_axes
            # cax, kw = make_axes(fig.gca())
            # title = fieldName if title is None else title
            # cbar = fig.colorbar(c[0],title=title, cax=cax, **kw)
            # fldRange = [min(self.CpStats[fieldName][dxnIdx,:]), max(self.CpStats[fieldName][dxnIdx,:])] if fldRange is None else fldRange
            # cbar.set_clim(fldRange[0],fldRange[1])
            ax.axis('equal')
            ax.axis('off')

            title = fieldName if title is None else title
            cbar = fig.colorbar(cObj[0], ax=ax, orientation=colBarOrientation)
            cbar.set_label(title, fontsize=14)

        return cObj

    def plotPanelCpStatContour(self, fieldName, dxnIdx=0, aIdx=0, showValueText=False, strFmt="{:.3g}", figSize=[15,10], ax=None, title=None, fldRange=None, nLvl=100, cmap='RdBu'):
        self.checkStatField(fieldName)
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            ax = fig.add_subplot()
        self.plotPanelField(self.CpStatsAreaAvg, fieldName, dIdx=dxnIdx, aIdx=aIdx, showValueText=showValueText, strFmt=strFmt, fldRange=fldRange, ax=ax, nLvl=nLvl, cmap=cmap)
        if newFig:
            self.plotEdges(ax=ax)
            ax.axis('equal')
            ax.axis('off')
        return

    def plotAreaAveragedStat(self, fig=None, axs=None, figSize=[15,10], 
                            plotExtremesPerNominalArea=True, nCols=3, areaFactor=1.0, invertYAxis=True,
                            label_min='Min', label_max='Max',
                            overlayThis_min=None, overlayLabel_min='', kwargs_overlay_min={'color':'k', 'linewidth':2, 'linestyle':'-'},
                            overlayThis_max=None, overlayLabel_max='', kwargs_overlay_max={'color':'k', 'linewidth':2, 'linestyle':'-'},
                            plotZoneGeom=True, insetBounds:Union[list,dict]=[0.6, 0.0, 0.4, 0.4], zoneShadeColor='darkgrey', kwargs_zonePlots={},
                            xLimits=None, yLimits=None, xLabel=None, yLabel=None,
                            kwargs_min={}, kwargs_max={}, kwargs_legend={},
                            kwargs_ax={'gridMajor':True, 'gridMinor':True}):
        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figSize)

            NumZones = self.NumZones
            nCols = min(nCols, NumZones)
            nRows = int(np.ceil(NumZones/nCols))
            axs = fig.subplots(nRows, nCols, sharex=True, sharey=True)

        zoneDict = self.zoneDict
        pnlA_all = self.panelAreas
        pnlA_nom = self.NominalPanelArea
        if plotExtremesPerNominalArea:
            areaAvgStats_min = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='min',extremesPerNominalArea='min')
            areaAvgStats_max = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='max',extremesPerNominalArea='max')
        else:
            areaAvgStats_min = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='min',extremesPerNominalArea='none')
            areaAvgStats_max = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='max',extremesPerNominalArea='none')

        if plotZoneGeom:
            zoneCol = {zoneDict[z][0]+zoneDict[z][1]: 'w' for z in zoneDict}

        for I, z in enumerate(zoneDict):
            zoneName = zoneDict[z][0] + ', ' + zoneDict[z][1]
            area = pnlA_nom*areaFactor if plotExtremesPerNominalArea else np.array(pnlA_all[z][2])*areaFactor
            maxVal = np.squeeze(areaAvgStats_max[z][2]['peakMax'])
            minVal = np.squeeze(areaAvgStats_min[z][2]['peakMin'])
            
            i, j = np.unravel_index(z, axs.shape)
            ax = axs[i,j]
            ax.semilogx(area, minVal, 'vr', label=label_min, **kwargs_min)
            ax.semilogx(area, maxVal, '^b', label=label_max, **kwargs_max)
            if overlayThis_min is not None:
                ax.semilogx(overlayThis_min[z][2]['area'], overlayThis_min[z][2]['value'], label=overlayLabel_min, **kwargs_overlay_min)
            if overlayThis_max is not None:
                ax.semilogx(overlayThis_max[z][2]['area'], overlayThis_max[z][2]['value'], label=overlayLabel_max, **kwargs_overlay_max)

            if invertYAxis:
                ax.invert_yaxis()
            if plotZoneGeom:
                zoneCol_copy = zoneCol.copy()
                zoneCol_copy[zoneDict[z][0]+zoneDict[z][1]] = zoneShadeColor
                bounds = insetBounds # if insetBounds is isinstance(insetBounds, list) else insetBounds[z]
                ax_inset = ax.inset_axes(bounds=bounds)
                self.plotZones(ax=ax_inset, zoneCol=zoneCol_copy, drawEdge=True, **kwargs_zonePlots)
                # ax_inset.axis('off')
                ax_inset.axis('equal')
                ax_inset.patch.set_facecolor('w')
                ax_inset.patch.set_alpha(0.7)
                ax_inset.tick_params(axis='both', which='major', length=0)
                ax_inset.set_xticklabels('')
                ax_inset.set_yticklabels('')
                for spine in ax_inset.spines.values():
                    spine.set_visible(False)

            if newFig:
                ax.set_title(zoneName)
                if i == axs.shape[0]-1:
                    xLabel = r'Area [$m^2$]' if xLabel is None else xLabel
                    ax.set_xlabel(xLabel)
                if j == 0:
                    yLabel = r'Peak Cp' if yLabel is None else yLabel
                    ax.set_ylabel(yLabel)
                if i == 0 and j == 0:
                    ax.legend(**kwargs_legend)
                if xLimits is not None:
                    ax.set_xlim(xLimits)
                if yLimits is not None:
                    ax.set_ylim(yLimits)

                formatAxis(ax, **kwargs_ax)

        if newFig:
            plt.show()
        return fig, axs
    
class BldgCps():
    def __init__(self) -> None:
        self._members = []
        self.parent = bldgCp()
        pass

    def fold(self):
        pass