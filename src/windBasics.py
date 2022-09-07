# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinu
"""

import numpy as np
import warnings
from scipy import signal

unitsCommonSI = {
        'L':'m',
        'T':'s',
        'V':'mps',
        'P':'Pa',
        'M':'kg'
        }
unitsNone = {
        'L':None,
        'T':None,
        'V':None,
        'P':None,
        'M':None
        }

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

def integLengthScale(vel,dt,meanU=None,rho_tol=0.0001,showPlots=False):
    """
    Computes the longitudianl integral length scale of a velocity time history 
    where Taylor's frozen turbulence hypothesis applies (Simiu and Yeo (2019)). 
    Returns xLu, xLv, and xLw components.

    Parameters
    ----------
    vel : 1-D or 2-D float array ; shape [nTime,nComponents]
        Time-history of one or more velocity components. If there are more 
        than one components in vel, the shape must be [nTime,nComponents] 
        and u, v, and w in indices 0, 1, and 2, respectively. If it is a 
        single component, either the mean longitudinal velocity U has to be 
        supplied or else the mean value of the signal will be used. This may 
        cause a problem for v and w or mean-remove u components. By default, 
        vel is assumed to contain all three non-mean-removed components.
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
    
    nComponents = np.shape(vel)[1]
    
    if nComponents == 1:
        UofT = vel
    elif nComponents == 2:
        UofT = vel[:,0]
        VofT = vel[:,1]
    elif nComponents == 3:
        UofT = vel[:,0]
        VofT = vel[:,1]
        WofT = vel[:,2]
    else:
        raise Exception("The maximum number of components is three. Check if the matrix needs to be transposed so that the first dimension is the time and the second components.")
    
    if meanU is not None:
        meanU = np.mean(UofT)
    
    xLu = meanU * integTimeScale(UofT-meanU, dt, rho_tol=rho_tol, showPlots=showPlots)[0]
    xLv = meanU * integTimeScale(VofT-np.mean(VofT), dt, rho_tol=rho_tol, showPlots=showPlots)[0]
    xLw = meanU * integTimeScale(WofT-np.mean(WofT), dt, rho_tol=rho_tol, showPlots=showPlots)[0]
    
    L = np.asarray([xLu, xLv, xLw],dtype=float)
    
    return L

def psd(x,ns):
    
    x = x - np.mean(x)
    n, Sxx = signal.welch(x,fs=ns)
    
    return n,Sxx


class spectra:
    def __init__(self, spectType='fromTH', UofT=[], VofT=[], WofT=[]):
        
        if spectType == 'fromTH':
            pass
        elif spectType == 'vonK':
            pass
        elif spectType == 'ESDU':
            pass
        elif spectType == 'fromFile':
            pass
        else:
            pass
            
        pass

class profile:
    
    isNormalized = False
    interpolateToZref = False
    nPts = 0
    Zref = None
    Uref = None
    IuRef = None
    IvRef = None
    IwRef = None
    iRef = None
    dt = None
    units = unitsNone
    
    def __updateUref(self):
        if self.nPts == 0:
            self.iRef = None
            self.Uref = self.IuRef = self.IvRef = self.IwRef = None
            return
        if self.interpolateToZref:
            self.iRef = None
            self.Uref = np.interp(self.Zref, self.Z, self.U)
            self.IuRef = np.interp(self.Zref, self.Z, self.Iu)
            self.IvRef = np.interp(self.Zref, self.Z, self.Iv)
            self.IwRef = np.interp(self.Zref, self.Z, self.Iw)
        else: # nearest value
            self.iRef = (np.abs(self.Z - self.Zref)).argmin()
            self.Uref = self.U[self.iRef]
            self.IuRef = self.Iu[self.iRef]
            self.IvRef = self.Iv[self.iRef]
            self.IwRef = self.Iw[self.iRef]
    
    def __init__(self,name="profile", Z=None, U=None, V=None, W=None, Zref=None, 
                 dt=None, UofT=None, VofT=None, WofT=None,
                 Iu=None, Iv=None, Iw=None, 
                 xLu=None, xLv=None, xLw=None,
                 Spect_H=None, interpolateToZref=False, units=unitsNone):
        self.name = name
        self.Z = Z  # [nPts]
        self.U = U  # [nPts]
        self.V = V  # [nPts]
        self.W = W  # [nPts]
        
        self.Zref = Zref
        
        self.dt = dt
        self.UofT = UofT  # [nPts x nTime]
        self.VofT = VofT  # [nPts x nTime]
        self.WofT = WofT  # [nPts x nTime]
        
        self.Iu = Iu  # [nPts]
        self.Iv = Iv  # [nPts]
        self.Iw = Iw  # [nPts]
        
        self.xLu = xLu  # [nPts]
        self.xLv = xLv  # [nPts]
        self.xLw = xLw  # [nPts]
        
        self.Spect_H = Spect_H
        
        self.interpolateToZref = interpolateToZref
        self.units = units
        
        # Compute params
        self.__computeFromTH()
        if self.Z is not None:
            self.nPts = len(self.Z)
            self.__updateUref()
        self.__computeSpectra()

    def __computeFromTH(self):
        if self.UofT is not None:
            if self.Z is None:
                raise Exception("Z values not found while UofT is given.")
            self.U = np.mean(self.UofT,axis=1)
            self.Iu = np.std(self.UofT,axis=1)/self.U
            self.xLu = np.zeros(self.nPts)
            for i in range(self.nPts):
                self.xLu[i] = integLengthScale(self.UofT[i,:], self.dt)
            
        if self.VofT is not None:
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate V stats.")
            self.V = np.mean(self.VofT,axis=1)
            self.Iv = np.std(self.VofT,axis=1)/self.U
            self.xLv = np.zeros(self.nPts)
            for i in range(self.nPts):
                self.xLv[i] = integLengthScale(self.VofT[i,:], self.dt, meanU=self.U[i])

        if self.WofT is not None:
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate W stats.")
            self.W = np.mean(self.WofT,axis=1)
            self.Iw = np.std(self.WofT,axis=1)/self.U
            self.xLw = np.zeros(self.nPts)
            for i in range(self.nPts):
                self.xLw[i] = integLengthScale(self.WofT[i,:], self.dt, meanU=self.U[i])

    def __computeSpectra(self):
        pass

    def writeToFile(self,outDir,nameSuffix=''):
        pass
    
    def readFromFile(self):
        pass
    
    def normalize(self):
        pass
    
    def plotProfiles(self):
        pass
    
    def plotTimeHistory(self):
        pass
    
    def plotSpectra(self):
        pass
    
    
class Profiles:
    def __init__(self, *args, **kwargs):
        pass

