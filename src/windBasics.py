# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinu
"""

import numpy as np
from scipy import signal

def integTimeScale(x,dt,rho_tol=0.0001,showPlots=False):
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

    x = x - np.mean(x)
    rhoAll = signal.correlate(x,x)
    iMiddle = int(np.floor(len(rhoAll)/2))
    rhoAll = rhoAll/rhoAll[iMiddle]
    rho = rhoAll[iMiddle:]
    
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

def integLengthScale(vel,dt,meanU=[],rho_tol=0.0001,showPlots=False):
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
        The default is [].
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
    UisGiven = len(meanU) > 0
    
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
    
    if not UisGiven:
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












