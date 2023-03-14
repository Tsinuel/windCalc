# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:25:30 2022

@author: Tsinu
"""

import numpy as np
import wind
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def processVelProfile(Z, vel, dt, zRef, interpMethod='nearest', getSpectAt=[], showPlots=(False,'')):
    """
    

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    vel : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    zRef : TYPE
        DESCRIPTION.
    interpMethod : TYPE, optional
        DESCRIPTION. The default is 'nearest'.
    getSpectAt : TYPE, optional
        DESCRIPTION. The default is [].
    showPlots : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    ref_U_TI_L : TYPE
        DESCRIPTION.
    Z : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.
    TI : TYPE
        DESCRIPTION.
    L : TYPE
        DESCRIPTION.
    freq : TYPE
        DESCRIPTION.
    Spect_H : TYPE
        DESCRIPTION.
    Spect_Others : TYPE
        DESCRIPTION.

    """
    # sort Z and vel if it is not sorted
    if not np.all(Z[:-1] <= Z[1:]):
        i = np.argsort(Z)
        Z = Z[i]
        vel = vel[:,i,:]
        warnings.warn("The input profile was not sorted. It has been sorted now and the new Z values are used (and returned) and make sure to use those, not the old ones.")
    
    nPts = len(Z)
    velMean = np.mean(vel,axis=0)
    U = velMean[:,0]
    TI = np.std(vel,axis=0) / U[:,None]
    
    L = np.zeros(np.shape(TI))
    for i in range(nPts):
        L[i,:] = wind.integLengthScale(vel[:,i,:],dt)
    
    iRef = (np.abs(Z - zRef)).argmin()
    if interpMethod == 'nearest':
        ref_U_TI_L = np.array([U[iRef],
                            TI[iRef,0],
                            TI[iRef,1],
                            TI[iRef,2],
                            L[iRef,0],
                            L[iRef,1],
                            L[iRef,2]])
    elif interpMethod == 'interp':
        ref_U_TI_L = np.array([np.interp(zRef,Z,U),
                            np.interp(zRef,Z,TI[:,0]),
                            np.interp(zRef,Z,TI[:,1]),
                            np.interp(zRef,Z,TI[:,2]),
                            np.interp(zRef,Z,L[:,0]),
                            np.interp(zRef,Z,L[:,1]),
                            np.interp(zRef,Z,L[:,2])])
        warnings.warn("Interpolation is only used for U, Iu, Iv, Iw, Lu, Lv, & Lw values, NOT the spectra. The spectra are always calculated at the nearest points.")
    
    freq,Suu = wind.psd(vel[:,iRef,0], 1/dt)
    Svv = wind.psd(vel[:,iRef,1], 1/dt)[1]
    Sww = wind.psd(vel[:,iRef,2], 1/dt)[1]
    Spect_H = np.concatenate((np.reshape(Suu,[-1,1]), np.reshape(Svv,[-1,1]), np.reshape(Sww,[-1,1])), axis=1)
    
    Spect_Others = []
    if len(getSpectAt) > 0:
        raise NotImplementedError()
        iSpectOthrs = np.zeros(np.shape(getSpectAt),dtype=int)
        for i,z in enumerate(getSpectAt):
            iSpectOthrs[i] = (np.abs(Z - z)).argmin()
    

    if showPlots[0]:
        pdf = PdfPages(showPlots[1])       
        fig = plt.figure() 
        
        # U
        plt.plot(U,Z,'sr',label='mean U',ms=3)
        plt.xlabel('U')
        plt.ylabel('Z')
        plt.legend()        
        pdf.savefig(fig)
        plt.clf()
        
        plt.plot(TI[:,0],Z,'sr',label='Iu',ms=3)
        plt.plot(TI[:,1],Z,'dg',label='Iv',ms=3)
        plt.plot(TI[:,2],Z,'ob',label='Iw',ms=3)
        plt.xlabel('TI')
        plt.ylabel('Z')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()
        
        plt.plot(L[:,0],Z,'sr',label='xLu',ms=3)
        plt.plot(L[:,1],Z,'dg',label='xLv',ms=3)
        plt.plot(L[:,2],Z,'ob',label='xLw',ms=3)
        plt.xlabel('xLi')
        plt.ylabel('Z')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()
        
        plt.loglog(freq,Spect_H[:,0],'-r',label='Suu',ms=3)
        plt.xlabel('freq [Hz]')
        plt.ylabel('Suu')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()

        plt.loglog(freq,Spect_H[:,1],'-g',label='Svv',ms=3)
        plt.xlabel('freq [Hz]')
        plt.ylabel('Svv')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()

        plt.loglog(freq,Spect_H[:,2],'-b',label='Sww',ms=3)
        plt.xlabel('freq [Hz]')
        plt.ylabel('Sww')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()
        
        pdf.close()

    return ref_U_TI_L, Z, U, TI, L, freq, Spect_H, Spect_Others

def readVelProfile(velFile,zRef):
    
    data = pd.read_csv(velFile,delimiter=',')
    
    Z = np.asarray(data.Z,dtype=float)
    U = np.asarray(data.U,dtype=float)
    TI = np.asarray(data.values[:,2:5],dtype=float)
    L = np.asarray(data.values[:,5:8],dtype=float)
    
    iRef = (np.abs(Z - zRef)).argmin()
    
    ref_U_TI_L = [U[iRef], TI[iRef,0], TI[iRef,1], TI[iRef,2], L[iRef,0], L[iRef,1], L[iRef,2]]
    
    return ref_U_TI_L, Z, U, TI, L