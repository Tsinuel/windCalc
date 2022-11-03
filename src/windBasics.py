# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinu
"""

import numpy as np
import pandas as pd
import warnings
import windPlotters as wplt
from scipy import signal

"""
===============================================================================
=============================== VARIABLES =====================================
===============================================================================
"""

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


"""
===============================================================================
=============================== FUNCTIONS =====================================
===============================================================================
"""

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

def psd(x,ns):
    
    x = x - np.mean(x)
    n, Sxx = signal.welch(x,fs=ns)
    
    return n,Sxx

def vonKarmanSpectra(n_=None,U_=None,Iu_=None,Iv_=None,Iw_=None,xLu_=None,xLv_=None,xLw_=None):
    
    Suu = lambda n,U,Iu,xLu: np.divide((4*xLu*(Iu**2)*U), np.power((1 + 70.8*np.power(n*xLu/U,2)),5/6))
    Svv = lambda n,U,Iv,xLv: np.divide( np.multiply( (4*xLv*(Iv**2)*U), (1 + 755.2*np.power(n*xLv/U,2) ) ), 
                                       np.power((1 + 283.1*np.power(n*xLv/U,2)),11/6))
    Sww = lambda n,U,Iw,xLw: np.divide( np.multiply( (4*xLw*(Iw**2)*U), (1 + 755.2*np.power(n*xLw/U,2) ) ), 
                                       np.power((1 + 283.1*np.power(n*xLw/U,2)),11/6))
    
    if n_ is not None and U_ is not None and Iu_ is not None and xLu_ is not None:
        Suu = Suu(n_,U_,Iu_,xLu_)
    if n_ is not None and U_ is not None and Iv_ is not None and xLv_ is not None:
        Svv = Svv(n_,U_,Iv_,xLv_)
    if n_ is not None and U_ is not None and Iw_ is not None and xLw_ is not None:
        Sww = Suu(n_,U_,Iw_,xLw_)
    
    return Suu, Svv, Sww



"""
===============================================================================
================================ CLASSES ======================================
===============================================================================
"""

class spectra:
    Suu_vonK = Svv_vonK = Sww_vonK = None
    UofT = VofT = WofT = None

    def __calculateSpectra(self,n_smpl,UofT,VofT,WofT):
        if UofT is not None:
            self.n, self.Suu = psd(UofT, n_smpl)
            self.U = np.mean(UofT)
            self.Iu = np.std(UofT)
            self.xLu = integLengthScale(UofT, 1/n_smpl)
        if VofT is not None:
            n, self.Svv = psd(VofT, n_smpl)
            self.Iv = np.std(VofT)
            self.xLv = integLengthScale(VofT, 1/n_smpl,meanU=self.U)
            if self.n is None:
                self.n = n
            elif not len(n) == len(self.n):
                raise Exception("UofT and VofT have different no. of time steps. The number of time steps in all UofT, VofT, and WofT must match for spectra calculation.")
                
        if WofT is not None:
            n, self.Sww = psd(WofT, n_smpl)
            self.Iw = np.std(WofT)
            self.xLw = integLengthScale(WofT, 1/n_smpl,meanU=self.U)
            if self.n is None:
                self.n = n
            elif not len(n) == len(self.n):
                raise Exception("WofT has different no. of time steps from UofT and VofT. All three must have the same number of time steps for spectra calculation.")
    
    def __calcVonK(self):
        Suu, Svv, Sww = vonKarmanSpectra()
        self.Suu_vonK = lambda n: Suu(n,self.U,self.Iu,self.xLu)
        self.Svv_vonK = lambda n: Svv(n,self.U,self.Iv,self.xLv)
        self.Sww_vonK = lambda n: Sww(n,self.U,self.Iw,self.xLw)
    
    def __init__(self, UofT=None, VofT=None, WofT=None, n_smpl=None, keepTH=False,
                 n=None, Suu=None, Svv=None, Sww=None,
                 Z=None, U=None, Iu=None, Iv=None, Iw=None):
        
        self.n = n
        self.Suu = Suu
        self.Svv = Svv
        self.Sww = Sww
        
        self.Z = Z
        self.U = U
        self.Iu = Iu
        self.Iv = Iv
        self.Iw = Iw
        
        if keepTH:
            self.UofT = UofT
            self.VofT = VofT
            self.WofT = WofT
        
        self.__calculateSpectra(n_smpl,UofT,VofT,WofT)
        self.__calcVonK()

    def plotSpectra(self, figFile=None, xLimits=None, figSize=[14,6]):
        n = (self.n,)
        val = (np.transpose(np.stack((self.Suu, self.Svv, self.Sww))),)
        ylabels = ("Suu","Svv","Sww")
        xlabel = 'n [Hz]'

        wplt.plotSpectra(
                        n, # ([n1,], [n2,], ... [nN,])
                        val, # ([n1,M], [n2,M], ... [nN,M])
                        dataLabels=(self.name,), # ("str1", "str2", ... "strN")
                        pltFile=figFile, # "/path/to/plot/file.pdf"
                        xLabel=xlabel,
                        yLabels=ylabels, # ("str1", "str2", ... "str_m")
                        xLimits=xLimits, # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                        # yLimits=[0,0.5], # [zMin, zMax]
                        figSize=figSize,
                        nCols=3
                        )

class profile:
    
    isNormalized = False
    interpolateToZref = False
    N_pts = 0
    Zref = None
    Uref = None
    IuRef = None
    IvRef = None
    IwRef = None
    iRef = None
    dt = None
    units = unitsNone
    
    def __updateUref(self):
        if self.N_pts == 0:
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

    def __computeFromTH(self):
        if self.UofT is not None:
            if self.Z is None:
                raise Exception("Z values not found while UofT is given.")
            self.U = np.mean(self.UofT,axis=1)
            self.Iu = np.std(self.UofT,axis=1)/self.U
            self.xLu = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLu[i] = integLengthScale(self.UofT[i,:], self.dt)
            
        if self.VofT is not None:
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate V stats.")
            self.V = np.mean(self.VofT,axis=1)
            self.Iv = np.std(self.VofT,axis=1)/self.U
            self.xLv = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLv[i] = integLengthScale(self.VofT[i,:], self.dt, meanU=self.U[i])

        if self.WofT is not None:
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate W stats.")
            self.W = np.mean(self.WofT,axis=1)
            self.Iw = np.std(self.WofT,axis=1)/self.U
            self.xLw = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLw[i] = integLengthScale(self.WofT[i,:], self.dt, meanU=self.U[i])
    
        self.__updateUref()
    
    def __computeSpectra(self):
        if self.iRef is None:
            self.Spect_Zref = None
            return
        uOfT = vOfT = wOfT = None
        if self.UofT is not None:
            uOfT = self.UofT[self.iRef,:]
        if self.VofT is not None:
            vOfT = self.UofT[self.iRef,:]
        if self.WofT is not None:
            wOfT = self.UofT[self.iRef,:]
        
        self.Spect_Zref = spectra(UofT=uOfT, VofT=vOfT, WofT=wOfT, n_smpl=self.n_smpl, Z=self.Zref )

    def __init__(self,
                name="profile", 
                profType=None, # {"continuous","discrete","scatter"}
                Z=None, Zref=None, dt=None, t=None,
                U=None, V=None, W=None, 
                UofT=None, VofT=None, WofT=None,
                Iu=None, Iv=None, Iw=None, 
                xLu=None, xLv=None, xLw=None,
                Spect_Zref=None, 
                fileName=None,
                interpolateToZref=False, units=unitsNone):
        self.name = name
        self.profType = profType
        self.Z = Z  # [N_pts]
        self.U = U  # [N_pts]
        self.V = V  # [N_pts]
        self.W = W  # [N_pts]
        
        self.Zref = Zref
        
        self.dt = dt
        self.n_smpl = None if dt is None else 1/dt
        self.t = t
        self.UofT = UofT  # [N_pts x nTime]
        self.VofT = VofT  # [N_pts x nTime]
        self.WofT = WofT  # [N_pts x nTime]
        
        self.Iu = Iu  # [N_pts]
        self.Iv = Iv  # [N_pts]
        self.Iw = Iw  # [N_pts]
        
        self.xLu = xLu  # [N_pts]
        self.xLv = xLv  # [N_pts]
        self.xLw = xLw  # [N_pts]
        
        self.Spect_Zref = Spect_Zref
        
        self.origFileName = fileName
        self.interpolateToZref = interpolateToZref
        self.units = units
        
        # Compute params
        self.__computeFromTH()
        if self.Z is not None:
            self.N_pts = len(self.Z)
        self.__computeSpectra()
        
        if fileName is not None:
            self.readFromFile(fileName)
        self.__updateUref()

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
    
    def readFromFile(self,fileName,getZrefFromU_Uh=False):
        data = pd.read_csv(fileName)
        self.Z = data.Z
        self.U = data.U
        self.Iu = data.Iu
        self.Iv = data.Iv
        self.Iw = data.Iw
        if 'U_Uh' in data.columns and self.Zref is None:
            idx = (np.abs(data.U_Uh - 1)).argmin()
            self.Zref = self.Z[idx]

        self.N_pts = len(self.Z)
        self.__updateUref()
    
    def normalize(self):
        pass
    
    def plotProfiles(self,figFile=None,xLimits=None,figSize=[14,6]):
        Z = (self.Z,)
        val = (np.transpose(np.stack((self.U, self.Iu, self.Iv, self.Iw))),)
        xlabels = ("U","Iu","Iv","Iw")
        zlabel = 'Z'

        wplt.plotProfiles(
                        Z, # ([n1,], [n2,], ... [nN,])
                        val, # ([n1,M], [n2,M], ... [nN,M])
                        dataLabels=(self.name,), # ("str1", "str2", ... "strN")
                        pltFile=figFile, # "/path/to/plot/file.pdf"
                        xLabels=xlabels, # ("str1", "str2", ... "str_m")
                        yLabel=zlabel,
                        xLimits=xLimits, # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                        # yLimits=[0,0.5], # [zMin, zMax]
                        figSize=figSize,
                        nCols=4
                        )
    
    def plotTimeHistory(self,figFile=None):
        if all((self.UofT is None, self.VofT is None, self.WofT is None)):
            raise Exception("At least one of UofT, VofT, or WofT has to be provided to plot time history.")
        if all((self.dt is None, self.t is None)):
            raise Exception("Either dt or t has to be provided to plot time history.")
        
        if self.UofT is not None:
            N_T = np.shape(self.UofT)[1]
        elif self.VofT is not None:
            N_T = np.shape(self.VofT)[1]
        else:
            N_T = np.shape(self.WofT)[1]
        if self.t is None:
            self.t = range(0,N_T,self.dt)
        if self.dt is None:
            self.dt = np.mean(np.diff(self.t))
        




class Profiles:
    def __init__(self, profs):
        self.N = len(profs)
        self.profiles = profs
    
    def plotProfiles(self,figFile=None,xLimits='auto',zLimits='auto',figSize=[14,6],normalize=True):
        Z = ()
        val = ()
        names = ()
        # maxU = 0
        if normalize:
            for i in range(self.N):
                Z += (self.profiles[i].Z/self.profiles[i].Zref,)
                val += (np.transpose(np.stack((self.profiles[i].U/self.profiles[i].Uref, self.profiles[i].Iu, self.profiles[i].Iv, self.profiles[i].Iw))),)
                names += (self.profiles[i].name,)
                # maxU = max(maxU, max(val[i][:,0]))
            xlabels = ("U/Uref","Iu","Iv","Iw")
            zlabel = 'Z/Zref'
        else:
            for i in range(self.N):
                Z += (self.profiles[i].Z,)
                val += (np.transpose(np.stack((self.profiles[i].U, self.profiles[i].Iu, self.profiles[i].Iv, self.profiles[i].Iw))),)
                names += (self.profiles[i].name,)
                # maxU = max(maxU, max(val[i][:,0]))
            xlabels = ("U","Iu","Iv","Iw")
            zlabel = 'Z'

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
    
    def plotTimeHistory(self):
        pass
    
    def plotSpectra(self, figFile=None, xLimits=None, figSize=[16,4], normalize=True, 
                    normSby='Uref' # {'Uref','sig_ui'}
                    ):
        n = ()
        val = ()
        names = ()
        if normalize:
            if normSby == 'Uref':
                for i in range(self.N):
                    n += (self.profiles[i].Spect_Zref.n * self.profiles[i].Spect_Zref.Z / self.profiles[i].Spect_Zref.U ,)
                    val += (np.transpose(np.stack(( np.multiply(self.profiles[i].Spect_Zref.n, self.profiles[i].Spect_Zref.Suu) / (self.profiles[i].Spect_Zref.U)**2 , 
                                                    np.multiply(self.profiles[i].Spect_Zref.n, self.profiles[i].Spect_Zref.Svv) / (self.profiles[i].Spect_Zref.U)**2 , 
                                                    np.multiply(self.profiles[i].Spect_Zref.n, self.profiles[i].Spect_Zref.Sww) / (self.profiles[i].Spect_Zref.U)**2
                                                    ))),)
                    names += (self.profiles[i].name,)
                ylabels = ("nSuu/Uref^2","nSvv/Uref^2","nSww/Uref^2")
                xlabel = 'n.Zref/Uref'
            elif normSby == 'sig_ui':
                raise NotImplemented("This normalization type has not been implemented")
        else:
            for i in range(self.N):
                n += (self.profiles[i].Spect_Zref.n,)
                val += (np.transpose(np.stack((self.profiles[i].Spect_Zref.Suu, self.profiles[i].Spect_Zref.Svv, self.profiles[i].Spect_Zref.Sww))),)
                names += (self.profiles[i].name,)
            ylabels = ("Suu","Svv","Sww")
            xlabel = 'n'

        wplt.plotSpectra(
                        n, # ([n1,], [n2,], ... [nN,])
                        val, # ([n1,M], [n2,M], ... [nN,M])
                        dataLabels=names, # ("str1", "str2", ... "strN")
                        pltFile=figFile, # "/path/to/plot/file.pdf"
                        yLabels=ylabels, # ("str1", "str2", ... "str_m")
                        xLabel=xlabel,
                        xLimits=xLimits, # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                        # yLimits=[0,0.5], # [zMin, zMax]
                        figSize=figSize,
                        nCols=3
                        )


