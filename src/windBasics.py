# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
import warnings
from scipy import signal

import windPlotters as wplt

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

def psd(x,fs,nAvg=8):
    x = x - np.mean(x)
    N = len(x)
    nblock = int(np.floor(N/nAvg))
    overlap = int(np.ceil(nblock/2))
    win = signal.hanning(nblock, True)
    N = nblock*nAvg
    f, Pxxf = signal.welch(x[0:N], fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True)
    return f, Pxxf

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

def vonKarmanSpectra___depricated(n_=None,U_=None,Iu_=None,Iv_=None,Iw_=None,xLu_=None,xLv_=None,xLw_=None):
    
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
class ESDU74:
    def __init__(self):
        pass

    def U(self):
        pass

    def Iu(self):
        pass

    def Iv(self):
        pass

    def Iw(self):
        pass

    def xLu(self):
        pass


class ESDU85:
    __Omega = 72.9e-6            # Angular rate of rotation of the earth ESDU 82026 ssA1

    def __init__(self,
                    phi=30,
                    z0=None,
                    Z=None, 
                    Zref=10,
                    Uref=10,
                    ):
        self.phi = phi # latitude in degrees
        self.z0 = z0
        self.Z = Z=np.sort(np.append(np.logspace(np.log10(0.5),np.log10(300),99),10)) if Z is None else Z
        self.Zref = Zref
        self.Uref = Uref

    def fitToThisIuRef(self,IuRef):
        pass

    def f(self):
        # Coriolis parameter            ESDU 82026, section A1
        if self.phi is None:
            return None
        else:
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
        if Z is None or self.z0 is None or self.Uref is None or self.Zref is None:
            return None
        else:
            return np.multiply(self.uStar()/0.4, (np.log(Z/self.z0) + 34.5*self.f()*Z/self.uStar()))
    
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
        Ro = self.uStar()/(self.f()*self.z0)
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

    def Suu(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            nu = np.logspace(-3,2,100)
            n = np.multiply(nu, np.divide(self.U(Z), self.xLu(Z)))
        else:
            nu = np.multiply(n, np.divide(self.xLu(Z), self.U(Z)))
        F1 = 1 + 0.455*np.exp(-0.76*np.divide(nu,np.power(self.alpha(Z),-0.8)))   # ESDU 85020, eq. B4.3
        term1 = np.multiply(self.beta1(Z), np.divide(2.987*np.divide(nu,self.alpha(Z)),
                                            np.power(1 + np.power(np.divide(2*np.pi*nu,self.alpha(Z)) ,2) ,5/6) ) 
                                            )
        term2 = np.multiply(self.beta2(Z), np.multiply(np.divide(np.divide(1.294*nu, self.alpha(Z)),
                                                         np.power(1 + np.power(np.divide(np.pi*nu, self.alpha(Z)),2) ,5/6) )  ,F1) )
        rSuu = term1 + term2  # ESDU 85020, eq. B4.1
        varU = np.power(np.multiply(self.Iu(Z),self.U(Z)),2)
        return n, np.divide(np.multiply(rSuu,varU), n)
        
    def F2(self,ni,Z=None):
        Z = self.Zref if Z is None else Z
        return 1 + 2.88*np.exp(-0.218*np.divide(ni,np.power(self.alpha(Z),-0.9)))   # ESDU 85020, eq. B4.4

    def Sii(self,ni,Z=None):
        Z = self.Zref if Z is None else Z
        term1 = np.multiply(self.beta1(Z), np.divide(2.987* np.multiply(1 + (8/3)*np.power(np.divide(4*np.pi*ni,self.alpha(Z)),2), ni/self.alpha(Z)) ,
                                            np.power(1 + np.power(np.divide(4*np.pi*ni,self.alpha(Z)) ,2), 11/6) ) 
                                            )
        term2 = np.multiply(self.beta2(Z), np.multiply(np.divide(np.divide(1.294*ni, self.alpha(Z)),
                                                         np.power(1 + np.power(np.divide(2*np.pi*ni, self.alpha(Z)),2) ,5/6) )  ,self.F2(ni,Z)) )
        return term1 + term2  # ESDU 85020, eq. B4.1
        
    def Svv(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            nv = np.logspace(-3,2,100)
            n = np.multiply(nv, np.divide(self.U(Z), self.xLv(Z)))
        else:
            nv = np.multiply(n, np.divide(self.xLv(Z), self.U(Z)))
        varV = np.power(np.multiply(self.Iv(Z),self.U(Z)),2)
        return n, np.divide(np.multiply(self.Sii(nv,Z),varV), n)

    def Sww(self,n=None,Z=None):
        Z = self.Zref if Z is None else Z
        if n is None:
            nw = np.logspace(-3,2,100)
            n = np.multiply(nw, np.divide(self.U(Z), self.xLw(Z)))
        else:
            nw = np.multiply(n, np.divide(self.xLw(Z), self.U(Z)))
        varW = np.power(np.multiply(self.Iw(Z),self.U(Z)),2)
        return n, np.divide(np.multiply(self.Sii(nw,Z),varW), n)

    def rf(self,n,Z=None,normZ='Z'):
        Z = self.Zref if Z is None else Z
        if normZ == 'Z':
            normZ = Z
        elif normZ == 'xLi':
            normZ = self.xLu(Z=Z)
        elif not type(normZ) == int or float:
            raise Exception("Unknown normalization height type. Choose from {'Z', 'xLi'} or specify a number.")
        Uref = self.U(Z=Z)
        return n * normZ/Uref

    def rSuu(self,n=None,normU='U',Z=None,normZ='Z'):
        Z = self.Zref if Z is None else Z
        if n is None:
            n = np.multiply(np.logspace(-3,2,100), np.divide(self.U(Z), self.xLu(Z)))
        if Z is None:
            return None
        if normU == 'U':
            normU = self.U(Z)
        elif normU == 'sigUi':
            normU = self.Iu(Z) * self.U(Z)
        return self.rf(n,Z=Z,normZ=normZ), np.multiply(n,self.Suu(n=n,Z=Z)[1])/(normU**2)

    def rSvv(self,n=None,normU='U',Z=None,normZ='Z'):
        Z = self.Zref if Z is None else Z
        if n is None:
            n = np.multiply(np.logspace(-3,2,100), np.divide(self.U(Z), self.xLv(Z)))
        if Z is None:
            return None
        if normU == 'U':
            normU = self.U(Z)
        elif normU == 'sigUi':
            normU = self.Iv(Z) * self.U(Z)
        return self.rf(n,Z=Z,normZ=normZ), np.multiply(n,self.Svv(n=n,Z=Z)[1])/(normU**2)

    def rSww(self,n=None,normU='U',Z=None,normZ='Z'):
        Z = self.Zref if Z is None else Z
        if n is None:
            n = np.multiply(np.logspace(-3,2,100), np.divide(self.U(Z), self.xLw(Z)))
        if Z is None:
            return None
        if normU == 'U':
            normU = self.U(Z)
        elif normU == 'sigUi':
            normU = self.Iw(Z) * self.U(Z)
        return self.rf(n,Z=Z,normZ=normZ), np.multiply(n,self.Sww(n=n,Z=Z)[1])/(normU**2)


class spectra:
    UofT = VofT = WofT = None

    """---------------------------------- Internals -----------------------------------"""
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
        
    def __init__(self, name=None, UofT=None, VofT=None, WofT=None, samplingFreq=None, 
                 n=None, Suu=None, Svv=None, Sww=None, nSpectAvg=8,
                 Z=None, U=None, Iu=None, Iv=None, Iw=None):
        
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
        
        self.UofT = UofT
        self.VofT = VofT
        self.WofT = WofT

        self.reAnalyze()

    """--------------------------------- Normalizers ----------------------------------"""
    def rf(self,n='auto',normZ='Z'):
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

    def rSuu(self,normU='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iu * self.U
        if self.n is None or self.Suu is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Suu)/(normU**2)

    def rSvv(self,normU='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iv * self.U
        if self.n is None or self.Svv is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Svv)/(normU**2)

    def rSww(self,normU='U'):
        if normU == 'U':
            normU = self.U
        elif normU == 'sigUi':
            normU = self.Iw * self.U
        if self.n is None or self.Sww is None or normU is None:
            return None
        else:
            return np.multiply(self.n,self.Sww)/(normU**2)
    
    """-------------------------------- Data handlers ---------------------------------"""
    def reAnalyze(self):
        self.__calculateSpectra()

    def writeSpecsToFile(self):
        pass

    def writeDataToFile(self):
        pass

    """--------------------------------- Fittings -------------------------------------"""
    def Suu_vonK(self,n=None,normalized=False,normU='U'):
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

    def Svv_vonK(self,n=None,normalized=False,normU='U'):
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

    def Sww_vonK(self,n=None,normalized=False,normU='U'):
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
    def plotSpectra(self, 
                    figFile=None, 
                    xLimits=None, 
                    yLimits='auto', # ([SuuMin, SuuMax], [SvvMin, SvvMax], [SwwMin, SwwMax])
                    figSize=[15,5], 
                    normalize=True,
                    normZ='Z',
                    normU='U',
                    plotType='loglog',
                    overlayVonK=False,
                    avoidZeroFreq=True
                    ):
        if normalize:
            n = (self.rf(normZ=normZ),)
            Suu = (self.rSuu(normU=normU),)
            Svv = (self.rSvv(normU=normU),)
            Sww = (self.rSww(normU=normU),)
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
            n = (self.n,)
            Suu = (self.Suu,)
            Svv = (self.Svv,)
            Sww = (self.Sww,)
            ylabels = (r"$S_{uu}$",r"$S_{vv}$",r"$S_{ww}$")
            xlabel = r"$n [Hz]$"
            drawXlineAt_rf1 = False
        names = (self.name,)
        if overlayVonK:
            names += ('vonK-'+self.name,)
            n += (n[0],)
            Suu += (self.Suu_vonK(self.n,normalized=normalize,normU=normU),)
            Svv += (self.Svv_vonK(self.n,normalized=normalize,normU=normU),)
            Sww += (self.Sww_vonK(self.n,normalized=normalize,normU=normU),)

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
                        drawXlineAt_rf1=drawXlineAt_rf1,
                        avoidZeroFreq=avoidZeroFreq
                        )

class profile:
    
    isNormalized = False
    interpolateToH = False
    N_pts = 0
    H = None
    Uh = None
    IuH = None
    IvH = None
    IwH = None
    iH = None
    dt = None
    units = unitsNone

    """---------------------------------- Internals -----------------------------------"""
    def __verifyData(self):
        pass

    def __updateUh(self):
        if self.N_pts == 0:
            self.iH = None
            self.Uh = self.IuH = self.IvH = self.IwH = None
            return
        if self.interpolateToH:
            self.iH = None
            self.Uh = np.interp(self.H, self.Z, self.U)
            self.IuH = np.interp(self.H, self.Z, self.Iu)
            self.IvH = np.interp(self.H, self.Z, self.Iv)
            self.IwH = np.interp(self.H, self.Z, self.Iw)
        else: # nearest value
            self.iH = (np.abs(self.Z - self.H)).argmin()
            self.Uh = self.U[self.iH]
            self.IuH = self.Iu[self.iH]
            self.IvH = self.Iv[self.iH]
            self.IwH = self.Iw[self.iH]

    def __computeFromTH(self):
        atLeastOneTHfound = False
        if self.UofT is not None:
            N_T = np.shape(self.UofT)[1]
            if self.Z is None:
                raise Exception("Z values not found while UofT is given.")
            self.U = np.mean(self.UofT,axis=1)
            self.Iu = np.std(self.UofT,axis=1)/self.U
            self.xLu = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLu[i] = integLengthScale(self.UofT[i,:], self.dt)
            atLeastOneTHfound = True
            
        if self.VofT is not None:
            N_T = np.shape(self.VofT)[1]
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate V stats.")
            self.V = np.mean(self.VofT,axis=1)
            self.Iv = np.std(self.VofT,axis=1)/self.U
            self.xLv = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLv[i] = integLengthScale(self.VofT[i,:], self.dt, meanU=self.U[i])
            atLeastOneTHfound = True

        if self.WofT is not None:
            N_T = np.shape(self.WofT)[1]
            if self.U is None or self.Z is None:
                raise Exception("Either Z or U(z) does not exist to calculate W stats.")
            self.W = np.mean(self.WofT,axis=1)
            self.Iw = np.std(self.WofT,axis=1)/self.U
            self.xLw = np.zeros(self.N_pts)
            for i in range(self.N_pts):
                self.xLw[i] = integLengthScale(self.WofT[i,:], self.dt, meanU=self.U[i])
            atLeastOneTHfound = True
        
        if self.t is None and self.dt is not None and atLeastOneTHfound:
            self.t = np.linspace(0,(N_T-1)*self.dt,num=N_T)
        if self.dt is None and self.t is not None and atLeastOneTHfound:
            self.dt = np.mean(np.diff(self.t))

        self.__updateUh()
        self.__computeSpectra()
    
    def __computeSpectra(self):
        if self.iH is None:
            self.SpectH = None
            return
        uOfT = vOfT = wOfT = None
        if self.UofT is not None:
            uOfT = self.UofT[self.iH,:]
        if self.VofT is not None:
            vOfT = self.UofT[self.iH,:]
        if self.WofT is not None:
            wOfT = self.UofT[self.iH,:]
        
        self.SpectH = spectra(name=self.name, UofT=uOfT, VofT=vOfT, WofT=wOfT, samplingFreq=self.samplingFreq, Z=self.H, nSpectAvg=self.nSpectAvg)

    def __init__(self,
                name="profile", 
                profType=None, # {"continuous","discrete","scatter"}
                Z=None, H=None, dt=None, t=None,
                U=None, V=None, W=None, 
                UofT=None, VofT=None, WofT=None,
                Iu=None, Iv=None, Iw=None, 
                xLu=None, xLv=None, xLw=None,
                SpectH=None, nSpectAvg=8,
                fileName=None,
                interpolateToH=False, units=unitsNone):
        self.name = name
        self.profType = profType
        self.Z = Z  # [N_pts]
        self.U = U  # [N_pts]
        self.V = V  # [N_pts]
        self.W = W  # [N_pts]
        
        self.H = H
        
        self.dt = dt
        self.samplingFreq = None if dt is None else 1/dt
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
        
        self.SpectH = SpectH
        self.nSpectAvg=nSpectAvg
        
        self.origFileName = fileName
        self.interpolateToH = interpolateToH
        self.units = units
        
        self.reAnalyze()

    """---------------------------------- Normalizers ---------------------------------"""
    def ZbyH(self,H=None):
        if H is None:
            H = self.H
        if H is None or self.Z is None:
            return None
        else:
            return self.Z/H

    def UbyUh(self,Uh=None):
        if Uh is None:
            Uh = self.Uh
        if Uh is None or self.U is None:
            return None
        else:
            return self.U/Uh

    def xLuByH(self,H=None):
        if H is None:
            H = self.H
        if H is None or self.Z is None:
            return None
        else:
            return self.xLu/H

    def xLvByH(self,H=None):
        if H is None:
            H = self.H
        if H is None or self.Z is None:
            return None
        else:
            return self.xLv/H

    def xLwByH(self,H=None):
        if H is None:
            H = self.H
        if H is None or self.Z is None:
            return None
        else:
            return self.xLw/H

    def normalize(self):
        pass

    """-------------------------------- Data handlers ---------------------------------"""
    def reAnalyze(self):
        self.__verifyData()
        if self.Z is not None:
            self.N_pts = len(self.Z)
        self.__computeFromTH()
        
        if self.origFileName is not None:
            self.readFromFile(self.origFileName)
        self.__updateUh()

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
    
    def readFromFile(self,fileName,getHfromU_Uh=False):
        data = pd.read_csv(fileName)
        self.Z = data.Z
        self.U = data.U
        self.Iu = data.Iu
        self.Iv = data.Iv
        self.Iw = data.Iw
        if 'U_Uh' in data.columns and self.H is None:
            idx = (np.abs(data.U_Uh - 1)).argmin()
            self.H = self.Z[idx]

        self.N_pts = len(self.Z)
        self.__updateUh()

    """--------------------------------- Plotters -------------------------------------"""
    def plotProfiles(self,figFile=None,xLimits=None,figSize=[15,5]):
        Z = (self.Z,)
        val = (np.transpose(np.stack((self.U, self.Iu, self.Iv, self.Iw))),)
        xlabels = (r"U",r"I_u",r"I_v",r"I_w")
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
    
    def plotTimeHistory(self,
                    figFile=None,
                    normalizeTime=False,
                    normalizeVel=False,
                    dataLabels='auto', # automatically taken from the profile object
                    xLabel='auto',
                    yLabels=("U(t)","V(t)","W(t)"), 
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
            U = self.UofT[self.iH,:]/self.Uh if normalizeVel else self.UofT[self.iH,:]
        if self.VofT is not None:
            N_T = np.shape(self.VofT)[1]
            V = self.VofT[self.iH,:]/self.Uh if normalizeVel else self.VofT[self.iH,:] 
        if self.WofT is not None:
            N_T = np.shape(self.WofT)[1]
            W = self.WofT[self.iH,:]/self.Uh if normalizeVel else self.WofT[self.iH,:] 
        
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
    
    def plotTimeHistory(self,
                    figFile=None,
                    xLabel='t [s]',
                    yLabels=("U(t)","V(t)","W(t)"), 
                    xLimits='auto', # [tMin,tMax]
                    yLimits='auto', # ([Umin, Umax], [Vmin, Vmax], [Wmin, Wmax])
                    figSize=[15, 5],
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
                    normZ='Z',
                    normU='U',
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

