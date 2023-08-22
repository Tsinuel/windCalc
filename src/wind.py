# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 18:23:09 2022

@author: Tsinuel Geleta
"""
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import copy

from typing import List,Literal,Dict,Tuple,Any,Union,Set
from scipy import signal
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import skew,kurtosis
from scipy.interpolate import interp1d
from matplotlib.patches import Arc
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# internal imports
import windPlotting as wplt
import windCAD
import windCodes as wc

#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================

UNIT_CONV = {
    'fps2mps':0.3048,
    'mm2m':0.001,
    'mps2fps':1/0.3048,
    'm2mm':1/0.001,
    'in2m':0.0254,
    'm2in':1/0.0254,
    'ft2m':0.3048,
    'm2ft':1/0.3048,
    'mph2mps':0.44704,
    'mps2mph':1/0.44704,
    'kph2mps':1/3.6,
    'mps2kph':3.6,
    }
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
VON_KARMAN_CONST = 0.4
VALID_ERROR_TYPES = ['RMSE', 'MAE', 'NMAE', 'SMAPE', 'MSLE', 'NRMSE', 'RAE', 'MBD', 'PCC', 'RAE', 'R^2']
NORMALIZED_ERROR_TYPES = ['NMAE', 'NRMSE', 'RAE']
with open(PATH_SRC+r'/refData/bluecoeff.json', 'r') as f:
    BLUE_COEFFS = json.load(f)

def_cols = list(mcolors.TABLEAU_COLORS.values())
def_cols2 = list(mcolors.CSS4_COLORS.values())
def_cols3 = list(mcolors.XKCD_COLORS.values())
def_cols4 = list(mcolors.BASE_COLORS.values())
def_cols5 = list(mcolors.CSS4_COLORS.values())
def_mk = ['o','s','^','v','<','>','D','P','X','*','+','x','1','2','3','4','8','p','h','H','d','|','_']
def_ls = ['-','--','-.',':']

#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

#---------------------------------- GENERAL ------------------------------------
def isValidNumericScalar(value):
    try:
        if value is not None and (isinstance(value, (int, float, complex)) or
                                  np.isinf(value) or np.isnan(value)):
            return True
    except (TypeError, ValueError):
        pass
    return False

def isValidNumericVector(value):
    try:
        if value is not None and isinstance(value, (list, tuple, np.ndarray)) and \
            all(isValidNumericScalar(x) for x in value):
            return True
    except (TypeError, ValueError):
        pass
    return False

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

def lowpass(x, fs, fc, axis=-1, order = 4, resample=False):
    Wn = fc / (fs / 2)
    b, a = signal.butter(order, Wn, 'low')
    y = signal.filtfilt(b, a, x, axis=axis)
    if resample:
        y = signal.resample(y, int(np.shape(y)[axis] * fc / fs), axis=axis)
    return y

def mathName(rawname):
    if isinstance(rawname, (list, tuple, np.ndarray)):
        return [mathName(x) for x in rawname]

    # wind field
    if rawname == 'U':
        return '$U$ [$m/s$]'
    elif rawname == 'U/Uh':
        return '$U/U_h$'
    elif rawname == 'Uh':
        return '$U_h$ [$m/s$]'
    elif rawname == 'V':
        return '$V$ [$m/s$]'
    elif rawname == 'V/Uh':
        return '$V/U_h$'
    elif rawname == 'W':
        return '$W$ [$m/s$]'
    elif rawname == 'W/Uh':
        return '$W/U_h$'
    elif rawname == 'Iu':
        return '$I_u$'
    elif rawname == 'Iv':
        return '$I_v$'
    elif rawname == 'Iw':
        return '$I_w$'
    elif rawname == 'xLu':
        return '$^xL_u$ [$m$]'
    elif rawname == 'xLu/H':
        return '$^xL_u/H$'
    elif rawname == 'xLv':
        return '$^xL_v$ [$m$]'
    elif rawname == 'xLv/H':
        return '$^xL_v/H$'
    elif rawname == 'xLw':
        return '$^xL_w$ [$m/s$]'
    elif rawname == 'xLw/H':
        return '$^xL_w/H$'
    elif rawname == 'uv': 
        return '$\\overline{u\'v\'}$ [$m^2/s^2$]'
    elif rawname == 'uv/Uh^2':
        return '$\\overline{u\'v\'}/U_h^2$'
    elif rawname == 'uw':
        return '$\\overline{u\'w\'}$ [$m^2/s^2$]'
    elif rawname == 'uw/Uh^2':
        return '$\\overline{u\'w\'}/U_h^2$'
    elif rawname == 'vw':
        return '$\\overline{v\'w\'}$ [$m^2/s^2$]'
    elif rawname == 'vw/Uh^2':
        return '$\\overline{v\'w\'}/U_h^2$'
    elif rawname == 'z0':
        return '$z_0$ [$m$]'
    elif rawname == 'Je':
        return '$Je$'
    # Spectra
    elif rawname == 'Suu':
        return '$S_{uu}$ [$m^2/s$]'
    elif rawname == 'Svv':
        return '$S_{vv}$ [$m^2/s$]'
    elif rawname == 'Sww':
        return '$S_{ww}$ [$m^2/s$]'
    elif rawname == 'nSuu/Uh^2':
        return '$nS_{uu}/U_h^2$'
    elif rawname == 'nSuu/u^2':
        return '$nS_{uu}/\\sigma_u^2$'
    elif rawname == 'nSvv/Uh^2':
        return '$nS_{vv}/U_h^2$'
    elif rawname == 'nSvv/v^2':
        return '$nS_{vv}/\\sigma_v^2$'
    elif rawname == 'nSww/Uh^2':
        return '$nS_{ww}/U_h^2$'
    elif rawname == 'nSww/w^2':
        return '$nS_{ww}/\\sigma_w^2$'
    elif rawname == 'n':
        return '$n$'
    elif rawname == 'f':
        return '$f$'
    elif rawname == 'nH/Uh':
        return '$nH/U_h$'
    elif rawname == 'n/Uh':
        return '$n/U_h$'
    # Geometry and scaling
    elif rawname == 'H':
        return '$H$ [$m$]'
    elif rawname == 'T':
        return '$T$ [$s$]'
    elif rawname == 'T_star':
        return '$TU_h/H$'
    elif rawname == 'n_smpl':
        return r'$n_{smpl}$ [$Hz$]'
    elif rawname == 'f_smpl':
        return r'$n_{smpl}H/U_h$'
    elif rawname == 'lScl':
        return '$\lambda_L$ = 1:'
    elif rawname == 'vScl':
        return '$\lambda_V$ = 1:'
    elif rawname == 'tScl':
        return '$\lambda_T$ = 1:'
    elif rawname == 'Re':
        return '$Re$'
    # Cp
    elif rawname == 'mean':
        return '$\\overline{C_p}$'
    elif rawname == 'std':
        return '$\\sigma_{C_p}$'
    elif rawname == 'peak' or rawname == 'peakMax':
        return '$\hat{C_p}$'
    elif rawname == 'peakMin':
        return '$\check{C}_p$'
    elif rawname == 'skewness':
        return '$\\gamma_{C_p}$'
    elif rawname == 'kurtosis':
        return '$\\kappa_{C_p}$'
    else:
        warnings.warn("Unknown rawname: "+rawname)
        return rawname

def fullName(rawname, abbreviate=False):
    if isinstance(rawname, (list, tuple, np.ndarray)):
        return [fullName(x, abbreviate=abbreviate) for x in rawname]

    if rawname == 'U':
        return 'Mean longitudinal velocity'
    elif rawname == 'U/Uh':
        return 'Mean longitudinal velocity normalized by mean roof height velocity'
    elif rawname == 'Uh':
        return 'Mean roof height velocity'
    elif rawname == 'V':
        return 'Mean lateral velocity'
    elif rawname == 'V/Uh':
        return 'Mean lateral velocity normalized by mean roof height velocity'
    elif rawname == 'W':
        return 'Mean vertical velocity'
    elif rawname == 'W/Uh':
        return 'Mean vertical velocity normalized by mean roof height velocity'
    elif rawname == 'Iu':
        return 'Longitudinal turbulence intensity'
    elif rawname == 'Iv':
        return 'Lateral turbulence intensity'
    elif rawname == 'Iw':
        return 'Vertical turbulence intensity'
    elif rawname == 'xLu':
        return 'Longitudinal integral length scale'
    elif rawname == 'xLu/H':
        return 'Longitudinal integral length scale normalized by mean roof height'
    elif rawname == 'xLv':
        return 'Lateral integral length scale'
    elif rawname == 'xLv/H':
        return 'Lateral integral length scale normalized by mean roof height'
    elif rawname == 'xLw':
        return 'Vertical integral length scale'
    elif rawname == 'xLw/H':
        return 'Vertical integral length scale normalized by mean roof height'
    elif rawname == 'uv': 
        return 'Covariance of longitudinal and lateral velocity fluctuations'
    elif rawname == 'uv/Uh^2':
        return 'Normalized covariance of longitudinal and lateral velocity fluctuations'
    elif rawname == 'uw':
        return 'Covariance of longitudinal and vertical velocity fluctuations'
    elif rawname == 'uw/Uh^2':
        return 'Normalized covariance of longitudinal and vertical velocity fluctuations'
    elif rawname == 'vw':
        return 'Covariance of lateral and vertical velocity fluctuations'
    elif rawname == 'vw/Uh^2':
        return 'Normalized covariance of lateral and vertical velocity fluctuations'
    elif rawname == 'z0':
        return 'Roughness length'
    elif rawname == 'Je':
        return 'Jensen number'
    # Geometry and scaling
    elif rawname == 'H':
        return 'Mean roof height'
    elif rawname == 'T':
        return 'Sampling duration'
    elif rawname == 'T_star':
        return 'Non-dimensional sampling duration'
    elif rawname == 'n_smpl':
        return 'Sampling frequency'
    elif rawname == 'f_smpl':
        return 'Non-dimensional sampling frequency'
    elif rawname == 'lScl':
        return 'Geometric length scaling factor'
    elif rawname == 'vScl':
        return 'Velocity scaling factor'
    elif rawname == 'tScl':
        return 'Time scaling factor'    
    elif rawname == 'Re':
        return 'Reynolds number'
    # Cp
    elif rawname == 'mean':
        return r'Mean $C_p$'
    elif rawname == 'std':
        if abbreviate:
            return r'Std. $C_p$'
        else:
            return r'Standard deviation of $C_p$'
    elif rawname == 'peak':
        return r'Peak $C_p$'
    elif rawname == 'peakMin':
        if abbreviate:
            return r'Peak $C_p^-$'
        else:
            return r'Negative peak $C_p$'
    elif rawname == 'peakMax':
        if abbreviate:
            return r'Peak $C_p^+$'
        else:
            return r'Positive peak $C_p$'
    elif rawname == 'skewness':
        if abbreviate:
            return r'Skew. $C_p$'
        else:
            return r'Skewness of $C_p$'
    elif rawname == 'kurtosis':
        if abbreviate:
            return r'Kurt. $C_p$'
        else:
            return r'Kurtosis of $C_p$'
    else:
        warnings.warn("Unknown rawname: "+rawname)
        return rawname

def measureError(cfd=None, exp=None, 
                 errorTypes:Literal['all','RMSE', 'MAE', 'NMAE', 'SMAPE', 'MSLE', 'NRMSE', 'RAE', 'MBD', 'PCC', 'RAE', 'R^2',]=['RMSE','NRMSE','MAE','PCC','R^2'],
                 returnEqn=False, cfdName='CFD', expName='Exp'):
    '''
    Parameters
    ----------
    model : np.ndarray
        The model data.
    data : np.ndarray
        The data to be compared with the model.
    errorTypes : str or list of str, optional
        The error type(s) to be calculated. The default is 'all'.
        Error definitions:
            RMSE: Root Mean Square Error. 
                    $$\sqrt{\frac{1}{N}\sum_{i=1}^{N}(model_i - data_i)^2}$$
            MAE: Mean Absolute Error.
                    $$\frac{1}{N}\sum_{i=1}^{N}|model_i - data_i|$$
            NMAE: Normalized Mean Absolute Error.
                    $$\frac{1}{N}\sum_{i=1}^{N}\frac{|model_i - data_i|}{\bar{data}}$$
            SMAPE: Symmetric Mean Absolute Percentage Error.
                    $$\frac{1}{N}\sum_{i=1}^{N}\frac{|model_i - data_i|}{(|model_i| + |data_i|)/2}$$
            MSLE: Mean Squared Logarithmic Error.
                    $$\frac{1}{N}\sum_{i=1}^{N}(\log(model_i + 1) - \log(data_i + 1))^2$$
            NRMSE: Normalized Root Mean Square Error.
                    $$\frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N}(model_i - data_i)^2}}{\bar{data}}$$
            RAE: Relative Absolute Error.
                    $$\frac{\sum_{i=1}^{N}|model_i - data_i|}{\sum_{i=1}^{N}|data_i - \bar{data}|}$$
            MBD: Mean Bias Deviation.
                    $$\frac{\sum_{i=1}^{N}(model_i - data_i)}{\sum_{i=1}^{N}data_i}$$
            PCC: Pearson Correlation Coefficient.
                    $$\frac{\sum_{i=1}^{N}(model_i - \bar{model})(data_i - \bar{data})}{\sqrt{\sum_{i=1}^{N}(model_i - \bar{model})^2}\sqrt{\sum_{i=1}^{N}(data_i - \bar{data})^2}}$$
            RAW: Relative Absolute Error.
                    $$\frac{\sum_{i=1}^{N}|model_i - data_i|}{\sum_{i=1}^{N}|data_i - \bar{data}|}$$

    Returns
    -------
    error : dict
        The error value(s) for the given error type(s).
    '''

    if errorTypes == 'all':
        errorTypes = VALID_ERROR_TYPES
    elif isinstance(errorTypes, str):
        errorTypes = [errorTypes]
    elif not isinstance(errorTypes, list):
        raise Exception("Unknown error type(s): "+str(errorTypes))
    
    errorTypes = [x.upper() for x in errorTypes]
    if not all(x in VALID_ERROR_TYPES for x in errorTypes):
        raise Exception("Unknown error type(s): "+str(errorTypes))
    
    eqn = {
        'RMSE': r'$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(' + cfdName + r'_i - ' + expName + r'_i)^2}$',
        'MAE': r'$MAE = \frac{1}{N}\sum_{i=1}^{N}|' + cfdName + r'_i - ' + expName + r'_i|$',
        'NMAE': r'$NMAE = \frac{1}{N}\sum_{i=1}^{N}\frac{\left|' + cfdName + r'_i - ' + expName + r'_i\right|}{' + expName + r'_{max} - ' + expName + r'_{min}}$',
        'SMAPE': r'$SMAPE = \frac{1}{N}\sum_{i=1}^{N}\frac{|' + cfdName + r'_i - ' + expName + r'_i|}{(|' + cfdName + r'_i| + |' + expName + r'_i|)/2}$',
        'MSLE': r'$MSLE = \frac{1}{N}\sum_{i=1}^{N}(\log(' + cfdName + r'_i + 1) - \log(' + expName + r'_i + 1))^2$',
        'NRMSE': r'$NRMSE = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N}(' + cfdName + r'_i - ' + expName + r'_i)^2}}{' + expName + r'_{max} - ' + expName + r'_{min}}$',
        'RAE': r'$RAE = \frac{\sum_{i=1}^{N}|' + cfdName + r'_i - ' + expName + r'_i|}{\sum_{i=1}^{N}|' + expName + r'_i - \bar{' + expName + r'}|}$',
        'MBD': r'$MBD = \frac{\sum_{i=1}^{N}(' + cfdName + r'_i - ' + expName + r'_i)}{\sum_{i=1}^{N}' + expName + r'_i}$',
        'PCC': r'$PCC = \frac{\sum_{i=1}^{N}(' + cfdName + r'_i - \bar{' + cfdName + r'})(data_i - \bar{data})}{\sqrt{\sum_{i=1}^{N}(' + cfdName + r'_i - \bar{' + cfdName + r'})^2}\sqrt{\sum_{i=1}^{N}(data_i - \bar{data})^2}}$',
        'R^2': r'$R^2 = 1 - \frac{\sum_{i=1}^{N}(' + cfdName + r'_i - ' + expName + r'_i)^2}{\sum_{i=1}^{N}(' + expName + r'_i - \bar{' + expName + r'})^2}$',
    }

    if cfd is not None and exp is not None:
        error = {}
        for errorType in errorTypes:
            if errorType == 'RMSE':
                error['RMSE'] = np.sqrt(np.mean((cfd - exp)**2))
            elif errorType == 'MAE':
                error['MAE'] = np.mean(np.abs(cfd - exp))
            elif errorType == 'NMAE':
                error['NMAE'] = np.mean(np.abs(cfd - exp)) / (np.max(exp) - np.min(exp))
            elif errorType == 'SMAPE':
                error['SMAPE'] = np.mean(np.abs(cfd - exp)) / (np.mean(np.abs(cfd)) + np.mean(np.abs(exp)))
            elif errorType == 'MSLE':
                error['MSLE'] = np.mean((np.log(cfd + 1) - np.log(exp + 1))**2)
            elif errorType == 'NRMSE': 
                error['NRMSE'] = np.sqrt(np.mean((cfd - exp)**2)) / (np.max(exp) - np.min(exp))
            elif errorType == 'RAE':
                error['RAE'] = np.sum(np.abs(cfd - exp)) / np.sum(np.abs(exp - np.mean(exp)))
            elif errorType == 'MBD':
                error['MBD'] = np.sum(cfd - exp) / np.sum(exp)
            elif errorType == 'PCC':
                error['PCC'] = pearsonr(cfd.flatten(), exp.flatten())[0]
            elif errorType == 'RAE':
                error['RAE'] = np.sum(np.abs(cfd - exp)) / np.sum(np.abs(exp - np.mean(exp)))
            elif errorType == 'R^2':
                error['R^2'] = r2_score(exp, cfd)
            else:
                raise Exception("Unknown error type: "+errorType)
    else:
        error = np.nan
    if returnEqn:
        return error, eqn
    else:
        return error

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

def psd(x, fs, nAvg=8, overlapRatio=0.5, window:Literal['hanning', 'hamming', 'bartlett', 'blackman']='hanning', kwargs_window={}, kwargs_welch={}, 
        showPlots=False):
    if np.isscalar(x):
        raise Exception("The input 'x' must be a vector.")
    x = x - np.mean(x)
    N = len(x)
    nblock = int(np.floor(N/nAvg)) # number of samples per block
    overlap = int(np.floor(nblock*overlapRatio)) # number of samples to overlap
    win = eval('signal.'+window+'(nblock, **kwargs_window)')
    N = len(x) - (len(x) % nblock) # number of samples to use 
    if N < nblock: # avoid error and get the simplest PSD possible
        N = nblock
    f, Pxxf = signal.welch(x[0:N], fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, **kwargs_welch)
    if showPlots:
        plt.figure()
        plt.loglog(f, Pxxf)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
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
    '''
    Parameters
    ----------
    n : np.ndarray
        Frequency vector.
    U : float
        Mean wind speed.
    Iu : float, optional
        Longitudinal turbulence intensity. The default is None.
    Iv : float, optional
        Lateral turbulence intensity. The default is None.
    Iw : float, optional
        Vertical turbulence intensity. The default is None.
    xLu : float, optional
        Longitudinal integral length scale. The default is None.
    xLv : float, optional
        Lateral integral length scale. The default is None.
    xLw : float, optional
        Vertical integral length scale. The default is None.

    Returns
    -------
    Suu : np.ndarray
        Longitudinal velocity spectrum.
    Svv : np.ndarray
        Lateral velocity spectrum.
    Sww : np.ndarray
        Vertical velocity spectrum.

    '''
    Suu = Svv = Sww = None
    if Iu is not None and xLu is not None:
        Suu = vonKarmanSuu(n,U,Iu,xLu)
    if Iv is not None and xLv is not None:
        Svv = vonKarmanSuu(n,U,Iv,xLv)
    if Iw is not None and xLw is not None:
        Sww = vonKarmanSuu(n,U,Iw,xLw)
    return Suu, Svv, Sww

def Coh_Davenport():
    pass

def logProfile(Z, z0, Uref=None, Zref=None, uStar=None, d=0.0, detailedOutput=False):
    '''
    Parameters
    ----------
    Z : np.ndarray
        Height vector.
    Uref : float
        Mean wind speed at Zref.
    Zref : float
        Reference height.
    uStar : float
        Friction velocity.
    z0 : float
        Roughness length.
    d : float, optional
        Displacement height. The default is 0.0.

    Returns
    -------
    U : np.ndarray
        Wind speed vector.

    '''
    if uStar is None:
        if Uref is None or Zref is None:
            raise Exception("Either 'uStar' or 'Zref' and 'Uref' are required.")
        uStar = Uref * VON_KARMAN_CONST / np.log((Zref - d)/z0)
    elif Uref is not None or Zref is not None:
        warnings.warn("Both 'uStar' and 'Zref' and 'Uref' are given. Ignoring the latter two.")        

    U_func = lambda Z: uStar / VON_KARMAN_CONST * np.log((Z - d)/z0)
    U = U_func(Z)
    if detailedOutput:
        return U, U_func, uStar
    else:
        return U

def fitVelDataToLogProfile(Z, U, Zref=None, Uref=None, d=0.0, debugMode=False, uStar_init=1.0, z0_init=0.0000001) -> Tuple[float, float, np.ndarray]:
    from scipy.optimize import minimize

    if Zref is None and Uref is not None:
        try:
            Zref = np.interp(Uref, U, Z)
        except:
            Zref = None
    elif Zref is not None and Uref is None:
        try:
            Uref = np.interp(Zref, Z, U)
        except:
            Uref = None
    with_constraint = Zref is not None and Uref is not None

    U_func = lambda Z, uStar, z0: uStar / VON_KARMAN_CONST * np.log((Z - d) / z0)

    def costFunc(x, Z, U):
        uStar, z0 = x
        fit_error = np.sum((U_func(Z, uStar, z0) - U) ** 2)

        # Add an extra term to penalize deviation from the reference values
        if with_constraint:
            uStar_ref, z0_ref = x
            ref_error = (U_func(Zref, uStar_ref, z0_ref) - Uref) ** 2
            fit_error += ref_error

        return fit_error

    # Define the equality constraint to force the fitted profile to pass through (Zref, Uref)
    def constraint_func(x):
        uStar, z0 = x
        return U_func(Zref, uStar, z0) - Uref

    constraint = {'type': 'eq', 'fun': constraint_func}

    # Initial guess for optimization
    x0 = [uStar_init, z0_init]

    if with_constraint:
        res = minimize(costFunc, x0, args=(Z, U), method='SLSQP', constraints=constraint)
    else:
        res = minimize(costFunc, x0, args=(Z, U), method='Nelder-Mead')
    uStar, z0 = res.x
    U_fit = U_func(Z, uStar, z0)
    
    nmape = np.mean(np.abs(U_fit - U) / U)
    if nmape > 0.1:
        warnings.warn("The fit is not good enough. Try different initial guesses.")
    
    if debugMode:
        print("uStar = " + str(uStar) + " m/s")
        print("z0 = " + str(z0) + " m\n")
        print(pd.DataFrame({'Z': Z, 'U': U, 'U_fit': U_fit}))
        plt.figure(figsize=(6, 8))
        plt.plot(U, Z, '.-k', label='U')
        plt.plot(U_fit, Z, '-r', label='U_fit')
        if with_constraint:
            plt.axhline(Zref, color='k', linestyle='--', label='Zref')
            plt.axvline(Uref, color='k', linestyle='-.', label='Uref')
        y = 0.8
        plt.annotate('uStar = ' + str(uStar) + ' m/s', xy=(0.05, y), xycoords='axes fraction')
        plt.annotate('z0 = ' + str(z0) + ' m', xy=(0.05, y-0.05), xycoords='axes fraction')
        plt.annotate(f'fitting RMSE = {np.sqrt(np.mean((U_fit - U) ** 2)):.3f} m/s', xy=(0.05, y-0.1), xycoords='axes fraction')
        plt.xlabel('Z')
        plt.ylabel('U')
        plt.legend()
        plt.show()

    return z0, uStar, U_fit

def fitVelToPowerLawProfile(Z, U, Zref=None, Uref=None, debugMode=False, alpha_init=1.0, U0_init=1.0) -> Tuple[float, float, np.ndarray]:
    from scipy.optimize import minimize

    if Zref is None and Uref is not None:
        try:
            Zref = np.interp(Uref, U, Z)
        except:
            Zref = None
    elif Zref is not None and Uref is None:
        try:
            Uref = np.interp(Zref, Z, U)
        except:
            Uref = None
    with_constraint = Zref is not None and Uref is not None
    raise NotImplemented()

def get_velTH_stats_1pt(UofT: np.ndarray=None, 
                    VofT: np.ndarray=None, 
                    WofT: np.ndarray=None,
                    timeAxis=1, dt=None,
                    fields: List[Literal['U','V','W','Iu','Iv','Iw','xLu','xLv','xLw','uv','uw','vw']]=DEFAULT_VELOCITY_STAT_FIELDS,
                    orderFields=True,
                    ):
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
    
    if orderFields:
        stats = {k: stats[k] for k in fields}

    return stats

def coherence(pts1: np.ndarray, pts2: np.ndarray,
                vel1: np.ndarray, vel2: np.ndarray, fs,
                timeAxis=1, 
                ):
    if timeAxis == 0:
        vel1 = np.transpose(vel1)
        vel2 = np.transpose(vel2)
        timeAxis = 1

    dist = np.sqrt(np.sum((pts1[:,np.newaxis,:] - pts2[np.newaxis,:,:])**2, axis=-1))
    
    f, Coh = signal.coherence(vel1, vel2, fs=fs, nperseg=len(vel1), noverlap=0)
    
    return f, Coh, dist

def fitESDUgivenIuRef(
                    Zref,
                    IuRef,
                    z0i=[1e-10,10.0],
                    Uref=10.0,
                    phi=30,
                    ESDUversion:Literal['ESDU74', 'ESDU85']='ESDU85',
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
    try:
        Iu_0 = es.Iu(Z=Zref)
    except:
        raise Exception("It failed to calculate Iu for z0 = "+str(z0_0)+". Try different initial estimates.")
    es.z0 = z0_1
    try:
        Iu_1 = es.Iu(Z=Zref)
    except:
        raise Exception("It failed to calculate Iu for z0 = "+str(z0_1)+". Try different initial estimates.")

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
        try:
            Iu = es.Iu(Z=Zref)
        except:
            raise Exception("It failed to calculate Iu for z0 = "+str(z0)+". Try different initial estimates.")
        err = IuRef - Iu
        if err*err_0 < 0:
            z0_1 = z0
            err_1 = err
        else:
            z0_0 = z0
            err_0 = err
    z0 = (z0_0 + z0_1)/2
    return z0, es
    
#---------------------------- SURFACE PRESSURE ---------------------------------
def peak_gumbel(x, axis:int=0, 
                specs: dict=DEFAULT_PEAK_SPECS,
                unevenDataManagement: Literal['remove', 'padMean', 'padZero']='remove',
                detailedOutput=False, debugMode=False):
    x = np.array(x, dtype=float)
    ndim = x.ndim
    xShp = x.shape
    if xShp[axis] < specs['Num_seg']:
        raise Exception(f"The time dimension of {xShp[axis]} along axis {axis} is less than number of segments, N of {specs['Num_seg']}. The input shape you gave is: {xShp}")
    if debugMode:
        print(f"x_orig: {np.shape(x)}")
        print(f"axis: {axis}")
        print(f"ndim: {ndim}")

    # Extract N min/max values
    nUnevenExcessData = xShp[axis] % specs['Num_seg']
    if nUnevenExcessData > 0:
        if nUnevenExcessData > 0.25*xShp[axis]:
            warnings.warn(f"Excess data points ({nUnevenExcessData}) is more than 25% of the total data points ({xShp[axis]}).")
        if debugMode:
            print(f"Excess data found.")
        if unevenDataManagement == 'remove':
            if debugMode:
                print(f"Removing {nUnevenExcessData} data points from the end.")
            # slice the excess data considering x can be any dimension. apply the slice along the axis of interest
            x = x[tuple([slice(None)]*axis+[slice(0,-nUnevenExcessData)])]
        elif unevenDataManagement == 'padMean':
            # pad along the axis of interest
            x = np.pad(x, (0,specs['Num_seg']-xShp[axis]%specs['Num_seg']), mode='mean')
            if debugMode:
                print(f"Padding {specs['Num_seg']-xShp[axis]%specs['Num_seg']} data points with mean value.")
        elif unevenDataManagement == 'padZero':
            x = np.pad(x, (0,specs['Num_seg']-xShp[axis]%specs['Num_seg']), mode='constant')
            if debugMode:
                print(f"Padding {specs['Num_seg']-xShp[axis]%specs['Num_seg']} data points with zero.")
        else:
            raise Exception("Unknown excessDataManagement method: "+unevenDataManagement)
    
    if debugMode:
        print(f"x: {np.shape(x)}")
    if ndim == 1:
        x = np.array_split(x, specs['Num_seg'])
    else:
        x = np.array_split(x, specs['Num_seg'], axis=axis)
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
    '''
    Returns the peak values of the given time history along the given axis.

    Parameters
    ----------
    x : np.ndarray
        The time history of the variable of interest. Should be N-dimensional.
    axis : int, optional
        The axis along which the peak values are to be calculated. The default is 0.
    specs : dict, optional
        The specifications for the peak calculation. The default is DEFAULT_PEAK_SPECS.
    debugMode : bool, optional
        If True, prints out the intermediate steps of the calculation. The default is False.
    detailedOutput : bool, optional
        If True, returns the details of the calculation. The default is False.

    Returns
    -------
    peakMin : np.ndarray
        The minimum peak values along the given axis.
    peakMax : np.ndarray
        The maximum peak values along the given axis.
    details : dict
        The details of the calculation. Only returned if detailedOutput is True.
        
    '''
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

#-------------------------------- PLOTTING -------------------------------------
def formatAxis(ax, gridMajor=True, gridMinor=False, tickLabelSize=10, labelSize=12,
               tickTop=True, tickRight=True, gridColor_mjr=[0.8,0.8,0.8], gridColor_mnr=[0.85,0.85,0.85], numFormat: Literal['general','scientific','default']='general',
               tickDirection='in',
            #    kwargs_grid_major={'linestyle':'-', 'linewidth':1, 'color':'k'}, 
            #    kwargs_grid_minor={},
               ):
    # def custom_format(x,pos):
    #     return f"{x:g}"
    if numFormat == 'scientific':
        custom_format = lambda x,pos: f"{x:.1e}"
    elif numFormat == 'general':
        custom_format = lambda x,pos: f"{x:g}"
    else:
        custom_format = None

    if custom_format is not None:
        ax.xaxis.set_major_formatter(custom_format)
        ax.yaxis.set_major_formatter(custom_format)
    ax.tick_params(axis='both', which='major', labelsize=tickLabelSize, direction=tickDirection, top=tickTop, right=tickRight)
    ax.tick_params(axis='both', which='minor', labelsize=tickLabelSize, direction=tickDirection, top=tickTop, right=tickRight)
    ax.xaxis.label.set_size(labelSize)
    ax.yaxis.label.set_size(labelSize)
    if gridMajor:
        ax.grid(True, which='major', linestyle='-', linewidth=1, color=gridColor_mjr)
    if gridMinor:
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color=gridColor_mnr)
    return ax

def getPlotParams(pltType:Literal['default','Prof-BLWT','Prof-LES','Prof-CONT','Spect-BLWT','Spect-LES','Spect-CONT']='default') -> dict:
    pass

def sub_cmap(cmap, start=0.0, stop=1.0, n=100, reverse=False):
    cmap = plt.get_cmap(cmap) if type(cmap) == str else cmap
    if reverse:
        return mcolors.LinearSegmentedColormap.from_list(
            f"{cmap.name}_sub_{start}_{stop}_{n}_r",
            cmap(np.linspace(start, stop, n))[::-1]
        )
    else:
        return mcolors.LinearSegmentedColormap.from_list(
            f"{cmap.name}_sub_{start}_{stop}_{n}",
            cmap(np.linspace(start, stop, n))
        )

#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

#------------------------------- WIND FIELD ------------------------------------
class spectra:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self, name=None, UofT=None, VofT=None, WofT=None, samplingFreq=None, 
                 n=None, Suu=None, Svv=None, Sww=None, nSpectAvg=8,
                 Z=None, U=None, Iu=None, Iv=None, Iw=None,
                 xLu=None, xLv=None, xLw=None, keepTH=True):
        
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

        self.Refresh()
        if not keepTH:
            self.UofT = None
            self.VofT = None
            self.WofT = None

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
    def Refresh(self):
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
        Suu = wc.vonKarmanSuu(n=n,U=self.U,Iu=self.Iu,xLu=self.xLu)
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
        Svv = wc.vonKarmanSvv(n=n,U=self.U,Iv=self.Iv,xLv=self.xLv)
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
        Sww = wc.vonKarmanSww(n=n,U=self.U,Iw=self.Iw,xLw=self.xLw)
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

    def copy(self):
        return copy.deepcopy(self)

    """--------------------------------- Plotters -------------------------------------"""
    def plotSpect_any(self, f, S, ax=None, fig=None, figsize=[15,5], label=None, xLabel=None, yLabel=None, xLimits=None, yLimits=None, 
                      plotType: Literal['loglog', 'semilogx', 'semilogy']='loglog', 
                      kwargs_plot={'color': 'k', 'linestyle': '-'}, kwargs_legend={'loc': 'lower left', 'fontsize': 12}, kwargs_ax={}):
        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figsize)
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
            formatAxis(ax, numFormat='default', **kwargs_ax)
            plt.show()
        return fig, ax

    def plot(self, fig=None, ax_Suu=None, ax_Svv=None, ax_Sww=None, figsize=[15,4], label=None, 
                    xLabel=None, yLabel_Suu=None, yLabel_Svv=None, yLabel_Sww=None,
                    xLimits=None, yLimits=None, 
                    normalize=True, normZ:Literal['Z','xLi']='Z', normU:Literal['U','sigUi']='U',         
                    plotType: Literal['loglog', 'semilogx', 'semilogy']='loglog', avoidZeroFreq=True, 
                    kwargs_plt={'color': 'k', 'linestyle': '-'}, 
                    overlayThese:dict=None, overlayType:Literal['single','scatter','all_and_avg','errorBars']='single', kwargs_overlay={}, kwargs_overlay_all={}, 
                    showLegend=True,  kwargs_legend={'loc': 'lower left', 'fontsize': 12}, 
                    kwargs_ax={}):
        def addOverlay(ax, fld, name='_', kwargs_overlay={}):
            if overlayThese is None:
                return
            # print("Overlaying something ...")
            if fld in overlayThese:
                if plotType == 'loglog':
                    ax.loglog()
                elif plotType == 'semilogx':
                    ax.semilogx()
                elif plotType == 'semilogy':
                    ax.semilogy()
                firstIdx = 1 if avoidZeroFreq else 0

                if overlayType == 'single':
                    ax.plot(overlayThese['rf'][firstIdx:], overlayThese[fld][firstIdx:], 
                            label=name, **kwargs_overlay)
                elif overlayType == 'scatter':
                    ax.scatter(overlayThese['rf'][firstIdx:,:], overlayThese[fld][firstIdx:,:], 
                               label=name, **kwargs_overlay)
                elif overlayType == 'all_and_avg':
                    # check if the overlay data has the following keys in it: 'rf', 'rf_avg', fld, fld+'_avg'
                    if not all([k in overlayThese for k in ['rf', 'rf_avg', fld, fld+'_avg']]):
                        raise Exception(f"Overlay data must have the following keys: 'rf', 'rf_avg', {fld}, {fld}'_avg'")
                    ax.plot(overlayThese['rf'][firstIdx:,:], overlayThese[fld][firstIdx:,...], 
                             **kwargs_overlay_all)
                    ax.plot(overlayThese['rf_avg'][firstIdx:], overlayThese[fld+'_avg'][firstIdx:], 
                            label=name, **kwargs_overlay)                
                elif overlayType == 'errorBars':
                    print("The boxplot for spectra needs revision!.")
                    warnings.warn("The boxplot for spectra needs revision!.")
                    if kwargs_overlay == {}:
                        kwargs_overlay = {
                            'widths': 0.25,
                            'notch': True,
                            'vert': False,
                            'showfliers': False,
                            'patch_artist': True,
                            'meanline': True,
                            'boxprops': dict(facecolor='w', color='k', linewidth=1.25),
                            'medianprops': dict(color='k', linewidth=1.25),
                            'whiskerprops': dict(color='k', linewidth=1.25),
                            'capprops': dict(color='k', linewidth=1.25)
                            }
                    # keep the existing y-axis ticks and labels as they will be overwritten by boxplot
                    yTicks = ax.get_yticks()
                    hadYLabel = ax.get_ylabel() != ''
                    bx = ax.boxplot(overlayThese[fld], positions=overlayThese['Z'], 
                            **kwargs_overlay)
                    ax.set_yticks(yTicks)
                    if hadYLabel:
                        ax.set_yticklabels(['{:g}'.format(y) for y in yTicks])
                    if name != '_':
                        bx['boxes'][0].set_label(name)
                        # return the legend handle for the boxplot
                        return bx #['boxes'][0]
                else:
                    raise NotImplementedError("Overlay type '{}' not implemented".format(overlayType))
                return None

        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figsize)
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

        yLimits = [yLimits]*3 if yLimits is None else yLimits

        if avoidZeroFreq and n[0] == 0:
            n = n[1:]
            Suu = Suu[1:]
            Svv = Svv[1:]
            Sww = Sww[1:]
        
        if ax_Suu is not None:
            name = overlayThese['name'] if overlayThese is not None and 'name' in overlayThese else '_'
            _ = addOverlay(ax_Suu, 'rSuu', name, kwargs_overlay=kwargs_overlay)
            self.plotSpect_any(n, Suu, ax=ax_Suu, fig=fig, figsize=figsize, label=label, xLabel=xLabel, yLabel=yLabel_Suu, xLimits=xLimits, yLimits=yLimits[0], 
                            kwargs_plot=kwargs_plt, plotType=plotType)
        if ax_Svv is not None:
            name = overlayThese['name'] if overlayThese is not None and 'name' in overlayThese else '_'
            _ = addOverlay(ax_Svv, 'rSvv', name, kwargs_overlay=kwargs_overlay)
            self.plotSpect_any(n, Svv, ax=ax_Svv, fig=fig, figsize=figsize, label=label, xLabel=xLabel, yLabel=yLabel_Svv, xLimits=xLimits, yLimits=yLimits[1],
                                kwargs_plot=kwargs_plt, plotType=plotType)
        if ax_Sww is not None:
            name = overlayThese['name'] if overlayThese is not None and 'name' in overlayThese else '_'
            _ = addOverlay(ax_Sww, 'rSww', name, kwargs_overlay=kwargs_overlay)
            self.plotSpect_any(n, Sww, ax=ax_Sww, fig=fig, figsize=figsize, label=label, xLabel=xLabel, yLabel=yLabel_Sww, xLimits=xLimits, yLimits=yLimits[2],
                                kwargs_plot=kwargs_plt, plotType=plotType)

        if showLegend:
            ax_Suu.legend(**kwargs_legend)

        if newFig:
            formatAxis(ax_Suu, numFormat='default', **kwargs_ax)
            formatAxis(ax_Svv, numFormat='default', **kwargs_ax)
            formatAxis(ax_Sww, numFormat='default', **kwargs_ax)
            plt.show()
        return fig, ax_Suu, ax_Svv, ax_Sww

class profile:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                name="profile", 
                profType: Union[Literal["continuous","discrete","scatter"], None] = None,
                X: np.ndarray=None,
                Y: np.ndarray=None,
                Z: np.ndarray=None,
                workSect_zLim: list=None,
                H=None, 
                dt=None, 
                t=None,
                lScl=1.0,
                UofT: np.ndarray=None, 
                VofT: np.ndarray=None,
                WofT: np.ndarray=None,
                pOfT: np.ndarray=None,
                fields=VALID_VELOCITY_STAT_FIELDS,
                stats: dict ={},
                SpectH: spectra =None, 
                nSpectAvg=8,
                fileName=None,
                keepTH=True,
                interpolateToH=False, 
                units=DEFAULT_SI_UNITS,
                kwargs_z0_fit={'fitTo':'Iu', # 'Iu' or 'U'
                               'uStar_init':1.0, 
                               'z0_init':0.001, 
                               'kwargs_z0Fit':{'z0i':[1e-10,1.0]}},
                ):
        '''
        Parameters
        ----------
        name : str, optional
            Name of the profile. The default is "profile".
        profType : Union[Literal["continuous","discrete","scatter"], None], optional
            Type of the profile. The default is None.
        X : np.ndarray, optional
            A vector of X coordinates for the profile. Shape: [N_pts]. The default is None.
        Y : np.ndarray, optional
            A vector of Y coordinates for the profile. Shape: [N_pts]. The default is None.
        Z : np.ndarray, optional
            A vector of Z coordinates for the profile. Shape: [N_pts]. The default is None.
        H : float, optional
            A reference height (e.g. building height) for the profile. The default is None.
        dt : float, optional
            Time step. The default is None.
        t : np.ndarray, optional
            A vector of time. The default is None.
        UofT : np.ndarray, optional
            A matrix of U velocity time series. Shape: [N_pts, nTime]. The default is None.
        VofT : np.ndarray, optional
            A matrix of V velocity time series. Shape: [N_pts, nTime]. The default is None.
        WofT : np.ndarray, optional
            A matrix of W velocity time series. Shape: [N_pts, nTime]. The default is None.
        pOfT : np.ndarray, optional
            A matrix of pressure time series at all or limited number of points from the 
            profile. This pressure is used for pressure coefficient calculation. 
            Shape: [N_pts, nTime]. The default is None.
        fields : List, optional
            A list of fields to be calculated. The default is found in VALID_VELOCITY_STAT_FIELDS.
        stats : Dict, optional
            A dictionary of pre-calculated statistics. The default is {}.
        SpectH : spectra, optional
            A pre-defined spectra object at the reference height. The default is None.
        nSpectAvg : int, optional
            Number of averaging windows for spectra calculation. The default is 8.
        fileName : str, optional
            Name of the file to read data from. The default is None.
        keepTH : bool, optional
            Keep time history data in memory. The default is True.
        interpolateToH : bool, optional
            Interpolate statistics to the reference height as opposed to using the nearest.
            The default is False.
        units : str, optional
            Units of the data. The default is DEFAULT_SI_UNITS.

        Returns
        -------
        None.
        '''

        self.name = name
        self.profType = profType
        self.X: Union[np.ndarray, None] = np.array(X, float) if X is not None else None
        self.Y: Union[np.ndarray, None] = np.array(Y, float) if Y is not None else None
        self.Z: Union[np.ndarray, None] = np.array(Z, float) if Z is not None else None
        self.stats_core: Dict = stats
        self.fields = fields

        self.H = H
        self.dt = dt
        self.samplingFreq = None if dt is None else 1/dt
        self.t = t
        self.lScl = lScl
        self.UofT: np.ndarray = UofT  # [N_pts x nTime]
        self.VofT: np.ndarray = VofT  # [N_pts x nTime]
        self.WofT: np.ndarray = WofT  # [N_pts x nTime]
        self.pOfT: np.ndarray = pOfT  # [N_pts x nTime]
        
        self.SpectH : spectra = SpectH
        self.nSpectAvg = nSpectAvg
        
        self.origFileName = fileName
        self.interpolateToH = interpolateToH
        self.units = units

        self.keepTH = keepTH

        self.z0_Iu = None
        self.z0_U = None
        self.kwargs_z0_fit = kwargs_z0_fit

        if workSect_zLim is None and self.Z is not None:
            self.workSect_zLim = [self.Z[0], self.Z[-1]]
        else:
            self.workSect_zLim = workSect_zLim

        self.Refresh()
        if not keepTH:
            self.t = None
            self.UofT = None
            self.VofT = None
            self.WofT = None
            self.pOfT = None

    
    def __verifyData(self):
        if self.UofT is not None and self.VofT is not None and self.WofT is not None:
            if not np.shape(self.UofT) == np.shape(self.VofT) == np.shape(self.WofT):
                raise Exception("UofT, VofT, and WofT must have the same shape.")
            if not np.shape(self.UofT)[0] == len(self.Z):
                raise Exception("UofT, VofT, and WofT must have the same number of points as Z.")

    def __computeVelStats(self):
        if all([self.UofT is None, self.VofT is None, self.WofT is None]):
            return
        N_T = np.shape(self.UofT)[1]
        
        if self.t is None and self.dt is not None:
            self.t = np.linspace(0,(N_T-1)*self.dt,num=N_T)
        if self.dt is None and self.t is not None:
            self.dt = np.mean(np.diff(self.t))

        self.stats_core = get_velTH_stats_1pt(UofT=self.UofT, VofT=self.VofT, WofT=self.WofT, dt=self.dt,
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
        
        self.SpectH = spectra(name=self.name, UofT=uOfT, VofT=vOfT, WofT=wOfT, samplingFreq=self.samplingFreq, Z=self.H, nSpectAvg=self.nSpectAvg, keepTH=self.keepTH)

    def __str__(self):
        return self.name

    """----------------------------------- Properties ---------------------------------"""
    @property
    def idx_eff(self):
        if self.Z is None or self.workSect_zLim is None:
            return None
        else:
            return np.where((self.Z >= self.workSect_zLim[0]) & (self.Z <= self.workSect_zLim[1]))[0]

    @property
    def Z_eff(self):
        if self.Z is None or self.idx_eff is None:
            return None
        else:
            return self.Z[self.idx_eff]

    @property
    def T(self):
        '''Duration of the time series in seconds'''
        if self.t is not None:
            return self.t[-1] - self.t[0]
        elif any([self.UofT is not None, self.VofT is not None, self.WofT is not None]) and self.dt is not None:
            return self.dt * (self.N_t-1)
        else:
            return None

    @property
    def T_star(self):
        '''Normalized duration, T.Uh/H'''
        dur = self.T
        if dur is None or self.H is None or self.Uh is None:
            return None
        else:
            return dur * self.Uh/self.H

    @property
    def N_t(self):
        if self.t is not None:
            return len(np.asarray(self.t))
        elif self.UofT is not None:
            return np.shape(self.UofT)[-1]
        elif self.VofT is not None:
            return np.shape(self.VofT)[-1]
        elif self.WofT is not None:
            return np.shape(self.WofT)[-1]
    
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
        if self.stats_core is None or 'U' not in self.stats_core:
            return None
        return self.stats_core['U']

    @property
    def Iu(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'Iu' not in self.stats_core:
            return None
        return self.stats_core['Iu']
    
    @property
    def Iv(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'Iv' not in self.stats_core:
            return None
        return self.stats_core['Iv']
    
    @property
    def Iw(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'Iw' not in self.stats_core:
            return None
        return self.stats_core['Iw']
    
    @property
    def xLu(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'xLu' not in self.stats_core:
            return None
        return self.stats_core['xLu']
    
    @property
    def xLv(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'xLv' not in self.stats_core:
            return None
        return self.stats_core['xLv']
    
    @property
    def xLw(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'xLw' not in self.stats_core:
            return None
        return self.stats_core['xLw']
    
    @property
    def uw(self) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'uw' not in self.stats_core:
            return None
        return self.stats_core['uw']

    @property
    def Je(self) -> Union[float, None]:
        ''' Computes the Jensen number, Je = H/z0, where z0 is the roughness length and H is the reference height.'''
        z0 = self.z0_Iu
        if self.stats_core is None or self.H is None or z0 is None:
            return None
        else:
            return self.H/z0
        
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
    
    @property
    def stats_norm(self):
        if self.stats_core is None:
            return None
        stats = {}
        for st in self.stats_core.keys():
            temp,_ = self.stat_norm(st)
            stats[st] = temp
        return stats
    
    @property
    def stats_norm_eff(self):
        if self.stats_core is None:
            return None
        stats = {}
        for st in self.stats_core.keys():
            temp,_ = self.stat_norm(st)
            stats[st] = temp[self.idx_eff]
        return stats

    @property
    def stats_at_H(self):
        if self.stats_core is None:
            return None
        stats = {}
        for st in self.stats_core.keys():
            stats[st] = self.stat_at_H(st)
        return stats

    """----------------------------------- Methods ------------------------------------"""
    def stat_at_H(self, field):
        if self.H is None or self.stats_core is None:
            return None
        else:
            return self.stats_core[field][self.H_idx]

    def stat_norm(self, field):
        '''Returns the normalized value of the given field at the given height and the name of the normalization
        
        Parameters
        ----------
        field : str
            Name of the field to be normalized. Choose from {'U','Iu','Iv','Iw','xLu','xLv','xLw','uv','uw','vw'}

        Returns
        -------
        normVal : np.ndarray
            Normalized value of the given field at the given height
        normName : str
            Name of the normalization

        Raises
        ------
        NotImplementedError
            If the normalization for the given field is not implemented yet

            '''
        if self.stats_core is None:
            return None, ''
        else:
            if field in ['U','V','W']:
                if self.Uh is None:
                    return None, ''
                return self.stats_core[field]/self.Uh, mathName(field+'/Uh')
            elif field in ['Iu','Iv','Iw']:
                return self.stats_core[field], mathName(field)
            elif field in ['xLu','xLv','xLw']:
                if self.H is None:
                    return None, ''
                return self.stats_core[field]/self.H, mathName(field+'/H')
            elif field in ['uv','uw','vw']:
                if self.Uh is None:
                    return None, ''
                return self.stats_core[field]/(self.Uh**2), mathName(field+'/Uh^2')
            else:
                raise NotImplementedError("Normalization for field '{}' not implemented".format(field))

    def stats_norm_interp_to_Z(self, Z):
        if self.stats_core is None:
            return None
        stats_full = self.stats_norm_eff
        Z_full = self.Z_eff
        stats = {}
        for st in stats_full.keys():
            stats[st] = np.interp(Z, Z_full, stats_full[st])
        return stats

    def paramsTable(self, normalized=True, fields=None,) -> dict:
        if fields is None:
            fields = list(self.stats_core.keys())
            fields.extend(['Name','H','Uh','T','n_smpl','z0','Je'])
            
        data = {}
        if 'Name' in fields:
            data['Name'] = self.name
        if 'H' in fields:
            data[mathName('H')+' @MS'] = self.H
        if 'Uh' in fields:
            data[mathName('Uh')+' @MS'] = self.Uh

        for st in self.stats_core.keys():
            if st in fields:                
                if normalized:
                    temp,name = self.stat_norm(st)
                    data[name] = temp[self.H_idx]
                else:
                    data[mathName(st)] = self.stat_at_H(st)
        if 'z0' in fields:
            data[mathName('z0')+' @FS'] = self.z0_Iu/self.lScl if self.z0_Iu is not None else None
        if 'Je' in fields:
            data[mathName('Je')] = self.Je

        if 'T' in fields:
            if normalized:
                data[mathName('T_star')] = self.T_star
            else:
                data[mathName('T')] = self.T
        if 'n_smpl' in fields:
            if normalized:
                if self.samplingFreq is None or self.H is None or self.Uh is None:
                    data[mathName('f_smpl')] = None
                else:
                    data[mathName('f_smpl')] = self.samplingFreq * self.H / self.Uh 
            else:
                data[mathName('n_smpl')] = self.samplingFreq

        return data

    """-------------------------------- Data handlers ---------------------------------"""
    def Refresh(self):
        self.__verifyData()
        self.__computeVelStats()
        # try:
        #     _ = self.fit_z0()
        # except:
        #     print("Could not fit z0. Call fit_z0() to see the error details and set the z0 value manually. Skipping...")

    def writeToFile__to_be_depricated(self,outDir,
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

    def copy(self):
        return copy.deepcopy(self)

    def getCoherence(self, 
                     D, 
                     dxn:Literal['x','y','z']='z', 
                     **kwargs):
        idx1 = [self.H_idx, self.H_idx+3, self.H_idx+6]
        pts1 = np.array([[0.0, 0.0, self.Z[i]] for i in idx1])
        dIdxs = [np.argmin(np.abs(self.Z-self.H-d)) for d in D]
        pts2 = np.array([[0.0, 0.0, self.Z[i]] for i in dIdxs])

        print(f"Shapes of pts1 and pts2: {np.shape(pts1)}, {np.shape(pts2)}")

        Coh = {}

        u1 = self.UofT[idx1,:]
        u2 = self.UofT[dIdxs,:]
        print(f"Shapes of u1 and u2: {np.shape(u1)}, {np.shape(u2)}")
        f, Coh['zu'], dist = coherence(pts1=pts1, pts2=pts2, vel1=u1, vel2=u2, fs=self.samplingFreq, timeAxis=1, **kwargs)

        v1 = np.array([self.VofT[self.H_idx,:]]).T
        v2 = np.array([self.VofT[dIdxs,:]]).T
        _, Coh['zv'], _ = coherence(pts1=pts1, pts2=pts2, vel1=v1, vel2=v2, fs=self.samplingFreq, timeAxis=1, **kwargs)

        w1 = np.array([self.WofT[self.H_idx,:]]).T
        w2 = np.array([self.WofT[dIdxs,:]]).T
        _, Coh['zw'], _ = coherence(pts1=pts1, pts2=pts2, vel1=w1, vel2=w2, fs=self.samplingFreq, timeAxis=1, **kwargs)

        return f, Coh, dist

    def U10_atModelScale(self, lScl) -> Union[np.ndarray, None]:
        if self.stats_core is None or 'U' not in self.stats_core:
            return None
        Z10 = 10*lScl
        return np.interp(Z10, self.Z, self.U)

    def fit_z0(self, fitTo:Literal['U','Iu']='Iu', uStar_init=None, z0_init=None, debugMode=True,
                  kwargs_z0Fit={}) -> Union[float, None]:
        
        kwargs_z0Fit = self.kwargs_z0_fit['kwargs_z0Fit'] if kwargs_z0Fit == {} else kwargs_z0Fit
        z0_init = self.kwargs_z0_fit['z0_init'] if z0_init is None else z0_init
        uStar_init = self.kwargs_z0_fit['uStar_init'] if uStar_init is None else uStar_init
        if self.stats_core is None:
            print("No statistics found. Cannot fit z0.")
            return None
        
        Z, U = self.Z_eff, self.U[self.idx_eff]
        self.z0_U, _, _ = fitVelDataToLogProfile(Z, U, Zref=self.H, Uref=self.Uh, d=0.0, debugMode=False, uStar_init=uStar_init, z0_init=z0_init,) # **kwargs_z0Fit)
        es_U = ESDU85(z0=self.z0_U/self.lScl)

        if self.SpectH is None or self.SpectH.Iu is None or self.H is None or self.lScl is None:
            return None
        self.z0_Iu, es_Iu = fitESDUgivenIuRef(self.H/self.lScl, self.SpectH.Iu, **kwargs_z0Fit)
        self.z0_Iu *= self.lScl

        if debugMode:
            profs = Profiles([self, es_Iu.toProfileObj(), es_U.toProfileObj()])
            profs.profiles[1].name = 'ESDU85_IuFit(z0={:.2g}m @FS)'.format(self.z0_Iu/self.lScl)
            profs.profiles[2].name = 'ESDU85_Ufit(z0={:.2g}m @FS)'.format(self.z0_U/self.lScl)
            _ = profs.plot__(fig=plt.figure(figsize=[16,6]),zLim=self.workSect_zLim/self.H, IuLim=[0,100], Ulim=[0,1.5],
                                col=['k','r','b'],
                                marker_Iu=   ['o','None','None'], linestyle_Iu=['-','-.',':' ],
                                marker_U=    ['o','None','None'], linestyle_U= ['-','--','-.'],
                                marker_Spect=   ['None','None','None'],
                                linestyle_Spect=['-','-.','--'],
                                alpha_Spect=[0.5, 1.0, 1.0],
                                IuLgndLoc='upper center', UlgndLoc='center',
                                # fontSz_axLbl=14, fontSz_axNum=12, fontSz_lgnd=12,
                                # freqLim=[1e-4, 5], rSuuLim=[1e-3,0.5],
                                )
        
        if fitTo == 'U':
            z0 = self.z0_U
        elif fitTo == 'Iu':
            z0 = self.z0_Iu
        else:
            raise NotImplementedError("Fitting to '{}' not implemented".format(fitTo))
        return z0

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
            yLabel = '$Z/H$' if yLabel is None else yLabel
        else:
            Z = self.Z
            H = self.H
            F = self.stats_core[fld]
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
            zLabel = '$Z/H$' if normalize else '$Z$'
        
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

    def plotProfile_basic2(self, fig=None, axs=None, figsize=[12,12], label=None, normalize=True,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None,
                            overlayThese:dict=None, overlayType:Literal['single','scatter','errorBars']='single', kwargs_overlay={}, 
                            yLimits=None, lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt={'color': 'k', 'linestyle': '-'}, kwargs_ax={}):
        '''
        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on. The default is None.
        axs : matplotlib.axes._subplots.AxesSubplot, optional
            Axes to plot on. The default is None.
        figsize : list, optional
            Figure size. The default is [12,10].
        label : str, optional
            Label for the profile. The default is None.
        normalize : bool, optional
            Normalize the profile. The default is True.
        xLabel : str, optional
            Label for the x-axis. The default is None.
        zLabel : str, optional
            Label for the z-axis. The default is None.
        xLimits_U : list, optional
            Limits for the x-axis of the U profile. The default is None.
        xLimits_Iu : list, optional
            Limits for the x-axis of the Iu profile. The default is None.
        xLimits_Iv : list, optional
            Limits for the x-axis of the Iv profile. The default is None.
        xLimits_Iw : list, optional
            Limits for the x-axis of the Iw profile. The default is None.
        xLimits_xLu : list, optional
            Limits for the x-axis of the xLu profile. The default is None.
        xLimits_xLv : list, optional
            Limits for the x-axis of the xLv profile. The default is None.
        xLimits_xLw : list, optional
            Limits for the x-axis of the xLw profile. The default is None.
        xLimits_uw : list, optional
            Limits for the x-axis of the uw profile. The default is None.
        overlayThese : dict, optional
            A dictionary of fields to overlay on the profile. If it include a field called 
            'name', it will be used in the legend. It must have a field called 'Z' which
            is the profile height. The normalization, if any, is assumed to be pre-applied.
            The default is None.
        '''
        def addOverlay(ax, fld, name='_', kwargs_overlay={}):
            if overlayThese is None:
                return
            # print("Overlaying something ...")
            if fld in overlayThese:
                if overlayType == 'single':
                    ax.plot(overlayThese[fld], overlayThese['Z'], 
                            label=name, **kwargs_overlay)
                elif overlayType == 'scatter':
                    ax.scatter(overlayThese[fld], overlayThese['Z'], 
                               label=name, **kwargs_overlay)
                elif overlayType == 'errorBars':
                    if kwargs_overlay == {}:
                        kwargs_overlay = {
                            'widths': 0.25,
                            'notch': True,
                            'vert': False,
                            'showfliers': False,
                            'patch_artist': True,
                            'meanline': True,
                            'boxprops': dict(facecolor='w', color='k', linewidth=1.25),
                            'medianprops': dict(color='k', linewidth=1.25),
                            'whiskerprops': dict(color='k', linewidth=1.25),
                            'capprops': dict(color='k', linewidth=1.25)
                            }
                    # keep the existing y-axis ticks and labels as they will be overwritten by boxplot
                    yTicks = ax.get_yticks()
                    hadYLabel = ax.get_ylabel() != ''
                    bx = ax.boxplot(overlayThese[fld], positions=overlayThese['Z'], 
                            **kwargs_overlay)
                    ax.set_yticks(yTicks)
                    if hadYLabel:
                        ax.set_yticklabels(['{:g}'.format(y) for y in yTicks])
                    if name != '_':
                        bx['boxes'][0].set_label(name)
                        # return the legend handle for the boxplot
                        return bx #['boxes'][0]
                else:
                    raise NotImplementedError("Overlay type '{}' not implemented".format(overlayType))
                return None
     
        newFig = False
        if fig is None:
            newFig = True
            fig, axs = plt.subplots(3,3)
            fig.set_size_inches(figsize)
        
        label = self.name if label is None else label
        if zLabel is None:
            zLabel = '$Z/H$' if normalize else '$Z$'
        
        bxPltObj = None
        if 'U' in self.stats_core.keys():
            self.plotProfile_any('U', ax=axs[0,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_U, yLimits=yLimits, kwargs=kwargs_plt)
            name = overlayThese['name'] if overlayThese is not None and 'name' in overlayThese else '_'
            bxPltObj = addOverlay(axs[0,0], 'U', name, kwargs_overlay=kwargs_overlay)
        if 'uw' in self.stats_core.keys():
            self.plotProfile_any('uw', ax=axs[0,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_uw, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[0,1], 'uw', kwargs_overlay=kwargs_overlay)
        if 'Iu' in self.stats_core.keys():
            self.plotProfile_any('Iu', ax=axs[1,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iu, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[1,0], 'Iu', kwargs_overlay=kwargs_overlay)
        if 'Iv' in self.stats_core.keys():
            self.plotProfile_any('Iv', ax=axs[1,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iv, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[1,1], 'Iv', kwargs_overlay=kwargs_overlay)
        if 'Iw' in self.stats_core.keys():
            self.plotProfile_any('Iw', ax=axs[1,2], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_Iw, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[1,2], 'Iw', kwargs_overlay=kwargs_overlay)
        if 'xLu' in self.stats_core.keys():
            self.plotProfile_any('xLu', ax=axs[2,0], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLu, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[2,0], 'xLu', kwargs_overlay=kwargs_overlay)
        if 'xLv' in self.stats_core.keys():
            self.plotProfile_any('xLv', ax=axs[2,1], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLv, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[2,1], 'xLv', kwargs_overlay=kwargs_overlay)
        if 'xLw' in self.stats_core.keys():
            self.plotProfile_any('xLw', ax=axs[2,2], label=label, normalize=normalize, xLabel=xLabel, yLabel=zLabel, xLimits=xLimits_xLw, yLimits=yLimits, kwargs=kwargs_plt)
            addOverlay(axs[2,2], 'xLw', kwargs_overlay=kwargs_overlay)
        
        if newFig:
            axs[0,2].axis('off')
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                legend_handles = [axs[0,0].get_lines()]
                if bxPltObj is not None:
                    print(bxPltObj is not None)
                    legend_handles.append(bxPltObj)
                fig.legend(legend_handles, bbox_transform=axs[0,2].transAxes, **lgnd_kwargs)
            for ax in axs.flatten():
                formatAxis(ax, **kwargs_ax)
            plt.show()
        return fig, axs, bxPltObj
    
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
        if self.SpectH is not None:
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

    def plotBasicStatsTable(self,
                                fig=None,
                                ax=None,
                                precision=2,):
        if self.stats_core is None:
            return
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = plt.subplot()
        ax.axis('off')
        ax.axis('tight')
        cell_text = []
        # manually add some parameters
        cell_text.append(['Name', self.name])
        cell_text.append(['H', np.round(self.H, precision)])
        cell_text.append(['Uh', np.round(self.Uh, precision)])
        
        for key in self.stats_core.keys():
            cell_text.append([key, np.round(self.stats_core[key][self.H_idx], precision)])
        # manually add some parameters
        cell_text.append(['T', self.T])
        cell_text.append(['T*', self.T_star])
        cell_text.append(['samplingFreq', self.samplingFreq])

        
        ax.table(cellText=cell_text,
                 colLabels=['Field','Value'],
                 loc='center')
        fig.tight_layout()
        plt.show()

class ESDU74:
    __Omega = 72.7e-6       # Angular rate of rotation of the earth in [rad/s].  ESDU 74031 sect. A.1
    __k = 0.4               # von Karman constant

    def __init__(self,
                    phi=30,
                    z0=0.03,
                    Z=None,
                    d=0.0,
                    Zref=10.0,
                    Uref=10.0,
                    ):
        self.phi = phi # latitude in degrees
        self.z0 = z0
        self.Zref = Zref
        self.Uref = Uref
        if Z is None:
            Z = np.sort(np.append(np.logspace(np.log10(1.01*z0+d),np.log10(300),99),10))
        self.Z = Z
        if not self.Zref in self.Z:
            self.Z = np.sort(np.append(self.Z, self.Zref))
        # remove all Z values that are smaller than z0
        self.Z = self.Z[self.Z >= self.z0]

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
        return self.U(self.Zg())        # ESDU 74031, eq. A.15

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

    def Fu(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return np.multiply(0.867 + 0.556*np.log10(Z) - 0.246*np.power(np.log10(Z),2), self.lambda_())     # ESDU 74031, eq. A.4a

    def Iu(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return np.divide(self.Fu(Z), 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def Iv(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        Fv = 0.655 + 0.201*np.log10(Z) - 0.095*np.power(np.log10(Z),2)     # ESDU 74031, eq. A.5
        return np.divide(Fv, 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def Fw(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return 0.381 + 0.172*np.log10(Z) - 0.062*np.power(np.log10(Z),2)     # ESDU 74031, eq. A.6

    def Iw(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        return np.divide(self.Fw(Z), 
                        np.log(np.divide(Z, self.z0)) )      # ESDU 74031, eq. A.3

    def uw(self,Z=None):
        Z = self.Zd(self.Z) if Z is None else self.Zd(Z)
        sigU_sigW = self.Iu(Z)*self.Iw(Z)*np.power(self.U(Z),2)
        Ro = np.divide(self.Ug(), self.f()*self.z0)
        Vstar_VG = np.divide(0.31, np.log10(Ro)) - 0.012
        alphaG = 58 - 5*np.log10(Ro)
        return sigU_sigW * -0.16/(self.Fu(Z)*self.Fw(Z)) * (1-(self.f()*Z*np.sin(alphaG))/(self.Ug()*(Vstar_VG)**2))     # ESDU 74031, eq. A.10

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
            name = f'ESDU-74 (z0={self.z0:.3g}m)'

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
        stats['uw'] = self.uw()

        return profile(name=name, 
                        profType="continuous", 
                        Z=self.Z, H=self.Zref,
                        stats=stats,
                        SpectH=spect)

class ESDU85:
    __Omega = 72.9e-6            # Angular rate of rotation of the earth [rad/s] ESDU 82026 ssA1

    def __init__(self,
                    phi=30.0,
                    z0=0.03,
                    Z=None,
                    d=0.0,
                    Zref=10.0,
                    Uref=10.0,
                    ):
        """
        
        This class implements the ESDU 85 wind profile. The default values are for a open terrain at 30 degrees latitude. The default Z vector is from 0.5m to 300m with 99 points logarithmically spaced and 10m added at the end. The default Zref is 10m. The default Uref is 10m/s.

        Args:
            phi (float, optional): Latitude in degrees. Defaults to 30.0 degrees.
            z0 (float, optional): Roughness length in meters. Defaults to 0.03m.
            Z (np.ndarray, optional): Heights in meters. Defaults to np.sort(np.append(np.logspace(np.log10(0.5),np.log10(300),99),10)).
            Zref (float, optional): Reference height in meters. Defaults to 10.0m
            Uref (float, optional): Reference velocity in m/s. Defaults to 10.0m/s.

        References:
            ESDU 85020
            ESDU 82026
        """
        self.phi = phi # latitude in degrees
        self.z0 = z0
        self.d = d  # zero-plane displacement not implemented
        self.Zref = Zref
        self.Uref = Uref
        if Z is None:
            Z = np.sort(np.append(np.logspace(np.log10(1.01*z0+d),np.log10(300),99),10))
        self.Z = Z
        if not self.Zref in self.Z:
            self.Z = np.sort(np.append(self.Z, self.Zref))
        # remove all Z values that are smaller than z0
        # self.Z = self.Z[self.Z >= self.z0]
        
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
    
    def uw(self,Z=None):
        Z = self.Z if Z is None else Z
        op1 = -1*np.multiply(np.multiply(self.Iu(Z),self.Iv(Z)), np.power(self.U(Z),2) )
        op2 = np.divide(1 - 2*Z/self.h(),
                          np.multiply(np.power(self.sigUbyUstar(Z),2), self.sigWbySigU(Z) ) )
        return np.multiply(op1,op2)     # ESDU 85020, eq. 4.7
    
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
            name = f'ESDU-85 (z0={self.z0:.3g}m)'

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
        stats['uw'] = self.uw()

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
                bldgName='myBuilding', H=None, He=None, Hr=None, B=None, D=None, roofSlope=None, lScl=1, valuesAreScaled=True, 
                faces: List[windCAD.face] = [], faces_file_basic=None, faces_file_derived=None, tapNos: List[int]=[],
                # Inputs for the derived class
                caseName='myBuilding_testCase',
                notes_Cp=" ",
                AoA_zero_deg_basisVector=None,
                AoA_rotation_direction:Literal['CW','CCW']=None,
                refProfile:profile=None,
                Zref_input=None,  # for the Cp TH being input below
                Uref_input=None,  # for the Cp TH being input below
                Uref_FS=None,     # Full-scale reference velocity for scaling purposes
                vScl=None,        # Scaling factor for velocity
                samplingFreq=None,
                fluidDensity=1.0,
                fluidKinematicViscosity=1.48e-5,
                AoA=None,
                CpOfT=None,  # Cp TH referenced to Uref at Zref
                badTaps=None, # tap numbers to remove
                reReferenceCpToH=True, # whether or not to re-reference Cp to building height velocity
                pOfT=None,
                p0ofT=None,
                CpStats=None,
                CpStats_fields: Literal['mean','std','peak','peakMin','peakMax','skewness','kurtosis'] = ['mean','std','peak'],
                peakSpecs=DEFAULT_PEAK_SPECS,
                keepTH=True,
                reScaleProfileToMatchUref=True,
                ):
        """
        This class handles the computation of building wind load by inheriting the windCAD.building 
        class. It also handles the computation of the Cp time history from the pressure time history.

        Args:
            bldgName (str, optional): 
                        Name of the building. Defaults to None.
            H (float, optional): 
                        The reference height of the building. Defaults to None.
            He (float, optional): 
                        Height of the building's eave. Defaults to None.
            Hr (float, optional): 
                        Height of the building's ridge. Defaults to None.
            B (float, optional):
                        Building width. Defaults to None.
            D (float, optional): 
                        Building depth. Defaults to None.
            roofSlope (float, optional): 
                        Roof slope in degrees. Defaults to None.
            lScl (float, optional): 
                        Length scale for the building. Defaults to 1.
            valuesAreScaled (bool, optional): 
                        Whether or not the values are scaled. Defaults to True.
            faces (List[windCAD.face], optional): 
                        List of faces of the building. Defaults to [].
            faces_file_basic (str, optional): 
                        Path to the basic face file. Defaults to None.
            faces_file_derived (str, optional): 
                        Path to the derived face file. Defaults to None.
            caseName (str, optional): 
                        Name of the case. Defaults to None.
            notes_Cp (str, optional): 
                        Notes for the Cp time history. Defaults to " ".
            AoA_zero_deg_basisVector (np.ndarray, optional): 
                        Basis vector for the zero degree AoA. Defaults to None.
            AoA_rotation_direction (Literal['CW','CCW'], optional): 
                        Rotation direction for the AoA. Defaults to None.
            refProfile (profile, optional): 
                        Reference profile for the Cp time history. Defaults to None.
            Zref_input (float, optional): 
                        The reference height used for the incoming Cp time history. 
                        It is used to re-reference the Cp time history to the building 
                        height. The units are in meters. Defaults to None.
            Uref_input (float, optional): 
                        The reference velocity used for the incoming Cp time history. 
                        It is used to re-reference the Cp time history to the building 
                        height. The units are in m/s. Defaults to None.
            Uref_FS (float, optional): 
                        The full-scale reference velocity used for scaling the velocity. 
                        The units are in m/s. Defaults to None.
            samplingFreq (float, optional): 
                        Sampling frequency of the Cp time history. The units are in Hz. 
                        Defaults to None.
            fluidDensity (float, optional): 
                        Fluid density in kg/m3. Defaults to 1.0 assuming normalized pressure.
            AoA (float, optional): 
                        Angle of attack in degrees. Defaults to None.
            CpOfT (np.ndarray, optional): 
                        Pressure coefficient time history defined as
                        Cp = (p-p0)/(0.5*rho*Uref^2) where p0 is the reference static
                        pressure, p is the pressure at point of interest, rho is the
                        fluid density, and Uref is the reference velocity. If both CpOfT
                        and pOfT are provided, then CpOfT is used. Defaults to None.
                        Shape: [N_AoA,Ntaps,Ntime]. 
            badTaps (List[int], optional): 
                        List of tap numbers to remove. Defaults to None.
            reReferenceCpToH (bool, optional): 
                        Whether or not to re-reference the Cp time history to the 
                        building height. Defaults to True.
            pOfT (np.ndarray, optional): 
                        Pressure time history in Pa used in the calculation of CpOfT 
                        (see the definition of 'CpOfT'). If CpOfT is provided, then this
                        is neglected. Defaults to None.
                        Shape: [N_AoA,Ntaps,Ntime].
            p0ofT (np.ndarray, optional): 
                        Reference static pressure time history in Pa used in the 
                        calculation of CpOfT (see the definition of 'CpOfT'). If CpOfT
                        is provided, then this is neglected. For CFD data, this is taken 
                        from a point far upstream (if blockage is very low) or far above
                        the building away from any boundaries. For wind tunnel data that
                        is provided as CpOfT, this is the static reference pressure used
                        by the scanners.
                        Defaults to None.
                        Shape: [N_AoA,Ntime].
            CpStats (dict, optional):
                        Dictionary of pre-calculated Cp statistics. If this is provided,
                        then the Cp time history is not re-calculated. Defaults to None.
                        Shape: dict{nFlds:[N_AoA,Ntaps]}.
            peakSpecs (dict, optional):
                        Dictionary of peak specifications. Defaults to DEFAULT_PEAK_SPECS.
            keepTH (bool, optional):
                        Whether or not to keep the Cp and pressure time histories.
                        Defaults to True.

        Raises:
            Exception: If the p and p0 time series for Cp calculation do not match in 
                        time steps.
            Exception: If the provided Z vector contains values higher than 300m which
                        is beyond the range provided in ESDU 82026, eq. A1.8.
            Exception: Unknown normalization height type. Choose from {'Z', 'xLi'} or
                        specify a number.

        References:
            ESDU 85020
            ESDU 82026

        """
        super().__init__(name=bldgName, H=H, He=He, Hr=Hr, B=B, D=D, roofSlope=roofSlope, lScl=lScl, valuesAreScaled=valuesAreScaled, faces=faces,
                        faces_file_basic=faces_file_basic, faces_file_derived=faces_file_derived, tapNos=tapNos)

        self.name = caseName
        self.notes_Cp = notes_Cp
        self.profile : profile = refProfile
        self.samplingFreq = samplingFreq
        self.fluidDensity = fluidDensity
        self.fluidKinematicViscosity = fluidKinematicViscosity

        self.Zref = Zref_input
        self.AoA = [AoA,] if np.isscalar(AoA) else AoA          # [N_AoA]
        self.Uref = np.array([Uref_input for _ in AoA]) if np.isscalar(Uref_input) else Uref_input
        self.Uref_FS = Uref_FS
        self.badTaps = badTaps
        self.AoA_zero_deg_basisVector = AoA_zero_deg_basisVector
        self.AoA_rotation_direction: Literal['CW','CCW'] = AoA_rotation_direction
        self.CpOfT = CpOfT      # [N_AoA,Ntaps,Ntime]
        self.pOfT = pOfT        # [N_AoA,Ntaps,Ntime]
        self.p0ofT = p0ofT      # [N_AoA,Ntime]
        self.CpStats = CpStats          # dict{nFlds:[N_AoA,Ntaps]}
        self.peakSpecs = peakSpecs
        self.CpStats_fields = CpStats_fields

        self.CpStatsAreaAvg = None      # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]
        self.velRatio = None

        self.__handleBadTaps()
        if reReferenceCpToH:
            self.__reReferenceCp()
        elif self.Zref is None:
            self.Zref = self.H
        
        if vScl is not None:
            self.vScl = vScl
            self.tScl = self.lScl/self.vScl
        elif self.Uref_FS is not None and self.Uref is not None and self.lScl is not None:
            self.vScl = np.mean(self.Uref)/self.Uref_FS
            self.tScl = self.lScl/self.vScl
        else:
            self.vScl = self.tScl = None
        if reScaleProfileToMatchUref:
            self.__reScaleProfileToMatchUref()
        self.Refresh()
        self.__N_t__ = None if self.CpOfT is None else np.shape(self.CpOfT)[-1]
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
        print(f"Computing Cp time history ...")
        p0ofT = 0.0 if self.p0ofT is None else self.p0ofT
        if not np.isscalar(p0ofT) and not np.shape(p0ofT)[-1] == np.shape(self.pOfT)[-1]:
            raise Exception(f"The p and p0 time series for Cp calculation do not match in time steps. Shapes of p0ofT : {np.shape(p0ofT)}, pOfT : {np.shape(self.pOfT)}")
        CpOfT = np.empty(np.shape(self.pOfT))
        print(f"Uref = {self.Uref}")
        print(f"Shape of self.pOfT = {np.shape(self.pOfT)}")
        print(f"Shape of p0ofT = {np.shape(p0ofT)}")
        print(f"Shape of pOfT = {np.shape(CpOfT)}")
        print(f"Shape of self.fluidDensity = {np.shape(self.fluidDensity)}")
        for i, Uref in enumerate(self.Uref):
            CpOfT[i,:,:] = np.divide(np.subtract(self.pOfT[i,:,:],p0ofT),
                                    0.5*self.fluidDensity*Uref**2)
        self.CpOfT = CpOfT 
        #np.divide(np.subtract(self.pOfT,p0ofT),
                                # 0.5*self.fluidDensity*self.Uref**2)
        pass

    def __computeAreaAveragedCp(self):
        if self.NumPanels == 0 or self.CpOfT is None:
            return
        print(f"Computing area-averaging ...")

        axT = len(np.shape(self.CpOfT))-1
        nT = np.shape(self.CpOfT)[-1]
        nAoA = self.NumAoA
        self.CpStatsAreaAvg = [] # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]

        for _, fc in enumerate(self.faces):
            print(f"    Computing area-averaging for face {fc.name} ...")
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
        if self.profile is None or self.Zref is None or self.Uref is None or self.AoA is None:
            return
        if self.CpOfT is None and self.CpStats is None:
            return
        
        vel = self.profile
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

    def __reScaleProfileToMatchUref(self, uref_tol=0.01):
        if self.profile is None or self.Uref is None or self.Zref is None:
            return
        Uref_avg = np.mean(self.Uref)
        U_ref_prof = np.interp(self.Zref, self.profile.Z, self.profile.U)
        if np.abs(Uref_avg - U_ref_prof) < uref_tol:
            return
        if self.profile.UofT is None or self.profile.dt is None or self.profile.VofT is None or self.profile.WofT is None:
            raise Exception(f"Cannot re-scale profile to match Uref of Cp. Profile time history (UofT, VofT, WofT, dt) is not available.")
        
        import inspect

        print(f"Re-scaling profile to match Uref ...")
        prof = self.profile.copy()
        U_ratio = Uref_avg / U_ref_prof
        time_ratio = 1/U_ratio
        print(f"    Uref_avg = {Uref_avg:.3f} m/s")
        print(f"    U_ref_prof = {U_ref_prof:.3f} m/s")
        print(f"    U_ratio = {U_ratio:.3f}")
        print(f"    time_ratio = {time_ratio:.3f}")

        params = inspect.signature(profile.__init__).parameters
        params = list(params.values())[1:]
        args = {}
        for p in params:
            if p.name == 'UofT':
                args[p.name] = prof.UofT.copy()*U_ratio
            elif p.name == 'VofT':
                args[p.name] = prof.VofT.copy()*U_ratio
            elif p.name == 'WofT':
                args[p.name] = prof.WofT.copy()*U_ratio
            elif p.name == 'dt':
                args[p.name] = prof.dt*time_ratio
            elif p.name in ['stats','SpectH',]:
                continue
            elif hasattr(prof, p.name):
                args[p.name] = getattr(prof, p.name)
        self.profile = profile(**args)
    
    def __str__(self):
        return self.name

    """-------------------------------- Properties ------------------------------------"""
    @property
    def dt(self):
        if self.samplingFreq is None:
            return None
        return 1/self.samplingFreq

    @property
    def N_t(self):
        if self.CpOfT is not None:
            return np.shape(self.CpOfT)[-1]
        elif self.pOfT is not None:
            return np.shape(self.pOfT)[-1]
        elif self.p0ofT is not None:
            return np.shape(self.p0ofT)[-1]
        else:
            return self.__N_t__
        
    @property
    def t(self):
        if self.dt is None or self.N_t is None:
            return None
        return np.arange(self.N_t)*self.dt
        
    @property
    def T(self):
        '''Duration of the time series in seconds'''
        if self.dt is None or self.N_t is None:
            return None
        return self.dt * self.N_t

    @property
    def T_star(self):
        '''Normalized duration, T.Uh/H'''
        dur = self.T
        if dur is None or self.Zref is None or self.Uref is None:
            return None
        else:
            return dur * self.Uref/self.Zref

    @property
    def Re(self):
        if self.Uref is None or self.Zref is None:
            return None
        return np.multiply(self.Uref, self.Zref) / self.fluidKinematicViscosity

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

    @property
    def allAreasForCpAavg(self) -> dict:
        areas = {k:np.array([]) for k in self.zoneDictKeys}
        for _,fc in enumerate(self.faces): # self.CpStatsAreaAvg # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                area_z = []
                for a, _ in enumerate(self.NominalPanelArea):
                    area_z.extend(fc.panelAreas[z][a])
                areas[zKey] = np.concatenate((areas[zKey], np.array(area_z)))
        return areas

    @property
    def CpAavg_envMin_peakMin_minPerA(self) -> dict:
        stats = {k:[] for k in self.zoneDictKeys}
        fld = 'peakMin'
        for f,fc in enumerate(self.faces): # self.CpStatsAreaAvg # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                val_z = []
                for a, _ in enumerate(self.NominalPanelArea):
                    val_z.append(np.min(np.array(self.CpStatsAreaAvg[f][z][a][fld]).flatten()))
                if stats[zKey] == []:
                    stats[zKey] = np.array(val_z)
                else:
                    stats[zKey] = np.minimum(stats[zKey], np.array(val_z))
        return stats
    
    @property
    def CpAavg_envMin_peakMin_allA(self) -> dict:
        stats = {k:[] for k in self.zoneDictKeys}
        fld = 'peakMin'
        for f,fc in enumerate(self.faces): # self.CpStatsAreaAvg # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                val_z = []
                for a, _ in enumerate(self.NominalPanelArea):
                    val_z.extend(np.min(np.array(self.CpStatsAreaAvg[f][z][a][fld]), axis=0))
                stats[zKey].extend(val_z)
        return stats
    
    @property
    def CpAavg_envMax_peakMax_maxPerA(self) -> dict:
        stats = {k:[] for k in self.zoneDictKeys}
        fld = 'peakMax'
        for f,fc in enumerate(self.faces):
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                val_z = []
                for a, _ in enumerate(self.NominalPanelArea):
                    val_z.append(np.max(np.array(self.CpStatsAreaAvg[f][z][a][fld]).flatten()))
                if stats[zKey] == []:
                    stats[zKey] = np.array(val_z)
                else:
                    stats[zKey] = np.maximum(stats[zKey], np.array(val_z))
        return stats
    
    @property
    def CpAavg_envMax_peakMax_allA(self) -> dict:
        stats = {k:[] for k in self.zoneDictKeys}
        fld = 'peakMax'
        for f,fc in enumerate(self.faces):
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                val_z = []
                for a, _ in enumerate(self.NominalPanelArea):
                    val_z.extend(np.max(np.array(self.CpStatsAreaAvg[f][z][a][fld]), axis=0))
                stats[zKey].extend(val_z)
        return stats

    """--------------------------------- Methods --------------------------------------"""
    def CpStatsAreaAvgCollected(self, mixNominalAreasFromAllZonesAndFaces=False, 
                        envelope:Literal['max','min','none']='none', 
                        extremesPerNominalArea:Literal['max','min','none']='none'):
        # [Nfaces][Nzones][Narea][Nflds][N_AoA,Npanels]
        zNames = self.zoneDictKeys
        # for z, zn in enumerate(self.zoneDict):
        #     zNames.append(self.zoneDictKeys[zn])
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
            for z,zKey in enumerate(fc.zoneDictKeys):
                zIdx = zNames.index(zKey)
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

    def Refresh(self):
        print(f"Refreshing {self.name}...")
        print(f"Verifying data ...")
        self.__verifyData()
        self.__computeCpTHfrom_p_TH()
        if self.CpOfT is not None:
            print(f"Computing Cp statistics ...")
            self.CpStats = get_CpTH_stats(self.CpOfT,axis=len(np.shape(self.CpOfT))-1,peakSpecs=self.peakSpecs, fields=self.CpStats_fields)
        self.__computeAreaAveragedCp()
        print(f"Done refreshing {self.name}.\n")
    
    def write(self):
        pass

    def checkStatField(self,fld):
        if self.CpStats is None:
            raise Exception(f"CpStats is not defined.")
        if fld in self.CpStats:
            return True
        else:
            raise Exception(f"The field {fld} is not a part of the available stat fields. Available stat fields: {list(self.CpStats.keys())}")
    
    def CandCLoad_factor(self, format:Literal['NBCC','ASCE']='ASCE', debugMode=False,):
        if self.T is None or self.tScl is None or self.H is None or self.profile is None:
            print(f"Cannot compute C&C Load factor. The following are required: T, tScl, H, profile")
            return None
        
        T = self.T
        full_scale_duration = T/self.tScl if np.isscalar(T) else np.mean(T)/self.tScl
        z0_MS = self.profile.z0_Iu
        z0 = z0_MS / self.lScl
        if debugMode:
            print(f"Computing C&C Load factor ...")
            print(f"full_scale_duration = {full_scale_duration}")
            print(f"z0 = {z0_MS:.3g}m (@MS), {z0:.3g}m (@FS)\n")

        factor = wc.CpConversionFactor(from_='simulated', to_=format, from_Z=self.H/self.lScl, from_gustDuration=full_scale_duration, from_z0=z0, debugMode=debugMode)
        if debugMode:
            print(f"C&C factor for case {self.name} = {factor}")
        return factor

    def writeCandCloadToXLSX(self, filePath, 
                            extremesPerNominalArea=True, areaFactor=1.0, 
                            format:Literal['default','NBCC','ASCE']='default', debugMode=False,
                            ):
        if self.T is None or self.tScl is None or self.H is None or self.profile is None:
            print(f"Cannot compute C&C Load factor. The following are required: T, tScl, H, profile")
            return None

        zoneDictKeys = self.zoneDictKeys
        area = {z: np.array(self.NominalPanelArea) for z in zoneDictKeys} if extremesPerNominalArea else self.allAreasForCpAavg
        if extremesPerNominalArea:
            peakMax = self.CpAavg_envMax_peakMax_maxPerA
            peakMin = self.CpAavg_envMin_peakMin_minPerA
        else:
            peakMax = self.CpAavg_envMax_peakMax_allA
            peakMin = self.CpAavg_envMin_peakMin_allA

        if format == 'default':
            valueScaleFactor = 1.0
        elif format == 'NBCC':
            valueScaleFactor = self.CandCLoad_factor(debugMode=debugMode, format='NBCC')
        elif format == 'ASCE':
            valueScaleFactor = self.CandCLoad_factor(debugMode=debugMode, format='ASCE')
        else:
            raise Exception(f"Unknown CandCLoadFormat: {format}")
        
        with pd.ExcelWriter(filePath) as writer:
            for I, zKey in enumerate(zoneDictKeys):
                df = pd.DataFrame({'Area [m^2]':area[zKey]*areaFactor, 
                                   'Peak Max':np.array(peakMax[zKey])*valueScaleFactor, 
                                   'Peak Min':np.array(peakMin[zKey])*valueScaleFactor})
                df.to_excel(writer, sheet_name=zKey, index=False)
        if debugMode:
            print(f"Saved C&C load to {filePath}")

    def paramsTable(self, normalized=True, fields=None) -> dict:
        
        data = {}
        if self.profile is not None:
            data = self.profile.paramsTable(normalized=normalized)
        Re = self.Re
        if Re is not None and not np.isscalar(Re):
            Re = [r for r in Re if r is not None]
            Re = np.mean(Re) if len(Re) > 0 else None
        data[mathName('Re')] = np.nan if Re is None else Re
        data['design '+mathName('Uh')+' @FS'] = np.nan if self.Uref_FS is None else self.Uref_FS
        data[mathName('lScl')] = np.nan if self.lScl is None else 1/self.lScl
        data[mathName('vScl')] = np.nan if self.vScl is None else 1/self.vScl
        data[mathName('tScl')] = np.nan if self.tScl is None else 1/self.tScl

        return data

    """--------------------------------- Plotters -------------------------------------"""
    def plotTapCpStatsPerAoA(self, figs=None, all_axes=None, addMarginDetails=False,
                            fields=['peakMin','mean','peakMax',],fldRange=[-15,10], tapsToPlot=None, includeTapName=True,
                            nCols=7, nRows=10, 
                            cols = ['r','k','b','g','m','r','k','b','g','m'],
                            mrkrs = ['v','o','^','s','p','d','.','*','<','>','h'], 
                            ls=['-','-','-','-','-','-','-','-','-','-',],
                            mrkrSize=2,
                            kwargs_perFld=None, 
                            simpleLabel=True,
                            xticks=None, nAoA_ticks=5, xlim=None, 
                            legend_bbox_to_anchor=(0.5, 0.905), pageNo_xy=(0.5,0.1), figsize=[15,20], sharex=True, sharey=True,
                            overlayThis=None, overlay_AoA=None, overlayLabel=None, kwargs_overlay=None):
        if kwargs_perFld is None:
            kwargs_perFld = [{'color':cols[i], 
                              'marker':mrkrs[i], 
                              'ls':ls[i],
                              'markersize':mrkrSize,
                              } for i in range(len(fields))]
        kwargs_overlay = [{} for _ in range(len(fields))] if kwargs_overlay is None else kwargs_overlay

        tapIdxs = self.tapIdx if tapsToPlot is None else self.tapIdxOf(tapsToPlot)
        
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
            addMarginDetails = True
            figs = []
            all_axes = []
        for p in range(nPages):
            if newFig:
                fig, axs = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex, sharey=sharey)
                all_axes.append(axs)
                figs.append(fig)
            else:
                axs = all_axes[p]
                fig = figs[p]
                
            for i in range(nPltPerPage):
                if tapPltCount >= nPltTotal:
                    break
                ax = axs[i//nCols,i%nCols]
                for f,fld in enumerate(fields):
                    label = fld if simpleLabel else self.name+' ('+fld+')'
                    ax.plot(self.AoA, self.CpStats[fld][:,tapIdx], label=label, **kwargs_perFld[f])
                    if overlayThis is not None:
                        ax.plot(overlay_AoA, overlayThis[fld][:,tapPltCount], label=overlayLabel+' ('+fld+')', **kwargs_overlay[f])
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
                if xlim is not None:
                    min_AoA, max_AoA = xlim
                else:
                    min_AoA, max_AoA = np.floor(np.min(self.AoA)/5)*5, np.ceil(np.max(self.AoA)/5)*5
                xticks = np.unique(np.floor(np.linspace(min_AoA, max_AoA, nAoA_ticks)/5)*5) if xticks is None else xticks
                ax.xaxis.set_ticks(xticks)
                ax.set_xlim([xticks[0],xticks[-1]])
                ax.tick_params(axis=u'both', which=u'both',direction='in')
                
                # ax.grid(which='both')
                if addMarginDetails:
                    formatAxis(ax)
                
                if i//nCols == nRows-1:
                    ax.set_xlabel(r'AoA')
                if i%nCols == 0:
                    ax.set_ylabel(r'$C_p$')
                tapPltCount += 1
                if tapPltCount < len(tapIdxs):
                    tapIdx = tapIdxs[tapPltCount]

            handles, labels = ax.get_legend_handles_labels()
            if newFig or addMarginDetails:
                pass
                
            if newFig:
                fig.legend(handles, fields, loc='upper center',ncol=len(fields), bbox_to_anchor=legend_bbox_to_anchor, bbox_transform=fig.transFigure)
                text = f"Page {p+1} of {nPages}\n Case: {self.name}"
                plt.annotate(text, xy=pageNo_xy, xycoords='figure fraction', ha='center', va='top')
                plt.show()
        if newFig:
            return figs, all_axes
        else:
            return None, None

    def plotTapCpStatContour(self, fieldName:Literal['mean','std','peak','peakMin','peakMax','skewness','kurtosis'], 
                             dxnIdx=None, envelopeType:Literal['high','low','both']='both', figSize=[15,10], ax=None, 
                            fldRange=None, nLvl=100, cmap='RdBu', extend='both', title=None, colBarOrientation='horizontal',
                            showValuesOnContour=True, kwargs_contourTxt={'inline':True, 'fmt':'%.2g','fontsize':4, 'colors':'k',},
                            showContourEdge=True, kwargs_contourEdge={'colors':'k', 'linewidths':0.3, 'linestyles':'solid', 'alpha':0.5},
                            ):
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
        cObj = self.plotTapField(ax=ax, field=data, fldRange=fldRange, nLvl=nLvl, cmap=cmap, extend=extend,
                                 showValuesOnContour=showValuesOnContour, kwargs_contourTxt=kwargs_contourTxt,
                                 showContourEdge=showContourEdge, kwargs_contourEdge=kwargs_contourEdge)
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

    def plotAreaAveragedStat___depricated(self, fig=None, axs=None, figSize=[15,10], 
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
        zoneDictKeys = self.zoneDictKeys
        pnlA_all = self.panelAreas__to_be_redacted
        pnlA_nom = np.array(self.NominalPanelArea)
        if plotExtremesPerNominalArea:
            areaAvgStats_min = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='min',extremesPerNominalArea='min')
            areaAvgStats_max = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='max',extremesPerNominalArea='max')
        else:
            areaAvgStats_min = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='min',extremesPerNominalArea='none')
            areaAvgStats_max = self.CpStatsAreaAvgCollected(mixNominalAreasFromAllZonesAndFaces=True,envelope='max',extremesPerNominalArea='none')

        if plotZoneGeom:
            zoneCol = {z: 'w' for z in zoneDictKeys}

        for I, z in enumerate(zoneDict):
            zoneName = zoneDictKeys[I]
            print(f"Plotting zone {zoneName} ({I+1} of {len(zoneDict)})")
            area = pnlA_nom*areaFactor if plotExtremesPerNominalArea else np.array(pnlA_all[z][2])*areaFactor
            print(f"Shape of area: {np.shape(area)}")
            maxVal = np.squeeze(areaAvgStats_max[z][2]['peakMax'])
            minVal = np.squeeze(areaAvgStats_min[z][2]['peakMin'])
            print(f"Shape of maxVal: {np.shape(maxVal)}")
            print(f"Shape of minVal: {np.shape(minVal)}")
            
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
                zoneCol_copy[zoneName] = zoneShadeColor
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

    def plotCandC_load(self, fig=None, axs=None,
                    figSize=[15,10], sharex=False, sharey=False,
                    plotExtremesPerNominalArea=True, nCols=3, areaFactor=1.0, CandCLoadFormat:Literal['default','NBCC','ASCE']='default', 
                    invertYAxis=False,
                    label_min='Min', label_max='Max',
                    overlayThese=None, overlayFactors=None, kwargs_overlay={'color':'k', 'linewidth':2, 'linestyle':'-'},
                    subplotLabels=None, subplotLabels_xy=[0.05,0.95], kwargs_subplotLabels={'fontsize':14},
                    legend_ax_idx=0,
                    debugMode=False,
                    plotZoneGeom=True, insetBounds:Union[list,dict]=[0.6, 0.0, 0.4, 0.4], zoneShadeColor='k', kwargs_zonePlots={},
                    xLimits=None, yLimits=None, xLabel=None, yLabel=None,
                    kwargs_min={}, kwargs_max={}, kwargs_legend={},
                    kwargs_ax={'gridMajor':True, 'gridMinor':True}):
        def addOverlay(ax, ar, val, overlayType, name='_', kwargs_overlay={}):
            if overlayThese is None:
                return
            if overlayType == 'single':
                ax.semilogx(ar, val, 
                        label=name, **kwargs_overlay)
            elif overlayType == 'errorBars':
                if kwargs_overlay == {}:
                    kwargs_overlay = {
                        'widths': 0.1,
                        'notch': True,
                        'vert': True,
                        'showfliers': False,
                        'patch_artist': True,
                        'meanline': True,
                        'boxprops': dict(facecolor='None', color='b', linewidth=1, edgecolor='b'),
                        'medianprops': dict(color='b', linewidth=1),
                        'whiskerprops': dict(color='b', linewidth=1),
                        'capprops': dict(color='b', linewidth=1.5,),
                        'whis': [5,95],
                        }
                yTicks = ax.get_yticks()
                hadYLabel = ax.get_ylabel() != ''
                bx = ax.boxplot(val, positions=ar, 
                        **kwargs_overlay)
                ax.set_yticks(yTicks)
                if hadYLabel:
                    ax.set_yticklabels(['{:g}'.format(y) for y in yTicks])
                if name != '_':
                    bx['boxes'][0].set_label(name)
                    return bx #['boxes'][0]
            else:
                raise NotImplementedError("Overlay type '{}' not implemented".format(overlayType))
            return None

        newFig = False
        if fig is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            plt.tight_layout()

            NumZones = self.NumZones
            nCols = min(nCols, NumZones)
            nRows = int(np.ceil(NumZones/nCols))
            axs = fig.subplots(nRows, nCols, sharex=sharex, sharey=sharey)

        zoneDictKeys = self.zoneDictKeys
        area = {z: np.array(self.NominalPanelArea) for z in zoneDictKeys} if plotExtremesPerNominalArea else self.allAreasForCpAavg
        if plotExtremesPerNominalArea:
            peakMax = self.CpAavg_envMax_peakMax_maxPerA
            peakMin = self.CpAavg_envMin_peakMin_minPerA
        else:
            peakMax = self.CpAavg_envMax_peakMax_allA
            peakMin = self.CpAavg_envMin_peakMin_allA

        if plotZoneGeom:
            zoneCol = {z: 'w' for z in zoneDictKeys}

        valueScaleFactor = 1.0 if CandCLoadFormat == 'default' else self.CandCLoad_factor(debugMode=debugMode, format=CandCLoadFormat)
        
        bxPltObj = None
        bxPltName = None
        for I, zKey in enumerate(zoneDictKeys):
            zoneName = zKey
            
            i, j = np.unravel_index(I, axs.shape)
            ax = axs[i,j]
            ax.semilogx(area[zKey]*areaFactor, np.array(peakMax[zKey])*valueScaleFactor, '^b', label=label_max, **kwargs_max)
            ax.semilogx(area[zKey]*areaFactor, np.array(peakMin[zKey])*valueScaleFactor, 'vk', label=label_min, **kwargs_min)
            ax.axhline(0, color='k', linestyle='-', linewidth=0.7)
            if overlayThese is not None:
                for ii, overlayThis in enumerate(overlayThese):
                    overlayFactor = overlayFactors[ii] if overlayFactors is not None else 1.0
                    ar = overlayThis['Min']['area'][zKey]
                    valMin = np.array(overlayThis['Min']['value'][zKey])*overlayFactor
                    valMax = np.array(overlayThis['Max']['value'][zKey])*overlayFactor
                    if np.shape(ar) == np.shape(valMin):
                        ax.semilogx(ar, valMin, label=overlayThis['Name'], **kwargs_overlay[ii])
                        ax.semilogx(ar, valMax, **kwargs_overlay[ii])
                    else:
                        dim = 0 if np.shape(ar)[0] == np.shape(valMin)[0] else 1
                        if dim == 1:
                            valMin = valMin.T
                            valMax = valMax.T
                        for a in range(np.shape(valMin)[dim]):
                            overlayLabel = overlayThis['Name'] if a == 0 else None
                            ax.semilogx(ar, valMin[:,a], label=overlayLabel, **kwargs_overlay[ii])
                            ax.semilogx(ar, valMax[:,a], **kwargs_overlay[ii])
                        bxPltName = overlayThis['Name']
                        bxPltObj = addOverlay(ax, [ar[0],], np.array(valMin).flatten(), 'errorBars', name=bxPltName, kwargs_overlay={})
                        addOverlay(ax, [ar[0],], np.array(valMax).flatten(), 'errorBars', name='_', kwargs_overlay={})

            if invertYAxis:
                ax.invert_yaxis()
            if plotZoneGeom:
                zoneCol_copy = zoneCol.copy()
                zoneCol_copy[zKey] = zoneShadeColor
                bounds = insetBounds # if insetBounds is isinstance(insetBounds, list) else insetBounds[z]
                ax_inset = ax.inset_axes(bounds=bounds)
                self.plotZones(ax=ax_inset, zoneCol=zoneCol_copy, showLegend=False, drawEdge=True, **kwargs_zonePlots)
                # ax_inset.axis('off')
                ax_inset.axis('equal')
                ax_inset.patch.set_facecolor('w')
                ax_inset.patch.set_alpha(0.7)
                ax_inset.tick_params(axis='both', which='major', length=0)
                ax_inset.set_xticklabels('')
                ax_inset.set_yticklabels('')
                for spine in ax_inset.spines.values():
                    spine.set_visible(False)

            if subplotLabels is not None:
                ax.annotate(subplotLabels[I], xy=subplotLabels_xy, xycoords='axes fraction', **kwargs_subplotLabels)

            if newFig:
                ax.set_title(zoneName)
                if i == axs.shape[0]-1:
                    xLabel = r'Area [$m^2$]' if xLabel is None else xLabel
                    ax.set_xlabel(xLabel)
                if j == 0:
                    yLabel = r'Peak Cp' if yLabel is None else yLabel
                    ax.set_ylabel(yLabel)
                if xLimits is not None:
                    ax.set_xlim(xLimits)
                if yLimits is not None:
                    ax.set_ylim(yLimits)

                formatAxis(ax, **kwargs_ax)

        if bxPltObj is None:
            handles, labels = axs[0,0].get_legend_handles_labels()
            axs[np.unravel_index(legend_ax_idx, axs.shape)].legend(handles, labels, **kwargs_legend)
        else:
            hndls, lbls = axs[0,0].get_legend_handles_labels()
            hndls.append(bxPltObj["boxes"][0])
            lbls.append(bxPltName)
            axs[np.unravel_index(legend_ax_idx, axs.shape)].legend(handles=hndls, labels=lbls, **kwargs_legend)
        
        # if there are remaining axes, remove them
        for I in range(len(zoneDictKeys), axs.size):
            i, j = np.unravel_index(I, axs.shape)
            axs[i,j].axis('off')

        if newFig:
            plt.show()
        return fig, axs
        
    def plotAoA_symbol(self, AoA, ax=None, figSize=[6,6], location: Literal['upper left','upper right','lower left','lower right']='lower left',
                       explicitLocation=None, marginFactor=1.0,
                       size=1.0, inwardArrow=True, drawDicorations=True):
        if self.AoA_zero_deg_basisVector is None or self.AoA_rotation_direction is None:
            raise Exception("AoA_zero_deg_basisVector and AoA_rotation_direction must be defined to plot the AoA symbol.")
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            ax = fig.add_subplot()
        
        bounds = self.boundingBoxPlt
        aoaZero = self.AoA_zero_deg_basisVector
        aoaRotDxn = self.AoA_rotation_direction
        xRange, yRange = bounds[1]-bounds[0], bounds[3]-bounds[2]
        basicSize = 0.05*np.mean([xRange, yRange])
        if np.isscalar(marginFactor):
            margin_x = marginFactor*basicSize
            margin_y = marginFactor*basicSize
        else:
            margin_x = marginFactor[0]*basicSize
            margin_y = marginFactor[1]*basicSize

        if explicitLocation is None:
            if location == 'upper left':
                xOrig = bounds[0] + basicSize + margin_x
                yOrig = bounds[3] - basicSize - margin_y
            elif location == 'upper right':
                xOrig = bounds[1] - basicSize - margin_x
                yOrig = bounds[3] - basicSize - margin_y
            elif location == 'lower left':
                xOrig = bounds[0] + basicSize + margin_x
                yOrig = bounds[2] + basicSize + margin_y
            elif location == 'lower right':
                xOrig = bounds[1] - basicSize - margin_x
                yOrig = bounds[2] + basicSize + margin_y
            else:
                raise Exception(f"Unknown location {location}")
        else:
            xOrig, yOrig = explicitLocation[0], explicitLocation[1]

        if drawDicorations:
            ax.plot(xOrig, yOrig, marker='+', color='k', markersize=30*size)
        
        xZero = [xOrig, xOrig + aoaZero[0]*basicSize*size]
        yZero = [yOrig, yOrig + aoaZero[1]*basicSize*size]
        if drawDicorations:
            ax.plot(xZero, yZero, color='k', linewidth=0.5)

        zeroAngle = np.rad2deg(np.arctan2(aoaZero[1], aoaZero[0]))
        original_aoa = AoA
        rotnDxn = 1 if aoaRotDxn == 'ccw' else -1
        AoA = rotnDxn*AoA + zeroAngle

        r_x = np.cos(np.deg2rad(AoA))*basicSize*size
        r_y = np.sin(np.deg2rad(AoA))*basicSize*size
        r_x_txt, r_y_txt = r_x, r_y

        if inwardArrow:
            arr_orig_x, arr_orig_y = xOrig+r_x*1.8, yOrig+r_y*1.8
            r_x *= -1
            r_y *= -1
        else:
            arr_orig_x, arr_orig_y = xOrig, yOrig

        ax.arrow(arr_orig_x, arr_orig_y, r_x, r_y,
                    head_width=0.3*basicSize*size, head_length=0.6*basicSize*size, fc='k', ec='k')
        ax.text(xOrig+r_x_txt*2.2, yOrig+r_y_txt*2.2,
                f"{original_aoa}\u00b0", fontsize=12*size, ha='center', va='center')

        if drawDicorations:
            arc_orig_x, arc_orig_y = xOrig, yOrig
            arc_radius = 1.0*basicSize*size
            arc_start_angle = zeroAngle
            arc_end_angle =   zeroAngle + rotnDxn*original_aoa
            ax.add_patch(Arc((arc_orig_x, arc_orig_y), arc_radius*2, arc_radius*2, 
                            theta1=arc_start_angle, theta2=arc_end_angle, 
                            linewidth=0.5, zorder=0))
        
        if newFig:
            ax.axis('equal')
            plt.show()
            return fig, ax

    def plotAoA_definition(self, AoAs=[0,90,180,270], ax=None, figSize=[6,6], location: Literal['upper left','upper right','lower left','lower right']='lower left',
                       explicitLocation=None, marginFactor=1.0, size=1.0, inwardArrow=True):
        if self.AoA_zero_deg_basisVector is None or self.AoA_rotation_direction is None:
            raise Exception("AoA_zero_deg_basisVector and AoA_rotation_direction must be defined to plot the AoA symbol.")
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure(figsize=figSize)
            ax = fig.add_subplot()

        for i, aoa in enumerate(AoAs):
            if i == 0:
                self.plotAoA_symbol(aoa, ax=ax, figSize=figSize, location=location, explicitLocation=explicitLocation, marginFactor=marginFactor, size=size, inwardArrow=inwardArrow)
            else:
                self.plotAoA_symbol(aoa, ax=ax, figSize=figSize, location=location, explicitLocation=explicitLocation, marginFactor=marginFactor, size=size, inwardArrow=inwardArrow, drawDicorations=False)

        if newFig:
            ax.axis('equal')
            plt.show()
            return fig, ax

#------------------------------- COLLECTIONS -----------------------------------
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

    def __getitem__(self, key):
        return self.profiles[key]
    
    def __setitem__(self, key, value):
        self.profiles[key] = value

    def __len__(self):
        return self.__numOfProfiles()
    
    def __str__(self):
        return str(self.profiles)
    
    @property
    def N(self):
        return len(self.profiles)

    def copy(self):
        return copy.deepcopy(self)

    def paramsTable(self, normalized=True, fields=None) -> dict:
        params = None
        for i, prof in enumerate(self.profiles):
            if i == 0:
                params = prof.paramsTable(normalized=normalized, fields=fields)
                for key in params.keys():
                    params[key] = [params[key],]
            else:
                temp = prof.paramsTable(normalized=normalized, fields=fields)
                # append each entry to the list
                for key in temp.keys():
                    if key in params.keys():
                        params[key].append(temp[key])
                    else:
                        params[key] = [None,]*i + [temp[key],]
                # loop through the params keys looking for those that are not in temp
                for key in params.keys():
                    if key not in temp.keys():
                        params[key].append(None)

        return params

    def plot(self, figsize=None, landscape=True, 
             kwargs_profile=None, 
             kwargs_spect=None):
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

        kwargs_profile = [{} if kwargs_profile is None else kwargs_profile[i] for i in range(self.N)]
        kwargs_spect = [{} if kwargs_spect is None else kwargs_spect[i] for i in range(self.N)]

        for i, prof in enumerate(self.profiles):
            if i == 0:
                kwargs_profile[i]['overlay_H'] = True
            else:
                kwargs_profile[i]['overlay_H'] = False
            prof.plot(fig=fig, ax_U=ax_U, ax_Iu=ax_Iu, ax_Spect=ax_Spect, kwargs_profile=kwargs_profile[i], kwargs_spect=kwargs_spect[i])

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
            ax.loglog(S.rf(), S.rSuu(normU='U'),
                       ls=ls_Spect_i, marker=mrkr_Spect_i, color=col_i, label=prof.SpectH.name, alpha=alpha_S_i)
        ax.set_xlabel(r"$nH/U_H$",fontsize=fontSz_axLbl)
        ax.set_ylabel(r"$nS_{uu}/U_H^2$",fontsize=fontSz_axLbl)
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
    
    def plotProfile_basic2(self, figsize=[12,12], label=None, hspace=0.3, wspace=0.3,
                           normalize=True,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None,
                            overlayThese:dict=None, overlayType:Literal['single','scatter','errorBars']='single', kwargs_overlay={}, 
                            yLimits=None, lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt=None, kwargs_ax={}):
        
        fig, axs = plt.subplots(3,3)
        fig.set_size_inches(figsize)
        # relax the space between subplots
        fig.subplots_adjust(hspace=hspace, wspace=wspace)

        kwargs_plt = [{} if kwargs_plt is None else kwargs_plt[i] for i in range(self.N)]

        if yLimits is None:
            
            allYlims = [] 
            for i, prof in enumerate(self.profiles):
                if prof.workSect_zLim is None:
                    continue
                else:
                    normFactor = 1/prof.H if normalize else 1
                    yl = [prof.workSect_zLim[0]*normFactor, prof.workSect_zLim[1]*normFactor] if prof.workSect_zLim is not None else None
                    allYlims.append(yl)
            if len(allYlims) == 0:
                yLimits = None
            else:
                yLimits = [min([y[0] for y in allYlims]), max([y[1] for y in allYlims])]

        for i, prof in enumerate(self.profiles):
            if i == 0 and overlayThese is not None:
                _,_,bxPltObj = prof.plotProfile_basic2(fig=fig, axs=axs, label=prof.name, normalize=normalize, xLabel=xLabel, zLabel=zLabel, xLimits_U=xLimits_U, 
                                                    xLimits_Iu=xLimits_Iu, xLimits_Iv=xLimits_Iv, xLimits_Iw=xLimits_Iw, 
                                                    xLimits_xLu=xLimits_xLu, xLimits_xLv=xLimits_xLv, xLimits_xLw=xLimits_xLw, xLimits_uw=xLimits_uw, 
                                                    overlayThese=overlayThese, overlayType=overlayType, kwargs_overlay=kwargs_overlay,
                                                    yLimits=yLimits, kwargs_plt=kwargs_plt[i], kwargs_ax=kwargs_ax)
            else:
                _,_,_ = prof.plotProfile_basic2(fig=fig, axs=axs, label=prof.name, normalize=normalize, xLabel=xLabel, zLabel=zLabel, xLimits_U=xLimits_U, xLimits_Iu=xLimits_Iu, 
                                        xLimits_Iv=xLimits_Iv, xLimits_Iw=xLimits_Iw, xLimits_xLu=xLimits_xLu, xLimits_xLv=xLimits_xLv, xLimits_xLw=xLimits_xLw, 
                                        xLimits_uw=xLimits_uw, 
                                        yLimits=yLimits, kwargs_plt=kwargs_plt[i], kwargs_ax=kwargs_ax)
                bxPltObj = None if i == 0 else bxPltObj
        
        axs[0,2].axis('off')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if bxPltObj is None:
                axs[0,2].legend(handles=axs[0,0].get_legend_handles_labels()[0],
                                labels=axs[0,0].get_legend_handles_labels()[1],
                                **lgnd_kwargs)
            else:
                hndls, lbls = axs[0, 0].get_legend_handles_labels()

                # Add the boxplot handle to the legend handles
                hndls.append(bxPltObj["boxes"][0])

                # Add the label for the boxplot handle
                lbls.append(overlayThese['name'])

                axs[0, 2].legend(handles=hndls, labels=lbls, **lgnd_kwargs)
        for ax in axs.flatten():
            formatAxis(ax, **kwargs_ax)
        # plt.show()
        return fig, axs

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
                    figsize=[15,5], 
                    normalize=True,
                    normZ:Literal['Z','xLi']='Z',
                    normU:Literal['U','sigUi']='U',
                    smoothFactor=1,
                    kwargs_smooth={},
                    plotType='loglog',
                    xLimits='auto', # [nMin,nMax]
                    yLimits='auto', # ([SuuMin, SuuMax], [SvvMin, SvvMax], [SwwMin, SwwMax])
                    overlayVonK=False, # Either one entry or array equal to N
                    kwargs_plt=None, # kwargs for the plot
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
        if np.isscalar(smoothFactor):
            smoothFactor = (smoothFactor,)*self.N
        for i in range(self.N):
            if overlayVonK[i]:
                n += (self.profiles[i].SpectH.rf(normZ=normZ),) if normalize else (self.profiles[i].SpectH.n,)
                if smoothFactor[i] > 1:
                    Suu += (self.profiles[i].SpectH.Suu_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                    Svv += (self.profiles[i].SpectH.Svv_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                    Sww += (self.profiles[i].SpectH.Sww_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU),)
                else:
                    Suu += (smooth(self.profiles[i].SpectH.Suu_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU), window_len=smoothFactor[i], **kwargs_smooth), )
                    Svv += (smooth(self.profiles[i].SpectH.Svv_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU), window_len=smoothFactor[i], **kwargs_smooth), )
                    Sww += (smooth(self.profiles[i].SpectH.Sww_vonK(self.profiles[i].SpectH.n,normalized=normalize,normU=normU), window_len=smoothFactor[i], **kwargs_smooth), )
                names += (self.profiles[i].SpectH.name+'-vonK',)
        Nplts = len(n)
        kwargs_plt = [{} if kwargs_plt is None else kwargs_plt[i] for i in range(Nplts)]

        fig = wplt.plotSpectra(
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
                        figSize=figsize,
                        plotType=plotType,
                        drawXlineAt_rf1=drawXlineAt_rf1,
                        kwargs_plt=kwargs_plt
                        )
        return fig

    def plotSpect(self, 
                fig=None, ax_Suu=None, ax_Svv=None, ax_Sww=None, figsize=[15,4], label=None, 
                xLabel=None, yLabel_Suu=None, yLabel_Svv=None, yLabel_Sww=None,
                xLimits=None, yLimits=None, 
                normalize=True, normZ:Literal['Z','xLi']='Z', normU:Literal['U','sigUi']='U',         
                plotType: Literal['loglog', 'semilogx', 'semilogy']='loglog', avoidZeroFreq=True, 
                overlayThese:dict=None, overlayType:Literal['single','scatter','errorBars']='single', kwargs_overlay={}, kwargs_overlay_all={},
                lgnd_kwargs={'loc': 'best', 'ncol': 1},
                kwargs_plt=None, kwargs_ax={}):
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(figsize)

        ax_Suu = axs[0] if ax_Suu is None else ax_Suu
        ax_Svv = axs[1] if ax_Svv is None else ax_Svv
        ax_Sww = axs[2] if ax_Sww is None else ax_Sww

        kwargs_plt = [{} if kwargs_plt is None else kwargs_plt[i] for i in range(self.N)]

        for i, prof in enumerate(self.profiles):
            if i == 0 and overlayThese is not None:
                _ = prof.SpectH.plot(fig=fig, ax_Suu=ax_Suu, ax_Svv=ax_Svv, ax_Sww=ax_Sww, figsize=figsize, label=prof.SpectH.name,
                                        xLabel=xLabel, yLabel_Suu=yLabel_Suu, yLabel_Svv=yLabel_Svv, yLabel_Sww=yLabel_Sww,
                                        xLimits=xLimits, yLimits=yLimits,
                                        normalize=normalize, normZ=normZ, normU=normU,
                                        plotType=plotType, avoidZeroFreq=avoidZeroFreq,
                                        overlayThese=overlayThese, overlayType=overlayType, kwargs_overlay=kwargs_overlay, kwargs_overlay_all=kwargs_overlay_all,
                                        kwargs_plt=kwargs_plt[i], kwargs_ax=kwargs_ax)
            else:
                _ = prof.SpectH.plot(fig=fig, ax_Suu=ax_Suu, ax_Svv=ax_Svv, ax_Sww=ax_Sww, figsize=figsize, label=prof.SpectH.name,
                                        xLabel=xLabel, yLabel_Suu=yLabel_Suu, yLabel_Svv=yLabel_Svv, yLabel_Sww=yLabel_Sww,
                                        xLimits=xLimits, yLimits=yLimits,
                                        normalize=normalize, normZ=normZ, normU=normU,
                                        plotType=plotType, avoidZeroFreq=avoidZeroFreq,
                                        kwargs_plt=kwargs_plt[i], kwargs_ax=kwargs_ax)
        
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=UserWarning)
        ax_Suu.legend(handles=ax_Suu.get_legend_handles_labels()[0],
                        labels=ax_Suu.get_legend_handles_labels()[1],
                        **lgnd_kwargs)
        
        for ax in axs.flatten():
            formatAxis(ax, **kwargs_ax, numFormat='default')
        # plt.show()
        return fig, axs

    def plotRefHeightStatsTable(self,
                                fig=None,figsize=[0,5],autoFigSize=True,figWidthFctr=1.0,ax=None, strFmt='{:.4g}', 
                                fontSz=None, normalize=True, colColors=None, colTxtColors=None, 
                                showBorder=False, 
                                kwargs_table={'loc':'center',
                                              'cellLoc': 'center',
                                              'bbox': [0.0, 0.0, 1.0, 1.0],}):
        if fig is None:
            fntFctr = 1.0 if fontSz is None else fontSz/10.0
            if autoFigSize:
                figsize[0] = 2*(self.N+1)*fntFctr*figWidthFctr
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.subplot()
        ax.axis('off')
        ax.axis('tight')

        # each profile will have its own column and each field will have its own row. If there is missing value or non-existent field, it will be filled with ' '
        cell_text = []
        for key in self.profiles[0].stats_core.keys():
            if normalize:
                # val = [np.round(prof.stat_norm(key)[prof.H_idx], precision) if key in prof.stats.keys() else ' ' for prof in self.profiles]
                val = []
                for prof in self.profiles:
                    if key in prof.stats_core.keys():
                        val_norm, key_name = prof.stat_norm(key)
                        if val_norm is None:
                            val.append(' ')
                        else:
                            val.append(strFmt.format(val_norm[prof.H_idx]))
                    else:
                        val.append(' ')
            else:
                key_name = key
                # val = [np.round(prof.stats[key][prof.H_idx], decimals) if key in prof.stats.keys() else ' ' for prof in self.profiles]
                val = [strFmt.format(prof.stats_core[key][prof.H_idx]) if key in prof.stats_core.keys() else ' ' for prof in self.profiles]
            val.insert(0,key_name)
            cell_text.append(val)

        table = ax.table(cellText=cell_text,
                    colLabels=['Field',*[prof.name for prof in self.profiles]],
                    **kwargs_table)
        if fontSz is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(fontSz)

        # decorative edits
        nRows = len(cell_text)+1
        if colColors is not None:
            for i in range(self.N):
                for j in range(nRows):
                    table[(j,i+1)].set_facecolor(colColors[i])
        if colTxtColors is not None:
            for i in range(self.N):
                for j in range(nRows):
                    table[(j,i+1)].set_text_props(color=colTxtColors[i])
        for i in range(self.N+1):
            table[(0,i)].set_text_props(weight='bold')
        for i in range(nRows):
            table[(i,0)].set_text_props(weight='bold')
        if not showBorder:
            for key, cell in table.get_celld().items():
                cell.set_linewidth(0)

        fig.tight_layout()
        plt.show()
        return fig, ax

    def plotParamsTable(self,
                        fig=None,figsize=[0,7],autoFigSize=True,figWidthFctr=1.0,ax=None, strFmt='{:.4g}',
                        fontSz=10, normalize=True, colColors=None, colTxtColors=None,
                        showBorder=True,
                        kwargs_table={'loc':'center',
                                        'cellLoc': 'center',
                                        'bbox': [0.0, 0.0, 1.0, 1.0],}):
        if fig is None:
            fntFctr = 1.0 if fontSz is None else fontSz/10.0
            if autoFigSize:
                figsize[0] = 2*(self.N+1)*fntFctr*figWidthFctr
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.subplot()
        ax.axis('off')
        ax.axis('tight')

        tableContent = self.paramsTable(normalized=normalize)
        cell_text = []
        for key in tableContent.keys():
            if key == 'name':
                continue
            val = []
            for i in range(self.N):
                v = tableContent[key][i]
                if not isValidNumericScalar(v):
                    v = np.nan                
                val.append(strFmt.format(v))
            val.insert(0,key)
            cell_text.append(val)

        table = ax.table(cellText=cell_text,
                    colLabels=['Field',*[prof.name for prof in self.profiles]],
                    **kwargs_table)
        if fontSz is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(fontSz)

        # decorative edits
        nRows = len(cell_text)+1
        if colColors is not None:
            for i in range(self.N):
                for j in range(nRows):
                    table[(j,i+1)].set_facecolor(colColors[i])
        if colTxtColors is not None:
            for i in range(self.N):
                for j in range(nRows):
                    table[(j,i+1)].set_text_props(color=colTxtColors[i])
        for i in range(self.N+1):
            table[(0,i)].set_text_props(weight='bold')
        # for i in range(nRows):
        #     table[(i,0)].set_text_props(weight='bold')
        if not showBorder:
            for key, cell in table.get_celld().items():
                cell.set_linewidth(0)

        fig.tight_layout()
        plt.show()
        return fig, ax

class BldgCps():
    def __init__(self,
                memberBldgs:List[bldgCp]=[],
                masterBldg:Union[bldgCp,None]=None,
                ) -> None:
        self.memberBldgs: List[bldgCp] = memberBldgs
        print(f"Number of member bldgs: {len(memberBldgs)}")

        self.master = memberBldgs[0] if masterBldg is None and len(memberBldgs) > 0 else masterBldg
        # super().__init__(caseName=mb.name, H=mb.H, He=mb.He, Hr=mb.Hr, B=mb.B, D=mb.D, roofSlope=mb.roofSlope, lScl=mb.lScl, valuesAreScaled=mb.valuesAreScaled, faces=mb.memberFaces,
        #                 # refProfile=mb.refProfile, samplingFreq=mb.samplingFreq, airDensity=mb.airDensity,
        #                 # Zref_input=mb.Zref, Uref_input=mb.Uref, Uref_FS=mb.Uref_FS, badTaps=mb.badTaps, AoA=mb.AoA, AoA_zero_deg_basisVector=mb.AoA_zero_deg_basisVector, 
        #                 # AoA_rotation_direction=mb.AoA_rotation_direction,
        #                 # CpOfT=mb.CpOfT, pOfT=mb.pOfT, p0ofT=mb.p0ofT, CpStats=mb.CpStats, peakSpecs=mb.peakSpecs,
        #                 )
        # self.master = masterBldg

    def __fold(self):
        flds = self.master.CpStats.keys()
        stat = {}
        for fld in flds:
            stat[fld] = np.zeros_like(self.master.CpStats[fld])
            for i, bldg in enumerate(self.memberBldgs):
                stat[fld] += bldg.CpStats[fld]
            stat[fld] /= self.N_bldgs
        pass

    """-------------------------------- Properties ------------------------------------"""
    @property
    def N_bldgs(self):
        return len(self.memberBldgs)
    

    """--------------------------------- Methods --------------------------------------"""
    def CpStats(self, errorPercentile=95.0,): 
        if len(self.memberBldgs) == 0 or self.master is None:
            return None
        flds = self.master.CpStats.keys()
        upper = {} 
        mean = {}
        lower = {}
        allData = {}
        for fld in flds:
            # initialize clctd{} by zeros like the master bldg CpStats with one extra dimension for the member bldgs as the last dimension
            allData[fld] = np.zeros(np.shape(self.master.CpStats[fld]) + (self.N_bldgs,))
            for i, bldg in enumerate(self.memberBldgs):
                # collect the CpStats of the member bldgs into the last dimension of clctd{} without explicitly knowing the dimensionality of the CpStats
                allData[fld][...,i] = bldg.CpStats[fld]
                
            mean[fld] = np.mean(allData[fld], axis=-1)
            upper[fld] = np.percentile(allData[fld], 50+errorPercentile/2, axis=-1)
            lower[fld] = np.percentile(allData[fld], 50-errorPercentile/2, axis=-1)
        return mean, upper, lower, allData
    
    def writeCandCloadToXLSX(self, filePath, 
                            extremesPerNominalArea=True, areaFactor=1.0, 
                            format:Literal['default','NBCC','ASCE']='default', debugMode=False,
                            ):
        if self.master.T is None or self.master.tScl is None or self.master.H is None or self.master.profile is None:
            print(f"Cannot compute C&C Load factor. The following are required: T, tScl, H, profile")
            return None

        data = {}
        for i, bldg in enumerate(self.memberBldgs):
            zoneDictKeys = bldg.zoneDictKeys
            if format == 'default':
                valueScaleFactor = 1.0
            elif format == 'NBCC':
                valueScaleFactor = bldg.CandCLoad_factor(debugMode=debugMode, format='NBCC')
            elif format == 'ASCE':
                valueScaleFactor = bldg.CandCLoad_factor(debugMode=debugMode, format='ASCE')
            else:
                raise Exception(f"Unknown CandCLoadFormat: {format}")
            area = {z: np.array(bldg.NominalPanelArea) for z in zoneDictKeys} if extremesPerNominalArea else bldg.allAreasForCpAavg
            if extremesPerNominalArea:
                peakMax = bldg.CpAavg_envMax_peakMax_maxPerA
                peakMin = bldg.CpAavg_envMin_peakMin_minPerA
            else:
                peakMax = bldg.CpAavg_envMax_peakMax_allA
                peakMin = bldg.CpAavg_envMin_peakMin_allA
            for zKey in zoneDictKeys:
                if i == 0:
                    data[zKey] = pd.DataFrame({'Area [m^2]':area[zKey]*areaFactor, 
                                   'Peak Max':np.array(peakMax[zKey])*valueScaleFactor, 
                                   'Peak Min':np.array(peakMin[zKey])*valueScaleFactor})
                else:
                    data[zKey] = pd.concat([data[zKey], 
                                    pd.DataFrame({'Area [m^2]':area[zKey]*areaFactor, 
                                    'Peak Max':np.array(peakMax[zKey])*valueScaleFactor, 
                                    'Peak Min':np.array(peakMin[zKey])*valueScaleFactor})], 
                                axis=0)
                    
        with pd.ExcelWriter(filePath) as writer:
            for zKey in zoneDictKeys:
                sheetName = zKey
                # make sure the sheet name is valid (replace apostrophe with 'prime', etc.)
                sheetName = sheetName.replace("'", "_prime")
                data[zKey].to_excel(writer, sheet_name=sheetName, index=False)

        if debugMode:
            print(f"Saved C&C load to {filePath}")

    def paramsTable(self, normalized=True, fields=None) -> dict:
        params = None
        for i, bldg in enumerate(self.memberBldgs):
            if i == 0:
                params = bldg.paramsTable(normalized=normalized, fields=fields)
                # change each entry to a list
                for key in params.keys():
                    params[key] = [params[key],]
            else:
                temp = bldg.paramsTable(normalized=normalized, fields=fields)
                # append each entry to the list
                for key in temp.keys():
                    params[key].append(temp[key])

        return params
    
    """--------------------------------- Plotters -------------------------------------"""
    def plotTapCpStatsPerAoA(self, 
                            fields=['peakMax','mean','peakMin'],fldRange=[-15,10], tapsToPlot=None, includeTapName=True,
                            nCols=7, nRows=10, 
                            cols = ['r','k','b','g','m','r','k','b','g','m'],
                            mrkrs = ['^','o','v','s','p','d','.','*','<','>','h'], 
                            ls=['-','-','-','-','-','-','-','-','-','-',],
                            mrkrSize=2, lw=0.5,
                            kwargs_perFld=None, 
                            xticks=None, nAoA_ticks=5, xlim=None, 
                            legend_bbox_to_anchor=(0.5, 0.905), nLgndCols=None, simpleLabel=False,
                            pageNo_xy=(0.5,0.1), 
                            figsize=[15,20], sharex=True, sharey=True,
                            overlayThis=None, overlay_AoA=None, overlayLabel=None, kwargs_overlay={},
                             ):
        if kwargs_perFld is None:
            kwargs_perFld = [[{'color':cols[c], 
                                'marker':mrkrs[i], 
                                'ls':ls[i],
                                'markersize':mrkrSize,
                                'linewidth':lw,
                                } for i in range(len(fields))]
                                for c in range(self.N_bldgs)] 
            

        tapIdxs = self.master.tapIdx if tapsToPlot is None else self.master.tapIdxOf(tapsToPlot)
        
        for fld in fields:
            self.master.checkStatField(fld)
        nPltPerPage = nCols * nRows
        nPltTotal = len(tapIdxs)
        nPages = int(np.ceil(nPltTotal/nPltPerPage))
        
        figs = []
        all_axes = []
        for p in range(nPages):
            fig, axs = plt.subplots(nRows, nCols, figsize=figsize, sharex=sharex, sharey=sharey)
            if nRows == 1 and nCols == 1:
                axs = np.array([[axs]])
            elif nRows == 1: # make sure axs is 2D
                axs = np.array([axs])
            elif nCols == 1:
                axs = np.array([[ax] for ax in axs])
            all_axes.append(axs)
            figs.append(fig)
        
        addMarginDetails = True
        for i, bldg in enumerate(self.memberBldgs):
            if i > 0:
                overlayThis = None
                addMarginDetails = False
            bldg.plotTapCpStatsPerAoA(figs=figs, all_axes=all_axes, addMarginDetails=addMarginDetails,
                                    fields=fields, fldRange=fldRange, tapsToPlot=tapsToPlot, includeTapName=includeTapName,
                                    kwargs_perFld=kwargs_perFld[i], 
                                    simpleLabel=simpleLabel,
                                    xticks=xticks, nAoA_ticks=nAoA_ticks, xlim=xlim, 
                                    legend_bbox_to_anchor=legend_bbox_to_anchor, pageNo_xy=pageNo_xy,
                                    overlayThis=overlayThis, overlay_AoA=overlay_AoA, overlayLabel=overlayLabel, kwargs_overlay=kwargs_overlay,
                                    nCols=nCols, nRows=nRows,
                                    )
        
        caseNames = [bldg.name for bldg in self.memberBldgs]
        nLgndCols = self.N_bldgs if nLgndCols is None else nLgndCols
        transposeLegend = False
        for p, (axs, fig) in enumerate(zip(all_axes, figs)):
            ax = axs[0,0]
            handles, labels = ax.get_legend_handles_labels()
            if transposeLegend:
                pass
            fig.legend(handles, labels, loc='upper center',ncol=nLgndCols, bbox_to_anchor=legend_bbox_to_anchor, bbox_transform=fig.transFigure)
            text = f"Page {p+1} of {nPages}\n Cases: {caseNames}"
            fig.text(pageNo_xy[0], pageNo_xy[1], text, ha='center', va='top')
        
        # turn off the remaining axes
        if len(tapIdxs) < nPltPerPage*nPages:
            axs = all_axes[-1]
            nTapsOnLastPage = len(tapIdxs) - nPltPerPage*(nPages-1)
            for i in range(nTapsOnLastPage, nPltPerPage):
                ax = axs[np.unravel_index(i, axs.shape)]
                ax.axis('off')
            
        return figs, all_axes

    def plotParamsTable(self,
                        fig=None,figsize=[0,8],autoFigSize=True,figWidthFctr=1.0,ax=None, strFmt='{:.4g}',
                        fontSz=10, normalize=True, colColors=None, colTxtColors=None,
                        showBorder=True,
                        kwargs_table={'loc':'center',
                                        'cellLoc': 'center',
                                        'bbox': [0.0, 0.0, 1.0, 1.0],}):
        if fig is None:
            fntFctr = 1.0 if fontSz is None else fontSz/10.0
            if autoFigSize:
                figsize[0] = 2*(self.N_bldgs+1)*fntFctr*figWidthFctr
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.subplot()
        ax.axis('off')
        ax.axis('tight')

        tableContent = self.paramsTable(normalized=normalize)
        cell_text = []
        for key in tableContent.keys():
            if key == 'Name':
                continue
            val = []
            for i in range(self.N_bldgs):
                v = tableContent[key][i]
                if not isValidNumericScalar(v):
                    v = np.nan     
                val.append(strFmt.format(v))
            val.insert(0,key)
            cell_text.append(val)

        table = ax.table(cellText=cell_text,
                    colLabels=['Field',*[bldg.name for bldg in self.memberBldgs]],
                    **kwargs_table)
        if fontSz is not None:
            table.auto_set_font_size(False)
            table.set_fontsize(fontSz)
        else:
            table.auto_set_font_size(True)

        # decorative edits
        nRows = len(cell_text)+1
        if colColors is not None:
            for i in range(self.N_bldgs):
                for j in range(nRows):
                    table[(j,i+1)].set_facecolor(colColors[i])
        if colTxtColors is not None:
            for i in range(self.N_bldgs):
                for j in range(nRows):
                    table[(j,i+1)].set_text_props(color=colTxtColors[i])
        for i in range(self.N_bldgs+1):
            table[(0,i)].set_text_props(weight='bold')
        # for i in range(nRows):
        #     table[(i,0)].set_text_props(weight='bold')
        if not showBorder:
            for key, cell in table.get_celld().items():
                cell.set_linewidth(0)

        fig.tight_layout()
        plt.show()
        return fig, ax

class validator():
    def __init__(self, 
                target:Union[None, bldgCp, BldgCps]=None,
                model:Union[None, bldgCp, BldgCps]=None,
                errorTypes_velStats = ['MAE', 'RMSE', 'NMAE', 'NRMSE'],  #['MAE', 'NRMSE', 'RMSE', 'R^2', 'SMAPE', 'PCC'],
                errorTypes_CpStats = ['MAE', 'RMSE', 'NMAE', 'NRMSE'],  #['MAE', 'NRMSE', 'RMSE', 'R^2', 'SMAPE', 'PCC'],
                errorTypes_CpAavg = ['MAE', 'RMSE', 'NMAE', 'NRMSE'],  #['MAE', 'NRMSE', 'RMSE', 'R^2', 'SMAPE', 'PCC'],
                extremesPerNominalArea=True,
                CandCLoadFormat:Literal['default','NBCC','ASCE']='default',
                combineMinAndMax = True,
                correctForIuDifference = False,
                ) -> None:
        self.target: bldgCp = target
        self.model: bldgCp = model
        self.errorTypes_velStats = errorTypes_velStats
        self.errorTypes_CpStats = errorTypes_CpStats
        self.errorTypes_CpAavg = errorTypes_CpAavg
        self.extremesPerNominalArea=extremesPerNominalArea
        self.CandCLoadFormat:Literal['default','NBCC','ASCE']=CandCLoadFormat
        self.combineMinAndMax = combineMinAndMax
        self.correctForIu = correctForIuDifference

        self.error_params = None
        self.error_velStats = None
        self.error_CpStats = None
        self.error_CpAavg = None
        self.diff_CpStats = None
        self.ndiff_CpStats = None
        self.commonAoA:list = []
        self.commonTapNo:list = []
        self.tapIdx_target = None
        self.tapIdx_model = None
        self.aoaIdx_target = None
        self.aoaIdx_model = None
        self.IuFctr = 1.0
        
        self.Refresh()

    def __calculateDiff_CpStats(self,):
        if self.target.CpStats is None or self.model.CpStats is None:
            raise Exception(f"Cannot compute CpStats error. The following are required: CpStats")
        if self.target.CpStats.keys() != self.model.CpStats.keys():
            raise Exception(f"Cannot compute CpStats error. The following are required: CpStats")
        
        self.diff_CpStats = {}
        self.ndiff_CpStats = {}
        for key in self.target.CpStats.keys():
            IuFctr = self.IuFctr if key in SCALABLE_CP_STATS else 1.0
            self.diff_CpStats[key] = self.model.CpStats[key][self.aoaIdx_model[:, np.newaxis]] * IuFctr - \
                                     self.target.CpStats[key][self.aoaIdx_target[:, np.newaxis]]
            targetValueRangePerAoA = np.max(self.target.CpStats[key][self.aoaIdx_target[:, np.newaxis]], axis=0) - \
                                        np.min(self.target.CpStats[key][self.aoaIdx_target[:, np.newaxis]], axis=0)
            self.ndiff_CpStats[key] = self.diff_CpStats[key] / targetValueRangePerAoA

    def __computeIuCorrection(self,):
        if not self.correctForIu:
            self.IuFctr = 1.0
            return
        if self.target is None or self.model is None or self.target.profile is None or self.model.profile is None:
            print(f"Cannot compute Iu correction factor. The following are required: target, model, target.profile, model.profile")
            self.IuFctr = 1.0
            return
        if self.target.profile.z0_Iu is None or self.model.profile.z0_Iu is None:
            print(f"Cannot compute Iu correction factor. The following are required: target.profile.z0_Iu, model.profile.z0_Iu")
            self.IuFctr = 1.0
            return
        if self.target.profile.z0_Iu == self.model.profile.z0_Iu:
            self.IuFctr = 1.0
            return
        
        z0_t = self.target.profile.z0_Iu / self.target.lScl
        z0_m = self.model.profile.z0_Iu / self.model.lScl

        self.IuFctr = wc.velRatio_exposureChange(from_z0=z0_m, to_z0=z0_t, zref=self.model.H/self.model.lScl) ** -2.0

    def __computeErrors_params(self, 
                                  errorMeasures_params:Literal['diff','diff_norm','ratio','orderOfMagnitude']=['diff','diff_norm','ratio','orderOfMagnitude']):
        tmp1 = {}
        if 'diff' in errorMeasures_params:
            tmp1['diff'] = {}
        if 'diff_norm' in errorMeasures_params:
            tmp1['diff_norm'] = {}
        if 'ratio' in errorMeasures_params:
            tmp1['ratio'] = {}
        if 'orderOfMagnitude' in errorMeasures_params:
            tmp1['orderOfMagnitude'] = {}
        
        tp = self.target.paramsTable()
        mp = self.model.paramsTable()
        for key in tp.keys():
            tp_i, mp_i = tp[key], mp[key]
            if not isValidNumericScalar(tp_i):
                tp_i = np.nan
            if not isValidNumericScalar(mp_i):
                mp_i = np.nan
            if 'diff' in errorMeasures_params:
                tmp1['diff'][key] = mp_i - tp_i
            if 'diff_norm' in errorMeasures_params:
                tmp1['diff_norm'][key] = (mp_i - tp_i)/tp_i
            if 'ratio' in errorMeasures_params:
                tmp1['ratio'][key] = mp_i/tp_i
            if 'orderOfMagnitude' in errorMeasures_params:
                tmp1['orderOfMagnitude'][key] = np.log10(mp_i/tp_i)

        self.error_params = tmp1

    def __computeErrors_velStats(self):
        Zt = np.array(self.target.profile.Z_eff)
        Zm = np.array(self.model.profile.Z_eff)
        zMin, zMax = np.max([np.min(Zt), np.min(Zm)]), np.min([np.max(Zt), np.max(Zm)])
        mask_t = np.logical_and(Zt >= zMin, Zt <= zMax)
        mask_m = np.logical_and(Zm >= zMin, Zm <= zMax)
        Z = np.sort(np.concatenate((Zt[mask_t], Zm[mask_m])))
        if len(Z) == 0:
            raise Exception(f"The target and model profiles do not overlap. Cannot compute error")

        stats_t = self.target.profile.stats_norm_interp_to_Z(Z)
        stats_m = self.model.profile.stats_norm_interp_to_Z(Z)
        tmp2 = {}
        for key in stats_t.keys():
            stats_t_i, stats_m_i = stats_t[key], stats_m[key]
            tmp2[key] = measureError(stats_m_i, stats_t_i, errorTypes=self.errorTypes_velStats)

        self.error_velStats = tmp2

    def __computeErrors_CpStats(self,):
        if self.target.CpStats is None or self.model.CpStats is None:
            raise Exception(f"Cannot compute CpStats error. The following are required: CpStats")
        if self.target.CpStats.keys() != self.model.CpStats.keys():
            raise Exception(f"Cannot compute CpStats error. The following are required: CpStats")
        if self.target.faces != self.model.faces:
            warnings.warn(f"The target and model faces seem to be referring to different 'faces' objects. This may result in failure to calculate CpStats error or incorrect values. Please check and make sure that all the face geometries, tap layouts, and other details are the same (except for the Cp data) between the target and model. The computation will continue but the results may be incorrect.")
        
        aoa_t = np.array(self.target.AoA)
        tapNo_t = np.array(self.target.tapNo)
        # tapIdx_t = np.array(self.target.tapIdx)

        aoa_m = np.array(self.model.AoA)
        tapNo_m = np.array(self.model.tapNo)
        # tapIdx_m = np.array(self.model.tapIdx)

        self.commonAoA = np.sort(np.intersect1d(aoa_t, aoa_m))
        self.commonTapNo = np.sort(np.intersect1d(tapNo_t, tapNo_m))
        if len(self.commonAoA) == 0 or len(self.commonTapNo) == 0:
            raise Exception(f"The target and model CpStats do not overlap (no common AoA or tapNo). Cannot compute error")

        self.tapIdx_target = np.array([np.where(tapNo_t == i)[0][0] for i in self.commonTapNo])
        self.tapIdx_model = np.array([np.where(tapNo_m == i)[0][0] for i in self.commonTapNo])

        self.aoaIdx_target = np.array([np.where(aoa_t == i)[0][0] for i in self.commonAoA])
        self.aoaIdx_model = np.array([np.where(aoa_m == i)[0][0] for i in self.commonAoA])

        tmp = {}
        tmp['AoA'] = self.commonAoA
        tmp['tapNo'] = self.commonTapNo
        for key in self.target.CpStats.keys():
            IuFctr = self.IuFctr if key in SCALABLE_CP_STATS else 1.0
            stat_t = self.target.CpStats[key][self.aoaIdx_target[:, np.newaxis], self.tapIdx_target]
            stat_m = self.model.CpStats[key][self.aoaIdx_model[:, np.newaxis], self.tapIdx_model] * IuFctr
            tmp[key] = measureError(stat_t, stat_m, errorTypes=self.errorTypes_CpStats)
            tmp[key]['perAoA'] = {}
            for i, aoa in enumerate(self.commonAoA):
                stat_t_i = stat_t[i,:]
                stat_m_i = stat_m[i,:] * IuFctr
                tmp[key]['perAoA'][aoa] = measureError(stat_t_i, stat_m_i, errorTypes=self.errorTypes_CpStats)

        self.error_CpStats = tmp

    def __computeErrors_CpAavg(self):
        if self.target.CpStatsAreaAvg is None or self.model.CpStatsAreaAvg is None:
            return
        if self.target.faces != self.model.faces:
            warnings.warn(f"The target and model faces seem to be referring to different 'faces' objects. This may result in failure to calculate CpAvg error or incorrect values. Please check and make sure that all the tap layouts, averaging panel definitions, and other details are the same (except for the Cp data) between the target and model. The computation will continue but the results may be incorrect.")
        
        bldg_t = self.target
        bldg_m = self.model

        zoneDictKeys = bldg_t.zoneDictKeys
        if zoneDictKeys != bldg_m.zoneDictKeys:
            raise Exception(f"Cannot compute CpAavg error. The zoneDictKeys of the target and model are not the same. Please check and make sure that all the tap layouts, averaging panel definitions, and other details are the same (except for the Cp data) between the target and model.")
        area = {z: np.array(bldg_t.NominalPanelArea) for z in zoneDictKeys} if self.extremesPerNominalArea else bldg_t.allAreasForCpAavg
        if self.extremesPerNominalArea:
            peakMax_t = bldg_t.CpAavg_envMax_peakMax_maxPerA
            peakMin_t = bldg_t.CpAavg_envMin_peakMin_minPerA
            peakMax_m = bldg_m.CpAavg_envMax_peakMax_maxPerA
            peakMin_m = bldg_m.CpAavg_envMin_peakMin_minPerA
        else:
            peakMax_t = bldg_t.CpAavg_envMax_peakMax_allA
            peakMin_t = bldg_t.CpAavg_envMin_peakMin_allA
            peakMax_m = bldg_m.CpAavg_envMax_peakMax_allA
            peakMin_m = bldg_m.CpAavg_envMin_peakMin_allA

        valueScaleFactor_t = 1.0 #if self.CandCLoadFormat == 'default' else bldg_t.CandCLoad_factor(debugMode=True, format=self.CandCLoadFormat)
        valueScaleFactor_m = self.IuFctr #1.0 if self.CandCLoadFormat == 'default' else bldg_m.CandCLoad_factor(debugMode=True, format=self.CandCLoadFormat)
        
        tmp = {}
        tmp['Area'] = area
        for I, zKey in enumerate(zoneDictKeys):
            pMax_t = np.array(peakMax_t[zKey])*valueScaleFactor_t
            pMin_t = np.array(peakMin_t[zKey])*valueScaleFactor_t
            pMax_m = np.array(peakMax_m[zKey])*valueScaleFactor_m
            pMin_m = np.array(peakMin_m[zKey])*valueScaleFactor_m
            if self.combineMinAndMax:
                p_t = np.concatenate((pMax_t, pMin_t))
                p_m = np.concatenate((pMax_m, pMin_m))
                tmp[zKey] = measureError(p_t, p_m, errorTypes=self.errorTypes_CpAavg)
            else:
                tmp[zKey] = {}
                tmp[zKey]['peakMax'] = measureError(pMax_t, pMax_m, errorTypes=self.errorTypes_CpAavg)
                tmp[zKey]['peakMin'] = measureError(pMin_t, pMin_m, errorTypes=self.errorTypes_CpAavg)

        self.error_CpAavg = tmp

    def Refresh(self):
        self.__computeIuCorrection()
        self.__computeErrors_params()
        self.__computeErrors_velStats()
        self.__computeErrors_CpStats()
        self.__computeErrors_CpAavg()
        self.__calculateDiff_CpStats()

    def compareProfiles(self):
        pass

    def plotError_velStats(self):
        pass

    def plotError_CpStats(self, fig=None, axs=None, figsize_per_ax=[4,4], 
                          nPltCols=3, 
                          fields:dict=None, 
                          errTypes:list=None, showErrTxt:bool=True,
                          lumpAllAoAs:bool=False, 
                          targetLabel='Target', modelLabel='Model',
                          percentLinesAt=[10,30], percentLinesAt_kwargs=None, 
                          xyLims=None,
                          kwargs_mainPlot={'color':'k', 'marker':'.', 'linestyle':''},
                          cols = def_cols,
                          kwargs_annotation={'xy':(0.95, 0.05), 'xycoords':'axes fraction', 'ha':'right', 'va':'bottom', 'backgroundcolor':[1,1,1,0.5],},
                          kwargs_legend={'loc':'best'},
                          ):
        errTypes = self.errorTypes_CpStats if errTypes is None else errTypes
        fields = self.target.CpStats.keys() if fields is None else fields
        nFlds = len(fields)
        nPltRows = int(np.ceil(nFlds/nPltCols))

        newFig = False
        if fig is None:
            figsize = [nPltCols*figsize_per_ax[0], nPltRows*figsize_per_ax[1]]
            fig, axs = plt.subplots(nPltRows, nPltCols, figsize=figsize)
            newFig = True

        axs = np.array(axs).flatten()
        
        if percentLinesAt_kwargs is None:
            percentLinesAt_kwargs = [{'color':cols[i], 'linestyle':'--', 'linewidth':0.5} for i in range(len(percentLinesAt))]

        xyLims = {fld:None for fld in fields} if xyLims is None else xyLims

        for i, fld in enumerate(fields):
            ax = axs[np.unravel_index(i, axs.shape)]
            ax.set_title(fullName(fld))
            ax.set_xlabel(targetLabel)
            ax.set_ylabel(modelLabel)
            ax.set_aspect('equal')

            if lumpAllAoAs:
                ax.plot(self.target.CpStats[fld][self.aoaIdx_target[:, np.newaxis],self.tapIdx_target], 
                        self.model.CpStats[fld][self.aoaIdx_model[:, np.newaxis],self.tapIdx_model] * self.IuFctr, 
                        **kwargs_mainPlot)
            else:
                for j, aoa in enumerate(self.commonAoA):
                    kwargs_mainPlot['color'] = cols[j]
                    kwargs_mainPlot['label'] = f'${aoa:.1f}^\circ$'
                    ax.plot(self.target.CpStats[fld][self.aoaIdx_target[j],self.tapIdx_target], 
                            self.model.CpStats[fld][self.aoaIdx_model[j],self.tapIdx_model] * self.IuFctr, 
                            **kwargs_mainPlot)

            if showErrTxt:
                errTxt = ''
                for j, errType in enumerate(errTypes):
                    errTxt += f"${errType} = {self.error_CpStats[fld][errType]:.3g}$"
                    if j < len(errTypes)-1:
                        errTxt += '\n'
                ax.annotate(errTxt, **kwargs_annotation)

            if fld in xyLims.keys() and xyLims[fld] is not None:
                ax.set_xlim(xyLims[fld])
                ax.set_ylim(xyLims[fld])

            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            minmax = np.array([np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])])
            ax.plot(minmax, minmax, 'k-')
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            for j, n in enumerate(percentLinesAt):
                ax.plot(minmax, minmax*(1+n/100), label=r'$\pm$'+f'{n}%', **percentLinesAt_kwargs[j])
                ax.plot(minmax, minmax*(1-n/100), **percentLinesAt_kwargs[j])
            ax.set_xlim(minmax)
            ax.set_ylim(minmax)
            formatAxis(ax=ax, gridMajor=False, gridMinor=False)
            if i == 0:
                ax.legend(**kwargs_legend)

        # turn off the remaining axes
        if nFlds < nPltCols*nPltRows:
            for i in range(nFlds, nPltCols*nPltRows):
                ax = axs[np.unravel_index(i, axs.shape)]
                ax.axis('off')

        if newFig:
            fig.tight_layout()
            plt.show()
        return fig, axs

    def plotError_CpStats_perAoA(self, fig=None, axs=None, figsize_per_ax=[4,4], 
                          errorTypePerField:dict={'mean':'RMSE', 'std':'RMSE', 'peakMax':'RMSE', 'peakMin':'RMSE'}, 
                          targetLabel='Target', modelLabel='Model',
                          percentLinesAt=[10,30], percentLinesAt_kwargs=None, 
                          kwargs_mainPlot={'color':'k', 'marker':'.', 'linestyle':''},
                          kwargs_annotation={'xy':(0.95, 0.05), 'xycoords':'axes fraction', 'ha':'right', 'va':'bottom'},
                          ):
        nFlds = len(errorTypePerField.keys())
        nPltCols = len(self.commonAoA)#+1
        nPltRows = nFlds

        newFig = False
        if fig is None:
            figsize = [nPltCols*figsize_per_ax[0], nPltRows*figsize_per_ax[1]]
            fig, axs = plt.subplots(nPltRows, nPltCols, figsize=figsize)
            newFig = True

        axs = np.array(axs).flatten()

        cols = ['r','b','g','m','r','k','b','g','m']
        if percentLinesAt_kwargs is None:
            percentLinesAt_kwargs = [{'color':cols[i], 'linestyle':'--', 'linewidth':0.5} for i in range(len(percentLinesAt))]

        flds = list(errorTypePerField.keys())
        for f, fld in enumerate(flds):
            for d, aoa in enumerate(self.commonAoA):
                ax = axs[np.unravel_index(f*nPltCols+d, axs.shape)]
                if f == 0:
                    ax.set_title(r'$\theta = $'+f'${aoa:.1f}^\circ$')
                if d == 0: # write the field name on the left column rotated by 90 degrees
                    ax.text(-0.25, 0.5, mathName(fld), 
                            horizontalalignment='center', verticalalignment='center', 
                            transform=ax.transAxes, rotation=90)
                ax.set_xlabel(targetLabel)
                ax.set_ylabel(modelLabel)
                ax.set_aspect('equal')

                ax.plot(self.target.CpStats[fld][self.aoaIdx_target[d],self.tapIdx_target], 
                        self.model.CpStats[fld][self.aoaIdx_model[d],self.tapIdx_model] * self.IuFctr, 
                        **kwargs_mainPlot)

                errTxt = ''
                for j, errType in enumerate(self.error_CpStats[fld]['perAoA'][aoa]):
                    errTxt += f"${errType} = {self.error_CpStats[fld]['perAoA'][aoa][errType]:.3g}$"
                    if j < len(self.error_CpStats[fld]['perAoA'][aoa])-1:
                        errTxt += '\n'
                ax.annotate(errTxt, **kwargs_annotation)

                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                minmax = np.array([np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])])
                ax.plot(minmax, minmax, 'k-')
                for j, n in enumerate(percentLinesAt):
                    ax.plot(minmax, minmax*(1+n/100), label=r'$\pm$'+f'{n}%', **percentLinesAt_kwargs[j])
                    ax.plot(minmax, minmax*(1-n/100), **percentLinesAt_kwargs[j])
                ax.set_xlim(minmax)
                ax.set_ylim(minmax)
                formatAxis(ax=ax)
                if d == 0 and f == 0:
                    ax.legend()
            
            # ax = axs[np.unravel_index(f*nPltCols+nPltCols-1, axs.shape)]
            # ax.set_title(errorTypePerField[fld])
            # ax.set_xlabel('AoA')
            # ax.set_ylabel(errorTypePerField[fld])
            # ax.plot(self.commonAoA, [self.error_CpStats[fld]['perAoA'][aoa][errorTypePerField[fld]] for aoa in self.commonAoA], '.k-')
            # formatAxis(ax=ax)
            
        if newFig:
            fig.tight_layout()
            plt.show()
        return fig, axs

    def plotError_CpAavg(self):
        pass

    def plotError_contour_CpStats(self, fieldName:Literal['mean','std','peak','peakMin','peakMax','skewness','kurtosis'], normalizedError:bool=True,
                            AoAs=None, envelopeType:Literal['None','high','low','both']='None', tLbl='Target', mLbl='Model',
                            figsize_per_ax=[5,5], ax=None, nCols=3, 
                            fldRange=None, nLvl=100, cmap='RdBu', extend='both', title=None, colBarOrientation='horizontal',
                            showValuesOnContour=True, kwargs_contourTxt={'inline':True, 'fmt':'%.2g','fontsize':4, 'colors':'gray'},
                            showContourEdge=True, kwargs_contourEdge={'colors':'k', 'linewidths':0.3, 'linestyles':'solid', 'alpha':0.3},
                            kwargs_Edge={'showName':False}, 
                            kwargs_AoAsymbol={'location':'lower right', 'marginFactor':1.0, 'drawDicorations':False},
                            ):
        if fieldName not in self.error_CpStats.keys():
            raise Exception(f"Unknown fieldName {fieldName}. Valid options are {self.error_CpStats.keys()}")
        AoAs = self.commonAoA if AoAs is None else AoAs
        aoaIdx = [np.where(self.commonAoA == aoa)[0][0] for aoa in AoAs]
        
        if envelopeType == 'both':
            # data = self.CpStatEnvlp[fieldName]
            raise NotImplementedError
        elif envelopeType == 'high':
            # data = self.CpStatEnvlp_high[fieldName]
            raise NotImplementedError
        elif envelopeType == 'low':
            # data = self.CpStatEnvlp_low[fieldName]
            raise NotImplementedError
        elif envelopeType == 'None':
            if normalizedError:
                data = np.array(self.ndiff_CpStats[fieldName][aoaIdx,:], dtype=float)
            else:
                data = np.array(self.diff_CpStats[fieldName][aoaIdx,:], dtype=float)
            if len(AoAs) > 1:
                data = np.squeeze(data)
        else:
            raise Exception(f"Unknown envelope type {envelopeType}")
        # else:
        #     data = np.squeeze(np.array(self.diff_CpStats[fieldName][aoaIdx,:], dtype=float))

        nAxs = len(aoaIdx)+1
        newFig = False
        if ax is None:
            newFig = True
            
            nCols = min(nCols, nAxs)
            nRows = int(np.ceil(nAxs/nCols))
            figsize = [nCols*figsize_per_ax[0], nRows*figsize_per_ax[1]]
            fig, axs = plt.subplots(nRows, nCols, figsize=figsize)

        axs = np.array(axs).flatten()

        for i, aoa in enumerate(AoAs):
            ax = axs[np.unravel_index(i, axs.shape)]
            # ax.set_title(r'$\theta = $'+f'${aoa:.1f}^\circ$')
            dtt = data[i,:] if len(AoAs) > 1 else data
            cObj = self.model.plotTapField(ax=ax, field=dtt, fldRange=fldRange, nLvl=nLvl, cmap=cmap, extend=extend,
                                    showValuesOnContour=showValuesOnContour, kwargs_contourTxt=kwargs_contourTxt,
                                    showContourEdge=showContourEdge, kwargs_contourEdge=kwargs_contourEdge)
            self.model.plotEdges(ax=ax, **kwargs_Edge)
            self.model.plotTaps(ax=ax, kwargs_dots={'color':'k', 'marker':'.', 'alpha':0.3, 'ls':'', 'ms':1}, )
            self.model.plotAoA_symbol(ax=ax, AoA=aoa, **kwargs_AoAsymbol, )
            ax.axis('equal')
            ax.axis('off')

        if len(aoaIdx) < nCols*nRows:
            axs = axs.flatten()
            for i in range(len(aoaIdx), nCols*nRows):
                ax = axs[np.unravel_index(i, axs.shape)]
                ax.axis('off')
            
        if newFig:
            def custom_formatter(x, pos):
                return f'{x:.2g}'
            
            if title is None:
                fld = mathName(fieldName)[1:-1]
                if normalizedError:
                    # the equation is (model - target) / (target_max - target_min)
                    title = r'$\frac{'+fld+r'^{'+mLbl+r'} - '+fld+r'^{'+tLbl+r'}}{'+fld+r'^{'+tLbl+r'}_{max} - '+fld+r'^{'+tLbl+r'}_{min}}$'
                else:
                    # the equation is model - target
                    title = r'$'+fld+r'^{'+mLbl+r'} - '+fld+r'^{'+tLbl+r'}$'
            cbar = fig.colorbar(cObj[0], ax=ax, orientation=colBarOrientation)
            cbar.set_ticks(np.linspace(fldRange[0], fldRange[1], 7))
            cbar.set_label(title, fontsize=14,)
            if colBarOrientation == 'horizontal':
                cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
            else:
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

            fig.tight_layout()
            plt.show()

        return fig, axs

    def plotError_barChart_CpStats(self, fig=None, axs=None, plotType:Literal['bar','line']='bar',
                                   errorType:List[Literal['MAE', 'RMSE', 'NMAE', 'NRMSE']]=None, plotNormalizedErrorsAsPercentage:bool=True, showErrEqn:bool=True,
                                   fields:List[str]=None,
                                   nPltCols:int=3, figsize_per_ax=[4,4], lumpAllAoAs:bool=False,
                                   tLbl='BLWT', mLbl='LES', yLims:dict=None, cols=def_cols,
                                   kwargs_annotation={'xy':(0.95, 0.95), 'xycoords':'axes fraction', 'ha':'right', 'va':'top', 'backgroundcolor':[1,1,1,0.5], 'fontsize':12},
                                   kwargs_legend={'loc':'best'},
                                   ):
        errorType = self.errorTypes_CpStats if errorType is None else errorType
        fields = list(self.target.CpStats.keys()) if fields is None else fields
        nFlds = len(fields)
        nErrTypes = len(errorType)
        nPltCols = min(nErrTypes, nPltCols)
        nPltRows = int(np.ceil(nErrTypes/nPltCols))

        newFig = False
        if fig is None:
            figsize = [nPltCols*figsize_per_ax[0], nPltRows*figsize_per_ax[1]]
            fig, axs = plt.subplots(nPltRows, nPltCols, figsize=figsize)
            newFig = True

        axs = np.array(axs).flatten()
        _, errorEqn = measureError(returnEqn=True, cfdName=mLbl, expName=tLbl)
        normFactor = 100.0 if plotNormalizedErrorsAsPercentage else 1.0
        normTxt = ' [%]' if plotNormalizedErrorsAsPercentage else ''

        for i, err in enumerate(errorType):
            nf = normFactor if err in NORMALIZED_ERROR_TYPES else 1.0
            ntxt = normTxt if err in NORMALIZED_ERROR_TYPES else ''
            ax = axs[np.unravel_index(i, axs.shape)]
            
            if lumpAllAoAs:
                if plotType == 'line':
                    ax.plot(np.arange(nFlds), [self.error_CpStats[fld][err]*nf for fld in fields], '.-')
                elif plotType == 'bar':
                    ax.bar(np.arange(nFlds), [self.error_CpStats[fld][err]*nf for fld in fields], width=0.8, color=cols[0])
                    ax.set_xticks(np.arange(nFlds))
                    ax.set_xticklabels(fullName(fields,abbreviate=True))
            else:
                if plotType == 'bar':
                    barWidthPerAoA = 0.8 / len(self.commonAoA)
                    barLocPerAoA = np.arange(nFlds) - 0.4 + barWidthPerAoA/2
                for j, aoa in enumerate(self.commonAoA):
                    if plotType == 'line':
                        ax.plot(np.arange(nFlds), [self.error_CpStats[fld]['perAoA'][aoa][err]*nf for fld in fields], '.-')
                    elif plotType == 'bar':
                        ax.bar(barLocPerAoA + j*barWidthPerAoA, [self.error_CpStats[fld]['perAoA'][aoa][err]*nf for fld in fields],
                                width=barWidthPerAoA, label=f'${aoa:5.1f}^\circ$', color=cols[j])
                        ax.set_xticks(np.arange(nFlds))
                        ax.set_xticklabels(fullName(fields,abbreviate=True))

            if showErrEqn:
                ax.annotate(errorEqn[err], **kwargs_annotation)

            if yLims is not None and err in yLims.keys():
                ax.set_ylim(yLims[err])

            # ax.set_xlabel(targetLabel)
            ax.set_ylabel(err+ntxt)

            # formatAxis(ax=ax, gridMajor=False, gridMinor=False)
            if i == 0:
                ax.legend(**kwargs_legend)

        if nErrTypes < nPltCols*nPltRows:
            for i in range(nErrTypes, nPltCols*nPltRows):
                ax = axs[np.unravel_index(i, axs.shape)]
                ax.axis('off')

        if newFig:
            fig.tight_layout()
            plt.show()
        return fig, axs

