# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:19:46 2022

@author: Tsinu
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

from matplotlib.backends.backend_pdf import PdfPages

# from PyPDF2 import PdfFileReader, PdfFileWriter  # to append pdfs
# https://stackoverflow.com/questions/38118510/append-page-to-existing-pdf-file-using-python-and-matplotlib



#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================

ctlProf = {
    "col":      ('k','r','g','m','b','c','r','g','b','k','m','c'),
    "lStyle":   ('-','--','-.',':','-','-','--','--','--','--','--','--'),
    "marker":   ('o','s','v','^','>','<','x','d','*','1','2','3'),
    "mrkrSize": (2,2,2,2,2,2,2,2,2,2,2,2)
    }

ctlSpect = {
    "col":      ('k','r','g','m','b','c','r','g','b','k','m','c'),
    "lStyle":   ('-','-','-','-','-','-','--','--','--','--','--','--'),
    "marker":   ('o','s','v','^','>','<','x','d','*','1','2','3'),
    "mrkrSize": (2,2,2,2,2,2,2,2,2,2,2,2)
    }

plot = {
        'loglog':plt.loglog,
        'plot':plt.plot,
        'semilogx':plt.semilogx,
        'semilogy':plt.semilogy
    }

#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

def plotProfiles(
                Z, # ([n1,], [n2,], ... [nN,])
                val, # ([n1,M], [n2,M], ... [nN,M])
                dataLabels=None, # ("str1", "str2", ... "strN")
                pltFile=None, # "/path/to/plot/file.pdf"
                xLabels=None, # ("str1", "str2", ... "str_m")
                yLabel=r"$Z$",
                xLimits='auto', # ([vMin1,vMax1], [vMin2,vMax2], ... [vMin_m,vMax_m])
                yLimits='auto', # [zMin, zMax]
                figSize=[15, 5],
                alwaysShowFig=False,
                nCols=3
                ):
    
    N = len(Z)
    M = val[0].shape[1]
    nRows = int(np.ceil((M/nCols)))
    
    if dataLabels is None:
        dataLabels = (['Dataset_1'])
        for i in range(1,N):
            dataLabels += (['Dataset_'+str(i+1)])
    if xLabels is None:
        xLabels = (['Var_1'])
        for i in range(1,M):
            xLabels += (['Var_'+str(i+1)])

    if yLimits == 'auto':
        yLimits = [0, max([item for sublist in Z for item in sublist])]
    
    if isinstance(pltFile, str):
        pdf = PdfPages(pltFile)
    else:
        pdf = pltFile
    fig = plt.figure(figsize=figSize) 

    for m in range(M):
        plt.subplot(nRows,nCols,m+1)
        for i in range(N):
            plt.plot(val[i][:,m], Z[i], color=ctlProf['col'][i], linestyle=ctlProf['lStyle'][i], label=dataLabels[i])
        plt.xlabel(xLabels[m])
        plt.ylabel(yLabel)
        if xLimits is not None and not xLimits == 'auto':
            plt.xlim(xLimits[m])
        plt.ylim(yLimits)
        plt.legend()
        plt.grid()
        plt.tight_layout()
    
    if isinstance(pltFile, str):
        pdf.savefig(fig)
        plt.clf()
    if alwaysShowFig:
        plt.show()

    if isinstance(pltFile, str):
        pdf.close()

def plotVelTimeHistories(
                    T, # ([n1,], [n2,], ... [nN,])
                    U=None, # ([n1,], [n2,], ... [nN,])
                    V=None, # ([n1,], [n2,], ... [nN,])
                    W=None, # ([n1,], [n2,], ... [nN,])
                    dataLabels=None, # ("str1", "str2", ... "strN")
                    pltFile=None, # "/path/to/plot/file.pdf"
                    xLabel=r"$t [s]$",
                    yLabels=(r"$U(t)$",r"$V(t)$",r"$W(t)$"), 
                    xLimits='auto', # [tMin,tMax]
                    yLimits='auto', # ([Umin, Umax], [Vmin, Vmax], [Wmin, Wmax])
                    figSize=[15, 5],
                    alwaysShowFig=False
                    ):

    nCols = 3 - (U,V,W).count(None)
    N = len(U)
    nRows = N

    if isinstance(pltFile, str):
        pdf = PdfPages(pltFile)
    else:
        pdf = pltFile
    fig = plt.figure(figsize=figSize) 

    for i in range(N):
        m = 0
        if U is not None:
            plt.subplot(nRows,nCols,(i*nCols)+m+1)            
            plt.plot(T[i], U[i], color=ctlProf['col'][i], label=dataLabels[i])
            plt.xlabel(xLabel)
            plt.ylabel(yLabels[0])
            if xLimits is not None and not xLimits == 'auto':
                plt.xlim(xLimits)
            if yLimits is not None and not yLimits == 'auto':
                plt.ylim(yLimits[0])
            plt.legend()
            plt.grid()
            plt.tight_layout()
            m += 1

        if V is not None:
            plt.subplot(nRows,nCols,(i*nCols)+m+1)            
            plt.plot(T[i], V[i], color=ctlProf['col'][i], label=dataLabels[i])
            plt.xlabel(xLabel)
            plt.ylabel(yLabels[1])
            if xLimits is not None and not xLimits == 'auto':
                plt.xlim(xLimits)
            if yLimits is not None and not yLimits == 'auto':
                plt.ylim(yLimits[1])
            if m == 0:
                plt.legend()
            plt.grid()
            plt.tight_layout()
            m += 1

        if W is not None:
            plt.subplot(nRows,nCols,(i*nCols)+m+1)            
            plt.plot(T[i], W[i], color=ctlProf['col'][i], label=dataLabels[i])
            plt.xlabel(xLabel)
            plt.ylabel(yLabels[2])
            if xLimits is not None and not xLimits == 'auto':
                plt.xlim(xLimits)
            if yLimits is not None and not yLimits == 'auto':
                plt.ylim(yLimits[2])
            if m == 0:
                plt.legend()
            plt.grid()
            plt.tight_layout()
            m += 1

    if isinstance(pltFile, str):
        pdf.savefig(fig)
        plt.clf()
    if alwaysShowFig:
        plt.show()

    if isinstance(pltFile, str):
        pdf.close()    
    
def plotSpectra(
                freq, # ([n1,], [n2,], ... [nN,])
                Suu=None, # ([n1,], [n2,], ... [nN,])
                Svv=None, # ([n1,], [n2,], ... [nN,])
                Sww=None, # ([n1,], [n2,], ... [nN,])
                dataLabels=None, # ("str1", "str2", ... "strN")
                plotType:Literal['loglog','plot','semilogx','semilogy']='loglog', # {'loglog','plot','semilogx','semilogy'}
                pltFile=None, # "/path/to/plot/file.pdf"
                xLabel=r"$n [Hz]$",
                yLabels=(r"$S_{uu}$",r"$S_{vv}$",r"$S_{ww}$"), 
                xLimits='auto', # [nMin,nMax]
                yLimits='auto', # ([SuuMin, SuuMax], [SvvMin, SvvMax], [SwwMin, SwwMax])
                figSize=None,
                legendOnAll=False,
                drawXlineAt_rf1=False,
                alwaysShowFig=False,
                avoidZeroFreq=True,
                kwargs_plt=None,
                ):

    nCols = 3 - (Suu,Svv,Sww).count(None)
    N = len(Suu) if Suu is not None else len(Svv) if Svv is not None else len(Sww) if Sww is not None else 0
    nRows = 1
    i_n0 = 1 if avoidZeroFreq else 0
    figSize = [5*nCols, 5] if figSize is None else figSize
    dataLabels = ('data',)*N if dataLabels is None else dataLabels

    if kwargs_plt is None:
        kwargs_plt = [{}] * N

    if isinstance(pltFile, str):
        pdf = PdfPages(pltFile)
    else:
        pdf = pltFile
    fig = plt.figure(figsize=figSize)

    m = 0
    if Suu is not None:
        plt.subplot(nRows,nCols,m+1)
        for i in range(N):
            plot[plotType](freq[i][i_n0:], Suu[i][1:], label=dataLabels[i], **kwargs_plt[i])
        plt.xlabel(xLabel)
        plt.ylabel(yLabels[0])
        if drawXlineAt_rf1:
            plt.axvline(x=1,linestyle='--',ls='--',c='k',lw=1.4)
        if xLimits is not None and not xLimits == 'auto':
            plt.xlim(xLimits)
        if yLimits is not None and not yLimits == 'auto':
            plt.ylim(yLimits[0])
        if m == 0 or legendOnAll:
            plt.legend()
        plt.grid(which='both')
        plt.tight_layout()
        m += 1

    if Svv is not None:
        plt.subplot(nRows,nCols,m+1)
        for i in range(N):
            plot[plotType](freq[i][i_n0:], Svv[i][1:], label=dataLabels[i], **kwargs_plt[i])
        plt.xlabel(xLabel)
        plt.ylabel(yLabels[1])
        if drawXlineAt_rf1:
            plt.axvline(x=1,linestyle='--',ls='--',c='k',lw=1.4)
        if xLimits is not None and not xLimits == 'auto':
            plt.xlim(xLimits)
        if yLimits is not None and not yLimits == 'auto':
            plt.ylim(yLimits[1])
        if m == 0 or legendOnAll:
            plt.legend()
        plt.grid(which='both')
        plt.tight_layout()
        m += 1

    if Sww is not None:
        plt.subplot(nRows,nCols,m+1)
        for i in range(N):
            plot[plotType](freq[i][i_n0:], Sww[i][1:], label=dataLabels[i], **kwargs_plt[i])
        plt.xlabel(xLabel)
        plt.ylabel(yLabels[2])
        if drawXlineAt_rf1:
            plt.axvline(x=1,linestyle='--',ls='--',c='k',lw=1.4)
        if xLimits is not None and not xLimits == 'auto':
            plt.xlim(xLimits)
        if yLimits is not None and not yLimits == 'auto':
            plt.ylim(yLimits[2])
        if m == 0 or legendOnAll:
            plt.legend()
        plt.grid(which='both')
        plt.tight_layout()
        m += 1

    if isinstance(pltFile, str):
        pdf.savefig(fig)
        plt.clf()
    if alwaysShowFig:
        plt.show()

    if isinstance(pltFile, str):
        pdf.close()
    return fig

def plotSpectra____depricated(
                freq, # ([n1,], [n2,], ... [nN,])
                Spect, # ([n1,M], [n2,M], ... [nN,M])
                dataLabels=None, # ("str1", "str2", ... "strN")
                pltFile=None, # "/path/to/plot/file.pdf"
                xLabel='n [Hz]',
                yLabels=None, # ("str1", "str2", ... "str_m")
                xLimits='auto',
                yLimits='auto',
                figSize=[10, 4.8],
                alwaysShowFig=False,
                nCols=3
                ):
    
    N = len(freq)
    M = Spect[0].shape[1]
    nRows = int(np.ceil((M/nCols)))
    
    if dataLabels is None:
        dataLabels = (['Dataset_1'])
        for i in range(1,N):
            dataLabels += (['Dataset_'+str(i+1)])
    if yLabels is None:
        yLabels = (['Spect_1'])
        for i in range(1,M):
            yLabels += (['Spect_'+str(i+1)])

    if xLimits == 'auto':
        xLimits = [min([item for sublist in freq for item in sublist]), max([item for sublist in freq for item in sublist])]
    
    if isinstance(pltFile, str):
        pdf = PdfPages(pltFile)
    else:
        pdf = pltFile
    fig = plt.figure(figsize=figSize)

    for m in range(M):
        plt.subplot(nRows,nCols,m+1)
        for i in range(N):
            plt.loglog(freq[i], Spect[i][:,m], color=ctlSpect['col'][i], label=dataLabels[i])
        plt.xlabel(xLabel)
        plt.ylabel(yLabels[m])
        if xLimits is not None and not xLimits == 'auto':
            plt.xlim(xLimits)
        if yLimits is not None and not yLimits == 'auto':
            plt.ylim(yLimits[m])
        plt.legend()
        plt.grid()
    
    if isinstance(pltFile, str):
        pdf.savefig(fig)
        plt.clf()
    if alwaysShowFig:
        plt.show()

    if isinstance(pltFile, str):
        pdf.close()

def plotProfiles____depricated(Z, U,
                 TI = (),
                 L = (),
                 yLabel = "Z",
                 xLabels = ["U","Iu","Iv","Iw","xLu","xLv","xLw"],
                 pltCtl = ctlProf,
                 plotNames = (),
                 pltFile = None,
                 showPlot = True,
                 zLimit = 'max',
                 Ulims = [0,2],
                 TIlims = [0.4, 0.3, 0.3],
                 Llims = 'max'):
    clProf = ctl[0]
    clSpec = ctl[1]
    
    N = len(Z)
    
    if len(plotNames) == 0:
        plotNames = (['Dataset_1'])
        for i in range(1,N):
            plotNames += (['Dataset_'+str(i+1)])
    
    Ulabel = 'U'
    Zlabel = 'Z'
    
    if normalizer[0]:
        for i in range(N):
            Z[i] /= normalizer[1][i]
            U[i] /= normalizer[2][i]
            if len(L) > 0:
                L[i] /= normalizer[1][i]
            Ulabel = 'U/Uref'
            Zlabel = 'Z/Zref'
    if lim_Z == 'max':
        lim_Z = max([item for sublist in Z for item in sublist])
    
    if isinstance(pltFile, str):
        pdf = PdfPages(pltFile)
    else:
        pdf = pltFile
    fig = plt.figure() 

    for i in range(N):
        plt.plot(U[i],Z[i], color=clProf["col"][i], label=plotNames[i], ms=3, marker=clProf["marker"][i])
    plt.xlabel(Ulabel)
    plt.ylabel(Zlabel)
    plt.xlim(lim_U)
    plt.ylim([0, lim_Z])
    plt.legend()
    plt.grid()
    plt.title('mean U')
    pdf.savefig(fig)
    plt.clf()

    if len(TI) > 0:
        for i in range(N):
            plt.plot(TI[i][:,0],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iu')
        plt.ylabel(Zlabel)
        plt.xlim([0, lim_TI[0]])
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.grid()
        plt.title('Iu')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(TI[i][:,1],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iv')
        plt.ylabel(Zlabel)
        plt.xlim([0, lim_TI[1]])
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.grid()
        plt.title('Iv')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(TI[i][:,2],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iw')
        plt.ylabel(Zlabel)
        plt.xlim([0, lim_TI[2]])
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.grid()
        plt.title('Iw')
        pdf.savefig(fig)
        plt.clf()

    if len(L) > 0:
        for i in range(N):
            plt.plot(L[i][:,0],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLu')
        plt.ylabel(Zlabel)
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.title('xLu')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(L[i][:,1],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLv')
        plt.ylabel(Zlabel)
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.title('xLv')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(L[i][:,2],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLw')
        plt.ylabel(Zlabel)
        plt.ylim([0, lim_Z])
        plt.legend()
        plt.title('xLw')
        pdf.savefig(fig)
        plt.clf()

    if isinstance(pltFile, str):
        pdf.close()

#===============================================================================
#================================ CLASSES ======================================
#===============================================================================


