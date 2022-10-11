# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:19:46 2022

@author: Tsinu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ctlProf = {
    "col":      ('r','g','b','k','m','c','r','g','b','k','m','c'),
    "lStyle":   ('-','-','-','-','-','-','--','--','--','--','--','--'),
    "marker":   ('o','s','v','^','>','<','x','d','*','1','2','3'),
    "mrkrSize": (2,2,2,2,2,2,2,2,2,2,2,2)
    }
ctlSpect = {
    "col":      ('r','g','b','k','m','c','r','g','b','k','m','c'),
    "lStyle":   ('-','-','-','-','-','-','--','--','--','--','--','--'),
    "marker":   ('o','s','v','^','>','<','x','d','*','1','2','3'),
    "mrkrSize": (2,2,2,2,2,2,2,2,2,2,2,2)
    }

def plotProfiles(Z,U,
                 TI=(), L=(), ctl=(ctlProf,ctlSpect),
                 plotNames=(),
                 normalizer=(False, [0.0], [0.0]),
                 pltFile='',
                 lim_Z = 'max',
                 lim_U = [0,2],
                 lim_TI=[0.4, 0.3, 0.3]):
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
    

def plotProfile(Z, X, zName, xName,
                dataName=None,
                file=None,
                zLim = 'max',
                xLim = None):
    plt.plot(X,Z,label=dataName)
    plt.xlabel(xName)
    plt.ylabel(zName)
    # plt.ylim([0, lim_Z])
    plt.legend()
    # plt.title('xLu')
    plt.show()

    pass