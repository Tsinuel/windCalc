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
                 normalize_zRef_Uref=(False, [0.0], [0.0]),
                 pltFile=''):
    clProf = ctl[0]
    clSpec = ctl[1]
    
    N = len(Z)
    
    if len(plotNames) == 0:
        plotNames = (['Dataset_1'])
        for i in range(1,N):
            plotNames += (['Dataset_'+str(i+1)])
    
    Ulabel = 'U'
    Zlabel = 'Z'
    
    if normalize_zRef_Uref[0]:
        for i in range(N):
            Z[i] /= normalize_zRef_Uref[1][i]
            U[i] /= normalize_zRef_Uref[2][i]
            if len(L) > 0:
                L[i] /= normalize_zRef_Uref[1][i]
            Ulabel = 'U/Uref'
            Zlabel = 'Z/Zref'
    
    pdf = PdfPages(pltFile)       
    fig = plt.figure() 

    for i in range(N):
        plt.plot(U[i],Z[i], color=clProf["col"][i], label=plotNames[i], ms=3, marker=clProf["marker"][i])
    plt.xlabel(Ulabel)
    plt.ylabel(Zlabel)
    plt.legend()        
    plt.title('mean U')
    pdf.savefig(fig)
    plt.clf()

    if len(TI) > 0:
        for i in range(N):
            plt.plot(TI[i][:,0],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iu')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('Iu')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(TI[i][:,1],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iv')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('Iv')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(TI[i][:,2],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('Iw')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('Iw')
        pdf.savefig(fig)
        plt.clf()

    if len(L) > 0:
        for i in range(N):
            plt.plot(L[i][:,0],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLu')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('xLu')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(L[i][:,1],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLv')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('xLv')
        pdf.savefig(fig)
        plt.clf()

        for i in range(N):
            plt.plot(L[i][:,2],Z[i],color=clProf["col"][i],label=plotNames[i],ms=3, marker=clProf["marker"][i])
        plt.xlabel('xLw')
        plt.ylabel(Zlabel)
        plt.legend()
        plt.title('xLw')
        pdf.savefig(fig)
        plt.clf()

      
    pdf.close()
    