# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:50:37 2018

@author: Tsinu
Description:
    The processes time series of surface pressure from LES by calculating the mean, std, peak etc
    and write it in a format readable by paraview. It can also be used to visualize wind tunnel data
    as long as the mesh is available
    
Potential improvement: 
    the current method is not memory effective as it calculated everything
    after loading every time-step file. This can be improved by calculating the 
    required parameters as the time files load and discard the loaded data once 
    the required values are calculated. It may also be implemented in OpenFOAM 
    itself so that it can calculate the values along with the simulation.
    
    Non-uniform time distribution handling

Version history:
   1.0 
      date : May 7, 2018
      changes: first code
"""

import numpy as np
import scipy.stats as sc
import os
from math import ceil
#from shutil import copy2

caseDir = os.getcwd()
funcName = 'wallPressure'
patchName = 'building'
fieldName = 'p'
workDir = caseDir+'/postProcessing/'+funcName
cutOffTime = 0.0

times = os.listdir(workDir)
fileType = 'vtk' #['vtk', 'raw']

if fileType == 'raw':
    first = True
    for t in times:
        lines = [line.split() for line in open(workDir+'/'+t+'/'+fieldName+'_'+patchName+'.raw')]
        npts = int(lines[0][3])
        del lines[0]
        del lines[0]
        p = np.asarray(lines)
        p = p.astype(float)
        p = p[:,3]
        if first:
            P = p
            T = float(t)
            first = False
        else:
            P = np.column_stack((P,p)) 
            T = np.row_stack((T,float(t)))
    lines = [line for line in open(caseDir+'/postProcessing/sample.vtk')]
    nPoints = int((lines[4].split())[1])
    iPoly = nPoints+6
    nPoly = int(lines[iPoly].split()[1])
    iAtt = iPoly + nPoly + 4
elif fileType == 'vtk':
    first = True
    for t in times:
        if float(t) < cutOffTime:
            continue
        lines = [line.split() for line in open(workDir+'/'+t+'/'+fieldName+'_'+patchName+'.vtk')]
        nPoints = int(lines[4][1])
        iPoly = nPoints+6
        nPoly = int(lines[iPoly][1])
        iAtt = iPoly + nPoly + 4
        end = iAtt+ int(nPoly/10)
        p = lines[iAtt:end][:]
        p = np.asarray(p)
        p = np.reshape(p,p.size)
        p = p.astype(float)
        # add last line if it exists
        if p.size != nPoly:
            for m in lines[iAtt+ceil(nPoly/10)-1][:]:
                p = np.append(p, float(m))
        
        if first:
            P = p
            T = float(t)
            first = False
        else:
            P = np.column_stack((P,p)) 
            T = np.row_stack((T,float(t)))

P_mean = np.mean(P,1)
P_std = np.std(P,1)
P_rms =  np.sqrt(np.mean(P**2,1))
P_skewness = sc.skew(P,1)
P_kurtosis = sc.kurtosis(P,1)
# TO DO: 
#   Implement non-Gaussian peak analysis of Simiu and Sadek


meanFile = open(caseDir+'/postProcessing/pMean_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
stdFile = open(caseDir+'/postProcessing/pStd_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
rmsFile = open(caseDir+'/postProcessing/pRms_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
skewFile = open(caseDir+'/postProcessing/pSkewness_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
kurtFile = open(caseDir+'/postProcessing/pKurtosis_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')

lines = [line for line in open(workDir+'/'+t+'/'+fieldName+'_'+patchName+'.vtk')]

for l in lines[0:(iAtt-1)][:]:
    meanFile.write(l)
    stdFile.write(l)
    rmsFile.write(l)
    skewFile.write(l)
    kurtFile.write(l)

meanFile.write('p_Mean 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in P_mean:
    meanFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        meanFile.write('\n')
meanFile.close()

stdFile.write('p_Std 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in P_std:
    stdFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        stdFile.write('\n')
stdFile.close()

rmsFile.write('p_Rms 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in P_rms:
    rmsFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        rmsFile.write('\n')
rmsFile.close()

skewFile.write('p_Skewness 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in P_skewness:
    skewFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        skewFile.write('\n')
skewFile.close()

kurtFile.write('p_Kurtosis 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in P_kurtosis:
    kurtFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        kurtFile.write('\n')
kurtFile.close()


