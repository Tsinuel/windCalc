# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:50:37 2018

@author: Tsinu
Description:
    The processes time series of surface pressure from LES by calculating the mean, std, peak etc
    and write it in a format readable by paraview. It can also be used to visualize wind tunnel data
    as long as the mesh is available
    
Potential improvement: 
    * the current method is not memory effective as it calculated everything
      after loading every time-step file. This can be improved by calculating the 
	  required parameters as the time files load and discard the loaded data once
	  the required values are calculated. It may also be implemented in OpenFOAM
	  itself so that it can calculate the values along with the simulation.
    * Implement non-Gaussian peak analysis of Simiu and Sadek
    * Non-uniform time distribution handling

Version history:
   1.0 
      date : May 7, 2018
      changes: first code
   1.1 
      date : Sep 17, 2019
      changes: 
         * calculate Cp instead of pressure
         * subtract ambient (like pitot of wind tunnel) pressure p0(requires measure pressure time 
           history in the free stream above the building in the computational domain) from measured 
           pressure p. This is specially important for CDRFG outputs because it imposes artificial 
           fluctuations all over the computational domain.

"""

import numpy as np
import scipy.stats as sc
import os
from math import ceil
#from shutil import copy2

caseDir = os.getcwd()
pFuncName = 'wallTester'  # the name used in 'controlDict' when the data is recorded
patchName = 'building'      # the patch name for the building surface
fieldName = 'p'             # the name of the field to process (default 'p')
pttFuncName = 'probes.V0'   # name of the profile with the 'pitot' point where p0 will be taken at
ptt_P0_ID = 57              # pitot point ID among the profile points
UrefFuncName = 'probes.x-80y20'  # name of the profile with normalizing velocity Uh
zRef = 12.19                # reference height (will be rounded off to the closest probe if it doesn't exist)
                            # for dynamic pressure calculation
rho = 1.125                 # Air density
cutOffTime = 0.0            # amount of time to chop-off at the beginning
fileType = 'vtk'            # recorded data type. Options:['vtk', 'raw']
writeCpFiles = True         # calculates the Cp for each time step and write the time history to a directory

# Read surface pressure data
pWorkDir = caseDir+'/postProcessing/'+pFuncName
times = os.listdir(pWorkDir)
if fileType == 'raw':
    first = True
    for t in times:
        lines = [line.split() for line in open(pWorkDir+'/'+t+'/'+fieldName+'_'+patchName+'.raw')]
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
        lines = [line.split() for line in open(pWorkDir+'/'+t+'/'+fieldName+'_'+patchName+'.vtk')]
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

# Read pitot pressure time history
pttWorkDir = caseDir+'/postProcessing/'+pttFuncName
pttTimes = os.listdir(pttWorkDir)
pttT_Dir = np.asarray(pttTimes,'float')
pttT_Dir = np.sort(pttT_Dir)
pttProbeData = []
for tProb in pttTimes:
    linesProbe = [line for line in open(pttWorkDir+'/'+tProb+'/'+'p')]
    probeCount = 0
    probeCoords = []
    for l in linesProbe:
        if (l[0:7] == '# Probe'):
            probeCount = probeCount+1
            probeCoords.append(np.asarray(((l.split('(')[1]).split(')')[0]).split(' '),'float'))
        else:
            break
    
    pttProbeData.append([dd.split() for dd in linesProbe[probeCount+2:-1]])
            
probeCoords = np.asarray(probeCoords)
#pttProbeData = np.concatenate([np.asarray(dd,'float') for dd in pttProbeData],axis=0)
ptttProbeData = np.asarray(pttProbeData[0],'float')
for i in range(1,len(pttProbeData)):
   ptttProbeData = np.append(ptttProbeData, np.asarray(pttProbeData[i],'float'),axis=0)
pttProbeData = ptttProbeData
pttT = pttProbeData[:,0]
pttProbeData = pttProbeData[:,1:]
tMinIdx = np.abs(pttT - cutOffTime).argmin()
tMaxIdx = tMinIdx + len(T)
p0 = pttProbeData[tMinIdx:tMaxIdx,ptt_P0_ID]

# Read reference velocity time history
UrefWorkDir = caseDir+'/postProcessing/'+UrefFuncName
UrefTimes = os.listdir(UrefWorkDir)
UrefT_Dir = np.asarray(UrefTimes,'float')
UrefT_Dir = np.sort(UrefT_Dir)
UrefProbeData = []
for tProb in UrefTimes:
    linesProbe = [line for line in open(UrefWorkDir+'/'+tProb+'/'+'U')]
    probeCount = 0
    probeCoords = []
    for l in linesProbe:
        if (l[0:7] == '# Probe'):
            probeCount = probeCount+1
            probeCoords.append(np.asarray(((l.split('(')[1]).split(')')[0]).split(' '),'float'))
        else:
            break
    
    UrefProbeData.append([dd.split() for dd in linesProbe[probeCount+2:-1]])
            
probeCoords = np.asarray(probeCoords)

UrefT = []
UrefData = []
for urefT in UrefProbeData:
    for urefLine in urefT:
        UrefT.append(urefLine[0])
        UrefDataLine = []        
        for u in range(1,len(urefLine),3):
            UrefDataLine.append([urefLine[u].split('(')[1], urefLine[u+1], urefLine[u+2].split(')')[0]])
        UrefData.append(np.asarray(UrefDataLine,'float'))

UrefT = np.asarray(UrefT,'float')
tMinIdx = np.abs(UrefT - cutOffTime).argmin()
tMaxIdx = tMinIdx + len(T)
U = np.asarray(UrefData,'float')
refIdx = np.abs(probeCoords[:,2] - zRef).argmin()
Uref = U[tMinIdx:tMaxIdx,refIdx,:]
UrefMean = np.mean(Uref[:,0])

# Calculate Cp time histories of the surface pressure
Cp = (P-p0)/(0.5*rho*UrefMean**2)

# Process the pressure data
Cp_mean = np.mean(Cp,1)
Cp_std = np.std(Cp,1)
Cp_rms =  np.sqrt(np.mean(Cp**2,1))
Cp_skewness = sc.skew(Cp,1)
Cp_kurtosis = sc.kurtosis(Cp,1)

# Write-out the output files
meanFile = open(caseDir+'/postProcessing/CpMean_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
stdFile = open(caseDir+'/postProcessing/CpStd_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
rmsFile = open(caseDir+'/postProcessing/CpRms_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
skewFile = open(caseDir+'/postProcessing/CpSkewness_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')
kurtFile = open(caseDir+'/postProcessing/CpKurtosis_'+str(cutOffTime)+'_'+str(t)+'.vtk','w')

lines = [line for line in open(pWorkDir+'/'+t+'/'+fieldName+'_'+patchName+'.vtk')]

for l in lines[0:(iAtt-1)][:]:
    meanFile.write(l)
    stdFile.write(l)
    rmsFile.write(l)
    skewFile.write(l)
    kurtFile.write(l)

meanFile.write('Cp_Mean 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in Cp_mean:
    meanFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        meanFile.write('\n')
meanFile.close()

stdFile.write('Cp_Std 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in Cp_std:
    stdFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        stdFile.write('\n')
stdFile.close()

rmsFile.write('Cp_Rms 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in Cp_rms:
    rmsFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        rmsFile.write('\n')
rmsFile.close()

skewFile.write('Cp_Skewness 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in Cp_skewness:
    skewFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        skewFile.write('\n')
skewFile.close()

kurtFile.write('Cp_Kurtosis 1 ' + str(nPoly) + ' float'+'\n')
count = 1
for p in Cp_kurtosis:
    kurtFile.write(str(p)+' ')
    count = count+1
    if np.remainder(count,10) == 0:
        kurtFile.write('\n')
kurtFile.close()

# Write Cp time history file into a directory
if writeCpFiles:
    if not os.path.exists(caseDir+'/postProcessing/'+pFuncName+'_Cp'):
        os.makedirs(caseDir+'/postProcessing/'+pFuncName+'_Cp')
    CpDir = caseDir+'/postProcessing/'+pFuncName+'_Cp'
    for i in range(0,len(T)):
        t = T[i]
        CpFile = open(caseDir+'/postProcessing/'+pFuncName+'_Cp/Cp_'+str(t[0])+'.vtk','w')
        for l in lines[0:(iAtt-1)][:]:
            CpFile.write(l)
        CpFile.write('Cp 1 ' + str(nPoly) + ' float'+'\n')
        count = 1
        for p in Cp[:,i]:
            CpFile.write(str(p)+' ')
            count = count+1
            if np.remainder(count,10) == 0:
                CpFile.write('\n')
        CpFile.close()
    
