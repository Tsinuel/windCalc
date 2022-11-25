# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:39:45 2018

@author: Tsinu
"""
import numpy as np
import os

caseDir = os.getcwd()
funcName = 'samplePlanes'
sampleName = 'inflowPlane'
sampleDir = caseDir+'/postProcessing/'+funcName
if not os.path.exists(caseDir+'/constant/boundaryData/inlet'):
    os.makedirs(caseDir+'/constant/boundaryData/inlet')
tMin = 3.5
tMax = 100
p = 5  # precision for writing time directories

timeNames = os.listdir(sampleDir)
times = np.asarray(timeNames)
times = times.astype(np.float)
tFirst = np.float64(min(x for x in times if x > tMin))

# Generate 'points' file
rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+timeNames[0]+'/U_'+sampleName+'.raw')]
del rawData[0]
del rawData[0]
pointsFile = open(caseDir+'/constant/boundaryData/inlet/points', 'w')
pointsFile.write("\n"+str(len(rawData))+"\n(")
for r in rawData:
    pointsFile.write("\n(" + " ".join(r.split(" ")[0:3]) + ")")
pointsFile.write("\n)")
pointsFile.close()

# Generate velocity files
for t in timeNames:
    if ((float(t) >= tMin) & (float(t) <= tMax)) :
        tNew = str(np.round(float(t)-tFirst,p))
        if float(tNew) < 0.0:
            continue
        if not os.path.exists(caseDir+'/constant/boundaryData/inlet/'+tNew):
            os.makedirs(caseDir+'/constant/boundaryData/inlet/'+tNew)
        else:
            continue

        rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+t+'/U_'+sampleName+'.raw')]
        del rawData[0]
        del rawData[0]
        uFile = open(caseDir+'/constant/boundaryData/inlet/'+tNew+"/U", 'w')
        uFile.write("\n"+str(len(rawData))+"\n(")
        for r in rawData:
            uFile.write("\n(" + " ".join(r.split(" ")[3:6]) + ")")
        uFile.write("\n)")
        uFile.close()
