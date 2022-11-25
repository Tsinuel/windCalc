# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:39:45 2018

@author: Tsinu
"""
import numpy as np
import os

caseDir = '/home/tgeleta/scratch/TTU/wtAsTarget/fullModel/FWT_z01_50_emp'
sampleName = 'inflowPlane'
sampleDir = caseDir+'/postProcessing/'+sampleName
os.makedirs(caseDir+'/constant/boundaryData/inlet')
tMin = 4.0
tMax = 30.0

timeNames = os.listdir(sampleDir)

# Generate 'points' file
rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+timeNames[0]+'/U_sectionPlane.raw')]
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
        rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+t+'/U_sectionPlane.raw')]
        del rawData[0]
        del rawData[0]
        os.makedirs(caseDir+'/constant/boundaryData/inlet/'+t)
        uFile = open(caseDir+'/constant/boundaryData/inlet/'+t+"/U", 'w')
        uFile.write("\n"+str(len(rawData))+"\n(")
        for r in rawData:
            uFile.write("\n(" + " ".join(r.split(" ")[3:6]) + ")")
        uFile.write("\n)")
        uFile.close()
