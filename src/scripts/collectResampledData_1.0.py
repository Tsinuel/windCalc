# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:20:27 2021

@author: Tsinu
Description: This code collects time history data in the form of .csv files 
    exported from paraview over multiple blocks 
"""

import numpy as np
import os
#import csv

caseDir = os.getcwd()

dictFile = [line.split() for line in open(caseDir+'/system/collectSampleDict')]
for d in dictFile:
    if d[0] == 'workDir':
        workDir = caseDir+'/'+d[1]+'/'
    elif d[0] == 'outDir':
        outDir = caseDir+'/'+d[1]+'/'
    elif d[0] == 'fileName':
        fileName = d[1]
    elif d[0] == 'nameTag':
        nameTag = d[1]
    elif d[0] == 'readFields':
        nFields = int(d[1])
        readFields = d[2:nFields+2]
    elif d[0] == 'writeFields':
        nFields = int(d[1])
        writeFields = d[2:nFields+2]
    elif d[0] == 'startTime':
        tStart = float(d[1])
    elif d[0] == 'dt':
        dt = float(d[1])
    else:
        continue
    
allFiles = os.listdir(workDir)
tInt = np.asarray([i.split('.',2)[1] for i in allFiles],int)
p = tInt.argsort()
tInt = tInt[p]
timeTag = str(tStart)+'_to_'+str(float(tInt[-1])*dt)

# read first file for point and field index extraction
csvFile = [line.split(",") for line in open(workDir+fileName+'.'+str(tInt[0])+'.csv')]
fieldIdx = []
for i, f in enumerate(readFields):
    fieldIdx.append(csvFile[0].index(f))
    field = np.asarray(csvFile[1:],float)[:,fieldIdx[-1]]
    field = np.insert(field,0,tStart)
    outFile = outDir + writeFields[i] + '_' + nameTag + timeTag
    with open(outFile,'a+') as file:
        np.savetxt(file, field, fmt='%.6f', newline=", ")
        file.write("\n")
    
ptIdx = [csvFile[0].index('"Points:0"'),csvFile[0].index('"Points:1"'),csvFile[0].index('"Points:2"\n')]

ptXYZ = np.asarray(csvFile[1:],float)[:,ptIdx]

np.savetxt(outDir+nameTag+'points', ptXYZ, fmt='%.6f',delimiter=',', header='X, Y, Z',comments='')

for t in tInt[1:]:
    csvFile = [line.split(",") for line in open(workDir+fileName+'.'+str(t)+'.csv')]
    for i, f in enumerate(readFields):
        field = np.asarray(csvFile[1:],float)[:,fieldIdx[i]]
        field = np.insert(field,0,tStart+float(t)*dt)
        outFile = outDir + writeFields[i] + '_' + nameTag + timeTag
        with open(outFile,'a+') as file:
            np.savetxt(file, field, fmt='%.6f', newline=", ")
            file.write("\n")
            file.close()
    

