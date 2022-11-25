# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:39:45 2018

@author: Tsinu

** With scaling
"""
import numpy as np
import os

caseDir = os.getcwd()
funcName = 'samplePlanes'
sampleName = 'inflowPlane'
sampleDir = caseDir+'/postProcessing/'+funcName
if not os.path.exists(caseDir+'/constant/boundaryData/inlet'):
    os.makedirs(caseDir+'/constant/boundaryData/inlet')
tMin = 5.0
tMax = 100
p = 8  # precision for writing time directories

# all input parameters before this are un-scaled values
tScl = 1.0 #22.52252252  # time scale
lScl = 1.0 #100          # length scale
vScl = lScl/tScl    # velocity scale

timeNames = os.listdir(sampleDir)
times = np.asarray(timeNames)
times = times.astype(np.float)
tFirst = np.float64(min(x for x in times if x > tMin))

# Generate 'points' file
print("Generating point file ... \n")
rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+timeNames[0]+'/U_'+sampleName+'.raw')]
del rawData[0]
del rawData[0]
pointsFile = open(caseDir+'/constant/boundaryData/inlet/points', 'w')
pointsFile.write("\n"+str(len(rawData))+"\n(")
for r in rawData:
    rn = np.asarray(r.split(" ")[0:3],'float64')  # convert to numeric array
    rn = np.multiply(rn,lScl)  # scale
    pointsFile.write("\n( " + str(rn[0]) + " " + str(rn[1]) + " " + str(rn[2]) + " )")
pointsFile.write("\n)")
pointsFile.close()
print("Done! generating point file. \n")

# Generate velocity files
print("Converting velocity files ... \n")
for t in timeNames:
    if ((float(t) >= tMin) & (float(t) <= tMax)) :
        tNew = str(np.round((float(t)-tFirst)*tScl,p))
        if float(tNew) < 0.0:
            print("    t = "+str(t)+" --> "+str(tNew)+"\t\t------ warning! skipping negative time value.")
            continue
        if not os.path.exists(caseDir+'/constant/boundaryData/inlet/'+tNew):
            os.makedirs(caseDir+'/constant/boundaryData/inlet/'+tNew)
        else:
            print("    t = "+str(t)+" --> "+str(tNew)+"\t\t------ warning! skipping existing time value.")
            continue

        rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+t+'/U_'+sampleName+'.raw')]
        del rawData[0]
        del rawData[0]
        uFile = open(caseDir+'/constant/boundaryData/inlet/'+tNew+"/U", 'w')
        uFile.write("\n"+str(len(rawData))+"\n(")
        for r in rawData:
            rn = np.asarray(r.split(" ")[3:6],'float64')
            rn = np.multiply(rn,vScl)  # scale
            uFile.write("\n(" + str(rn[0]) + " " + str(rn[1]) + " " + str(rn[2]) + ")")
        uFile.write("\n)")
        uFile.close()
        print("    t = "+str(t)+" --> "+str(tNew))
print("Done! writing velocity files. \n")

print("Writing inflowDict file to "+ caseDir+"/constant/boundaryData/inflowDict. \n")
inflowDictFile = open(caseDir+'/constant/boundaryData/inflowDict', 'w')
inflowDictFile.write("""/*--------------------------------*- C++ -*----------------------------------*
| =========                |                                                 |
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration     | Version:  4.1                                   |
|   \  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""")
inflowDictFile.write("maxInflowTime\t\t"+str(np.round((max(times)-tFirst)*tScl,p))+";\n\n")
inflowDictFile.write("// ************************************************************************* //\n")
inflowDictFile.close()

print("Done! Writing inflowDict file to "+ caseDir+"/constant/boundaryData/inflowDict. \n")
