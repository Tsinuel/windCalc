# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:39:45 2018

@author: Tsinu

** With scaling
"""
import numpy as np
import os

caseDir = os.getcwd()

#outDir = caseDir+'/constant/filteredBD/boundaryData'
#funcName = 'samplePlanes'
#sampleName = 'inflowPlane'
#tMin = 1.0
#tMax = 100
#p = 8  # precision for writing time directories
## all input parameters before this are un-scaled values
#tScl = 1.0 #22.52252252  # time scale
#lScl = 1.0 #100          # length scale

dictFile = [line.split() for line in open(caseDir+'/system/inflowConvertDict')]
for d in dictFile:
    if d[0] == 'funcName':
        funcName = d[1]
    elif d[0] == 'sampleName':
        sampleName = d[1]
    elif d[0] == 'unscaledOutDir':
        unscaledOutDir = d[1]
    elif d[0] == 'scaledOutDir':
        scaledOutDir = d[1]
    elif d[0] == 'writeUnscaledVelocity':
        writeUnscaledVelocity = d[1] == 'True'
    elif d[0] == 'writeScaledVelocity':
        writeScaledVelocity = d[1] == 'True'
    elif d[0] == 'tMin':
        tMin = float(d[1])
    elif d[0] == 'tMax':
        tMax = float(d[1])
    elif d[0] == 'dt':
        dt = float(d[1])
    elif d[0] == 'precision':
        p = int(d[1])
    elif d[0] == 'timeScale':
        tScl = float(d[1])
    elif d[0] == 'lengthScale':
        lScl = float(d[1])
    else:
        continue

vScl = lScl/tScl    # velocity scale
tMin = round(tMin/dt,0)*dt   # round off to the closest time step
tMax = round(tMax/dt,0)*dt

if (not writeUnscaledVelocity) and (not writeScaledVelocity):
    print("Both 'writeUnscaledVelocity' and 'writeScaledVelocity' in ./system/inflowConvertDict are set to 'False'. Script is exiting. Change one or both of them to 'True' to run the conversion.")
    exit()

sampleDir = caseDir+'/postProcessing/'+funcName
if (writeScaledVelocity and not os.path.exists(scaledOutDir+'/inlet')):
    os.makedirs(scaledOutDir+'/inlet')
if (writeUnscaledVelocity and not os.path.exists(unscaledOutDir+'/inlet')):
    os.makedirs(unscaledOutDir+'/inlet')

timeNames = os.listdir(sampleDir)
times = np.asarray(timeNames)
times = times.astype(np.float)
timeNames = [x for _,x in sorted(zip(times,timeNames))]

# Generate 'points' file
print("Generating point file ... \n")
rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+timeNames[0]+'/U_'+sampleName+'.raw')]
del rawData[0]
del rawData[0]
if writeScaledVelocity:
    S_pointsFile = open(scaledOutDir+'/inlet/points', 'w')
    S_pointsFile.write("\n"+str(len(rawData))+"\n(")
if writeUnscaledVelocity:
    U_pointsFile = open(unscaledOutDir+'/inlet/points', 'w')
    U_pointsFile.write("\n"+str(len(rawData))+"\n(")

for r in rawData:
    U_rn = np.asarray(r.split(" ")[0:3],'float64')  # convert to numeric array
    S_rn = np.round(np.multiply(U_rn,lScl),p)  # scale
    if writeScaledVelocity:
        S_pointsFile.write("\n( " + str(S_rn[0]) + " " + str(S_rn[1]) + " " + str(S_rn[2]) + " )")
    if writeUnscaledVelocity:
        U_pointsFile.write("\n( " + str(U_rn[0]) + " " + str(U_rn[1]) + " " + str(U_rn[2]) + " )")
if writeScaledVelocity:
    S_pointsFile.write("\n)")
    S_pointsFile.close()
if writeUnscaledVelocity:
    U_pointsFile.write("\n)")
    U_pointsFile.close()
print("Done! generating point file. \n")
print("   Files are written to: " + scaledOutDir)
print("                    and: " + unscaledOutDir + "\n")

# Generate velocity files
print("Converting velocity files ... \n")
print("  > t = original_time \t\t--> \tscaled_time \t\t------ Notes")
print("    t = original_time \t\t--> \tunscaled_time \t\t------ Notes\n")
runScaled = writeScaledVelocity
runUnscaled = writeUnscaledVelocity
for t in timeNames:
    if ((float(t) >= tMin) and (float(t) <= tMax)):
        S_tNew = str(np.round((float(t)-tMin)*tScl,p))
        U_tNew = str(np.round((float(t)-tMin),p))
        if float(U_tNew) < 0.0:
            print("    t = "+str(t)+" --> "+str(U_tNew)+", "+str(S_tNew)+"\t\t------ warning! skipping negative time value.")
            continue
        if writeScaledVelocity:
            if not os.path.exists(scaledOutDir+'/inlet/'+S_tNew):
                os.makedirs(scaledOutDir+'/inlet/'+S_tNew)
            else:
                if os.path.exists(scaledOutDir+'/inlet/'+str(S_tNew)+"/U"):
                    print(f"  > t_o = {str(t):<15}\t-->\tt_s = {str(S_tNew):<15}\t------ warning! skipping existing time value.")
                    runScaled = False
        else:
            runScaled = False
        
        if writeUnscaledVelocity:
            if not os.path.exists(unscaledOutDir+'/inlet/'+U_tNew):
                os.makedirs(unscaledOutDir+'/inlet/'+U_tNew)
            else:
                if os.path.exists(unscaledOutDir+'/inlet/'+str(U_tNew)+"/U"):
                    print(f"    t_o = {str(t):<15}\t-->\tt_s = {str(U_tNew):<15}\t------ warning! skipping existing time value.")
                    runUnscaled = False
        else:
            runUnscaled = False
#        print(str(runUnscaled)+"\t"+str(runScaled))
            
        if (not runScaled and not runUnscaled):
            runScaled = writeScaledVelocity
            runUnscaled = writeUnscaledVelocity
            continue

        rawData =  [line.rstrip('\n') for line in open(sampleDir+'/'+t+'/U_'+sampleName+'.raw')]
        del rawData[0]
        del rawData[0]
        if runScaled:
            S_uFile = open(scaledOutDir+'/inlet/'+S_tNew+"/U", 'w')
            S_uFile.write("\n"+str(len(rawData))+"\n(")
        if runUnscaled:
            U_uFile = open(unscaledOutDir+'/inlet/'+U_tNew+"/U", 'w')
            U_uFile.write("\n"+str(len(rawData))+"\n(")
        
        for r in rawData:
            U_rn = np.asarray(r.split(" ")[3:6],'float64')
            S_rn = np.multiply(U_rn,vScl)  # scale
            if runUnscaled:
                U_uFile.write("\n(" + str(U_rn[0]) + " " + str(U_rn[1]) + " " + str(U_rn[2]) + ")")
            if runScaled:
                S_uFile.write("\n(" + str(S_rn[0]) + " " + str(S_rn[1]) + " " + str(S_rn[2]) + ")")
        if runScaled:
            S_uFile.write("\n)")
            S_uFile.close()
            print(f"  > t_o = {str(t):<15}\t-->\tt_s = {str(S_tNew):<15}")
        if runUnscaled:
            U_uFile.write("\n)")
            U_uFile.close()
            print(f"    t_o = {str(t):<15}\t-->\tt_s = {str(U_tNew):<15}")
        
    runScaled = writeScaledVelocity
    runUnscaled = writeUnscaledVelocity

print("Done! writing velocity files. \n")

if writeScaledVelocity:
    print("Writing inflowDict file to "+ scaledOutDir +". \n")
    
    inflowDictFile = open(scaledOutDir+'/inflowDict', 'w')
    inflowDictFile.write("""/*--------------------------------*- C++ -*----------------------------------*
| =========                |                                                 |
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration     | Version:  4.1                                   |
|   \  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""")
    inflowDictFile.write("maxInflowTime\t\t"+str(np.round((max(times)-tMin)*tScl,p))+";\n\n")
    inflowDictFile.write("// ************************************************************************* //\n")
    inflowDictFile.close()
    
    print("Done! Writing inflowDict file to "+ scaledOutDir +". \n")

if writeUnscaledVelocity:
    print("Writing inflowDict file to "+ unscaledOutDir +". \n")
    
    inflowDictFile = open(unscaledOutDir+'/inflowDict', 'w')
    inflowDictFile.write("""/*--------------------------------*- C++ -*----------------------------------*
| =========                |                                                 |
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration     | Version:  4.1                                   |
|   \  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""")
    inflowDictFile.write("maxInflowTime\t\t"+str(np.round((max(times)-tMin),p))+";\n\n")
    inflowDictFile.write("// ************************************************************************* //\n")
    inflowDictFile.close()
    
    print("Done! Writing inflowDict file to "+ unscaledOutDir +". \n")
