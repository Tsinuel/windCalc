# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:53:01 2020

@author: Tsinu
Description:
    Converts patchCloud data in distributed file form to the format of probe files. The set probes are 
    used to sample points on surfaces with a tolerance so that it won't miss because of small mesh 
    movement issues.
    
Potential improvement: 
    * 

Version history:
   1.0 
      date : April 6, 2020
      changes: first code
"""

import os
import time
import numpy as np
import shutil

sTime = time.time()
caseDir = os.getcwd()   #"D:/WorkFolder/Research_works/TTU/CFD_data/simData/ttuPSmBS02.0/postProcessing/roundOffTest" 

# inputs
inDir = caseDir+"/postProcessing/probeLineEFGH"
lineName = "EFGH"
field = "p"
removeOrig = True

outDir = caseDir+"/postProcessing/probes.tapsEFGH"
#
if os.path.exists(outDir):
    oldDirs = os.listdir(outDir)
    
else:
    os.mkdir(outDir)

inTimes = os.listdir(inDir)
inTimesN = np.asarray(inTimes,'float')
inTimesN = np.sort(inTimesN)

# write probe header
if not os.path.exists(outDir+"/"+str(inTimesN[0])):
    os.mkdir(outDir+"/"+str(inTimesN[0]))
outFile = open(outDir+"/"+str(inTimesN[0])+"/"+field,'w')
lines = [line.split() for line in open(inDir+"/"+str(inTimesN[0])+"/"+lineName+"_"+field+".xy")]
lines = np.asarray(lines,'float')
coord = lines[:,0:3]
for i in range(0,len(coord)):
    outFile.write("# Probe "+str(i)+" ("+str(coord[i,0])+" "+str(coord[i,1])+" "+str(coord[i,2])+")\n")
outFile.write("#\tProbe")
for i in range(0,len(coord)):
    outFile.write("\t"+str(i))
outFile.write("\n#\t Time\n")

for t in inTimesN:
    lines = [line.split() for line in open(inDir+"/"+'{:g}'.format(t)+"/"+lineName+"_"+field+".xy")]
    P = np.asarray(lines,'float')[:,3]
    outFile.write(str(t))
    for p in P:
        outFile.write("\t"+str(p))
    outFile.write("\n")
    if removeOrig:
        shutil.rmtree(inDir+"/"+'{:g}'.format(t))

outFile.close()
eTime = time.time()
print("Elapsed time = {:g} secs".format(eTime-sTime))
