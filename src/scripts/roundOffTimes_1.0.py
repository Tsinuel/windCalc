# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:50:37 2018

@author: Tsinu
Description:
    It rounds off time directories to the specified precision. This is useful for rounding off unnecessary decimals from 
    OpenFOAM output (e.g. inflow, sample data etc.)
    
Potential improvement: 
    * 

Version history:
   1.0 
      date : Feb 28, 2020
      changes: first code
"""

import os
import time

sTime = time.time()
caseDir = "/home/tgeleta/scratch/TTU/prmStud/ttuPSf.C.0/postProcessing/wallPressure"
p = 5  # round off precision
print("Rounding off numeric directories in "+caseDir+" to "+str(p)+" decimal places ...\n")
allDir = os.listdir(caseDir)
for d in allDir:
    rd = "{:g}".format(round(float(d),p))
    if d != rd:
        os.rename(caseDir+"/"+d,caseDir+"/"+rd)
        print(rd+"\t<--\t"+d)
	
print("\nDone!!\n")

eTime = time.time()
print("Elapsed time = {:g} secs".format(eTime-sTime))

