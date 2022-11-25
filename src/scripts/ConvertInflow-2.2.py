# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:39:45 2018

@author: Tsinu
"""
import numpy as np
import os
import shutil

homeDir = os.getcwd()
if not os.path.exists(homeDir+'/GeneratedTimes'):
    os.makedirs(homeDir+'/GeneratedTimes')

appendTime=1  # 1 to append, 0 to replace
dt = 0.07
tPrev = 0.0
if appendTime == 1:
    # find the largest time value from existing directory names
    existingGenDirs = os.listdir(homeDir+'/GeneratedTimes')
    existingGenDirs = np.asarray(existingGenDirs)
    latestGenDir = -1
    if existingGenDirs.size > 0:
        latestGenDir = float(max(existingGenDirs))
        existingTimes = os.listdir(homeDir+'/GeneratedTimes/'+str(int(latestGenDir)))
        existingTimes = np.asarray(existingTimes)
        existingTimes = existingTimes.astype(float)
        if existingTimes.size > 0:
            tPrev = float(max(existingTimes))+dt
        #debugFile = open(homeDir+'/debugFile','w')
        #debugFile.write("\n".join(existingTimes))
        #debugFile.close()
    thisGenDir = str(int(latestGenDir + 1))
else:
    shutil.rmtree(homeDir+'/GeneratedTimes/')
    thisGenDir = '0'
print('Converting velocity for '+str(thisGenDir)+'th time ...')
uDirs = [line.rstrip('\n') for line in open(homeDir+'/directories')]
converted = [line.rstrip('\n') for line in open(homeDir+'/convStatus')]

os.makedirs(homeDir+'/GeneratedTimes/'+thisGenDir)
for uD in uDirs:
    print('Converting directory: '+uD+' ...')
    if int(converted[uDirs.index(uD)]) == 1:
        print('Skipped directory: '+uD + '\n')
        continue
    uDir = homeDir+'/'+uD
    uLines = [line.rstrip('\n') for line in open(uDir+'/U')]
    nt = int(uLines[1])
    nP = int(uLines[4])
    del uLines[0]
    del uLines[0]
    del uLines[0]
    del uLines[-1]
    
    nPl = nP+5
    for i in range(0,nt):
        U = uLines[(i*nPl):(i+1)*nPl]
        t = tPrev + float(i)*dt
        os.makedirs(homeDir+'/GeneratedTimes/'+thisGenDir+'/'+str(np.round(t,6)))
        Ufile = open(homeDir+'/GeneratedTimes/'+thisGenDir+'/'+str(np.round(t,6))+'/U', 'w')
        Ufile.write("\n".join(U))
        Ufile.close()
    tPrev = t + dt
    converted[uDirs.index(uD)] = str(1)
    print('Finished converting directory: '+uD)
    print('Finished at time: '+ str(np.round(t,6)) + '\n')

print('\n\nGenerated files have been placed in "'+homeDir+'\\GeneratedTimes\\'+thisGenDir)
statusFile = open(homeDir+'/convStatus', 'w')
statusFile.write("\n".join(converted))
statusFile.close()

