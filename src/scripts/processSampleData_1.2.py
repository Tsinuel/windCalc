# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:50:37 2018

@author: Tsinu
Description:
    The processes time series of surface pressure from LES by calculating the mean, std, peak etc
    and write it in a format readable by paraview. It can also be used to visualize wind tunnel data
    as long as the mesh is available
    
Potential improvement: 
    * the current method is not memory effective as it calculated everything after loading every 
      time-step file. This can be improved by calculating the required parameters as the time files 
      load and discard the loaded data once the required values are calculated. It may also be 
      implemented in OpenFOAM itself so that it can calculate the values along with the simulation.
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
   1.1.1
      date : Sep 30, 2019
      changes: 
         * writes output into one file and multiple fields
		 * output log file
   1.1.2
      date : Feb 4, 2020
      changes: 
         * round off time before reading the time directories {lines 108 to 111}
   1.2
      date : Feb 4, 2020
      changes: 
         * program name changed from 'processBldgCpStats' to 'processSampleData'
         * Added capability of processing velocity data. 

"""

import numpy as np
import scipy.stats as sc
import os
from math import ceil
import time

sTime = time.time()
caseDir = os.getcwd()
dictFile = [line.split() for line in open(caseDir+'/system/calcCpDict')]
for d in dictFile:
    if d[0] == 'pFuncName':
        pFuncName = d[1]
    elif d[0] == 'patchName':
        patchName = d[1]
    elif d[0] == 'fieldName':
        fieldName = d[1]
    elif d[0] == 'pttFuncName':
        pttFuncName = d[1]
    elif d[0] == 'fixedP_0Value':
        fixedP_0Value = float(d[1])
    elif d[0] == 'ptt_P0_ID':
        ptt_P0_ID = int(d[1])
    elif d[0] == 'UrefFuncName':
        UrefFuncName = d[1]
    elif d[0] == 'zRef':
        zRef = float(d[1])
    elif d[0] == 'rho':
        rho = float(d[1])
    elif d[0] == 'writeCpFiles':
        if d[1] == 'True':
            writeCpFiles = True
        elif d[1] == 'False':
            writeCpFiles = False
        else:
            raise Exception("writeCpFiles option in the input dictionary should either be True or False. (Case senstive)")
    elif d[0] == 'uFuncName':
        uFuncName = d[1]
    elif d[0] == 'uSectNames':
        uSectNames = d[2:2+int(d[1])]
    elif d[0] == 'timeRange':
        timeRange = [float(d[1]), float(d[2])]
    elif d[0] == 'fileType':
        fileType = d[1]
    elif d[0] == 'timeTol':
        timeTol = float(d[1])
    else:
        continue

# Read surface pressure data
if pFuncName != '***':
    pWorkDir = caseDir+'/postProcessing/'+pFuncName
    print('>> Started processing surface pressure\n\n')
    
    times = os.listdir(pWorkDir)
    if len(times) < 2:
        raise Exception('Less than two time directories found in: ' + pWorkDir + '. A minimum of 2 instantaneous data files have to be provided')
    tSurf = np.asarray(times,'float')
    tSurf.sort()
    roundDigits = -int(np.log10(timeTol))
    for t in times:
        if str(round(float(t), roundDigits)) != t:
            os.rename(pWorkDir+'/'+str(t), pWorkDir+'/{:g}'.format(round(float(t), roundDigits)))
    dtSurf = round(tSurf[1]-tSurf[0],roundDigits)
    for i in range(1,len(tSurf)):
        if abs(abs(tSurf[i]-tSurf[i-1]) - dtSurf) > timeTol:
            print('WARNING! Non-uniform time step detected in the surface data. The first time step will be used. '+ str(np.round(tSurf[i]-tSurf[i-1],9)) +'\n')
    print('>> Found t = ' + str(tSurf[0]) + ' to ' + str(tSurf[-1]) + ' at ' + str(dtSurf)+'\n\n')
    # truncate time
    tMinIdx = np.abs(tSurf - timeRange[0]).argmin()
    tMaxIdx = np.abs(tSurf - timeRange[1]).argmin()
    tSurf = tSurf[tMinIdx:tMaxIdx]
    print('   Using t = ' + str(tSurf[0]) + ' to ' + str(tSurf[-1]) + ' at ' + str(dtSurf)+'\n\n')
    
    # works with vtk files only
    
    first = True
    n = len(tSurf)
    c = np.ceil(n/100)
    print("   Reading surface files ...")
    for i in range(0,n):
        lines = [line.split() for line in open(pWorkDir+'/'+'{:g}'.format(tSurf[i])+'/'+fieldName+'_'+patchName+'.vtk')]
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
            T = tSurf[i]
            first = False
        else:
            P = np.column_stack((P,p)) 
            T = np.row_stack((T,tSurf[i]))
        if (c%(i+1)):
            print(" t = " + str(tSurf[i]) + "\t\t" + str(round(float(i/n)*100, 3)) + "%" )
    print("\r{0}%".format(int(float(i)/n*100)))
    print(" Done! Reading data from file. \n\n")
    
    # Read pitot pressure time history
    print("   Reading p0 data ...")
    if (pttFuncName != "***"):  # read P0 from monitored data
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
        if len(pttProbeData) < len(tSurf):
            print('WARNING! Available reference pressure data is shorter than the requested time range. The reference pressure will be zero for the extra time range.')
        p0 = tSurf*0
        for i in range(0,len(tSurf)):
            idx = np.abs(pttT - tSurf[i]).argmin()
        #    print("idx = "+str(idx))
            if abs(pttT[idx] - tSurf[i]) > timeTol:
                print('WARNING! No matching time of P0 data is found for surface data at '+ str(tSurf[i]) + '. The closest value is: ' + str(pttT[idx]))
            p0[i] = pttProbeData[idx,ptt_P0_ID]
        print(">> Reference ('pitot') pressure, P0. \n   P0_mean = " + str(np.mean(p0)) + "\n   Values: \n")
        for p in p0:
        	print(str(p)+" \t")
        print("\n\n")
    else:  # set P0 to fixedP_0Value defined in the dictionary
        p0 = fixedP_0Value
        print("   Using a fixed P0 value of "+str(fixedP_0Value)+"\n\n")
        
    # Read reference velocity time history
    print("   Reading Uref data ...")
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
    U = np.asarray(UrefData,'float')
    refIdx = np.abs(probeCoords[:,2] - zRef).argmin()
    
    if len(UrefT) < len(tSurf):
        print('WARNING! Available reference velocity data is shorter than the requested time range. The velocity will be zero for the extra time range.')
    Uref = np.zeros([len(tSurf), 3])
    for i in range(0,len(tSurf)):
        idx = np.abs(UrefT - tSurf[i]).argmin()
    #    print("idx = "+str(idx))
        if abs(UrefT[idx] - tSurf[i]) > timeTol:
            print('WARNING! No matching time of Uref data is found for surface data at '+ str(tSurf[i]) + '. The closest value is: ' + str(UrefT[idx]))
        Uref[i,:] = U[idx,refIdx,:]
    UrefMean = np.mean(Uref[:,0])
    print(">> Reference velocity, Uref. \n   UrefMean = "+ str(UrefMean) +"\n   Zref = " + str(probeCoords[refIdx,2]) + " \n   Values: \n")
    for u in Uref:
    	print(str(u)+" \t")
    print("\n")
    print("   Rho = "+ str(rho) + "\n\n")
    
    # Calculate Cp time histories of the surface pressure
    # print("   Array sizes\n      P: "+ str(P.shape) + "   p0:"+ str(p0.shape) + ". \n\n")
    Cp = (P-p0)/(0.5*rho*UrefMean**2)
    print(">> Done calculating Cp. \n\n")
    
    # Process the pressure data
    Cp_mean = np.mean(Cp,1)
    Cp_std = np.std(Cp,1)
    Cp_rms =  np.sqrt(np.mean(Cp**2,1))
    Cp_skewness = sc.skew(Cp,1)
    Cp_kurtosis = sc.kurtosis(Cp,1)
    Cp_obsrvMin = np.amin(Cp,axis=1)
    Cp_obsrvMax = np.amax(Cp,axis=1)
    Cp_obsrvPeak = np.zeros(Cp_obsrvMax.shape)
    for i in range(0,len(Cp_obsrvMax)):
        Cp_obsrvPeak[i] = max(Cp_obsrvMin[i],Cp_obsrvMax[i],key=abs)
    numOfProperties = 8  # number of properties to be written to the statistics vtk file
    
    # Write-out the output files
    print("Writing calculated values to stat file ...\n")
    cPfile = open(caseDir+'/postProcessing/Cp_' + pFuncName + '_' + patchName + '_' + '{:g}'.format(tSurf[0])+'_'+str(tSurf[-1])+'.vtk','w')
    
    lines = [line for line in open(pWorkDir+'/{:g}'.format(tSurf[0])+'/'+fieldName+'_'+patchName+'.vtk')]
    for l in lines[0:(iAtt-2)][:]:
        cPfile.write(l)
    cPfile.write('FIELD attributes ' + str(numOfProperties) + '\n')
    
    cPfile.write('Cp_Mean 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_mean:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing mean Cp\n")
    
    cPfile.write('Cp_Std 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_std:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing std Cp\n")
    
    cPfile.write('Cp_Rms 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_rms:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing rms Cp\n")
    
    cPfile.write('Cp_Skewness 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_skewness:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing skewness Cp\n")
    
    cPfile.write('Cp_Kurtosis 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_kurtosis:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing kurtosis Cp\n")
    
    cPfile.write('Cp_obsrvMin 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_obsrvMin:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing observed Min Cp\n")
    
    cPfile.write('Cp_obsrvMax 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_obsrvMax:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing Cp_obsrvMax Cp\n")
    
    cPfile.write('Cp_obsrvPeak 1 ' + str(nPoly) + ' float'+'\n')
    count = 1
    for p in Cp_obsrvPeak:
        cPfile.write(str(p)+' ')
        count = count+1
        if np.remainder(count,10) == 0:
            cPfile.write('\n')
    cPfile.write('\n')
    print("Done! writing Cp_obsrvPeak Cp\n")
    
    cPfile.close()
    
    # Write Cp time history file into a directory
    print("Writing instantaneous Cp files ...\n")
    if writeCpFiles:
        if not os.path.exists(caseDir+'/postProcessing/'+pFuncName+'_Cp'):
            os.makedirs(caseDir+'/postProcessing/'+pFuncName+'_Cp')
        CpDir = caseDir+'/postProcessing/'+pFuncName+'_Cp'
        w = int(np.log10(len(T)))+1
        for i in range(0,len(T)):
            t = T[i]
            CpFile = open(caseDir+'/postProcessing/'+pFuncName+'_Cp/Cp_'+str(i).zfill(w)+'.vtk','w')
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
        print("Done! writing instantaneous Cp files.\n")
    else:
        print("Skipped writing instantaneous Cp files. This option is suppressed by the input dictionary. \n"+
              "        Change 'writeCpFiles' to 'True' to enable it.\n")
    print("Cleaning memory ...")
    try:
        del Cp
        del Cp_kurtosis
        del Cp_mean
        del Cp_obsrvMax
        del Cp_obsrvMin
        del Cp_obsrvPeak
        del P
        del U
        del lines
    except:
        print("Warning! One or more of Cp calculation variables could not be deleted.")    
        
# Process velocity sections
if uFuncName != '***':
    print("*************************************************************************************\n")
    print("Processing velocity data!\n")
    
    uWorkDir = caseDir+'/postProcessing/'+uFuncName
    times = os.listdir(uWorkDir)
    if len(times) < 2:
        raise Exception('Less than two time directories found in: ' + uWorkDir + '. A minimum of 2 instantaneous data files have to be provided')

    tSurf = np.asarray(times,'float')
    tSurf.sort()
    roundDigits = -int(np.log10(timeTol))
    for t in times:
        if str(round(float(t), roundDigits)) != t:
            os.rename(uWorkDir+'/'+str(t), uWorkDir+'/{:g}'.format(round(float(t), roundDigits)))
    dtSurf = round(tSurf[1]-tSurf[0],roundDigits)
    for i in range(1,len(tSurf)):
        if abs(abs(tSurf[i]-tSurf[i-1]) - dtSurf) > timeTol:
            print('WARNING! Non-uniform time step detected in the velocity section data. The first time step will be used. '+ str(np.round(tSurf[i]-tSurf[i-1],9)) +'\n')
    print('>> Found t = ' + str(tSurf[0]) + ' to ' + str(tSurf[-1]) + ' at ' + str(dtSurf)+'\n\n')
    # truncate time
    tMinIdx = np.abs(tSurf - timeRange[0]).argmin()
    tMaxIdx = np.abs(tSurf - timeRange[1]).argmin()
    tSurf = tSurf[tMinIdx:tMaxIdx]
    print('   Using t = ' + str(tSurf[0]) + ' to ' + str(tSurf[-1]) + ' at ' + str(dtSurf)+'\n\n')


    for uSectName in uSectNames:
        print("Processing velocity data for: " + uSectName + "\n")
    
    	    # works with vtk files only
        first = True
        n = len(tSurf)
        c = np.ceil(n/100)
        print("   Reading surface files ...")
        for i in range(0,n):
            lines = [line.split() for line in open(uWorkDir+'/'+'{:g}'.format(tSurf[i])+'/U_'+uSectName+'.vtk')]
            nPoints = int(lines[4][1])
            iPoly = nPoints+6
            nPoly = int(lines[iPoly][1])
            iAtt = iPoly + nPoly + 4
            nCells = int(lines[iAtt-1][2])
            u = lines[iAtt:]
            u = np.asarray(u,'float')
            
            if first:
                U = u
                T = tSurf[i]
                first = False
            else:
                U = np.dstack([U,u])
                T = np.row_stack((T,tSurf[i]))
            if (c%(i+1)):
                print("\r{0} %".format(int(float(i/n)*100)))
        print("\r{0} %".format(int(float(i)/n*100)))
        print(" Done! Reading data from file. \n\n")
    
        # Compute the statistics of the current time
        U_mean = np.mean(U,2)
        U_std = np.std(U,2)
        U_skewness = sc.skew(U,2)
        U_kurtosis = sc.kurtosis(U,2)
        numOfProperties = 4
    
        # Write-out the output files
        print("Writing calculated values to stat file ...\n")
        uFile = open(caseDir+'/postProcessing/U_' + uSectName + '_' + '{:g}'.format(tSurf[0])+'_'+str(tSurf[-1])+'.vtk','w')
        
        lines = [line for line in open(uWorkDir+'/{:g}'.format(tSurf[0])+'/U_'+uSectName+'.vtk')]
        for l in lines[0:(iAtt-2)][:]:
            uFile.write(l)
        uFile.write('FIELD attributes ' + str(numOfProperties) + '\n')
        
        uFile.write('U_Mean 3 ' + str(nCells) + ' float \n')
        count = 1
        for ui in range(0,nCells-1):
            uFile.write(str(U_mean[ui,0]) + ' ' + str(U_mean[ui,1]) + ' ' + str(U_mean[ui,2]) + '\n')
            count = count+1
        print("Done! writing mean U\n")
        
        uFile.write('U_std 3 ' + str(nCells) + ' float \n')
        count = 1
        for ui in range(0,nCells-1):
            uFile.write(str(U_std[ui,0]) + ' ' + str(U_std[ui,1]) + ' ' + str(U_std[ui,2]) + '\n')
            count = count+1
        print("Done! writing std U\n")
        
        uFile.write('U_skewness 3 ' + str(nCells) + ' float \n')
        count = 1
        for ui in range(0,nCells-1):
            uFile.write(str(U_skewness[ui,0]) + ' ' + str(U_skewness[ui,1]) + ' ' + str(U_skewness[ui,2]) + '\n')
            count = count+1
        print("Done! writing skewness of U\n")
        
        uFile.write('U_kurtosis 3 ' + str(nCells) + ' float \n')
        count = 1
        for ui in range(0,nCells-1):
            uFile.write(str(U_kurtosis[ui,0]) + ' ' + str(U_kurtosis[ui,1]) + ' ' + str(U_kurtosis[ui,2]) + '\n')
            count = count+1
        print("Done! writing kurtosis of U\n")
        
        uFile.close()


	
print("\nDone!!\n")

eTime = time.time()
print("Elapsed time = {:g} secs".format(eTime-sTime))
