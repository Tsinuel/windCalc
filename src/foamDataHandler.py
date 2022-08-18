# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:49:36 2022

@author: Tsinu
"""
import numpy as np
import os
import warnings
import pandas as pd
import scipy.interpolate as scintrp
import matplotlib.pyplot as plt

import windLoadCaseProcessors as wProc
import windPlotters as wPlt

__tol_time = 1e-7
__tol_data = 1e15


__showPlots = False
__precision = 6

# Inflow data handlers

def readProfiles(file,requiredFields):
    if ~os.path.exists(file):
        FileNotFoundError()
    profiles = pd.read_csv(file)
    
    if 'Z' not in profiles:
        if 'z' in profiles:
            profiles.rename(columns={'z':'Z'})
        else:
            raise Exception("All profiles must have Z column. Check profile:"+ file)
    
    for f in requiredFields:
        if f not in profiles:
            raise Exception("The profile column '"+ f +"' is listed in 'requiredFields' but not found in the profile: "+file)
    
    return profiles

def readBDformattedFile(file):
    # file in the format of boundaryData. Can be points or velocity
    entries = [line.split() for line in open(file)]
    entries = list(filter(None, entries))
    entries = entries[2:-1]
    
    if len(entries[0]) == 3:
        entries = [[s[0].strip('('), s[1], s[2].strip(')')] for s in entries]
        entries = pd.DataFrame.from_records(entries)
    elif len(entries[0]) == 5: 
        entries = pd.DataFrame.from_records(entries)
        entries = entries.drop([0,4], axis=1)
    
    return entries.astype('float64')

def writeBDformattedFile(file,vect):
    f = open(file,"w")
    f.write(str(len(vect))+"\n")
    f.close()
    f = open(file,"a")
    f.write("(\n")
    np.savetxt(f,vect,fmt='(%1.8g %1.8g %1.8g)',delimiter=' ')
    f.write(")")
    f.close()    

def getProfileScaleFactor(origProf,targProf,scaleBy):
    def plotProf(z,orig,targ,f,name):
        if not __showPlots:
            return
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(orig,z,'k-',label='orig')
        ax1.plot(targ,z,'r-',label='target')
        plt.xlabel(name)
        plt.ylabel('Z')
        ax1.legend()
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(f,z,'k+-',label='Target/original')
        plt.xlabel('Correction factor ('+name+')')
        plt.ylabel('Z')
        ax2.legend()
        plt.show()

    Z = np.unique(np.sort(np.append(np.asarray(origProf.Z), np.asarray(targProf.Z))))
    factor = Z
    cNames = ['Z']
    
    if 'U' in scaleBy:
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
        orig = intrp(Z)
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.U),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('U')
        plotProf(Z,orig,targ,f,'U')
    
    if 'Iu' in scaleBy:
        # intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),bounds_error=False,fill_value=(origProf.Iu[0],origProf.iloc[-1].Iu))
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),fill_value='extrapolate')
        orig = intrp(Z)
        # intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iu),bounds_error=False,fill_value=(targProf.Iu[0],targProf.iloc[-1].Iu))
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iu),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iu')
        plotProf(Z,orig,targ,f,'Iu')
    
    if 'Iv' in scaleBy:
        # intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),bounds_error=False,fill_value=(origProf.Iv[0],origProf.iloc[-1].Iv))
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),fill_value='extrapolate')
        orig = intrp(Z)
        # intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iv),bounds_error=False,fill_value=(targProf.Iv[0],targProf.iloc[-1].Iv))
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iv),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iv')
        plotProf(Z,orig,targ,f,'Iv')
    
    if 'Iw' in scaleBy:
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iw),bounds_error=False,fill_value=(origProf.Iw[0],origProf.iloc[-1].Iw))
        orig = intrp(Z)
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iw),bounds_error=False,fill_value=(targProf.Iw[0],targProf.iloc[-1].Iw))
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iw')
        plotProf(Z,orig,targ,f,'Iw')
    
    factor = pd.DataFrame(np.transpose(factor),columns=cNames)
    return factor

def scaleVelocity(UofT,VofT,WofT,
                  Z, scaleBy,
                  origUofZ, fUofZ, fIuOfZ, fIvOfZ, fIwOfZ):
    
    U_old = origUofZ(Z)
    u_old = UofT - U_old
    
    U_new = fUofZ(Z)*U_old + fIuOfZ(Z)*u_old
    
    V_new = fIvOfZ(Z)*VofT
    
    W_new = fIwOfZ(Z)*WofT
        
    if __showPlots:
        fig1 = plt.figure()
        ax = fig1.add_subplot(1,1,1)
        ax.plot(UofT,Z,'.k',label='orig',ms=1)
        ax.plot(U_new,Z,'.r',label='scaled',ms=1)
        ax.plot(U_old,Z,'xb',label='meanU-old',ms=1)
        ax.plot(fUofZ(Z)*U_old,Z,'xg',label='meanU-new',ms=1)
        plt.xlabel('U')
        plt.ylabel('Z')
        ax.legend()
        plt.show()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,1,1)
        ax.plot(VofT,Z,'.k',label='orig',ms=1)
        ax.plot(V_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('V')
        plt.ylabel('Z')
        ax.legend()
        plt.show()
        
        fig3 = plt.figure()
        ax = fig3.add_subplot(1,1,1)
        ax.plot(WofT,Z,'.k',label='orig',ms=1)
        ax.plot(W_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('W')
        plt.ylabel('Z')
        ax.legend()
        plt.show()

    return U_new, V_new, W_new

def readInflowDict(infFile):
    inflDict = {}
    dictFile = [line.split() for line in open(infFile)]
    
    for d in dictFile:
        if d == []:
            continue
        elif d[0] == 'inflowDir':
            inflDict["inflowDir"] = d[1]
        elif d[0] == 'inflowFormat':
            inflDict["inflowFormat"] = d[1]
        elif d[0] == 'outputDir':
            inflDict["outputDir"] = d[1]
        elif d[0] == 'targProfFile':
            inflDict["targProfFile"] = d[1]
        elif d[0] == 'origProfFile':
            inflDict["origProfFile"] = d[1]
        elif d[0] == 'dt':
            inflDict["dt"] = float(d[1])
        elif d[0] == 'precision':
            inflDict["precision"] = int(d[1])
        elif d[0] == 'lengthScale':
            inflDict["lScl"] = float(d[1])
        elif d[0] == 'timeScale':
            inflDict["tScl"] = float(d[1])
        elif d[0] == 'scaleBy':
            inflDict["scaleBy"] = d[1:]
        else:
            continue
        
    return inflDict

def scaleInflowData(caseDir):
    # caseDir = 'D:/tempData_depot/simData_CandC/ttuPSpcOP15.7'
    inflDict = readInflowDict(caseDir+'/system/scaleInflowDict')
    
    origProf = readProfiles(inflDict["origProfFile"], inflDict["scaleBy"])
    targProf = readProfiles(inflDict["targProfFile"], inflDict["scaleBy"])
    origProf.Z = origProf.Z*inflDict["lScl"]
    profSclFctr = getProfileScaleFactor(inflDict["origProf"], targProf, inflDict["scaleBy"])
    
    origUofZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
    # origIuOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),fill_value='extrapolate')
    # origIvOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),fill_value='extrapolate')
    # origIwOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iw),bounds_error=False,fill_value=(origProf.Iw[0],origProf.iloc[-1].Iw))
    
    fUofZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.U),fill_value='extrapolate')
    fIuOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iu),fill_value='extrapolate')
    fIvOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iv),fill_value='extrapolate')
    fIwOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iw),bounds_error=False,fill_value=(profSclFctr.Iw[0],profSclFctr.iloc[-1].Iw))
    
    if inflDict["inflowFormat"] == 'boundaryData':
        ptsFile = inflDict["inflowDir"] + '/inlet/points'
        points = readBDformattedFile(ptsFile)
        points.columns = ['X','Y','Z']
        
        inletDir = inflDict["inflowDir"]+'/inlet'
        times = [ name for name in os.listdir(inletDir) if os.path.isdir(os.path.join(inletDir, name)) ]
        times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
        
        if not os.path.exists(inflDict["outputDir"]+'/inlet'):
            os.makedirs(inflDict["outputDir"]+'/inlet')
        file = inflDict["outputDir"]+'/inlet/points'
        vect = np.transpose(np.asarray((points.X, points.Y, points.Z)))
        writeBDformattedFile(file,vect)
    
        for t in times:
            tNew = str(round(t[1]*inflDict["tScl"],__precision))
            if os.path.exists(inflDict["outputDir"]+'/inlet/'+tNew):
                print("--- skipping \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)
                continue
            else:
                os.makedirs(inflDict["outputDir"]+'/inlet/'+tNew)
            velFile = inflDict["inflowDir"]+'/'+t[0]+'/U'
            if not os.path.exists(velFile):
                print("--- File not found! " + velFile)
                continue
            vel = readBDformattedFile(velFile)
            vel.columns = ['u','v','w']
            
            U,V,W = scaleVelocity(vel.u, vel.v, vel.w,
                                    points.Z, inflDict["scaleBy"],
                                    origUofZ, fUofZ, fIuOfZ, fIvOfZ, fIwOfZ)
            file = inflDict["outputDir"]+'/inlet/'+tNew+'/U'
            vect = np.transpose(np.asarray((U, V, W)))
            writeBDformattedFile(file,vect)
            
            print("Scaling \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)
    
        
    elif inflDict["inflowFormat"] == 'vtk':
        raise NotImplementedError("vtk inflow format not implemented")
    else:
        raise NotImplementedError("Unknown inflow data format.")
    
    inflowDictFile = open(caseDir+'/constant/inflowDict', 'w')
    inflowDictFile.write("""/*--------------------------------*- C++ -*----------------------------------*
    | =========                |                                                 |
    | \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \    /   O peration     | Version:  4.1                                   |
    |   \  /    A nd           | Web:      www.OpenFOAM.org                      |
    |    \/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    """)
    inflowDictFile.write("maxInflowTime\t\t"+str(tNew)+";\n\n")
    inflowDictFile.write("// ************************************************************************* //\n")
    inflowDictFile.close()

def extractSampleProfileFromInflow(inletDir,figFile,tMax):
    from scipy import spatial
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    points = readBDformattedFile(inletDir + '/points')
    points.columns = ['X','Y','Z']
    
    coords = []
    for x, y in zip(points.Y, points.Z):
        coords.append((x,y))
    inletPlane = spatial.KDTree(coords)
    
    profPts = []    
    for z in np.linspace(min(points.Z), max(points.Z), 100):
        profPts.append((0,z))
        
    idx = inletPlane.query(profPts)[1]
    Z = np.asarray(points.Z[idx],dtype=float)
    
    times = [ name for name in os.listdir(inletDir) if os.path.isdir(os.path.join(inletDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    
    velOfT = []
    for t in times:
        velFile = inletDir+'/'+t[0]+'/U'
        vel = readBDformattedFile(velFile)
        vel.columns = ['u','v','w']
        
        if len(velOfT) == 0:
            velOfT = np.reshape(np.asarray(vel.values[idx,:]),[1,-1,3])
        else:
            velOfT = np.append(velOfT, np.reshape(np.asarray(vel.values[idx,:]),[1,-1,3]), axis=0)
        print("t = "+t[0])
        if t[1] > tMax:
            break

    Umean = np.mean(velOfT[:,:,0],axis=0)
    Iu = np.std(velOfT[:,:,0],axis=0)/Umean
    Iv = np.std(velOfT[:,:,1],axis=0)/Umean
    Iw = np.std(velOfT[:,:,2],axis=0)/Umean

    pdf = PdfPages(figFile)       
    fig = plt.figure(figsize=[15.0,12.0]) 
    
    # plt.plot(points.Y,points.Z, 'xk')
    # plt.plot(Z*0.0,Z, 'sr')
    # pdf.savefig(fig)
    # plt.clf()

    plt.figure()
    plt.plot(Umean,Z, '-xk')
    plt.xlabel('U')
    plt.ylabel('Z')
    pdf.savefig(fig)
    plt.clf()

    plt.figure()
    plt.plot(Iu,Z, '-xk',label='Iu')
    plt.plot(Iv,Z, '-dr',label='Iv')
    plt.plot(Iw,Z, '-+b',label='Iw')
    plt.xlabel('TI')
    plt.ylabel('Z')
    plt.legend()
    pdf.savefig(fig)
    plt.clf()

    pdf.close()


# Probe readers and related functions

def __readProbe_singleT(file,field):
    """
    Read time-history data of a field from a single OpenFOAM probe file.

    Parameters
    ----------
    file : str
        The file name of the probe data.
    field : str
        The field to be read. e.g., 'p', 'U'

    Returns
    -------
    probes : np.array(float)
        Sampling points (i.e., probe locations).
    time : np.array(float)
        Time vector.
    data : np.array(float)
        The time history data of the field recorded with the probes.

    """
    probes = np.zeros([0])
    data = np.zeros([0])
    time  = np.zeros([0])
    with open(file, "r") as f:
        for line in f:
            line = line.replace('(','')
            line = line.replace(')','')
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.split()
                    prb_i = np.reshape(np.asarray(line[3:6],dtype=float),[1,3])
                    if len(probes) == 0:
                        probes = prb_i
                    else:
                        probes = np.append(probes,prb_i,axis=0)
                else:
                    continue
            else:
                line = line.split()
                if field == 'p':
                    if len(line[1:]) < len(probes): # inclomplete line
                        continue
                    d = np.reshape(np.asarray(line[1:],dtype=(float)),[1,-1])
                elif field == 'U':
                    if len(line[1:])/3 < len(probes): # inclomplete line
                        continue
                    d = np.reshape(np.asarray(line[1:],dtype=(float)),[1,-1,3])

                if len(data) == 0:
                    data = d
                else:
                    data = np.append(data, d,axis=0)
                time = np.append(time,float(line[0]))
   
    return probes, time, data

def readProbe(probeName, postProcDir, field, trimTimeSegs=[[0,0]], trimOverlap=True, 
              shiftTimeToZero=True, removeOutOfDomainProbes=True):
    """
    Read probe time-history data from an OpenFOAM case.

    Parameters
    ----------
    probeName : str
        The probe name given in the controlDict.
    postProcDir : str
        The postProcessing directory with the data.
    field : str
        The field name to read from the probe files. e.g., 'p', 'U'
    trimOverlap : bool, optional
        Whether or not to remove time overlap between consecutive records that 
        commonly happens in OpenFOAM when the simulation is re-run from 
        latestTime. The default is True.
    shiftTimeToZero : bool, optional
        Whether or not to shift the time vector in such a way that the time 
        starts from zero. It is significant if there has been some trimming 
        and clipping on the time history. The default is True.
    removeOutOfDomainProbes : bool, optional
        Whether or not to remove probes which lie outside the computational 
        domain indicated by absurdly high values like 1e+300. They do not 
        normally result in errors. The default is True.

    Returns
    -------
    probes : np.array(float)    : shape [nPoints, components]
        Sampling points (i.e., probe locations).
    T : np.array(float)    : shape [nTime, ]
        Time vector.
    data : np.array(float)    : shape [nTime, nPoints, components] for U, [nTime, nPoints] for p
        The time history data of the field recorded with the probes.

    Examples
    --------
    >>> caseDir = "D:/Tsinu/..../windCalc/data/exampleOFcase/"
    >>> postProcDir = caseDir+"postProcessing/"
    >>> probeName = "probes.tapsABCD"
    >>> probes,time,p = readProbe(probeName, postProcDir, "p")
    >>> np.shape(probes)
    (46, 3)
    >>> np.shape(time)
    (3661,)
    >>> np.shape(p)
    (3661, 46)
    
    """
    probeDir = postProcDir+probeName+"/"
    times = [ name for name in os.listdir(probeDir) if os.path.isdir(os.path.join(probeDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    
    probes = np.zeros([0])
    T = np.zeros([0])
    data = np.zeros([0])
    
    for t in times:
        fileName = probeDir+t[0]+"/"+field
        
        (probes,time,d) = __readProbe_singleT(fileName, field)
        
        if len(T) == 0:
            T = time
            data = d
        else:
            T = np.append(T,time)
            data = np.append(data,d,axis=0)

    if trimOverlap:
        T,idx = np.unique(T,return_index=True)
        data = data[idx,:]
        
    dt = np.diff(T)
    if max(dt)-min(dt) > __tol_time:
        warnings.warn("\n\nWARNING! Non-uniform time step detected in '"+probeName+"'.\n\n")
        
    if shiftTimeToZero:
        dt = np.mean(dt)
        T = np.linspace(0, (len(T)-1)*dt, num=len(T))
        
    if removeOutOfDomainProbes:
        if field == 'p':
            removeIdx = np.where(np.prod(abs(data) > __tol_data,axis=0))
        elif field == 'U':
            removeIdx = np.where(np.prod(np.prod(abs(data) > __tol_data,axis=2),axis=0))
        probes = np.delete(probes,removeIdx,0)
        data = np.delete(data,removeIdx,1)
        
    return probes, T, data
    
def processVelProfile(caseDir, probeName, targetProfile,
                          writeToDataFile=False,
                          showPlots=False,
                          exportPlots=True):
    
    # caseDir = "D:/OneDrive - The University of Western Ontario/Documents/PhD/Thesis/CodeRepositories/windCalc/data/exampleOFcase/"
    postProcDir = caseDir+"postProcessing/"
    # probeName = "probes.tapsABCD"
    # probeName = "probes.V1"
    figFile = caseDir+"figures.pdf"
    
    probes,time,vel = readProbe(probeName, postProcDir, "U", trimTimeSegs=[[0,0.5]])
    
    Z = probes[:,2]
    dt = np.mean(np.diff(time))
    zRef = 0.08
    
    ref_U_TI_L, Z, U, TI, L, freq, Spect_H, Spect_Others = wProc.processVelProfile(Z,vel,dt,zRef)
    
    ref_U_TI_L_wt, Z_wt, U_wt, TI_wt, L_wt = wProc.readVelProfile(targetProfile, zRef)
    
    wPlt.plotProfiles((Z, Z_wt),
                      (U, U_wt),
                     TI=(TI, TI_wt), 
                     L=(L, L_wt),
                     normalize_zRef_Uref=(False, [zRef, zRef], [ref_U_TI_L[0], ref_U_TI_L_wt[0]]),
                     plotNames=('LES','BLWT'),
                     pltFile=figFile)
    
    
 