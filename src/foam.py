# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:49:36 2022

@author: Tsinuel Geleta
"""
import numpy as np
import os
import warnings
import pandas as pd
import scipy.interpolate as scintrp
import matplotlib.pyplot as plt
from typing import List,Literal,Dict,Tuple,Any

from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

import wind
# import windLoadCaseProcessors__toBeRemoved as wProc
import windPlotting as wPlt

__tol_time = 1e-7
__tol_data = 1e15


__showPlots = False
__precision = 6

#===============================================================================
#=============================  FUNCTIONS  =====================================
#===============================================================================

#-----------------------  Inflow data handlers  --------------------------------
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

def getProfileScaleFactor(Z_intrp, origProf, targProf, scaleBy, scalingFile=None, figFile=''):
    
    def plotProf(Z_calc, orig, targ, fCalc, Z_intrp, fIntrp, name, pdf):
        if pdf == '':
            return
        fig = plt.figure() 
        fig.add_subplot(1,2,1)
        plt.plot(orig,Z_calc,'k-',label='orig')
        plt.plot(targ,Z_calc,'r-',label='target')
        plt.xlabel(name)
        plt.ylabel('Z')
        plt.legend()
        
        fig.add_subplot(1,2,2)
        plt.plot(fCalc,Z_calc,'dk',label='Calculated')
        plt.plot(fIntrp[::50],Z_intrp[::50],'.r',label='Interpolated',markersize=2)
        plt.xlabel('Correction factor (Targ/Orig)')
        plt.ylabel('Z')
        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.clf()
        
    if scalingFile is not None:
        scaleFctrs = readProfiles(scalingFile, ["U", "Iu", "Iv", "Iw"])
        fU_intrp = scintrp.interp1d(scaleFctrs.Z, scaleFctrs.U, fill_value='extrapolate')
        fIu_intrp = scintrp.interp1d(scaleFctrs.Z, scaleFctrs.Iu, fill_value='extrapolate')
        fIv_intrp = scintrp.interp1d(scaleFctrs.Z, scaleFctrs.Iv, fill_value='extrapolate')
        fIw_intrp = scintrp.interp1d(scaleFctrs.Z, scaleFctrs.Iw, fill_value='extrapolate')

    else:
        Z_calc = np.unique(np.sort(np.append(np.asarray(origProf.Z), np.asarray(targProf.Z))))
        
        if 'U' in scaleBy:
            intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
            orig = intrp(Z_calc)
            intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.U),fill_value='extrapolate')
            targ = intrp(Z_calc)
            fU_calc = targ/orig
            intrp = scintrp.interp1d(Z_calc, fU_calc, fill_value='extrapolate')
            fU_intrp = intrp(Z_intrp)
            plotProf(Z_calc,orig,targ,fU_calc, Z_intrp, fU_intrp,'U',figFile)
        else:
            fU_intrp = np.ones(Z_intrp.shape())
        
        if 'Iu' in scaleBy:
            intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),fill_value='extrapolate')
            orig = intrp(Z_calc)
            intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iu),fill_value='extrapolate')
            targ = intrp(Z_calc)
            fIu_calc = targ/orig
            intrp = scintrp.interp1d(Z_calc, fIu_calc, fill_value='extrapolate')
            fIu_intrp = intrp(Z_intrp)
            plotProf(Z_calc,orig,targ,fIu_calc, Z_intrp, fIu_intrp,'Iu',figFile)
        else:
            fIu_intrp = np.ones(Z_intrp.shape())
    
        if 'Iv' in scaleBy:
            intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),fill_value='extrapolate')
            orig = intrp(Z_calc)
            intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iv),fill_value='extrapolate')
            targ = intrp(Z_calc)
            fIv_calc = targ/orig
            intrp = scintrp.interp1d(Z_calc, fIv_calc, fill_value='extrapolate')
            fIv_intrp = intrp(Z_intrp)
            plotProf(Z_calc,orig,targ,fIv_calc, Z_intrp, fIv_intrp,'Iv',figFile)
        else:
            fIv_intrp = np.ones(Z_intrp.shape())
        
        if 'Iw' in scaleBy:
            intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iw),bounds_error=False,fill_value=(origProf.Iw[0],origProf.iloc[-1].Iw))
            orig = intrp(Z_calc)
            intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iw),bounds_error=False,fill_value=(targProf.Iw[0],targProf.iloc[-1].Iw))
            targ = intrp(Z_calc)
            fIw_calc = targ/orig
            intrp = scintrp.interp1d(Z_calc, fIw_calc, fill_value='extrapolate')
            fIw_intrp = intrp(Z_intrp)
            plotProf(Z_calc,orig,targ,fIw_calc, Z_intrp, fIw_intrp,'Iw',figFile)
        else:
            fIw_intrp = np.ones(Z_intrp.shape())
    
    return fU_intrp,fIu_intrp,fIv_intrp,fIw_intrp

def scaleVelocity(UofT,VofT,WofT,
                  Z, scaleBy,
                  origUofZ, fU, fIu, fIv, fIw,
                  figFile=''):
    
    U_MeanCorr = fU*UofT
    U_old = fU*origUofZ(Z)
    U_new = U_old + fIu*(U_MeanCorr-U_old)
    
    V_new = fU*fIv*VofT
    
    W_new = fU*fIw*WofT
        
    if len(figFile) > 0:
        pdf = PdfPages(figFile)       
        fig = pdf.figure() 
        fig.add_subplot(1,1,1)
        plt.plot(UofT,Z,'.k',label='orig',ms=1)
        plt.plot(U_new,Z,'.r',label='scaled',ms=1)
        # plt.plot(U_old,Z,'xb',label='meanU-old',ms=1)
        # plt.plot(fUofZ(Z)*U_old,Z,'xg',label='meanU-new',ms=1)
        plt.xlabel('U')
        plt.ylabel('Z')
        plt.legend()
        # pdf.savefig(fig)
        # plt.clf()
        
        fig.add_subplot(1,1,1)
        plt.plot(VofT,Z,'.k',label='orig',ms=1)
        plt.plot(V_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('V')
        plt.ylabel('Z')
        plt.legend()
        # pdf.savefig(fig)
        # plt.clf()
        
        fig.add_subplot(1,1,1)
        plt.plot(WofT,Z,'.k',label='orig',ms=1)
        plt.plot(W_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('W')
        plt.ylabel('Z')
        plt.legend()
        pdf.savefig(fig)
        plt.clf()
        
        pdf.close()

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
        elif d[0] == 'scaleFactorFile':
            inflDict["scaleFactorFile"] = d[1]
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

def getClosest2DcoordsTo(X,Y,Zin):
    from scipy import spatial
    coords = []
    for x, y in zip(X, Y):
        coords.append((x,y))
    inletPlane = spatial.KDTree(coords)
    
    pts = []    
    for z in Zin:
        pts.append((0,z))
        
    idx = inletPlane.query(pts)[1]
    
    return idx

def scaleInflowData(caseDir,tMin,tMax,H,writeInflow=True,smplName=''):
    
    # caseDir = 'D:/tempData_depot/simData_CandC/ttuPSpcOP15.7'
    inflDict = readInflowDict(caseDir+'/system/scaleInflowDict')
    if smplName == '':
        smplName = 'inflowScaling_'+str(tMin)+'_to_'+str(tMax)
    if not os.path.exists(inflDict["outputDir"]+'/inlet'):
        os.makedirs(inflDict["outputDir"]+'/inlet')
    pdfDoc = PdfPages(inflDict["outputDir"]+'/'+smplName+'.pdf')
    
    
    ptsFile = inflDict["inflowDir"] + '/inlet/points'
    points = readBDformattedFile(ptsFile)
    points.columns = ['X','Y','Z']
    
    Z_smpl = np.linspace(0.001,max(points.Z)-0.01,100)
    idx = getClosest2DcoordsTo(points.Y,points.Z,Z_smpl)
    Z_smpl = np.asarray(points.Z[idx],dtype=float)

    
    
    # prep for scaling
    origProf = readProfiles(inflDict["origProfFile"], inflDict["scaleBy"])
    targProf = readProfiles(inflDict["targProfFile"], inflDict["scaleBy"])
    # origProf.Z = origProf.Z*inflDict["lScl"]
    fU,fIu,fIv,fIw = getProfileScaleFactor(points.Z, origProf, targProf, inflDict["scaleBy"],scalingFile=inflDict["scaleFactorFile"], figFile=pdfDoc)
    
    origUofZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
    
    
    inletDir = inflDict["inflowDir"]+'/inlet'
    times = [ name for name in os.listdir(inletDir) if os.path.isdir(os.path.join(inletDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    
    file = inflDict["outputDir"]+'/inlet/points'
    vect = np.transpose(np.asarray((points.X, points.Y, points.Z)))
    if writeInflow:
        writeBDformattedFile(file,vect)

    Vsmpl_in = []
    Vsmpl_out = []
    for t in times:
        if (t[1] < tMin or t[1] > tMax):
            continue
        tNew = str(round(t[1]*inflDict["tScl"],__precision))
        if writeInflow:
            if os.path.exists(inflDict["outputDir"]+'/inlet/'+tNew):
                print("--- skipping \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)
                continue
            else:
                os.makedirs(inflDict["outputDir"]+'/inlet/'+tNew)
        velFile = inletDir+'/'+t[0]+'/U'
        if not os.path.exists(velFile):
            print("--- File not found! " + velFile)
            continue
        vel = readBDformattedFile(velFile)
        vel.columns = ['u','v','w']

        U,V,W = scaleVelocity(vel.u, vel.v, vel.w,
                                points.Z, inflDict["scaleBy"],
                                origUofZ, fU(points.Z), fIu(points.Z), fIv(points.Z), fIw(points.Z))
        file = inflDict["outputDir"]+'/inlet/'+tNew+'/U'
        vect = np.transpose(np.asarray((U, V, W)))
        
        if len(Vsmpl_in) == 0:
            Vsmpl_in = np.reshape(np.asarray(vel.values[idx,:]),[1,-1,3])
            Vsmpl_out = np.reshape(np.asarray(vect[idx,:]),[1,-1,3])
        else:
            Vsmpl_in = np.append(Vsmpl_in, np.reshape(np.asarray(vel.values[idx,:]),[1,-1,3]), axis=0)
            Vsmpl_out = np.append(Vsmpl_out, np.reshape(np.asarray(vect[idx,:]),[1,-1,3]), axis=0)
        
        if writeInflow:
            writeBDformattedFile(file,vect)
        
        print("Scaling \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)
            
    sfix = datetime.now().strftime("%Y-%m-%d_%H:%M")
    sfix = str(tMin)+'_to_'+str(tMax)
    dt = times[1][1]-times[0][1]
    velUnscaled = wind.profile(name="Unscaled",Z=Z_smpl,dt=dt,H=H,
                    UofT=np.transpose(Vsmpl_in[:,:,0]),
                    VofT=np.transpose(Vsmpl_in[:,:,1]),
                    WofT=np.transpose(Vsmpl_in[:,:,2]))
    velUnscaled.writeToFile(inflDict["outputDir"],nameSuffix=sfix,writeTH=True)

    dt = round(times[1][1] * inflDict["tScl"], __precision)  - round(times[0][1] * inflDict["tScl"], __precision)
    velScaled = wind.profile(name="Scaled",Z=Z_smpl,dt=dt,H=H,
                    UofT=np.transpose(Vsmpl_out[:,:,0]),
                    VofT=np.transpose(Vsmpl_out[:,:,1]),
                    WofT=np.transpose(Vsmpl_out[:,:,2]))
    velScaled.writeToFile(inflDict["outputDir"],nameSuffix=sfix,writeTH=True)
 
    if writeInflow:
        inflowDictFile = open(inflDict["outputDir"]+'/inflowDict', 'w')
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

    pdfDoc.close()

    return velUnscaled, velScaled

def extractSampleProfileFromInflow(inletDir,outPath,figFile,tMax,H):
    
    points = readBDformattedFile(inletDir + '/points')
    points.columns = ['X','Y','Z']
    
    Z = np.linspace(0.001,max(points.Z)-0.01,100)
    idx = getClosest2DcoordsTo(points.Y,points.Z,Z)
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
    
    np.savetxt(outPath+"_UofT.csv",velOfT[:,:,0],delimiter=",")
    np.savetxt(outPath+"_VofT.csv",velOfT[:,:,1],delimiter=",")
    np.savetxt(outPath+"_WofT.csv",velOfT[:,:,2],delimiter=",")
    # ref_U_TI_L, Z, U, TI, L, freq, Spect_H, Spect_Others = wProc.processVelProfile(Z, 
    #                                                                                velOfT, 
    #                                                                                times[1][1]-times[0][1], 
    #                                                                                H)
    
    # wPlt.plotProfiles([Z],
    #                   [U],
    #                   TI=[TI], 
    #                   L=[L],
    #                   plotNames=(),
    #                   pltFile=figFile)
    pass

#===============================================================================
#=================  Probe readers and related functions  =======================
#===============================================================================
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

    @author: Tsinuel Geleta
    """
    probes = np.zeros([0])
    data = np.zeros([0])
    time  = np.zeros([0])
    with open(file, "r") as f:
        for line in f:
            line = line.replace('(','')
            line = line.replace(')','')
            line = line.replace(',',' ')
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

def readProbe(probeName, postProcDir, field, trimTimeSegs:List[List[float]]=None, trimOverlap=True, 
              shiftTimeToZero=True, removeOutOfDomainProbes=True, showLog=True):
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
        Whether or not to remove time overlap between consecutive records that commonly happens in OpenFOAM when the simulation is re-run from latestTime. The default is True.
    shiftTimeToZero : bool, optional
        Whether or not to shift the time vector in such a way that the time starts from zero. It is significant if there has been some trimming and clipping on the time history. The default is True.
    removeOutOfDomainProbes : bool, optional
        Whether or not to remove probes which lie outside the computational domain indicated by absurdly high values like 1e+300. They do not normally result in errors. The default is True.

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
    
    @author: Tsinuel Geleta
    """
    probeDir = postProcDir+probeName+"/"
    times = [ name for name in os.listdir(probeDir) if os.path.isdir(os.path.join(probeDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    
    probes = np.zeros([0])
    T = np.zeros([0])
    data = np.zeros([0])
    
    for t in times:
        fileName = probeDir+t[0]+"/"+field
        if showLog:
            print(f"           reading {field} from: {fileName}")
        
        (probes,time,d) = __readProbe_singleT(fileName, field)
        
        if len(T) == 0:
            T = time
            data = d
        else:
            T = np.append(T,time)
            data = np.append(data,d,axis=0)

    if trimOverlap:
        if showLog:
            print("    Trimming overlapping times.")
        T,idx = np.unique(T,return_index=True)
        data = data[idx,:]
        
    dt = np.diff(np.unique(np.sort(T)))
    if max(dt)-min(dt) > __tol_time:
        msg = f"WARNING! Non-uniform time step detected in '{probeName}'. The highest difference in time step is: {max(dt)-min(dt)}"
        warnings.warn(msg)
        
    if trimTimeSegs is not None:
        if showLog:
            print(f"   Trimming times: {trimTimeSegs}")
        idx = []
        for seg in trimTimeSegs:
            if not len(seg) == 2:
                msg = "The length of each 'trimTimeSegs' must be a pair of start and end time to trim out. The probelematic segment is: "+str(seg)
                raise Exception(msg)
            if seg[0] > seg[1]:
                msg = "The first entry of every 'trimTimeSegs' must be less than the second. The probelematic segment is: "+str(seg)
                raise Exception(msg)
            s, e = np.argmin(np.abs(T - seg[0])), np.argmin(np.abs(T - seg[1]))
            idx.extend(range(s,e))
        T = np.delete(T, idx)
        data = np.delete(data, idx, axis=0)

    if shiftTimeToZero:
        dt = np.mean(dt)
        if showLog:
            print(f"    Adopted time step: {dt}")
        T = np.linspace(0, (len(T)-1)*dt, num=len(T))

    if removeOutOfDomainProbes:
        if field == 'p':
            removeIdx = np.where(np.prod(abs(data) > __tol_data,axis=0))
        elif field == 'U':
            removeIdx = np.where(np.prod(np.prod(abs(data) > __tol_data,axis=2),axis=0))
        probes = np.delete(probes,removeIdx,0)
        data = np.delete(data,removeIdx,1)
        
    return probes, T, data
    
def processVelProfile(caseDir, probeName, targetProfile=None,
                        name=None,
                        normalize=True,
                        writeToDataFile=False,
                        trimTimeSegs:List[List[float]]=[[0,0.5]],
                        shiftTimeToZero=True,
                        H=None,
                        showPlots=False,
                        showLog=True,
                        exportPlots=True):
    
    caseName = os.path.abspath(caseDir).split(os.sep)[-1]
    postProcDir = caseDir+"/postProcessing/"
    outDir = caseDir+"/_processed/"
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    if showLog:
        print("Processing OpenFOAM case:\t"+caseDir)
        print("Probe read from:\t\t"+postProcDir+probeName)
        print("Target profile read from:\t"+str(targetProfile))
    
        print("  >> Reading probe data ...")
    probes,time,vel = readProbe(probeName, postProcDir, "U", trimTimeSegs=trimTimeSegs, showLog=showLog, shiftTimeToZero=shiftTimeToZero)
    if showLog:
        print("             << Done!")
    
    Z = probes[:,2]
    dt = np.mean(np.diff(time))
    
    if showLog:
        print("  >> Processing profile data.")
    name = caseName+"__"+probeName if name is None else name
    vel_LES = wind.profile(name=name,Z=Z, UofT=np.transpose(vel[:,:,0]), VofT=np.transpose(vel[:,:,1]), 
                          WofT=np.transpose(vel[:,:,2]), H=H, dt=dt, units=wind.DEFAULT_SI_UNITS)
    if showLog:
        print("             << Done!")
    
    if writeToDataFile:
        print("  >> Writing data to file.")
        vel_LES.writeToFile(outDir=outDir,writeTH=True,writeProfiles=True)
        print("             << Done!")
    
    if targetProfile is None:
        vel_EXP = None
        profiles = wind.Profiles((vel_LES,))
    else:
        vel_EXP = wind.profile(name="target",fileName=targetProfile,H=H)
        profiles = wind.Profiles((vel_EXP, vel_LES))
    if showLog:
        print("  >> Finished reading probe data.")
    figFile = outDir+vel_LES.name+"_profiles.pdf"
    if exportPlots:
        profiles.plotProfiles(figFile,normalize=normalize)

    return vel_LES
    
def writeProbeDict(file, points, fields=['p','U'], writeControl='adjustableRunTime', writeInterval='$probeWriteTime', includeLines=[], overwrite=False, precision=8, width=10):
    if os.path.exists(file):
        if not overwrite:
            msg = "The file "+file+" already exists."
            raise Exception(msg)
        else:
            msg = "The file "+file+" exists and will be overwritten."
            warnings.warn(msg)
    shp = np.shape(points)
    if not (shp[1] == 3):
        msg = "The points matrix must have three columns (x,y,z). The current shape is:"+str(shp)
        raise Exception(msg)

    with open(file,'w') as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |                                                   \n")
        f.write("  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox             \n")
        f.write("   \\\\    /   O peration     |                                                   \n")
        f.write("    \\\\  /    A nd           | Web:      www.OpenFOAM.org                        \n")
        f.write("     \\\\/     M anipulation  |                                                   \n")
        f.write("------------------------------------------------------------------------------- \n")
        f.write("Description                                                                     \n")
        f.write("    Writes out values of fields from cells nearest to specified locations.      \n")
        f.write("                                                                                \n")
        f.write("\\*---------------------------------------------------------------------------*/ \n")
        f.write("\n")
        f.write("#includeEtc \"caseDicts/postProcessing/probes/probes.cfg\"\n")
        for inc in includeLines:
            f.write(inc+"\n")
        f.write("\n")
        f.write("type            probes;\n")
        f.write("libs            (\"libsampling.so\");\n")
        f.write("\n")
        f.write(f"writeControl    {writeControl};\n")
        f.write(f"writeInterval   {writeInterval};\n")
        f.write("\n")
        f.write("fields\n")
        f.write("(\n")
        for fld in fields:
            f.write(f"    {fld}\n")
        f.write(");\n")
        f.write("\n")
        f.write("probeLocations \n")
        f.write("( \n")
        for row in points:
            f.write("({:>{width}.{precision}g} {:>{width}.{precision}g} {:>{width}.{precision}g})\n".format(row[0], row[1], row[2], width=width, precision=precision))
        f.write("); \n")
        f.write("\n")
        f.write("// ************************************************************************* //\n")


    pass

#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

class foamCase:
    def __init__(self, 
                caseDir=None,
                name=None,
                trimTimeSegs:List[List[float]]=[0,0.5],
                velProbeName_main=None,
                velProbeName_all=None,
                cpProbeName=None,
                ) -> None:
        self.caseDir = caseDir
        self.name = name
        self.trimTimeSegs = trimTimeSegs
        self.velProbeName_main = velProbeName_main
        self.velProbeName_all = velProbeName_all
        self.cpProbeName = cpProbeName

        self.mainProfile = None
        self.__allProfiles = None

    def Update(self, **kwargs):
        if self.caseDir is not None:
            if self.velProbeName_main is not None:
                self.mainProfile = processVelProfile(self.caseDir, self.velProbeName_main, name=self.name, 
                                                     exportPlots=False,trimTimeSegs=self.trimTimeSegs, H=self.H, showLog=False, **kwargs)
            if self.velProbeName_all is not None:
                self.__allProfiles = []
                for probe in self.velProbeName_all:
                    self.__allProfiles.append(processVelProfile(self.caseDir, probe, name=probe, exportPlots=False, 
                                                                trimTimeSegs=self.trimTimeSegs, H=self.H, showLog=False, **kwargs))

    @property
    def allProfiles(self):
        if self.__allProfiles is not None:
            return wind.Profiles(self.__allProfiles)
        else:
            return None

class inflowTuner:
    def __init__(self,
                 H = None,
                 nSpectAvg=8,
                 target:wind.profile=None,
                 inflows:wind.Profiles=None,
                 incidents:wind.Profiles=None,
                 ) -> None:
        self.H = H
        self.nSpectAvg = nSpectAvg
        self.target = target
        self.inflows = inflows
        self.incidents = incidents

    @property
    def allProfiles(self):
        profs = []
        if self.target is not None:
            profs.append(self.target)
        if self.inflows is not None:
            profs.extend(self.inflows.profiles)
        if self.incidents is not None:
            profs.extend(self.incidents.profiles)
        return wind.Profiles(profs)

    def targetProfileTable(self, castToUniform=False, nPts=1000, smoothWindow=1, kwargs_smooth={}) -> pd.DataFrame:
        if self.target is None:
            return None
        if np.isscalar(smoothWindow):
            smoothWindow = [smoothWindow,]*7

        table = pd.DataFrame()
        if castToUniform:
            table['Z'] = np.linspace(self.target.Z[0], np.array(self.target.Z)[-1], nPts)
            table['U'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.U), smoothWindow[0], **kwargs_smooth)
            table['Iu'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.Iu), smoothWindow[1], **kwargs_smooth)
            table['Iv'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.Iv), smoothWindow[2], **kwargs_smooth)
            table['Iw'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.Iw), smoothWindow[3], **kwargs_smooth)
            table['xLu'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.xLu), smoothWindow[4], **kwargs_smooth)
            table['xLv'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.xLv), smoothWindow[5], **kwargs_smooth)
            table['xLw'] = wind.smooth(np.interp(table['Z'], self.target.Z, self.target.xLw), smoothWindow[6], **kwargs_smooth)
        else:
            table['Z'] = self.target.Z
            table['U'] = wind.smooth(self.target.U, smoothWindow, **kwargs_smooth)
            table['Iu'] = wind.smooth(self.target.Iu, smoothWindow, **kwargs_smooth)
            table['Iv'] = wind.smooth(self.target.Iv, smoothWindow, **kwargs_smooth)
            table['Iw'] = wind.smooth(self.target.Iw, smoothWindow, **kwargs_smooth)
            table['xLu'] = wind.smooth(self.target.xLu, smoothWindow, **kwargs_smooth)
            table['xLv'] = wind.smooth(self.target.xLv, smoothWindow, **kwargs_smooth)
            table['xLw'] = wind.smooth(self.target.xLw, smoothWindow, **kwargs_smooth)
        return table
    
    def addInflow(self, caseName, sampleName, name=None):

        tFile = caseName+"/"+sampleName+"/time.npy"
        Ufile = caseName+"/"+sampleName+"/UofT.npy"
        Vfile = caseName+"/"+sampleName+"/VofT.npy"
        Wfile = caseName+"/"+sampleName+"/WofT.npy"
        probeFile = caseName+"/"+sampleName+"/probes.npy"

        time = np.load(tFile)
        UofT = np.load(Ufile)
        VofT = np.load(Vfile)
        WofT = np.load(Wfile)
        probes = np.load(probeFile)

        dt = time[2] - time[1]

        prof = wind.profile(name=name, Z=probes[:,2], UofT=UofT, VofT=VofT, WofT=WofT, H=self.H, dt=dt, nSpectAvg=self.nSpectAvg)
        if self.inflows is None:
            self.inflows = wind.Profiles([prof,])
        else:
            self.inflows.profiles.append(prof)

    def addIncident(self, caseDir, probeName, name=None, trimTimeSegs=[[0, 1.0],], showLog=False):
        prof = processVelProfile(caseDir,probeName=probeName,name=name, exportPlots=False,showLog=showLog, trimTimeSegs=trimTimeSegs,H=self.H)
        if self.incidents is None:
            self.incidents = wind.Profiles([prof,])
        else:
            self.incidents.profiles.append(prof)

    def __curveRatio(self, x1, x2, y1, y2, smoothWindow=1, castToUniform=True, nPoints=100, kwargs_smooth={}):
        if castToUniform:
            all_x = np.linspace(np.min(np.concatenate((x1, x2))), np.max(np.concatenate((x1, x2))), nPoints)
        else:
            all_x = np.unique(np.sort(np.concatenate((x1, x2))))
        all_y1 = np.interp(all_x, x1, y1, left=y1[0], right=y1[-1])
        all_y2 = np.interp(all_x, x2, y2, left=y2[0], right=y2[-1])
        all_y = wind.smooth(all_y1/all_y2, window_len=smoothWindow, **kwargs_smooth)
        return all_x, all_y

    def __profRatio(self, prof1, prof2, smoothWindow=1, castToUniform=True, nPoints=100, LES_useRange_Z=[0, 100], kwargs_smooth={}):
        table = pd.DataFrame()
        s = np.argmin(np.abs(prof2.Z - LES_useRange_Z[0]))
        e = np.argmin(np.abs(prof2.Z - LES_useRange_Z[1]))
        
        table['Z'], table['U'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.U, prof2.U[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                                   kwargs_smooth=kwargs_smooth)
        _, table['Iu'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.Iu, prof2.Iu[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['Iv'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.Iv, prof2.Iv[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['Iw'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.Iw, prof2.Iw[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLu'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.xLu, prof2.xLu[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLv'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.xLv, prof2.xLv[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLw'] = self.__curveRatio(prof1.Z, prof2.Z[s:e], prof1.xLw, prof2.xLw[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        e = np.argmin(np.abs(table['Z'] - LES_useRange_Z[1]))
        for comp in ['U', 'Iu', 'Iv', 'Iw', 'xLu', 'xLv', 'xLw']:
            table[comp][e:] = table[comp][e-1]
        return table

    def getScaledTarget(self, smoothWindow=1, scaleByInflow=False, nPoints=100, castToUniform=True, LES_useRange_Z=[0, 100], kwargs_smooth={}):
        # if self.target is None or self.incidents is None or (self.inflows is None and scaleByInflow):
        #     return None, None, None
        
        target = self.targetProfileTable(smoothWindow=smoothWindow, kwargs_smooth=kwargs_smooth)
        # target = self.targetProfileTable(smoothWindow=10, castToUniform=True, kwargs_smooth={'window':'hamming', 'mode':'valid', 'usePadding':True, 
        #                                                                                 'paddingFactor':2, 'paddingMode':'edge'})
        names = []
        if scaleByInflow:
            if self.inflows is None:
                profs = wind.Profiles()
                names = []
            else:
                profs = self.inflows.copy()
                names = [prof.name for prof in self.inflows.profiles]
        else:
            if self.incidents is None:
                profs = wind.Profiles()
                names = []
            else:
                profs = self.incidents.copy()
                names = [prof.name for prof in self.incidents.profiles]
        
        factors = []
        scaledTables = []
        for i, prof in enumerate(profs.profiles):
            factors.append(self.__profRatio(self.target, prof, smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, LES_useRange_Z=LES_useRange_Z, kwargs_smooth=kwargs_smooth))

            scaledTable = target.copy()
            for col in ['U', 'Iu', 'Iv', 'Iw', 'xLu', 'xLv', 'xLw']:
                scaledTable[col] *= factors[i][col]
            scaledTables.append(scaledTable)
        return scaledTables, factors, names

    def plotProfiles(self, figsize=[15,12], zLim=[0,10], normalize=True, lw = 1.5, ms=3,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None,
                            lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt=None, kwargs_ax={}):
        longListOfColors = ['r', 'b', 'g', 'm', 'k', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'navy', ]
       
        if kwargs_plt is None:
            kwargs_plt = []
            kwargs_plt.append({'color':'k', 'lw':1, 'ls':'None', 'marker':'o', 'ms':ms})
            if self.inflows is not None:
                for i, prof in enumerate(self.inflows.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'--'})
            if self.incidents is not None:
                for i, prof in enumerate(self.incidents.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'-'})

        self.allProfiles.plotProfile_basic2(figsize=figsize, yLimits=zLim, normalize=normalize, xLimits_U=xLimits_U, xLimits_Iu=xLimits_Iu, xLimits_Iv=xLimits_Iv, 
                                            xLimits_Iw=xLimits_Iw, xLimits_uw=xLimits_uw, xLimits_xLu=xLimits_xLu, xLimits_xLv=xLimits_xLv, xLimits_xLw=xLimits_xLw, 
                                            xLabel=xLabel, zLabel=zLabel, lgnd_kwargs=lgnd_kwargs, kwargs_plt=kwargs_plt, kwargs_ax=kwargs_ax)