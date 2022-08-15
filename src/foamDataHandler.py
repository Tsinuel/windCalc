# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:49:36 2022

@author: Tsinu
"""
import numpy as np
import os
import warnings
import windLoadCaseProcessors as wProc

__tol_time = 1e-7
__tol_data = 1e15

def __foamReadProbe_singleT(file,field):
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

def foamReadProbe(probeName, postProcDir, field, trimTimeSegs=[[0,0]], trimOverlap=True, 
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
        
        (probes,time,d) = __foamReadProbe_singleT(fileName, field)
        
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
    
def foamProcessVelProfile(caseDir, probeName,
                          writeToDataFile=False,
                          showPlots=False,
                          exportPlots=True):
    
    # caseDir = "D:/OneDrive - The University of Western Ontario/Documents/PhD/Thesis/CodeRepositories/windCalc/data/exampleOFcase/"
    postProcDir = caseDir+"postProcessing/"
    # probeName = "probes.tapsABCD"
    # probeName = "probes.V1"
    
    probes,time,vel = foamReadProbe(probeName, postProcDir, "U")
    
    Z = probes[:,2]
    dt = np.mean(np.diff(time))
    
    ref_U_TI_L, Z, U, TI, L, freq, Spect_H, Spect_Others = wProc.processVelProfile(Z,vel,dt,0.08)
    
 