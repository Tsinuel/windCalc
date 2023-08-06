# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:49:36 2022

@author: Tsinuel Geleta

This module contains functions for reading and processing OpenFOAM data.

The module is part of the windCalc package.

Functions
---------
meshSizeFromCutoffFreq : Calculate the mesh size from the cutoff frequency and the spectral densities of the velocity components.
readProfiles : Read profiles from a file.
readBDformattedFile : Read a file in the format of boundaryData. Can be points or velocity.
writeBDformattedFile : Write a file in the format of boundaryData. Can be points or velocity.
getProfileScaleFactor : Calculate the scale factor for a profile.
scaleVelocity : Scale velocity components.
readInflowDict : Read the inflow dictionary.
getClosest2DcoordsTo : Get the closest 2D coordinates to a given set of coordinates.
scaleInflowData : Scale inflow data.
extractSampleProfileFromInflow : Extract a sample profile from inflow data.
readProbe : Read probe data from an OpenFOAM case.
processVelProfile : Process velocity profiles.
readVelProfile : Read velocity profiles.
readSurfacePressure : Read surface pressure data.

Classes
-------
foamCase : A class for handling OpenFOAM cases.
inflowTuner : A class for scaling inflow data.

"""
import numpy as np
import os
import glob
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
import datetime


TIME_STEP_TOLERANCE = 1e-7
MAX_DATA_LIMIT = 1e+50
SHOWLOG = False

__precision = 6

#===============================================================================
#=============================  FUNCTIONS  =====================================
#===============================================================================
def printlog(msg, showLog=SHOWLOG):
    if showLog:
        print(msg)

#---------------------------  General CFD  -------------------------------------
def meshSizeFromCutoffFreq(nc, Suu, Svv, Sww):
    """
    Calculate the mesh size from the cutoff frequency and the spectral densities
    of the velocity components.

    Parameters
    ----------
    fc : float
        The cutoff frequency.
    Suu : np.array(float)
        The spectral density of the longitudinal velocity component.
    Svv : np.array(float)
        The spectral density of the lateral velocity component.
    Sww : np.array(float)
        The spectral density of the vertical velocity component.

    Returns
    -------
    dx : float
        The mesh size.

    References
    ----------
    [1]  Geleta, T.N., Bitsuamlak, G.T., 2022. Validation metrics and turbulence 
        frequency limits for LES-based wind load evaluation for low-rise 
        buildings. J. Wind Eng. Ind. Aerodyn. 231, 105210. 
        https://doi.org/10.1016/j.jweia.2022.105210
    """
    Suu = np.asarray(Suu)
    Svv = np.asarray(Svv)
    Sww = np.asarray(Sww)
    
    dx = np.sqrt( (np.pi/(2*nc)) * (Suu+Svv+Sww) )
    
    return dx

#-------------------------------- IO  ------------------------------------------
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

def readVTKfile(vtkFile, fieldName, showLog=True):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFile)
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    if showLog:
        print("Reading field '"+fieldName+"' from file: "+vtkFile)
        print("    No. of points: "+str(data.GetNumberOfPoints()))
        print("    No. of cells: "+str(data.GetNumberOfCells()))
        print("    No. of arrays: "+str(data.GetPointData().GetNumberOfArrays()))

    if fieldName == 'p':
        field = data.GetPointData().GetScalars('p')
    elif fieldName == 'U':
        field = data.GetPointData().GetVectors('U')
    else:
        raise Exception("The field '"+fieldName+"' is not available in the file: "+vtkFile)
    points = data.GetPoints().GetData()
    points = vtk_to_numpy(points)
    field = vtk_to_numpy(field)
    if showLog:
        print("    Shape of points: "+str(np.shape(points)))
        print("    Shape of the field: "+str(np.shape(field)))
    return points, field

def readRAWfile(file, numHeader=2, showLog=True):
    # space delimited column file with the following header:
    # U  POINT_DATA 266505
    #  x  y  z  U_x  U_y  U_z
    with open(file) as f:
        first_line = f.readline()
        second_line = f.readline()
    nCols = len(second_line.split())-1
    nRows = int(first_line.split()[-1])
    if showLog:
        print("Reading file: "+file)
        print("    No. of columns: "+str(nCols))
        print("    No. of rows: "+str(nRows))
    data = pd.read_csv(file, skiprows=numHeader, sep=' ', header=None, names=second_line.split()[1:])
    if showLog:
        print("    Shape of the data: "+str(np.shape(data)))
    return data

#-----------------------  Inflow data handlers  --------------------------------
def convertSectionSampleToBoundaryDataInflow(caseDir, sectionName, fileName, 
                                             outDir=None, 
                                             shiftTimeBy=None, 
                                             overwrite=False,
                                             checkMatchingPoints=True,
                                             pointDistanceTolerance=1e-6,
                                             timeOutputPrecision=6,
                                             showLog=True, 
                                             detailedLog=False,
                                             num_processors=1,
                                             ):
    if not os.path.exists(caseDir):
        raise Exception("The case directory '"+caseDir+"' does not exist.")
    sectDir = caseDir+"/postProcessing/"+sectionName+"/"
    if not os.path.exists(sectDir):
        raise Exception("The section directory '"+sectDir+"' does not exist.")
    if outDir is None:
        outDir = caseDir+"/constant/boundaryData/inlet/"
    os.makedirs(outDir, exist_ok=True)
    if showLog:
        print("Converting section sample to boundaryData inflow.")
        print("    Reading section sample from: "+sectDir+fileName)
        print("    Writing boundaryData inflow to: "+outDir)
    
    times = [ name for name in os.listdir(sectDir) if os.path.isdir(os.path.join(sectDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    times = [x[0] for x in times]
    if len(times) == 0:
        print("    No time directories found in: "+sectDir)
        return
    pointsFile = outDir+"points"
    if os.path.exists(pointsFile):
        points_old = readBDformattedFile(pointsFile)
    pointsFileHandled = False
    shiftTimeBy = -1.0*float(times[0]) if shiftTimeBy is None else shiftTimeBy

    if showLog:
        print("    Shifting time by: "+str(shiftTimeBy))
    
    # if num_processors > 1:
    #     import multiprocessing as mp
    #     pool = mp.Pool(processes=num_processors)
    #     for t in times:
    #         pool.apply_async(__convertSectionSampleToBoundaryDataInflow_singleT, args=(sectDir, t, fileName, outDir, shiftTimeBy, overwrite, pointsFileHandled, pointsFile, pointDistanceTolerance, timeOutputPrecision, showLog, detailedLog))
    #     pool.close()
    #     pool.join()
    for t in times:
        t_out = str(round(float(t)+shiftTimeBy,timeOutputPrecision))
        timeDir_out = outDir+"/"+t_out+"/"
        if os.path.exists(timeDir_out) and not overwrite:
            print("    Skipping existing time step: "+t_out+"/U")
            continue
        file = glob.glob(os.path.join(sectDir, t, fileName+".*"))
        if len(file) == 0:
            print("    The file '"+sectDir+t+"/"+fileName+"' does not exist.")
            continue
        if len(file) > 1:
            print("    Multiple files found for '"+sectDir+t+"/"+fileName+"'.")
            print("    The first file will be used.")
        file = file[0]
        ext = os.path.splitext(file)[1]
        if ext == '.vtk':
            points, vel = readVTKfile(file, 'U', showLog=detailedLog)
        elif ext == '.raw':
            data = readRAWfile(file, showLog=detailedLog)
            points = data[['x','y','z']].to_numpy()
            vel = data[['U_x','U_y','U_z']].to_numpy()
        else:
            raise NotImplementedError("The file extension '"+ext+"' is not supported.")
        if not pointsFileHandled:
            if overwrite or not os.path.exists(pointsFile):
                writeBDformattedFile(pointsFile, points)
                points_old = points
            pointsFileHandled = True
            if showLog:
                print("    Writing points to: "+pointsFile)
                print("    Writing velocity to: <output_dir> = "+outDir)
        else:
            if not np.array_equal(points, points_old) and checkMatchingPoints:
                # throw error if the number of points does not match
                if not len(points) == len(points_old):
                    raise Exception("The number of points in the file '"+sectDir+t+"/"+fileName+ext+"' do not match with the number of points in the file '"+pointsFile+"'.")
                if showLog:
                    print("      Reordering velocity to match the points in existing 'points' file.")
                from scipy.spatial import KDTree
                tree = KDTree(points)
                distance,idx = tree.query(points_old)
                # check the maximum distance between the points
                if max(distance) > pointDistanceTolerance:
                    raise Exception("The points in the file '"+sectDir+t+"/"+fileName+ext+"' do not match with the points in the file '"+pointsFile+"'.")
                if showLog:
                    print("      Distance between the points: max = "+str(max(distance))+", mean = "+str(np.mean(distance)))
                points = points[idx,:]
                vel = vel[idx,:]
        if overwrite or not os.path.exists(timeDir_out):
            os.makedirs(timeDir_out, exist_ok=True)
            writeBDformattedFile(timeDir_out+"U", vel)
            if showLog:
                print("    "+t+"/"+fileName+ext+" \t-->  "+t_out+"/U")
        else:
            if showLog:
                print("    Skipping existing time step: "+t_out+"/U")
    if showLog:
        print("    << Finished converting section sample to boundaryData inflow.")

#-----------------  Probe readers and related functions  -----------------------
def read_OF_probe_single(file,field):
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
    points = np.zeros([0])
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
                    if len(points) == 0:
                        points = prb_i
                    else:
                        points = np.append(points,prb_i,axis=0)
                else:
                    continue
            else:
                line = line.split()
                if field == 'p':
                    if len(line[1:]) < len(points): # inclomplete line
                        continue
                    d = np.reshape(np.asarray(line[1:],dtype=(float)),[1,-1])
                elif field == 'U':
                    if len(line[1:])/3 < len(points): # inclomplete line
                        continue
                    d = np.reshape(np.asarray(line[1:],dtype=(float)),[1,-1,3])
                else:
                    raise Exception("The probe reader is not implemented for field '"+field+"'.")

                if len(data) == 0:
                    data = d
                else:
                    data = np.append(data, d,axis=0)
                time = np.append(time,float(line[0]))
   
    return points, time, data

def read_OF_probe(probeName, postProcDir, field, trimTimeSegs:List[List[float]]=None, trimOverlap=True, 
              shiftTimeToZero=True, removeOutOfDomainProbes=True, showLog=True, skipNoneExistingFiles=True):
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
    
    points = np.zeros([0])
    T = np.zeros([0])
    data = np.zeros([0])
    
    for t in times:
        fileName = probeDir+t[0]+"/"+field
        if not os.path.exists(fileName):
            if skipNoneExistingFiles:
                if showLog:
                    print(f"           skipping non-existing file: {fileName}")
                continue
            else:
                msg = f"ERROR! The file '{fileName}' does not exist."
                raise Exception(msg)
        if showLog:
            print(f"           Reading {field} from: {fileName}")
        
        (points,time,d) = read_OF_probe_single(fileName, field)
        if showLog:
            tStart = time[0] if len(time) > 0 else np.nan
            tEnd = time[-1] if len(time) > 0 else np.nan
            print(f"                {len(points)} probes with {len(time)} time steps ({tStart} to {tEnd})")
            print(f"                No. of overlapping time steps with previously read data: {len(np.intersect1d(T,time))}")
            print(f"                Shape of data: {np.shape(d)}")
        
        if len(time) == 0:
            if showLog:
                print(f"             No data found in: {fileName}")
            continue

        if len(T) == 0:
            T = time
            data = d
        else:
            T = np.append(T,time)
            data = np.append(data,d,axis=0)

    if trimOverlap:
        if showLog:
            print("      Trimming overlapping times.")
        T,idx = np.unique(T,return_index=True)
        data = data[idx,:]
        
    dt = np.diff(np.unique(np.sort(T)))
    if showLog:
        print(f"      Deviation from fixed time step: mean = {np.mean(np.abs(dt))}, max = {max(np.abs(dt))}, standard deviation = {np.std(dt)}")
    if max(dt)-min(dt) > TIME_STEP_TOLERANCE:
        msg = f"WARNING! Non-uniform time step detected in '{probeName}'. The highest difference in time step is: {max(dt)-min(dt)}"
        warnings.warn(msg)
        
    if trimTimeSegs is not None:
        if showLog:
            print(f"      Trimming explicitly defined time segments: {trimTimeSegs}")
        idx = []
        for seg in trimTimeSegs:
            if not len(seg) == 2:
                msg = "The length of each 'trimTimeSegs' must be a pair of start and end time to trim out. The problematic segment is: "+str(seg)
                raise Exception(msg)
            if seg[0] > seg[1]:
                msg = "The first entry of every 'trimTimeSegs' must be less than the second. The problematic segment is: "+str(seg)
                raise Exception(msg)
            s, e = np.argmin(np.abs(T - seg[0])), np.argmin(np.abs(T - seg[1]))
            idx.extend(range(s,e))
        T = np.delete(T, idx)
        data = np.delete(data, idx, axis=0)

    if shiftTimeToZero:
        dt = np.mean(dt)
        if showLog:
            print(f"      Shifting time to start from zero with a time step of {dt}.")            
        T = np.linspace(0, (len(T)-1)*dt, num=len(T))

    if removeOutOfDomainProbes:
        if field == 'p':
            removeIdx = np.where(np.prod(abs(data) > MAX_DATA_LIMIT,axis=0))
        elif field == 'U':
            removeIdx = np.where(np.prod(np.prod(abs(data) > MAX_DATA_LIMIT,axis=2),axis=0))
        if showLog:
            if len(removeIdx[0]) > 0:
                print(f"      Removing {len(removeIdx[0])} out-of-domain probes.")
        points = np.delete(points,removeIdx,0)
        data = np.delete(data,removeIdx,1)
        
    if showLog:
        print(f"      No. of probes: {len(points)}")
        print(f"      No. of time steps: {len(T)}")
        print(f"      Shape of data: {np.shape(data)}")
        print("      << Finished reading probe data.")

    return points, T, data
    
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
    os.makedirs(outDir, exist_ok=True)
    if showLog:
        print("Processing OpenFOAM case:\t"+caseDir)
        print("Probe read from:\t\t"+postProcDir+probeName)
        print("Target profile read from:\t"+str(targetProfile))
    
        print("  >> Reading probe data ...")
    probes,time,vel = read_OF_probe(probeName, postProcDir, "U", trimTimeSegs=trimTimeSegs, showLog=showLog, shiftTimeToZero=shiftTimeToZero)
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

def readVelProfile(caseDir, probeName, 
                    name=None,
                    trimTimeSegs:List[List[float]]=[[0,0.5]],
                    shiftTimeToZero=True,
                    readPressure=True,
                    H=None,
                    showLog=True, 
                    writeToFile=False,
                    outDir=None,
                    readFromNPY_file=False,
                    kwargs_profile={},
                    ):
    '''
    Read velocity profile from an OpenFOAM case.

    Parameters
    ----------
    caseDir : str
        The case directory.
    probeName : str
        The probe name given in the controlDict.
    name : str, optional
        The name of the profile. The default is None.
    trimTimeSegs : List[List[float]], optional
        A list of time segments to be trimmed out. The default is [[0,0.5]].
    shiftTimeToZero : bool, optional
        Whether or not to shift the time vector in such a way that the time starts from zero. It is significant if there has been some trimming and clipping on the time history. The default is True.
    H : float, optional
        The referece height for normalization. The default is None.
    showLog : bool, optional
        Whether or not to show log messages. The default is True.

    Returns
    -------
    prof : wind.profile
        The velocity profile object.
        
    Examples
    --------
    >>> caseDir = "D:/Tsinu/..../windCalc/data/exampleOFcase/"
    >>> probeName = "probes.tapsABCD"
    >>> prof = readVelProfile(caseDir, probeName)
    >>> prof.name
    'exampleOFcase__probes.tapsABCD'
    >>> prof.Z
    array([0.001, 0.002, 0.003, ..., 0.997, 0.998, 0.999])
    >>> prof.UofZ
    '''
    caseName = os.path.abspath(caseDir).split(os.sep)[-1]
    postProcDir = caseDir+"/postProcessing/"
    if writeToFile or readFromNPY_file:
        if outDir is None:
            outDir = caseDir+"/_processed/"
        os.makedirs(outDir, exist_ok=True)
    if showLog:
        print("Processing OpenFOAM case:\t"+caseDir)
        print("Probe read from:\t\t"+postProcDir+probeName)
        print("  >> Reading probe data ...")
    if not readFromNPY_file:
        probes,time,vel = read_OF_probe(probeName, postProcDir, "U", trimTimeSegs=trimTimeSegs, showLog=showLog, shiftTimeToZero=shiftTimeToZero)
        if writeToFile:
            if showLog:
                print("  >> Writing data to NPY file.")
                print("             << Done!")
            np.save(outDir+probeName+"_vel.npy",vel)
            np.save(outDir+probeName+"_probes.npy",probes)
            np.save(outDir+probeName+"_time.npy",time)
    else:
        if showLog:
            print("  >> Reading data from NPY file.")
        vel = np.load(outDir+probeName+"_vel.npy")
        probes = np.load(outDir+probeName+"_probes.npy")
        time = np.load(outDir+probeName+"_time.npy")
    if showLog:
        print("             << Done!")
    
    Z = probes[:,2]
    X = probes[:,0]
    Y = probes[:,1]
    dt = np.mean(np.diff(time))
    
    if showLog:
        print("  >> Processing profile data.")

    if readPressure:
        if showLog:
            print("  >> Reading pressure data ...")

        # list all time directories
        sampleDir = postProcDir+probeName+"/"
        times = [ name for name in os.listdir(sampleDir) if os.path.isdir(os.path.join(sampleDir, name)) ]
        contains_p = [name for name in times if os.path.exists(sampleDir+name+"/p")] 
        if any(contains_p):
            if not readFromNPY_file:
                pressure,_,time_p = readSurfacePressure(caseDir=caseDir, probeName=probeName, trimTimeSegs=trimTimeSegs, shiftTimeToZero=shiftTimeToZero,
                                                                showLog=showLog)
                pressure = np.transpose(pressure)
                idxOfP = np.where(np.isin(np.round(time_p,decimals=6),np.round(time,decimals=6),assume_unique=True))[0]
                pressure = pressure[:,idxOfP]
                time_p = time_p[idxOfP]

                idxOfV = np.where(np.isin(np.round(time,decimals=6),np.round(time_p,decimals=6),assume_unique=True))[0]
                vel = vel[idxOfV,:,:]
                time = time[idxOfV]

                if writeToFile:
                    if showLog:
                        print("  >> Writing data to NPY file.")
                        print("             << Done!")
                    np.save(outDir+probeName+"_pressure.npy",pressure)
                    # np.save(outDir+probeName+"_time_p.npy",time_p)
            else:
                if showLog:
                    print("  >> Reading data from NPY file.")
                pressure = np.load(outDir+probeName+"_pressure.npy")
                # time_p = np.load(outDir+probeName+"_time_p.npy")
        else:
            pressure = None
            # time_p = None
    if showLog:
        print("             << Done!")
        print("  >> Finished reading probe data.")
    
    name = caseName+"__"+probeName if name is None else name
    if len(np.shape(vel)) == 3:
        prof = wind.profile(name=name, X=X, Y=Y, Z=Z, UofT=np.transpose(vel[:,:,0]), VofT=np.transpose(vel[:,:,1]), 
                            WofT=np.transpose(vel[:,:,2]), H=H, dt=dt, units=wind.DEFAULT_SI_UNITS,
                            pOfT=pressure,
                            **kwargs_profile)
    else:
        print("WARNING! The velocity data is not a time history. The profile will be created without.")
        prof = None
    return prof

def readSurfacePressure(caseDir, probeName, 
                    trimTimeSegs:List[List[float]]=[[0,0.5]],
                    shiftTimeToZero=True,
                    showLog=True
                    ):
    # caseName = os.path.abspath(caseDir).split(os.sep)[-1]
    postProcDir = caseDir+"/postProcessing/"
    outDir = caseDir+"/_processed/"
    os.makedirs(outDir, exist_ok=True)
    if showLog:
        print("Processing OpenFOAM case:\t"+caseDir)
        print("Probe read from:\t\t"+postProcDir+probeName)
        print("  >> Reading probe data ...")
    probes,time,pressure = read_OF_probe(probeName, postProcDir, "p", trimTimeSegs=trimTimeSegs, showLog=showLog, shiftTimeToZero=shiftTimeToZero)
    if showLog:
        print("             << Done!")
    
    # name = caseName+"__"+probeName if name is None else name

    if showLog:
        print("             << Done!")
        print("  >> Finished reading probe data.")
    
    return pressure,probes,time

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

        self.Refresh()

    def Refresh(self, **kwargs):
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
                 refProfiles:wind.Profiles=None,
                 ) -> None:
        self.H = H
        self.nSpectAvg = nSpectAvg
        self.target = target
        self.inflows = inflows
        self.incidents = incidents
        self.refProfiles = refProfiles

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

    def targetProfileTable(self, castToUniform=False, nPts=1000, smoothen=False, smoothWindow=1, kwargs_smooth={}) -> pd.DataFrame:
        if self.target is None:
            return None
        if np.isscalar(smoothWindow):
            smoothWindow = [smoothWindow,]*7

        table = pd.DataFrame()
        if castToUniform:
            table['Z'] = np.linspace(self.target.Z[0], np.array(self.target.Z)[-1], nPts)
            table['U'] = np.interp(table['Z'], self.target.Z, self.target.U)
            table['Iu'] = np.interp(table['Z'], self.target.Z, self.target.Iu)
            table['Iv'] = np.interp(table['Z'], self.target.Z, self.target.Iv)
            table['Iw'] = np.interp(table['Z'], self.target.Z, self.target.Iw)
            table['xLu'] = np.interp(table['Z'], self.target.Z, self.target.xLu)
            table['xLv'] = np.interp(table['Z'], self.target.Z, self.target.xLv)
            table['xLw'] = np.interp(table['Z'], self.target.Z, self.target.xLw)
        else:
            table['Z'] = self.target.Z
            table['U'] = self.target.U
            table['Iu'] = self.target.Iu
            table['Iv'] = self.target.Iv
            table['Iw'] = self.target.Iw
            table['xLu'] = self.target.xLu
            table['xLv'] = self.target.xLv
            table['xLw'] = self.target.xLw

        if smoothen:
            table['U'] = wind.smooth(table['U'], smoothWindow[0], **kwargs_smooth)
            table['Iu'] = wind.smooth(table['Iu'], smoothWindow[1], **kwargs_smooth)
            table['Iv'] = wind.smooth(table['Iv'], smoothWindow[2], **kwargs_smooth)
            table['Iw'] = wind.smooth(table['Iw'], smoothWindow[3], **kwargs_smooth)
            table['xLu'] = wind.smooth(table['xLu'], smoothWindow[4], **kwargs_smooth)
            table['xLv'] = wind.smooth(table['xLv'], smoothWindow[5], **kwargs_smooth)
            table['xLw'] = wind.smooth(table['xLw'], smoothWindow[6], **kwargs_smooth)

        return table
    
    def addInflow(self, caseName, sampleName, name=None, nSpectAvg=None):

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

        nSpectAvg = self.nSpectAvg if nSpectAvg is None else nSpectAvg

        prof = wind.profile(name=name, Z=probes[:,2], UofT=UofT, VofT=VofT, WofT=WofT, H=self.H, dt=dt, nSpectAvg=nSpectAvg)
        if self.inflows is None:
            self.inflows = wind.Profiles([prof,])
        else:
            self.inflows.profiles.append(prof)

    def addIncident(self, caseDir, probeName, name=None, readFromNPY_file=True, writeToDataFile=False,
                    trimTimeSegs=[[0, 1.0],], showLog=True, kwargs_profile={},
                    kwargs_readVelProfile={},):
        
        prof = readVelProfile(caseDir=caseDir,probeName=probeName,name=name, showLog=showLog, trimTimeSegs=trimTimeSegs,H=self.H, readFromNPY_file=readFromNPY_file,
                                writeToFile=writeToDataFile, kwargs_profile=kwargs_profile,
                              **kwargs_readVelProfile)
        if self.incidents is None:
            self.incidents = wind.Profiles([prof,])
        else:
            self.incidents.profiles.append(prof)

    def __curveRatio____redacted(self, x1, x2, y1, y2, smoothWindow=1, castToUniform=True, nPoints=100, kwargs_smooth={}):
        if castToUniform:
            all_x = np.linspace(np.min(np.concatenate((x1, x2))), np.max(np.concatenate((x1, x2))), nPoints)
        else:
            all_x = np.unique(np.sort(np.concatenate((x1, x2))))
        all_y1 = np.interp(all_x, x1, y1, left=y1[0], right=y1[-1])
        all_y2 = np.interp(all_x, x2, y2, left=y2[0], right=y2[-1])
        all_y = wind.smooth(all_y1/all_y2, window_len=smoothWindow, **kwargs_smooth)
        return all_x, all_y

    def __profRatio____redacted(self, prof1, prof2, smoothWindow=1, castToUniform=True, nPoints=100, LES_useRange_Z=[0, 100], kwargs_smooth={}):
        table = pd.DataFrame()
        s = np.argmin(np.abs(prof2.Z - LES_useRange_Z[0]))
        e = np.argmin(np.abs(prof2.Z - LES_useRange_Z[1]))
        
        table['Z'], table['U'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.U, prof2.U[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                                   kwargs_smooth=kwargs_smooth)
        _, table['Iu'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.Iu, prof2.Iu[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['Iv'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.Iv, prof2.Iv[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['Iw'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.Iw, prof2.Iw[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLu'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.xLu, prof2.xLu[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLv'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.xLv, prof2.xLv[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        _, table['xLw'] = self.__curveRatio____redacted(prof1.Z, prof2.Z[s:e], prof1.xLw, prof2.xLw[s:e], smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, 
                                           kwargs_smooth=kwargs_smooth)
        e = np.argmin(np.abs(table['Z'] - LES_useRange_Z[1]))
        for comp in ['U', 'Iu', 'Iv', 'Iw', 'xLu', 'xLv', 'xLw']:
            table[comp][e:] = table[comp][e-1]
        return table

    def getScaledTarget__redacted(self, smoothWindow=1, scaleByInflow=False, nPoints=100, castToUniform=True, LES_useRange_Z=[0, 100], kwargs_smooth={}):
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
            factors.append(self.__profRatio____redacted(self.target, prof, smoothWindow=smoothWindow, castToUniform=castToUniform, nPoints=nPoints, LES_useRange_Z=LES_useRange_Z, kwargs_smooth=kwargs_smooth))

            scaledTable = target.copy()
            for col in ['U', 'Iu', 'Iv', 'Iw', 'xLu', 'xLv', 'xLw']:
                scaledTable[col] *= factors[i][col]
            scaledTables.append(scaledTable)
        return scaledTables, factors, names

    def writeProfile(self, rounds=0, dir=None, name='profile', caseName='infl', figsize=[15,15], zLim=[0,10], scaleByFixedRefHeightRatio=True, description=" ",
                     debugMode=False, smoothTheTarget=True, smoothRatios=False, compensateFor_xLi_in_Ii=False,
                     zMax_scaling=None, zMin_scaling=None, scale_xLi=True, applyLimitedSmoothing=False,
                     smoothWindow_ratios=[50, 50, 50, 50, 200, 150, 200], 
                     smoothWindow_target=[50, 50, 50, 50, 200, 150, 200], 
                     kwargs_smooth={'window':'hamming', 'mode':'valid', 'usePadding':True, 'paddingFactor':2, 'paddingMode':'edge'},
                     ):
        def plotThese(profs:List[pd.DataFrame]=[], names=[], ratio:pd.DataFrame=None, mainTitle=caseName, markers=None, color=None, lwgts=None, lss=None,
                      saveToFile=True):
            if not debugMode:
                return
            if len(profs) == 0:
                return
            fig, axs = plt.subplots(3,3, figsize=figsize)
            # set the facecolor of the figure
            fig.set_facecolor('w')
            axs = axs.flatten()
            flds = ['U', 'Iu', 'Iv', 'Iw', 'xLu', 'xLv', 'xLw']
            axIdxs = [0, 3, 4, 5, 6, 7, 8]
            axs[1].axis('off')
            axs[2].axis('off')
            ax_legend = axs[1]
            # mrkrs = ['.', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_'] 
            longListOfMarkers = ['.', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_', 'o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_', 'o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_', 'o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_', 'o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '8', 'D', 'P', 'X', '*', 'H', '1', '2', '3', '4', '+', 'x', '|', '_', ]
            markers = longListOfMarkers if markers is None else markers
            longListOfColors = ['k', 'b', 'r', 'g', 'm', 'c', 'y', 'tab:gray', 'tab:olive', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', ]
            cols = longListOfColors if color is None else color
            lwgts = [1,]*len(profs) if lwgts is None else lwgts
            lss = ['-',]*len(profs) if lss is None else lss
            
            for ii in range(len(flds)):
                ax = axs[axIdxs[ii]]
                for jj, prof in enumerate(profs):
                    ax.plot(prof[flds[ii]], prof['Z']/self.target.H, label=names[jj], lw=lwgts[jj], ls=lss[jj], marker=markers[jj], ms=2, c=cols[jj%len(cols)])
                if ratio is not None:
                    ax2 = ax.twiny()
                    ax2.plot(ratio[flds[ii]], ratio['Z']/self.target.H, label='ratio', lw=2, ls='--', c='c',marker='None', ms=2)
                    ax2.set_xlabel('ratio')
                    ax2.axvline(1.0, c='k', ls='--', lw=1)
                    ax2.xaxis.label.set_color('c')
                    ax2.tick_params(axis='x', colors='c')
                    ax2.spines['top'].set_color('c')
                    wind.formatAxis(ax2, gridMajor=False, gridMinor=False, tickDirection='out')
                ax.set_xlabel(flds[ii])
                ax.set_ylabel('Z/H')
                ax.set_ylim(zLim)
                wind.formatAxis(ax)
            # show all legends from ax and ax2 in one legend
            handles, labels = ax.get_legend_handles_labels()
            if ratio is not None:
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles.extend(handles2)
                labels.extend(labels2)
            ax_legend.legend(handles, labels, loc='center', ) #bbox_to_anchor=(0.5, 0.5))
            
            plt.tight_layout()
            if mainTitle is not None:
                fig.suptitle(mainTitle)
            if dir is not None and saveToFile:
                plt.savefig(dir+caseName+"_round"+str(rounds)+".svg", dpi=300)

        def interpToZ(prof, Z, merge=True):
            if merge:
                Z = np.unique(np.sort(np.concatenate((prof.Z, Z))))
            prof_out = pd.DataFrame()
            prof_out['Z'] = Z
            prof_out['U'] = np.interp(Z, prof.Z, prof.U,) # left=prof.U[0], right=prof.U[-1])
            prof_out['Iu'] = np.interp(Z, prof.Z, prof.Iu) #, left=prof.Iu[0], right=prof.Iu[-1])
            prof_out['Iv'] = np.interp(Z, prof.Z, prof.Iv) #, left=prof.Iv[0], right=prof.Iv[-1])
            prof_out['Iw'] = np.interp(Z, prof.Z, prof.Iw) #, left=prof.Iw[0], right=prof.Iw[-1])
            prof_out['xLu'] = np.interp(Z, prof.Z, prof.xLu) #, left=prof.xLu[0], right=prof.xLu[-1])
            prof_out['xLv'] = np.interp(Z, prof.Z, prof.xLv) #, left=prof.xLv[0], right=prof.xLv[-1])
            prof_out['xLw'] = np.interp(Z, prof.Z, prof.xLw) #, left=prof.xLw[0], right=prof.xLw[-1])
            return prof_out
        
        def smooth_profile(prof,imin=0, imax=-1, smoothWindow=smoothWindow_ratios, kwargs_smooth=kwargs_smooth):
            prof_out = prof.copy()
            prof_out['U'][imin:imax] = wind.smooth(prof['U'][imin:imax], smoothWindow[0], **kwargs_smooth)
            prof_out['Iu'][imin:imax] = wind.smooth(prof['Iu'][imin:imax], smoothWindow[1], **kwargs_smooth)
            prof_out['Iv'][imin:imax] = wind.smooth(prof['Iv'][imin:imax], smoothWindow[2], **kwargs_smooth)
            prof_out['Iw'][imin:imax] = wind.smooth(prof['Iw'][imin:imax], smoothWindow[3], **kwargs_smooth)
            prof_out['xLu'][imin:imax] = wind.smooth(prof['xLu'][imin:imax], smoothWindow[4], **kwargs_smooth)
            prof_out['xLv'][imin:imax] = wind.smooth(prof['xLv'][imin:imax], smoothWindow[5], **kwargs_smooth)
            prof_out['xLw'][imin:imax] = wind.smooth(prof['xLw'][imin:imax], smoothWindow[6], **kwargs_smooth)
            return prof_out

        def prof_ratio(prof_target, prof_attempt, scaleByFixedRatio, smooth, zMaxCommon=None, zMinCommon=None, include_xLi=True):
            printlog('    Scaling profile ...')
            if not np.array_equal(prof_target.Z, prof_attempt.Z):
                Z = np.unique(np.sort(np.concatenate((prof_target.Z, prof_attempt.Z))))
                prof_attempt = interpToZ(prof_attempt, Z)
                prof_target = interpToZ(prof_target, Z)
            ratio = pd.DataFrame()
            if scaleByFixedRatio:
                printlog("    Scaling type\t\t\t: Fixed roof-height ratio.")
                hIdx = np.argmin(np.abs(prof_target.Z - self.H))
                ratio['Z'] = prof_target['Z']
                ratio['U'] = np.ones_like(prof_target['Z'])*prof_target['U'][hIdx]/prof_attempt['U'][hIdx]
                ratio['Iu'] = np.ones_like(prof_target['Z'])*prof_target['Iu'][hIdx]/prof_attempt['Iu'][hIdx]
                ratio['Iv'] = np.ones_like(prof_target['Z'])*prof_target['Iv'][hIdx]/prof_attempt['Iv'][hIdx]
                ratio['Iw'] = np.ones_like(prof_target['Z'])*prof_target['Iw'][hIdx]/prof_attempt['Iw'][hIdx]
                ratio['xLu'] = np.ones_like(prof_target['Z'])*prof_target['xLu'][hIdx]/prof_attempt['xLu'][hIdx]
                ratio['xLv'] = np.ones_like(prof_target['Z'])*prof_target['xLv'][hIdx]/prof_attempt['xLv'][hIdx]
                ratio['xLw'] = np.ones_like(prof_target['Z'])*prof_target['xLw'][hIdx]/prof_attempt['xLw'][hIdx]
            else:
                printlog("    Scaling type\t\t\t: Full-profile ratio.")
                ratio['Z'] = prof_target['Z']
                ratio['U'] = prof_target['U']/prof_attempt['U']
                ratio['Iu'] = prof_target['Iu']/prof_attempt['Iu']
                ratio['Iv'] = prof_target['Iv']/prof_attempt['Iv']
                ratio['Iw'] = prof_target['Iw']/prof_attempt['Iw']
                ratio['xLu'] = prof_target['xLu']/prof_attempt['xLu']
                ratio['xLv'] = prof_target['xLv']/prof_attempt['xLv']
                ratio['xLw'] = prof_target['xLw']/prof_attempt['xLw']

            if zMinCommon is None:
                printlog(f"    Scaling range\t\t\t: Full profile from min Z = {np.min(ratio['Z'])} ...")
                imin = 0
            else:
                imin = np.argmin(np.abs(ratio['Z'] - zMinCommon))
                printlog("    Scaling range\t\t\t: "+str(zMinCommon)+" < Z < "+str(zMaxCommon))
                ratio['U'][:imin] = ratio['U'][imin]
                ratio['Iu'][:imin] = ratio['Iu'][imin]
                ratio['Iv'][:imin] = ratio['Iv'][imin]
                ratio['Iw'][:imin] = ratio['Iw'][imin]
                ratio['xLu'][:imin] = ratio['xLu'][imin]
                ratio['xLv'][:imin] = ratio['xLv'][imin]
                ratio['xLw'][:imin] = ratio['xLw'][imin]
            if zMaxCommon is None:
                printlog("    Scaling range\t\t\t: Full profile.")
                imax = -1
            else:
                imax = np.argmin(np.abs(ratio['Z'] - zMaxCommon))
                printlog("    Scaling range\t\t\t: "+str(zMinCommon)+" < Z < "+str(zMaxCommon))
                ratio['U'][imax:] = ratio['U'][imax]
                ratio['Iu'][imax:] = ratio['Iu'][imax]
                ratio['Iv'][imax:] = ratio['Iv'][imax]
                ratio['Iw'][imax:] = ratio['Iw'][imax]
                ratio['xLu'][imax:] = ratio['xLu'][imax]
                ratio['xLv'][imax:] = ratio['xLv'][imax]
                ratio['xLw'][imax:] = ratio['xLw'][imax]

            if smooth:
                if applyLimitedSmoothing:
                    printlog("    Smoothing\t\t\t: Limited smoothing.")
                    ratio = smooth_profile(ratio, imin, imax)
                else:
                    ratio = smooth_profile(ratio)
            if not include_xLi:
                ratio['xLu'] = 1.0
                ratio['xLv'] = 1.0
                ratio['xLw'] = 1.0

            return ratio

        def profileObj_to_df(prof):
            df = pd.DataFrame()
            df['Z'] = prof.Z
            df['U'] = prof.U
            df['Iu'] = prof.Iu
            df['Iv'] = prof.Iv
            df['Iw'] = prof.Iw
            df['xLu'] = prof.xLu
            df['xLv'] = prof.xLv
            df['xLw'] = prof.xLw
            return df

        def multiply_profiles(prof1, prof2):
            if not np.array_equal(prof1['Z'], prof2['Z']):
                Z = np.unique(np.sort(np.concatenate((prof1['Z'], prof2['Z']))))
                prof2 = interpToZ(prof2, Z)
                prof1 = interpToZ(prof1, Z)
            prof_out = prof1.copy()
            prof_out['U'] *= prof2['U']
            if compensateFor_xLi_in_Ii:
                prof_out['Iu'] *= prof2['Iu']/prof2['xLu']
                prof_out['Iv'] *= prof2['Iv']/prof2['xLv']
                prof_out['Iw'] *= prof2['Iw']/prof2['xLw']
            else:
                prof_out['Iu'] *= prof2['Iu']
                prof_out['Iv'] *= prof2['Iv']
                prof_out['Iw'] *= prof2['Iw']
            prof_out['xLu'] *= prof2['xLu']
            prof_out['xLv'] *= prof2['xLv']
            prof_out['xLw'] *= prof2['xLw']
            return prof_out

        def getPrintTxt(ratio_i, ratio_cum):
            txt = 'Case name\t\t\t\t\t: '+caseName+'\n'
            txt += 'Description\t\t\t\t\t: '+description+'\n'
            txt += 'Date\t\t\t\t\t\t: '+str(datetime.datetime.now())+'\n'
            txt += 'Scaling rounds\t\t\t\t: '+str(rounds)+'\n'
            txt += 'Scale by fixed Zref ratio\t: '+str(scaleByFixedRefHeightRatio)+'\n'
            txt += 'Scale xLi\t\t\t\t\t: '+str(scale_xLi)+'\n'
            txt += 'Compensate for xLi in Ii\t: '+str(compensateFor_xLi_in_Ii)+'\n'
            txt += 'Smooth the target\t\t\t: '+str(smoothTheTarget)+'\n'
            txt += 'Smooth ratios\t\t\t\t: '+str(smoothRatios)+'\n'
            txt += 'Smooth window (ratios) \t\t: '+str(smoothWindow_ratios)+'\n'
            txt += 'Smooth window (target) \t\t: '+str(smoothWindow_target)+'\n'
            txt += 'Apply limited smoothing\t\t: '+str(applyLimitedSmoothing)+'\n'
            txt += 'Smoothing kwargs\t\t\t: '+str(kwargs_smooth)+'\n'
            txt += 'Zmax scaling\t\t\t\t: '+str(zMax_scaling)+'\n'
            txt += 'Zmin scaling\t\t\t\t: '+str(zMin_scaling)+'\n'
            if dir is not None:
                txt += 'Profile written to\t\t\t: '+str(dir)+'/'+name+'\n'
            else:
                txt += 'Profile written to\t\t\t: None\n'
            txt += f'\nScaling info at roof height (H = {self.H}):\n'

            for r in range(rounds):
                i = np.argmin(np.abs(ratio_i[r+1]['Z'] - self.H))
                txt += f"    Scaling round \t\t\t\t\t: {r+1}\n"
                txt += f"    Fixed ratio scaling \t\t\t: {scaleByFixedRefHeightRatio[r]}\n"
                txt += f"    Scale xLi \t\t\t\t\t\t: {scale_xLi[r]}\n"
                txt += f"    Apply smoothing to the ratio \t: {smoothRatios[r]}\n"
                txt += f"    Ratio \t\t\t\t\t\t\t= {profs_in_names[0]} / {profs_in_names[-1]}\n"
                txt += f"    Roof height ratio (current) \t: U = {ratio_i[r+1]['U'][i]:.3f}, Iu = {ratio_i[r+1]['Iu'][i]:.3f}, Iv = {ratio_i[r+1]['Iv'][i]:.3f}, "+\
                        f"Iw = {ratio_i[r+1]['Iw'][i]:.3f}, xLu = {ratio_i[r+1]['xLu'][i]:.3f}, xLv = {ratio_i[r+1]['xLv'][i]:.3f}, xLw = {ratio_i[r+1]['xLw'][i]:.3f}\n"
                txt += f"    Roof height ratio (cumulative) \t: U = {ratio_cum[r+1]['U'][i]:.3f}, Iu = {ratio_cum[r+1]['Iu'][i]:.3f}, Iv = {ratio_cum[r+1]['Iv'][i]:.3f}, "+\
                        f"Iw = {ratio_cum[r+1]['Iw'][i]:.3f}, xLu = {ratio_cum[r+1]['xLu'][i]:.3f}, xLv = {ratio_cum[r+1]['xLv'][i]:.3f}, xLw = {ratio_cum[r+1]['xLw'][i]:.3f}\n"
                txt += '\n'
            txt += '\n'
            return txt

        '''
        # target_0 = self.targetProfileTable(smoothWindow=smoothWindow, castToUniform=True, kwargs_smooth=kwargs_smooth)
        # r = 0
        # profs_in = [target_0,]
        # profs_out = []
        # profs_in_names = [self.target.name,]
        # profs_out_names = []
        # ratios = [target_0/target_0]
        # ratios[0] = ratios[0].fillna(1.0)
        # scale_xLi = [scale_xLi,]*(rounds+1) if np.isscalar(scale_xLi) else scale_xLi
        # scaleByFixedRefHeightRatio = [scaleByFixedRefHeightRatio,]*(rounds+1) if np.isscalar(scaleByFixedRefHeightRatio) else scaleByFixedRefHeightRatio

        # while r < rounds:
        #     print('Scaling round: '+str(r))
        #     profs_out.append(profileObj_to_df(self.incidents.profiles[r]))
        #     profs_out_names.append(self.incidents.profiles[r].name)
        #     zMax_scaling = 0.8*min(np.array(profs_out[r]['Z'])[-1], np.array(target_0['Z'])[-1]) if zMax_scaling is None else zMax_scaling
        #     zMin_scaling = 1.2*max(np.array(profs_out[r]['Z'])[0], np.array(target_0['Z'])[0]) if zMin_scaling is None else zMin_scaling
        #     ratios.append(prof_ratio(profs_in[r], profs_out[r], zMaxCommon=zMax_scaling, zMinCommon=zMin_scaling, scaleByFixedRatio=scaleByFixedRefHeightRatio[r], 
        #                              include_xLi=scale_xLi[r] ))
        #     ratios[r+1] = multiply_profiles(ratios[r], ratios[r+1],)
        #     profs_in.append(multiply_profiles(profs_in[0], ratios[r+1],))
        #     profs_in_names.append(profs_out_names[r-1]+' (scaled)')
        #     print('    Input \t\t\t\t: '+ str(profs_in_names[r]))
        #     print('    Latest ED LES \t\t\t: '+ str(profs_out_names[r]))
        #     print(f"    Latest roof height ratio \t\t: U = {ratios[r]['U'][0]:.3f}, Iu = {ratios[r]['Iu'][0]:.3f}, Iv = {ratios[r]['Iv'][0]:.3f}, "+
        #           f"Iw = {ratios[r]['Iw'][0]:.3f}, xLu = {ratios[r]['xLu'][0]:.3f}, xLv = {ratios[r]['xLv'][0]:.3f}, xLw = {ratios[r]['xLw'][0]:.3f}")
        #     print(f"    Cumulative roof height ratio \t: U = {ratios[r+1]['U'][0]:.3f}, Iu = {ratios[r+1]['Iu'][0]:.3f}, Iv = {ratios[r+1]['Iv'][0]:.3f}, "+
        #             f"Iw = {ratios[r+1]['Iw'][0]:.3f}, xLu = {ratios[r+1]['xLu'][0]:.3f}, xLv = {ratios[r+1]['xLv'][0]:.3f}, xLw = {ratios[r+1]['xLw'][0]:.3f}\n")
        #     r += 1
        # prof_latest_incident = profileObj_to_df(self.incidents.profiles[r-1]) if rounds > 0 else None
        # ratio = ratios[-1]
        # prof = profs_in[-1]
        '''

        target_0 = self.targetProfileTable(smoothWindow=smoothWindow_target, castToUniform=True, smoothen=smoothTheTarget, kwargs_smooth=kwargs_smooth)
        ratio_i = [prof_ratio(target_0, target_0, True, smoothTheTarget),]
        ratio_cum = ratio_i.copy()
        profs_cum = [target_0,]
        profs_in_names = [self.target.name,]

        smoothRatios = [smoothRatios,]*(rounds+1) if np.isscalar(smoothRatios) else smoothRatios
        
        for r in range(rounds):
            LES = profileObj_to_df(self.incidents.profiles[r])
            profs_in_names.append(self.incidents.profiles[r].name)
            zMax_scaling = 0.8*min(np.array(LES['Z'])[-1], np.array(target_0['Z'])[-1]) if zMax_scaling is None else zMax_scaling
            zMin_scaling = 1.2*max(np.array(LES['Z'])[0], np.array(target_0['Z'])[0]) if zMin_scaling is None else zMin_scaling

            ratio = prof_ratio(target_0, LES, zMaxCommon=zMax_scaling, zMinCommon=zMin_scaling, scaleByFixedRatio=scaleByFixedRefHeightRatio[r],
                               smooth=smoothRatios[r], include_xLi=scale_xLi[r])
            ratio_i.append(ratio)
            ratio_cum.append(multiply_profiles(ratio_cum[-1], ratio))
            profs_cum.append(multiply_profiles(target_0, ratio_cum[r+1]))

        print(getPrintTxt(ratio_i, ratio_cum))
        prof = profs_cum[-1]
        ratio = ratio_cum[-1]

        if dir is not None:
            if not os.path.exists(dir):
                os.makedirs(dir)
            prof.to_csv(dir+'/'+name, index=False, sep=' ', float_format='%.6e', header=False)
            print('Profile written to: '+dir+'/'+name)
            with open(dir+'/'+caseName+'_info.txt', 'w') as f:
                f.write(getPrintTxt(ratio_i, ratio_cum))
                        
        if debugMode:
            toPlot = [profileObj_to_df(self.target), target_0]
            names = ['target'+' ('+self.target.name+')', 'target(smooth)']
            # if r > 0:
            #     toPlot.append(prof_latest_incident)
            #     names.append('latest EDS ('+self.incidents.profiles[r-1].name+')')
            toPlot.append(prof)
            names.append('next_prof')
            plotThese(profs=toPlot, names=names, ratio=ratio, mainTitle=caseName, color=['k', 'k', 'r', 'g'], lwgts=[1, 2, 2, 3], lss=['None', '-', '-', '-'], 
                      markers=['.', 'None', 'None', 'None'])
            
            toPlot = [profileObj_to_df(self.target), target_0]
            names = ['target'+' ('+self.target.name+')', 'target(smooth)']
            for r in range(rounds+1):
                toPlot.append(profs_cum[r])
                names.append(profs_in_names[r])
            plotThese(profs=toPlot, names=names, ratio=None, mainTitle=caseName, 
                      color=['k', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'tab:gray', 'tab:olive', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:blue', 'tab:green', 'tab:red',], 
                      lwgts=[1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, ], 
                      lss=['None', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', ],
                      markers=['.', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None'])
            
        return prof

    def plotProfiles(self, figsize=[15,12], zLim=[0,10], normalize=True, lw = 1.5, ms=3,
                            xLabel=None, zLabel=None, xLimits_U=None, xLimits_Iu=None, xLimits_Iv=None, xLimits_Iw=None, 
                            xLimits_xLu=None, xLimits_xLv=None, xLimits_xLw=None, xLimits_uw=None, 
                            includeInflows=True, includeIncidents=True, includeRefProfiles=True,
                            lgnd_kwargs={'bbox_to_anchor': (0.5, 0.5), 'loc': 'center', 'ncol': 1},
                            kwargs_plt=None, kwargs_ax={},
                            kwargs_StatsTable={}):
        longListOfColors = ['r', 'b', 'g', 'm', 'k', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'navy', ]
       
        if kwargs_plt is None:
            kwargs_plt = []
            kwargs_plt.append({'color':'k', 'lw':0.3, 'ls':'-', 'marker':'.', 'ms':ms})
            if self.inflows is not None and includeInflows:
                for i, _ in enumerate(self.inflows.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'--'})
            if self.incidents is not None and includeIncidents:
                for i, _ in enumerate(self.incidents.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'-'})
            if self.refProfiles is not None and includeRefProfiles:
                for i, _ in enumerate(self.refProfiles.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw+1, 'ls':'-.'})

        profs = []
        if self.target is not None:
            profs.append(self.target)
        if self.inflows is not None and includeInflows:
            profs.extend(self.inflows.profiles)
        if self.incidents is not None and includeIncidents:
            profs.extend(self.incidents.profiles)
        if self.refProfiles is not None and includeRefProfiles:
            profs.extend(self.refProfiles.profiles)
        profs = wind.Profiles(profs)

        table = profs.plotRefHeightStatsTable(colTxtColors=[p['color'] for p in kwargs_plt], 
                                      fontSz=12,
                                      **kwargs_StatsTable)

        fig = profs.plotProfile_basic2(figsize=figsize, yLimits=zLim, normalize=normalize, xLimits_U=xLimits_U, xLimits_Iu=xLimits_Iu, xLimits_Iv=xLimits_Iv, 
                                            xLimits_Iw=xLimits_Iw, xLimits_uw=xLimits_uw, xLimits_xLu=xLimits_xLu, xLimits_xLv=xLimits_xLv, xLimits_xLw=xLimits_xLw, 
                                            xLabel=xLabel, zLabel=zLabel, lgnd_kwargs=lgnd_kwargs, kwargs_plt=kwargs_plt, kwargs_ax=kwargs_ax)
        
        return fig, table
    
    def plotSpectra(self, figSize=[15,5], lw = 1.5, ms=3,
                    xLimits='auto', yLimits='auto', includeInflows=True, includeIncidents=True, includeRefProfiles=True,
                    kwargs_plt=None, 
                    kwargs_pltSpectra={}):
        longListOfColors = ['r', 'b', 'g', 'm', 'k', 'c', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'navy', ]
        if kwargs_plt is None:
            kwargs_plt = []
            kwargs_plt.append({'color':'k', 'lw':1, 'ls':'-', 'marker':'None', 'ms':ms, 'alpha':0.7})
            if self.inflows is not None and includeInflows:
                for i, prof in enumerate(self.inflows.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'--', 'alpha':0.7})
            if self.incidents is not None and includeIncidents:
                for i, prof in enumerate(self.incidents.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw, 'ls':'-', 'alpha':0.7})
            if self.refProfiles is not None and includeRefProfiles:
                for i, prof in enumerate(self.refProfiles.profiles):
                    kwargs_plt.append({'color':longListOfColors[i], 'lw':lw+1, 'ls':'-'})
        
        profs = []
        if self.target is not None:
            profs.append(self.target)
        if self.inflows is not None and includeInflows:
            profs.extend(self.inflows.profiles)
        if self.incidents is not None and includeIncidents:
            profs.extend(self.incidents.profiles)
        if self.refProfiles is not None and includeRefProfiles:
            profs.extend(self.refProfiles.profiles)
        profs = wind.Profiles(profs)

        return profs.plotSpectra(figsize=figSize, xLimits=xLimits, yLimits=yLimits, kwargs_plt=kwargs_plt, **kwargs_pltSpectra)













#===============================================================================
#=============================== GRAVEYARD =====================================
#===============================================================================
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
