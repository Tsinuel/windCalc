# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 9:20:05 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
import os
import warnings


#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================


#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

def extractTapDataFromVTK(vtkFile,tapCoords):
    import vtkmodules.all as vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from scipy.spatial import KDTree
    from multiprocessing import Pool

    # vtkDir = r'..\data\ttuMSfl0450.0\postProcessing\wallPressure'
    # vtkTimes = os.listdir(vtkDir)
    # vtkName = 'p_building.vtk'
    # file = r'D:\OneDrive - The University of Western Ontario\Documents\PhD\Thesis\CFD_simulations\2022_08_TTU_WL_LES\OF_inputFiles\45deg\WTtaps_45deg.csv'
    # tapCoords = pd.read_csv(file)
    # print(vtkTimes)

    reader = vtk.vtkPolyDataReader()
    # first = True
    # for vtk_i in vtkTimes:
    # vtkFile = vtkDir+'/'+vtk_i+'/'+vtkName
    print(vtkFile)
    reader.SetFileName(vtkFile)
    reader.ReadAllFieldsOn()
    reader.Update()
    polydata = reader.GetOutput()
    
    # if first:
    nCells = polydata.GetNumberOfCells()
    cellCenters = np.empty([nCells,3])
    cells = KDTree(cellCenters)
    distance,idx = cells.query(tapCoords)

    # for i in range(nCells):
    #     pts = polydata.GetCell(i).GetPoints()    
    #     cell_i_vrtx = np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())])
    #     cellCenters[i,:] = np.mean(cell_i_vrtx,axis=0) 

    p = vtk_to_numpy(polydata.GetCellData().GetArray(0))
    return np.reshape(p[idx],[-1,1])

def extractTapDataFromCSV(fileDir,tapCoords,writeToFile=True,showLog=True):
    import glob
    from scipy.spatial import KDTree

    data = pd.read_csv(fileDir+"/points.csv")
    pts = data.to_numpy()
    cells = KDTree(pts)
    distance,idx = cells.query(tapCoords)
    newPts = pts[idx,:]

    files = glob.glob(os.path.join(fileDir, "p_*.csv"))
    times = [int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(fileDir) if filename.startswith("p_")]
    files = sorted(list(zip(files, np.asarray(times).astype(int))), key=lambda x: x[1])
    files = [x[0] for x in files]

    first = True
    for f in files:
        if showLog:
            print("Reading: "+f+"   ...")
        data = pd.read_csv(f)
        if first:
            p = np.reshape(data.to_numpy()[idx],[-1,1])
            first = False
        else:
            p = np.concatenate((p,np.reshape(data.to_numpy()[idx],[-1,1])),axis=1)
    if writeToFile:
        np.save(fileDir+"/all_pOfT",p)
        np.save(fileDir+"/tapCoords_closest",newPts)
    
    return p,newPts,distance

def writeSpecFile(file,keyword,value,writeType):
    pass



#===============================================================================
#================================ CLASSES ======================================
#===============================================================================


