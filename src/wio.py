# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 9:20:05 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
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
    # if first:
    #     p_tap = np.reshape(p[idx],[-1,1])
    #     first = False
    # else:
    #     p_tap = np.concatenate((p_tap, np.reshape(p[idx],[-1,1])), axis=1)

def readPSSfile(file_pssr,file_pssd):
    import scipy.io as sio
    
    #def readPSSfile(file_pssr,file_pssd):
    with open(file_pssr,'rb') as f:
        tester=np.multiply(np.fromfile(f,dtype='int16',count=-1),10/65536)
    
    WTTDATALOG=sio.loadmat(file_pssd)['WTTDATALOG']
    channel_count=int(WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['StopAdd'][0,0][0][0]-WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['StartAdd'][0,0][0][0]+1)
    
    modules_used_count=int(WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['ModulesInUse'][0,0][0].size)
    modules_used=WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['ModulesInUse'][0,0][0]
    
    data=np.reshape(tester,
        (int(np.divide(np.divide(tester.size,modules_used_count),channel_count)),
        int(np.multiply(modules_used_count,channel_count))))
    
    for analog_module in WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['AnalogModules'][0,0][0]:
        analog_index=np.arange(analog_module-1,data.shape[1],modules_used_count)
        analog=data[:,analog_module-1:data.shape[1]:modules_used_count]
        data=np.delete(data,analog_index,axis=1)
    
        pth_modules=modules_used_count-WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['AnalogModules'][0,0][0].size
        
        if WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['ValidCal'][0,0][0][0]==1:
            toBeReshaped = WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['CAL'][0,0][0][0]['Z'][0:pth_modules,:]
            newShape = [1,((pth_modules)*channel_count)]
            reshaped = np.reshape(toBeReshaped, newShape, 'F') #.transpose()
            numer = data - reshaped #np.tile(reshaped, (data.shape[0],1))
            toBeReshaped2 = WTTDATALOG['APPSPE'][0,0][0,0][0][0][0]['MAN']['CAL'][0,0][0][0]['Q'][0:pth_modules,:]
            newShape2 = [1,((pth_modules)*channel_count)]
            reshaped2 = np.reshape(toBeReshaped2,newShape2, 'F') #.transpose()
            # denom = np.tile(reshaped2, (data.shape[0],1))
            data=np.divide(numer,reshaped2)    
    cp_data=np.zeros((data.shape))
    
    for i in range(int(data.shape[1]/channel_count)-1):
        cp_data[:,i*channel_count:((i+1)*channel_count):1]=data[:,i:int(data.shape[1]):pth_modules]

    cp_data=np.array(cp_data,order='F')
    analog=np.array(analog)
    header=[]
    for mod in modules_used:
        for i in range(16):
            header.append(str(mod)+"{0:0>3}".format(i+1))

    return cp_data,analog,WTTDATALOG

def readCobraProbeData():
    raise NotImplemented

def writeSpecFile(file,keyword,value,writeType):
    pass





#===============================================================================
#================================ CLASSES ======================================
#===============================================================================
