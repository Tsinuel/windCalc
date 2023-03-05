# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 9:20:05 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
import os
import glob
import warnings
import wind as wd


#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================
BAROCEL_RANGE_FACTORS = { # Obtained from Anthony @ BLWTL
    "Range":        [1, .3, .1, .03, .01, .003, .001],
    "Range_factor": [94.606, 51.818, 29.917, 16., 9.461, 5.182, 2.992],
}

#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

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

class BLWTL_HFPI:
    def __init__(self,
                 caseDir:str=None,
                 Z_MainPitot=1.48,
                 pitotChnl_main:int=None,
                 pitotChnl_xref:int=None,
                 pitotChnl_20in:int=None,
                 pitotChnl_Uh:int=None,
                 Ntaps=None,
                 barocelRangeFactor=29.917,
                 ) -> None:
        self.caseDir = caseDir
        self.Z_MainPitot = Z_MainPitot
        self.pitotChnl_main = pitotChnl_main
        self.pitotChnl_xref = pitotChnl_xref
        self.pitotChnl_20in = pitotChnl_20in
        self.pitotChnl_Uh = pitotChnl_Uh
        self.barocelRangeFactor = barocelRangeFactor
        self.Ntaps = Ntaps

        self.files_pssd = None
        self.files_pssr = None
        self.AoA = None
        self.UrefMain_TH = None
        self.WTTDATALOG = None
        self.CpTH = None
        self.sampleRate = None

        self.Update()
        
    def Update(self):
        pass

    def read(self):
        self.files_pssr = glob.glob(os.path.join(self.caseDir, '*.pssr'))
        file_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in self.files_pssr]
        self.files_pssd = [os.path.join(self.caseDir, file+'.pssd') for file in file_names]

        self.AoA = np.array([])
        self.sampleRate = np.array([])
        self.WTTDATALOG = []
        N_AoA = len(self.files_pssd)

        for d,(file_pssd,file_pssr) in enumerate(zip(self.files_pssd,self.files_pssr)):
            cp_data,analog,WTTDATALOG = readPSSfile(file_pssr,file_pssd)
            self.WTTDATALOG.append(WTTDATALOG)

            if d == 0:
                N_t,Ntaps = np.shape(cp_data)
                self.CpTH = np.zeros((N_AoA,Ntaps,N_t)) # [N_AoA,Ntaps,Ntime]
                self.UrefMain_TH = np.zeros((N_AoA,N_t))
            self.CpTH[d,:,:] = np.transpose(cp_data)
            self.UrefMain_TH[d,:] = self.barocelRangeFactor * np.sqrt(analog[:,self.pitotChnl_main]) * wd.fps2mps
            self.AoA = np.append(self.AoA,WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][16][0][0])
            self.sampleRate = np.append(self.sampleRate, WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][12][0][0])

        self.UrefMain = np.mean(self.UrefMain_TH,axis=1)
        
