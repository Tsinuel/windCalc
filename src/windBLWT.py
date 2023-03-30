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
# 
# Obtained from Anthony @ BLWTL
BAROCEL_RANGE_FACTORS = { 
    "Range":        [1, .3, .1, .03, .01, .003, .001],
    "Range_factor": [94.606, 51.818, 29.917, 16., 9.461, 5.182, 2.992],
}

XREF_PITOT_QUEUE_TOLERANCE = 0.01 # xref

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
                 userNotes=None,
                 Z_MainPitot=1.48,
                 analogChannels_idxs:dict=None,
                 pressureExtraChannels_tapNos:dict=None,
                 lowpassFreq=None,
                 Ntaps=None,
                 barocelRangeFactor=29.917,
                 ) -> None:
        self.caseDir = caseDir
        self.userNotes = userNotes
        self.Z_MainPitot = Z_MainPitot
        if analogChannels_idxs is None:
            analogChannels_ex = {
                'main_pitot': 1,
                'xref_pitot': 3,
                'sync_switch': 6,
            }
            print("analogChannels is not provided. It has the form of:")
            print(analogChannels_ex)
        self.analogChannels_idxs = analogChannels_idxs
        
        if pressureExtraChannels_tapNos is None:
            pressureExtraChannels_ex = {
                'main_pitot_zero': 2909,
                'main_pitot_queue': 2910,
                '20inch_pitot_zero': 2907,
                '20inch_pitot_queue': 2908,
                'Uh_pitot_zero': 2905,
                'Uh_pitot_queue': 2906,
            }
            print("pressureExtraChannels_tapNos is not provided. It has the form of:")
            print(pressureExtraChannels_ex)
        self.pressureExtraChannels_tapNos = pressureExtraChannels_tapNos
        self.lowpassFreq = lowpassFreq

        self.barocelRangeFactor = barocelRangeFactor
        self.Ntaps = Ntaps

        self.files_pssd = None
        self.files_pssr = None
        self.AoA = None
        self.WTTDATALOG = None
        self.CpTH: np.array() = None
        self.sampleRate_orig = None
        self.sampleRate = None
        self.testDuration = None
        self.floorExposure = None
        self.tapNos = None
        self.analogChannels = None
        self.pressureExtraChannels = None

        self.Update()
        
    def Update(self):
        self.read()
        

    def read(self):
        self.files_pssr = glob.glob(os.path.join(self.caseDir, '*.pssr'))
        file_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in self.files_pssr]
        self.files_pssd = [os.path.join(self.caseDir, file+'.pssd') for file in file_names]

        self.AoA = np.array([])
        self.sampleRate_orig = np.array([])
        self.WTTDATALOG = []
        N_AoA = len(self.files_pssd)

        for d,(file_pssd,file_pssr) in enumerate(zip(self.files_pssd,self.files_pssr)):
            cp_data,analog,WTTDATALOG = readPSSfile(file_pssr,file_pssd)
            analog = np.transpose(analog)
            cp_data = np.transpose(cp_data)
            self.WTTDATALOG.append(WTTDATALOG)
            self.AoA = np.append(self.AoA,float(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][24][0][0][5][0][0]))
            sampleFreq = WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][12][0][0]
            self.sampleRate_orig = np.append(self.sampleRate_orig, sampleFreq)
            if self.lowpassFreq is not None:
                cp_data = wd.lowpass(cp_data,fs=sampleFreq,fc=self.lowpassFreq, axis=1, resample=True)
                analog = wd.lowpass(analog,fs=sampleFreq,fc=self.lowpassFreq, axis=1, resample=True)
            if d == 0:
                Ntaps,N_t = np.shape(cp_data)
                CpTH = np.zeros((N_AoA,Ntaps,N_t))
                Nchnl,_ = np.shape(analog)
                anTH = np.zeros((N_AoA,Nchnl,N_t))
            CpTH[d,:,:] = cp_data[0:Ntaps,:]
            anTH[d,:,:] = analog
        if self.lowpassFreq is None:
            self.sampleRate = self.sampleRate_orig
        else:
            self.sampleRate = np.ones_like(self.sampleRate_orig)*self.lowpassFreq

        tapNos = list(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][3][0])
        self.barocelRangeFactor = WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][7][0][3]
        self.testDuration = float(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][6][0][0])
        self.floorExposure = WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][7]
        self.CpTH = CpTH[:,:self.Ntaps,:]
        self.analogData = anTH
        self.tapNos = tapNos[:self.Ntaps]

        if self.analogChannels_idxs is not None:
            self.analogChannels = {}
            for dId in self.analogChannels_idxs:
                idx = self.analogChannels_idxs[dId]
                self.analogChannels[dId] = anTH[:,idx,:]

        if self.pressureExtraChannels_tapNos is not None:
            self.pressureExtraChannels = {}
            for dId in self.pressureExtraChannels_tapNos:
                tNo = self.pressureExtraChannels_tapNos[dId]
                idx = tapNos.index(tNo)
                self.pressureExtraChannels[dId] = CpTH[:,idx,:]

    @property
    def Uref(self):
        urefTH = self.Uref_TH
        if urefTH is None:
            return None
        return np.mean(urefTH,axis=-1)

    @property
    def Uref_TH(self):
        if self.analogChannels is None:
            return None
        if not 'main_pitot' in self.analogChannels:
            return None
        return np.sqrt(self.analogChannels['main_pitot'])  * self.barocelRangeFactor * wd.fps2mps

    @property
    def U_XRef(self):
        uTH = self.U_XRef_TH
        if uTH is None:
            return None
        return np.mean(uTH,axis=-1)

    @property
    def U_XRef_TH(self):
        if self.analogChannels is None:
            return None
        if not 'xref_pitot' in self.analogChannels:
            return None
        return np.sqrt(self.analogChannels['xref_pitot'])  * self.barocelRangeFactor * wd.fps2mps
