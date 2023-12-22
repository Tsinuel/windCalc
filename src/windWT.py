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
import json


#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================
# 
# Obtained from Anthony @ BLWTL
BAROCEL_RANGE_FACTORS = { 
    "Range":        [1, .3, .1, .03, .01, .003, .001],
    "Range_factor": [94.606, 51.818, 29.917, 16., 9.461, 5.182, 2.992],
}

XCHECK_PITOT_Q_TOLERANCE = 0.01 # xref

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
                 Ntaps=-1,
                 AoAsToRead=None,
                 trimTimeStart=None,
                 trimTimeEnd=None,
                 ) -> None:
        self.caseDir = caseDir
        self.userNotes = userNotes
        self.Z_MainPitot = Z_MainPitot
        self.lowpassFreq = lowpassFreq
        self.Ntaps = Ntaps
        self.AoAsToRead = AoAsToRead
        self.trimTimeStart = trimTimeStart
        self.trimTimeEnd = trimTimeEnd

        if analogChannels_idxs is None:
            analogChannels_ex = {
                'main_pitot': 0,
                'xcheck_pitot': 3,
                'sync_switch': 6,
            }
            print("Index (zero-based) of 'analogChannels' is not provided. It has the form of:")
            print(json.dumps(analogChannels_ex, indent=4))
        self.analogChannels_idxs = analogChannels_idxs
        
        if pressureExtraChannels_tapNos is None:
            pressureExtraChannels_ex = {
                'main_pitot_zero': 2909,
                'main_pitot_q': 2910,
                '20inch_pitot_zero': 2907,
                '20inch_pitot_q': 2908,
                'Uh_pitot_zero': 2905,
                'Uh_pitot_q': 2906,
            }
            print("'pressureExtraChannels_tapNos' is not provided. It has the form of:")
            print(json.dumps(pressureExtraChannels_ex, indent=4))
        self.pressureExtraChannels_tapNos = pressureExtraChannels_tapNos

        self.files_pssd = None
        self.files_pssr = None
        self.AoA = None
        self.all_WTTDATALOGs = None
        self.CpTH: np.array() = None
        self.sampleRate_orig = None
        self.sampleRate = None
        self.testDuration = None
        self.floorExposure = None
        self.tapNos = None
        self.analogData = None
        self.pressureExtraChannels = None
        self.barocelRangeFactors = None
        self.CAL_factors = []
        
        self.Refresh()
        
    def Refresh(self):
        self.read()

    def read(self):
        print("Reading HFPI data from: {}".format(self.caseDir))
        files_pssr = glob.glob(os.path.join(self.caseDir, '*.pssr'))
        file_names = [os.path.splitext(os.path.basename(file_name))[0] for file_name in files_pssr]
        files_pssd = [os.path.join(self.caseDir, file+'.pssd') for file in file_names]
        self.files_pssr = []
        self.files_pssd = []

        self.AoA = np.array([])
        self.sampleRate_orig = np.array([])
        self.all_WTTDATALOGs = []
        N_RawFiles = len(files_pssd) 
        N_AoA = N_RawFiles if self.AoAsToRead is None else len(self.AoAsToRead)

        i = 0
        for d,(file_pssd,file_pssr) in enumerate(zip(files_pssd,files_pssr)):
            print("   Reading file: {}".format(file_pssd))
            cp_data,analog,WTTDATALOG = readPSSfile(file_pssr,file_pssd)
            AoA = float(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][24][0][0][5][0][0])
            if self.AoAsToRead is not None:
                if not AoA in self.AoAsToRead:
                    print(f"       AoA {AoA} is not in the list of AoAs to read: {self.AoAsToRead}. Skipping.")
                    continue
            self.files_pssd.append(file_pssd)
            self.files_pssr.append(file_pssr)
            analog = np.transpose(analog)
            cp_data = np.transpose(cp_data)
            self.all_WTTDATALOGs.append(WTTDATALOG)
            self.AoA = np.append(self.AoA,AoA)
            self.CAL_factors.append({
                                'Z':        WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][2][0][0][1], 
                                'Zrms':     WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][2][0][0][2],
                                'Q':        WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][2][0][0][3],
                                'Qerr':     WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][2][0][0][4],
                                'MainH2O':  WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][2][0][0][5][0][0],
                                })
            sampleFreq = np.squeeze(WTTDATALOG["SampleRate"][0][0])
            self.sampleRate_orig = np.append(self.sampleRate_orig, sampleFreq)
            if self.lowpassFreq is not None:
                cp_data = wd.lowpass(cp_data,fs=sampleFreq,fc=self.lowpassFreq, axis=1, resample=True)
                analog = wd.lowpass(analog,fs=sampleFreq,fc=self.lowpassFreq, axis=1, resample=True)
            if d == 0:
                Ntaps,N_t = np.shape(cp_data)
                CpTH = np.zeros((N_AoA,Ntaps,N_t))
                Nchnl,_ = np.shape(analog)
                anTH = np.zeros((N_AoA,Nchnl,N_t))
            CpTH[i,:,:] = cp_data[0:Ntaps,:]
            anTH[i,:,:] = analog
            i += 1
            
        if self.lowpassFreq is None:
            self.sampleRate = self.sampleRate_orig
        else:
            self.sampleRate = np.ones_like(self.sampleRate_orig)*self.lowpassFreq

        tapNos = list(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][3][0])
        self.barocelRangeFactors = {
                                    'main_pitot':   WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][7][0][3],
                                    'xcheck_pitot': WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][4][0][0][8][0][3]
                                    }
        self.testDuration = float(WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][6][0][0])
        self.floorExposure = WTTDATALOG["APPSPE"][0][0][0][0][0][0][0][7]
        self.CpTH = CpTH[:,:self.Ntaps,:]
        self.analogData = anTH
        self.tapNos = tapNos[:self.Ntaps]

        if self.analogChannels_idxs is not None:
            self.analogData = {}
            for dId in self.analogChannels_idxs:
                idx = self.analogChannels_idxs[dId]
                self.analogData[dId] = anTH[:,idx,:]

        if self.pressureExtraChannels_tapNos is not None:
            self.pressureExtraChannels = {}
            for dId in self.pressureExtraChannels_tapNos:
                tNo = self.pressureExtraChannels_tapNos[dId]
                idx = tapNos.index(tNo)
                self.pressureExtraChannels[dId] = CpTH[:,idx,:]

        print("   A total of {} AoA were read.".format(N_AoA))
        print("   Shape of CpTH: {}".format(np.shape(self.CpTH)))
        print("   Done reading HFPI data.\n")

    @property
    def Uref(self):
        urefTH = self.Uref_TH
        if urefTH is None:
            return None
        return np.mean(urefTH,axis=-1)

    @property
    def Uref_TH(self):
        if self.analogData is None:
            print("No analog data found.")
            return None
        if not 'main_pitot' in self.analogData.keys():
            print("The key 'main_pitot' is not found in the analog data dictionary.")
            return None
        if not 'main_pitot' in self.barocelRangeFactors.keys():
            print("The key 'main_pitot' is not found in the barocelRangeFactors dictionary.")
            return None
        voltage = np.asarray(self.analogData['main_pitot'])
        if np.any(voltage < 0.0):
            warnings.warn("Negative voltage detected in main pitot. Assuming that the pitot ID is not correct.")
            return None
        return np.sqrt(voltage)  * self.barocelRangeFactors['main_pitot'] * wd.UNIT_CONV['fps2mps']

    @property
    def Uref_Xchk(self):
        uTH = self.Uref_Xchk_TH
        if uTH is None:
            return None
        return np.mean(uTH,axis=-1)

    @property
    def Uref_Xchk_TH(self):
        if self.analogData is None:
            print("No analog channel data found.")
            return None
        if not 'xcheck_pitot' in self.analogData.keys():
            print("The key 'xcheck_pitot' is not found in the analog data dictionary.")
            return None
        if not 'xcheck_pitot' in self.barocelRangeFactors.keys():
            print("The key 'xcheck_pitot' is not found in the barocelRangeFactors dictionary.")
            return None
        voltage = np.asarray(self.analogData['xcheck_pitot'])
        if np.any(voltage < 0):
            warnings.warn("Negative voltage detected in cross-check pitot. Assuming that the pitot ID is not correct.")
            return None
        return np.sqrt(voltage)  * self.barocelRangeFactors['xcheck_pitot'] * wd.UNIT_CONV['fps2mps']

    @property
    def description(self):
        description = "HFPI data from BLWTL\n"
        description += "Case directory: {}\n".format(self.caseDir)
        description += "User notes: {}\n".format(self.userNotes)
        description += "Main pitot Z: {} m\n".format(self.Z_MainPitot)
        description += "Lowpass frequency: {} Hz\n".format(self.lowpassFreq)
        description += "Barocel range factor: {}\n".format(json.dumps(self.barocelRangeFactors, indent=4))
        description += "Ntaps: {}\n".format(self.Ntaps)
        description += "AoA: {} deg.\n".format(self.AoA)
        description += "Sample rate (raw): {} Hz\n".format(self.sampleRate_orig)
        description += "Sample rate (filtered): {} Hz\n".format(self.sampleRate)
        description += "Test duration: {} sec.\n".format(self.testDuration)
        description += "Floor exposure: \n\t\tBank\tBlock height (in)\n" #.format(self.floorExposure)
        description +=              "".join(["\t\t"+str(int(item[0]))+"\t"+str(item[1])+"\n" for item in self.floorExposure]) 
        description += "Analog channels indecies: {}\n".format(json.dumps(self.analogChannels_idxs, indent=4))
        description += "Pressure extra channels tap Nos.: {}\n".format(json.dumps(self.pressureExtraChannels_tapNos, indent=4))

        return description

    @property
    def trimTimeStartIdx(self):
        if self.trimTimeStart is None:
            return 0
        sf = self.sampleRate if np.isscalar(self.sampleRate) else self.sampleRate[0]
        return int(self.trimTimeStart * sf)
    
    @property
    def trimTimeEndIdx(self):
        if self.trimTimeEnd is None:
            return -1
        sf = self.sampleRate if np.isscalar(self.sampleRate) else self.sampleRate[0]
        return int(self.trimTimeEnd * sf)
    
    def indexOfTaps(self, tapNos):
        if isinstance(tapNos, int):
            tapNos = [tapNos]
        return [self.tapNos.index(tapNo) for tapNo in tapNos]
    
    def CpOfT_atTaps(self, tapNos, AoA=None):
        if AoA is None:
            AoA = self.AoA
        idx = self.indexOfTaps(tapNos)
        return self.CpTH[:,idx,:] if np.isscalar(AoA) else self.CpTH[:,idx,:][:,np.newaxis,:]