# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 01:40:48 2022

@author: Tsinu
"""

import sys
import os
import numpy as np

def main(args):
    print(args)
    vtkDir = args[0]
    vtkOutDir = args[1]
    fld = args[2]
    vtkName = args[3]
    
    import shutil
    vtkTimes = os.listdir(vtkDir)
    if not os.path.exists(vtkOutDir):
        os.mkdir(vtkOutDir)
    vtkTimesNum = np.asarray(vtkTimes,dtype=float)
    vtkTimesSorted = sorted(zip(vtkTimesNum,vtkTimes),key=lambda x: x[0])
    vtkTimesSorted = list(vtkTimesSorted)
    for v in vtkTimesSorted:
        srcFile = vtkDir+'/'+v[1]+'/'+vtkName
        newName = fld+'_'+str(int(round(v[0]*10000,4))).zfill(7)+'.vtk'
        destFile = vtkOutDir+'/'+newName
        print(srcFile+'-->'+newName)
        shutil.copy(srcFile,destFile)
    
if __name__ == "__main__":
    main(sys.argv[1:])
