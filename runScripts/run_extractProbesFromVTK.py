# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 01:40:48 2024

@author: Tsinu
"""

import sys
import numpy as np
import windOF as foam

def main(args):
    print(args)
    vtkDir = args[0]
    fieldName = args[1] # e.g. 'U'
    filePrefix = args[2] # e.g. 'U_yNorm_'
    timeMultiplier = float(args[3]) # factor to convert the integral form of the time in the vtk file names to seconds
    outputDir = args[4]
    outFilePrefix = args[5] # e.g. 'SurfProbExtr_'
    numOfSplitWriteFiles = int(args[6])
    nProcessors = int(args[7])
    
    z1 = np.linspace(0, 0.08, 100)
    x = np.full_like(z1, 0.0)
    y = np.full_like(z1, 0.0)

    x2 = np.linspace(0, 1.6, 1000)
    x = np.append(x, x2)
    y = np.append(y, np.full_like(x2, 0))
    z = np.append(z1, np.full_like(x2, 0.08))
    step_surf_points = np.column_stack([x, y, z])
    
    foam.extractDataAtPoints(vtkDir=vtkDir, 
                            query_points=step_surf_points, 
                            fieldName=fieldName,
                            filePrefix=filePrefix, 
                            timeMultiplier=timeMultiplier,
                            num_processors=nProcessors,
                            outputDir=outputDir,
                            numOfSplitWriteFiles=numOfSplitWriteFiles,
                            outFilePrefix=outFilePrefix,
                            showLog=True,
                            kwargs_extSingle={'distanceTolerance':1e-3}, 
                            )
    
if __name__ == "__main__":
    main(sys.argv[1:])
