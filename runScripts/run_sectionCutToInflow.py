# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 22:20:15 2023

@author: Tsinuel
"""

import sys
import foam

def main(args):
    print(args)
    caseDir = args[0]
    sectionName = args[1]
    fileName = args[2]
    shiftTimeBy = float(args[3])
    overwrite = bool(float(args[4]))
    checkMatchingPoints = bool(float(args[5]))
    outDir = args[6]

    print("caseDir: ", caseDir)
    print("sectionName: ", sectionName)
    print("fileName: ", fileName)
    print("shiftTimeBy: ", shiftTimeBy)
    print("overwrite: ", overwrite)
    print("outDir: ", outDir)
        
    foam.convertSectionSampleToBoundaryDataInflow(caseDir=caseDir, 
                                                sectionName=sectionName, 
                                                fileName=fileName,
                                                shiftTimeBy=shiftTimeBy,
                                                overwrite=overwrite,
                                                checkMatchingPoints=checkMatchingPoints,
                                                pointDistanceTolerance=1e-3,
                                                showLog=True,
                                                timeOutputPrecision=6,
                                                detailedLog=False,
                                                outDir=outDir,
                                                )
    
if __name__ == "__main__":
    main(sys.argv[1:])

