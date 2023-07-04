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
    overwrite = bool(args[4])
    outDir = args[5]

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
                                                pointDistanceTolerance=1e-3,
                                                showLog=True,
                                                timeOutputPrecision=6,
                                                detailedLog=False,
                                                outDir=outDir,
                                                )
    
if __name__ == "__main__":
    main(sys.argv[1:])

