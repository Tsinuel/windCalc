# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 22:20:15 2023

@author: Tsinuel
"""

import sys
import windOF

def main(args):
    print(args)
    caseDir = args[0]
    sectionName = args[1]
    ratioFile = args[2]
    profFile = args[3]
    overwrite = bool(float(args[4]))
    checkMatchingPoints = bool(float(args[5]))
    outDir = args[6]
    shiftTimeBy = float(args[7])
    timeRange = [float(args[8]), float(args[9])]

    print("caseDir: ", caseDir)
    print("sectionName: ", sectionName)
    print("ratioFile: ", ratioFile)
    print("profFile: ", profFile)
    print("overwrite: ", overwrite)
    print("checkMatchingPoints: ", checkMatchingPoints)
    print("outDir: ", outDir)
    print("shiftTimeBy: ", shiftTimeBy)
    print("timeRange: ", timeRange)

    windOF.scaleInflowFromSectionSample(caseDir=caseDir, 
                                        sectionName=sectionName,
                                        ratioFile=ratioFile, 
                                        profFile=profFile, 
                                        overwrite=overwrite,
                                        checkMatchingPoints=checkMatchingPoints,
                                        outputDir=outDir,
                                        timeRange=timeRange,
                                        shiftTimeBy=shiftTimeBy,
                                        )

if __name__ == "__main__":
    main(sys.argv[1:])

