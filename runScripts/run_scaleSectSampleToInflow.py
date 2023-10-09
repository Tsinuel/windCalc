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

    print("caseDir: \t\t", caseDir)
    print("sectionName: \t\t", sectionName)
    print("ratioFile: \t\t", ratioFile)
    print("profFile: \t\t", profFile)
    print("overwrite: \t\t", overwrite)
    print("checkMatchingPoints: \t", checkMatchingPoints)
    print("outDir: \t\t", outDir)
    print("shiftTimeBy: \t\t", shiftTimeBy)
    print("timeRange: \t\t", timeRange)

    windOF.scaleInflowFromSectionSample(caseDir=caseDir, 
                                        sectionName=sectionName,
                                        ratioFile=ratioFile, 
                                        profFile=profFile, 
                                        overwrite=overwrite,
                                        checkMatchingPoints=checkMatchingPoints,
                                        outputDir=outDir,
                                        timeRange=timeRange,
                                        shiftTimeBy=shiftTimeBy,
                                        timeOutputPrecision=10,
                                        )

if __name__ == "__main__":
    main(sys.argv[1:])

