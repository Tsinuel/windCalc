# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 01:40:48 2022

@author: Tsinu
"""

import sys
import windOF

def main(args):
    print(args)
    
    caseDir = args[0]
    tMax = float(args[1])
    
    inletDir = caseDir+'constant/boundaryData/inlet/'
    outPath = caseDir+'constant/boundaryData/sample_'
    figFile = caseDir+'constant/boundaryData/sampleProfiles.pdf'    
    H = 0.08
    
    windOF.extractSampleProfileFromInflow(inletDir, outPath, figFile, tMax, H)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
