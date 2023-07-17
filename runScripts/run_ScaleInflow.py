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
    tMin = float(args[1])
    tMax = float(args[2])
    H = float(args[3])
    write = bool(float(args[4]))
    windOF.scaleInflowData(caseDir, tMin, tMax, H, writeInflow=write, smplName="sampleInfl")
    
if __name__ == "__main__":
    main(sys.argv[1:])
