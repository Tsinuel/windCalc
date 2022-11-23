# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 01:40:48 2022

@author: Tsinu
"""

import sys
# import numpy as np
import foam

def main(args):
    caseDir = args[0]
    tMin = float(args[1])
    tMax = float(args[2])
    H = float(args[3])
    foam.scaleInflowData(caseDir, tMin, tMax, H, writeInflow=False)
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])