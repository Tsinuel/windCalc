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
    probeName = args[1]
    targetProfile = args[2]
    
    
    windOF.processVelProfile(caseDir, probeName, targetProfile,
                          normalize=True,
                          writeToDataFile=True,
                          showPlots=False,
                          exportPlots=True,
                          )
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
