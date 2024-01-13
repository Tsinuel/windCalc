# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 01:40:48 2022

@author: Tsinu
"""

import sys
import spatialAvg_FFS as ffs

def main(args):
    print(args)
    dir = args[0]
    fileStartsWith = args[1]
    outFile = args[2]
    
    ffs.average_data(dir, fileStartsWith, outFile)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
