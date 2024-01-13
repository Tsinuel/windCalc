# -*- coding: utf-8 -*-
"""
Created on Jan 7, 2024 11:00 PM

@author: Tsinuel Geleta

"""

import numpy as np
import pandas as pd
import os

def average_data(dir, fileStartsWith, outFile, axis='y', outCoordVal=0.0, convertNaNtoZero=False,
                 fields_dont_avg = ['Points:0', 'Points:1', 'Points:2'],
                 ):
    # list all files in the directory that start with fileStartsWith
    files = [f for f in os.listdir(dir) if f.startswith(fileStartsWith)]
    
    df_avg = pd.read_csv(dir + files[0])
    print("Reading file " + files[0])
    if axis == 'x':
        df_avg['Points:0'] = outCoordVal
    elif axis == 'y':
        df_avg['Points:1'] = outCoordVal
    elif axis == 'z':
        df_avg['Points:2'] = outCoordVal
    else:
        raise ValueError("axis must be one of ['x','y','z']")
    
    # replace nan with 0.0 and warn the user how many nans were replaced
    if convertNaNtoZero and df_avg.isnull().values.any():
        percentNAN = df_avg.isnull().sum().sum() / df_avg.size
        print("WARNING: " + str(percentNAN*100) + "% of the data is in file " + files[0] + " is nan. Replacing with 0.0")
        df_avg = df_avg.fillna(0.0)    
    
    N = 1.0
    # loop over the rest of the files
    for i in range(1,len(files)):
        print("Reading file " + files[i])
        df = pd.read_csv(dir + files[i])
        
        if convertNaNtoZero and df.isnull().values.any():
            percentNAN = df.isnull().sum().sum() / df.size
            print("WARNING: " + str(percentNAN*100) + "% of the data in file " + files[i] + " is nan. Replacing with 0.0")
            df = df.fillna(0.0)
            
        # loop over the fields in the file
        for field in df.columns:
            if field in fields_dont_avg:
                continue
            else:
                df_avg[field] = (N*df_avg[field] + df[field]) / (N+1)
            
        N += 1.0
    
    # use scientific notation when saving the file
    df_avg.to_csv(outFile, index=False, float_format='%.6E')
    print("Saved file " + outFile)
    