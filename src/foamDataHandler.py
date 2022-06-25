# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:49:36 2022

@author: Tsinu
"""
import numpy as np

def of_readProbe(fileName, trimOverlap = True):
    probes = []
    p = []
    time  = []

    with open(fileName, "r") as f:
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(','')
                    line = line.replace(')','')
                    line = line.split()
                    probes.append([float(line[3]),float(line[4]),float(line[5])])
                else:
                    continue
            else:
                line = line.split()
                time.append(float(line[0]))
                p_probe_i = np.zeros([len(probes)])
                for i in  range(len(probes)):
                    p_probe_i[i] = float(line[i + 1])
                p.append(p_probe_i)
   
    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
   
    return probes, time, p

