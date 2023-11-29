# Author: Tsinuel Geleta
# Date: November 1, 2023

"""
This module contains functions and classes to handle the calculation of separated shear layer (SSL) in a given planar flow field.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import wind

class curvedCoordSys:
    def __init__(self, origin, s_x, s_y, xRange, yRange):
        self.origin = origin
        self.s_x = s_x
        self.s_y = s_y
        self.xRange = xRange
        self.yRange = yRange
        
    def __calc_n_axis(self):
        # calculate the normal vector
        n = np.cross(self.s_x, self.s_y)
        n = n / np.linalg.norm(n)
        return n
    

class SSL:
    def __init__(self, x, y, U, V, W, 
                 uu, vv, ww, uv, vw, uw,
                 p, P, pp, tke, nut, nutMean, nutVar,
                 origin, normal, xDim, pathSmoothWin=15,
                 ):
        if xDim == 1:
            # transpose all the quantities
            x, y = np.transpose(x), np.transpose(y)
            U, V, W = np.transpose(U), np.transpose(V), np.transpose(W)
            uu, vv, ww = np.transpose(uu), np.transpose(vv), np.transpose(ww)
            uv, vw, uw = np.transpose(uv), np.transpose(vw), np.transpose(uw)
            p, P = np.transpose(p), np.transpose(P)
            pp = np.transpose(pp)
            tke = np.transpose(tke)
            nut = np.transpose(nut)
            nutMean = np.transpose(nutMean)
            nutVar = np.transpose(nutVar)            
            xDim = 0
            
        self.x = x
        self.y = y
        self.U = U
        self.V = V
        self.W = W
        self.uu = uu
        self.vv = vv
        self.ww = ww
        self.uv = uv
        self.vw = vw
        self.uw = uw
        self.p = p
        self.P = P
        self.pp = pp
        self.tke = tke
        self.nut = nut
        self.nutMean = nutMean
        self.nutVar = nutVar
        self.origin = origin
        self.normal = normal
        self.xDim = xDim
        self.pathSmoothWin = pathSmoothWin       
        
        # calculated quantities
        self.shape = np.shape(self.x) if np.shape(self.x) == np.shape(self.y) else None
        self.ssl_x = None
        self.ssl_y = None
        
        self.Refresh()
        
        
    def __calcSSLPath(self):
        # calculate the SSL path
        self.ssl_x = np.zeros(self.shape[0])
        self.ssl_y = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            max_tke_idx = np.argmax(self.tke[i,:])
            self.ssl_x[i] = self.x[i,max_tke_idx]
            self.ssl_y[i] = self.y[i,max_tke_idx]
            
        # smooth the SSL path using a moving average
        self.ssl_y = pd.Series(self.ssl_y).rolling(self.pathSmoothWin, min_periods=1, center=True).mean()
        
        # remove the part left of the origin
        origin_idx = np.argmin(np.abs(self.ssl_x - self.origin[0]))
        self.ssl_x = self.ssl_x[origin_idx:]
        self.ssl_y = self.ssl_y[origin_idx:]
        
    def Refresh(self):
        self.__calcSSLPath()