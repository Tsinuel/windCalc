# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:56:05 2022

@author: Tsinuel Geleta
"""

import numpy as np
import pandas as pd
import os
import warnings
import shapely.geometry as shp

from shapely.ops import voronoi_diagram
from shapely.validation import make_valid


#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================


#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================




def getIntersection(pA, pB, allowMultiPolygon=True):
    
    # pA = shp.Polygon([])
    # pB = shp.Polygon([])
    # overlap = shp.Polygon([])
    
    if not pA.is_valid:
        pA = make_valid(pA)
        if pA.geom_type == "MultiPolygon":
            temp = pA
            pA = None
            for geom in temp.geoms:
                if geom.area/temp.area > 0.001:
                    if pA is None:
                        pA = geom
                    else:
                        pA = pA.union(geom)
        elif pA.geom_type == "GeometryCollection":
            temp = pA
            pA = None
            for geom in temp.geoms:
                if geom.area/temp.area > 0.001:
                    if geom.geom_type == "Polygon":
                        if pA is None:
                            pA = geom
                        else:
                            pA = pA.union(geom)

    if not pB.is_valid:
        pB = make_valid(pB)
        if pB.geom_type == "MultiPolygon":
            temp = pB
            pB = None
            for geom in temp.geoms:
                if geom.area/temp.area > 0.001:
                    if pB is None:
                        pB = geom
                    else:
                        pB = pB.union(geom)
        elif pB.geom_type == "GeometryCollection":
            temp = pB
            pB = None
            for geom in temp.geoms:
                if geom.geom_type == "Polygon":
                    if geom.area/temp.area > 0.001:
                        if pB is None:
                            pB = geom
                        else:
                            pB = pB.union(geom)
    
    overlap = None
    if pA.is_valid and pB.is_valid:
        if pA.intersects(pB):
            overlap = pA.intersection(pB)
            if overlap.geom_type == "Polygon":
                pass
            elif overlap.geom_type == "GeometryCollection":
                temp = overlap
                overlap = None
                for geom in temp.geoms:
                    if geom.geom_type == "Polygon":
                        if overlap is None:
                            overlap = geom
                        else:
                            overlap = overlap.union(geom)
            elif overlap.geom_type == "MultiPolygon":
                if allowMultiPolygon:
                    warnings.warn("MultiPolygon obtained while intersecting. Make sure that the receiving function is capable of treating it.")
                else:
                    overlap = None
            elif overlap.geom_type == "LineString":
                overlap = None
            elif overlap.geom_type == "Point":
                overlap = None
            else:
                raise Exception(f"Unknown geometry type {overlap.geom_type} obtained while intersecting.")
        elif pA.within(pB):
            overlap = pA
        elif pB.within(pA):
            overlap = pB
        else:
            overlap = None
    else:
        print("One or both of the input geometries are invalid to perform intersection.")
        print(f"geom A: {pA}")
        print(f"geom B: {pB}")
        plt.figure()
        x,y = pA.exterior.xy
        plt.plot(x,y,'-b',lw=1.0)
        x,y = pB.exterior.xy
        plt.plot(x,y,'-r',lw=1.0)
        plt.axis('equal')
        overlap = None
    
    return overlap

def intersects(geomA,geomB):
    yesItDoes = geomA.intersects(geomB) or geomA.within(geomB) or geomB.within(geomA)
    return yesItDoes

def sortVoronoi(coords,tribs=None):
    points = shp.MultiPoint(coords)
    if tribs is None:
        tribs = voronoi_diagram(coords)

    idxs = []
    for t,trib in enumerate(tribs.geoms):
        for p,pt in enumerate(points.geoms):
            if pt.within(trib):
                idxs.append(p)
    tribs = shp.MultiPolygon([tribs.geoms[i] for i in idxs])
    return tribs

def trimmedVoronoi(bound,coords):
    inftyTribs = voronoi_diagram(shp.MultiPoint(coords))
    boundPolygon = shp.Polygon(bound)
    tribs = []
    for g in inftyTribs.geoms:
        newRig = getIntersection(boundPolygon,g)
        if newRig is not None:
            tribs.append(newRig)
        else:
            continue
    idx = []
    for ic in range(np.shape(coords)[0]):
        for it,trib in enumerate(tribs):
            if shp.Point(coords[ic,:]).within(trib):
                idx.append(it)
                break
    tribs = shp.MultiPolygon([tribs[i] for i in idx])
    
    return tribs

def meshRegionWithPanels(region,area,minAreaFactor=0.5,debug=False):
    if debug:
        import matplotlib.pyplot as plt
        # print(f"Region: shape {region.shape}, {region}\n")
        # print(f"NominalArea = {area}")

    # generate a uniform grid of panel center points covering the extents of the region
    xMin, xMax = min(region[:,0]), max(region[:,0])
    yMin, yMax = min(region[:,1]), max(region[:,1])
    Dx = xMax-xMin
    Dy = yMax-yMin
    N = max(int(np.round(Dx/np.sqrt(area),0)), 1)
    dx = Dx/N
    M = max(int(np.round(Dy/np.sqrt(area),0)), 1)
    dy = Dy/M

    x = np.linspace(xMin+dx/2,xMax-dx/2,N-1) if N > 2 else (xMax-xMin)/2.0
    y = np.linspace(yMin+dy/2,yMax-dy/2,M-1) if M > 2 else (yMax-yMin)/2.0
    X,Y = np.meshgrid(x,y)
    XY = np.concatenate((np.reshape(X,[-1,1]), np.reshape(Y,[-1,1])),axis=1)

    # generate the panels with the regular points. Take a note of the panels with area less than minAreaFactor*targetPanelArea for merging
    points = shp.MultiPoint(XY)
    regions = voronoi_diagram(points)
    bound = shp.Polygon(region)

    if debug:
        # print(f"xRange = {Dx}, yRange = {Dy}")
        # print(f"N = {N}, M = {M}")
        # plt.figure(figsize=[15,10])
        # plt.plot(region[:,0],region[:,1],'-k',lw=2)
        # plt.plot(X,Y,'.k')
        plt.figure(figsize=[15,10])
        x,y = bound.exterior.xy
        plt.plot(x,y,'-b',lw=3)
        plt.axis('equal')

    panels = []
    goodPts = []
    for g in regions.geoms:
        newRig = getIntersection(bound,g)
        if newRig is not None:
            panels.append(newRig)
        else:
            continue
        # x,y = newRig.exterior.xy
        # plt.plot(x,y,'-r',lw=0.5)
        
        if newRig.area > minAreaFactor*area:
            center = newRig.centroid
            goodPts.append(center.xy)
    goodPts = np.reshape(np.asarray(goodPts,dtype=float),[-1,2])

    # regenerate the panels without those that had areas less than minAreaFactor*targetPanelArea
    points = shp.MultiPoint(goodPts)
    regions = voronoi_diagram(points)

    if debug:
        plt.plot(goodPts[:,0],goodPts[:,1],'.k')

    panels = []
    summedArea = 0
    areas = []
    for g in regions.geoms:
        newRig = getIntersection(bound,g)
        if newRig is not None:
            panels.append(newRig)
        else:
            continue
        
        areas.append(newRig.area)
        x,y = newRig.exterior.xy
        panels.append(newRig)
        # print(f"      xy shape of panel_i: {np.shape(xy)},    total panels shape: {np.shape(panels)}")
        
        if debug:
            summedArea += newRig.area
            plt.plot(x,y,'-r',lw=0.5)

    if len(panels) == 0:
        panels.append(bound)

    if debug:
        # plt.xlim([-0.15,0.12])
        # plt.ylim([-0.03,0.05])
        print(f"Zone area = {bound.area}, summed area = {summedArea}")
        plt.axis('equal')
        plt.show()

    # print(f"Shape: {len(panels)}")
    panels = np.asarray(panels,dtype=object)
    return panels, areas

def calculateTapWeightsPerPanel(panels,tapsAll,tapIdxAll, wghtTolerance=0.0000001):    
    weights = np.array([],dtype=object)
    tapIdxs = np.array([],dtype=object)
    overlaps = ()
    for pnl in panels:
        A = pnl.area
        w = []
        idx = []
        ovlps = []
        for tap,i in zip(tapsAll,tapIdxAll):
            ovlp = getIntersection(pnl,tap)
            if ovlp is not None:
                idx.append(i)
                w.append(ovlp.area/A)
                ovlps.append(ovlp)
            else:
                continue
        if abs(sum(w)-1) > wghtTolerance:
            warnings.warn(f"The sum of area weights {sum(w)} from involved taps does not add up to 1 within the tolerance of 0.1%.")
        weights = np.append(weights, w)
        tapIdxs = np.append(tapIdxs, idx)
        overlaps += (ovlps,)
    
    return weights, tapIdxs, overlaps



#===============================================================================
#================================ CLASSES ======================================
#===============================================================================
