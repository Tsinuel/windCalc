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
import json
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from numpy import linalg
from shapely.ops import voronoi_diagram
from shapely.validation import make_valid
from typing import List, Tuple, Dict, Literal
from scipy.interpolate import griddata
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================

def_cols = list(mcolors.TABLEAU_COLORS.values())


#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

#--------------------- TAPS, PANELS, & AREA AVERAGING --------------------------
def getIntersection(pA, pB, allowMultiPolygon=True, showLog=False):
    """
    Get the intersection of two polygons. The function checks if the polygons are valid and if they intersect. If they intersect, the function returns the intersection. If the intersection is a MultiPolygon, the function returns None if allowMultiPolygon is False. If allowMultiPolygon is True, the function returns the MultiPolygon.
    
    Parameters
    ----------
    pA : shapely.geometry.Polygon
        The first polygon.
    pB : shapely.geometry.Polygon
        The second polygon.
    allowMultiPolygon : bool, optional
        If True, the function returns a MultiPolygon if the intersection is a MultiPolygon. If False, the function returns None if the intersection is a MultiPolygon. The default is True.
    showLog : bool, optional
        If True, the function prints the details of the polygons and the intersection. The default is False.
        
    Returns
    -------
    overlap : shapely.geometry.Polygon or shapely.geometry.MultiPolygon or None
        The intersection of the two polygons. If the intersection is a MultiPolygon and allowMultiPolygon is True, the function returns the MultiPolygon. If the intersection is a MultiPolygon and allowMultiPolygon is False, the function returns None. If the polygons do not intersect, the function returns None.
        
    """
    
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
            elif overlap.geom_type == "LineString" or overlap.geom_type == "Point" or overlap.geom_type == "MultiLineString":
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

def trimmedVoronoi(bound,query_points,query_points_can_be_on_boundary=False,showLog=False,showPlot=False):
    if showLog:
        print(f"Generating tributaries ...")
        print(f"Shape of bound: {np.shape(bound)}")
        print(f"Shape of coords: {np.shape(query_points)}")
    if showPlot:
        ax = plt.figure(figsize=[10,8]).add_subplot(111)
    
    inftyTribs = voronoi_diagram(shp.MultiPoint(query_points))
    if showPlot:
        for i,g in enumerate(inftyTribs.geoms):
            x,y = g.exterior.xy
            if i == 0:
                ax.plot(x,y,'-g',lw=1,label='Tributaries (infinite)')
            else:
                ax.plot(x,y,'-g',lw=1)
    if showLog:
        print(f"Shape of inftyTribs: {np.shape(inftyTribs.geoms)}")
    
    boundPolygon = shp.Polygon(bound)
    tribs = []
    for i, g in enumerate(inftyTribs.geoms):
        newRig = getIntersection(boundPolygon,g)
        if showLog:
            print(f"Type of newRig: {type(newRig)}")
            print(f"Area of newRig: {newRig.area}")
        if showPlot:
            x,y = g.exterior.xy
            ax.plot(x,y,'-g',lw=1)
            x,y = newRig.exterior.xy
            if i == 0:
                ax.plot(x,y,'--m',lw=1,label='Tributaries (bounded intermediate)')
            else:
                ax.plot(x,y,'--m',lw=1)
            
            
        if newRig is not None:
            tribs.append(newRig)
        else:
            continue
    idx = []
    areas = []
    for ic in range(np.shape(query_points)[0]):
        for it,trib in enumerate(tribs):
            isToBeIncluded = False
            if query_points_can_be_on_boundary:
                if shp.Point(query_points[ic,:]).touches(trib):
                    isToBeIncluded = True
            else:
                if shp.Point(query_points[ic,:]).within(trib):
                    isToBeIncluded = True
            if isToBeIncluded:
                idx.append(it)
                areas.append(trib.area)
                break
    idx = np.asarray(idx,dtype=int)
    areas = np.asarray(areas,dtype=float)
    totalArea = np.sum(areas)
    boundArea = boundPolygon.area
    if abs(totalArea - boundArea) > 0.00001*boundArea:
        msg = f"The sum of tributary areas {totalArea} is not equal to the area of the bound {boundArea}."
        warnings.warn(msg)

    temp = []
    for i in idx:
        temp.append(tribs[i])
        if not type(tribs[i]) == shp.Polygon:
            warnings.warn(f"Type of tributary {i} is {type(tribs[i])}. There is a risk of voronoi not completely tiling the bound.")
    tribs = shp.GeometryCollection(temp)

    # tribs = shp.MultiPolygon([tribs[i] for i in idx])

    if showLog:
        print(f"Shape of tribs: {np.shape(tribs.geoms)}")
    if showPlot:
        x,y = boundPolygon.exterior.xy
        ax.plot(x,y,'-b',lw=3,label='Bound')
        ax.plot(query_points[:,0],query_points[:,1],'.r',label='Query points')
        for i,g in enumerate(tribs.geoms):
            x,y = g.exterior.xy
            if i == 0:
                ax.plot(x,y,'-r',lw=1,label='Tributaries (bounded)')
            else:
                ax.plot(x,y,'-r',lw=1)
        ax.axis('equal')
        ax.legend()
        plt.show()
    
    return tribs

def meshRegionWithPanels(region,area,minAreaFactor=0.5,debug=False) -> Tuple[shp.MultiPolygon, List[float]]:
    if debug:
        print(f"Region: shape {region.shape}, {region}\n")
        print(f"NominalArea = {area}")

    # generate a uniform grid of panel center points covering the extents of the region
    xMin, xMax = min(region[:,0]), max(region[:,0])
    yMin, yMax = min(region[:,1]), max(region[:,1])
    Dx = xMax-xMin
    Dy = yMax-yMin
    d = np.sqrt(area)
    N = int(np.ceil(Dx/d))
    dx = Dx/N
    M = int(np.ceil(Dy/d))
    dy = Dy/M

    x = np.linspace(xMin+dx/2,xMax-dx/2,N)
    y = np.linspace(yMin+dy/2,yMax-dy/2,M)
    X,Y = np.meshgrid(x,y)
    XY = np.concatenate((np.reshape(X,[-1,1]), np.reshape(Y,[-1,1])),axis=1)
    isSinglePoint = len(X) == 1 and len(Y) == 1
    if debug:
        print(f"x Range: {xMin} to {xMax} \t\t y Range: {yMin} to {yMax}")
        print(f"dx = {dx}, dy = {dy}")
        print(f"Point grid: {N} X {M}")
        print(f"Grid \t\tx: {x}\n\t\ty: {y} ")
        print(f"Grid mesh shape: {np.shape(X)}")

    # generate the panels with the regular points. Take a note of the panels with area less than minAreaFactor*targetPanelArea for merging
    points = shp.MultiPoint(XY)
    regions = voronoi_diagram(points)
    bound = shp.Polygon(region)

    if debug:
        plt.figure(figsize=[15,10])
        x,y = bound.exterior.xy
        plt.plot(x,y,'-b',lw=3)
        plt.axis('equal')

    panels1 = []
    goodPts = []
    for g in regions.geoms:
        newRig = getIntersection(bound,g)
        if newRig is not None:
            panels1.append(newRig)
        else:
            continue
        if newRig.area > minAreaFactor*area:
            center = newRig.centroid
            goodPts.append(center.xy)
        if debug:
            x,y = newRig.exterior.xy
            # plt.fill(x,y,'b',alpha=0.3)
            # plt.plot(x,y,'-b',alpha=1,lw=0.5)
    goodPts = np.reshape(np.asarray(goodPts,dtype=float),[-1,2])

    # regenerate the panels without those that had areas less than minAreaFactor*targetPanelArea
    points = shp.MultiPoint(goodPts)
    regions = voronoi_diagram(points)

    if debug:
        plt.plot(goodPts[:,0],goodPts[:,1],'.k')

    panels = []
    areas = []
    for i,g in enumerate(regions.geoms):
        newRig = getIntersection(bound,g)
        if newRig is not None:
            panels.append(newRig)
        else:
            continue
        areas.append(newRig.area)
        x,y = newRig.exterior.xy
        if debug:
            plt.fill(x,y,'r',alpha=0.5)
            plt.plot(x,y,'--k',alpha=1,lw=1)
            plt.text(np.mean(x),np.mean(y),str(i))

    if len(panels) == 0 or isSinglePoint:
        panels.append(bound)
        areas = [bound.area,]
        if debug:
            x,y = bound.exterior.xy
            plt.fill(x,y,'r',alpha=0.5)
            plt.plot(x,y,'--k',alpha=1,lw=1)

    if debug:
        print(f"Zone area = {bound.area}, summed area = {np.sum(areas)}")
        plt.axis('equal')
        plt.show()

    panels = shp.MultiPolygon(panels)
    return panels, areas

def meshRegionWithWiggle(region, area_orig, minAreaFactor=0.5, debug=False, 
                        tol_factorFromRegionArea=0.001, wiggle_h=1.001, wiggle_r=1.001, wiggleFactorLimit=1.5, maxNumTrial=1000) -> Tuple[shp.MultiPolygon, List[float]]:
    '''
    This function is similar to meshRegionWithPanels() but it allows for a wiggle factor to be applied to the target area.
    The wiggle factor is applied in a loop until the area of the panels is within the tolerance of the target area.

    Parameters
    ----------
    region : np.ndarray
        A 2D array of the vertices of the region to be meshed.
    area_orig : float
        The target area of the panels.
    minAreaFactor : float, optional (default=0.5)
        The minimum area factor to be used in the first iteration of the meshing.
    debug : bool, optional (default=False)
        If True, the meshing process will be plotted.
    tol_factorFromRegionArea : float, optional (default=0.001) 
        The tolerance of the difference between the area of the panels and the area of the region.
    wiggle_h : float, optional (default=1.001) 
        The wiggle factor to be applied to the target area.
    wiggle_r : float, optional (default=1.001)
        The rate of increase of the wiggle factor.
    wiggleFactorLimit : float, optional (default=1.2)
        The maximum wiggle factor to be applied.

    Returns
    -------
    panels : shp.MultiPolygon
        A shapely MultiPolygon object of the panels.
    areas : List[float]
        A list of the areas of the panels.
    '''
    if debug:
        print(f"Region: shape {region.shape}, {region}\n")
        print(f"NominalArea = {area_orig}")
        print("n \tfactor \t\tArea")
    area = area_orig
    panels, areas = meshRegionWithPanels(region, area, minAreaFactor=minAreaFactor, debug=False)
    zoneArea = shp.Polygon(region).area
    tiledArea = panels.area

    all_areas = [area,]
    sign = 1
    count = 0
    while np.abs(zoneArea - tiledArea) > tol_factorFromRegionArea*zoneArea:
        area = area_orig
        sign *= -1
        factor = np.power(wiggle_h, sign*wiggle_r*count)
        if factor > wiggleFactorLimit:
            print(f"Reached the limit of wiggle factor of {wiggleFactorLimit}. Breaking the loop.")
            break
        if count > maxNumTrial:
            print(f"Reached the maximum number of trials of {count}. Breaking the loop.")
            break
        area *= factor
        all_areas.append(area)
        count += 1

        try:
            panels, areas = meshRegionWithPanels(region, area, minAreaFactor=minAreaFactor, debug=False)
        except:
            if debug:
                print(f"An error occured while meshing the region with area = {area}. Skipping it to the next area.")
            continue
        zoneArea = shp.Polygon(region).area
        tiledArea = panels.area
        if debug:
            print(f"{count} \t{factor:.6f} \t{area:.6f}")
    if debug:
        plt.figure()
        x,y = shp.Polygon(region).exterior.xy
        plt.plot(x,y,'-b',lw=3)
        for p in panels.geoms:
            x,y = p.exterior.xy
            plt.fill(x,y,'r',alpha=0.5)
            plt.plot(x,y,'-k',alpha=1,lw=0.5)
        plt.axis('equal')
        plt.show()

        plt.figure()
        plt.plot(all_areas,'-o')
        plt.xlabel('Iteration')
        plt.ylabel('Tested areas')
        plt.show()
        print(f"Finito! Final nominal area: {area}")

    return panels, areas

def calculateTapWeightsPerPanel(panels:shp.MultiPolygon,tapsAll,tapIdxAll, wghtTolerance=0.0000001, showLog=True):    
    weights = ()
    tapIdxs = ()
    overlaps = ()
    errIdxs = []
    for p,pnl in enumerate(panels.geoms):
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
            errIdxs.append(p)
            warnings.warn(f"The sum of area weights {sum(w)} from involved taps does not add up to 1 within the tolerance of 0.1%.")
        weights += (w,)
        tapIdxs += (idx,)
        overlaps += (ovlps,)
        if showLog:
            print(f"\t\t\t\tNo. of taps in panel: {len(w)}, \tError in wights: {1-sum(w)}")
    
    return weights, tapIdxs, overlaps, errIdxs

#--------------------------- COORDINATE SYSTEMS --------------------------------
def transform(geomIn:np.ndarray,orig:np.ndarray,T:np.ndarray,inverseTransform:bool=False,debug:bool=False):
    '''
    Transform the geometry from the local coordinate system to the global coordinate 
    system or vice versa.
    
    Parameters
    ----------
    geomIn : np.ndarray
        The geometry in the local coordinate system. The shape of the array should be 
        [N,2] or [N,3].
    orig : np.ndarray
        The origin of the local coordinate system. The shape of the array should be [2,] 
        or [3,].
    T : np.ndarray
        The transformation matrix. The shape of the array should be [2,2] or [3,3].
    inverseTransform : bool, optional
        If True, the transformation is from the global coordinate system to the local 
        coordinate system. The default is False.
    debug : bool, optional
        If True, the function prints the details of the transformation. The default is 
        False.
        
    Returns
    -------
    geomOut : np.ndarray
        The geometry in the global coordinate system. The shape of the array is [N,2] or 
        [N,3].
    '''
    # geomIn = np.asarray(geomIn,dtype=float)
    # orig = np.asarray(orig,dtype=float)
    # T = np.asarray(T,dtype=float)
    if debug:
        print("Raw input:")
        print(f"geomIn: {geomIn}")
        print(f"orig: {orig}")
        print(f"T: {T}")
    
    N,M = np.shape(geomIn)
    if len(orig) == 3 and M == 2:
        geomIn = np.append(geomIn,np.zeros((N,1)),axis=1)
        if debug:
            print(f"geomIn after readjusting and adding zeros: {geomIn}")
    if inverseTransform:
        # add the origin to the geometry dotted with the inverse of the transformation matrix
        geomOut = np.transpose(np.dot(np.linalg.inv(T),np.transpose(geomIn)))
        if debug:
            print(f"Inverse of T: {np.linalg.inv(T)}")
            print(f"shape of geomOut: {np.shape(geomOut)}")
            print(f"geomOut after rotation by the inverse of T: {geomOut}")
        geomOut = np.add(geomOut, orig)
        if debug:
            print(f"shape of geomOut: {np.shape(geomOut)}")
            print(f"geomOut after translation to the origin: {geomOut}")        
    else:
        geomOut = np.add(geomIn,orig)
        if debug:
            print(f"geomOut after translation to the origin: {geomOut}")
        geomOut = np.transpose(np.dot(T,np.transpose(geomOut)))
        if debug:
            print(f"shape of geomOut: {np.shape(geomOut)}")
            print(f"geomOut after rotation by T: {geomOut}")
        # geomOut =np.transpose(np.dot(T,np.transpose(np.add(geomIn,orig))))
        
    return geomOut

def tranform2D(x, y, O, e1, e2, translateFirst=True,):
    if translateFirst:
        # ans = np.array([[np.sum(xyi*basisVectors[0]), np.sum(xyi*basisVectors[1])] for xyi in xy-origin], dtype=float)
        ans_x = (x-O[0])*e1[0] + (y-O[1])*e1[1]
        ans_y = (x-O[0])*e2[0] + (y-O[1])*e2[1]
    else:
        ans_x = (x*e1[0] + y*e1[1]) - O[0]
        ans_y = (x*e2[0] + y*e2[1]) - O[1]
    return np.array(ans_x, dtype=float), np.array(ans_y, dtype=float)

def translate(x, y, t, z=None):
    x, y = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
    t = np.array(t,dtype=float)
    if z is None:
        return x+t[0], y+t[1]
    else:
        z = np.asarray(z,dtype=float)
        return x+t[0], y+t[1], z+t[2]
    
    

#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

#-------------------------- BUILDING ENVELOPE (WIND) ---------------------------
class face:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                ID=None,
                name="Unnamed face",
                faceType:Literal['roof','wall','other']=None,
                note=None,
                origin=None,
                basisVectors=None,
                origin_plt=None,
                basisVectors_plt=None,
                vertices=None,
                tapNo: List[int]=None,
                tapIdx: List[int]=None,
                tapName: List[str]=None,
                # badTaps=None,
                # allBldgTaps=None,
                tapCoord=None,
                zoneDict=None,
                nominalPanelAreas=None,
                numOfNominalPanelAreas=5,
                file_basic=None,
                file_derived=None,
                showDetailedLog=False,
                ):
        """
        Represents a 2D planar face of a building. The face is defined by its vertices in the face-local coordinate system. The face-local coordinate system is defined by its origin and basis vectors in the main 3D coordinate system. The face-local coordinate system is also defined by its origin and basis vectors in the 2D coordinate system for plotting. The face is also defined by its taps. The taps are defined by their numbers, names, and coordinates in the face-local coordinate system. Component and cladding load zoning can be defined for the face according to a pre-defined dictionary that specifies the design code zonings.

        Parameters
        ----------
        ID : int, optional
            The ID of the face. The default is None.
        name : str, optional
            The name of the face. The default is None.
        note : str, optional
            A note about the face. The default is None.
        origin : np.ndarray, optional
            The origin of the face-local coordinate system. The default is None.
        basisVectors : np.ndarray, optional
            The basis vectors of the face-local coordinate system. The default is None.
        origin_plt : np.ndarray, optional
            The origin of the face-local coordinate system for plotting. The default is None.
        basisVectors_plt : np.ndarray, optional
            The basis vectors of the face-local coordinate system for plotting. The default is None.
        vertices : np.ndarray, optional
            [Nv, 2]
            The vertices of the face in face-local coordinate system. The default is None.
        tapNo : List[int], optional
            The tap numbers of the face. The default is None.
        tapIdx : List[int], optional
            The tap indices in the main building-level list of taps. The default is None.
        tapName : List[str], optional
            The tap names of the face. The default is None.
        badTaps : List[int], optional
            The tap numbers of the bad taps in the face. The default is None.
        allBldgTaps : List[int], optional
            The tap numbers of all the taps of the building. The default is None.
        tapCoord : np.ndarray, optional
            The coordinates of the taps of the face in the face-local coordinate system. The default is None.
        zoneDict : dict, optional
            The zoning dictionary of the face. It is defined as a dictionary with the following structure:
                zoneDict = {
                                0:['Design Code','zone name', np.array(<vertices of the zone>)],
                        }
            The 'Design code' is the building code, such as NBCC, ASCE, etc. The 'zone name' is the name of the zone, such as 'z1', 'z2', etc. The vertices define the zone boundary. The default is None.
        nominalPanelAreas : np.ndarray, optional
            List of the nominal panel areas to generate panels of the face. The default is None.
        numOfNominalPanelAreas : int, optional
            The number of nominal panel areas to generate panels of the face. The default is 5.
        file_basic : str, optional
            The path to the file that contains the basic information of the face. The default is None.
        file_derived : str, optional
            The path to the file that contains the derived information of the face. The default is None.

        Returns
        -------
        None.
        """
        # basics
        self.ID = ID
        self.name = name
        self.faceType:Literal['roof','wall','other'] = faceType
        self.note = note
        self.origin = origin                # origin of the face-local coordinate system
        self.basisVectors = basisVectors    # [[3,], [3,], [3,]] basis vectors of the local coord sys in the main 3D coord sys.
        self.origin_plt = origin_plt                # 2D origin of the face-local coordinate system for plotting
        self.basisVectors_plt = basisVectors_plt    # [[2,], [2,]] basis vectors of the local coord sys in 2D for plotting.
        self.vertices = np.array(vertices,dtype=float)            # [Nv, 2] face corners that form a non-self-intersecting polygon. No need to close it with the last edge.
        self.zoneDict = zoneDict            # dict of dicts: zone names per each zoning. e.g., [['NBCC-z1', 'NBCC-z2', ... 'NBCC-zM'], ['ASCE-a1', 'ASCE-a2', ... 'ASCE-aQ']]
        self.panelAreas_nominal = nominalPanelAreas  # a vector of nominal areas to generate panels. The panels areas will be close but not necessarily exactly equal to these.
        self.numOfNominalPanelAreas = numOfNominalPanelAreas

        # Tap things
        self.tapNo: List[int] = tapNo                  # [Ntaps,]
        # self.tapNo_all = self.tapNo                    # includes the bad taps
        self.tapIdx: List[int] = tapIdx                # [Ntaps,] tap indices in the main matrix of the entire building
        self.tapName: List[str] = tapName              # [Ntaps,]
        self.badTaps: List[int] = None
        self.RemovedBadTaps: dict = None
        self.tapCoord = tapCoord            # [Ntaps,2]   ... from the local face origin
        
        # derived
        self.tapTribs: shp.MultiPolygon = None
        self.panels : shp.MultiPolygon = None
        self.panelAreas = None
        self.tapWghtPerPanel = None     # [N_zones][N_area][N_panels][N_taps]
        self.tapIdxPerPanel = None
        self.error_in_panels = None
        self.error_in_zones = None

        # fill derived
        # self.__handleBadTaps(allBldgTaps)
        if file_basic is None:
            self.Refresh(showDetailedLog)
        elif file_derived is None:
            self.readFromFile(file_basic=file_basic)
        else:
            self.readFromFile(file_basic=file_basic,file_derived=file_derived)
            
    def __str__(self):
        return self.name

    def __handleBadTaps(self,allBldgTaps):
        if self.badTaps is None:
            return
        if self.tapNo is None:
            warnings.warn("List of bad taps is provided for a face without defined tap numbers. Define the taps or the bad taps will be ignored.")
            return
        if allBldgTaps is None:
            warnings.warn("If bad taps are provided, the full building tap list has to be provided as well so the updated indices can be determined.")
            return
        
        badIdx = np.where(np.in1d(self.tapNo, self.badTaps))[0]
        self.tapNo = np.delete(self.tapNo, badIdx)
        if self.tapIdx is not None:
            allBadIdx = np.where(np.in1d(self.tapNo, self.badTaps))[0]
            bldgTaps = np.delete(allBldgTaps, allBadIdx)
            self.tapIdx = np.where(np.isin(bldgTaps, self.tapNo))[0]
        if self.tapName is not None:
            self.tapName = np.delete(self.tapName, badIdx)
        if self.tapCoord is not None:
            self.tapCoord = np.delete(self.tapCoord, badIdx, axis=0)

    def __generateTributaries(self, showDetailedLog=False):
        if self.vertices is None or self.tapCoord is None:
            return

        self.tapTribs = trimmedVoronoi(self.vertices, self.tapCoord, showLog=showDetailedLog)

    def __defaultZoneAndPanelConfig(self):
        if self.zoneDict is None and self.panelAreas_nominal is None:
            return
        if self.zoneDict == {}:
            self.zoneDict = {
                                0:['Default','Default', np.array(self.vertices)],
                        }
        if self.panelAreas_nominal is None and self.zoneDict is not None:
            bound = shp.Polygon(self.vertices)
            maxArea = bound.area
            minArea = maxArea
            for trib in self.tapTribs.geoms:
                minArea = min((minArea, trib.area))

            self.panelAreas_nominal = np.logspace(np.log10(minArea), np.log10(maxArea), self.numOfNominalPanelAreas)

    def __generatePanels(self, showDetailedLog=False):
        self.__defaultZoneAndPanelConfig()
        print(f"Generating panels ...")
        # print(f"Shape of zones: {np.shape(self.zones)}")
        if self.tapTribs is None or self.zoneDict is None:
            return
        if showDetailedLog:
            print(f"Shape of tapTribs: {np.shape(self.tapTribs)}")
            print(f"Shape of nominalPanelAreas: {np.shape(self.panelAreas_nominal)}")

        self.panels : shp.MultiPolygon = ()    # [nZones][nAreas][nPanels]
        # self.panelAreas = ()    # [nZones][nAreas][nPanels]
        self.tapWghtPerPanel = ()    # [nZones][nAreas][nPanels][nTapsPerPanel]
        self.tapIdxPerPanel = ()   # [nZones][nAreas][nPanels][nTapsPerPanel]
        self.error_in_panels = ()
        self.error_in_zones = ()
        for z,zn in enumerate(self.zoneDict):
            # print(f"\tWorking on: {self.zoneDict[zn][0]}-{self.zoneDict[zn][1]}")
            zoneBoundary = self.zoneDict[zn][2]
            panels_z = ()
            panelA_z = ()
            pnlWeights_z = ()
            tapIdxByPnl_z = ()
            err_pnl_z = ()
            err_zn_z = ()
            for a,area in enumerate(self.panelAreas_nominal):
                # print(f"\t\tWorking on nominal area: {area}")
                # print(f"zoneBoundary: {zoneBoundary}, area: {area}")
                # pnls,areas = meshRegionWithPanels(zoneBoundary,area,debug=False)
                pnls,areas = meshRegionWithWiggle(zoneBoundary,area,debug=False) 

                tapsInSubzone = []
                idxInSubzone = []
                zonePlygn = shp.Polygon(zoneBoundary)
                for i,tap in zip(self.tapIdx,self.tapTribs.geoms):
                    if intersects(tap,zonePlygn):
                        tapsInSubzone.append(tap)
                        idxInSubzone.append(i)
                wght, Idx, overlaps, errIdxs = calculateTapWeightsPerPanel(pnls,tapsInSubzone,idxInSubzone,showLog=False)
                
                panels_z += (pnls,)
                panelA_z += (areas,)
                pnlWeights_z += (wght,)
                tapIdxByPnl_z += (Idx,)
                err_pnl_z += (errIdxs,)

                errorInArea = 100*(zonePlygn.area - sum(areas))/zonePlygn.area
                # print(f"\t\t\tArea check: \t Zone area = {zonePlygn.area}, \t sum of all panel areas = {sum(areas)}, \t Error = {errorInArea}%")
                # print(f"\t\tGenerated number of panels: {len(pnls.geoms)}")
                
                if abs(errorInArea) > 1.0:
                    err_zn_z += (a,)
                    warnings.warn(f"The difference between Zone area and the sum of its panel areas exceeds the tolerance level.")
                
            self.panels += (panels_z,)
            # self.panelAreas += (panelA_z,)
            self.tapWghtPerPanel += (pnlWeights_z,)
            self.tapIdxPerPanel += (tapIdxByPnl_z,)
            self.error_in_panels += (err_pnl_z,)
            self.error_in_zones += (err_zn_z,)

        self.panelAreas = []
        for _, panels_z in enumerate(self.panels):
            pnlArea_z = []
            for _, panels_a in enumerate(panels_z):
                pnlArea_a = []
                for _, pnl in enumerate(panels_a.geoms):
                    pnlArea_a.append(pnl.area)
                pnlArea_z.append(pnlArea_a)
            self.panelAreas.append(pnlArea_z)
        
        # print(f"Shape of 'panels': {np.shape(np.array(self.panels,dtype=object))}")
        # print(f"Shape of 'panelAreas': {np.shape(np.array(self.panelAreas,dtype=object))}")
        # print(f"Shape of 'pnlWeights': {np.shape(np.array(self.tapWghtPerPanel,dtype=object))}")
        # print(f"Shape of 'tapIdxByPnl': {np.shape(np.array(self.tapIdxPerPanel,dtype=object))}")
        print(f"Done generating panels ...")
        print(f"Error summary in paneling:")
        print(json.dumps(self.panelingErrors, indent=4))
        # print(f"Indecies of nominalPanelArea per each zone with error of unequal area between total sum of panel areas vs. zone: \n\t\t\t{self.error_in_zones}")
        # print(f"Indecies of panels within nominalPanelArea within zone with tap weights that do not sum to 1 : \n\t\t\t{self.error_in_panels}")

    """--------------------------------- Properties -----------------------------------"""
    @property
    def tapCoord3D(self):
        return transform(self.tapCoord, self.origin, self.basisVectors)

    @property
    def tapCoordPlt(self):
        return transform(self.tapCoord, self.origin_plt, self.basisVectors_plt)

    @property
    def orientation(self):
        if self.basisVectors is None:
            return None
        return np.arctan2(self.basisVectors[1][1], self.basisVectors[1][0])
    
    @property
    def faceNormal(self):
        if self.basisVectors is None:
            return None
        return np.cross(self.basisVectors[0],self.basisVectors[1])
    
    @property
    def orientation_plt(self):
        if self.basisVectors_plt is None:
            return None
        return np.arctan2(self.basisVectors_plt[1][1], self.basisVectors_plt[1][0])

    @property
    def vertices3D(self):
        # return transform(self.vertices, self.origin, self.basisVectors, inverse=True, debug=False)
        return self.toGlobalCoord(self.vertices)

    @property
    def vertices3D_inv____redacted(self):
        # 3D vertices in the face-local coordinate system
        return transform(self.vertices, self.origin, np.linalg.inv(self.basisVectors))

    @property
    def verticesPlt(self):
        return transform(self.vertices, self.origin_plt, self.basisVectors_plt)

    @property
    def boundingBox(self):
        if self.vertices is None:
            return None
        return np.array([min(self.vertices[:,0]), max(self.vertices[:,0]), 
                         min(self.vertices[:,1]), max(self.vertices[:,1])])
        
    @property
    def aspectRatio(self):
        if self.vertices is None:
            return None
        xtnts = self.boundingBox
        return (xtnts[1]-xtnts[0])/(xtnts[3]-xtnts[2])

    @property
    def NumTaps(self):
        num = 0 if self.tapNo is None else len(self.tapNo)
        return num

    @property
    def NumPanels(self):
        if self.panelAreas_nominal is None:
            return 0
        num = np.int32(np.multiply(self.panelAreas_nominal,0))
        if self.panels is None:
            return num
        for a,_ in enumerate(self.panelAreas_nominal):
            for z,_ in enumerate(self.panels):
                num[a] += len(self.panels[z][a].geoms)
        return np.sum(num)

    @property
    def NumPanelsPerArea(self):
        if self.panelAreas_nominal is None:
            return 0
        num = np.int32(np.multiply(self.panelAreas_nominal,0))
        if self.panels is None:
            return num
        for a,_ in enumerate(self.panelAreas_nominal):
            for z,_ in enumerate(self.panels):
                num[a] += len(self.panels[z][a].geoms)
        return num

    @property
    def panelAreas__redacted(self):
        return None
        if self.panelAreas_nominal is None or self.panels is None:
            return None
        
        panelAreas = []
        for _, panels_z in enumerate(self.panels):
            pnlArea_z = []
            for _, panels_a in enumerate(panels_z):
                pnlArea_a = []
                for _, pnl in enumerate(panels_a.geoms):
                    pnlArea_a.append(pnl.area)
                pnlArea_z.append(pnlArea_a)
            panelAreas.append(pnlArea_z)
        return panelAreas

    @property
    def zoneDictUniqe(self):
        allZones = {}
        i = 0
        for val in self.zoneDict.values():
            isNewVal = True
            for acceptedZone in allZones.values():
                if val[0] == acceptedZone[0] and val[1] == acceptedZone[1]:
                    isNewVal = False
                    break
            if isNewVal:
                x = list(val)[0:2]
                x.append([])
                allZones[i] = x
                i += 1
        return allZones

    @property
    def zoneDictKeys(self):
        zoneDict = self.zoneDictUniqe
        return [val[0]+', '+val[1] for val in zoneDict.values()]

    @property
    def panelingErrors(self):
        if self.error_in_zones is None or self.error_in_panels is None:
            return None
        errDict = {}
        for z,zn in enumerate(self.zoneDict):
            zn_ID = self.zoneDict[zn][0] + ' -- ' + self.zoneDict[zn][1]
            errDict[zn_ID] = {f"nom. areas idxs with tiling errors {self.panelAreas_nominal}": self.error_in_zones[z]}
            errDict[zn_ID]['tap idxs with weight errors'] = {}
            for a,area in enumerate(self.panelAreas_nominal):
                errDict[zn_ID]['tap idxs with weight errors'][f"A={area}"] = self.error_in_panels[z][a]
        return errDict

    """-------------------------------- Data handlers ---------------------------------"""
    def RemoveBadTaps(self, badTaps, idxInData):
        self.RemovedBadTaps = {} if self.RemovedBadTaps is None else self.RemovedBadTaps
        
        thisFaceHasChanged = False
        tapIdxList = list(self.tapIdx)
        for bt,idxData in zip(badTaps, idxInData):
            print(f"Face {self.name}: Removing bad tap {bt} with index in data {idxData} ...")
            for i in range(len(tapIdxList)):
                if tapIdxList[i] > idxData:
                    # print(f"  ... data Index is higher than {self.tapIdx[i]} in face {self.name}")
                    tapIdxList[i] -= 1
            if bt in self.tapNo:
                print(f"  ... found it in face {self.name}")
                thisFaceHasChanged = True
                idx = np.where(self.tapNo == bt)[0]
                self.tapNo = np.delete(self.tapNo, idx)
                tapIdxList = np.delete(tapIdxList, idx)
                if self.tapName is not None:
                    self.tapName = np.delete(self.tapName, idx)
                self.tapCoord = np.delete(self.tapCoord, idx, axis=0)
                self.RemovedBadTaps[bt] = {
                                            'IndexInData': idxData,
                                            'Face': self.name, 
                                            'tapName': self.tapName[idx] if self.tapName is not None else '', 
                                            'tapCoord': list(self.tapCoord[idx]) if self.tapCoord is not None else [],
                                           }
                    
        self.tapIdx = tapIdxList
        
        if thisFaceHasChanged:
            print(f"Removed bad taps from face {self.name}:\n\t{self.RemovedBadTaps}")
            self.Refresh()
    
    def Refresh(self, showDetailedLog=False):
        self.__generateTributaries(showDetailedLog)
        self.__generatePanels(showDetailedLog)

    def copy(self):
        return copy.deepcopy(self)

    def to_dict(self,getDerived=False):
        basic = {}
        basic['ID'] = self.ID
        basic['name'] = self.name
        basic['note'] = self.note
        basic['origin'] = self.origin
        basic['basisVectors'] = self.basisVectors
        basic['vertices'] = self.vertices.tolist()
        basic['zoneDict'] = self.zoneDict
        if basic['zoneDict'] is not None:
            for val in basic['zoneDict']:
                if not basic['zoneDict'][val][2] == []:
                    temp = list(basic['zoneDict'][val][2])
                    for i,x in enumerate(temp):
                        temp[i] = list(x)
                    basic['zoneDict'][val][2] = temp
        basic['nominalPanelAreas'] = self.panelAreas_nominal
        basic['numOfNominalPanelAreas'] = self.numOfNominalPanelAreas
        basic['tapNo'] = None if self.tapNo is None else self.tapNo.tolist()
        basic['tapIdx'] = None if self.tapIdx is None else self.tapIdx.tolist()
        basic['tapName'] = None if self.tapName is None else self.tapName.tolist()
        basic['tapCoord'] = None if self.tapCoord is None else self.tapCoord.tolist()

        derived = None
        if getDerived:
            derived = {}
            derived['tapTribs'] = self.tapTribs
            derived['panels'] = self.panels
            derived['panelAreas'] = self.panelAreas
            derived['tapWghtPerPanel'] = self.tapWghtPerPanel
            derived['tapIdxPerPanel'] = self.tapIdxPerPanel
            derived['error_in_panels'] = self.error_in_panels
            derived['error_in_zones'] = self.error_in_zones
        
        return basic, derived
        
    def from_dict(self,basic,derived=None):
        self.ID = basic['ID']
        self.name = basic['name']
        self.note = basic['note']
        self.origin = basic['origin']
        self.basisVectors = basic['basisVectors']
        self.vertices = np.array(basic['vertices'])
        self.zoneDict = basic['zoneDict']
        if self.zoneDict is not None:
            for val in self.zoneDict:
                self.zoneDict[val][2] = np.array(self.zoneDict[val][2])
        self.panelAreas_nominal = basic['nominalPanelAreas']
        self.numOfNominalPanelAreas = basic['numOfNominalPanelAreas']
        self.tapNo = np.array(basic['tapNo'])
        self.tapIdx = np.array(basic['tapIdx'])
        self.tapName = np.array(basic['tapName'])
        self.tapCoord = np.array(basic['tapCoord'])

        if derived is not None:
            self.tapTribs = derived['tapTribs']
            self.panels = derived['panels']
            self.panelAreas = derived['panelAreas']
            self.tapWghtPerPanel = derived['tapWghtPerPanel']
            self.tapIdxPerPanel = derived['tapIdxPerPanel']
            self.error_in_panels = derived['error_in_panels']
            self.error_in_zones = derived['error_in_zones']
        else:
            self.Refresh()

    def writeToFile(self,file_basic, file_derived=None):
        getDerived = file_derived is not None
        basic, derived = self.to_dict(getDerived=getDerived)

        with open(file_basic, 'w') as f:
            json.dump(basic,f, indent=4, separators=(',', ':'))
        
        if file_derived is not None:
            with open(file_derived, 'w') as f:
                json.dump(derived,f, indent=4, separators=(',', ':'))
    
    def readFromFile(self,file_basic, file_derived=None):
        with open(file_basic, 'r') as f:
            basic = json.load(f)
        
        if file_derived is not None:
            with open(file_derived, 'r') as f:
                derived = json.load(f)
        else:
            derived = None

        self.from_dict(basic, derived=derived)

    def getNearestTapsToLine__to_be_redacted(self, start: np.ndarray, end: np.ndarray, distTolerance: float=0.0001, debug=False):
        """
        Returns the tap number that is nearest to the line defined by the start and end points. The distance is measured in the face-local coordinate system.

        Parameters
        ----------
        start : np.ndarray
            [2,]
            The start point of the line.
        end : np.ndarray
            [2,]
            The end point of the line.
        distTolerance : float, optional
            The tolerance of the distance between the line and the tap. The default is 0.0001.

        Returns
        -------
        tapNos : List[int]
            The tap numbers that are nearest to the line.
        tapIdxs : List[int]
            The tap indices in the main building-level list of taps that are nearest to the line.
        dist_from_start : List[float]
            The distance of the taps from the start point of the line.
        """
        if self.tapCoord is None:
            return None, None
        if debug:
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot()
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-')
            ax.plot(self.vertices[:,0], self.vertices[:,1], 'k-')
            ax.plot(self.tapCoord[:,0], self.tapCoord[:,1], '.k')
            ax.plot(start[0], start[1], 'ro')
            ax.plot(end[0], end[1], 'go')
            # ax.axis('equal')
            # plt.show()
            
        start, end = np.array(start), np.array(end)
        
        tapNos = []
        tapIdxs = []
        dist_from_start = []
        for t,tp in enumerate(self.tapCoord):
            d = linalg.norm(np.cross(end-start, start-tp))/linalg.norm(end-start)
            if d <= distTolerance:
                tapNos.append(self.tapNo[t])
                tapIdxs.append(self.tapIdx[t])
                dist_from_start.append(linalg.norm(tp-start))
        # sort the tapNos and tapIdxs by distance from the start point
        tapNos = [x for _,x in sorted(zip(dist_from_start,tapNos))]
        tapIdxs = [x for _,x in sorted(zip(dist_from_start,tapIdxs))]
        
        if debug:
            ax.plot(self.tapCoord[tapIdxs,0], self.tapCoord[tapIdxs,1], 'xr')
            # plot the tolerance margin around the line
            start_tol_upper_point = start + distTolerance*(end-start)/linalg.norm(end-start)
            start_tol_lower_point = start - distTolerance*(end-start)/linalg.norm(end-start)
            end_tol_upper_point = end + distTolerance*(end-start)/linalg.norm(end-start)
            end_tol_lower_point = end - distTolerance*(end-start)/linalg.norm(end-start)
            print(f"start_tol_upper_point: {start_tol_upper_point}")
            print(f"start: {start}")
            print(f"start_tol_lower_point: {start_tol_lower_point}")
            print(f"end_tol_upper_point: {end_tol_upper_point}")
            print(f"end: {end}")
            print(f"end_tol_lower_point: {end_tol_lower_point}")
            # plot a filled polygon
            ax.fill([start_tol_upper_point[0], end_tol_upper_point[0], end_tol_lower_point[0], start_tol_lower_point[0]],
                    [start_tol_upper_point[1], end_tol_upper_point[1], end_tol_lower_point[1], start_tol_lower_point[1]],
                    'r', alpha=0.8)
            # edges of the polygon
            ax.plot([start_tol_upper_point[0], end_tol_upper_point[0]], [start_tol_upper_point[1], end_tol_upper_point[1]], 'r-', alpha=0.3)
            
            ax.axis('equal')
            plt.show()
        
        return tapNos, tapIdxs, dist_from_start
    
    def indexOfTap(self, tapNo):
        if self.tapNo is None:
            return None
        return np.where(np.isin(self.tapNo, tapNo))[0]

    def getTapsInFence(self, fence, debug=False):
        if self.tapCoord is None:
            return None
        fence = shp.Polygon(fence)
        # localIdx = fence.contains(shp.MultiPoint(self.tapCoord))
        tapIdx_inFace = []
        tapIdx_inData = []
        for i,tp in enumerate(self.tapCoord):
            if debug:
                print(f"tap {i}: {tp}")
                print(f"tap {i} is in fence: {fence.contains(shp.Point(tp))}")
            if fence.contains(shp.Point(tp)):
                tapIdx_inFace.append(i)
                tapIdx_inData.append(self.tapIdx[i])
        return tapIdx_inFace, tapIdx_inData

    def toGlobalCoord(self, points_inLocalCoords, debug=False, kwargs={}):
        points_inLocalCoords = np.array(points_inLocalCoords)
        
        # vertics_local = np.hstack([fc.vertices, np.zeros((len(fc.vertices),1))])
        if points_inLocalCoords.shape[1] == 2:
            points_inLocalCoords = np.hstack([points_inLocalCoords, np.zeros((len(points_inLocalCoords),1))])
            
        origin_localInGlobal = np.array(self.origin)
        basisVectors_local = np.array(self.basisVectors)

        transformation_matrix_local2global = np.linalg.inv(basisVectors_local)
        transformation_matrix_global2local = np.linalg.inv(transformation_matrix_local2global)

        points_global = points_inLocalCoords @ transformation_matrix_local2global + origin_localInGlobal
        points_backToLocal = (points_global - origin_localInGlobal) @ transformation_matrix_global2local 

        the_two_are_the_same = np.allclose(points_inLocalCoords, points_backToLocal)
        
        if not the_two_are_the_same:
            print(f"Error in transformation from local to global coordinates.")
            print(f"Local:\n{np.array2string(points_inLocalCoords, precision=3, separator=',', suppress_small=True)}")
            print(f"Global:\n{np.array2string(points_global, precision=3, separator=',', suppress_small=True)}")
            print(f"Back to Local:\n{np.array2string(points_backToLocal, precision=3, separator=',', suppress_small=True)}")
            return None
        
        if debug:
            print(f"points_inLocalCoords:\n{np.array2string(points_inLocalCoords, precision=3, separator=',', suppress_small=True)}")
            print(f"transformation_matrix_local2global:\n{np.array2string(transformation_matrix_local2global, precision=3, separator=',', suppress_small=True)}")
            print(f"points_global:\n{np.array2string(points_global, precision=3, separator=',', suppress_small=True)}")
            print(f"re-transformation successfull: {the_two_are_the_same}")
        
        return points_global
        
        # # transform(self.vertices, self.origin, self.basisVectors, inverse=True, debug=False)
        # return transform(geomIn=points, orig=self.origin, T=self.basisVectors, inverseTransform=True, debug=debug, **kwargs)

    def toLocalCoord(self, points_inGlobalCoords, debug=False, kwargs={}):
        points_inGlobalCoords = np.array(points_inGlobalCoords)
        # return transform(geomIn=points, orig=self.origin, T=self.basisVectors, inverseTransform=False, debug=debug, **kwargs)
        Origin_ofLocal_inGlobalSys = np.array(self.origin)
        basisVectors_local = np.array(self.basisVectors)

        transformation_matrix_local2global = np.linalg.inv(basisVectors_local)
        transformation_matrix_global2local = np.linalg.inv(transformation_matrix_local2global)

        # points_local = points_inGlobalCoords @ transformation_matrix_global2local + Origin_ofLocal_inGlobalSys
        # points_backToGlobal = (points_local - Origin_ofLocal_inGlobalSys) @ transformation_matrix_local2global
        
        points_local = (points_inGlobalCoords - Origin_ofLocal_inGlobalSys) @ transformation_matrix_global2local
        points_backToGlobal = points_local @ transformation_matrix_local2global + Origin_ofLocal_inGlobalSys
        
        the_two_are_the_same = np.allclose(points_inGlobalCoords, points_backToGlobal)
        if not the_two_are_the_same:
            print(f"Error in transformation from global to local coordinates.")
            print(f"Global: {points_inGlobalCoords}")
            print(f"Local: {points_local}")
            print(f"Back to Global: {points_backToGlobal}")
            return None
        
        if debug:
            print(f"Origin_ofLocal_inGlobalSys:\n{np.array2string(Origin_ofLocal_inGlobalSys, precision=3, separator=',', suppress_small=True)}\n")
            print(f"basisVectors_local:\n{np.array2string(basisVectors_local, precision=3, separator=',', suppress_small=True)}\n")
            print(f"points_inGlobalCoords:\n{np.array2string(points_inGlobalCoords, precision=3, separator=',', suppress_small=True)}\n")
            print(f"transformation_matrix_global2local:\n{np.array2string(transformation_matrix_global2local, precision=3, separator=',', suppress_small=True)}\n")
            print(f"points_local:\n{np.array2string(points_local, precision=3, separator=',', suppress_small=True)}\n")
            print(f"re-transformation successfull: {the_two_are_the_same}\n")
        
        return points_local
        

    """--------------------------------- Plotters -------------------------------------"""
    def plotLocalAxes(self, ax=None, showLabels=True, drawOrigin=True, drawBasisVectors=True, vectorSize=1.0, 
                    kwargs_vec_x={'arrowstyle':'->', 'lw':1.0, 'color':'r', 'connectionstyle':'arc3,rad=0.0'},
                    kwargs_vec_y={'arrowstyle':'->', 'lw':1.0, 'color':'b', 'connectionstyle':'arc3,rad=0.0'},
                    kwargs_Origin={'color':'k', 'marker':'o', 'ms':2.0, 'mfc':'k', 'mec':'none'},
                    kwargs_x_label={'ha':'center', 'va':'center', 'color':'r', 'backgroundcolor':[1,1,1,0.3], 'fontsize':14},
                    kwargs_y_label={'ha':'center', 'va':'center', 'color':'b', 'backgroundcolor':[1,1,1,0.3], 'fontsize':14},
                    ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        orig = tranform2D(self.origin_plt[0], self.origin_plt[1], [0,0], self.basisVectors_plt[0], self.basisVectors_plt[1])
        
        if drawOrigin:
            ax.plot(orig[0], orig[1], **kwargs_Origin)
        if drawBasisVectors:
            # arrowLength = vectorSize*np.max([np.max(self.vertices[:,0])-np.min(self.vertices[:,0]), np.max(self.vertices[:,1])-np.min(self.vertices[:,1])])
            # use fixed length relative to the axis bounding box (average of the x and y axis lengths) determined from axis limits
            axLimits = ax.axis()
            arrowLength = vectorSize*0.1*np.mean([axLimits[1]-axLimits[0], axLimits[3]-axLimits[2]])
            
            if showLabels:
                ax.text((orig[0]+self.basisVectors_plt[0][0]*arrowLength)*1.1, orig[1]+self.basisVectors_plt[0][1]*arrowLength/2, 'x', **kwargs_x_label)
                ax.text(orig[0]+self.basisVectors_plt[1][0]*arrowLength/2, (orig[1]+self.basisVectors_plt[1][1]*arrowLength)*1.1, 'y', **kwargs_y_label)
        
            # use ax.annotate with 'arrowtext' to plot the arrows with text instead of the ax.arrow and ax.text
            ax.annotate('  ', xy=(orig[0]+self.basisVectors_plt[0][0]*arrowLength, orig[1]+self.basisVectors_plt[0][1]*arrowLength), 
                        xytext=(orig[0], orig[1]), 
                        arrowprops=kwargs_vec_x, ha='center', va='center')
            ax.annotate(' ', xy=(orig[0]+self.basisVectors_plt[1][0]*arrowLength, orig[1]+self.basisVectors_plt[1][1]*arrowLength),
                        xytext=(orig[0], orig[1]), 
                        arrowprops=kwargs_vec_y, ha='center', va='center')
        
        if newFig:
            ax.axis('equal')
    
    def plotEdges(self, ax=None, showName=True, drawOrigin=False, drawLocalAxes=False, basisVectorLength=1.0, fill=False, 
                  kwargs_Edge={'color':'k', 'lw':1.0, 'ls':'-'}, 
                  kwargs_Name={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.3]},
                  kwargs_Fill={'facecolor':[0.8,0.8,0.8,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'},
                  kwargs_lclAxs={'showLabels':True, 'drawOrigin':True, 'drawBasisVectors':True, 'vectorSize':1.0,
                                    'kwargs_vec_x':{'color':'r', 'lw':1.0, 'ls':'-'}, 'kwargs_vec_y':{'color':'b', 'lw':1.0, 'ls':'-'},
                                    'kwargs_Origin':{'color':'k', 'marker':'o', 'ms':5.0, 'mfc':'k', 'mec':'k'},
                                    'kwargs_x_label':{'ha':'center', 'va':'center', 'color':'r', 'backgroundcolor':[1,1,1,0.3]},
                                    'kwargs_y_label':{'ha':'center', 'va':'center', 'color':'b', 'backgroundcolor':[1,1,1,0.3]},
                                    }
                  ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.verticesPlt)
        ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
        if fill:
            ax.fill(xy[:,0], xy[:,1], **kwargs_Fill)
        if showName:
            txt = self.name
            txt = txt.replace('_', '\_')
            ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), txt, **kwargs_Name)
        if drawOrigin:
            # ax.plot(self.origin_plt[0], self.origin_plt[1], 'ko')
            pass
        if drawLocalAxes:
            self.plotLocalAxes(ax=ax, **kwargs_lclAxs)
        if newFig:
            ax.axis('equal')

    def plotEdges_3D(self, ax=None, showName=True, drawOrigin=False, drawBasisVectors=False, basisVectorLength=1.0, fill=True,
                    kwargs_Edge={'color':'k', 'lw':1.0, 'ls':'-'},
                    kwargs_Name={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.3]},
                    kwargs_Fill={'facecolor':[0.0,0.0,0.5,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'},
                    ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        xyz = np.array(self.vertices3D)
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], **kwargs_Edge)
        if fill:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            vertices = xyz
            ax.add_collection3d(Poly3DCollection([vertices], facecolors=kwargs_Fill['facecolor'], edgecolors=kwargs_Fill['edgecolor'], lw=kwargs_Fill['lw'], ls=kwargs_Fill['ls']))
        # # overlay a transparent filled polygon
        # verts = xyz
        # verts = np.append(verts, [verts[0]], axis=0)
        # poly = Poly3DCollection([verts], alpha=0.3)
        # ax.add_collection3d(poly, zs='z')
        
        if newFig:
            ax.axis('equal')
    
    def plotTaps(self, ax=None, tapsToPlot=None, showTapNo=False, showTapName=False, textOffset_tapNo=[0,0], textOffset_tapName=[0,0],
                 kwargs_dots={'color':'k', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                 kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        if tapsToPlot is None:
            tapsToPlot = self.tapNo
        
        tapIdx = np.where(np.isin(self.tapNo, tapsToPlot))[0]
        if len(tapIdx) == 0:
            # print(f"No taps to plot for face {self.name}. Skipping ...")
            return
        foundTapNos = self.tapNo[tapIdx]
        foundTapCoord = self.tapCoordPlt[tapIdx]
        foundTapName = []
        if showTapName and self.tapName is not None and len(self.tapName) > 0:
            foundTapName = np.array(self.tapName)[tapIdx]
            
        xy = np.array(foundTapCoord)
        if showTapNo:
            for t,tpNo in enumerate(foundTapNos):
                ax.text(xy[t,0]+textOffset_tapNo[0], xy[t,1]+textOffset_tapNo[1], str(tpNo), **kwargs_text)
        if showTapName and len(foundTapName) > 0:
            for t,tpName in enumerate(foundTapName):
                ax.text(xy[t,0]+textOffset_tapName[0], xy[t,1]+textOffset_tapName[1], str(tpName), **kwargs_text)
        ax.plot(xy[:,0], xy[:,1], **kwargs_dots)
        if newFig:
            ax.axis('equal')
    
    def plotTribs(self, ax=None, 
                  kwargs_Edge={'color':'r', 'lw':0.5, 'ls':'-', 'marker':'None', 'markersize':3},):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for trib in self.tapTribs.geoms:
            if type(trib) == shp.Polygon:
                xy = transform(np.transpose(trib.exterior.xy), self.origin_plt, self.basisVectors_plt) 
            elif type(trib) == shp.MultiPolygon:
                xy = []
                for t in trib.geoms:
                    xy.extend(transform(np.transpose(t.exterior.xy), self.origin_plt, self.basisVectors_plt))
                xy = np.array(xy)
            else:
                warnings.warn(f"Unknown geometry type: {type(trib)}")
                continue
            ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
        if newFig:
            ax.axis('equal')
        
    def plotZones(self, ax=None, zoneCol=None, drawEdge=False, fill=True, showLegend=True, zonesToPlot=None, overlayZoneIdx=False,
                  kwargs_Edge={'color':'k', 'lw':0.5, 'ls':'-'}, 
                  kwargs_Fill={}, 
                  kwargs_text={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.3]},
                  kwargs_Legend={'loc':'upper left', 'bbox_to_anchor':(1.05, 1), 'borderaxespad':0.}):
        if zoneCol is None:
            nCol = len(self.zoneDictUniqe)
            c = plt.cm.tab20(np.linspace(0,1,nCol))
            zoneCol = {}
            for i,z in enumerate(self.zoneDictUniqe):
                zn = self.zoneDictUniqe[z]
                zoneCol[zn[0]+', '+zn[1]] = c[i]
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        if zonesToPlot is None:
            zonesToPlot = self.zoneDict.keys()
        for zDict in self.zoneDict:
            if zDict not in zonesToPlot:
                continue
            xy = transform(self.zoneDict[zDict][2], self.origin_plt, self.basisVectors_plt)
            if drawEdge:
                ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
            if fill:
                zn = self.zoneDict[zDict]
                colIdx = zn[0]+', '+zn[1]
                ax.fill(xy[:,0], xy[:,1], facecolor=zoneCol[colIdx], **kwargs_Fill)
            if overlayZoneIdx:
                ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), str(zDict), **kwargs_text)
        if newFig:
            ax.axis('equal')
        if showLegend:
            patches = []
            for i,z in enumerate(zoneCol):
                patches.append(mpatches.Patch(color=zoneCol[z], label=z))
            ax.legend(handles=patches, **kwargs_Legend)

        return zoneCol

    def plotPanels(self, ax=None, aIdx=0, panelsToPlot=None, fill=False, overlayPanelIdx=False,
                    kwargs_Edge={'color':'g', 'lw':0.5, 'ls':'-', 'marker':'None', 'markersize':3}, 
                    kwargs_Fill={'facecolor':[0,1,0,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'},
                    kwargs_text={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.0]},
                   ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        if panelsToPlot is None:
            panelsToPlot = {}
            for z,zKey in enumerate(self.zoneDict):
                panelsToPlot[zKey] = {}
                for a, aKey in enumerate(self.panelAreas_nominal):
                    panelsToPlot[zKey][a] = np.arange(len(self.panels[z][a].geoms))
                    
        for zIdx,zKey in enumerate(self.zoneDict):
            if zKey not in panelsToPlot.keys():
                continue
            for pIdx,p in enumerate(self.panels[zIdx][aIdx].geoms):
                if pIdx not in panelsToPlot[zKey][aIdx]:
                    continue
                xy = transform(np.transpose(p.exterior.xy), self.origin_plt, self.basisVectors_plt)
                ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
                if fill:
                    ax.fill(xy[:,0], xy[:,1], **kwargs_Fill)
                if overlayPanelIdx:
                    ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), str(pIdx), **kwargs_text)
        if newFig:
            ax.axis('equal')

    def plotTapField(self, field, dx=None, ax=None, fldRange=None, nLvl=100, cmap='RdBu', extend='both', 
                     showValuesOnContour=True, kwargs_contourTxt={'inline':True, 'fmt':'{:.3g}', 'fontsize':6, 'colors':'k', },
                     showContourEdge=True, kwargs_contourEdge={'colors':'k', 'linewidths':0.5, 'linestyles':'solid'},
                     ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.tapCoordPlt)
        xO, yO = xy[:,0], xy[:,1]
        fld = field[self.tapIdx]
        
        if dx is None:
            nx = 30
            dx = np.ptp(self.vertices[:,0])/nx
        else:
            nx = int(np.ceil(np.ptp(self.vertices[:,0])/dx))
        dy = dx * np.ptp(self.vertices[:,1])/np.ptp(self.vertices[:,0])
        ny = int(np.ceil(np.ptp(self.vertices[:,1])/dy))

        x = np.linspace(min(xO), max(xO), nx)
        y = np.linspace(min(yO), max(yO), ny)
        X, Y = np.meshgrid(x,y)
        Z = griddata((xO,yO), fld, (X,Y))
        
        if fldRange is None:
            levels = np.linspace(min(field), max(field), nLvl)
        else:
            levels = np.linspace(fldRange[0], fldRange[1], nLvl)
        cObj = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend=extend,)
        if showContourEdge:
            cObj_l = ax.contour(X, Y, Z, levels=levels, **kwargs_contourEdge)
        else:
            cObj_l = cObj
        if showValuesOnContour:
            ax.clabel(cObj_l, cObj_l.levels, **kwargs_contourTxt)
            
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')
        return cObj

    def plotPanelField(self, field, fldName, dIdx=0, aIdx=0, showValueText=False, strFmt="{:.3g}", fldRange=None, ax=None, nLvl=100, cmap='RdBu'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        if fldRange is None:
            for z,_ in enumerate(self.zoneDict):
                val = field[z][aIdx][fldName][dIdx, :]
                if z == 0:
                    fldRange = [min(val), max(val)]
                else:
                    fldRange = [min([min(val), fldRange[0]]), max([max(val), fldRange[1]])]

        for z,_ in enumerate(self.zoneDict):
            for p,pnl in enumerate(self.panels[z][aIdx].geoms):
                xy = transform(np.transpose(pnl.exterior.xy), self.origin_plt, self.basisVectors_plt)
                val = field[z][aIdx][fldName][dIdx, p]
                valNorm = (val - fldRange[0])/(fldRange[1]-fldRange[0])
                cmap = plt.get_cmap(cmap)
                ax.fill(xy[:,0], xy[:,1], color=cmap(valNorm))
                if showValueText:
                    ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), strFmt.format(val),
                            ha='center', va='center', backgroundcolor=[1,1,1,0.3])
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')
            
        return

    def plot(self, figFile=None,xLimits=None,figSize=[15,5], showAxis=True, dotSize=3,
             overlayTaps=True, overlayTribs=True, overlayPanels=False, overlayZones=True, useSubplotForDifferentNominalAreas=True, nSubPlotCols=3):

        if overlayPanels:
            if useSubplotForDifferentNominalAreas:
                plt.figure(figsize=figSize)
            for a,area in enumerate(self.panelAreas_nominal):
                if useSubplotForDifferentNominalAreas:
                    nRows = int(np.ceil((len(self.panelAreas_nominal))/nSubPlotCols))
                    plt.subplot(nRows, nSubPlotCols, a+1)
                else:
                    plt.figure(figsize=figSize)
                xy = np.array(self.vertices)
                plt.plot(xy[:,0], xy[:,1], '-k', lw=2)
                if overlayTaps:
                    plt.plot(self.tapCoord[:,0],self.tapCoord[:,1],'.k',markersize=dotSize)
                if overlayTribs:
                    for trib in self.tapTribs.geoms:
                        x,y = trib.exterior.xy
                        plt.plot(x,y,':r',lw=0.5)
                for z,zone in enumerate(self.zoneDict.values()):
                    for p in self.panels[z][a]:
                        x,y = p.exterior.xy
                        plt.plot(x,y,'-b',lw=0.5)
                    pass
                if overlayZones:
                    self.plotZones(ax=plt.gca(), zoneCol=None, drawEdge=False, fill=True, showLegend=False, zonesToPlot=None, overlayZoneIdx=False,
                                      kwargs_Edge={'color':'k', 'lw':0.5, 'ls':'-'}, 
                                      kwargs_Fill={'facecolor':[0,1,0,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'}, 
                                      kwargs_text={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.3]},
                                      kwargs_Legend={'loc':'upper left', 'bbox_to_anchor':(1.05, 1), 'borderaxespad':0.})
                    # for zDict in self.zoneDict:
                    #     xy = self.zoneDict[zDict][2]
                    #     plt.plot(xy[:,0], xy[:,1], '--k', lw=2)
                plt.axis('equal')
                if not showAxis:
                    ax = plt.gca()
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                if not useSubplotForDifferentNominalAreas:
                    plt.show()
            if useSubplotForDifferentNominalAreas:
                plt.show()
        else:
            plt.figure(figsize=figSize)
            xy = np.array(self.vertices)
            plt.plot(xy[:,0], xy[:,1], '-k', lw=2)
            if overlayTaps:
                plt.plot(self.tapCoord[:,0],self.tapCoord[:,1],'.k',markersize=dotSize)
            if overlayTribs:
                for trib in self.tapTribs.geoms:
                    x,y = trib.exterior.xy
                    plt.plot(x,y,':r',lw=0.5)
            if overlayZones:
                for zDict in self.zoneDict:
                    xy = self.zoneDict[zDict][2]
                    plt.plot(xy[:,0], xy[:,1], '--k', lw=2)
            plt.axis('equal')
            if not showAxis:
                ax = plt.gca()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_frame_on(False)
            plt.show()

    def plotTilingErrors(self, ax=None, col='r', dotSz=3, lw=0.5, ls='-', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        self.plotEdges(ax=ax, col='k', dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

class Faces:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                memberFaces: List[face]=[],
                tapNos: List[int]=[],
                file_basic=None,
                file_derived=None,
                ):
        self._currentIndex = 0
        self._memberFaces: List[face] = memberFaces
        self.tapNo_all_lumped = tapNos

        if file_basic is not None and file_derived is not None:
            self.readFromFile(file_basic=file_basic,file_derived=file_derived)
        elif file_basic is not None:
            self.readFromFile(file_basic=file_basic)
    
    def _numOfMembers(self):
        if self.memberFaces is None:
            return 0
        return len(self.memberFaces)

    def __iter__(self):
        return self

    def __next__(self) -> face:
        if self._currentIndex < self._numOfMembers():
            member = self._memberFaces[self._currentIndex]
            self._currentIndex += 1
            return member
        else:
            self._currentIndex = 0
            raise StopIteration

    def __getitem__(self, key) -> face:
        return self._memberFaces[key]
    
    def __setitem__(self, key: int, value: face):
        self._memberFaces[key] = value

    def __str__(self) -> str:
        name = 'Faces: '
        for f in self.memberFaces:
            name += f.name + ', '
        return name
        
    """--------------------------------- Properties -----------------------------------"""
    @property
    def memberFaces(self) -> List[face]:
        return self._memberFaces
    @memberFaces.setter
    def memberFaces(self,value: List[face]) -> None:
        self._memberFaces = value

    @property
    def tapNo(self) -> List[int]:
        if self._numOfMembers() == 0:
            return self.tapNo_all_lumped
        tapNo = []
        for f in self.memberFaces:
            tapNo.extend(f.tapNo)
        return tapNo

    # @property
    # def tapNo_all(self) -> List[int]:
    #     tapNo = []
    #     for f in self.memberFaces:
    #         tapNo.extend(f.tapNo_all)
    #     return tapNo

    @property
    def tapIdx(self) -> List[int]:
        if self._numOfMembers() == 0:
            return np.arange(len(self.tapNo_all_lumped))
        tapIdx = []
        for f in self.memberFaces:
            tapIdx.extend(f.tapIdx)
        return tapIdx

    @property
    def tapName(self) -> List[str]:
        if self._numOfMembers() == 0:
            return None
        tapName = []
        for f in self.memberFaces:
            if f.tapName is None:
                tapName_i = ['']*f.NumTaps
            else:
                tapName_i = f.tapName
            tapName.extend(tapName_i)
        return tapName
    
    @property
    def RemovedBadTaps(self) -> Dict[str, List[int]]:
        if self._numOfMembers() == 0:
            return None
        removedTaps = {}
        for f in self.memberFaces:
            if f.RemovedBadTaps is None:
                continue
            for key in f.RemovedBadTaps:
                if key in removedTaps.keys():
                    raise ValueError(f"Key {key} already exists in the dictionary.")
                removedTaps[key] = f.RemovedBadTaps[key]
                
        return removedTaps
    
    @property
    def panelArea_nominal(self) -> List[float]:
        nomArea = self._memberFaces[0].panelAreas_nominal
        for f in self.memberFaces:
            if f.panelAreas_nominal != nomArea:
                raise ValueError('All faces must have the same nominal panel areas.')
        return nomArea
    
    @property
    def panelAreas_every(self) -> dict:
        areas = {k:np.array([]) for k in self.zoneDictKeys}
        for _,fc in enumerate(self.memberFaces): # self.CpStatsAreaAvg # [Nface][Nzones][N_area]{nFlds}[N_AoA,Npanels]
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                area_z = []
                for a, _ in enumerate(self.panelArea_nominal):
                    area_z.extend(fc.panelAreas[z][a])
                areas[zKey] = np.concatenate((areas[zKey], np.array(area_z)))
        return areas
    
    @property
    def panelAreas_groupAvg(self) -> dict:
        areas = {k:{ar:[] for ar in self.panelArea_nominal} for k in self.zoneDictKeys}
        for _,fc in enumerate(self.memberFaces):
            for z, zone in enumerate(fc.zoneDict):
                zKey = fc.zoneDict[zone][0] + ', ' + fc.zoneDict[zone][1]
                for a, ar in enumerate(self.panelArea_nominal):
                    areas[zKey][ar].extend(fc.panelAreas[z][a])
        aAvg = {k:np.array([], dtype=float) for k in self.zoneDictKeys}
        for zKey in areas:
            for ar in areas[zKey]:
                aAvg[zKey] = np.append(aAvg[zKey], np.mean(areas[zKey][ar]))
                
        return aAvg
    
    @property
    def zoneDict(self):
        allZones = {}
        i = 0
        for fc in self.memberFaces:
            for val in fc.zoneDict.values():
                isNewVal = True
                for acceptedZone in allZones.values():
                    if val[0] == acceptedZone[0] and val[1] == acceptedZone[1]:
                        isNewVal = False
                        break
                if isNewVal:
                    x = list(val)[0:2]
                    x.append([])
                    allZones[i] = x
                    i += 1
        return allZones

    @property
    def wallTaps(self):
        tapNos = []
        for fc in self.memberFaces:
            if fc.faceType == 'wall':
                tapNos.extend(fc.tapNo)
        return tapNos
    
    @property
    def roofTaps(self):
        tapNos = []
        for fc in self.memberFaces:
            if fc.faceType == 'roof':
                tapNos.extend(fc.tapNo)
        return tapNos

    @property
    def zoneDictKeys(self):
        zoneDict = self.zoneDict
        return [val[0]+', '+val[1] for val in zoneDict.values()]
        
    @property
    def NumZones(self):
        return len(self.zoneDict)

    @property
    def NumTaps(self):
        num = 0
        for fc in self.memberFaces:
            num += fc.NumTaps
        return num

    @property
    def NumPanels(self):
        if len(self.memberFaces) == 0:
            return 0
        num = 0
        for fc in self.memberFaces:
            num += fc.NumPanels
        return num

    @property
    def error_in_panels(self):
        val = []
        for fc in self._memberFaces:
            val.append(fc.error_in_panels)
        return val

    @property
    def error_in_zones(self):
        val = []
        for fc in self._memberFaces:
            val.append(fc.error_in_zones)
        return val

    @property
    def NumPanelsPerArea(self):
        if len(self.memberFaces) == 0:
            return 0
        for i,fc in enumerate(self.memberFaces):
            if i == 0:
                num = fc.NumPanelsPerArea
            else:
                num = np.add(fc.NumPanelsPerArea,num)
        return num

    @property
    def tapWeightsPerPanel(self):
        if self._numOfMembers() == 0:
            return None
        mainZoneDict = self.zoneDict
        Weights = list([[]]*len(mainZoneDict))   # [nZones][nPanels][nTapsPerPanel_i]
        for fc in self.memberFaces:
            for z, zonei in enumerate(fc.zoneDict.values()):
                zone = list(zonei)[0:2]
                zone.append([])
                idxZ = list(mainZoneDict.values()).index(zone) # index of the current zone in the main zoneDict
                for a,_ in enumerate(fc.panelAreas_nominal):
                    Weights[idxZ].extend(fc.tapWghtPerPanel[z][a])     # [nZones][nAreas][nPanels][nTapsPerPanel]
        return Weights

    @property
    def tapIdxsPerPanel(self):
        if self._numOfMembers() == 0:
            return None, None, None
        mainZoneDict = self.zoneDict
        tapIdxs = list([[]]*len(mainZoneDict))   # [nZones][nPanels][nTapsPerPanel_i]
        for fc in self.memberFaces:
            for z, zonei in enumerate(fc.zoneDict.values()):
                zone = list(zonei)[0:2]
                zone.append([])
                idxZ = list(mainZoneDict.values()).index(zone) # index of the current zone in the main zoneDict
                for a,_ in enumerate(fc.panelAreas_nominal):
                    tapIdxs[idxZ].extend(fc.tapIdxPerPanel[z][a]) 
        return tapIdxs

    @property
    def panelAreas__to_be_redacted(self):
        if self._numOfMembers() == 0:
            return None
        zNames = []
        for z, zn in enumerate(self.zoneDict):
            zNames.append(self.zoneDict[zn][0]+'_'+self.zoneDict[zn][1])
        pnlAreas = self.zoneDict
        for zm, zone_m in enumerate(pnlAreas):
            pnlAreas[zone_m][2] = []

        for f,fc in enumerate(self.faces):
            for z,zone in enumerate(fc.zoneDict):
                zIdx = zNames.index(fc.zoneDict[zone][0]+'_'+fc.zoneDict[zone][1])
                for a,_ in enumerate(fc.nominalPanelAreas):
                    pnlAreas[zIdx][2].extend(fc.panelAreas[z][a])
        return pnlAreas

    @property
    def panelIdxRanges(self):
        # mainZoneDict = self.zoneDict
        # s = []   # [nZones][nPanels][nTapsPerPanel_i]
        # for fc in self.members:
        #     for z, zonei in enumerate(fc.zoneDict.values()):
        #         zone = list(zonei)[0:2]
        #         zone.append([])
        #         idxZ = list(mainZoneDict.values()).index(zone) # index of the current zone in the main zoneDict
        #         for a,_ in enumerate(fc.nominalPanelAreas):
        #             Weights[idxZ].extend(fc.tapWghtPerPanel[z][a])     # [nZones][nAreas][nPanels][nTapsPerPanel]
        return None
        pass

    @property
    def panelingErrors(self):
        if self._numOfMembers() == 0:
            return None
        errDict = {}
        for f, fc in enumerate(self._memberFaces):
            errDict[f"Face {f+1}"] = fc.panelingErrors
        return errDict
    
    @property
    def boundingBoxPlt(self):
        if self._numOfMembers() == 0:
            return None
        minX, maxX, minY, maxY = [], [], [], []
        for fc in self.memberFaces:
            minX.append(np.min(fc.verticesPlt[:,0]))
            maxX.append(np.max(fc.verticesPlt[:,0]))
            minY.append(np.min(fc.verticesPlt[:,1]))
            maxY.append(np.max(fc.verticesPlt[:,1]))
        bb = [np.min(minX), np.max(maxX), np.min(minY), np.max(maxY)]
        return bb

    """-------------------------------- Data handlers ---------------------------------"""
    def Refresh(self):
        for fc in self.memberFaces:
            fc.Refresh()

    def tapIdxOf(self,tapNo,returnIdxInDataMatrixInstead=False) -> List[int]:
        tapNo = [tapNo,] if np.isscalar(tapNo) else tapNo
        tapNo = np.array(tapNo)
        allIdx = np.array(self.tapIdx)
        allTapNo = np.array(self.tapNo)
        if not np.shape(allIdx) == np.shape(allTapNo):
            msg = f"The tapIdx (shape: {np.shape(allIdx)}) does not match the tapNo (shape: {np.shape(allTapNo)}) in this object."
            raise Exception(msg)
        
        foundTaps_idxInDataMtx = []
        foundTaps_idx = []
        notFoundTaps = []
        for t in tapNo:
            if t in allTapNo:
                localIdx = np.where(allTapNo == t)[0][0]
                foundTaps_idx.append(localIdx)
                foundTaps_idxInDataMtx.append(allIdx[localIdx])
            else:
                notFoundTaps.append(t)

        if len(notFoundTaps) > 0:
            msg = f"ERROR: The following taps are not found in the object: {notFoundTaps}"
            raise Exception(msg)
        if len(foundTaps_idxInDataMtx) == 1:
            return foundTaps_idxInDataMtx[0]
        if returnIdxInDataMatrixInstead:
            return foundTaps_idxInDataMtx
        else:
            return foundTaps_idx

    def writeToFile(self,file_basic, file_derived=None):
        getDerived = file_derived is not None
        basic = {}
        derived = {}
        for i,fc in enumerate(self.memberFaces):
            basic[i], derived[i] = fc.to_dict(getDerived=getDerived)

        with open(file_basic, 'w') as f:
            json.dump(basic,f, indent=4, separators=(',', ':'))
        
        if file_derived is not None:
            with open(file_derived, 'w') as f:
                json.dump(derived,f, indent=4, separators=(',', ':'))

    def readFromFile(self, file_basic, file_derived=None):
        with open(file_basic, 'r') as f:
            basic = json.load(f)
        
        if file_derived is not None:
            with open(file_derived, 'r') as f:
                derived = json.load(f)
        else:
            derived = None

        self.memberFaces = []
        for i, bsc in enumerate(basic):
            mem = face()
            if derived is None:
                mem.from_dict(basic=basic[bsc])
            else:
                mem.from_dict(basic=basic[bsc],derived=derived[bsc])
            self.memberFaces.append(mem)

    def copy(self):
        newFaces = []
        for fc in self.memberFaces:
            newFaces.append(fc.copy())
        return Faces(memberFaces=newFaces,
                     tapNos=self.tapNo_all_lumped.copy(),
                     )

    """--------------------------------- Plotters -------------------------------------"""
    def plotLocalAxes(self, ax=None, showLabels=True, drawOrigin=True, drawBasisVectors=True, vectorSize=1.0, 
                    kwargs_vec_x={'arrowstyle':'->', 'lw':1.0, 'color':'r', 'connectionstyle':'arc3,rad=0.0'},
                    kwargs_vec_y={'arrowstyle':'->', 'lw':1.0, 'color':'b', 'connectionstyle':'arc3,rad=0.0'},
                    kwargs_Origin={'color':'k', 'marker':'o', 'ms':2.0, 'mfc':'k', 'mec':'none'},
                    kwargs_x_label={'ha':'center', 'va':'center', 'color':'r', 'backgroundcolor':[1,1,1,0.3], 'fontsize':14},
                    kwargs_y_label={'ha':'center', 'va':'center', 'color':'b', 'backgroundcolor':[1,1,1,0.3], 'fontsize':14},
                ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotLocalAxes(ax=ax, showLabels=showLabels, drawOrigin=drawOrigin, drawBasisVectors=drawBasisVectors, vectorSize=vectorSize, 
                            kwargs_vec_x=kwargs_vec_x, kwargs_vec_y=kwargs_vec_y, kwargs_Origin=kwargs_Origin,
                            kwargs_x_label=kwargs_x_label, kwargs_y_label=kwargs_y_label)
        if newFig:
            ax.axis('equal')
    
    def plotEdges(self, ax=None, showName=True, fill=False, 
                  kwargs_Edge={'color':'k', 'lw':1.0, 'ls':'-'},
                  kwargs_Fill={'facecolor':[0.8,0.8,0.8,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'},
                  ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotEdges(ax=ax, showName=showName, fill=fill,
                         kwargs_Edge=kwargs_Edge, kwargs_Fill=kwargs_Fill)
        if newFig:
            ax.axis('equal')

    def plotTaps(self, ax=None, tapsToPlot=None, showTapNo=False, showTapName=False, textOffset_tapNo=[0,0], textOffset_tapName=[0,0],
                 kwargs_dots={'color':'k', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                 kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        if tapsToPlot is None:
            tapsToPlot = self.tapNo
        for fc in self._memberFaces:
            fc.plotTaps(ax=ax, tapsToPlot=tapsToPlot, showTapNo=showTapNo, showTapName=showTapName, textOffset_tapNo=textOffset_tapNo, textOffset_tapName=textOffset_tapName,
                        kwargs_dots=kwargs_dots, kwargs_text=kwargs_text)
        if newFig:
            ax.axis('equal')

    def plotTribs(self, ax=None, 
                  kwargs_Edge={'color':'r', 'lw':0.5, 'ls':'-', 'marker':'None', 'markersize':3},
                  ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotTribs(ax=ax, kwargs_Edge=kwargs_Edge)
        if newFig:
            ax.axis('equal')

    def plotZones(self, ax=None, zoneCol=None, drawEdge=True, fill=True, showLegend=True,
                  kwargs_Edge={'color':'k', 'lw':0.5, 'ls':'-'}, 
                  kwargs_Fill={'alpha':0.7,},
                  kwargs_Legend={'loc':'upper left', 'bbox_to_anchor':(1.05, 1), 'borderaxespad':0.}):
        if zoneCol is None:
            nCol = len(self.zoneDict)
            c = plt.cm.tab20(np.linspace(0,1,nCol))
            zoneCol = {}
            for i,z in enumerate(self.zoneDict):
                zn = self.zoneDict[z]
                zoneCol[zn[0]+', '+zn[1]] = c[i]
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotZones(ax=ax, zoneCol=zoneCol, drawEdge=drawEdge, fill=fill, showLegend=False, kwargs_Edge=kwargs_Edge, kwargs_Fill=kwargs_Fill)
        if newFig:
            ax.axis('equal')
        patches = []
        for i,z in enumerate(zoneCol):
            patches.append(mpatches.Patch(color=zoneCol[z], label=z))
        if showLegend:
            lg = ax.legend(handles=patches, **kwargs_Legend)
        else:
            lg = None 

        return zoneCol, lg, patches

    def plotPanels(self, ax=None, aIdx=0, 
                   kwargs_Edge={'color':'b', 'lw':0.5, 'ls':'-', 'marker':'None', 'markersize':3},
                   ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotPanels(ax=ax, aIdx=aIdx, kwargs_Edge=kwargs_Edge)
        if newFig:
            ax.axis('equal')

    def plotPanels_AllAreas(self, 
                    figsize=(15,5), fig=None, axs=None, nCols=3,
                    areaUnit='', areaFactor=1.0, areaFmt="{:.4f}",
                    plotEdges=True, plotTaps=False, plotZones=True,
                    areaLabel_xy=(0.5, 0), 
                    addSubPlotLabels=True,
                    subPlotLabels=None,
                    subPlotLabel_xy=(0.05, 0.95),
                    kwargs_faceEdge={'color':'k', 'lw':1.0, 'ls':'-'},
                    kwargs_taps={},
                    kwargs_zoneEdge={'color':'k', 'lw':0.5, 'ls':'-'},
                    kwargs_zoneFill={'alpha':0.5,},
                    kwargs_pnlEdge={'color':'k', 'lw':0.3, 'ls':'-'},
                    kwargs_areaLabel={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'medium', },
                    kwargs_subPlotLabel={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'medium', },
                    ):
        newFig = False
        areas = self.panelArea_nominal
        if axs is None:
            newFig = True
            nRows = int(np.ceil(len(areas)/nCols))
            fig, axs = plt.subplots(nRows, nCols, figsize=figsize)
            axs = axs.flatten()

        if subPlotLabels is None and addSubPlotLabels:
            # create a list of subplot labels from a, b, c, ... If it exceeds 26, raise an error and prompt the user to provide a list of labels
            subPlotLabels = []
            for i in range(26):
                subPlotLabels.append('('+chr(i+97)+')')
            if len(areas) > 26:
                raise Exception("The number of subplots exceeds 26. Please provide a list of subplot labels.")

        for a,area in enumerate(areas):
            if plotZones:
                self.plotZones(ax=axs[a], showLegend=False, kwargs_Fill=kwargs_zoneFill, kwargs_Edge=kwargs_zoneEdge)
            if plotTaps:
                self.plotTaps(ax=axs[a], showTapNo=False, **kwargs_taps)
            self.plotPanels(ax=axs[a], aIdx=a, kwargs_Edge=kwargs_pnlEdge)
            if plotEdges:
                self.plotEdges(ax=axs[a], showName=False, kwargs_Edge=kwargs_faceEdge)
            area_string = areaFmt.format(areaFactor*area)
            area_string = r"$A_j="+area_string+r"$ "+areaUnit
            axs[a].annotate(area_string, xy=areaLabel_xy, xycoords='axes fraction', **kwargs_areaLabel)
            if addSubPlotLabels:
                axs[a].annotate(subPlotLabels[a], xy=subPlotLabel_xy, xycoords='axes fraction', **kwargs_subPlotLabel)
            
        if newFig:
            for a in range(len(axs)):
                axs[a].axis('equal')
                axs[a].axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
        return fig, axs

    def plotTapField(self, field, ax=None, dx=None, fldRange=None, nLvl=100, cmap='RdBu', extend='both',
                     showValuesOnContour=True, kwargs_contourTxt={'inline':True, 'fmt':'{:.3g}', 'fontsize':6, 'colors':'k'},
                     showContourEdge=True, kwargs_contourEdge={'colors':'k', 'linewidths':0.5, 'linestyles':'solid'},):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        contours = []
        for fc in self._memberFaces:
            c = fc.plotTapField(ax=ax, field=field, dx=dx, fldRange=fldRange, nLvl=nLvl, cmap=cmap, extend=extend,
                                showValuesOnContour=showValuesOnContour, kwargs_contourTxt=kwargs_contourTxt,
                                showContourEdge=showContourEdge, kwargs_contourEdge=kwargs_contourEdge,)
            contours.append(c)
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')
        return contours

    def plotPanelField(self, field, fldName, dIdx=0, aIdx=0, showValueText=False, strFmt="{:.3g}", fldRange=None, ax=None, nLvl=100, cmap='RdBu'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for f,fc in enumerate(self._memberFaces):
            fc.plotPanelField(field[f], fldName, dIdx=dIdx, aIdx=aIdx, showValueText=showValueText, strFmt=strFmt, fldRange=fldRange, ax=ax, nLvl=nLvl, cmap=cmap)
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')

class samplingLine:
    def __init__(self, 
                name="Unnamed line",
                parentFace: face=None,
                start_xy =None,
                end_xy=None,
                fringeDistance=0.00001,
                fringeDistanceMode: Literal['absolute','relative']='absolute',
                ):
        self.name = name
        self.parentFace: face = parentFace
        self.fringeDistance = fringeDistance
        self.fringeDistanceMode: Literal['absolute','relative'] = fringeDistanceMode
        self.start_xy = start_xy
        self.end_xy = end_xy
        
    def __str__(self):
        return 'Line name: '+self.name
    
    @property
    def origin(self):
        return self.start_xy
    
    @property
    def xy(self):
        if self.start_xy is None or self.end_xy is None:
            return None
        return np.array([self.start_xy, self.end_xy], dtype=float)
    
    @property
    def length(self):
        if self.start_xy is None or self.end_xy is None:
            return None
        vec = np.array([self.end_xy[0]-self.start_xy[0], self.end_xy[1]-self.start_xy[1]])
        return linalg.norm(vec)
    
    @property
    def basisVectors(self):
        if self.start_xy is None or self.end_xy is None:
            return None
        vec = np.array([self.end_xy[0]-self.start_xy[0], self.end_xy[1]-self.start_xy[1]])
        vec = vec/linalg.norm(vec)
        return np.array([vec, np.array([-vec[1], vec[0]])])
    
    @property
    def orientation(self):
        if self.start_xy is None or self.end_xy is None:
            return None
        return np.arctan2(self.end_xy[1]-self.start_xy[1], self.end_xy[0]-self.start_xy[0])
    
    @property
    def reverseBasisVectors(self):
        if self.start_xy is None or self.end_xy is None:
            return None
        invBasis = [[self.basisVectors[0,0], -self.basisVectors[0,1]],
                    [-self.basisVectors[1,0], self.basisVectors[1,1]]]
        return invBasis
    
    @property
    def fringeDistance_abs(self):
        if self.fringeDistance is None or self.start_xy is None or self.end_xy is None:
            return None
        if self.fringeDistanceMode == 'absolute':
            return self.fringeDistance
        elif self.fringeDistanceMode == 'relative':
            return self.fringeDistance*self.length
        else:
            raise Exception('Invalid fringeDistanceMode. Valid options are: absolute, relative.')
    
    @property
    def fringeZone_localCoord(self):
        # a polygon object of a rectangle that defines the boundary of the line to snatch taps from
        # coordinates are in the local coordinate system of the line
        if self.start_xy is None or self.end_xy is None or self.fringeDistance is None:
            return None
        # the corners are fringeDistance above and below the local x-axis at the origin and the end of the line
        f = self.fringeDistance_abs
        L = self.length
        return np.array([[0, f],
                        [L, f],
                        [L, -f],
                        [0, -f],
                        [0, f]])
        
    @property
    def fringeZone_faceCoord(self):
        if self.fringeDistance is None or self.start_xy is None or self.end_xy is None:
            return None
        x, y = self.fringeZone_localCoord[:,0], self.fringeZone_localCoord[:,1]
        return self.toFaceCoords(x, y)
       
    @property
    def tapIdx_inData(self):
        if self.parentFace is None:
            return None
        _, tapIdx_inData = self.parentFace.getTapsInFence(self.fringeZone_faceCoord)
        return np.asarray(tapIdx_inData, dtype=int)

    @property
    def tapIdx_inFace(self):
        if self.parentFace is None:
            return None
        lclIdx, _ = self.parentFace.getTapsInFence(self.fringeZone_faceCoord)
        return lclIdx
       
    @property
    def tapNo(self):
        if self.parentFace is None:
            return None
        return self.parentFace.tapNo[self.tapIdx_inFace]
       
    @property
    def tapCoord(self):
        return self.parentFace.tapCoord[self.tapIdx_inFace]
    
    @property
    def tapCoordLocal(self):
        return self.toLocalCoords(self.tapCoord[:,0], self.tapCoord[:,1])
    
    @property
    def L(self):
        if self.parentFace is None or self.tapCoordLocal is None:
            return None
        return self.tapCoordLocal[:,0]
    
    @property
    def sortOrder(self):
        if self.L is None:
            return None
        return np.argsort(self.L)
        
    @property
    def tapName(self):
        return self.parentFace.tapName[self.tapIdx_inData]
    
    '''---------------------- Properties in plot coordinates --------------------------'''
    @property
    def xy_plt(self):
        if self.start_xy is None or self.end_xy is None or self.parentFace is None:
            return None
        xy = np.array([self.start_xy, self.end_xy])
        return self.toFaceCoords_plt(xy[:,0], xy[:,1])
    
    @property
    def orientation_Plt(self):
        if self.start_xy is None or self.end_xy is None or self.parentFace is None or self.parentFace.orientation_plt is None or self.parentFace.orientation is None:
            return None
        ori = self.orientation
        # add the angle difference of the parent face between e1 (basis vector in x) and e1_plt (basis vector in x_plt)
        ori += self.parentFace.orientation_plt - self.parentFace.orientation
        return ori
    
    @property
    def tapCoord_Plt(self):
        return self.parentFace.tapCoordPlt[self.tapIdx_inFace]
    
    @property
    def fringeZone_faceCoord_plt(self):
        if self.fringeDistance is None or self.start_xy is None or self.end_xy is None:
            return None
        xy = self.fringeZone_faceCoord
        return self.toFaceCoords_plt(xy[:,0], xy[:,1])
    
    '''-------------------------------- Data handlers ---------------------------------'''
    def toLocalCoords(self, x, y):
        e1 = self.basisVectors[0]
        e2 = self.basisVectors[1]
        x,y = tranform2D(x, y, self.origin, e1, e2, translateFirst=True,)
        return np.array([x,y]).T
    
    def toFaceCoords(self, x, y, forPlot=False):
        x, y = np.array(x), np.array(y)
        e1, e2 = self.reverseBasisVectors[0], self.reverseBasisVectors[1]
        x,y = tranform2D(x, y, np.dot(-1,self.origin), e1, e2, translateFirst=False,)
        if forPlot:
            # return self.toFaceCoords_plt(np.array([x,y]).T)
            return self.toFaceCoords_plt(x, y)
        else:
            return np.array([x,y]).T
    
    def toFaceCoords_plt(self, x, y):
        x, y = np.array(x), np.array(y)
        e1, e2 = self.parentFace.basisVectors_plt[0], self.parentFace.basisVectors_plt[1]
        # return transform(xy, self.parentFace.origin_plt, self.parentFace.basisVectors_plt)
        x, y = tranform2D(x, y, self.parentFace.origin_plt, e1, e2, translateFirst=True,)
        return np.array([x,y]).T
    
    def faceCoordOf_L(self, L):
        return self.toFaceCoords(L, np.zeros_like(L))        
    
    '''--------------------------------- Plotters -------------------------------------'''
    def plotLine(self, ax=None, showName=True, 
                addArrowHead=False, arrowPosition:Literal['start','middle','end']='end', arrowSize=0.0005, arrowHeadSize=(None,None),
                alignTextToLine=False, txtDistFromLine=0.0,
                kwargs_Edge={'color':'k', 'lw':1.2, 'ls':'-', 'marker':'x', 'markersize':3}, 
                kwargs_Name={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.8], 'fontsize':'medium', },
                kwargs_Arrow={'head_width':0.004, 'head_length':0.012, 'fc':'k', 'ec':'k','ls':'-','lw':1.0},
                ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = self.xy_plt
        ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
        if addArrowHead:
            vector = self.toFaceCoords_plt(self.basisVectors[0,0], self.basisVectors[0,1])*arrowSize
            if arrowPosition == 'start':
                xy = xy[0,:]
                kwargs_Arrow['head_starts_at_zero'] = True
            elif arrowPosition == 'middle':
                xy = self.toFaceCoords(self.length/2-linalg.norm(self.basisVectors[0,:])*arrowSize, 0, forPlot=True)
                kwargs_Arrow['length_includes_head'] = True
            elif arrowPosition == 'end':
                xy = xy[1,:]
                xy = xy - vector
                kwargs_Arrow['length_includes_head'] = True
            else:
                raise Exception('Invalid arrowPosition. Valid options are: start, middle, end.')
            kwargs_Arrow['head_width'] = arrowHeadSize[0] if arrowHeadSize[0] is not None else kwargs_Arrow['head_width']
            kwargs_Arrow['head_length'] = arrowHeadSize[1] if arrowHeadSize[1] is not None else kwargs_Arrow['head_length']
            ax.arrow(xy[0], xy[1], vector[0], vector[1], 
                     **kwargs_Arrow)
            
        if showName:
            txt = self.name.replace('_', '\_')
            rotation = np.rad2deg(self.orientation_Plt)  if alignTextToLine else 0
            txt_xy = self.toFaceCoords((self.length/2,), (txtDistFromLine,), forPlot=True)[0]
            ax.text(txt_xy[0], txt_xy[1], txt, rotation=rotation,
                    **kwargs_Name)
            
        if newFig:
            ax.axis('equal')
    
    def plotBasisVectors(self, ax=None,):
        pass
    
    def plotFringeZone(self, ax=None, fill=True, edges=False,
                       kwargs_fill={'facecolor':[1,0,0,0.3], 'edgecolor':'None', 'lw':0.5, 'ls':'-'},
                       ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = self.fringeZone_faceCoord_plt
        if fill:
            kwargs_fill['facecolor'] = kwargs_fill['facecolor'] if 'facecolor' in kwargs_fill else [1,0,0,0.3]
        if edges:
            kwargs_fill['edgecolor'] = kwargs_fill['edgecolor'] if 'edgecolor' in kwargs_fill else 'k'
            kwargs_fill['lw'] = kwargs_fill['lw'] if 'lw' in kwargs_fill else 0.5
            kwargs_fill['ls'] = kwargs_fill['ls'] if 'ls' in kwargs_fill else '--'
            
        ax.fill(xy[:,0], xy[:,1], **kwargs_fill)
        
        if newFig:
            ax.axis('equal')
    
    def plotTaps(self, ax=None, showTapNo=False, showTapName=False, connectToLine=True, showProjectedLocationInstead=False,
                 textOffset_tapNo=[0,0], textOffset_tapName=[0,0],
                 kwargs_dots={'color':'r', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                 kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                 kwargs_connector={'color':'k', 'lw':0.5, 'ls':'--'},
                 ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
            
        xy = self.tapCoord_Plt
        xy_proj = self.toFaceCoords(self.tapCoordLocal[:,0], np.zeros_like(self.tapCoordLocal[:,0]), forPlot=True)
        if showProjectedLocationInstead:
            xy = xy_proj
        ax.plot(xy[:,0], xy[:,1], **kwargs_dots)
        if showTapNo:
            for i,xyi in enumerate(xy):
                ax.text(xyi[0]+textOffset_tapNo[0], xyi[1]+textOffset_tapNo[1], str(self.tapNo[i]), **kwargs_text)
        if showTapName:
            for i,xyi in enumerate(xy):
                ax.text(xyi[0]+textOffset_tapName[0], xyi[1]+textOffset_tapName[1], str(self.tapName[i]), **kwargs_text)
        if connectToLine:
            for i,(xyi, xyi_p) in enumerate(zip(xy, xy_proj)):
                ax.plot([xyi[0], xyi_p[0]], [xyi[1], xyi_p[1]], **kwargs_connector)
                
        if newFig:
            ax.axis('equal')
    
    def plot(self, ax=None, plotParentFace=True, detailed=False,
             kwargs_line=None, kwargs_fringeZone=None, kwargs_taps=None, 
             kwargs_faceEdge=None, kwargs_faceTaps=None):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        
        if kwargs_line is None:
            self.plotLine(ax=ax,)
        else:
            self.plotLine(ax=ax, **kwargs_line)
            
        if kwargs_taps is None:
            self.plotTaps(ax=ax,)
        else:
            self.plotTaps(ax=ax, **kwargs_taps)
        
        if detailed:
            if kwargs_fringeZone is None:
                self.plotFringeZone(ax=ax,)
            else:
                self.plotFringeZone(ax=ax, **kwargs_fringeZone)
                
            self.plotBasisVectors(ax=ax,)
                
            if plotParentFace:
                if kwargs_faceEdge is None:
                    self.parentFace.plotEdges(ax=ax,)
                else:
                    self.parentFace.plotEdges(ax=ax, **kwargs_faceEdge)
                if kwargs_faceTaps is None:
                    self.parentFace.plotTaps(ax=ax, showTapNo=False, kwargs_dots={'color':'k', 'marker':'.', 'ms':1, 'ls':''})
                else:
                    self.parentFace.plotTaps(ax=ax, showTapNo=False, **kwargs_faceTaps)

        if newFig:
            ax.axis('equal')
            ax.axis('off')

class SamplingLines:
    def __init__(self, 
                lines: List[samplingLine]=[],
                ):
        self.lines: List[samplingLine] = lines
        
    def __str__(self):
        return str([l.name for l in self.lines])
    
    def tapIdx(self, sortByL=True):
        tapIdx = []
        for l in self.lines:
            # print(f"Line {l.name}: l.tapIdx_inData: {l.tapIdx_inData}, type: {type(l.tapIdx_inData)}")
            # print(f"Line {l.name}: l.sortOrder: {l.sortOrder}, type: {type(l.sortOrder)}")
            # print(f"Line {l.name}: l.tapIdx_inData[l.sortOrder]: {np.asarray(l.tapIdx_inData)[l.sortOrder]}")
            if sortByL:
                tapIdx.extend(l.tapIdx_inData[l.sortOrder])
            else:
                tapIdx.extend(l.tapIdx_inData)
        return tapIdx
        
    def tapNo(self, sortByL=True):
        tapNo = []
        for l in self.lines:
            if sortByL:
                tapNo.extend(l.tapNo[l.sortOrder])
            else:
                tapNo.extend(l.tapNo)
        return tapNo
    
    @property
    def L(self):
        tap_L = []
        cummulativeLength = 0
        for l in self.lines:
            L = l.L[l.sortOrder] + cummulativeLength
            tap_L.extend(L)
            cummulativeLength += l.length
        return tap_L
    
    @property
    def joint_L(self):
        jL = []
        cummulativeLength = 0
        for l in self.lines:
            jL.append(cummulativeLength)
            cummulativeLength += l.length
        return jL
    
    @property
    def length(self):
        if len(self.lines) == 0:
            return None
        return np.sum([l.length for l in self.lines])
    
    def tapName(self, sortByL=True):
        tapName = []
        for l in self.lines:
            if sortByL:
                tapName.extend(l.tapName[l.sortOrder])
            else:
                tapName.extend(l.tapName)
        return tapName
    
    '''-------------------------------- Data handlers ---------------------------------'''
    
    def copy(self) -> 'SamplingLines':
        return copy.deepcopy(self)
    
    '''--------------------------------- Plotters -------------------------------------'''
    def plot(self, ax=None, plotParentFace=True, detailed=False, 
             kwargs_line=None, kwargs_fringeZone=None, kwargs_taps=None, 
             kwargs_faceEdge=None, kwargs_faceTaps=None):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        
        for l in self.lines:
            l.plot(ax=ax, plotParentFace=plotParentFace, detailed=detailed,
                   kwargs_line=kwargs_line, kwargs_fringeZone=kwargs_fringeZone, kwargs_taps=kwargs_taps, 
                   kwargs_faceEdge=kwargs_faceEdge, kwargs_faceTaps=kwargs_faceTaps)
        
        if newFig:
            ax.axis('equal')
            ax.axis('off')

class building(Faces):
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self, 
                name="Unnamed building",
                H=None,     # average roof height
                He=None,    # eave height
                Hr=None,    # ridge height 
                B=None,     # shorter plan-inscribing-rectangle width
                D=None,     # longer plan-inscribing-rectangle width
                roofSlope=None,
                lScl=1.0,   # length scale
                valuesAreScaled=True, # weather or not the dimensions are scaled
                faces: List[face]=[], 
                tapNos: List[int]=[],
                faces_file_basic=None, 
                faces_file_derived=None,
                ):
        super().__init__(memberFaces=faces, tapNos=tapNos, file_basic=faces_file_basic, file_derived=faces_file_derived)
        self.name = name
        self.H = H
        self.He = He
        self.Hr = Hr
        self.B = B
        self.D = D
        self.roofSlope = roofSlope
        self.lScl = lScl
        self.valuesAreScaled = valuesAreScaled
    
    def __str__(self):
        return 'Building name: '+self.name+'\n'+super().__str__()
    
    """--------------------------------- Properties -----------------------------------"""
    @property
    def faces(self) -> List[face]:
        return self._memberFaces
    @faces.setter
    def faces(self,value: List[face]) -> None:
        self._memberFaces = value

    @property
    def boundingBox(self):
        if len(self._memberFaces) == 0:
            return None
        # loop over all faces and get the vertices3D and then get the min and max of x, y, z
        x = []
        y = []
        z = []
        for fc in self._memberFaces:
            verts = fc.vertices3D
            x.extend(verts[:,0])
            y.extend(verts[:,1])
            z.extend(verts[:,2])
        return np.array([[min(x), max(x)], [min(y), max(y)], [min(z), max(z)]])

    """-------------------------------- Data handlers ---------------------------------"""
    def writeToFile(self, file_basic, file_derived=None) -> None:
        # write the basic building details to the file 
        super().writeToFile(file_basic, file_derived)
        # finally add derived things if any
        pass

    def copy(self) -> 'building':
        return copy.deepcopy(self)

    """--------------------------------- Plotters -------------------------------------"""
    def plotBldg_3D(self, ax=None, figsize=(10,10), showOrigin=False, 
                    showAxis=True, axisArrowSize=1.0, axisVisibility:Literal['on','off']='off',
                    showTaps=False, showTapNo=False, showTapName=False, textOffset_tapNo=[0,0], textOffset_tapName=[0,0],
                    kwargs_taps={'color':'k', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                    kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                    kwargs_view={'elev':30, 'azim':30},
                    ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
            
        if showOrigin:
            # draw the origin
            ax.scatter(0, 0, 0, color='k', marker='o', s=5)
            
        # draw the axes with arrows
        if showAxis:
            ax.quiver(0, 0, 0, axisArrowSize, 0, 0, color='r', arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, 0, axisArrowSize, 0, color='g', arrow_length_ratio=0.1)
            ax.quiver(0, 0, 0, 0, 0, axisArrowSize, color='b', arrow_length_ratio=0.1)
            
            
        for fc in self._memberFaces:
            fc.plotEdges_3D(ax=ax, showName=False)
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # rotate the view
        ax.view_init(**kwargs_view)
                
        bounds = self.boundingBox  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        
        # set the axis to equal
        ax.axis('equal')
        ax.axis(axisVisibility)

        
        
        # set the x, y, z limits to proportionally fit the building
        
        
        # if xlim is not None:
        #     ax.set_xlim(xlim)
        # if ylim is not None:
        #     ax.set_ylim(ylim)
        # if zlim is not None:
        #     ax.set_zlim(zlim)

#---------------------------- BUILDING STRUCTURE ---------------------------------#
class node_CAD:
    # a node of a 2D or 3D frame
    def __init__(self, x: float, y: float=None, z: float=None, ID: int=None, name=None, 
                 fixedDOF: List[bool]=[False, False, False, False, False, False],
                 DOF_indices: List[int]=[None, None, None, None, None, None],
                 nodeType: Literal['support','joint','internal']='joint',
                 connectedElements: List['element_CAD']=[],
                 connectedPanels: List['panel_CAD']=[],
                 parentFrame: 'frame2D_CAD'=None,
                ) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.ID: int = ID
        self.name: str = name
        # self.connectionType: Literal['fixed','pinned','roller','free'] = connectionType
        self.fixedDOF: List[bool] = fixedDOF
        self.DOFindices: List[int] = DOF_indices
        self.nodeType: Literal['support','joint','internal'] = nodeType
        self.connectedElements: List[element_CAD] = connectedElements
        self.connectedPanels: List[panel_CAD] = connectedPanels
        self.parentFrame: frame2D_CAD = parentFrame
        
    @property
    def is1D(self) -> bool:
        return self.z is None and self.y is None
    
    @property
    def is2D(self) -> bool:
        return self.z is None and self.y is not None
    
    @property
    def is3D(self) -> bool:
        return self.z is not None and self.y is not None
    
    @property
    def loc(self) -> np.ndarray:
        if self.is1D:
            return np.array([self.x], dtype=float)
        elif self.is2D:
            return np.array([self.x, self.y], dtype=float)
        elif self.is3D:
            return np.array([self.x, self.y, self.z], dtype=float)
        else:
            return None

    @property
    def loc_2D(self) -> np.ndarray:
        if self.parentFrame is None:
            return None
        # transform the 3D location to 2D location within the plane defined by origin and basis vectors of the parentFrame_2D
        loc = self.loc
        # if loc is 1D or 2D append a zero to make it 3D
        if len(loc) == 1: # add two zeros
            loc = np.append(loc, [0, 0])
        elif len(loc) == 2:
            loc = np.append(loc, 0)
            
        # get the projection of the location on the plane
        return transform(loc, self.parentFrame.origin, self.parentFrame.basisVectors)
    
    @property
    def idxs_inConnectedElements(self) -> List[int]:
        return [e.nodes.index(self) for e in self.connectedElements]
    
    @property
    def idxs_inConnectedPanels(self) -> List[int]:
        return [p.supportNodes.index(self) for p in self.connectedPanels]
    
    '''@property
    def degreesOfFreedom(self) -> List[bool]:
        if self.connectionType == 'fixed':
            return [False, False, False, False, False, False]
        elif self.connectionType == 'pinned':
            return [True, True, True, False, False, False]
        elif self.connectionType == 'roller':
            return [True, True, False, False, False, False]
        elif self.connectionType == 'free':
            return [True, True, True, True, True, True]
        else:
            raise Exception('Invalid connectionType. Valid options are: fixed, pinned, roller, free.')'''
    
    def __str__(self):
        # ID, name, loc, connectionType, nodeType, parent frame, connectedElements, connectedPanels
        msg = 'Node ID:\t\t'+str(self.ID)+'\nname:\t\t\t'+self.name+'\nLocation:\t\t'+str(self.loc)+'\nDoF fixed:\t'+self.fixedDOF+'\nNode type:\t\t'+self.nodeType
        msg += '\nParent frame:\t\t'+self.parentFrame.name+' ['+str(self.parentFrame.ID)+']'
        msg += '\nConnected elements:\t'+str([e.ID for e in self.connectedElements])
        msg += '\nConnected panels:\t'+str([p.ID for p in self.connectedPanels])
        return msg
    
    def __repr__(self):
        return 'Node at: '+str(self.loc)
    
    def __eq__(self, other: 'node_CAD') -> bool:
        return np.all(self.loc == other.loc)
    
    def __ne__(self, other: 'node_CAD') -> bool:
        return not np.all(self.loc == other.loc)
    
    def __add__(self, other: 'node_CAD') -> 'node_CAD':
        return node_CAD(self.loc + other.loc)
    
    def __sub__(self, other: 'node_CAD') -> 'node_CAD':
        return node_CAD(self.loc - other.loc)
    
    def __hash__(self):
        return hash(tuple(self.loc))
    
    def copy(self) -> 'node_CAD':
        return copy.deepcopy(self)

    ###---------------------- Data handlers ----------------------###
    
    def toLocalCoord(self, face: 'face') -> np.ndarray:
        return face.toLocalCoord(points_inGlobalCoords=self.loc)
    
    def toGlobalCoord(self, face: 'face') -> np.ndarray:
        return face.toGlobalCoord(points_inLocalCoords=self.loc)
    
    ###---------------------- Plotters ----------------------###
    
    def plot(self, ax=None, showName=False, textOffset=[0,0],
                kwargs_node={'color':'k', 'marker':'o', 'ms':5, 'ls':'None'},
                kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
            newFig = False
            if ax is None:
                newFig = True
                fig = plt.figure()
                ax = fig.add_subplot()
            loc = self.loc_2D
            ax.plot(loc[0], loc[1], **kwargs_node)
            if showName:
                ax.text(self.loc[0]+textOffset[0], self.loc[1]+textOffset[1], str(self.ID), **kwargs_text)
            if newFig:
                ax.axis('equal')
                ax.axis('off')
                
    def plot3D(self, ax=None, showName=False, textOffset=[0,0,0],
                kwargs_node={'color':'k', 'marker':'o', 'ms':5, 'ls':'None'},
                kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
            newFig = False
            if ax is None:
                newFig = True
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
            ax.scatter(self.loc[0], self.loc[1], self.loc[2], **kwargs_node)
            if showName:
                ax.text(self.loc[0]+textOffset[0], self.loc[1]+textOffset[1], self.loc[2]+textOffset[2], str(self.ID), **kwargs_text)
            if newFig:
                ax.axis('equal')
                ax.axis('off')
    
class panel_CAD: 
    def __init__(self,
                parentFace: face,
                vertices_global: np.ndarray = None,
                # center: np.ndarray = None,
                ID: int = None,
                name: str = None,
                supportNodes: List[node_CAD] = [],
                ) -> None:
        self.parentFace: face = parentFace
        self.vertices_global: np.ndarray = vertices_global
        # self.center: np.ndarray = center
        self.ID: int = ID
        self.name: str = name
        self.supportNodes: List[node_CAD] = supportNodes
    
    @property
    def area(self) -> float:
        return self.polygon.area
    
    @property
    def normal(self) -> np.ndarray:
        return self.parentFace.faceNormal
    
    @property
    def centroid_local(self) -> np.ndarray:
        return np.mean(self.vertices_local, axis=0)
    
    @property
    def centroid_global(self) -> np.ndarray:
        return self.parentFace.toGlobalCoord(points_inLocalCoords=[self.centroid_local])
    
    @property
    def vertices_local(self) -> np.ndarray:
        crnrs = self.parentFace.toLocalCoord(self.vertices_global,debug=False)
        # crnrs = transform(self.vertices_global, self.parentFace.origin, self.parentFace.basisVectors, inverseTransform=True, debug=False)
        
        # if the z values are not zero, set them to zero and warn the user
        if not np.allclose(crnrs[:,2], 0):
            warnings.warn(f"Panel {self.ID} has non-zero local z values. Setting them to zero.")
            crnrs[:,2] = 0
        return crnrs
    
    @property
    def vertices_plt(self) -> np.ndarray:
        return transform(self.vertices_local, self.parentFace.origin_plt, self.parentFace.basisVectors_plt)
    
    @property
    def polygon(self) -> shp.Polygon:
        return shp.Polygon(self.vertices_local)
        
    @property
    def supportNode_IDs(self) -> List[int]:
        return [n.ID for n in self.supportNodes]
        
    @property
    def supportLocations_global(self) -> np.ndarray:
        return np.array([n.loc for n in self.supportNodes])
    
    @property
    def supportLocations_local(self) -> np.ndarray:
        ans = self.parentFace.toLocalCoord(self.supportLocations_global, debug=False)
        # if the z values are not zero, set them to zero and warn the user
        if not np.allclose(ans[:,2], 0):
            warnings.warn(f"Panel {self.ID} has support nodes with non-zero local z values. Setting them to zero.")
            ans[:,2] = 0
        return ans
        
    @property
    def supportTributaries(self) -> shp.MultiPolygon:
        '''Generate the tributaries for the supportNodes of the panel using voronoi tessellation.
        Returns:
            shp.MultiPolygon: A multipolygon object containing the tributaries.
        '''
        # trimmedVoronoi(self.vertices, self.tapCoord, showLog=showDetailedLog)
        return trimmedVoronoi(bound=self.vertices_local, query_points=self.supportLocations_local, query_points_can_be_on_boundary=True, showLog=False, showPlot=False)

    @property
    def supportAreaShares(self) -> np.ndarray:
        '''Get the area shares of the supportNodes of the panel.
        Returns:
            np.ndarray: The area shares of the supportNodes.
        '''
        polygon = self.polygon
        supportAreas = []
        for i,tri in enumerate(self.supportTributaries.geoms):
            out = tri.intersection(polygon)
            if out.is_empty:
                supportAreas.append(0.0)
                warnings.warn(f"Support node {self.supportNodes[i]} of panel {self.ID} does not fall within the panel area.")
            else:
                supportAreas.append(out.area)
        return np.array(supportAreas, dtype=float)

    @property
    def supportAreaShares_normalized(self) -> np.ndarray:
        '''Get the normalized area shares of the supportNodes of the panel.
        Returns:
            np.ndarray: The normalized area shares of the supportNodes.
        '''
        return self.supportAreaShares/self.area

    @property
    def edges(self) -> List[List[int]]:
        # list of list of node IDs forming the edges of the panel (also include the last edge that connects the last node to the first node)
        return [[self.supportNodes[i].ID, self.supportNodes[i+1].ID] for i in range(len(self.supportNodes)-1)] + [[self.supportNodes[-1].ID, self.supportNodes[0].ID]]

    ###---------------------- Data handlers ----------------------###
    def __str__(self) -> str:
        # ID, parentFace, supportNodes[.ID]
        fcStr = self.parentFace.name+'['+str(self.parentFace.ID)+']' if self.parentFace is not None else 'None'
        supportNodesStr = [n.name+'['+str(n.ID)+']' for n in self.supportNodes] 
        return 'Panel ID:\t'+str(self.ID)+'\nParent face:\t'+fcStr+'\nSupport nodes:\t'+str(supportNodesStr)
    
    def getInvolvedTaps(self, debug=False, showPlots=False) -> np.ndarray:
        '''Get the tap tributaries that fall within the panel area.
        Returns:
            np.ndarray: The indices of the tap tributaries that fall within the panel area.
        '''
        # create a shaplley polygon object from the vertices of the panel
        panelPoly = self.polygon
        if showPlots:
            fig, ax = plt.subplots()
            fig.set_size_inches(12,12)
            ax.plot(*panelPoly.exterior.xy, color='k', label=f"Panel boundary")
            ax.set_title(f"Taps overlapping with panel {self.ID}")
            ax.plot(self.parentFace.vertices[:,0], self.parentFace.vertices[:,1], color='b', lw=2.0, label=f"Parent face boundary")
            ax.axhline(0, color='k', lw=0.5)
            ax.axvline(0, color='k', lw=0.5)
            
        # loop through the tap tributaries of the parent face and check if they fall (overlap) within the panel's area
        tapIdxs_inFace = []
        overlappingAreas = []
        cnt = 0
        for i,trib in enumerate(self.parentFace.tapTribs.geoms):
            out = panelPoly.intersection(trib)
            if out.is_empty:
                continue
            # only consider the overlapping area if it is a polygon, a multipolygon, or a collection containing polygons. Otherwise, ignore it.
            if isinstance(out, shp.Polygon):
                tapIdxs_inFace.append(i)
                overlappingAreas.append(out.area)
            elif isinstance(out, shp.MultiPolygon):
                counted = False
                area = 0.0
                for poly in out:
                    if isinstance(poly, shp.Polygon):
                        if not counted:
                            tapIdxs_inFace.append(i)
                            counted = True
                        area += poly.area
                if area > 0.0:
                    overlappingAreas.append(area)
            elif isinstance(out, shp.GeometryCollection):
                counted = False
                area = 0.0
                for geom in out:
                    if isinstance(geom, shp.Polygon):
                        if not counted:
                            tapIdxs_inFace.append(i)
                            counted = True
                        area += geom.area
                if area > 0.0:
                    overlappingAreas.append(area)
            if debug:
                print(f"Tap {self.parentFace.tapNo[i]} overlaps with the panel area by {out.area}")
            if showPlots:
                ax.plot(*trib.exterior.xy, color=def_cols[cnt], lw=1.5, label=f"Tap {self.parentFace.tapNo[i]}")
                ax.fill(*out.exterior.xy, color=def_cols[cnt], alpha=0.3, label=f"Overlap with tap {self.parentFace.tapNo[i]}")
                # plot the tap points as well
                tapLoc = self.parentFace.tapCoord[i]
                ax.plot(tapLoc[0], tapLoc[1], 'o', color=def_cols[cnt],)
                
            cnt += 1
        
        if showPlots:
            ax.legend()
            ax.axis('equal')
            plt.show()
        
        tapIdxs_inFace = np.array(tapIdxs_inFace, dtype=int)
        overlappingAreas = np.array(overlappingAreas, dtype=float)
        if np.abs(self.area - np.sum(overlappingAreas)) > 1e-6:
            warnings.warn(f"Panel {self.ID} area ({self.area}) and the sum of overlapping tap areas ({np.sum(overlappingAreas)}) do not match within a tolerance of 1e-6.")
            
        return tapIdxs_inFace, overlappingAreas

    ###------------------------ Plotters ------------------------###
    def plot(self, ax=None, showName=False, textOffset=[0,0],
                kwargs_node={'color':'k', 'marker':'o', 'ms':5, 'ls':'None'},
                kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
            newFig = False
            if ax is None:
                newFig = True
                fig = plt.figure()
                ax = fig.add_subplot()
            verts = self.vertices_plt
            ax.plot(verts[:,0], verts[:,1], **kwargs_node)
            if showName:
                ax.text(self.centroid_global[0]+textOffset[0], self.centroid_global[1]+textOffset[1], str(self.ID), **kwargs_text)
            if newFig:
                ax.axis('equal')
                ax.axis('off')

    def plot_3D(self, ax=None, showName=False, textOffset=[0,0,0],
                kwargs_node={'color':'k', 'marker':'o', 'ms':5, 'ls':'None'},
                kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
            newFig = False
            if ax is None:
                newFig = True
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
            verts = self.vertices_global
            ax.plot(verts[:,0], verts[:,1], verts[:,2], **kwargs_node)
            if showName:
                ax.text(self.centroid_global[0]+textOffset[0], self.centroid_global[1]+textOffset[1], self.centroid_global[2]+textOffset[2], str(self.ID), **kwargs_text)
            if newFig:
                ax.axis('equal')
                ax.axis('off')

class element_CAD:
    # a structural element that forms a 2D or 3D frame (e.g. beam, column, etc.)
    def __init__(self, startNode: node_CAD, endNode: node_CAD, internalNodes: List[node_CAD]=[],
                 ID: int=None, name=None,
                 parentFrame: 'frame2D_CAD'=None,
                 ) -> None:
        self.startNode: node_CAD = startNode
        self.endNode: node_CAD = endNode
        self.internalNodes: List[node_CAD] = internalNodes
        self.ID: int = ID
        self.name = name
        self.parentFrame: frame2D_CAD = parentFrame
    
    @property
    def is2D(self) -> bool:
        return self.startNode.is2D and self.endNode.is2D
    
    @property
    def is3D(self) -> bool:
        return self.startNode.is3D and self.endNode.is3D
    
    @property
    def length(self) -> float:
        return linalg.norm(self.endNode.loc - self.startNode.loc)

    @property
    def x(self) -> np.ndarray:
        return np.array([self.startNode.loc[0], self.endNode.loc[0]], dtype=float)
    
    @property
    def y(self) -> np.ndarray:
        return np.array([self.startNode.loc[1], self.endNode.loc[1]], dtype=float)
    
    @property
    def z(self) -> np.ndarray:
        if self.is2D:
            return np.array([0, 0], dtype=float)
        else:
            return np.array([self.startNode.loc[2], self.endNode.loc[2]], dtype=float)
    
    @property
    def loc(self) -> np.ndarray:
        return np.array([self.startNode.loc, self.endNode.loc], dtype=float)
    
    @property
    def loc_2D(self) -> np.ndarray:
        if self.parentFrame is None:
            return None
        loc = self.loc
        if self.is2D:
            loc = np.hstack(loc, np.zeros((2,1)))
        return transform(loc, self.parentFrame.origin, self.parentFrame.basisVectors)
    
    @property
    def orientation(self) -> float:
        if self.is2D:
            return np.arctan2(self.endNode.loc[1]-self.startNode.loc[1], self.endNode.loc[0]-self.startNode.loc[0])
        elif self.is3D:
            # two angles: one in the xy-plane and the other the vertical angle between the xy-plane and the line
            xy_angle = np.arctan2(self.endNode.loc[1]-self.startNode.loc[1], self.endNode.loc[0]-self.startNode.loc[0])
            xyz_angle = np.arctan2(self.endNode.loc[2]-self.startNode.loc[2], self.length)
            return {'xy-plane [rad]':xy_angle, 'xy-z--plane [rad]':xyz_angle}
        else:
            return None
        
    @property
    def orientation_deg(self) -> float:
        return np.rad2deg(self.orientation)
    
    @property
    def orientation_unitVector(self) -> np.ndarray:
        if self.is2D:
            return (self.endNode.loc - self.startNode.loc)/self.length
        elif self.is3D:
            return (self.endNode.loc - self.startNode.loc)/linalg.norm(self.endNode.loc - self.startNode.loc)
        else:
            return None
    
    @property
    def centroid(self) -> np.ndarray:
        return 0.5*(self.nodes[0].loc + self.nodes[1].loc)

    '''@property
    def panelCentroids_fromStartNode(self) -> np.ndarray:
        start = self.startNode.loc
        # if the start point is a 2D element, add a zero z-coordinate to the start node
        if len(start) == 2:
            start = np.append(start, 0)
        C = []
        for p in self.panels:
            C.append(p.centroid_global - start)
        return np.array(C, dtype=float)

    @property
    def panelMomentArm(self) -> np.ndarray:
        start = self.startNode.loc
        # if the start point is a 2D element, add a zero z-coordinate to the start node
        if len(start) == 2:
            start = np.append(start, 0)
        orientation = self.orientation_unitVector
        L = []
        for p in self.panels:
            global_arm = linalg.norm(p.centroid_global - start)
            # project the global arm onto the line of the element
            local_arm = np.dot(p.centroid_global - start, orientation)
            L.append(local_arm)
            
        return np.array(L, dtype=float)
    
    @property
    def panelMomentArm_global(self) -> np.ndarray:
        start = self.startNode.loc
        # if the start point is a 2D element, add a zero z-coordinate to the start node
        if len(start) == 2:
            start = np.append(start, 0)
        L = []
        for p in self.panels:
            L.append(linalg.norm(p.centroid_global - start))
        return np.array(L, dtype=float)'''
    
    @property
    def nodes(self) -> List[node_CAD]:
        return [self.startNode, *self.internalNodes, self.endNode]
    
    @property
    def segments(self) -> List[List[int]]:
        # list of list of node IDs forming the segments of the element
        return [[self.nodes[i].ID, self.nodes[i+1].ID] for i in range(len(self.nodes)-1)]
    
    @property
    def connectedPanels(self) -> List[panel_CAD]:
        debug = False
        panels = []
        for n in self.nodes:
            for p in n.connectedPanels:
                if p not in panels:
                    panels.append(p)
        # remove the panels that do not have the element as a boundary
        connectedPnls = []
        for seg in self.segments:
            if debug:
                print(f"Segment: {seg}")
            # a segment may be connected to multiple panels, 
            for p in panels:
                isBoundary = False
                # go through the panel.edges and check if any of them is the same as the segment (check comutationally, i.e. [1,2] is the same as [2,1])
                for edge in p.edges:
                    if debug:
                        print(f"\tEdge: {edge}")
                    if (seg[0] == edge[0] and seg[1] == edge[1]) or (seg[0] == edge[1] and seg[1] == edge[0]):
                        if debug:
                            print(f"\t\tPanel {p.ID} is connected to segment {seg}******************")
                        isBoundary = True
                        break
                    else:
                        if debug:
                            print(f"\t\tPanel {p.ID} is not connected to segment {seg}")
                if isBoundary and p not in connectedPnls:
                    connectedPnls.append(p)
                    # do not break here, as the segment may be connected to multiple panels
        return connectedPnls
                    
        
    
    
    ###---------------------- Data handlers ----------------------###
    def copy(self) -> 'element_CAD':
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        # ID, name, startNode.ID, endNode.ID, length, orientation, connectedPanels, parentFrame
        msg = 'Element ID:\t\t'+str(self.ID)+'\n'
        msg += 'name:\t\t\t'+self.name+'\n'
        msg += 'Start node:\t\t'+str(self.startNode.ID)+'\n'
        msg += 'End node:\t\t'+str(self.endNode.ID)+'\n'
        msg += 'Length:\t\t\t'+str(self.length)+'\n'
        msg += 'Orientation:\t\t'+str(self.orientation)+'\n'
        msg += 'Connected panels:\t'+str([p.ID for p in self.connectedPanels])+'\n'
        msg += 'Parent frame:\t\t'+self.parentFrame.name+' ['+str(self.parentFrame.ID)+']'
        return msg
    
    ###---------------------- Plotters ----------------------###
    
    def plot(self, ax=None, showName=False, textOffset=[0,0],
                kwargs_line={'color':'b', 'lw':3, 'ls':'-'},
                kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
            newFig = False
            if ax is None:
                newFig = True
                fig = plt.figure()
                ax = fig.add_subplot()
            loc = self.loc_2D
            ax.plot(loc[:,0], loc[:,1], **kwargs_line)
            if showName:
                ax.text(self.centroid[0]+textOffset[0], self.centroid[1]+textOffset[1], str(self.ID), **kwargs_text)
            if newFig:
                ax.axis('equal')
                ax.axis('off')

class frame2D_CAD:
    
    # a 2D frame that forms the structural skeleton of a building
    def __init__(self, parentBuilding: building,
                 ID: int=None, name=None,
                 nodes: List[node_CAD]=[],
                 elements: List[element_CAD]=[],
                 ) -> None:
        self.NODE_PLANE_TOLERANCE = 1e-3
        
        self.ID = ID
        self.name = name
        self.parentBuilding: building = parentBuilding
        self._nodes: List[node_CAD] = nodes
        self.elements : List[element_CAD] = elements
        
    
    def __redefineDOF_indices(self):
        # assign the degrees of freedom indices to the nodes
        # get the number of nodes
        n = len(self.nodes)
        # get the number of degrees of freedom (both fixed and free)
        N = 6*n
        iDOF = 0
        for i,n in enumerate(self.nodes):
            pass
        
    def __redefineBoundaryConditions(self):
        # assign the boundary conditions to the nodes
        pass
        

    @property
    def nodes(self) -> List[node_CAD]:
        return self._nodes
    @nodes.setter
    def nodes(self, nodes: List[node_CAD]):
        self._nodes = nodes
        self.__redefineDOF_indices()
        self.__redefineBoundaryConditions()
    
    @property
    def supportNodes(self) -> List[node_CAD]:
        return [n for n in self.nodes if n.nodeType == 'support']
    
    @property
    def internalNodes(self) -> List[node_CAD]:
        return [n for n in self.nodes if n.nodeType == 'internal']
    
    @property
    def joints(self) -> List[node_CAD]:
        return [n for n in self.nodes if n.nodeType == 'joint']
    
    @property
    def nodeIDs(self) -> List[int]:
        return [n.ID for n in self.nodes]
    
    @property
    def elementIDs(self) -> List[int]:
        return [el.ID for el in self.elements]
    
    @property
    def panels(self) -> List[panel_CAD]:
        panels = []
        for n in self.nodes:
            for p in n.connectedPanels:
                if p not in panels:
                    panels.append(p)
                    
        return panels
    
    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.nodePoints, axis=0)
    
    @property
    def nodeIDs(self) -> List[int]:
        return [n.ID for n in self.nodes]
    
    @property
    def elementIDs(self) -> List[int]:
        return [el.ID for el in self.elements]
    
    @property
    def planeNormal(self) -> np.ndarray:
        # returns a normal vector to the plane of the frame
        if len(self.elements) == 0:
            return None
        
        # get the optimal plane (least squares) that fits all the nodes of the frame
        # get the node points
        nodePoints = self.nodePoints
        # get the covariance matrix of the node points (it may not be centered at the origin)
        cov = np.cov(nodePoints, rowvar=False)
        # get the eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = linalg.eig(cov)
        # get the eigenvector corresponding to the smallest eigenvalue
        idx = np.argmin(eigvals)
        planeNormal = eigvecs[:,idx]
        
        # if the deviation of the points from the plane is too high, then the plane is not optimal
        # check the deviation of the points from the plane
        deviations = np.abs(np.dot(nodePoints, planeNormal))
        # subtract the mean deviation
        deviations -= np.mean(deviations)        
        if np.max(deviations) > self.NODE_PLANE_TOLERANCE:
            # print a warning with info about the deviation and the node(s) that cause it
            print(f"Warning: The deviation of the nodes from the plane is too high. The maximum deviation is {np.max(deviations)}.")
            print(f"The node(s) that cause the deviation are:")
            for i,dev in enumerate(deviations):
                if dev > self.NODE_PLANE_TOLERANCE:
                    print(f"    Node ID: {self.nodes[i].ID}, Deviation: {dev}")
                    
        return planeNormal
        
    @property
    def origin(self) -> np.ndarray:
        # use the projection of the vector to the centroid onto the plane normal as the origin
        centroid = self.centroid
        norm = self.planeNormal
        return np.dot(centroid, norm)*norm
        
    @property
    def basisVectors(self) -> np.ndarray:
        # get the local basis vectors of the frame
        # get the plane normal
        norm = self.planeNormal
        # get the first basis vector as the unit vector in the direction of the first element
        e1 = self.elements[0].orientation_unitVector
        # get the second basis vector as the cross product of the plane normal and the first basis vector
        e2 = np.cross(norm, e1)
        return np.array([e1, e2, norm])
        
    @property
    def nodePoints(self) -> np.ndarray:
        return np.array([n.loc for n in self.nodes])
        
    def __str__(self) -> str:
        # ID, name, parentBuilding.name, nodes.ID, elements.ID, if(panels.ID)
        msg = 'Frame ID:\t'+str(self.ID)+'\nname:\t\t'+self.name+'\nBuilding:\t'+self.parentBuilding.name+'\nNodes:\t\t'+str(self.nodeIDs)+'\nElements:\t'+str(self.elementIDs)
        if len(self.panels) > 0:
            msg += '\nPanels:\t\t'+str([p.ID for p in self.panels])
        return msg
