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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from shapely.ops import voronoi_diagram
from shapely.validation import make_valid
from typing import List, Tuple, Dict
from scipy.interpolate import griddata

#===============================================================================
#==================== CONSTANTS & GLOBAL VARIABLES  ============================
#===============================================================================


#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

#--------------------- TAPS, PANELS, & AREA AVERAGING --------------------------
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

def trimmedVoronoi(bound,coords,showLog=False):
    if showLog:
        print(f"Generating tributaries ...")
        print(f"Shape of bound: {np.shape(bound)}")
        print(f"Shape of coords: {np.shape(coords)}")
    
    inftyTribs = voronoi_diagram(shp.MultiPoint(coords))
    if showLog:
        print(f"Shape of inftyTribs: {np.shape(inftyTribs.geoms)}")
    
    boundPolygon = shp.Polygon(bound)
    tribs = []
    for g in inftyTribs.geoms:
        newRig = getIntersection(boundPolygon,g)
        if newRig is not None:
            tribs.append(newRig)
        else:
            continue
    idx = []
    areas = []
    for ic in range(np.shape(coords)[0]):
        for it,trib in enumerate(tribs):
            if shp.Point(coords[ic,:]).within(trib):
                idx.append(it)
                areas.append(trib.area)
                break
    idx = np.asarray(idx,dtype=int)
    areas = np.asarray(areas,dtype=float)
    totalArea = np.sum(areas)
    boundArea = boundPolygon.area
    if abs(totalArea - boundArea) > 0.00001*boundArea:
        warnings.warn(f"The sum of tributary areas {totalArea} is not equal to the area of the bound {boundArea}.")

    temp = []
    for i in idx:
        temp.append(tribs[i])
        if not type(tribs[i]) == shp.Polygon:
            warnings.warn(f"Type of tributary {i} is {type(tribs[i])}. There is a risk of voronoi not completely tiling the bound.")
    tribs = shp.GeometryCollection(temp)

    # tribs = shp.MultiPolygon([tribs[i] for i in idx])

    if showLog:
        print(f"Shape of tribs: {np.shape(tribs.geoms)}")
        plt.figure(figsize=[15,10])
        x,y = boundPolygon.exterior.xy
        plt.plot(x,y,'-b',lw=3)
        plt.plot(coords[:,0],coords[:,1],'.r')
        for g in tribs.geoms:
            x,y = g.exterior.xy
            plt.plot(x,y,'-r',lw=1)
        plt.axis('equal')
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
def transform(geomIn,orig,T):
    N,M = np.shape(geomIn)
    if len(orig) == 3 and M == 2:
        geomIn = np.append(geomIn,np.zeros((N,1)),axis=1)
    geomOut =np.transpose(np.dot(T,np.transpose(np.add(geomIn,orig))))
    return geomOut


#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

#----------------------------- BUILDING MODELS ---------------------------------
class face:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                ID=None,
                name=None,
                note=None,
                origin=None,
                basisVectors=None,
                origin_plt=None,
                basisVectors_plt=None,
                vertices=None,
                tapNo: List[int]=None,
                tapIdx: List[int]=None,
                tapName: List[str]=None,
                badTaps=None,
                allBldgTaps=None,
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
        self.note = note
        self.origin = origin                # origin of the face-local coordinate system
        self.basisVectors = basisVectors    # [[3,], [3,], [3,]] basis vectors of the local coord sys in the main 3D coord sys.
        self.origin_plt = origin_plt                # 2D origin of the face-local coordinate system for plotting
        self.basisVectors_plt = basisVectors_plt    # [[2,], [2,]] basis vectors of the local coord sys in 2D for plotting.
        self.vertices = np.array(vertices,dtype=float)            # [Nv, 2] face corners that form a non-self-intersecting polygon. No need to close it with the last edge.
        self.zoneDict = zoneDict            # dict of dicts: zone names per each zoning. e.g., [['NBCC-z1', 'NBCC-z2', ... 'NBCC-zM'], ['ASCE-a1', 'ASCE-a2', ... 'ASCE-aQ']]
        self.nominalPanelAreas = nominalPanelAreas  # a vector of nominal areas to generate panels. The panels areas will be close but not necessarily exactly equal to these.
        self.numOfNominalPanelAreas = numOfNominalPanelAreas

        # Tap things
        self.tapNo: List[int] = tapNo                  # [Ntaps,]
        self.tapNo_all = self.tapNo                    # includes the bad taps
        self.tapIdx: List[int] = tapIdx                # [Ntaps,] tap indices in the main matrix of the entire building
        self.tapName: List[str] = tapName              # [Ntaps,]
        self.badTaps: List[int] = badTaps
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
        self.__handleBadTaps(allBldgTaps)
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
        if self.zoneDict is None and self.nominalPanelAreas is None:
            return
        if self.zoneDict == {}:
            self.zoneDict = {
                                0:['Default','Default', np.array(self.vertices)],
                        }
        if self.nominalPanelAreas is None and self.zoneDict is not None:
            bound = shp.Polygon(self.vertices)
            maxArea = bound.area
            minArea = maxArea
            for trib in self.tapTribs.geoms:
                minArea = min((minArea, trib.area))

            self.nominalPanelAreas = np.logspace(np.log10(minArea), np.log10(maxArea), self.numOfNominalPanelAreas)

    def __generatePanels(self, showDetailedLog=False):
        self.__defaultZoneAndPanelConfig()
        print(f"Generating panels ...")
        # print(f"Shape of zones: {np.shape(self.zones)}")
        if self.tapTribs is None or self.zoneDict is None:
            return
        if showDetailedLog:
            print(f"Shape of tapTribs: {np.shape(self.tapTribs)}")
            print(f"Shape of nominalPanelAreas: {np.shape(self.nominalPanelAreas)}")

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
            for a,area in enumerate(self.nominalPanelAreas):
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
    def vertices3D(self):
        return transform(self.vertices, self.origin, self.basisVectors)

    @property
    def verticesPlt(self):
        return transform(self.vertices, self.origin_plt, self.basisVectors_plt)

    @property
    def NumTaps(self):
        num = 0 if self.tapNo is None else len(self.tapNo)
        return num

    @property
    def NumPanels(self):
        if self.nominalPanelAreas is None:
            return 0
        num = np.int32(np.multiply(self.nominalPanelAreas,0))
        if self.panels is None:
            return num
        for a,_ in enumerate(self.nominalPanelAreas):
            for z,_ in enumerate(self.panels):
                num[a] += len(self.panels[z][a].geoms)
        return np.sum(num)

    @property
    def NumPanelsPerArea(self):
        if self.nominalPanelAreas is None:
            return 0
        num = np.int32(np.multiply(self.nominalPanelAreas,0))
        if self.panels is None:
            return num
        for a,_ in enumerate(self.nominalPanelAreas):
            for z,_ in enumerate(self.panels):
                num[a] += len(self.panels[z][a].geoms)
        return num

    @property
    def panelAreas__redacted(self):
        return None
        if self.nominalPanelAreas is None or self.panels is None:
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
            errDict[zn_ID] = {f"nom. areas idxs with tiling errors {self.nominalPanelAreas}": self.error_in_zones[z]}
            errDict[zn_ID]['tap idxs with weight errors'] = {}
            for a,area in enumerate(self.nominalPanelAreas):
                errDict[zn_ID]['tap idxs with weight errors'][f"A={area}"] = self.error_in_panels[z][a]
        return errDict

    """-------------------------------- Data handlers ---------------------------------"""
    def Refresh(self, showDetailedLog=False):
        self.__generateTributaries(showDetailedLog)
        self.__generatePanels(showDetailedLog)

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
        basic['nominalPanelAreas'] = self.nominalPanelAreas
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
        self.nominalPanelAreas = basic['nominalPanelAreas']
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

    def getNearestTapsToLine(self, start: np.ndarray, end: np.ndarray, distTolerance: float=0.0001):
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
        tapNos = []
        tapIdxs = []
        dist_from_start = []
        for t,tp in enumerate(self.tapCoord):
            d = np.linalg.norm(np.cross(end-start, start-tp))/np.linalg.norm(end-start)
            if d <= distTolerance:
                tapNos.append(self.tapNo[t])
                tapIdxs.append(self.tapIdx[t])
                dist_from_start.append(np.linalg.norm(tp-start))
        # sort the tapNos and tapIdxs by distance from the start point
        tapNos = [x for _,x in sorted(zip(dist_from_start,tapNos))]
        tapIdxs = [x for _,x in sorted(zip(dist_from_start,tapIdxs))]
        return tapNos, tapIdxs, dist_from_start
    

    """--------------------------------- Plotters -------------------------------------"""
    def plotEdges(self, ax=None, showName=True, drawOrigin=False, drawBasisVectors=False, basisVectorLength=1.0,
                  kwargs_Edge={'color':'k', 'lw':1.0, 'ls':'-'}, 
                  kwargs_Name={'ha':'center', 'va':'center', 'color':'k', 'backgroundcolor':[1,1,1,0.3]},
                  ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.verticesPlt)
        ax.plot(xy[:,0], xy[:,1], **kwargs_Edge)
        if showName:
            ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), self.name, **kwargs_Name)
        if drawOrigin:
            # ax.plot(self.origin_plt[0], self.origin_plt[1], 'ko')
            pass
        if drawBasisVectors:
            # basis vectors of the local coord sys in 2D (not for the plotting).
            # windowSize = 0.05*np.max([np.max(xy[:,0])-np.min(xy[:,0]), np.max(xy[:,1])-np.min(xy[:,1])])
            # arrowLength = basisVectorLength*windowSize
            # ax.arrow(self.origin_plt[0], self.origin_plt[1], self.basisVectors_plt[0][0]*arrowLength, self.basisVectors_plt[0][1]*arrowLength, color='r', width=0.01*windowSize, head_width=0.15*windowSize, head_length=0.2*windowSize)
            # ax.arrow(self.origin_plt[0], self.origin_plt[1], self.basisVectors_plt[1][0]*arrowLength, self.basisVectors_plt[1][1]*arrowLength, color='g', width=0.01*windowSize, head_width=0.15*windowSize, head_length=0.2*windowSize)
            # ax.text(self.origin_plt[0]+self.basisVectors_plt[0][0]*arrowLength + 0.15*windowSize, self.origin_plt[1]+self.basisVectors_plt[0][1]*arrowLength + 0.15*windowSize, 'x', **kwargs_Name)
            # ax.text(self.origin_plt[0]+self.basisVectors_plt[1][0]*arrowLength + 0.15*windowSize, self.origin_plt[1]+self.basisVectors_plt[1][1]*arrowLength + 0.15*windowSize, 'y', **kwargs_Name)
            pass
        if newFig:
            ax.axis('equal')

    def plotTaps(self, ax=None, showTapNo=False, 
                 kwargs_dots={'color':'k', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                 kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},
                ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.tapCoordPlt)
        ax.plot(xy[:,0], xy[:,1], **kwargs_dots)
        if showTapNo:
            for t,tpNo in enumerate(self.tapNo):
                ax.text(xy[t,0], xy[t,1], str(tpNo), **kwargs_text)
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
                for a, aKey in enumerate(self.nominalPanelAreas):
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

    def plot(self, figFile=None,xLimits=None,figSize=[15,5], showAxis=False, dotSize=3,
             overlayTaps=False, overlayTribs=False, overlayPanels=False, overlayZones=False, useSubplotForDifferentNominalAreas=True, nSubPlotCols=3):

        if overlayPanels:
            if useSubplotForDifferentNominalAreas:
                plt.figure(figsize=figSize)
            for a,area in enumerate(self.nominalPanelAreas):
                if useSubplotForDifferentNominalAreas:
                    nRows = int(np.ceil((len(self.nominalPanelAreas))/nSubPlotCols))
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
                    for zDict in self.zoneDict:
                        xy = self.zoneDict[zDict][2]
                        plt.plot(xy[:,0], xy[:,1], '--k', lw=2)
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

    @property
    def tapNo_all(self) -> List[int]:
        tapNo = []
        for f in self.memberFaces:
            tapNo.extend(f.tapNo_all)
        return tapNo

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
    def NominalPanelArea(self) -> List[float]:
        nomArea = self._memberFaces[0].nominalPanelAreas
        for f in self.memberFaces:
            if f.nominalPanelAreas != nomArea:
                raise ValueError('All faces must have the same nominal panel areas.')
        return nomArea

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
                for a,_ in enumerate(fc.nominalPanelAreas):
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
                for a,_ in enumerate(fc.nominalPanelAreas):
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

    def tapIdxOf(self,tapNo) -> List[int]:
        tapNo = [tapNo,] if np.isscalar(tapNo) else tapNo
        tapNo = np.array(tapNo)
        allIdx = np.array(self.tapIdx)
        allTapNo = np.array(self.tapNo)
        if not np.shape(allIdx) == np.shape(allTapNo):
            msg = f"The tapIdx (shape: {np.shape(allIdx)}) does not match the tapNo (shape: {np.shape(allTapNo)}) in this object."
            raise Exception(msg)
        
        foundTapIdx = []
        notFoundTaps = []
        for t in tapNo:
            if t in allTapNo:
                idx = np.where(allTapNo == t)[0][0]
                foundTapIdx.append(idx)
            else:
                notFoundTaps.append(t)

        # foundTapIdx = allIdx[np.isin(allTapNo, tapNo).nonzero()[0]]
        # notFoundTaps = tapNo[np.where(~np.isin(tapNo, allTapNo))[0]]
        if len(notFoundTaps) > 0:
            msg = f"ERROR: The following taps are not found in the object: {notFoundTaps}"
            raise Exception(msg)
        if len(foundTapIdx) == 1:
            return foundTapIdx[0]
        return foundTapIdx

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

    """--------------------------------- Plotters -------------------------------------"""
    def plotEdges(self, ax=None, showName=True, 
                  kwargs_Edge={'color':'k', 'lw':1.0, 'ls':'-'},
                  ):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotEdges(ax=ax, showName=showName, kwargs_Edge=kwargs_Edge)
        if newFig:
            ax.axis('equal')

    def plotTaps(self, ax=None, showTapNo=False, 
                 kwargs_dots={'color':'k', 'lw':0.5, 'ls':'None', 'marker':'.', 'markersize':3},
                 kwargs_text={'ha':'left', 'va':'top', 'color':'k', 'backgroundcolor':[1,1,1,0.0], 'fontsize':'small', 'rotation':45},):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._memberFaces:
            fc.plotTaps(ax=ax, showTapNo=showTapNo, kwargs_dots=kwargs_dots, kwargs_text=kwargs_text)
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
                    kwargs_faceEdge={'color':'k', 'lw':1.0, 'ls':'-'},
                    kwargs_taps={},
                    kwargs_zoneEdge={'color':'k', 'lw':0.5, 'ls':'-'},
                    kwargs_zoneFill={'alpha':0.5,},
                    kwargs_pnlEdge={'color':'k', 'lw':0.3, 'ls':'-'},
                    ):
        newFig = False
        areas = self.NominalPanelArea
        if axs is None:
            newFig = True
            nRows = int(np.ceil(len(areas)/nCols))
            fig, axs = plt.subplots(nRows, nCols, figsize=figsize)
            axs = axs.flatten()

        for a,area in enumerate(areas):
            if plotZones:
                self.plotZones(ax=axs[a], showLegend=False, kwargs_Fill=kwargs_zoneFill, kwargs_Edge=kwargs_zoneEdge)
            if plotTaps:
                self.plotTaps(ax=axs[a], showTapNo=False, **kwargs_taps)
            self.plotPanels(ax=axs[a], aIdx=a, kwargs_Edge=kwargs_pnlEdge)
            if plotEdges:
                self.plotEdges(ax=axs[a], showName=False, kwargs_Edge=kwargs_faceEdge)
            area_string = areaFmt.format(areaFactor*area)
            axs[a].set_title(f"Area: {area_string} {areaUnit}")
            
        if newFig:
            for a in range(len(axs)):
                axs[a].axis('equal')
                axs[a].axis('off')
            plt.tight_layout()
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

class building(Faces):
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self, 
                name=None,
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

    """-------------------------------- Data handlers ---------------------------------"""
    def writeToFile(self, file_basic, file_derived=None) -> None:
        # write the basic building details to the file 
        super().writeToFile(file_basic, file_derived)
        # finally add derived things if any
        pass

class samplingLine:
    def __init__(self, 
                name=None,
                faces: List[face]=[],
                startEndPointPairs: List[List[float]]=[],
                labels: List[str]=[],
                dist_tolerance=1e-3,
                iterpolationMethod='nearest',
                ):
        self.name = name
        self.faces: List[face] = faces
        self.startEndPointPairs: List[List[List[float]]] = startEndPointPairs # [nFaces][2][2]
        self.labels: List[str] = labels
        self.dist_tolerance = dist_tolerance
        self.iterpolationMethod = iterpolationMethod

        self.tapNo = []
        self.tapIdx = []
        self.d_tap = []
        self.d_vertices = []

        self._identifyTaps()

    def __str__(self):
        return 'Line name: '+self.name
    
    def _identifyTaps(self):
        if len(self.faces) == 0:
            return
        self.tapNo = []
        self.tapIdx = []
        self.d_tap = []
        for f, fc in enumerate(self.faces):
            tapNos, tapIdxs, dist_from_start = fc.getNearestTapsToLine(self.startEndPointPairs[f][0], self.startEndPointPairs[f][1], distTolerance=self.dist_tolerance)
            self.tapNo.extend(tapNos)
            self.tapIdx.extend(tapIdxs)
            self.d_tap.extend(dist_from_start)
            
    
