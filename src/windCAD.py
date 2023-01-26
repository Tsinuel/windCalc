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

def meshRegionWithPanels(region,area,minAreaFactor=0.5,debug=False) -> Tuple[shp.MultiPolygon, List[float]]:
    if debug:
        print(f"Region: shape {region.shape}, {region}\n")
        print(f"NominalArea = {area}")

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
        # panels.append(newRig)
        
        if debug:
            summedArea += newRig.area
            plt.fill(x,y,'r',alpha=0.5)
            plt.plot(x,y,'-k',alpha=0.5,lw=0.5)

    if len(panels) == 0:
        panels.append(bound)

    if debug:
        print(f"Zone area = {bound.area}, summed area = {summedArea}")
        plt.axis('equal')
        plt.show()

    # print(f"Shape: {len(panels)}")
    panels = shp.MultiPolygon(panels)
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
                tapCoord=None,
                zoneDict=None,
                nominalPanelAreas=None,
                numOfNominalPanelAreas=5,
                file_basic=None,
                file_derived=None,
                ):
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
        self.tapNo: List[int] = tapNo                  # [Ntaps,]
        self.tapIdx: List[int] = tapIdx                # [Ntaps,] tap indices in the main matrix of the entire building
        self.tapName: List[str] = tapName              # [Ntaps,]
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
        if file_basic is None:
            self.Update()
        elif file_derived is None:
            self.readFromFile(file_basic=file_basic)
        else:
            self.readFromFile(file_basic=file_basic,file_derived=file_derived)
            
    def __str__(self):
        return self.name

    def __generateTributaries(self):
        if self.vertices is None or self.tapCoord is None:
            return
        self.tapTribs = trimmedVoronoi(self.vertices, self.tapCoord)

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

    def __generatePanels(self):
        self.__defaultZoneAndPanelConfig()
        print(f"Generating panels ...")
        # print(f"Shape of zones: {np.shape(self.zones)}")
        if self.tapTribs is None or self.zoneDict is None:
            return

        self.panels : shp.MultiPolygon = ()    # [nZones][nAreas][nPanels]
        self.panelAreas = ()    # [nZones][nAreas][nPanels]
        self.tapWghtPerPanel = ()    # [nZones][nAreas][nPanels][nTapsPerPanel]
        self.tapIdxPerPanel = ()   # [nZones][nAreas][nPanels][nTapsPerPanel]
        self.error_in_panels = ()
        self.error_in_zones = ()
        for z,zn in enumerate(self.zoneDict):
            print(f"\tWorking on: {self.zoneDict[zn][0]}-{self.zoneDict[zn][1]}")
            zoneBoundary = self.zoneDict[zn][2]
            panels_z = ()
            panelA_z = ()
            pnlWeights_z = ()
            tapIdxByPnl_z = ()
            err_pnl_z = ()
            err_zn_z = ()
            for a,area in enumerate(self.nominalPanelAreas):
                print(f"\t\tWorking on nominal area: {area}")
                print(f"zoneBoundary: {zoneBoundary}, area: {area}")
                pnls,areas = meshRegionWithPanels(zoneBoundary,area,debug=False)

                tapsInSubzone = []
                idxInSubzone = []
                zonePlygn = shp.Polygon(zoneBoundary)
                for i,tap in zip(self.tapIdx,self.tapTribs.geoms):
                    if intersects(tap,zonePlygn):
                        tapsInSubzone.append(tap)
                        idxInSubzone.append(i)
                wght, Idx, overlaps, errIdxs = calculateTapWeightsPerPanel(pnls,tapsInSubzone,idxInSubzone)
                
                panels_z += (pnls,)
                panelA_z += (areas,)
                pnlWeights_z += (wght,)
                tapIdxByPnl_z += (Idx,)
                err_pnl_z += (errIdxs,)

                errorInArea = 100*(zonePlygn.area - sum(areas))/zonePlygn.area
                print(f"\t\t\tArea check: \t Zone area = {zonePlygn.area}, \t sum of all panel areas = {sum(areas)}, \t Error = {errorInArea}%")
                print(f"\t\tGenerated number of panels: {len(pnls.geoms)}")
                
                if abs(errorInArea) > 1.0:
                    err_zn_z += (a,)
                    warnings.warn(f"The difference between Zone area and the sum of its panel areas exceeds the tolerance level.")
                
            self.panels += (panels_z,)
            self.panelAreas += (panelA_z,)
            self.tapWghtPerPanel += (pnlWeights_z,)
            self.tapIdxPerPanel += (tapIdxByPnl_z,)
            self.error_in_panels += (err_pnl_z,)
            self.error_in_zones += (err_zn_z,)

        print(f"Shape of 'panels': {np.shape(np.array(self.panels,dtype=object))}")
        print(f"Shape of 'panelAreas': {np.shape(np.array(self.panelAreas,dtype=object))}")
        print(f"Shape of 'pnlWeights': {np.shape(np.array(self.tapWghtPerPanel,dtype=object))}")
        print(f"Shape of 'tapIdxByPnl': {np.shape(np.array(self.tapIdxPerPanel,dtype=object))}")
        print(f"Indecies of nominalPanelArea per each zone with error of unequal area between total sum of panel areas vs. zone: \n\t\t\t{self.error_in_zones}")
        print(f"Indecies of panels within nominalPanelArea within zone with tap weights that do not sum to 1 : \n\t\t\t{self.error_in_panels}")

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
                num[a] += len(self.panels[z][a])
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

    """-------------------------------- Data handlers ---------------------------------"""
    def Update(self):
        self.__generateTributaries()
        self.__generatePanels()

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
            self.Update()

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

    """--------------------------------- Plotters -------------------------------------"""
    def plotEdges(self, ax=None, showName=True, col='k', dotSz=3, lw=2, ls='-', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.verticesPlt)
        ax.plot(xy[:,0], xy[:,1], 
                ls=ls, color=col, lw=lw, marker=mrkr, markersize=dotSz)
        if showName:
            ax.text(np.mean([min(xy[:,0]), max(xy[:,0])]), np.mean([min(xy[:,1]), max(xy[:,1])]), self.name, 
                    ha='center', va='center', color=col, backgroundcolor=[1,1,1,0.3],)
        if newFig:
            ax.axis('equal')

    def plotTaps(self, ax=None, showTapNo=False, fontsize='small', col='k', dotSz=3, lw=2, ls='None', mrkr='.'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        xy = np.array(self.tapCoordPlt)
        ax.plot(xy[:,0], xy[:,1], 
                ls=ls, color=col, lw=lw, marker=mrkr, markersize=dotSz)
        if showTapNo:
            for t,tpNo in enumerate(self.tapNo):
                ax.text(xy[t,0], xy[t,1], str(tpNo),
                        ha='left', va='top', rotation=45, fontsize=fontsize, color=col, backgroundcolor=[1,1,1,0.3],)
            pass
        if newFig:
            ax.axis('equal')
    
    def plotTribs(self, ax=None, col='r', dotSz=3, lw=0.5, ls='-.', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for trib in self.tapTribs.geoms:
            xy = transform(np.transpose(trib.exterior.xy), self.origin_plt, self.basisVectors_plt) 
            ax.plot(xy[:,0], xy[:,1], 
                    ls=ls, color=col, lw=lw, marker=mrkr, markersize=dotSz)
        if newFig:
            ax.axis('equal')
        
    def plotZones(self, ax=None, col='b', dotSz=3, lw=1.5, ls='--', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for zDict in self.zoneDict:
            xy = transform(self.zoneDict[zDict][2], self.origin_plt, self.basisVectors_plt)
            ax.plot(xy[:,0], xy[:,1], 
                    ls=ls, color=col, lw=lw, marker=mrkr, markersize=dotSz)
        if newFig:
            ax.axis('equal')

    def plotPanels(self, aIdx=0, ax=None, col='g', dotSz=3, lw=0.5, ls='-', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for z,_ in enumerate(self.zoneDict):
            for p in self.panels[z][aIdx].geoms:
                xy = transform(np.transpose(p.exterior.xy), self.origin_plt, self.basisVectors_plt)
                ax.plot(xy[:,0], xy[:,1], 
                    ls=ls, color=col, lw=lw, marker=mrkr, markersize=dotSz)
        if newFig:
            ax.axis('equal')

    def plotTapField(self, field, dx=None, ax=None, fldRange=None, nLvl=100, cmap='RdBu'):
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
        ax.contourf(X, Y, Z,
                levels=levels, cmap=cmap)
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')

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


class Faces:
    """---------------------------------- Internals -----------------------------------"""
    def __init__(self,
                    members: List[face]=[],
                    file_basic=None,
                    file_derived=None,
                    ):
        self._currentIndex = 0
        self._members: List[face] = members

        if file_basic is not None and file_derived is not None:
            self.readFromFile(file_basic=file_basic,file_derived=file_derived)
        elif file_basic is not None:
            self.readFromFile(file_basic=file_basic)
    
    def _numOfMembers(self):
        if self.members is None:
            return 0
        return len(self.members)

    def __iter__(self):
        return self

    def __next__(self) -> face:
        if self._currentIndex < self._numOfMembers():
            member = self._members[self._currentIndex]
            self._currentIndex += 1
            return member
        else:
            self._currentIndex = 0
            raise StopIteration

    def __getitem__(self, key) -> face:
        return self._members[key]
    
    def __setitem__(self, key: int, value: face):
        self._members[key] = value

    def __str__(self) -> str:
        name = 'Faces: '
        for f in self.members:
            name += f.name + ', '
        return name
        
    """--------------------------------- Properties -----------------------------------"""
    @property
    def members(self) -> List[face]:
        return self._members
    @members.setter
    def members(self,value: List[face]) -> None:
        self._members = value

    @property
    def tapNo(self) -> List[int]:
        tapNo = []
        for f in self.members:
            tapNo.extend(f.tapNo)
        return tapNo

    @property
    def tapIdx(self) -> List[int]:
        tapIdx = []
        for f in self.members:
            tapIdx.extend(f.tapIdx)
        return tapIdx

    @property
    def tapIdx(self) -> List[int]:
        tapIdx = []
        for f in self.members:
            tapIdx.extend(f.tapIdx)
        return tapIdx

    @property
    def tapName(self) -> List[str]:
        tapName = []
        for f in self.members:
            tapName.extend(f.tapName)
        return tapName
    
    @property
    def zoneDict(self):
        allZones = {}
        i = 0
        for fc in self.members:
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
    def NumZones(self):
        return len(self.zoneDict)

    @property
    def NumTaps(self):
        num = 0
        for fc in self.members:
            num += fc.NumTaps
        return num

    @property
    def NumPanels(self):
        if len(self.members) == 0:
            return 0
        num = 0
        for fc in self.members:
            num += fc.NumPanels
        return num

    @property
    def error_in_panels(self):
        val = []
        for fc in self._members:
            val.append(fc.error_in_panels)
        return val

    @property
    def error_in_zones(self):
        val = []
        for fc in self._members:
            val.append(fc.error_in_zones)
        return val

    @property
    def NumPanelsPerArea(self):
        if len(self.members) == 0:
            return 0
        for i,fc in enumerate(self.members):
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
        for fc in self.members:
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
        for fc in self.members:
            for z, zonei in enumerate(fc.zoneDict.values()):
                zone = list(zonei)[0:2]
                zone.append([])
                idxZ = list(mainZoneDict.values()).index(zone) # index of the current zone in the main zoneDict
                for a,_ in enumerate(fc.nominalPanelAreas):
                    tapIdxs[idxZ].extend(fc.tapIdxPerPanel[z][a]) 
        return tapIdxs

    @property
    def panelAreas(self):
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

    """-------------------------------- Data handlers ---------------------------------"""
    def Update(self):
        for fc in self.members:
            fc.Update()

    def writeToFile(self,file_basic, file_derived=None):
        getDerived = file_derived is not None
        basic = {}
        derived = {}
        for i,fc in enumerate(self.members):
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

        self.members = []
        for i, bsc in enumerate(basic):
            mem = face()
            if derived is None:
                mem.from_dict(basic=basic[bsc])
            else:
                mem.from_dict(basic=basic[bsc],derived=derived[bsc])
            self.members.append(mem)

    """--------------------------------- Plotters -------------------------------------"""
    def plotEdges(self, ax=None, showName=True, col='k', dotSz=3, lw=2, ls='-', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._members:
            fc.plotEdges(ax=ax, showName=showName, col=col, dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

    def plotTaps(self, ax=None, showTapNo=False, fontsize='small', col='k', dotSz=3, lw=2, ls='None', mrkr='.'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._members:
            fc.plotTaps(ax=ax, showTapNo=showTapNo, fontsize=fontsize, col=col, dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

    def plotTribs(self, ax=None, col='r', dotSz=3, lw=0.5, ls='-.', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._members:
            fc.plotTribs(ax=ax, col=col, dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

    def plotZones(self, ax=None, col='b', dotSz=3, lw=1.5, ls='--', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._members:
            fc.plotZones(ax=ax, col=col, dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

    def plotPanels(self, ax=None, aIdx=0, col='g', dotSz=3, lw=0.5, ls='-', mrkr='None'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for fc in self._members:
            fc.plotPanels(ax=ax, aIdx=aIdx, col=col, dotSz=dotSz, lw=lw, ls=ls, mrkr=mrkr)
        if newFig:
            ax.axis('equal')

    def plotTapField(self, field, ax=None, dx=None, fldRange=None, nLvl=100, cmap='RdBu'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        contours = []
        for fc in self._members:
            c = fc.plotTapField(ax=ax, field=field, dx=dx, fldRange=fldRange, nLvl=nLvl, cmap=cmap)
            contours.append(c)
        if newFig:
            plt.colorbar()
            ax.axis('equal')
            ax.axis('off')
        return

    def plotPanelField(self, field, fldName, dIdx=0, aIdx=0, showValueText=False, strFmt="{:.3g}", fldRange=None, ax=None, nLvl=100, cmap='RdBu'):
        newFig = False
        if ax is None:
            newFig = True
            fig = plt.figure()
            ax = fig.add_subplot()
        for f,fc in enumerate(self._members):
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
                faces_file_basic=None, 
                faces_file_derived=None,
                ):
        super().__init__(members=faces, file_basic=faces_file_basic, file_derived=faces_file_derived)
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
        return self._members
    @faces.setter
    def faces(self,value: List[face]) -> None:
        self._members = value

    """-------------------------------- Data handlers ---------------------------------"""
    def writeToFile(self, file_basic, file_derived=None) -> None:
        # write the basic building details to the file 
        super().writeToFile(file_basic, file_derived)
        # finally add derived things if any
        pass

