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

class face:

    def __init__(self,
                name=None,
                ID=None,
                bldg=None,
                origin=None,
                basisVectors=None,
                vertices=None,
                tapNo=None,
                tapIdx=None,
                tapName=None,
                tapCoord=None,
                zoningCodeNames=None,
                zoneNames=None,
                zones=None,
                nominalPanelAreas=None,
                numOfNominalPanelAreas=5,
                ):
        self.name = name
        self.ID = ID
        self.bldg = bldg
        self.origin = origin                # origin of the face-local coordinate system
        self.basisVectors = basisVectors    # [[3,], [3,], [3,]] basis vectors of the local coord sys in the main 3D coord sys.

        self.vertices = vertices            # [Nv, 2] face corners that form a non-self-intersecting polygon. No need to close it with the last edge.
        self.zoningCodeNames = zoningCodeNames
        self.zoneNames = zoneNames          # dict of dicts: zone names per each zoning. e.g., [['NBCC-z1', 'NBCC-z2', ... 'NBCC-zM'], ['ASCE-a1', 'ASCE-a2', ... 'ASCE-aQ']]
        self.zones = zones                  # list of array of arrays: vertices in local coord sys of each subzone (r) belonging to each zone (z or a) from each code.
                                            #   e.g., zones = 
                                            #         [ [[z1r1,2], [z1r2,2], ... [z1rN1,2]], [[z2r1,2], [z2r2,2], ... [z2rN2,2]], ....... [[zMr1,2], [zMr2,2], ... [zMrNM,2]],  # NBCC
                                            #           [[a1r1,2], [a1r2,2], ... [a1rN1,2]], [[a2r1,2], [a2r2,2], ... [a2rN2,2]], ....... [[aQr1,2], [aQr2,2], ... [aQrNQ,2]] ] # ASCE
                                            #          NBCC has M number of zones. Its zone 1 has N1 number of subzones defined by the list vertices in zones[0][0]
                                            #          zones[1][Q][NQ] is the coordinates of NQ'th subzone belonging to the Q'th zone of ASCE
                                            # If you don't want to apply any zoning, put the face vertices as [[[[Nv,2],],],]
        self.nominalPanelAreas = nominalPanelAreas  # a vector of nominal areas to generate panels. The panels areas will be close but not necessarily exactly equal to these.
        self.numOfNominalPanelAreas = numOfNominalPanelAreas
        self.tapNo = tapNo                  # [Ntaps,]
        self.tapIdx = tapIdx                # [Ntaps,] tap indices in the main matrix of the entire building
        self.tapName = tapName              # [Ntaps,]
        self.tapCoord = tapCoord            # [Ntaps,2]   ... from the local face origin

        self.tapTribs = None
        self.panels = None
        self.tapWghtPerPanel = None
        self.tapIdxPerPanel = None
        self.panelAreas = None


        self.__generateTributaries()
        self.__generatePanels()
    
    def __str__(self):
        return self.name

    def __generateTributaries(self):
        if self.vertices is None or self.tapCoord is None:
            return
        self.tapTribs = trimmedVoronoi(self.vertices, self.tapCoord)

    def __defaultZoneAndPanelConfig(self):
        if self.zones is None and self.vertices is not None:
            self.zoningCodeNames = {
                            0:'Default',
                        }
            self.zoneNames = {
                            0:{
                                0:'Default',
                            },
                        }
            self.zones = [
                            [
                                [
                                    self.vertices, # subzone: None
                                ], # Zone: Default
                            ], # Code: Default
                        ]
        if self.nominalPanelAreas is None:
            bound = shp.Polygon(self.vertices)
            maxArea = bound.area
            minArea = maxArea
            for trib in self.tapTribs.geoms:
                minArea = min((minArea, trib.area))

            self.nominalPanelAreas = np.linspace(minArea, maxArea, self.numOfNominalPanelAreas)

    def __generatePanels(self):
        self.__defaultZoneAndPanelConfig()
        if self.tapTribs is None or self.zones is None:
            return

        self.panels = ()
        self.tapWghtPerPanel = ()
        self.tapIdxPerPanel = ()
        for c,code in enumerate(self.zones):
            panels_c = ()
            pnlWeights_c = ()
            tapIdxByPnl_c = ()
            for z,zone in enumerate(code):
                panels_z = ()
                pnlWeights_z = ()
                tapIdxByPnl_z = ()
                for a,area in enumerate(self.nominalPanelAreas):
                    panels_a = np.asarray([],dtype=object)
                    pnlWeights_a = np.asarray([],dtype=object)
                    tapIdxByPnl_a = np.asarray([],dtype=object)
                    for r,subzone in enumerate(zone):
                        pnls,areas = meshRegionWithPanels(subzone,area,debug=False)
                        panels_a = np.append(panels_a, pnls)

                        tapsInSubzone = []
                        idxInSubzone = []
                        subzonePlygn = shp.Polygon(subzone)
                        for i,tap in zip(self.tapIdx,self.tapTribs.geoms):
                            if intersects(tap,subzonePlygn):
                                tapsInSubzone.append(tap)
                                idxInSubzone.append(i)
                        wght, Idx, overlaps = calculateTapWeightsPerPanel(pnls,tapsInSubzone,idxInSubzone)
                        pnlWeights_a = np.append(pnlWeights_a, wght)
                        tapIdxByPnl_a = np.append(tapIdxByPnl_a, Idx)
                    panels_z += (panels_a,)
                    pnlWeights_z += (pnlWeights_a,)
                    tapIdxByPnl_z += (tapIdxByPnl_a,)
                panels_c += (panels_z,)
                pnlWeights_c += (pnlWeights_z,)
                tapIdxByPnl_c += (tapIdxByPnl_z,)
            self.panels += (panels_c,)
            self.tapWghtPerPanel += (pnlWeights_c,)
            self.tapIdxPerPanel += (tapIdxByPnl_c,)

        print(f"Shape of 'panels': {np.shape(np.array(self.panels,dtype=object))}")
        print(f"Shape of 'pnlWeights': {np.shape(np.array(self.tapWghtPerPanel,dtype=object))}")
        print(f"Shape of 'tapIdxByPnl': {np.shape(np.array(self.tapIdxPerPanel,dtype=object))}")

        debugMode = False
        if debugMode:
            # plot panels
            for c,code in enumerate(self.zones):
                for a,area in enumerate(self.nominalPanelAreas):
                    plt.figure(figsize=[20,10])
                    plt.plot(taps[:,0],taps[:,1],'.k')
                    for t in tribs.geoms:
                        x,y = t.exterior.xy
                        plt.plot(x,y,':r',lw=0.5)
                    for z,zone in enumerate(code):
                        for subzone in zone:
                            plt.plot(subzone[:,0],subzone[:,1],'--k',lw=2.0)
                        for p in panels[c][z][a]:
                            # print(p)
                            x,y = p.exterior.xy
                            plt.plot(x,y,'-b',lw=0.5)
                    plt.axis('equal')
                    plt.show()




    def tapCoord3D(self):
        pass

    def vertices3D(self):
        pass




class Faces:
    def __init__(self,members=[]):
        self._currentIndex = 0
        self.members = members
    
    def _numOfMembers(self):
        if self.members is None:
            return 0
        return len(self.members)

    def __iter__(self):
        return self

    def __next__(self):
        if self._currentIndex < self._numOfMembers():
            member = self.members[self._currentIndex]
            self._currentIndex += 1
            return member
        raise StopIteration

class building:
    def __init__(self,
                name=None,
                H=None,     # average roof height
                He=None,    # eave height
                Hr=None,    # ridge height 
                B=None,     # shorter plan-inscribing-rectangle width
                D=None,     # longer plan-inscribing-rectangle width
                roofSlope=None,
                faces=None, # list of faces
                lScl=1.0,   # length scale
               ):
        self.name = name
        self.H = H
        self.He = He
        self.Hr = Hr
        self.B = B
        self.D = D
        self.roofSlope = roofSlope
        self.faces = faces
        self.lScl = lScl
    
    def __str__(self):
        return self.name

