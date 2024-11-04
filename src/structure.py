# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4, 2024

@author: Tsinuel Geleta
"""
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import copy
import inspect

from typing import List,Literal,Dict,Tuple,Any,Union,Set
from scipy import signal
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import skew,kurtosis
from scipy.interpolate import interp1d
from matplotlib.patches import Arc
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from windCAD import SamplingLines

# internal imports
import windPlotting as wplt
import windCAD
import windCodes as wc
import wind

#------------------------------- STRUCTURES -----------------------------------
'''
Definitions
===========

--> FORCE COEFFICIENT (Cf):
    Generally defined as the ratio of the force on a component (panel, node, element, frame, 
    etc) to the dynamic pressure times the reference area of the entire building.
        Cf  = F/(0.5*rho*U^2*Aref) 
            = (Cp*0.5*rho*U^2*Acomp)/(0.5*rho*U^2*Aref) 
    To calculate it from the pressure coefficient Cp, we need to multiply Cp by the area 
    ratio between the component and the reference area.
        ---------------------------
        |  Cf = Cp * (Acomp/Aref) |
        ---------------------------
            - Acomp is the area of the component
            - Aref is the reference area for the entire building
            - Cp is the pressure coefficient on the component
    
--> MOMENT COEFFICIENT (Cm):
    Generally defined as the ratio of the moment on a component (panel, node, element, frame,
    etc) to the dynamic pressure times the reference area of the entire building times the
    reference length of the entire building.
        Cm  = M/(0.5*rho*U^2*Aref*Lref) 
            = (Cm*0.5*rho*U^2*Acomp*Lcomp)/(0.5*rho*U^2*Aref*Lref)
        -----------------------------------------
        |  Cm = Cm * (Acomp*Lcomp)/(Aref*Lref)  |
        -----------------------------------------
            - Acomp is the area of the component
            - Aref is the reference area for the entire building
            - Lcomp is the length of the component
            - Lref is the reference length for the entire building
            - Cm is the moment coefficient on the component
'''
class panel(windCAD.panel_CAD):
    def __init__(self,
                parentFace: windCAD.face = None,
                vertices: np.ndarray = None,
                ID: int = None,
                name: str = None,
                supportNodes: List[windCAD.node_CAD] = [],
                parentBldg: wind.bldgCp = None,
                ) -> None:
        super().__init__(parentFace=parentFace, vertices_global=vertices, ID=ID, name=name, supportNodes=supportNodes)
        
        self.parentBldg : wind.bldgCp = parentBldg
    
    @property
    def Cf_TH(self) -> np.ndarray:  # [N_AoA, N_components=3, Ntime]
        debug = False
        if self.parentBldg is None:
            return None
        # get involved taps
        tapIdxs_inFace, overlappingAreas = self.getInvolvedTaps(debug=debug,showPlots=debug)
        if debug:
            print(f"tapIdxs_inFace = {tapIdxs_inFace}")
            print(f"overlappingAreas = {overlappingAreas}")
        mainTapIdxs = self.parentFace.tapIdx[tapIdxs_inFace]
        if debug:
            print(f"mainTapIdxs = {mainTapIdxs}")
        # CpTH has shape [N_AoA, Ntaps, Ntime]
        # to integrate over the area, we need to multiply self.parentBldg.CpOfT[:,mainTapIdxs,:] by the overlappingAreas, sum over the taps and divide 
        # by the total area
        # [N_AoA, 1, N_supports, Ntime]
        
        CpOfT = self.parentBldg.CpOfT[:,mainTapIdxs,:]
        if debug:
            print(f"Shape of CpOfT = {np.shape(CpOfT)}")
        overlappingAreas = overlappingAreas[np.newaxis,:,np.newaxis]
        if debug:
            print(f"Shape of overlappingAreas = {np.shape(overlappingAreas)}")
        CpTH_aAvgd = np.sum(CpOfT*overlappingAreas, axis=1)/self.area
        # CpTH_aAvgd = CpTH_aAvgd[:,np.newaxis,:]
        
        Aref = self.parentBldg.A_ref # [N_components=3]
        # area ratio between self.area over each of the reference areas
        Area_ratio = np.array([self.area/Aref[i] for i in range(3)], dtype=float)
        # use self.normal to get the direction of the force and multiply each component by the corresponding area ratio
        A_multiplier = np.array([self.normal[i]*Area_ratio[i] for i in range(3)], dtype=float)
        
        # Cf_TH [N_AoA, N_components=3, Ntime]
        Cf_TH = CpTH_aAvgd[:,np.newaxis,:]*A_multiplier[np.newaxis,:,np.newaxis]
        if debug:
            print(f"Shape of Cf_TH = {np.shape(Cf_TH)}")
        return Cf_TH
    
    @property
    def Cf_TH_atSupports(self) -> np.ndarray:  # [N_AoA, N_components=3, N_supports, Ntime]
        debug = False
        areaShares = self.supportAreaShares_normalized
        if debug:
            print(f"areaShares = {areaShares} with shape {np.shape(areaShares)}")
        # Cf_TH [N_AoA, N_components=3, Ntime]
        Cf_TH = self.Cf_TH
        if debug:
            print(f"Shape of Cf_TH = {np.shape(Cf_TH)}")
        # Cf_TH_atSupports [N_AoA, N_components=3, N_supports, Ntime]
        Cf_TH_sup = Cf_TH[:,:,np.newaxis,:]*areaShares[np.newaxis,np.newaxis,:,np.newaxis]
        if debug:
            print(f"Shape of Cf_TH_sup = {np.shape(Cf_TH_sup)}")
        return Cf_TH_sup
        
    
        
class node(windCAD.node_CAD):
    def __init__(self, x: float, y: float=None, z: float=None, ID: int=None, name=None, 
                 fixedDOF: List[bool]=[False,False,False,False,False,False],
                 DOF_indices: List[int]=[None, None, None, None, None, None],
                 nodeType: Literal['support','joint','internal']='joint',
                 connectedElements: List[windCAD.element_CAD]=[],
                 connectedPanels: List[windCAD.panel_CAD]=[],
                 parentFrame: windCAD.frame2D_CAD=None,
                ) -> None:
        super().__init__(x=x, y=y, z=z, ID=ID, name=name, fixedDOF=fixedDOF, DOF_indices=DOF_indices, nodeType=nodeType, connectedElements=connectedElements, connectedPanels=connectedPanels, parentFrame=parentFrame)
        
        
    @property
    def Cf_TH(self) -> np.ndarray:
        # Cf_TH [N_AoA, N_components=3, Ntime]
        # sum over the connected panels (contributions from each panel)
        for i, panel in enumerate(self.connectedPanels):
            this_nodes_id_in_panel = panel.supportNodes.index(self)
            if i == 0:
                Cf_TH = panel.Cf_TH_atSupports[:,:,this_nodes_id_in_panel,:]
            else:
                Cf_TH += panel.Cf_TH_atSupports[:,:,this_nodes_id_in_panel,:]
        return Cf_TH
        
    
class element(windCAD.element_CAD):
    def __init__(self,
                 startNode: windCAD.node_CAD, endNode: windCAD.node_CAD, internalNodes: List[windCAD.node_CAD]=[],
                 ID: int=None, name=None,
                 parentFrame: windCAD.frame2D_CAD=None,
                 E: float=None, 
                 A: float=None, 
                 Iy: float=None,
                 Iz: float=None,
                 J: float=None,                 
                ) -> None:
        super().__init__(startNode=startNode, endNode=endNode, internalNodes=internalNodes, ID=ID, name=name, parentFrame=parentFrame)
        
        self.E: float = E   # Young's modulus
        self.A: float = A   # cross-sectional area
        self.Iy: float = Iy # moment of inertia about the y-axis
        self.Iz: float = Iz # moment of inertia about the z-axis
        self.J: float = J   # torsional constant
        
        
        
    @property
    def local_stiffness_matrix(self) -> np.ndarray:
        # local stiffness matrix [6,6]
        L = self.length
        E = self.E
        A = self.A
        Iy = self.Iy
        Iz = self.Iz
        J = self.J
        
        EA_L = E * A / L
        EI_y = 12 * E * Iy / L**3
        EI_z = 12 * E * Iz / L**3
        GJ_L = E * J / L
        EI_y_L2 = 6 * E * Iy / L**2
        EI_z_L2 = 6 * E * Iz / L**2
        EI_y_L = 4 * E * Iy / L
        EI_z_L = 4 * E * Iz / L
        EI_y_2L = 2 * E * Iy / L
        EI_z_2L = 2 * E * Iz / L
        
        # Local stiffness matrix (12x12) in local coordinates
        k = np.array([
            [  EA_L,        0,        0,        0,        0,        0,     -EA_L,        0,        0,        0,        0,        0],
            [     0,     EI_z,        0,        0,        0,  EI_z_L2,         0,    -EI_z,        0,        0,        0,  EI_z_L2],
            [     0,        0,     EI_y,        0, -EI_y_L2,        0,         0,        0,    -EI_y,        0, -EI_y_L2,        0],
            [     0,        0,        0,     GJ_L,        0,        0,         0,        0,        0,    -GJ_L,        0,        0],
            [     0,        0, -EI_y_L2,        0,   EI_y_L,        0,         0,        0,  EI_y_L2,        0,  EI_y_2L,        0],
            [     0,  EI_z_L2,        0,        0,        0,   EI_z_L,         0, -EI_z_L2,        0,        0,        0,  EI_z_2L],
            [ -EA_L,        0,        0,        0,        0,        0,      EA_L,        0,        0,        0,        0,        0],
            [     0,    -EI_z,        0,        0,        0, -EI_z_L2,         0,     EI_z,        0,        0,        0, -EI_z_L2],
            [     0,        0,    -EI_y,        0,  EI_y_L2,        0,         0,        0,     EI_y,        0,  EI_y_L2,        0],
            [     0,        0,        0,    -GJ_L,        0,        0,         0,        0,        0,     GJ_L,        0,        0],
            [     0,        0, -EI_y_L2,        0,  EI_y_2L,        0,         0,        0,  EI_y_L2,        0,   EI_y_L,        0],
            [     0,  EI_z_L2,        0,        0,        0,  EI_z_2L,         0, -EI_z_L2,        0,        0,        0,   EI_z_L]
        ])
        
        return k
                        
        
    
    
    @property
    def transformation_matrix(self) -> np.ndarray:
        # transformation matrix [12,12]
        L = self.L
        
        # Direction cosines along the element's axis
        cx = (self.endNode.x - self.startNode.x) / L
        cy = (self.endNode.y - self.startNode.y) / L
        cz = (self.endNode.z - self.startNode.z) / L
        
        # Compute a perpendicular vector to define local axes
        if cx != 0 or cz != 0:
            v_x = np.array([-cz, 0, cx])  # arbitrary vector orthogonal to cx, cz
        else:
            v_x = np.array([0, 1, 0])  # if the element is along the y-axis

        # Normalize v_x to be the local x-axis perpendicular to the element's axis
        v_x = v_x / np.linalg.norm(v_x)
        
        # Local y-axis vector
        v_y = np.cross([cx, cy, cz], v_x)
        v_y = v_y / np.linalg.norm(v_y)
        
        # Define transformation matrix for local to global coordinates
        T = np.zeros((12, 12))
        
        # Assign rotation matrix to transformation matrix blocks
        R = np.array([
            [cx, cy, cz],
            v_x,
            v_y
        ])
        
        # Fill T matrix with rotation blocks
        T[:3, :3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:, 9:] = R
        
        return T
            
    
    @property
    def global_stiffness_matrix(self) -> np.ndarray:
        # global stiffness matrix [6,6]
        k = self.local_stiffness_matrix
        T = self.transformation_matrix
        k_global = T.T @ k @ T
        return k_global
        
        
class frame2D(windCAD.frame2D_CAD):
    def __init__(self,
                 parentBuilding: wind.bldgCp,
                 ID: int, name=None,
                 nodes: List[windCAD.node_CAD]=[],
                 elements: List[windCAD.element_CAD]=[],
                ) -> None:
        super().__init__(ID=ID, name=name,parentBuilding=parentBuilding, nodes=nodes, elements=elements)
        self.parentBuilding : wind.bldgCp = parentBuilding
        
    @property
    def Cf_TH(self) -> np.ndarray:
        # Cf_TH [N_AoA, N_components=3, N_nodes, Ntime]
        
        Cf_TH = None
        # collect the Cf_TH from each node
        for i, node in enumerate(self.nodes):
            if i == 0:
                Cf_TH = node.Cf_TH[:,:,np.newaxis,:]
            else:
                Cf_TH = np.concatenate((Cf_TH, node.Cf_TH[:,:,np.newaxis,:]), axis=2)
                
        return Cf_TH
    
    @property
    def Cf_Stats(self) -> dict:
        Cf_stats = wind.get_CpTH_stats(self.Cf_TH,axis=-1, peakSpecs=self.parentBuilding.peakSpecs)
        return Cf_stats

