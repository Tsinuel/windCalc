# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24, 2023 11:00:00 AM

@author: Tsinuel Geleta
"""
import numpy as np
import pandas as pd
import os
import warnings
import shapely.geometry as shp
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import json
import copy

from typing import List,Literal,Dict,Tuple,Any,Union,Set
from scipy import signal
from scipy.stats import skew,kurtosis
from scipy.interpolate import interp1d
from matplotlib.patches import Arc
from matplotlib.patches import Patch

# internal imports


#===============================================================================
#=============================== FUNCTIONS =====================================
#===============================================================================

def velRatio_exposureChange(from_z0, to_z0=0.03, zg=500, zref=10):
    """
    Calculates the velocity ratio for a change in exposure category

    Parameters
    ----------
    from_z0 : float
        The roughness length of the original exposure category
    to_z0 : float, optional
        The roughness length of the new exposure category. The default is 0.03.
    zg : float, optional
        The height where the two exposure categories are assumed to be equal. The default is 500.
    zref : float, optional
        The reference height where the velocity ratio is calculated. The default is 10.

    Returns
    -------
    float
        The velocity ratio for a change in exposure from the original to the new exposure category.
    """

    uStar_ratio = np.log(zg / from_z0) / np.log(zg / to_z0)
    return uStar_ratio * np.log(zref / to_z0) / np.log(zref / from_z0)

def velRatio_heightChange(from_Z, to_Z=10, z0=0.03):
    """
    Calculates the velocity ratio for a change in height

    Parameters
    ----------
    from_Z : float
        The height of the original velocity
    to_Z : float, optional
        The height of the new velocity. The default is 10.
    z0 : float, optional
        The roughness length of the surface. The default is 0.03.

    Returns
    -------
    float
        The velocity ratio for a change in height from the original to the new height.
    """

    return np.log(to_Z / z0) / np.log(from_Z / z0)

def velRatio_gustDurationChange(from_gustDuration, to_gustDuration=3):
    """
    Calculates the velocity ratio for a change in gust duration, in seconds, using the Durst factor.

    Parameters
    ----------
    from_gustDuration : float
        The gust duration in seconds of the original velocity
    to_gustDuration : float, optional
        The gust duration of the new velocity. The default is 3 seconds.

    Returns
    -------
    float
        The velocity ratio for a change in gust duration from the original to the new gust duration.
    """

    return getDurstFactor(to_gustDuration) / getDurstFactor(from_gustDuration)

def getDurstFactor(gustDuration):
    """Returns the velocity ratio to convert from 1 hr to the input gust duration.

    Parameters:
    ----------
        gustDuration (float): The output gust duration in seconds.

    Returns:
    -------
        float: The velocity ratio to convert from 1 hr to the input gust duration.

    References:
    ----------
        Durst, C.S., 1960. Wind speeds over short periods of time. Meteorol. Mag. 
            89, 181-187.
        ASCE/SEI, 2021. Minimum Design Loads and Associated Criteria for Buildings and 
            Other Structures. ASCE Standard ASCE/SEI 7-22, ASCE standard. American Society 
            of Civil Engineers, Reston, VA, USA. https://doi.org/10.1061/9780784415788
    """
    
    if any(gustDuration < 1.03780849700000) or any(gustDuration > 3600):
        raise ValueError("The gust duration must be between 1.037808497 and 3600 seconds.")
    gustDur = [1.03780849700000,1.12767394700000,1.22532098700000,1.33142343500000,1.45694344800000,1.57538111200000,1.70810738900000,1.86913930100000,2.00713704800000,
               2.19136209900000,2.38111554700000,2.58730004200000,2.81133837300000,3.05477653000000,3.31929437600000,3.60671723400000,3.97673384700000,4.35593123500000,
               4.73311808800000,5.14296613700000,5.58830356600000,6.07220345400000,6.59800498600000,7.16933648900000,7.79014047400000,8.46470084200000,9.19767243900000,
               9.99411318600000,10.8595189700000,11.7998616000000,12.8216299400000,13.9318748000000,15.1382574900000,16.4491027200000,17.8734560900000,19.4211464200000,
               21.1028536600000,22.9301825500000,24.9157426800000,27.0732355500000,29.4175490800000,31.9648603600000,34.7327472800000,37.7403098400000,41.0083019000000,
               44.5592744800000,48.4177312900000,52.6102978700000,57.1659053000000,62.1159898600000,67.4947099300000,73.3391817300000,79.6897354200000,86.5901934300000,
               94.0881728100000,102.235413900000,111.088137200000,120.707431600000,131.159675700000,142.516995800000,154.857763800000,168.267138200000,182.837651200000,
               198.669847400000,215.872978200000,234.565754800000,254.877168000000,276.947377900000,300.928681700000,326.986563800000,355.300838400000,386.066890000000,
               419.497021800000,455.821920700000,495.292249100000,538.180374500000,584.782249200000,635.419452800000,690.441410500000,750.227804900000,815.191196200000,
               885.779868400000,962.480923500000,1045.82364200000,1136.38313600000,1234.78431700000,1341.70621000000,1457.88663500000,1584.12730400000,1721.29934900000,
               1870.34933400000,2032.30578900000,2208.28630400000,2399.50524700000,2607.28213400000,2833.05074500000,3078.36901100000,3344.92976600000,3600]
    v_t_v3600 = [1.55811434100000,1.55601528500000,1.55444228200000,1.55128020300000,1.54957932800000,1.54695585600000,1.54491784700000,1.54151005900000,1.53832623700000,
                 1.53464510900000,1.53160764800000,1.52716545800000,1.52373747500000,1.51947963300000,1.51528151900000,1.51108734300000,1.50438942800000,1.49865733900000,
                 1.49377581200000,1.48752745700000,1.48176725500000,1.47571416100000,1.46931936000000,1.46272929800000,1.45589516000000,1.44886576100000,1.44154347000000,
                 1.43397710200000,1.42611784400000,1.41825858500000,1.41035051000000,1.40195428300000,1.39497369900000,1.38584524300000,1.37715612500000,1.36851582200000,
                 1.35968025700000,1.35069824700000,1.34181386700000,1.33288067200000,1.32389866200000,1.31462376000000,1.30515359700000,1.29714789300000,1.28806825200000,
                 1.27884216500000,1.26961607900000,1.26043880700000,1.25131035100000,1.24252360200000,1.23319988500000,1.22416906000000,1.21547994100000,1.20622944700000,
                 1.19858985700000,1.18994955300000,1.18209029400000,1.17423103500000,1.16617651500000,1.15841488700000,1.15065325900000,1.14381912000000,1.13654564500000,
                 1.12971150700000,1.12302381400000,1.11653138300000,1.11023421300000,1.10379059700000,1.09890906900000,1.09392991200000,1.08816970900000,1.08319055100000,
                 1.07845547000000,1.07396446500000,1.06952227500000,1.06534508200000,1.06151657000000,1.05707438100000,1.05414546400000,1.05087484100000,1.04828763100000,
                 1.04594449800000,1.04325965800000,1.04047718800000,1.03759708700000,1.03427764800000,1.03125110100000,1.02827336900000,1.02549089900000,1.02246435200000,
                 1.01968188100000,1.01675296500000,1.01387286400000,1.01079750200000,1.00830792300000,1.00614913700000,1.00386027600000,1.00239036100000,1]
    return np.interp(gustDuration, gustDur, v_t_v3600)

def CpConversionFactor(from_: Literal['simulated', 'NBCC', 'ASCE'], to_: Literal['NBCC', 'ASCE'], 
                        from_Z: Union[float, np.ndarray], from_z0: float=None, from_gustDuration: float=None, to_z0: float=0.03, 
                        to_Z: float=10, 
                        zg: float=500, 
                        ):
    """
    Calculates the lumped reference velocity conversion factor between experimental/CFD data, 
    building codes, and other sources of wind loads. It considers the change in exposure 
    category, height, and gust duration.

    Parameters
    ----------
    from_type : Literal['simulated', 'NBCC', 'ASCE']
        The type of the input wind load. If 'simulated', the gust duration must be specified.
    to_type : Literal['NBCC', 'ASCE']
        The type of the output wind load.
    in_z0 : float
        The roughness length of the input wind load. If it is 'simulated', it is the best
        fitted roughness length. If it is 'NBCC' or 'ASCE', it is the roughness length of the
        open terrain is used.
    in_gustDuration : float
        The gust duration of the input wind load. If it is 'simulated', it is a required input.
        If it is 'NBCC' or 'ASCE', it is the gust duration of the building codes, i.e., 3600
        seconds for 'NBCC' and 3 seconds for 'ASCE' are used unless specified.
    to_z0 : float, optional
        The roughness length of the output wind load. The default is 0.03.
    zref : float, optional
        The reference height where the velocity ratio is calculated. The default is 10.
    zg : float, optional
        The height where the two exposure categories are assumed to be equal. The default is 500.

    Returns
    -------
    factor : float
        The lumped reference velocity conversion factor between the input and output wind loads.

    References
    ----------
    ASCE/SEI, 2021. Minimum Design Loads and Associated Criteria for Buildings and
        Other Structures. ASCE Standard ASCE/SEI 7-22, ASCE standard. American Society
        of Civil Engineers, Reston, VA, USA. https://doi.org/10.1061/9780784415788
    NRCC, 2022. National Building Code of Canada: 2020, 15th ed, Government of Canada. 
        National Research Council of Canada, Ottawa, Canada. 
        https://doi.org/10.4224/w324-hv93
    St Pierre, L.M., Kopp, G.A., Surry, D., Ho, T.C.E., 2005. The UWO contribution to the 
        NIST aerodynamic database for wind loads on low buildings: Part 2. Comparison of 
        data with wind load provisions. J. Wind Eng. Ind. Aerodyn. 93, 31-59. 
        https://doi.org/10.1016/j.jweia.2004.07.007
    """
    from_Z = np.array(from_Z) if isinstance(from_Z, list) or isinstance(from_Z, tuple) else from_Z
    if from_ == 'simulated':
        if from_gustDuration is None:
            raise ValueError("The gust duration must be specified for simulated wind load inputs.")
        if from_z0 is None:
            raise ValueError("The roughness length must be specified for simulated wind load inputs.")

    elif from_ == 'NBCC': 
        from_gustDuration = 3600 if from_gustDuration is None else from_gustDuration
        from_z0 = 0.03 if from_z0 is None else from_z0
    elif from_ == 'ASCE':
        from_gustDuration = 3 if from_gustDuration is None else from_gustDuration
        from_z0 = 0.03 if from_z0 is None else from_z0

    if to_ == 'NBCC':
        to_gustDuration = 3600
    elif to_ == 'ASCE':
        to_gustDuration = 3
    
    F_exp = velRatio_exposureChange(from_z0=from_z0, to_z0=to_z0, zg=zg, zref=to_Z)
    F_height = velRatio_heightChange(from_Z=from_Z, to_Z=to_Z, z0=from_z0)
    F_gustDuration = velRatio_gustDurationChange(from_gustDuration=from_gustDuration, to_gustDuration=to_gustDuration)

    factor = (F_exp * F_height * F_gustDuration) ** (-2)
    return factor



def NBCC2020_CpCg(Figure:Literal['4.1.7.6.-A', '4.1.7.6.-B', '4.1.7.6.-C', '4.1.7.6.-D', '4.1.7.6.-E', '4.1.7.6.-F', '4.1.7.6.-G', '4.1.7.6.-H'], subfig='a'):
    if Figure == '4.1.7.6.-E':
        if subfig == 'a':
            CpCg = {}
            CpCg['Name'] = 'NBCC 2020'

            CpCg['Min'] = {}
            CpCg['Min']['area'] = {}
            CpCg['Min']['area']["NBCC 2020, Zone c"] = [0.1, 1, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone s"] = [0.1, 2, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone r"] = [0.1, 0.85, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone e"] = [0.5, 1, 50, 200]
            CpCg['Min']['area']["NBCC 2020, Zone w"] = [0.5, 1, 50, 200]

            CpCg['Min']['value'] = {}
            CpCg['Min']['value']["NBCC 2020, Zone c"] = [-5, -5, -4, -4]
            CpCg['Min']['value']["NBCC 2020, Zone s"] = [-3.6, -3.6, -2.65, -2.65]
            CpCg['Min']['value']["NBCC 2020, Zone r"] = [-2.5, -2.5, -2, -2]
            CpCg['Min']['value']["NBCC 2020, Zone e"] = [-2.1, -2.1, -1.5, -1.5]
            CpCg['Min']['value']["NBCC 2020, Zone w"] = [-1.8, -1.8, -1.5, -1.5]

            CpCg['Max'] = {}
            CpCg['Max']['area'] = {}
            CpCg['Max']['area']["NBCC 2020, Zone c"] = [0.1, 1, 8.5, 100]
            CpCg['Max']['area']["NBCC 2020, Zone s"] = [0.1, 1, 8.5, 100]
            CpCg['Max']['area']["NBCC 2020, Zone r"] = [0.1, 1, 8.5, 100]
            CpCg['Max']['area']["NBCC 2020, Zone e"] = [0.5, 1, 50, 200]
            CpCg['Max']['area']["NBCC 2020, Zone w"] = [0.5, 1, 50, 200]

            CpCg['Max']['value'] = {}
            CpCg['Max']['value']["NBCC 2020, Zone c"] = [0.8, 0.8, 0.5, 0.5]
            CpCg['Max']['value']["NBCC 2020, Zone s"] = [0.8, 0.8, 0.5, 0.5]
            CpCg['Max']['value']["NBCC 2020, Zone r"] = [0.8, 0.8, 0.5, 0.5]
            CpCg['Max']['value']["NBCC 2020, Zone e"] = [1.75, 1.75, 1.3, 1.3]
            CpCg['Max']['value']["NBCC 2020, Zone w"] = [1.75, 1.75, 1.3, 1.3]
    elif Figure == '4.1.7.6.-F':
        if subfig == 'a':
            CpCg = {}
            CpCg['Name'] = 'NBCC 2020'

            CpCg['Min'] = {}
            CpCg['Min']['area'] = {}
            CpCg['Min']['area']["NBCC 2020, Zone c"] = [0.1, 1, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone s'"] = [0.1, 1, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone s"] = [0.1, 1, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone r"] = [0.1, 1, 10, 100]
            CpCg['Min']['area']["NBCC 2020, Zone e"] = [0.5, 1, 50, 200]
            CpCg['Min']['area']["NBCC 2020, Zone w"] = [0.5, 1, 50, 200]

            CpCg['Min']['value'] = {}
            CpCg['Min']['value']["NBCC 2020, Zone c"] = [-4.9, -4.9, -3.1, -3.1]
            CpCg['Min']['value']["NBCC 2020, Zone s'"] = [-4.2, -4.2, -3.1, -3.1]
            CpCg['Min']['value']["NBCC 2020, Zone s"] = [-3.5, -3.5, -2.6, -2.6]
            CpCg['Min']['value']["NBCC 2020, Zone r"] = [-3, -3, -2.6, -2.6]
            CpCg['Min']['value']["NBCC 2020, Zone e"] = [-2.1, -2.1, -1.5, -1.5]
            CpCg['Min']['value']["NBCC 2020, Zone w"] = [-1.8, -1.8, -1.5, -1.5]

            CpCg['Max'] = {}
            CpCg['Max']['area'] = {}
            CpCg['Max']['area']["NBCC 2020, Zone c"] = [0.1, 1, 10, 100]
            CpCg['Max']['area']["NBCC 2020, Zone s'"] = [0.1, 1, 10, 100]
            CpCg['Max']['area']["NBCC 2020, Zone s"] = [0.1, 1, 10, 100]
            CpCg['Max']['area']["NBCC 2020, Zone r"] = [0.1, 1, 10, 100]
            CpCg['Max']['area']["NBCC 2020, Zone e"] = [0.5, 1, 50, 200]
            CpCg['Max']['area']["NBCC 2020, Zone w"] = [0.5, 1, 50, 200]

            CpCg['Max']['value'] = {}
            CpCg['Max']['value']["NBCC 2020, Zone c"] = [1.1, 1.1, 0.75, 0.75]
            CpCg['Max']['value']["NBCC 2020, Zone s'"] = [1.1, 1.1, 0.75, 0.75]
            CpCg['Max']['value']["NBCC 2020, Zone s"] = [1.1, 1.1, 0.75, 0.75]
            CpCg['Max']['value']["NBCC 2020, Zone r"] = [1.1, 1.1, 0.75, 0.75]
            CpCg['Max']['value']["NBCC 2020, Zone e"] = [1.75, 1.75, 1.3, 1.3]
            CpCg['Max']['value']["NBCC 2020, Zone w"] = [1.75, 1.75, 1.3, 1.3]
    return CpCg

def ASCE7_22_GCp(Figure:Literal['30.3-2A', '30.3-2B', '30.3-2C', '30.3-2D', '30.3-2E', '30.3-2F', '30.3-2G', '30.3-4'], subfig='a'):
    if Figure == '30.3-2C':
        if subfig == 'a':
            GCp = {}
            GCp['Name'] = 'ASCE 7-22'
            '''
            ft^2:      __,    10,         100,        200,        500,        __
            m^2:      0.1,    0.9290304,  9.290304,   18.580608,  46.45152,   100

            GCp['Min'] = {}
            GCp['Min']['area'] = {}
            GCp['Min']['area']['ASCE 7-22, Zone 1'] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Min']['area']['ASCE 7-22, Zone 2'] = [0.1, 0.9290304, 9.290304, 100]
            GCp['Min']['area']['ASCE 7-22, Zone 3'] = [0.1, 0.9290304, 9.290304, 100]
            GCp['Min']['area']['ASCE 7-22, Zone 4'] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Min']['area']['ASCE 7-22, Zone 5'] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Min']['value'] = {}
            GCp['Min']['value']['ASCE 7-22, Zone 1'] = [-1.5, -1.5, -0.8, -0.8]
            GCp['Min']['value']['ASCE 7-22, Zone 2'] = [-2.5, -2.5, -1.2, -1.2]
            GCp['Min']['value']['ASCE 7-22, Zone 3'] = [-3.0, -3.0, -1.4, -1.4]
            GCp['Min']['value']['ASCE 7-22, Zone 4'] = [-1.1, -1.1, -0.8, -0.8]
            GCp['Min']['value']['ASCE 7-22, Zone 5'] = [-1.4, -1.4, -0.8, -0.8]

            GCp['Max'] = {}
            GCp['Max']['area'] = {}
            GCp['Max']['area']['ASCE 7-22, Zone 1'] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']['ASCE 7-22, Zone 2'] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']['ASCE 7-22, Zone 3'] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']['ASCE 7-22, Zone 4'] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Max']['area']['ASCE 7-22, Zone 5'] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Max']['value'] = {}
            GCp['Max']['value']['ASCE 7-22, Zone 1'] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']['ASCE 7-22, Zone 2'] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']['ASCE 7-22, Zone 3'] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']['ASCE 7-22, Zone 4'] = [1.0, 1.0, 0.7, 0.7]
            GCp['Max']['value']['ASCE 7-22, Zone 5'] = [1.0, 1.0, 0.7, 0.7]
            '''

            GCp['Min'] = {}
            GCp['Min']['area'] = {}
            GCp['Min']['area']["NBCC 2020, Zone r"] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Min']['area']["NBCC 2020, Zone s"] = [0.1, 0.9290304, 9.290304, 100]
            GCp['Min']['area']["NBCC 2020, Zone c"] = [0.1, 0.9290304, 9.290304, 100]
            GCp['Min']['area']["NBCC 2020, Zone w"] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Min']['area']["NBCC 2020, Zone e"] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Min']['value'] = {}
            GCp['Min']['value']["NBCC 2020, Zone r"] = [-1.5, -1.5, -0.8, -0.8]
            GCp['Min']['value']["NBCC 2020, Zone s"] = [-2.5, -2.5, -1.2, -1.2]
            GCp['Min']['value']["NBCC 2020, Zone c"] = [-3.0, -3.0, -1.4, -1.4]
            GCp['Min']['value']["NBCC 2020, Zone w"] = [-1.1, -1.1, -0.8, -0.8]
            GCp['Min']['value']["NBCC 2020, Zone e"] = [-1.4, -1.4, -0.8, -0.8]

            GCp['Max'] = {}
            GCp['Max']['area'] = {}
            GCp['Max']['area']["NBCC 2020, Zone r"] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']["NBCC 2020, Zone s"] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']["NBCC 2020, Zone c"] = [0.1, 0.9290304, 18.580608, 100]
            GCp['Max']['area']["NBCC 2020, Zone w"] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Max']['area']["NBCC 2020, Zone e"] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Max']['value'] = {}
            GCp['Max']['value']["NBCC 2020, Zone r"] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']["NBCC 2020, Zone s"] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']["NBCC 2020, Zone c"] = [0.6, 0.6, 0.3, 0.3]
            GCp['Max']['value']["NBCC 2020, Zone w"] = [1.0, 1.0, 0.7, 0.7]
            GCp['Max']['value']["NBCC 2020, Zone e"] = [1.0, 1.0, 0.7, 0.7]

    elif Figure == '30.3-4':
        if subfig == 'a':
            GCp = {}
            GCp['Name'] = 'ASCE 7-22'
            
            GCp['Min'] = {}
            GCp['Min']['area'] = {}
            GCp['Min']['area']["NBCC 2020, Zone r"] = [0.1, 0.9, 9.3, 100]
            GCp['Min']['area']["NBCC 2020, Zone s'"] = [0.1, 0.9, 9.3, 100]
            GCp['Min']['area']["NBCC 2020, Zone s"] = [0.1, 0.9, 9.3, 100]
            GCp['Min']['area']["NBCC 2020, Zone c"] = [0.1, 0.9, 9.3, 100]
            GCp['Min']['area']["NBCC 2020, Zone w"] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Min']['area']["NBCC 2020, Zone e"] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Min']['value'] = {}
            GCp['Min']['value']["NBCC 2020, Zone r"] = [-1.6, -1.6, -1.4, -1.4]
            GCp['Min']['value']["NBCC 2020, Zone s'"] = [-2.2, -2.2, -1.7, -1.7]
            GCp['Min']['value']["NBCC 2020, Zone s"] = [-2.2, -2.2, -1.7, -1.7]
            GCp['Min']['value']["NBCC 2020, Zone c"] = [-2.7, -2.7, -1.7, -1.7]
            GCp['Min']['value']["NBCC 2020, Zone w"] = [-1.1, -1.1, -0.8, -0.8]
            GCp['Min']['value']["NBCC 2020, Zone e"] = [-1.4, -1.4, -0.8, -0.8]

            GCp['Max'] = {}
            GCp['Max']['area'] = {}
            GCp['Max']['area']["NBCC 2020, Zone r"] = [0.1, 0.9, 9.3, 100]
            GCp['Max']['area']["NBCC 2020, Zone s'"] = [0.1, 0.9, 9.3, 100]
            GCp['Max']['area']["NBCC 2020, Zone s"] = [0.1, 0.9, 9.3, 100]
            GCp['Max']['area']["NBCC 2020, Zone c"] = [0.1, 0.9, 9.3, 100]
            GCp['Max']['area']["NBCC 2020, Zone w"] = [0.1, 0.9290304, 46.45152, 100]
            GCp['Max']['area']["NBCC 2020, Zone e"] = [0.1, 0.9290304, 46.45152, 100]

            GCp['Max']['value'] = {}
            GCp['Max']['value']["NBCC 2020, Zone r"] = [0.6, 0.6, 0.4, 0.4]
            GCp['Max']['value']["NBCC 2020, Zone s'"] = [0.6, 0.6, 0.4, 0.4]
            GCp['Max']['value']["NBCC 2020, Zone s"] = [0.6, 0.6, 0.4, 0.4]
            GCp['Max']['value']["NBCC 2020, Zone c"] = [0.6, 0.6, 0.4, 0.4]
            GCp['Max']['value']["NBCC 2020, Zone w"] = [1.0, 1.0, 0.7, 0.7]
            GCp['Max']['value']["NBCC 2020, Zone e"] = [1.0, 1.0, 0.7, 0.7]

    return GCp


#===============================================================================
#================================ CLASSES ======================================
#===============================================================================

class ASCE_49_21:
    def __init__(self) -> None:
        pass

class NBCC_2020:
    def __init__(self) -> None:
        pass

class ASCE_7_22:
    def __init__(self) -> None:
        pass

    def __str__(self):
        return 'ASCE-7-16'
