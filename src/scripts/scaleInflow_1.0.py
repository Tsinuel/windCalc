# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:23:49 2022

@author: Tsinu

Description: Scales OpenFOAM inflow based on a target and current set of profiles.


** With scaling
"""
import numpy as np
import pandas as pd
import scipy.interpolate as scintrp
import matplotlib.pyplot as plt
import os

__showPlots = False
__precision = 6

def readProfiles(file,scaleBy):
    if ~os.path.exists(file):
        FileNotFoundError()
    profiles = pd.read_csv(file)
    
    if 'Z' not in profiles:
        if 'z' in profiles:
            profiles.rename(columns={'z':'Z'})
        else:
            raise Exception("All profiles must have Z column. Check profile:"+ file)
    
    for f in scaleBy:
        if f not in profiles:
            raise Exception("The profile column '"+ f +"' is listed in 'scaleBy' but not found in the profile: "+file)
    
    return profiles

def readBDformattedFile(file):
    # file in the format of boundaryData. Can be points or velocity
    entries = [line.split() for line in open(file)]
    entries = list(filter(None, entries))
    entries = entries[2:-1]
    
    if len(entries[0]) == 3:
        entries = [[s[0].strip('('), s[1], s[2].strip(')')] for s in entries]
        entries = pd.DataFrame.from_records(entries)
    elif len(entries[0]) == 5: 
        entries = pd.DataFrame.from_records(entries)
        entries = entries.drop([0,4], axis=1)
    
    return entries.astype('float64')

def writeBDformattedFile(file,vect):
    f = open(file,"w")
    f.write(str(len(vect))+"\n")
    f.close()
    f = open(file,"a")
    f.write("(\n")
    np.savetxt(f,vect,fmt='(%1.8g %1.8g %1.8g)',delimiter=' ')
    f.write(")")
    f.close()    

def getProfileScaleFactor(origProf,targProf,scaleBy):
    def plotProf(z,orig,targ,f,name):
        if not __showPlots:
            return
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(orig,z,'k-',label='orig')
        ax1.plot(targ,z,'r-',label='target')
        plt.xlabel(name)
        plt.ylabel('Z')
        ax1.legend()
        
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(f,z,'k+-',label='Target/original')
        plt.xlabel('Correction factor ('+name+')')
        plt.ylabel('Z')
        ax2.legend()
        plt.show()

    Z = np.unique(np.sort(np.append(np.asarray(origProf.Z), np.asarray(targProf.Z))))
    factor = Z
    cNames = ['Z']
    
    if 'U' in scaleBy:
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
        orig = intrp(Z)
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.U),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('U')
        plotProf(Z,orig,targ,f,'U')
    
    if 'Iu' in scaleBy:
        # intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),bounds_error=False,fill_value=(origProf.Iu[0],origProf.iloc[-1].Iu))
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),fill_value='extrapolate')
        orig = intrp(Z)
        # intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iu),bounds_error=False,fill_value=(targProf.Iu[0],targProf.iloc[-1].Iu))
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iu),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iu')
        plotProf(Z,orig,targ,f,'Iu')
    
    if 'Iv' in scaleBy:
        # intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),bounds_error=False,fill_value=(origProf.Iv[0],origProf.iloc[-1].Iv))
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),fill_value='extrapolate')
        orig = intrp(Z)
        # intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iv),bounds_error=False,fill_value=(targProf.Iv[0],targProf.iloc[-1].Iv))
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iv),fill_value='extrapolate')
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iv')
        plotProf(Z,orig,targ,f,'Iv')
    
    if 'Iw' in scaleBy:
        intrp = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iw),bounds_error=False,fill_value=(origProf.Iw[0],origProf.iloc[-1].Iw))
        orig = intrp(Z)
        intrp = scintrp.interp1d(np.asarray(targProf.Z), np.asarray(targProf.Iw),bounds_error=False,fill_value=(targProf.Iw[0],targProf.iloc[-1].Iw))
        targ = intrp(Z)
        f = targ/orig
        factor = np.vstack((factor,f))
        cNames.append('Iw')
        plotProf(Z,orig,targ,f,'Iw')
    
    factor = pd.DataFrame(np.transpose(factor),columns=cNames)
    return factor

def scaleVelocity(UofT,VofT,WofT,
                  Z, scaleBy,
                  origUofZ, fUofZ, fIuOfZ, fIvOfZ, fIwOfZ):
    
    U_old = origUofZ(Z)
    u_old = UofT - U_old
    
    U_new = fUofZ(Z)*U_old + fIuOfZ(Z)*u_old
    
    V_new = fIvOfZ(Z)*VofT
    
    W_new = fIwOfZ(Z)*WofT
        
    if __showPlots:
        fig1 = plt.figure()
        ax = fig1.add_subplot(1,1,1)
        ax.plot(UofT,Z,'.k',label='orig',ms=1)
        ax.plot(U_new,Z,'.r',label='scaled',ms=1)
        ax.plot(U_old,Z,'xb',label='meanU-old',ms=1)
        ax.plot(fUofZ(Z)*U_old,Z,'xg',label='meanU-new',ms=1)
        plt.xlabel('U')
        plt.ylabel('Z')
        ax.legend()
        plt.show()
        
        fig2 = plt.figure()
        ax = fig2.add_subplot(1,1,1)
        ax.plot(VofT,Z,'.k',label='orig',ms=1)
        ax.plot(V_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('V')
        plt.ylabel('Z')
        ax.legend()
        plt.show()
        
        fig3 = plt.figure()
        ax = fig3.add_subplot(1,1,1)
        ax.plot(WofT,Z,'.k',label='orig',ms=1)
        ax.plot(W_new,Z,'.r',label='scaled',ms=1)
        plt.xlabel('W')
        plt.ylabel('Z')
        ax.legend()
        plt.show()

    return U_new, V_new, W_new

caseDir = os.getcwd()
# caseDir = 'D:/tempData_depot/simData_CandC/ttuPSpcOP15.7'

dictFile = [line.split() for line in open(caseDir+'/system/scaleInflowDict')]
for d in dictFile:
    if d == []:
        continue
    elif d[0] == 'inflowDir':
        inflowDir = d[1]
    elif d[0] == 'inflowFormat':
        inflowFormat = d[1]
    elif d[0] == 'outputDir':
        outputDir = d[1]
    elif d[0] == 'targProfFile':
        targProfFile = d[1]
    elif d[0] == 'origProfFile':
        origProfFile = d[1]
    elif d[0] == 'dt':
        dt = float(d[1])
    elif d[0] == 'precision':
        p = int(d[1])
    elif d[0] == 'lengthScale':
        lScl = float(d[1])
    elif d[0] == 'timeScale':
        tScl = float(d[1])
    elif d[0] == 'scaleBy':
        scaleBy = d[1:]
    else:
        continue


origProf = readProfiles(origProfFile, scaleBy)
targProf = readProfiles(targProfFile, scaleBy)
origProf.Z = origProf.Z*lScl
profSclFctr = getProfileScaleFactor(origProf,targProf,scaleBy)

origUofZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.U),fill_value='extrapolate')
origIuOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iu),fill_value='extrapolate')
origIvOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iv),fill_value='extrapolate')
origIwOfZ = scintrp.interp1d(np.asarray(origProf.Z), np.asarray(origProf.Iw),bounds_error=False,fill_value=(origProf.Iw[0],origProf.iloc[-1].Iw))

fUofZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.U),fill_value='extrapolate')
fIuOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iu),fill_value='extrapolate')
fIvOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iv),fill_value='extrapolate')
fIwOfZ = scintrp.interp1d(np.asarray(profSclFctr.Z), np.asarray(profSclFctr.Iw),bounds_error=False,fill_value=(profSclFctr.Iw[0],profSclFctr.iloc[-1].Iw))

if inflowFormat == 'boundaryData':
    ptsFile = inflowDir + '/inlet/points'
    points = readBDformattedFile(ptsFile)
    points.columns = ['X','Y','Z']
    
    inflowDir = inflowDir+'/inlet'
    times = [ name for name in os.listdir(inflowDir) if os.path.isdir(os.path.join(inflowDir, name)) ]
    times = sorted(list(zip(times, np.asarray(times).astype(float))), key=lambda x: x[1])
    
    if not os.path.exists(outputDir+'/inlet'):
        os.makedirs(outputDir+'/inlet')
    file = outputDir+'/inlet/points'
    vect = np.transpose(np.asarray((points.X, points.Y, points.Z)))
    writeBDformattedFile(file,vect)

    for t in times:
        tNew = str(round(t[1]*tScl,__precision))
        if os.path.exists(outputDir+'/inlet/'+tNew):
            print("--- skipping \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)
            continue
        else:
            os.makedirs(outputDir+'/inlet/'+tNew)
        velFile = inflowDir+'/'+t[0]+'/U'
        if not os.path.exists(velFile):
            print("--- File not found! " + velFile)
            continue
        vel = readBDformattedFile(velFile)
        vel.columns = ['u','v','w']
        
        U,V,W = scaleVelocity(vel.u, vel.v, vel.w,
                                points.Z, scaleBy,
                                origUofZ, fUofZ, fIuOfZ, fIvOfZ, fIwOfZ)
        file = outputDir+'/inlet/'+tNew+'/U'
        vect = np.transpose(np.asarray((U, V, W)))
        writeBDformattedFile(file,vect)
        
        print("Scaling \t t-old = "+t[0]+"\t\t--> t-new = "+tNew)

    
elif inflowFormat == 'vtk':
    raise NotImplementedError("vtk inflow format not implemented")
else:
    raise NotImplementedError("Unknown inflow data format.")

inflowDictFile = open(caseDir+'/constant/boundaryData/inflowDict', 'w')
inflowDictFile.write("""/*--------------------------------*- C++ -*----------------------------------*
| =========                |                                                 |
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration     | Version:  4.1                                   |
|   \  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

""")
inflowDictFile.write("maxInflowTime\t\t"+str(tNew)+";\n\n")
inflowDictFile.write("// ************************************************************************* //\n")
inflowDictFile.close()
