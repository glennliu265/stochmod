#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:49:05 20
@author: gliu21

"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time



# Stochmad Synth
h = np.array([450,500,400,300,
     200,100,100,100,
     100,100,250,400
     ])

kprev,hout = scm.find_kprev(h)
viz.viz_kprev(h,kprev)
print(kprev)

t  = 12
#%% Test with explicit printouts
temp_ts = np.array([
    -1,-2,-1,0,
    2,3,2,3,
    -1,2,1,0,
    ])


#plt.plot(np.arange(1,13,1),temp_ts)
# Here's a section to print verbose statements to test indexing for entrain
Td0 = 0
for t in np.arange(12,24,1):
    m  = (t+1)%12
    if m == 0:
        m = 12
    
    if kprev[m-1]==0:
        print("No entrainment on month %i"%m)
        temp_ts = np.concatenate((temp_ts,[0,]))
        continue
    print("t=%i, Month is %i"%(t,m))

    
    # Get information about the last month
    m0 = m - 1
    if m0 == 0:
        m0 = 12
    
    # Find # of months since the anomaly was formed
    k1m = (m - np.floor(kprev[m-1])) % 12
    k0m = (m - np.floor(kprev[m0-1])) % 12
    if k1m == 0:
        k1m = 12
    if k0m == 0:
        k0m = 12   

    
    # Get Index
    kp1 = int(t - k1m)
    kp0 = int(t - k0m)
    print("kprev is %.2f for month %i, or %i months ago at t=%i"% (kprev[m-1],m,k1m,kp1))
    print("kprev is %.2f for month %i, or %i months ago at t=%i"% (kprev[m0-1],m0,k0m,kp0))
    
    
    # Trying to get the value instead
    kval = kprev[m-1]-np.floor(kprev[m-1])
    
    # Interpolate
    Td1 = np.interp(kval,[0,1],[temp_ts[kp1],temp_ts[kp1+1]])
    #print("since %.2f is between %i and %i... "%(kprev[m-1],kp1,kp1+1))
    print("since %.2f is between %i and %i... "%(kval,0,1))
    print("\tTd1 is interpolated to %.2f, between %.2f and %.2f"%(Td1,temp_ts[kp1],temp_ts[kp1+1]))
    #print("this corresponds to t=%i, where T=%.2f"%(kp1,Td1))
    
    if kprev[m0-1]==0:#m0-1 == h.argmin():
        Td0 = Td1
        print("Since m0=%i, or first entraining month, Td=Td1"%(h.argmin()))
    elif Td0 == 0:
        kval = kprev[m0-1]-np.floor(kprev[m0-1])
        Td0 = np.interp(kval,[kp0,kp0+1],[temp_ts[kp0],temp_ts[kp0+1]])
        #print("since %.2f is between %i and %i... "%(kprev[m0-1],kp0,kp0+1))
        print("since %.2f is between %i and %i... "%(kval,0,1))
        print("\tTd0 is interpolated to %.2f, between %.2f and %.2f"%(Td0,temp_ts[kp0],temp_ts[kp0+1]))
                    
    Td = (Td1+Td0)/2
    print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0))       
    Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
    temp_ts = np.concatenate((temp_ts,[Td,]))
    print("--------------------------------\n")
    
#%% Rewrite Above as Function

def calc_Td(t,index,values,prevmon=False,debug=False):
    """

    Parameters
    ----------
    t : INT
        Timestep (in months, where t(0) = Jan)
    index : ARRAY [12]
        Time of entraining month
    values : ARRAY [t]
        Array of values to interpolate
    prevmon : BOOL, optional
        Set to True to calculate Td0. The default is False.
    debug : TYPE, optional
        Set to True to print outputs. The default is False.

    Returns
    -------
    Td: INT if prevmon=False, Tuple (Td1,Td0) if prevmon=False

    """
    
    # Initialize month array
    months = []
    m1  = (t+1)%12
    if m1 == 0:
        m1 = 12
    months.append(m1)
    
    # Option to include previous month
    if prevmon:
        m0 = m1-1
        if m0==0:
            m0 = 12
        months.append(m0)
    if debug:
        print("t=%i, Month is %i"%(t,m1))

    # Loop for each month
    Td = []
    for m in months:
        print("\tCalculating Td for m=%i"%m)
        
        # For m0, check if index=0 and skip if so (first entraining month)
        if (len(months)>1) and (m==months[-1]):
            if index[m-1] == 0:
                Td.append(Td[0])
                print("\t\tSince m0=%i, or first entraining month, Td0=Td1" % m)
                continue
        
        # Find # of months since the anomaly was formed
        k1m = (m1 - np.floor(index[m-1])) % 12
        if k1m == 0:
            k1m = 12
        
        # Get Index in t
        kp1 = int(t - k1m)
        if debug:
            print("\t\tkprev is %.2f for month %i, or %i months ago at t=%i"% (index[m-1],m,k1m,kp1))
        
        # Retrieve value between 0 and 1
        kval = index[m-1]-np.floor(index[m-1])
        
        # Interpolate
        Td1 = np.interp(kval,[0,1],[values[kp1],values[kp1+1]])
        if debug:
            print("\t\tsince %.2f is between %i and %i... "%(kval,0,1))
            print("\t\t\tTd is interpolated to %.2f, between %.2f and %.2f"%(Td1,values[kp1],values[kp1+1]))
        Td.append(Td1)
    
    if prevmon: # return Td=[Td1,Td0]
        return Td
    else: # Just return Td1
        return Td[0]

# Start Loop
temp_ts1 = np.array([
    -1,-2,-1,0,
    2,3,2,3,
    -1,2,1,0,

    ])
Td0 = None
for t in np.arange(12,24,1):
    m  = (t+1)%12
    if m == 0:
        m = 12
        
    # Skip if no entrainment
    if kprev[m-1]==0:
        print("No entrainment on month %i"%m)
        temp_ts1 = np.concatenate((temp_ts1,[0,]))
        Td0 = None
        continue
    
    if Td0 is None:
        Td1,Td0 = calc_Td(t,kprev,temp_ts1,prevmon=True,debug=True)
    else:
        Td1 = calc_Td(t,kprev,temp_ts1,prevmon=False,debug=True)
    
    Td = (Td1+Td0)/2
    
    print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0)) 
    Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
    temp_ts1 = np.concatenate((temp_ts1,[Td,]))
    print("--------------------------------\n")

fig,ax = plt.subplots(1,1)
ax.plot(temp_ts,label='Out')
ax.plot(temp_ts1,label='Func',linestyle='dashed')
ax.legend()
    
#%% Now test for a specific point
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'   

mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

bboxsim  = [-100,20,-20,90] # Simulation Box


nlon,nlat,nmon = mld.shape

# Reshape to 1d array
mld = mld.reshape(nlon*nlat,nmon)
kprevall = kprevall.reshape(nlon*nlat,nmon)
#plt.plot(mld[klon*klat,:])

# Load lat/lon
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
lon        = np.squeeze(loaddamp['LON1'])
lat        = np.squeeze(loaddamp['LAT'])

# Load Data
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month


# Load Latitude and Longitude
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
LON         = np.squeeze(loaddamp['LON1'])
LAT         = np.squeeze(loaddamp['LAT'])

# Load Damping
mconfig = "SLAB_PIC"
damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")

# Note: what is the second dimension for?
dampingr,lonr,latr = proc.sel_region(damping,LON,LAT,bboxsim)
hclim,_,_ = proc.sel_region(mld,LON,LAT,bboxsim)
kprev,_,_ = proc.sel_region(kprevall,LON,LAT,bboxsim)
    
klon,klat = proc.find_latlon(-30,50,lonr,latr)
h = hclim[klon,klat,:]
kprev,hout = scm.find_kprev(h)
viz.viz_kprev(h,kprev)
print(kprev)

#%% Test out point
# Start Loop
temp_ts1 = np.array([
    -1,-2,-1,0,
    2,3,2,3,
    -1,2,1,0,

    ])
Td0 = None
for t in np.arange(12,24,1):
    m  = (t+1)%12
    if m == 0:
        m = 12
        
    # Skip if no entrainment
    if kprev[m-1]==0:
        print("No entrainment on month %i"%m)
        temp_ts1 = np.concatenate((temp_ts1,[0,]))
        Td0 = None
        continue
    
    if Td0 is None:
        Td1,Td0 = calc_Td(t,kprev,temp_ts1,prevmon=True,debug=True)
    else:
        Td1 = calc_Td(t,kprev,temp_ts1,prevmon=False,debug=True)
    
    Td = (Td1+Td0)/2
    
    print("Td is %.2f, which is average of Td1=%.2f, Td0=%.2f"%(Td,Td1,Td0)) 
    Td0 = np.copy(Td1)# Copy Td1 to Td0 for the next loop
    temp_ts1 = np.concatenate((temp_ts1,[Td,]))
    print("--------------------------------\n")

fig,ax = plt.subplots(1,1)
ax.plot(temp_ts1,label='Func',linestyle='dashed')
ax.legend()                