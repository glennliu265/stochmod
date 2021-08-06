#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working/Test Script to calc_kprev, but interannual varying mixed layer depth

Created on Tue Jul 13 09:19:46 2021

@author: gyl
"""

import numpy as np
import xarray as xr
import tqdm

import matplotlib.pyplot as plt


#%% User Edits
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
fname   = "HMXL_CESM1-FULL-PIC_lon-30_lat50.npy"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210715/"


entrain1 = -1 # Value to assign for the first mixing
entrain0 = 0  # Value when there is no entrainment

debug = True

#%% Read in the File

h     = np.load(datpath+fname)/100
ntime = h.shape[0]
hmon  = h.reshape(int(ntime/12),12) 

plt.plot(hmon.mean(0))

#%%

# Test
#dt = h/np.roll(h,-1)



# Looping for time, consider the mixed layer's progression
kprev = np.zeros(ntime)*np.nan
for t in tqdm.tqdm(range(ntime)):
    
    # Wrap around for end value
    if t >= (ntime-1):
        dt = h[t]/h[0]
    else: # Forward step comparison 
        dt = h[t]/h[t+1]
    
    # Skip points where the mixed layer is detraining or unchanging
    if dt >= 1:
        kprev[t] = entrain0
        continue
    
    
    # Find the last index where h had the same value
    hdiff = h - h[t]
    hdiff = hdiff[:t] # Restrict to values before current timestep
    kgreat = np.where(hdiff > 0)[0] # Find values deeper than current MLD
    if len(kgreat) == 0:  # If no values are found, assume first time entraining to this depth
        kprev[t] = entrain1
        continue
    else:
        kd  = kgreat[-1] # Take index of most recent value
        # Linear interpolate to approximate index
        kfind = np.interp(h[t],[h[kd],h[kd+1]][::-1],[kd,kd+1][::-1])
        
        if kfind == float(t):
            kprev[t] = entrain1
        else:
            kprev[t] = kfind
            
            
            
#%% Make into function
def calc_kprev_lin(h,entrain1=-1,entrain0=0):
    """
    Estimate detrainment indices given a timeseries of mixed-layer
    depths. Uses/Assumptions:
        - Linear interpolation
        - Forward direction for entrainment/detrainment (h[t], h[t+1])
        - For last value, assumes t+1 is the first value (periodic)
    
    Inputs
    ------
        h : ARRAY [time]
            Timeseries of mixed layer depth
        
        --- Optional ---
        
        entrain1 : Numeric (Default = -1)
            Placeholder value for first time MLD reaches a given depth
        entrain0 : Numeric (Default = 0)
            Placeholder value for detraining months
    Output
    ------
        kprev : ARRAY [time]
            Indices where detrainment occurred    
    """
    # Preallocate, get dimensions
    ntime = h.shape[0]
    kprev = np.zeros(ntime)*np.nan
    
    # Looping for each step, get index of previous step
    for t in range(ntime):
        
        # Wrap around for end value
        if t >= (ntime-1):
            dt = h[t]/h[0]
        else: # Forward step comparison 
            dt = h[t]/h[t+1]
        
        # Skip points where the mixed layer is detraining or unchanging
        if dt >= 1:
            kprev[t] = entrain0
            continue
        
        
        # Find the last index where h had the same value
        hdiff = h - h[t]
        hdiff = hdiff[:t] # Restrict to values before current timestep
        kgreat = np.where(hdiff > 0)[0] # Find values deeper than current MLD
        if len(kgreat) == 0:  # If no values are found, assume first time entraining to this depth
            kprev[t] = entrain1
            continue
        else:
            kd  = kgreat[-1] # Take index of most recent value
            # Linear interpolate to approximate index
            kfind = np.interp(h[t],[h[kd],h[kd+1]][::-1],[kd,kd+1][::-1])
            
            if kfind == float(t):
                kprev[t] = entrain1
            else:
                kprev[t] = kfind
        # End Loop
    return kprev

kprev = calc_kprev_lin(h)
        



#%% Plot sample selection
k = 190 # Starting Index
prange = 50 # Range to plot
plotint = np.arange(k,k+prange,1) # Indices

# Create connector lines
connex = [((im+1,kprev[im]+1),(h[im],h[im])) for im in plotint if kprev[im] > 0]

# Initialize Figure
fig,ax = plt.subplots(1,1,figsize=(14,4))

#kmax = kprev == entrain1
#kdetrain = kprev == entrain0
#kentrain = kprev > 0

# Plot MLD
ax.plot(plotint+1,h[plotint],marker="x",label="MLD")
ax.legend()
#ax.scatter((plotint+1)[kentrain],h[kentrain],marker="x",color="r")
#ax.scatter((plotint+1)[kmax],h[kmax],marker="x",color="r")

# Plot connector lines
[ax.plot(connex[m][0],connex[m][1],label="") for m in range(len(connex))]
[ax.annotate("%.2f"%(connex[m][0][1]),(connex[m][0][1],connex[m][1][1])) for m in range(len(connex))]

# Labels
ax.set_ylabel("MLD (m)")
ax.set_ylabel("Index")
ax.set_title("Detrainment Month Calculation for t = %i to %i" % (plotint[0],plotint[-1]) )

