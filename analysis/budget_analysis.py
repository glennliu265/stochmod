#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Budget Analysis

Created on Mon Mar 28 16:58:22 2022

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

import sys


import cartopy.crs as ccrs

#%% User Edits

amvbbox = [-80,0,10,60]


#%%
# (0) Data Prep
# Load in data
dp = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
fn = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run200_ampq0_method5_dmp0_budget_Qek.npz"
ld = np.load(dp+fn,allow_pickle=True)

T = ld['sst'] # [Model x Lon x Lat x Time]
lon = ld['lon']
lat = ld['lat']



entrain_term = ld['entrain_term']
entrain_ann  = proc.ann_avg(entrain_term,2) 

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

sys.path.append("/Users/gliu/Downloads/06_School/06_Fall2021/12860/materials_2020/CVD_Tutorials/")
import cvd_utils as cvd

#%% Compute/Load the AMV Index
"""
5) awgt:  number to indicate weight type
            0 = no weighting
            1 = cos(lat)
            2 = sqrt(cos(lat))
            
"""

# Compute the AMV Index
amvid,amvpat = proc.calc_AMVquick(T.squeeze(),lon,lat,amvbbox,dropedge=5)

# Now get indices of positive.negative points
pks,_ = signal.find_peaks(np.abs(amvid))

# Separate into positive and negative
idneg = [pk for pk in pks if amvid[pk] <= 0]
idpos = [pk for pk in pks if amvid[pk] > 0]


zerocross = np.where(np.diff(np.sign(amvid)))[0]

#%% Plot the AMV Index
t      = np.arange(0,len(amvid))
fig,ax = plt.subplots(1,1,figsize=(16,4))

ax     = cvd.plot_anomaly(t,amvid,ax=ax)
ax.scatter(idneg,amvid[idneg],marker="d",color="darkblue")
ax.scatter(idpos,amvid[idpos],marker="x",color="darkred")
ax.scatter(zerocross,amvid[zerocross],marker="o",color='yellow')
ax.set_xlabel("Years")
ax.set_ylabel("AMV Index")

#%% Find Increasing/Decreasing Segments

_,nlon,nlat,ntime = T.shape



decr = np.zeros((nlon,nlat))*np.nan # Decrease
incr = np.zeros((nlon,nlat))*np.nan # Inrease


decr_ids = []
incr_ids = []
# Loop through each zero crossing
for zero in range(len(zerocross)):
    idzero = zerocross[zero]
    
    # Find nearest pos/neg peaks
    kpos   = idpos[np.abs(idpos-idzero).argmin()]
    kneg   = idneg[np.abs(idneg-idzero).argmin()]
    
    #print("Nearest crossings to %i are Neg: %i, Pos: %i"%(idzero,kneg,kpos))
    # Record intevals to corresponding array (Note: need to check if this is indexing properly)
    if kpos > kneg: # Increasing
        incr_ids.append(np.arange(kneg,kpos+1,1))
    elif kpos < kneg: # Decreasing
        decr_ids.append(np.arange(kpos,kneg+1,1))
        
        
        
n_incr = len(incr_ids)
n_decr = len(decr_ids)
print("Found %i increasing sections, %i decreasing sections."% (n_incr,n_decr))
    
#%% Integrate over the segments

invar = entrain_ann

decr = np.zeros((nlon,nlat,n_decr))*np.nan # Decrease
incr = np.zeros((nlon,nlat,n_incr))*np.nan # Inrease

# Integrate Decreasing values
for d in range(n_decr):
    
    ids_in      = decr_ids[d]
    decr[:,:,d] = invar[:,:,ids_in].sum(-1)
    
    
# Integrate increasing values
for d in range(n_incr):
    ids_in      = incr_ids[d]
    incr[:,:,d] = invar[:,:,ids_in].sum(-1)
    


#%% PLot mean values

plotvars = [decr,incr]
plotnames = ["Decreasing","Increasing"]

proj = ccrs.PlateCarree()
fig,axs = plt.subplots(1,2,figsize=(12,4),constrained_layout=True,
                      subplot_kw={'projection':proj})

for i in range(2):
    ax = axs[i]
    ax = viz.add_coast_grid(ax,bbox=amvbbox,fill_color='gray',ignore_error=True)
    
    plotvar = plotvars[i].mean(-1).T
    vm = 2 * np.nanstd(plotvar.flatten())
    
    
    pcm = ax.pcolormesh(lon,lat,plotvar,vmin=-vm,vmax=vm,cmap='cmo.balance')
    fig.colorbar(pcm,ax=ax)
    ax.set_title(plotnames[i])
    
    
    
    
#%%



p_done = []
step   = 1 # How many peaks ahead to search (default is 1)
for p in range(len(pks)-1): # Loop through each peak
    pk = pks[p]
    
    # Search until there is a sign change
    if np.sign(amvid[pk]) == np.sign(amvid[pk+1]):
        print("No sign changes betweens point %i and %i" % (pk,pk+1))
        same_sign  = True
        search_step  = 1
        while same_sign:
            same_sign = np.sign(amvid[pk]) == np.sign(amvid[pk+search_step])
            search_step += 1
            
            
            
            
            
            
            
        
        
        print("No sign change for index %i" % pk)
        step += 1 # Now search 2 steps ahead
        continue
    
    
    
    
    
    if amvid[pk] > amvid[pk+1]: # Decreasing
        do x
        
    elif amvid[pk] < amvid[pk+1]: # Increasing
        
    
    







#%%
# (1) Find index of peaks, identify if + or - AMV

# (2) Integrate terms leading up to the peak 

# (3) Take the average(?) to see the general contribution of each term during the
# buildup

