#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget Analysis

Created on Mon Mar 28 16:58:22 2022

@author: gliu

"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from scipy import signal
from tqdm import tqdm
import sys
#%% Import Modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

sys.path.append("/Users/gliu/Downloads/06_School/06_Fall2021/12860/materials_2020/CVD_Tutorials/")
import cvd_utils as cvd

#%% User Edits

amvbbox = [-80,0,0,65]
def load_sm_terms(ld,vnames=None):
    
    # Get list of variables
    if vnames is None:
        vnames = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]
        
    # Load in the data, and take the annual averages
    for v,vname in tqdm(enumerate(vnames)):
        
        vld    = ld[vname].squeeze()
        vld_ann = proc.ann_avg(vld,2) 
        
        if v == 0:
            # Get Lat/Lon/Dimensions
            lon          = ld['lon']
            lat          = ld['lat']
            nlon,nlat,ntime = vld.shape
            nyr          = int(ntime/12)
            
            # Preallocate
            sm_vars = np.zeros((nlon,nlat,nyr,len(vnames)))*np.nan
        sm_vars[:,:,:,v] = vld_ann.copy()
        
        
        if vname == "sst":
            # Save and load separate copy
            T_ann = vld_ann.copy()
    
    return sm_vars,T_ann
#%%
# (0) Data Prep
# Load in data

vnames        = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]

dp            = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
#fn           = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run200_ampq0_method5_dmp0_budget_Qek.npz"
#ld           = np.load(dp+fn,allow_pickle=True)

sm_vars,T_ann = load_sm_terms(ld,vnames=vnames)

#%% Load for each one

fns = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02i_ampq0_method5_dmp0_budget_Qek.npz" % (s) for s in np.arange(0,10,1)]
sm_vars_all = []
T_all       = []
for f,fn in tqdm(enumerate(fns)):
    
    ld            = np.load(dp+fn,allow_pickle=True) 
    sm_vars,T_ann = load_sm_terms(ld,vnames=None)
    sm_vars_all.append(sm_vars)
    T_all.append(T_ann)



# Convert to 1-D
T_all       = np.array(T_all)       # [run x lon x lat x time]
sm_vars_all = np.array(sm_vars_all) # [run x lon x lat x time x variable]

nrun,nlon,nlat,nyr,nvar = sm_vars_all.shape()
#%% Transpose to proper dimensions and stack [lon x lat x RUN x TIME x var]
T_all       = T_all.transpose(1,2,0,3).reshape(nlon,nlat,nrun*nyr)
sm_vars_all = sm_vars_all.transpose(1,2,0,3,4).reshape(nlon,nlat,nrun*nyr,nvar)

"""
T_all = [lon x lat x year]
sm_vars_all = [lon x lat x year x run]
"""
#%%
#%% Old, non-functionized version for loading in variables

vnames = ["sst","damping_term","forcing_term","entrain_term","ekman_term"]


# Load in the data, and take the annual averages
for v,vname in tqdm(enumerate(vnames)):
    
    vld    = ld['vname'].squeeze()
    vldann = proc.ann_avg(vld,2) 
    
    if v == 0:
        # Get Lat/Lon/Dimensions
        lon          = ld['lon']
        lat          = ld['lat']
        nlon,nlat,ntime = vld.shape
        nyr          = int(ntime/12)
        
        # Preallocate
        sm_vars = np.zeros((nlon,nlat,nyr,len(vnames)))*np.nan
    sm_vars[:,:,:,v] = vann.copy()
    
    
    if vname == "sst":
        # Save and load separate copy
        T_ann = vld.copy()

entrain_term = ld['entrain_term'] # [Lon x Lat x Time]
entrain_ann  = proc.ann_avg(entrain_term,2) 

damping_term = ld['damping_term']
damping_ann  = proc.ann_avg(damping_term,2)

T            = ld['sst'] # [Model x Lon x Lat x Time]
T_ann        = proc.ann_avg(T.squeeze(),2)

Fprime       = ld['forcing_term']
Fprime_ann   = proc.ann_avg(Fprime,2)

Qek          = ld['ekman_term']
Qek_ann      = proc.ann_avg(Qek,2)
#%%



#%% Compute/Load the AMV Index
"""
5) awgt:  number to indicate weight type
            0 = no weighting
            1 = cos(lat)
            2 = sqrt(cos(lat))
            
"""

# Compute the AMV Index
amvid,amvpat = proc.calc_AMVquick(T_all.squeeze(),lon,lat,amvbbox,dropedge=5,anndata=True)

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
    
#%% Integrate over the segments (for a single variable)

invar  = sm_vars_all[:,:,:,0]#entrain_ann
#invar = damping_ann
#invar = T_ann
#invar  = Fprime_ann
#invar = Qek_ann

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
    
#%% Compute for each variable



invar  = sm_vars_all[:,:,:]#entrain_ann
#invar = damping_ann
#invar = T_ann
#invar = Fprime_ann
#invar = Qek_ann

decr = np.zeros((nlon,nlat,n_decr,nvar))*np.nan # Decrease
incr = np.zeros((nlon,nlat,n_incr,nvar))*np.nan # Inrease

for v in range(nvar):
    
    invar = sm_vars_all[:,:,:,v]
    
    # Integrate Decreasing values
    for d in range(n_decr):
        
        ids_in      = decr_ids[d]
        decr[:,:,d] = invar[:,:,ids_in].sum(-1)
        
        
    # Integrate increasing values
    for d in range(n_incr):
        ids_in      = incr_ids[d]
        incr[:,:,d] = invar[:,:,ids_in].sum(-1)
    


#%% Plot mean values

plotvars  = [decr,incr]
plotnames = ["Decreasing","Increasing"]

proj = ccrs.PlateCarree()
fig,axs = plt.subplots(1,2,figsize=(12,4),constrained_layout=True,
                      subplot_kw={'projection':proj})

for i in range(2):
    ax = axs[i]
    ax = viz.add_coast_grid(ax,bbox=amvbbox,fill_color='gray',ignore_error=True)
    
    plotvar = plotvars[i].mean(-1).T
    vm = 4 * np.nanstd(plotvar.flatten())
    
    
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

