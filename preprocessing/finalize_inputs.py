#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalize Inputs into Stochastic Model

Created on Sat Nov 21 22:39:21 2020

@author: gliu
"""
#%%

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
from dask.distributed import Client,progress
import dask

import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature

import matplotlib.colors as mc

#%% Set Paths

#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20201123_AMVTele/'
input_path  = datpath + 'model_input/'

# Plotting range
cmbal = cmocean.cm.balance

# Settings
debug = True # Visualize images
bbox  = [260-360, 360-360, 0, 80]

# Load in Mask
msk = np.load(datpath+"landicemask_enssum.npy")


#%% Load some background data

# Load Longitude for processing
lon360 =  np.load(datpath+"CESM_lon360.npy")

# Load in latitude for plotting
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])

# Load in land/ice mask

#%% Finalize DJFM Forcing Patterns

# 1) Historical CESM1-LE Run, Forcing Setup (DJFM NAO and EAP) -----

# Load in the data
naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
naof_hist = naoforcing.mean(1).squeeze()
naof_hist = naof_hist.transpose(2,1,0)# --> [lon x lat x pc]
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO_corr.npz")
eofall    = npzdata['eofall']    # [ens x lat x lon x pc]
nao_hist = eofall.mean(0)
nao_hist = nao_hist.transpose(1,0,2) # --> [lon x lat x pc]
varexpall = npzdata['varexpall'] # [ens x pc]
varexp_hist = varexpall.mean(0)

# Apply Mask
naof_hist *= msk.T[:,:,None]

# Flip longitude
lon180,naof_hist = proc.lon360to180(lon360,naof_hist)
_,nao_hist = proc.lon360to180(lon360,nao_hist)

# Visualize Results
if debug:
    cint = np.arange(-50,55,5)
    fig,axs = plt.subplots(2,1,subplot_kw={"projection":ccrs.PlateCarree()})
    
    ax = axs[0]
    ax = viz.init_map(bbox,ax=ax)
    cf1 = ax.contourf(lon180,lat,naof_hist[:,:,0].T,levels=cint,cmap=cmbal)
    ax  = viz.plot_contoursign(naof_hist[:,:,0].T,lon180,lat,cint,ax=ax,clab=True,lw=0.5)
    
    fig.colorbar(cf1,ax=ax)
    ax.set_title("$NAO_{DJFM}$ (EOF 1, Variance Explained = %.2f)"% (varexp_hist[0]*100)+r"%")
    
    ax = axs[1]
    ax = viz.init_map(bbox,ax=ax)
    cf2 = ax.contourf(lon180,lat,naof_hist[:,:,1].T,levels=cint,cmap=cmbal)
    ax  = viz.plot_contoursign(naof_hist[:,:,1].T,lon180,lat,cint,ax=ax,clab=True,lw=0.5)
    fig.colorbar(cf2,ax=ax)
    ax.set_title("$EAP_{DJFM}$ (EOF 2, Variance Explained = %.2f)"% (varexp_hist[1]*100)+r"%")
    
    plt.suptitle("CESM1-LE (Historical 1920-2005, Ens. Avg.)",x=0.6,y=0.95)
    plt.savefig(outpath+"CESM_Historical_DJFM_Forcing.png",dpi=200,bbox_inches='tight')
    
# Save output
np.save("%sNAO_EAP_NHFLX_Forcing_DJFM_HIST.npy" % (input_path),naof_hist)
np.save("%sNAO_EAP_PSL_DJFM_HIST.npy" %(input_path),nao_hist)


# 2) SLAB CESM1-LE Run, Forcing Setup (DJFM NAO and EAP) -----

# Load in SLAB PIC data
npzslab = np.load(datpath+"EOF_NAO_DJFM_PIC_SLAB.npz")
naof_slab = npzslab['nhflx_pattern'] # [Lat x Lon x PC]
naof_slab = naof_slab.transpose(1,0,2) # [Lon x Lat x PC]
nao_slab = npzslab['psl_pattern']
nao_slab = nao_slab.reshape(192,288,3)
nao_slab = nao_slab.transpose(1,0,2)
varexp_slab = npzslab['varexp']

# Apply mask
naof_slab *= msk.T[:,:,None]

# Flip Longitude
lon180,naof_slab = proc.lon360to180(lon360,naof_slab)
_,nao_slab = proc.lon360to180(lon360,nao_slab)

# Visualize Results
if debug:
    cint = np.arange(-50,55,5)
    fig,axs = plt.subplots(2,1,subplot_kw={"projection":ccrs.PlateCarree()})
    
    ax = axs[0]
    ax = viz.init_map(bbox,ax=ax)
    cf1 = ax.contourf(lon180,lat,naof_slab[:,:,0].T,levels=cint,cmap=cmbal)
    ax  = viz.plot_contoursign(naof_slab[:,:,0].T,lon180,lat,cint,ax=ax,clab=True,lw=0.5)
    
    fig.colorbar(cf1,ax=ax)
    ax.set_title("$NAO_{DJFM}$ (EOF 1, Variance Explained = %.2f)"% (varexp_slab[0]*100)+r"%")
    
    ax = axs[1]
    ax = viz.init_map(bbox,ax=ax)
    cf2 = ax.contourf(lon180,lat,naof_slab[:,:,1].T,levels=cint,cmap=cmbal)
    ax  = viz.plot_contoursign(naof_slab[:,:,1].T,lon180,lat,cint,ax=ax,clab=True,lw=0.5)
    fig.colorbar(cf2,ax=ax)
    ax.set_title("$EAP_{DJFM}$ (EOF 2, Variance Explained = %.2f)"% (varexp_slab[1]*100)+r"%")
    
    plt.suptitle("CESM1-Slab (Preindustrial Control 101-1001)",x=0.6,y=0.95)
    plt.savefig(outpath+"CESM_SLAB_DJFM_Forcing.png",dpi=200,bbox_inches='tight')
    
# Save output
np.save("%sNAO_EAP_NHFLX_Forcing_DJFM_SLAB.npy" % (input_path),naof_slab)
np.save("%sNAO_EAP_PSL_DJFM_SLAB.npy" %(input_path),nao_slab)


#%% Finalize Heat Fluxes




#%% Finalize Mixed Layer Depths


