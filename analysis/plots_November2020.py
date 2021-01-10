#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for the November 2020 Meeting

@author: gliu
"""



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

#%%
#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20201123_AMVTele/'
input_path  = datpath + 'model_input/'


# Plotting range
bbox = [260-360, 360-360, 0, 65]
cmbal = cmocean.cm.balance

#%% Load in Lat/LON
# Load damping variables (calculated in hfdamping matlab scripts...)
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
lon         = np.squeeze(loaddamp['LON1'])
lat         = np.squeeze(loaddamp['LAT'])
damping     = loaddamp['ensavg']


#%% Load in NAO/EAP EOF Pattern (SLP and NHFLX)

# Load in historical data
naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
naof_hist = naoforcing.mean(1).squeeze()
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO_corr.npz")
eofall    = npzdata['eofall']    # [ens x lat x lon x pc]
nao_hist = eofall.mean(0)
nao_hist = nao_hist.transpose(2,0,1) # [PC x Lat x Lon]

# Load in SLAB PIC data
npzslab = np.load(datpath+"EOF_NAO_DJFM_PIC_SLAB.npz")
naof_slab = npzslab['nhflx_pattern'] # [Lat x Lon x PC]
naof_slab = naof_slab.transpose(2,0,1)
nao_slab = npzslab['psl_pattern']
nao_slab = nao_slab.reshape(192,288,3)
nao_slab = nao_slab.transpose(2,0,1)

#%% Visualize NAO and EAP forcing for each case


fig,axs = plt.subplots(2,1,subplot_kw={"projection":ccrs.PlateCarree()})

ax = axs[0]
ax = viz.init_map(bbox,ax=ax)
pcm1 = ax.pcolormesh(lon,lat,naof_hist[0,:,:])

