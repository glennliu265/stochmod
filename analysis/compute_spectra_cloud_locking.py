#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Spectra for Cloud Locking Experiments

Copied sections from make_landice_mask.py from predict_amv repo (2023.07.09)

Created on Sun Jul  9 20:08:37 2023

@author: gliu
"""

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cartopy.crs as ccrs
import xarray as xr

import time
import sys
import glob

import nitime.algorithms as tsa

# xxxxxxxxxxxx
#%% User Edits
# xxxxxxxxxxxx

# Set Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM1_Cloud_Locking/"
outpath = datpath+"proc/"

# Indicate variable
varname = "TS"

# Indicate Mask Settings
ice_thres       = 0.05                     # Mask out if grid ever exceeds this value
mask_sep        = True                     # Save separate masks (in addition to combined)
save_max        = True                     # Output max concentration for debugging


# Land-ice mask to preload (calculated by make_landice_mask.py from predict_amv)
liname = "CESM1LE_HTR_limask_pacificmask_enssum_lon-90to20_lat0to90.nc"
lipath = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/Masks/"

# Region to make calculations over
bbox            = [-90,20,0,90]

# Power Spectra Options
dt              = 3600*24*30
nsmooth         = 10

# Other Toggles
debug           = True  # Set to True to see debugging plots


# ======================
#%% Import some packages
# ======================
# Import my own custom modules....
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/amv/")
import viz,proc

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
import scm


# ------------------------------------------------
#%% Load existing land/ice mask and update with the icemask from cloud-locked simulation...
# ------------------------------------------------

# 1. Make the ice mask for cloud-locked simulation
# ------------------------------------------------
# Load ice variables, lat/lon/time
ds_ice  = xr.open_dataset(datpath+"E1850C5CN.cld.2hr.1yrrpt.qfluxadj.latvary.cam.h0.ICEFRAC.000101-040005.nc")
ds_ice  = ds_ice.isel(ilev=-1).ICEFRAC.load() # Select the lowest level (1000)
#ds_ice  = proc.lon360to180_xr(ds_ice)
#ds_ice  = ds_ice.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
lon,lat,time=ds_ice.lon.values,ds_ice.lat.values,ds_ice.time.values
ntime,nlat,nlon = ds_ice.shape
# Make mask
ds_ice_max = ds_ice.max('time')
ice_mask   = np.ones((nlat,nlon))
ice_mask[ds_ice_max.values > ice_thres] = np.nan
if debug:
    
    fig,axs = plt.subplots(2,1,subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True,figsize=(12,8))
    pcm1=axs[0].pcolormesh(lon,lat,ds_ice_max)
    fig.colorbar(pcm1,ax=axs[0],fraction=0.025)
    axs[0].set_title("Max Ice Concentration (Cloud-Locked Simulation)")
    pcm2=axs[1].pcolormesh(lon,lat,ice_mask)
    fig.colorbar(pcm2,ax=axs[1],fraction=0.025)
    axs[1].set_title("Ice Mask, %.2f Threshold (Cloud-Locked Simulation)" % ice_thres)





# 2. Load the variable of interest and apply ice mask
# ----------------------------------------------------
ncname = "E1850C5CN.cld.2hr.1yrrpt.qfluxadj.latvary.cam.h0.%s.000101-040005.nc" % varname
ds_var = xr.open_dataset(datpath+ncname)
ds_var = ds_var.isel(ilev=-1)[varname].load() # Select surface level
ds_var = ds_var * ice_mask # Apply ice mask
if debug:
    ds_var.mean('time').plot()
    
    
# 3. Slice to the region
# ----------------------------------------------------
ds_var  = proc.lon360to180_xr(ds_var)
ds_var  = ds_var.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

# 4. Load and apply an existing land/ice mask from CESM1 
# ----------------------------------------------------
ds_mask = xr.open_dataset(lipath+liname)
ds_var  = ds_var * ds_mask.MASK.values.squeeze()
if debug:
    ds_var.mean('time').plot()

#%% Data Preprocessing

# 4. Remove mean seasonal cycle
# ----------------------------------------------------
ds_manom       = proc.xrdeseason(ds_var)
var_manom      = ds_manom.values
lonr,latr      = ds_manom.lon.values,ds_manom.lat.values


# 5. Remove a linear trend
# ----------------------------------------------------
detrend_output = proc.detrend_dim(var_manom,0)
var_detrended  = detrend_output[0]


# 6. Debug by looking at a point and computing the spectra, determine smoothing amount
# ----------------------------------------------------
klon,klat      = proc.find_latlon(-30,50,lonr,latr)
ts_point       = var_detrended[:,klat,klon]

# Test out different smoothing across adjacent bands (Daniell window)
nsmooths       = [10,25,50,100,]
pct            = 0.10
sst_in         = [ts_point,]*len(nsmooths)
spec_est       = scm.quick_spectrum(sst_in,nsmooths,pct,return_dict=True)
freqs          = spec_est['freqs']
specs          = spec_est['specs']

# Log Log Plot
fig,ax    = plt.subplots(1,1)
for n in range(len(nsmooths)):
    ax.loglog(freqs[n][1:]*dt,specs[n]/dt,label="n_smooth=%i" % nsmooths[n])
ax.axvline([1/(12*10)],color="k",label="Decade",ls='dashed')
ax.axvline([1/(12)],color="gray",label="Annual",ls="solid")
ax.axvline([1/(12*75)],color="orange",label="75-year",ls="dotted")
ax.axvline([1/(12*100)],color="red",label="Centennial",ls="dotted")
ax.legend()
ax.set_xlim([1/1500,1/6])
ax.set_ylim([1e-1,1e2])
ax.set_xlabel("Frequency (cycles per month)")
ax.set_ylabel("Power")

# Linear Linear Plot
fig,ax    = plt.subplots(1,1)
for n in range(len(nsmooths)):
    ax.plot(freqs[n][1:]*dt,specs[n]/dt,label="n_smooth=%i" % nsmooths[n])
ax.axvline([1/(12*10)],color="k",label="Decade",ls='dashed')
ax.axvline([1/(12)],color="gray",label="Annual",ls="solid")
ax.axvline([1/(12*75)],color="orange",label="75-year",ls="dotted")
ax.axvline([1/(12*100)],color="red",label="Centennial",ls="dotted")
ax.legend()
ax.set_xlim([1/(12*100),1/(12*5)])
ax.set_ylim([1e-1,1e2])
ax.set_xlabel("Frequency (cycles per month)")
ax.set_ylabel("Power")

# 7. Compute the power spectra
# ----------------------------------------------------
nsmooth_select = 25
nfreqs         = len(specs[0]) 
nlonr,nlatr    = len(lonr),len(latr)
spec_est_all   = np.zeros((nfreqs,nlatr,nlonr)) * np.nan
for o in tqdm(range(nlonr)):
    for a in range(nlatr):
        ts_point       = var_detrended[:,a,o]
        if np.all(np.isnan(ts_point)):
            continue
        spec_est       = scm.quick_spectrum([ts_point,],[nsmooth_select,],pct,return_dict=True)
        spec_est_all[:,a,o] = spec_est['specs'][0]

# 8. Save it somewhere
# ----------------------------------------------------
out_dict = {
    "lon"       : lonr,
    "lat"       : latr,
    "specs"     : spec_est_all,
    "freq"      : spec_est['freqs'][0],
    "nsmooth"   : nsmooth_select,
    "taper_pct" : pct
    }
outname = "%sCloudLocking_%s_Spectra_nsmooth%i.npz" % (outpath,varname,nsmooth_select)
np.savez(outname,**out_dict,allow_pickle=True)

#%%






