#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute the net heat flux variable for cloud locking experiments

Qnet = FSNS - (FLNS + LHFLX + SHFLX)
- Copied sections from compute_spectra_cloud_locking

Created on Mon Jul 10 10:23:01 2023

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
varnames = ["FSNS","FLNS","SHFLX","LHFLX"]

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

# 1. Load the variables of interest
# ----------------------------------------------------
ds_all = []
for v in range(4):
    varname = varnames[v]
    ncname  = "E1850C5CN.cld.2hr.1yrrpt.qfluxadj.latvary.cam.h0.%s.000101-040005.nc" % varname
    ds_var  = xr.open_dataset(datpath+ncname)
    ds_var = ds_var.isel(ilev=-1)[varname].load() # Select surface level
    ds_all.append(ds_var)

# 2. Sum. Rename. Save.
# ----------------------------------------------------
for v in range(4):
    varname = varnames[v]
    if (varname == "FSNS") and (v == 0):
        ds_nhflx = ds_all[v]
    else:
        ds_nhflx = ds_nhflx - ds_all[v]
    
    
# Do a check
lonf=330
latf=50
chk_values = [ds.sel(lon=lonf,lat=latf,method='nearest').isel(time=22).values.item() for ds in [ds_nhflx,ds_all[0],ds_all[1],ds_all[2],ds_all[3]]]
print("%.2f = %.2f - (%.2f + %.2f + %.2f)" % tuple(chk_values))


# Rename and save
# ---------
ds_nhflx = ds_nhflx.to_dataset(name="NHFLX") 
if debug:
    ds_nhflx.mean('time').NHFLX.plot()
ncname  = "%sE1850C5CN.cld.2hr.1yrrpt.qfluxadj.latvary.cam.h0.NHFLX.000101-040005.nc" % datpath
ds_nhflx.to_netcdf(ncname,
         encoding={"NHFLX": {'zlib': True}})
print("Saving netCDF to %s."% (ncname))
