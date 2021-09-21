#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 20:18:29 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time
import os
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
from amv import proc

#%%
# User Edits 
outpath    = "/home/glliu/02_Figures/01_WeeklyMeetings/20210114/"
outdatpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/wind/"

# Load Lat lon
lon = np.load("/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon360.npy")
lat = np.load("/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy")


# Load season data for all points
vname   = "V"
units   = "m/s"
mconfig = "SLAB"
lonf    = None
latf    = None

#%% # Functions are taken from Investigate_Forcing.ipynb from notebooks folder on stormtrack

# Define preprocessing variable
def load_PIC(mconfig,vname,varpath=False,debug=False,return_nc=False):
    """
    NOTE: Currently only supports atmospheric variables on stormtrack...
    """
    # Create list of ncfiles depending on inputs
    if ~varpath: # Set default variable path on stormtrack
        varpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/"
    ncpath = '%satm/proc/tseries/monthly/%s/' % (varpath,vname)
    if mconfig == 'SLAB':
        ncsearch =  'e.e11.E1850C5CN.f09_g16.001.cam.h0.%s.*.nc' % vname
    elif mconfig == 'FULL':
        ncsearch =  'b.e11.B1850C5CN.f09_g16.005.cam.h0.%s.*.nc' % vname
    else:
        print("ERROR: Set mconfig to <SLAB> or <FULL>")
        return 0
    nclist = glob.glob(ncpath+ncsearch)
    nclist.sort()
    if debug:
        print("Found %i files, from \n %s to... \n %s" % (len(nclist),nclist[0],nclist[-1]))
    if len(nclist) == 0:
        print("ERROR! No Files Found")
        return 0
    
    if return_nc:
        return nclist
    
    # Open dataset
    dsall = xr.open_mfdataset(nclist,concat_dim='time',preprocess=preprocess)
    return dsall

    
def fix_febstart(ds):
    if ds.time.values[0].month != 1:
        print("Warning, first month is %s"% ds.time.values[0])
        # Get starting year, must be "YYYY"
        startyr = str(ds.time.values[0].year)
        while len(startyr) < 4:
            startyr = '0' + startyr
        nmon = ds.time.shape[0] # Get number of months
        # Corrected Time
        correctedtime = xr.cftime_range(start=startyr,periods=nmon,freq="MS",calendar="noleap")
        ds = ds.assign_coords(time=correctedtime) 
    return ds
    
#%%time
#varout  = retrieve_seasonal_cycle(vname,mconfig,lonf,latf,outpath,deseason=deseason,debug=True,allpoints=True)
#dsout   = load_PIC(mconfig,vname)

# Create preprocessing function
varkeep  = [vname,'time','lat','lon','lev'] 
def preprocess(ds,varlist=varkeep):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)
    # Select the ground level
    ds = ds.isel(lev=-1)
    return ds

nclist = load_PIC(mconfig,vname,return_nc=True)



#%%

for i in tqdm(range(len(nclist))):
    
    # Read in the variables
    ds = xr.open_dataset(nclist[i])
    ds = preprocess(ds)
    ds[vname]
    
    # For first iteration, copy
    var_i = ds[vname].values
    time_i = ds['time'].values
    if i == 0:
        var_out  = var_i.copy()
        time_out = time_i.copy()
        lon = ds['lon'].values
        lat = ds['lat'].values
    else: # Other iterations, concatenate
        var_out  = np.concatenate([var_out,var_i],axis=0) # Stack variable along time dimension
        time_out = np.hstack([time_out,time_i]) # Stack time along first dimension
        

