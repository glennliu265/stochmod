#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Monthly Anomalies from CESM (PreIndustrial Control Runs)

Uses xarray to load in data on stormtrack. Calculates mean seasonal cycle, then
subtracts it from the data. Saves both as netCDF [time x lat x lon]

This is the script version of calc_CESM_anom.ipynb

Created on Fri Sep 24 14:42:21 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import glob
import time
import os
import sys

from tqdm import tqdm

# Import custom packages
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
from amv import proc

#%% Some Functions
def print_time(line=""):
    global st # Grab global time
    print("%sCompleted in %.2fs" % (line,time.time()-st))
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

def preprocess(ds):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist] # Takes varlist from global...
    ds = ds.drop(remvar)
    # Select the ground level
    ds = ds.isel(lev=-1)
    return ds



def start():
    return time.time()
#%% User Edits

# Saving location
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/TS/"

# Variable selection
mconfig   = "FULL"
vname     = "wind"
debug     = True
varlist   = [vname,'time','lat','lon','lev']

# Loading Method
use_loop  = False # Set true to loop and load individually
# For CESM-SLAB, TS the time differences were quite negligible:
# open_mfdataset() : 3m 23s
# loop             : 3m 25s
#%% Main Script

print("Loading %s from PIC-%s" % (vname,mconfig))

# Get list of netCDF files
nclist    = load_PIC(mconfig,vname,debug=debug,return_nc=True)

if use_loop:
    # Open file by file
    st = start()
    tsall = []
    for i in tqdm(range(len(nclist))):
        ds = xr.open_dataset(nclist[i])
        ds = preprocess(ds)
        tsall.append(ds[vname].values)
    
    # Concantenate
    ts = np.concatenate(tsall,axis=0)
    print_time(line="Loaded by looping ")
    
    
    
else: 
    
    # Open all at once
    st = start()
    ds = xr.open_mfdataset(nclist,concat_dim='time',preprocess=preprocess,parallel=True)
    print_time(line="Opened DS ")
    
    # Load in values
    st = start()
    ts    = ds.TS.values
    lon   = ds.lon.values
    lat   = ds.lat.values
    times = ds.time.values
    print_time(line="Loaded into ndarrays ")
    
# Calculate Climatology
st = start()
ntime,nlat,nlon = ts.shape
nyr             = int(ntime/12)
climts,tsmon = proc.calc_clim(ts,0,returnts=True)
print(tsmon.shape,climts.shape)
print_time(line="Calculated Climatology ")

# Calculate Anomaly
st = start()
tsanom = tsmon - climts[None,:,:,:]
tsanom = tsanom.reshape(ntime,nlat,nlon)
print(tsanom.shape)
print_time(line="Calculated Anomaly ")

# Create DataArrays
da_tsanom = proc.numpy_to_da(tsanom,times,lat,lon,vname,savenetcdf=None)
da_tsclim = proc.numpy_to_da(climts,np.arange(1,13,1),lat,lon,vname+"_clim",savenetcdf=None)

# Save Anomaly NetCDF
st = start()
savename = "%s%s_anom_PIC_%s.nc" % (outpath,vname,mconfig)
da_tsanom.to_netcdf(savename,
         encoding={vname: {'zlib': True,'_FillValue':-2147483647,"scale_factor":0.0001,'dtype':int}
                  })
print_time(line="Saved to %s" % savename)

# Save Clim NetCDF
st = start()
savename = "%s%s_clim_PIC_%s.nc" % (outpath,vname,mconfig)
da_tsclim.to_netcdf(savename,
         encoding={vname+"_clim": {'zlib': True,'_FillValue':-2147483647,"scale_factor":0.0001,'dtype':int}
                  })
print_time(line="Saved to %s " % savename)

# Note: for comparisons of compression/precision loss, please use the ipython notebook...