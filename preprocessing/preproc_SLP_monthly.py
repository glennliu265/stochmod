#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

preproc_SLP_monthly.py

Preprocessing script that reads in PSL (SLP) data for each of the 42 ensemble 
members. The data is cut to 1920-2005 and the time and latitude dimensions are
corrected (roundoff difference in ens35-42). All unnecessary variables are also
cleaned from the file.

Monthly anomalies are computed from the data, and the ensemble mean is removed.
Data is saved separately for each ensemble member as nc files at [outpath] 
with the names - [SLP_ens%03d.nc].

Note: thus script requires a CESM1_LATLON.mat file that contains the lat/lon
values for CESM1LE. The path is indicated in the preprocess variable, but
this can be changed. This is to correct the aforementioned error in decimal
places for ens35-42.

Created on Fri Aug 21 13:33:38 2020

@author: gliu

"""


import xarray as xr
import time
import glob
import numpy as np

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model")
from amv import proc

#%% # Functions

# Preprocessing function (need to edit to make it work within open_mfdataset)
def preprocess(ds,varkeep):
    """correct time dimension and drop unwanted variables"""
    
    # Correct time dimension to start in Jan rather than Feb
    if ds.time.values[0].month != 1:
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
        print("\tCorrected Start to: %s; End to: %s" %  (str(ds.time.values[0]),str(ds.time.values[-1])))

    
    # Load proper latitude to accomodate different indexing
    from scipy.io import loadmat
    lat = np.squeeze(loadmat("/home/glliu/01_Data/CESM1_LATLON.mat")['LAT'])
    if np.any(~(lat == ds.lat.values)):
        ds = ds.assign_coords(lat=lat)
        print("\tReassigning latitude values ")
    
    # Drop variables unless it is in "varkeep"
    dsvars = list(ds.variables)
    varrem = [i for i in dsvars if i not in varkeep]
    ds = ds.drop(varrem)
    
    return ds

def xrdeseason(ds):
    """ Remove seasonal cycle..."""
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')

#%% User Edits

# Outputpath
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/SLP/"

#Path to SLP Data and glob expression
varname = 'PSL'
datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/%s/" % varname
ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.%s.*.nc" % varname

# Variables to keep
varkeep = ['PSL','lon','lat','time','ensemble'] # Variables to Keep

# Ensemble numbers
mnum = np.concatenate([np.arange(1,36),np.arange(101,108,1)])

#%% Read in raw SLP Data
allstart = time.time()
print("Now running preproc_SLP_monthly, time=%.2f"%allstart)
print("\tData will be read from %s"%(datpath))
print("\tData will be saved to %s" %(outpath))

# Get list of ncfiles
globby = datpath+ncsearch
nclist =glob.glob(globby)
nclist = [x for x in nclist if "OIC" not in x]
nclist.sort()

# Concatenate Data
cstart = time.time()
# Read in and concatenate each file
for e in range(len(nclist)):
    startloop = time.time()
    psl = xr.open_dataset(nclist[e])
    psl = preprocess(psl,varkeep)
    if e == 0:
        pslall = psl.copy()
    else:
        pslall = xr.concat([pslall,psl],dim="ensemble")
    print("Completed ensemble %i of 42 in %.2fs" % (e+1,time.time()-startloop))
print("Completed concatenation in %.2fs" % (time.time()-cstart))



#% Select time period after 1920-01-01 and remove seasonal and ensemble mean [2m8s]
dsna   = pslall.sel(time=slice('1920-01-01','2005-12-31'))
dsna2 = xrdeseason(dsna)
#dsna2 = dsna2 - dsna2.mean('ensemble')



# Save data
cstart = time.time()
for e in range(len(nclist)):
    
    ensnum = mnum[e]
    
    start = time.time()
    
    # Get data for ensemble member
    ds = dsna2.isel(ensemble=e)
         
    # Save member to netcdf
    ds.to_netcdf("%sSLP_ens%03d.nc"%(outpath,ensnum))

    print("Saved ensemble # %03d in %fs" % (ensnum,time.time()-start))
print("Saved data in %.2fs" % (time.time()-cstart))
print("SLP preprocessing completed in %.2fs and saved to %s" % (time.time()-allstart,outpath))