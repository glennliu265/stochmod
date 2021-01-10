#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

calc_DJFM_mean

Calculates the DJFM mean and cuts the specified region
for NHFLX and SLP data that has been preprocessed by:
preproc_NHFLX_monthly.py
preproc_SLP_monthly.py

Created on Fri Aug 21 14:19:07 2020

@author: gliu
"""

import numpy as np
import xarray as xr
import time

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model")
from amv import proc
import scm
#%% Functions

    
    
    



#%% User Edits
varnames = ('NHFLX','SLP')

# Ensemble numbers
mnum = np.concatenate([np.arange(1,36),np.arange(101,108,1)])

# Cut to region
bbox = 1 # Set to 1 to specify region to cut to
lonW = -90
lonE = 20
latS = 20
latN = 75
djfm = [12,1,2,3] # Seasons to keep

lonname = 'lon'
latname = 'lat'

#%%
allstart = time.time()

for v in range(len(varnames)):
    vstart = time.time()
    # Get variable name and path
    vn = varnames[v]
    datpath  =  "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % vn
    
    # Create list of variables
    nclist = ["%s%s_ens%03d.nc" % (datpath,vn,e) for e in mnum]
    
    # Open dataset
    ds = xr.open_mfdataset(nclist,
                           concat_dim='ensemble',
                           combine='nested',
                           compat='identical', # seems to be strictest setting...not sure if necessary
                           parallel="True",
                           join="exact" # another strict selection...
                           )
    
    # Add ensemble as a dimension
    ds = ds.assign_coords({'ensemble':np.arange(1,len(mnum)+1,1)})
    
    # Take the DJFM Mean
    season = ds.sel(time=np.in1d(ds['time.month'],djfm))
    dsw = season.groupby('time.year').mean('time')
    
    # Cut to specific region (assumes CESM data is in degrees East)
    if bbox == 1:
        # Convert to degrees East
        if lonW < 0:
            lonW += 360
        if lonE < 0:
            lonE += 360
        
        # Select North Atlantic Region for NAO Calculation...
        if lonW > lonE: # Cases crossing the prime meridian
            #print("Crossing Prime Meridian!")
            dsna = dsw.where((dsw[lonname]>=lonW) | (dsw[lonname]<=lonE),drop=True).sel(lat=slice(latS,latN))
        else:
            dsna = dsw.sel(lon=slice(lonW,lonE),lat=slice(latS,latN))
    else:
        dsna = dsw.copy()
            
        
    # Save to netcdf
    dsna.to_netcdf("%s%s_DJFMavg_bbox%i"%(datpath,vn,bbox))
    print("%s Completed in %.2f" % (time.time()-vstart))
print("calc_DJFmean Completed in %.2f" % (time.time()-allstart))    
    
    