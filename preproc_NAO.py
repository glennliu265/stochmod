#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:31:57 2020

@author: gliu
"""

import xarray as xr
import time
import glob
import numpy as np

start = time.time()

# Set variable names
varname = 'PSL'
lonname = "lon"
latname = "lat"
timename = "time"
levelname = "lev"
varkeep = [varname,lonname,latname,timename,levelname]

# NAO Calculation Settings
bbox = 1 # Set to 1 to specify region to cut to
lonW = -90
lonE = 40
latS = 20
latN = 80
djfm = [12,1,2,3] # Seasons to keep

# File names
datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/%s/" % varname
ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.PSL.*.nc"
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NAO_Forcing_DataProc"

# --
globby = datpath+ncsearch
nclist =glob.glob(globby)

nclist = [x for x in nclist if "OIC" not in x]
nclist.sort()

i = 0
for nc in nclist:
    startloop = time.time()
    # Open datasets        
    ds = xr.open_dataset(nc)
    
    # Correct time issue (model should start on January, but time is expressed as "days since...")
    if ds.time.values[0].month != 1:
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
        print("\tCorrected Start to: %s; End to: %s" %  (str(ds.time.values[0]),str(ds.time.values[-1])))
    
    
    # Drop variables unless it is in "varkeep"
    dsvars = list(ds.variables)
    varrem = [i for i in dsvars if i not in varkeep]
    ds = ds.drop(varrem)
    
    
    #%% Slice to region, vertical level, and time (CESM1-LE Specific)
    
    if bbox == 1:
        # Convert to degrees East
        if lonW < 0:
            lonW += 360
        if lonE < 0:
            lonE += 360
        
        # Select North Atlantic Region for NAO Calculation...
        if lonW > lonE: # Cases crossing the prime meridian
            #print("Crossing Prime Meridian!")
            dsna = ds.where((ds[lonname]>=lonW) | (ds[lonname]<=lonE),drop=True).sel(lat=slice(latS,latN))
        else:
            dsna = ds.sel(lon=slice(lonW,lonE),lat=slice(latS,latN))
    else:
        dsna = ds
    
    #%% Further slice for latitude and bottom pressure level
    dsna = dsna.isel(lev=-1)
    
    #%% Select time period after 1920-01-01
    dsna = dsna.sel(time=slice('1920-01-01','2005-12-31'))
    
    #%% select djfm
    season = dsna.sel(time= np.in1d(dsna['time.month'],djfm))
    
    # take monthly anomaly
    dsm = season.groupby('time.month') - season.groupby('time.month').mean('time') # Calculate monthly anomalies
    
    # take winter average
    dswm = dsm.groupby('time.year').mean('time')
    
    
    # Just copy for first iteration to make container dataset
    if i == 0:
        dsall = dswm.copy()
    else:
        
        # Small roundoff error in latitude...
        if np.any(~(dswm.lat.values == dsall.lat.values)):
            dswm = dswm.assign_coords(lat=dsall.lat.values)
            print("\tReassigning latitude values for ENS%i"% (i+1))
        dsall = xr.concat([dsall,dswm],dim="ensemble")
    
    i += 1
    print("Finished concatenating ensemble %i of %i in %.2fs" % (i,len(nclist),time.time()-startloop))
    if np.any(np.isnan(dsall.PSL.values)):
        print("Error on %i"%i)
        break

    
    
# Save output
outpath = '/home/glliu/01_Data/'
outname = varname+'_NAOproc.nc'
dsall.to_netcdf(outpath+outname)
elapsed = time.time() - start
tprint = "Saved data in  %.2fs" % (elapsed)
print(tprint)