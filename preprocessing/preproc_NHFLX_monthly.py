#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Aug 19 17:29:01 2020

Take raw heat flux data, compute anomalies, deforce and deseason, and apply land ice mask.
Ported over from [NHFLX_Regress_Monthly.ipynb]. Need to run to check...

Takes in raw NHFLX data from CESM1LE
    1. Restricts to specified lat/lon, and 1920 onwards
    2. Calculates monthly anomaly
    3. Remove ensemble mean


@author: gliu
"""
import xarray as xr
import time
import glob


import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
from amv import proc,viz

#%% Functions

def preprocess(ds,varkeep):
    """ 
    Preprocessing script to correct time issues and drop variables
    
    ds      : input xarray DataSet
    varkeep : list of variables to keep
    
    """
    
    
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
    
    return ds

def xrdeseason(ds):
    """ Remove seasonal cycle..."""
    return ds.groupby('time.month') - ds.groupby('time.month').mean('time')

#%% User Edits

# Path to data
datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NAO_Forcing_DataProc/"
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NHFLX/"

# Set variable names for nhflx extraction
varnames = ["FLNS","FSNS","SHFLX","LHFLX"]
lonname = "lon"
latname = "lat"
timename = "time"
levelname = "lev"
varkeep = [lonname,latname,timename,levelname] + varnames

# Some other options
bbox = 0 # Currently not slicing data
lonW = 0
lonE = 0
latS = 0
latN = 0

#%% Part 1: Preprocess NHFLX Data
allstart = time.time()

# Create list of nc files
nclists = {}
for varname in varnames:
    datpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/%s/" % varname
    ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.%s.*.nc" % varname

    # Get file names
    globby = datpath+ncsearch
    nclist =glob.glob(globby)
    
    # Eliminate OIC files and sort
    nclist = [x for x in nclist if "OIC" not in x]
    nclist.sort()
    
    # Store in Dictionary
    nclists[varname]=nclist
    print("Found %i files for %s" %(len(nclist),varname))
nens = len(nclists[varnames[0]])

# Loop through each member to calculate net heat flux 
i = 0
for e in range(nens):
    
    startloop = time.time()
    
    # Loop through each flux variable and sum
    VV = 0
    for var in varnames:
        
        # Pull dataset name
        ncname = nclists[var][e]
           
        # Open dataset      
        ds = xr.open_dataset(ncname)
    
        # Correct time and drop unneeded variables
        ds = preprocess(ds,varkeep)
        
        #% Slice to region, vertical level, and time (CESM1-LE Specific)
    
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
        
        #%% Get variable
        flux = dsna[var]
        if var == "FSNS":
            flux *= -1
        
        # For first iteration, rename variable to NHFLX
        if VV == 0:
            
            # Copy object to new dataset
            ds_nhflx = flux.to_dataset(name="NHFLX") 
        else:
            # Add object to new dataset (theoretically should sum along the variable )
            ds_nhflx += flux
        VV += 1
    
    # Rename for convenience
    dsna = ds_nhflx
    
    # Calculate monthly anomaly
    #dsna = dsna.groupby('time.month') - dsna.groupby('time.month').mean('time')
     
    # Just copy for first iteration to make container dataset
    if i == 0:
        dsall = dsna.copy()
    else:
        
        # Small roundoff error in latitude...
        if np.any(~(dsna.lat.values == dsall.lat.values)):
            dsna = dsna.assign_coords(lat=dsall.lat.values)
            print("\tReassigning latitude values for ENS%i"% (i+1))
        dsall = xr.concat([dsall,dsna],dim="ensemble")
    
    i += 1
    print("Finished concatenating ensemble %i of %i in %.2fs" % (i,len(nclist),time.time()-startloop))
    if np.any(np.isnan(dsall['NHFLX'].values)):
        print("Error on %i"%i)
        break
    
    
    
#%% Remove ensemble mean
# Add explicit ensemble coordinate
dsall.assign_coords({"ensemble": np.arange(1,43,1)})

# Calculate monthly anomalies
dsm = xrdeseason(dsall)

# Calculate Ensemble mean and save
ensmean = dsm.mean('ensemble')
ensmean.to_netcdf("%sNHFLX_ensemble_mean.nc"%(outpath))

# Remove Ensemble Mean
dsm = dsm - ensmean

# Apply landice mask and save output

mnum = np.concatenate([np.arange(1,36),np.arange(101,108,1)])
maskloc = '/home/glliu/01_Data/masks/'

for e in range(nens):
    start = time.time()
    
    # Get data for ensemble member
    ds = dsm.isel(ensemble=e)
     
    # Get Mask
    ensnum = mnum[e]
    maskmat = "%slandicemask_ensnum%03d.mat" % (maskloc,ensnum) # Lon x Lat
     
    # Apply Mask
    mask = loadmat(maskmat)['mask_landice'] 
    ds = ds * np.transpose(mask[None,:,:],(0,2,1))
    
    # Save member to netcdf
    ds.to_netcdf("%sNHFLX_ens%03d.nc"%(outpath,ensnum))

    print("Saved ensemble # %03d in %fs" % (ensnum,time.time()-start))
    
print("Saved NHFLX files to %s. Completed script in %.2fs" % (outpath,time.time()-allstart))