#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:40:44 2020

@author: gliu
"""


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
varnames = ["FLNS","FSNS","SHFLX","LHFLX"]
lonname = "lon"
latname = "lat"
timename = "time"
levelname = "lev"
varkeep = [lonname,latname,timename,levelname] + varnames

# NAO Calculation Settings
bbox = 0 # Set to 1 to specify region to cut to
lonW = -90
lonE = 40
latS = 20
latN = 80
djfm = [12,1,2,3] # Seasons to keep

# Set output path
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/NAO_Forcing_DataProc/"


# Create dictionary of file names to loop over
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
    



i = 0
for e in range(nens):
    
    startloop = time.time()
    
    # Loop through each variable and sum
    VV = 0
    for var in varnames:
        
        # Pull dataset name
        ncname = nclists[var][e]
           
        # Open dataset      
        ds = xr.open_dataset(ncname)
    
        # Correct time and drop unneeded variables
        ds = preprocess(ds,varkeep)
        
        
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
    if np.any(np.isnan(dsall['NHFLX'].values)):
        print("Error on %i"%i)
        break


# Save output
#outpath = '/home/glliu/01_Data/'
outname = 'NHFLX_NAOproc.nc'
dsall.to_netcdf(outpath+outname)
elapsed = time.time() - start
tprint = "Saved data in  %.2fs" % (elapsed)
print(tprint)