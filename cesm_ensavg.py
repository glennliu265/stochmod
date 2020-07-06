#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:10:43 2020

Test script to take the climatological monthly mean for a dataset and output it.

@author: gliu
"""

import xarray as xr
import time
import glob

# # Note: change variable name in dataset
# def preprocess(ds):
    
#     # Copy array
#     if ds.time.shape[0] > 1032:
    
#         ds = ds.sel(time=slice('1920-01-01','2005-12-31'))
    
#     return ds
    

# Set variable name
varname = 'HMXL'

# Variables to keep
varlist = [varname,'TLONG','TLAT','time']

#%% Set paths and glob expression
ncpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/" + varname + "/"
ncnames = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h." + varname + ".*.nc"
globby = ncpath+ncnames


# Get list of variables for testing
nclist = glob.glob(globby)
nctest = nclist[0]

# Open first file to see if cftime decoded correctly
dstest = xr.open_dataset(nctest)
if dstest.time.values[0].month != 1:
    print("First Timestep is not January, but: %s" %  str(dstest.time.values[0]))
    montherr = 1
else:
    montherr = 0

#%% Open dataset


# Preprocess variable if it is applicable
def preprocess(ds):
    if ds.time.values[0].month != 1:
    
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
    return ds
        
        
# Open dataset with preprocessing
ds = xr.open_mfdataset(globby,
                       concat_dim="ensemble",
 #                      combine="by_coords",
                       preprocess=preprocess,
                       parallel=True)

#Correct time variable if applicable (might be not needed,can remove later)
if montherr != 1:
    startyr = str(ds.time.values[0].year)
    correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
    ds['time'] = correctedtime
    ds.assign_coords(time=correctedtime) 
    print("Corrected Start to: %s; End to: %s" %  (str(ds.time.values[0]),str(ds.time.values[-1])))
    
# Drop variables not in list
dsvars = list(ds.variables)
remvar = [i for i in dsvars if i not in varlist]
ds = ds.drop(remvar)

# Slice to the proper time period
# NOTE: one month offset in time (starts at 1920-02-01)
ds = ds.sel(time=slice('1920-01-01','2005-12-31'))

# Compute Monthly means
varmon = ds[varname].groupby('time.month').mean('time')   

# Save output
start = time.time()

outpath = '/home/glliu/01_Data/'
outname = varname+'_HTR_clim.nc'
varmon.to_netcdf(outpath+outname)
elapsed = time.time() - start
tprint = "Saved data in  %.2fs" % (elapsed)
print(tprint)

