#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:10:43 2020

Test script to take the climatological monthly mean for a dataset and output it.

@author: gliu
"""




import xarray as xr
import time


# # Note: change variable name in dataset
# def preprocess(ds):
    
#     # Copy array
#     if ds.time.shape[0] > 1032:
    
#         ds = ds.sel(time=slice('1920-01-01','2005-12-31'))
    
#     return ds
    


# Set variable name
varname = 'HMXL'

# Variables to keep
varlist = [varname,'TLON','TLAT','time']

# Set paths and glob expression
ncpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/" + varname + "/"
ncnames = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h." + varname + ".*.nc"
globby = ncpath+ncnames

# Open dataset
ds = xr.open_mfdataset(globby,
                       concat_dim="ensemble",
 #                      combine="by_coords",
 #                      preprocess=preprocess,
                       parallel=True)


# Drop variables not in list
dsvars = list(ds.variables)
remvar = [i for i in dsvars if i not in varlist]
ds.drop(remvar)


# Slice to the proper time period
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

