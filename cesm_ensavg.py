#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:10:43 2020

Test script to take the climatological monthly mean for a dataset and output it.

@author: gliu
"""




import xarray as xr


varname = 'HMXL'

# Set paths and glob expression
ncpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/" + varname + "/"
ncnames = "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h." + varname + ".*.nc"
globby = ncpath+ncnames

# Open dataset
ds = xr.open_mfdataset(globby,
                       concat_dim="ensemble")

# Compute Monthly means
varmon = ds[varname].groupby('time.month').mean('time')  

# Save output
outpath = '/home/glliu/01_Data/'
outname = varname+'_HTR_clim.nc'
varmon.to_netcdf(outpath+outname)
