#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert matfile containing CESM1-LE Heat Flux Damping to NetCDF Files

Beginning of script taken from viz_hfdamping.py.

Created on Fri Jan 14 15:02:44 2022


@author: gliu
"""


from scipy.io import loadmat,savemat
import numpy as np
from scipy import stats
import time

import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import viz,proc
import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature
#%%

# Indicate Settings
flux    = "NHFLX"  # Flux Name
monwin  = 3        # 3 month smoothing or 1 month only
dof     = 82       # Degrees of Freedom for Significaxnce Testing
p       = 0.20     # p-value
tails   = 2        # two-tailed or one-tailed test...
lags    = [1,2]      # indicate lags to use
mode    = 4      # (1) No mask (2) SST only (3) Flx only (4) Both


# Toggles
save_netcdf = True

# Set Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/2020913/"

# Plotting
bbox = [280, 360, 0, 90]
cmap = cmocean.cm.tempo

# Set string for lagmax
lagmax      = lags[-1]
save_allens = 0 # Option to save output that has been averaged for lags, but not ens


#%% Load Necessary Data

if save_netcdf:
    
    # Load Matfile
    # ------------

    # Load Lat Lon
    mat1 = loadmat(datpath+"CESM1_LATLON.mat")
    lon = np.squeeze(mat1["LON"])
    lat = np.squeeze(mat1["LAT"])
    
    # Load damping variable [lon x lat x ens x mon x lag]
    mat2 = loadmat("%s%s_damping_ensorem1_monwin%i.mat" % (datpath,flux,monwin))
    damping = mat2['damping'] # [lon x lat x ens x mon x lag]
    
    # Load correlation coefficients [lon x lat x ens x mon x lag]
    mat3 = loadmat("%s%s_rho_ensorem1_monwin%i.mat" % (datpath,flux,monwin))
    rflx = mat3['rho']
    
    # Load SST autoorrelation coefficients
    mat4 = loadmat("%sSST_rho_ensorem1_monwin%i.mat"% (datpath,monwin))
    rsst = mat4['rsst']
    
    #% PChange dimensions to meet NetCDF Conventions 
    # -----------------------------------------------
    """
    from: https://cfconventions.org/cf-conventions/cf-conventions.html#dimensions
    
    If any or all of the dimensions of a variable have the interpretations of 
    "date or time" (T), "height or depth" (Z), "latitude" (Y), or "longitude" (X) 
    then we recommend, but do not require, those dimensions to appear in the 
    relative order T, then Z, then Y, then X ... All other dimensions should, 
    whenever possible, be placed to the left of the spatiotemporal dimensions.
    
    """
    
    # Transpose to [ens x lag x mon x lat x lon]
    invars  = [damping,rflx,rsst]
    outvars = []
    for v in invars:
        print(v.shape)
        vout = v.transpose(2,4,3,1,0)
        print(vout.shape)
        outvars.append(vout)
    dampingr,rflxr,rsstr = outvars
    
    # Make dimensions for data array
    ens_num = np.arange(1,43,1)
    lags    = np.arange(1,4,1)
    months  = np.arange(1,13,1)
    dims = {'ensemble'   : ens_num,
            'lag_month' : lags,
            'month'      : months,
            'latitude'   : lat,
            'longitude'  : lon
            }
    
    # Set some attributes
    varnames = ("nhflx_damping",
                "sst_flx_crosscorr",
                "sst_autocorr")
    varlnames = ("Net Heat Flux Damping",
                 "SST-Heat Flux Cross Correlation",
                 "SST Autocorrelation")
    units     = ("W/m2/degC",
                 "Correlation",
                 "Correlation")
    
    # Convert from ndarray to data array
    # ----------------------------------
    das = []
    for v,name in enumerate(varnames):
        attr_dict = {'long_name':varlnames[v],
                     'units':units[v]}
        da = xr.DataArray(outvars[v],
                    dims=dims,
                    coords=dims,
                    name = name,
                    attrs=attr_dict
                    )
        if v == 0:
            ds = da.to_dataset() # Convert to dataset
        else:
            ds = ds.merge(da) # Merge other datasets
            
        # Append to list if I want to save separate dataarrays
        das.append(ds)
    
    #% Save as netCDF
    # ---------------
    st = time.time()
    encoding_dict = {name : {'zlib': True} for name in varnames} 
    savename = "%sCESM1-LE_NHFLX_Damping_raw.nc" % (datpath)
    print("Saving as " + savename)
    ds.to_netcdf(savename,
             encoding=encoding_dict)
    print("Saved in %.2fs" % (time.time()-st))

else: # Load NetCDF
    # st = time.time()
    # savename = "%sCESM1-LE_NHFLX_Damping_raw.nc" % (datpath)
    # ds = xr.open_dataset(savename)
    
    
    # damping = ds.nhflx_damping.values
    # rflx    = ds.sst_flx_crosscorr.values
    # rsst    = ds.sst_autocorr.values
    # lon     = ds.lon.values
    # lat     = ds.lat.values
    # mon     = ds.month.values
    # ens     = ds.ensemble.values
    # lag     = ds.lag_month.values
    
    print("Opened in %.2fs" % (time.time()-st))

#%% Make some visualizations
