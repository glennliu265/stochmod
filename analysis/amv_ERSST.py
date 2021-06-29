#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate/Preprocess SST from ERSST Dataset
Created on Thu May 27 21:21:36 2021

@author: gliu
"""

from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs
import xarray as xr


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz


#%% User Input

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"
ncname  = "ERSST5.nc"

datpathout = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"

# Detrending Options
method  = 2
startyr = 1900 
#%% Load data

ds = xr.open_dataset(datpath+ncname)
sst = ds['sst'].values
lat = ds['lat'].values
lon = ds['lon'].values
times = ds['time'].values

# Round off to the nearest year
sst = sst[:-11,:,:]


nmon,nlat,nlon = sst.shape
nyrs = int(nmon/12)
#%% Preprocess the data

# Flip array
sstflip = np.flip(sst,axis=1)
latnew = np.flip(lat)

# def flip_lat(hlat,hsst):
#     """
#     Flip latitude of variable goes from 90 to -90

#     Parameters
#     ----------
#     hlat : TYPE
#         DESCRIPTION.
#     hsst : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     hlatnew : TYPE
#         DESCRIPTION.
#     hsstnew : TYPE
#         DESCRIPTION.

#     """
#     # Find north and south latitude points
#     hsouth = np.where(hlat <= 0)
#     hnorth = np.where(hlat > 0)
    
#     # Find corresponding points in data
#     hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
#     hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]
    
#     # Stitch things together, reversing the order 
#     hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
#     hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)
    
#     return hlatnew,hsstnew


#%% Remove Seasonal Cycle first and plot

# Flip to lon x lat x time
hsstnew = sstflip.transpose(2,1,0)

# Deseason
dsfirst = np.reshape(hsstnew,(nlon,nlat,nyrs,12))
scycle  = np.mean(dsfirst,axis=2)[:,:,None,:]
dsfirst = dsfirst -scycle
dsfirst = np.reshape(dsfirst,(nlon,nlat,nmon))


# Detrend
start= time.time()
indata = dsfirst.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(indata,1)
x = np.arange(0,nmon,1)

if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T

# Replace back into dataset
dtdata = np.zeros((nlon*nlat,nmon))*np.nan
dtdata[okpts,:] = okdt
dtdata = dtdata.reshape(nlon,nlat,nmon)
print("Detrended in %.2fs" % (time.time()-start))



# # Plot Seasonal Cycle Removal and Detrended
lonf = -30+360
latf = 64
tper = np.arange(0,nmon)
klon,klat = proc.find_latlon(lonf,latf,lon,latnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdata[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

hlat = latnew
hsst = dtdata
hlon = lon
hyr  = times

# Save data (MONTHLY)
hadname  = "%sERSST_detrend%i_startyr%i.npz" % (datpathout,method,startyr)
np.savez(hadname,**{
    'sst':dtdata,
    'lat':latnew,
    'lon':lon,
    'yr':hyr},allow_pickle=True)


#%% Calculate and plot AMV
bbox = [-80,0 ,0,65]
for i in range(2):
    if bbox[i] <= 0:
        bbox[i] += 360

h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)


