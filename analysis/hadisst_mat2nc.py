#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert HadISST from matfile to ncfile
Quick scrap processing script..

Created on Tue Oct 20 22:42:57 2020

@author: gliu
"""

# Reprocess the hadISST data


from scipy.io import loadmat
import xarray as xr
import numpy as np


datpath = "/Users/gliu/Downloads/2020_Fall/6.862/Project/Data/"
ncname  = "hadisst.1870_2018.mat"




# Load in observation SST data to compare
obvhad = loadmat(datpath+ncname)
hlat = np.squeeze(obvhad['LAT'])
hlon = np.squeeze(obvhad['LON'])
hyr  = obvhad['YR'].squeeze()
hsst = obvhad['SST']


# Change hsst to lon x lat x time
hsst = np.transpose(hsst,(2,1,0))


#%% Fix Latitude Dimensions for HSST
# currently it is arranged 90:-90, need to flip to south first

# Find north and south latitude points
hsouth = np.where(hlat <= 0)
hnorth = np.where(hlat > 0)

# Find corresponding points in data
hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]

# Stitch things together, reversing the order 
hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)

mons = xr.cftime_range(start=str(hyr[0])+"-01-01",end=str(hyr[-1])+"-12-01",freq="MS") 


# Save as a revived ncfile
ds = xr.DataArray(hsstnew.transpose(2,1,0),
                coords=[mons,hlatnew,hlon],
                dims=["time","lat","lon"])

# Name variable and compress!
ds = ds.to_dataset(name='sst')
ds.to_netcdf(datpath+"hadisst.1870-01-01_2018-12-01.nc",encoding={'sst': {'zlib': True}})