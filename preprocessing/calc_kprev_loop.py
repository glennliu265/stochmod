#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate indices of detrainment month for a global array of MLD timeseries
(Currently written to work with CESM1-PreIndustrial Control run)

Created on Tue Jul 13 13:38:41 2021

@author: gyl
"""
import tqdm
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


import sys
stormtrack = 1
if stormtrack == 0:
    #projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    #datpath     = projpath + '01_Data/'
    datpath = "/Users/gyl/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    sys.path.append("/Users/gyl/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gyl/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    #sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    #sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")


elif stormtrack == 1: # NOTE... need to edit this
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/HMXL/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
import scm
#%% User Edits

#datpath = "/Users/gyl/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input"
fname   = "HMXL_PIC.nc"
#figpath = "/Users/gyl/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210715/"

entrain1 = -1 # Value to assign for the first mixing
entrain0 = 0  # Value when there is no entrainment

#%% Load the data

st    = time.time()
ds    = xr.open_dataset(datpath+fname)
lon   = ds.lon.values
lat   = ds.lat.values
hmxl  = ds.HMXL.values
times = ds.time.values
print("Loaded data in %.2fs"% (time.time()-st))

#ds.close()

#%% Calculate Kprev for each point
st    = time.time()
# Preallocate
nlon,nlat,ntime = hmxl.shape
kprevall = np.zeros((nlon,nlat,ntime)) * np.nan

# Loop for each point
for o in tqdm.tqdm(range(nlon)): # loop lon
    for a in range(nlat): # loop lat
        hpt = hmxl[o,a,:] # get point
        if np.any(np.isnan(hpt)): # Skip Land/Ice Points
            continue
        kprev = scm.calc_kprev_lin(hpt,entrain1=entrain1,entrain0=entrain0)
        kprevall[o,a,:] = kprev
print("Calculated kprev in %.2fs"% (time.time()-st))               
        
#%% Save kprev

np.save("%skprev_all_CESM-PIC.npy"%datpath,kprevall)
        
        