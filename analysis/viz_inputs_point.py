#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Input Parameters (Seasonal Cycle) at a Point

Created on Tue Apr 27 01:20:49 2021

@author: gliu
"""



import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx

from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
from scipy import signal
from tqdm import tqdm

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import scm
import time
import cmocean


#%% User Edits

projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20210424/'
input_path  = datpath + 'model_input/'


mconfig = "FULL_PIC"
if mconfig == "FULL_PIC":
    configname = "Fully-Coupled PiC"

bboxplot  = [-100,20,-10,80]

# ------------
#%% Load Data

# Lat and Lon
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp       = loadmat(input_path+dampmat)
lon            = np.squeeze(loaddamp['LON1'])
lat            = np.squeeze(loaddamp['LAT'])

# Stochastic Model Input
if "PIC" in mconfig: # Load Mixed layer variables (preprocessed in prep_mld.py)
    hclim = np.load(input_path+"FULL_PIC_HMXL_hclim.npy")
    kprevall    = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
else: # Load Historical
    hclim         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
    kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# Load Slab MLD
hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    

# Load full MLD field
ds = xr.open_dataset(input_path+"HMXL_PIC.nc")
mldfullpic = ds.HMXL.values

# Reshape to separate month and season
nlat,nlon,ntime = mldfullpic.shape
nyr = int(ntime/12)
mldfullpic = mldfullpic.reshape(nlat,nlon,nyr,12)

# All variables are lon 180
damping = np.load(input_path+"SLAB_PIC"+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy") # Lon 180
forcing = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")



# ---------------------------
#%% Retrieve data for a point
# ---------------------------
lonf = -30
latf = 50

klon,klat = proc.find_latlon(lonf,latf,lon,lat)

mldpt = hclim[klon,klat,:]
damppt = damping[klon,klat,:]








