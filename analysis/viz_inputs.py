#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_inputs

Script to visualize model inputs

Starts out identical to stochmod region, but visualizes inputs instead of
running the model


Created on Sat Aug 22 19:23:00 2020

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
from dask.distributed import Client,progress
import dask

#%% User Edits -----------------------------------------------------------------           
# Point Mode
pointmode = 0 # Set to 1 to output data for the point speficied below
lonf = -30
latf = 50

# ID of the run (determines random number sequence if loading or generating)
runid = "002"

# White Noise Options. Set to 1 to load data
genrand   = 0  # Set to 1 to regenerate white noise time series, with runid above

# Forcing Type
# 0 = completely random in space time
# 1 = spatially unform forcing, temporally varying
# 2 = NAO-like NHFLX Forcing (DJFM), temporally varying 
# 3 = NAO-like NHFLX Forcing, with NAO (DJFM) and NHFLX (Monthly)
# 4 = NAO-like NHFLX Forcing, with NAO (Monthly) and NHFLX (Monthly)
funiform = 5     # Forcing Mode (see options above)
fscale   = 10    # Value to scale forcing by

# Integration Options
nyr      = 1000        # Number of years to integrate over
t_end    = 12*nyr      # Calculates Integration Period
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
T0       = 0           # Initial temperature [degC]
hfix     = 50          # Fixed MLD value (meters)

# Set Constants
cp0      = 3850 # Specific Heat [J/(kg*C)]
rho      = 1025 # Density of Seawater [kg/m3]

# Set Integration Region
lonW = -100
lonE = 20
latS = -20
latN = 90

#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20200823/'

# Set input path
input_path  = datpath + 'model_input/'

# Set up some strings for labeling
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monsfull=('January','Febuary','March','April','May','June','July','August','September','October','November','December')

## ------------ Script Start -------------------------------------------------
print("Now Running stochmod_region with the following settings: \n")
print("funiform  = " + str(funiform))
print("genrand   = " + str(genrand))
print("runid     = " + runid)
print("pointmode = " + str(pointmode))
print("fscale    = " + str(fscale))
print("nyr       = " + str(nyr))
print("Data will be saved to %s" % datpath)
allstart = time.time()
# --------------
# %% Load Variables -------------------------------------------------------------
# --------------

# Load damping variables (calculated in hfdamping matlab scripts...)
dampmat     = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp    = loadmat(input_path+dampmat)
LON         = np.squeeze(loaddamp['LON1'])
LAT         = np.squeeze(loaddamp['LAT'])
damping     = loaddamp['ensavg']

# Load Mixed layer variables (preprocessed in prep_mld.py)
mld         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# Save Options are here
saveforcing = 0 # Save Forcing for each point

# ------------------
# %% Restrict to region ---------------------------------------------------------
# ------------------

# Note: what is the second dimension for?
klat = np.where((LAT >= latS) & (LAT <= latN))[0]
if lonW < 0:
    klon = np.where((LON >= lonW) & (LON <= lonE))[0]
else:
        
    klon = np.where((LON <= lonW) & (LON >= lonE))[0]
          
# Restrict Damping Region
dampingr = damping[klon[:,None],klat[None,:],:]
lonr = np.squeeze(LON[klon])
latr = np.squeeze(LAT[klat])

# Restrict MLD variables to region
hclim = mld[klon[:,None],klat[None,:],:]
kprev = kprevall[klon[:,None],klat[None,:],:]

# Get lat and long sizes
lonsize = lonr.shape[0]
latsize = latr.shape[0]
np.save(datpath+"lat.npy",latr)
np.save(datpath+"lon.npy",lonr)

# %% Load and Prep NAO Forcing... <Move to separate script?>


if funiform > 1:
    # Load Longitude for processing
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    
    # Load (NAO-NHFLX)_DJFM Forcing
    if funiform == 2:
        
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1 and take ensemble average
        NAO1 = np.mean(naoforcing[0,:,:,:],0) # [Lat x Lon]
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
        
    elif funiform == 3:
        
        # Load NAO Forcing and take ensemble average
        naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
        NAO1 = np.nanmean(naoforcing,0) * -1  # Multiply by -1 to flip flux sign convention
        
        
    elif funiform == 4:
        
        # # Load Forcing and take ensemble average
        # naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC.npz")['eofall'] #[Ens x Mon x Lat x Lon]
        # NAO1 = np.nanmean(naoforcing,0)
    
          # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
        # Take ensemble average, then sum EOF 1 and EOF2
        NAO1 = naoforcing[:,:,:,:,0].mean(0)
    
    elif funiform == 5:
        # Load NAO Forcing and prepare (Prepared in regress_NAO_pattern.py)
        naoforcing = np.load(input_path+"NAO_EAP_NHFLX_ForcingDJFM.npy") #[PC x Ens x Lat x Lon]
        
        # Select PC1 and take ensemble average
        NAO1 = naoforcing[0:2,:,:,:].mean(1).sum(0)# [Lat x Lon] # Take mean along ensemble dimension, sum along pc 1-2
        NAO1 = NAO1[None,:,:] # [1 x Lat x Lon]
    elif funiform == 7:
        
        # Load Forcing and take ensemble average
        naoforcing = np.load(datpath+"NAO_Monthly_Regression_PC123.npz")['flxpattern'] #[Ens x Mon x Lat x Lon]
        
        # Take ensemble average, then sum EOF 1 and EOF2
        NAO1 = naoforcing[:,:,:,:,:2].mean(0).sum(3)
    
    # Transpose to [Lon x Lat x Time]
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude to Degrees East
    lon180,NAO1 = proc.lon360to180(lon360,NAO1)
    
    # Test Plot
    #plt.pcolormesh(NAO1[:,:,0].T)
    
    NAO1 = NAO1[klon[:,None],klat[None,:],:]
    
    # Convert from W/m2 to C/S for the three different mld options
    NAOF = scm.convert_NAO(hclim,NAO1,dt,rho=rho,cp0=cp0,hfix=hfix)
else:
    NAOF = 1
    
    
# ----------------------------
# %% Set-up damping parameters
# ----------------------------

lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=1,rho=rho,cp0=cp0,hfix=hfix)



#%% Try Looking at some forcing values from CESM

cesmforce = xr.open_dataset(datpath+'NHFLX_ForcingStats_Converted_NotMonthly_hvarmode2_ensavg.nc')


fstd = cesmforce.fstd.values
clon = cesmforce.lon.values
clat = cesmforce.lat.values

clon1,fstd=proc.lon360to180(clon,fstd.transpose(1,0))


# Get values  for different regions
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]

regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
