#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Regress DJFM NHFLX to DJFM SLP PC (NAO and EAP)

see "viz_NAO_DJFM.py" to visualize the output of this script.

-------

Reads in the npz "Manual_EOF_Calc_NAO_corr.npz"

Created on Fri Jul 10 11:38:34 2020

@author: gliu

"""

import xarray as xr
import numpy as np
import time
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc


#%% User Edits
# Path to data
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath  = projpath + '01_Data/'
outpath = projpath+'/02_Figures/20200823/'
flux = 'NHFLX' # [ 'FLNS','FSNS','LHFLX','SHFLX','RHFLX','THFLX','NHFLX','PSL']

# %% Load data

# Load Results from EOF Analysis
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO_corr.npz")
eofall    = npzdata['eofall']    # [ens x lat x lon x pc]
pcall     = npzdata['pcall']     # [ens x yr x pc]
varexpall = npzdata['varexpall'] # [ens x pc]

# Load Data base on flux (Combine separately for RHFLX and THFLX)
if flux == 'THFLX':
    
    # Load Latent Heat Fluxes
    ncname1 = 'LHFLX_NAOproc.nc'
    ds1 = xr.open_dataset(datpath+ncname1)
    lhflx = ds1['LHFLX']
    
    # Load Sensible Heat Fluxes
    ncname2 = 'SHFLX_NAOproc.nc'
    ds2 = xr.open_dataset(datpath+ncname2)
    shflx = ds2['SHFLX']
    
    # Sum the two fluxes
    flx = lhflx+shflx
    
elif flux == 'RHFLX':
    
    # Load Shortwave 
    ncname1 = 'FSNS_NAOproc.nc'
    ds1 = xr.open_dataset(datpath+ncname1)
    fsns = ds1['FSNS']
    
    # Load Sensible Heat Fluxes
    ncname2 = 'FLNS_NAOproc.nc'
    ds2 = xr.open_dataset(datpath+ncname2)
    flns = ds2['FLNS']
    
    # Sum the two fluxes
    flx = fsns+flns
    
else:
    
    ncname = "%s_NAOproc.nc" % flux
    ds = xr.open_dataset(datpath+ncname)
    flx = ds[flux]
    

# Read in Coordinate values
lon = flx['lon'].values
lat = flx['lat'].values
time = flx['year'].values
flx = flx.values

# %% Prepare data
var = np.copy(flx)*-1 # Note, multiply by negative 1 to convert to upwards negative

# Get dimension sizes
nens,nyr,nlat,nlon = var.shape
npc = pcall.shape[2]

# Combine lat and lon dimensions
var = np.reshape(var,(nens,nyr,nlat*nlon))

#%% Regress for each mode and ensemble membe

varpattern = np.zeros((npc,nens,nlat*nlon))

for n in range(npc):
    for e in range(nens):
        
        pcin = pcall[e,:,n]
        datain = var[e,...]
        varpattern[n,e,:],_ = proc.regress_2d(pcin,datain)
        
        msg = '\rCompleted Regression for PC %02i/%02i, ENS %02i/%02i' % (n+1,npc,e+1,nens)
        print(msg,end="\r",flush=True)

# Reshape variable [pc, ens, lat, lon]
varout = np.reshape(varpattern,(npc,nens,nlat,nlon))

F = np.copy(varout)

outname = datpath+"NAO_EAP_%s_ForcingDJFM.npy" % flux 
np.save(outname,F)

