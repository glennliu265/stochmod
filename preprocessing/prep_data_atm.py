#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Counterpart to prep_data_ocn.py

Consolidates atmospheric data from CESM1


Copied sections from ...
- Investigate_Forcing.ipynb on (03/24/2022)

Created on Thu Mar 24 11:20:57 2022

@author: gliu
"""


import xarray as xr
import numpy as np
import glob
import time
import os
import scipy
from tqdm import tqdm

import matplotlib.pyplot as plt

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
from amv import proc

#%% User Edits

varname = "TS"
mconfig = "FULL_HTR"

# Preprocessing Option
detrend       = "linear" # Type of detrend (see scipy.signal.detrend)
if "HTR" in mconfig:
    detrend = "EnsAvg"

# Set Bounding Box
bbox          = [-80,0,0,65] # Set Bounding Box
bboxfn        = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])

# User Edits 
#outpath    = "/home/glliu/02_Figures/01_WeeklyMeetings/20210114/"
#outdatpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/"
datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % (varname)

# Output Name
savename       = "%s%s_%s_%s_DT%s.nc" % (datpath,varname,mconfig,bboxfn,detrend)
print("Saving output to %s"% savename)

#%%


def get_nclist(mconfig,vname,varpath=None,debug=False):
    """
    
    mconfig : STR - Model configuration ("FULL_HTR","FULL_PIC","SLAB_PIC")
    vname   : STR - Name of the variable in the preindustrial control run
    
    NOTE: Currently only supports atmospheric variables on stormtrack...
    
    """
    # Create list of ncfiles depending on inputs
    if varpath is None: # Set default variable path on stormtrack
        varpath = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/"
        ncpath = '%satm/proc/tseries/monthly/%s/' % (varpath,vname)
    
    if "PIC" in mconfig: # Load Preindustrial Control
        if mconfig == 'SLAB':
            ncsearch =  'e.e11.E1850C5CN.f09_g16.001.cam.h0.%s.*.nc' % vname
        elif mconfig == 'FULL':
            ncsearch =  'b.e11.B1850C5CN.f09_g16.005.cam.h0.%s.*.nc' % vname
    elif "HTR" in mconfig:
        ncsearch = "b.e11.B20TRC5CNBDRD.f09_g16.*.cam.h0.TS.*.nc"
    else:
        print("ERROR: Set mconfig to <SLAB> or <FULL>")
        return None
    # Get list of ncfiles
    # -------------------
    nclist = glob.glob(ncpath+ncsearch)
    nclist.sort()
    nclist = [nc for nc in nclist if "OIC" not in nc]
    if debug:
        print("Found %i files, from \n %s to... \n %s" % (len(nclist),nclist[0],nclist[-1]))
    if len(nclist) == 0:
        print("ERROR! No Files Found")
        return None
    return nclist


def load_PIC(mconfig,vname,varpath=None,debug=False,use_mf=False):
    """
    
    mconfig : STR - Model configuration ("FULL_HTR","FULL_PIC","SLAB_PIC")
    vname   : STR - Name of the variable in the preindustrial control run
    
    NOTE: Currently only supports atmospheric variables on stormtrack...
    
    """
    
    # Create preprocessing function
    varkeep  = [vname,'time','lat','lon','lev'] 
    def preprocess(ds,varlist=varkeep):
        """"preprocess dataarray [ds],dropping variables not in [varlist] and 
        selecting surface variables at [lev=-1]"""
        # Drop unwanted dimension
        dsvars = list(ds.variables)
        remvar = [i for i in dsvars if i not in varlist]
        ds = ds.drop(remvar)
        # Select the ground level
        ds = ds.isel(lev=-1)
        return ds
    
    # Open dataset
    if "PIC" in mconfig:
        combine_method ='by_coords'
        combine_dim     ='time'
    elif "HTR" in mconfig:
        combine_method ='nested'
        combine_dim     ='ensemble'
    
    # Open mfdataset
    if use_mf:
        st = time.time()
        dsall = xr.open_mfdataset(nclist,concat_dim=combine_dim,
                                  combine=combine_method,
                                  preprocess=preprocess,join="exact")
        print("Opened dataset in %.2fs"% (time.time()-st))
    else:
        dsall = []
        for i,nc in tqdm(enumerate(nclist)):
            ds = xr.open_dataset(nc)
            ds = preprocess(ds,varlist=varkeep)
            
            # Small roundoff error in latitude...
            if i >0:
                if np.any(~(ds.lat.values == dsall[0].lat.values)):
                    ds = ds.assign_coords(lat=dsall[0].lat.values)
                    print("\tReassigning latitude values for ENS%i"% (i+1))
            dsall.append(ds)
        dsall = xr.concat(dsall,dim=combine_dim)    
    
    return dsall

def load_combineflux(fluxname,mconfig,downward_positive=True,debug=False):
    vnames = ["FSNS","FLNS","LHFLX","SHFLX"]
    if fluxname == 'NHFLX':
        inflx = vnames
    elif fluxname == 'THFLX':
        inflx = vnames[2:]
    elif fluxname == 'RHFLX':
        inflx = vnames[:2]
    else:
        print("Please enter one of [NHFLX,THFLX,RHFLX]")
        return 0
    for v,vname in enumerate(inflx):
        print("Now Loading %s"%vname)
        ds = load_PIC(mconfig,vname,debug=debug)
        ds = fix_febstart(ds)
        #print(ds)
        ds = ds[vname]
        
        if vname == 'FSNS':
            ds *= -1
        
        if v == 0:
            fluxout = ds.copy()
        else:
            savevalori = fluxout[0,44,44].copy()
            fluxout += ds
            savevaladd = ds[0,44,44]
            if debug:
                print("%.2f + %.2f = %.2f"%(savevalori,savevaladd,fluxout[0,44,44].copy()))
    if downward_positive:
        fluxout *= -1
    fluxout = fluxout.rename(fluxname)
    return fluxout

#%%
st     = time.time()

# Get list of netCDF files from stormtrack
nclist = get_nclist(mconfig,varname,debug=True)

# Load data into dataset
ds_all = load_PIC(mconfig,varname,debug=True)
print("Opened dataset in %.2fs"% (time.time()-st))

# Flip longitude to degrees west
ds_flip = proc.lon360to180_xr(ds_all)

# Crop a region
ds_reg = ds_flip.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

# Load to NumPy Arrays
st_2 = time.time()
invar = ds_reg[varname].values # [(ensemble) x time x lat x lon]
lon   = ds_reg.lon.values
lat   = ds_reg.lat.values
times  = ds_reg.time.values
print("Loaded data to np-arrays in %.2fs" % (time.time()-st_2))


# Remove monthly anomalies
# -------------------------
if "HTR" in mconfig:
    time_axis = 1
else:
    time_axis = 0
climavg,tsyrmon = proc.calc_clim(invar,dim=time_axis,returnts=True)
invar_anom      = tsyrmon - climavg[:,None,...]

# Detrend
# -------
if "HTR" in mconfig: # Remove ensemble average
    invar_dt = invar_anom - np.mean(invar_anom,axis=0,keepdims=True)
    nens,nyr,nmon,nlat,nlon = invar_dt.shape
    newshape = (nens,nyr*nmon,nlat,nlon)
    coords_dict = {"ensemble" : np.arange(1,nens+1,1),
                   "time" : times,
                   "lat" : lat,
                   "lon" : lon
                   }
else:
    invar_dt = scipy.signal.detrend(invar_dt,axis=time_axis,method=detrend)
    nyr,nmon,nlat,nlon = invar_dt.shape
    newshape = (nyr*nmon,nlat,nlon)
    coords_dict = {"time" : times,
                   "z_t" : ds_reg.z_t.values,
                   "lat" : lat,
                   "lon" : lon
                   }

# Reshape array to recombine monxyr to time
invar_dt = invar_dt.reshape(newshape)

# Place back in data array [ (ensemble) x time x lon x lat]
dsa_reg_dt = xr.DataArray(invar_dt,coords=coords_dict,
                          dims=coords_dict,name=varname)
# Save output
# -----------
dsa_reg_dt.to_netcdf(savename,
                     encoding={varname: {'zlib': True}})
print("Saved output to %s in %.2fs" % (savename,time.time()-st))

