#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate AMV for the CESM Slab Model


Created on Fri Nov 13 11:45:17 2020

@author: gliu
"""


import xarray as xr
import numpy as np
import glob
import time

import sys
sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
from amv import proc

#%% User Edits

# Mode
mode = 'FULL' # "SLAB or FULL"

# TS [time lat lon]
varkeep = ['TS','time','lat','lon','lev'] 

# Subset data for enso index calculation
#bbox = [280, 0, 0, 65]
bbox = [270, 0, 0, 90]

# Filtering Options
order = 5
cutofftimemon = 120 # In Months
cutofftimeyr  = 10 # In Years


# Outpath
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/01_hfdamping/hfdamping_PIC_SLAB/AMV/"
outname = "EOF_AMV_PIC_FULL.npz"

#%% Functions 
    
# Define preprocessing variable
def preprocess(ds,varlist=varkeep):
    """"preprocess dataarray [ds],dropping variables not in [varlist] and 
    selecting surface variables at [lev=-1]"""
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)
    
    # Select the ground level
    ds = ds.isel(lev=-1)
    
    # # Correct first month (Note this isn't working)
    # if ds.time.values[0].month != 1:
    #     startyr = str(ds.time.values[0].year)
    #     endyr = str(ds.time.values[-1].year)
    #     correctedtime = xr.cftime_range(start=startyr,end=endyr,freq="MS",calendar="noleap") 
    #     ds = ds.assign_coords(time=correctedtime) 
    
    return ds

# Get List of nc files for preindustrial control
ncpath = r'/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/atm/proc/tseries/monthly/TS/'
if mode == 'SLAB':
    ncsearch = 'e.e11.E1850C5CN.f09_g16.001.cam.h0.TS.*.nc'
elif mode == 'FULL':
    ncsearch = 'b.e11.B1850C5CN.f09_g16.005.cam.h0.TS.*.nc'
nclist = glob.glob(ncpath+ncsearch)
nclist.sort()
nclist


# Open dataset
st = time.time()
dsall = xr.open_mfdataset(nclist,concat_dim='time',preprocess=preprocess)
print("Opened in %.2fs"%(time.time()-st))


# Apply Landice Mask
mask = np.load('/home/glliu/01_Data/00_Scrap/landicemask_enssum.npy')
dsall *= mask[None,:,:]


# Slice to region
dsreg = dsall.sel(lat=slice(bbox[2],bbox[3]))
dsreg = dsreg.where((dsreg.lon>=bbox[0]) | (dsreg.lon<=40),drop=True)

# Read out the variables
st = time.time()
ts = dsreg.TS.values
lon = dsreg.lon.values
lat = dsreg.lat.values
times = dsreg.time.values
print("Data loaded in %.2fs"%(time.time()-st))

#%% Calculate AMV

# Calculate Monthly Anomalies
ntime,nlat,nlon = ts.shape 
ts = ts.reshape(int(np.ceil(ntime/12)),12,nlat,nlon) # Separate mon/year dimensions
manom = ts.mean(0)
tsanom = ts - manom[None,:,:,:]
yranom = tsanom.mean(1).transpose(2,1,0) # Save annual data (lon x lat x time)
tsanom = tsanom.reshape(tsanom.shape[0]*12,nlat,nlon) # Recombine mon/year
tsanom = tsanom.transpose(2,1,0) # --> [lon x lat x time]

# Flip Longitude dimensions
lonw,tsanom = proc.lon360to180(lon,tsanom)
lonw,yranom = proc.lon360to180(lon,yranom)

# Calculate AMV Index
if bbox[0] > 180: # Also convert first longitude coordinate to degrees west
    bbox[0] -= 360
if bbox[1] > 180:
    bbox[1] -= 360
amvidmon,aavgmon = proc.calc_AMV(lonw,lat,tsanom,bbox,order,cutofftimemon,1)
amvidyr,aavgyr = proc.calc_AMV(lonw,lat,yranom,bbox,order,cutofftimeyr,1)

# Regress back to get amv spatial pattern
nyr = yranom.shape[2]

amvpatterns = []
tdims   = [nyr,ntime]
temps   = [yranom,tsanom]
aidx = [amvidyr,amvidmon] 

for i in range(2):

    invar = temps[i]
    tdim = tdims[i]
    amvid = aidx[i]
        
    # Regress to standardized AMV Index
    anorm  = amvid/amvid.std(0)
    pat    = proc.regress2ts(invar,anorm,nanwarn=0)
    
    # Append to output array    
    amvpatterns.append(pat)

# Save Output
st = time.time()
np.savez(outpath+outname,**{
         'patterns': amvpatterns,
         'indices': aidx,
         'amvnames':['annual','monthly'],
         'aavg': [aavgyr,aavgmon],
         'lon': lonw,
         'lat':lat,
         'times':times}
        )

print("Data saved in %.2fs"%(time.time()-st))