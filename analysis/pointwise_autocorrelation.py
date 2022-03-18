#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CESM or stochastic model outputs
Support separate calculation for warm and cold anomalies

Based on pointwise_autocorrelation.py

Created on Thu Mar 17 17:09:18 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import numpy as np
import xarray as xr
from tqdm import tqdm 
import time

import cartopy.crs as ccrs
#%% Select dataset to postprocess

# Autocorrelation parameters
lags   = np.arange(0,37)
lagname = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

# Set Output Directory
figpath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220317/'
proc.makedir(figpath)


# Postprocess Continuous SM  Run
# ------------------------------
datpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
outpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
fnames   = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
onames     = ["spectra_%s_Fprime_rolln0_ampq0_method5_dmp0_run2%02d.nc" % (lagname,i) for i in range(10)]
mnames     = ["constant h","vary h","entraining"] 

# Postproess Continuous CESM Run
# ------------------------------
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FULL","SLAB"] 
onames     = ["spectra_%s_PIC-%s.nc" % (lagname,mnames[i]) for i in range(2)]

# Other Params
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

#%% Read in the data

sst_fn = fnames[0]
print("Processing: "+sst_fn)

# Load in SST [model x lon x lat x time] Depending on the file format
if 'npy' in sst_fn:
    print("Loading .npy")
    sst = np.load(datpath+sst_fn)
    # NOTE: Need to write lat/lon loader
elif 'npz' in sst_fn:
    print("Loading .npz")
    ld  = np.load(datpath+sst_fn,allow_pickle=True)
    lon = ld['lon']
    lat = ld['lat']
    sst = ld['sst'] # [model x lon x lat x time]
elif 'nc' in sst_fn:
    print("Loading netCDF")
    ds  = xr.open_dataset(datpath+sst_fn)
    
    ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
    
    
    lon = ds.lon.values
    lat = ds.lat.values
    sst = ds.SST.values

#%% Now we have sst [lon x lat x time]

# First things first, combine lat/lon, remove nan points
thresholds = [-1,0,1]

# Get Dimensions
nlon,nlat,ntime = sst.shape
nyr             = int(ntime/12)
npts            = nlon*nlat
nlags           = len(lags)
nthres          = len(thresholds)

# Combine space, remove NaN points
sst = sst.reshape(npts,ntime)
sst_valid,knan,okpts = proc.find_nan(sst,1) # [finepoints,time]
npts_valid = sst_valid.shape[0] 

# Split to Year x Month
sst_valid = sst_valid.reshape(npts_valid,nyr,12)

# Preallocate
class_count = np.zeros((npts_valid,12,nthres+1)) # [pt x eventmonth x threshold]
sst_acs     = np.zeros((npts_valid,12,nthres+1,nlags,12))  # [pt x eventmonth x threshold x lag x basemonth]
sst_cfs     = np.zeros((npts_valid,12,nthres+1,nlags,12,2))  # [pt x eventmonth x threshold x lag x basemonth x bounds]

# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    
    # For that month, determine which years fall into which thresholds [pts,years]
    sst_mon = sst_valid[:,:,im] # [pts x yr]
    sst_mon_classes = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)
    
    for th in range(nthres+1): # Loop for each threshold
        for pt in tqdm(range(npts_valid)): # Loop for each point
            
            # Get years which fulfill criteria
            yr_mask     = np.where(sst_mon_classes[pt,:] == th)[0]
            sst_in      = sst_valid[pt,yr_mask,:] # [year,month]
            sst_in      = sst_in.T
            class_count[pt,im,th] = len(yr_mask) # Record # of events 
            
            for bm in range(12): # Loop for each base month
                # Compute the lagcovariance
                ac = proc.calc_lagcovar(sst_in,sst_in,lags,bm+1,1) # [lags]
                cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                
                sst_acs[pt,im,th,:,bm] = ac.copy()
                sst_cfs[pt,im,th,:,bm,:]  = cf.copy()
                # End Loop Basemonth
            # End Loop Point
        # End Loop Threshold
    # End Loop Event Month


#%% Now Replace into original matrices

# Preallocate
count_final = np.zeros((npts,12,nthres+1)) * np.nan
acs_final   = np.zeros((npts,12,nthres+1,nlags,12)) * np.nan
cfs_final   = np.zeros((npts,12,nthres+1,nlags,12,2)) * np.nan

# Replace
count_final[okpts,...] = class_count
acs_final[okpts,...]   = sst_acs
cfs_final[okpts,...]   = sst_cfs

# Reshape output
count_final = count_final.reshape(nlon,nlat,12,nthres+1)
acs_final   = acs_final.reshape(nlon,nlat,12,nthres+1,nlags,12)
cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres+1,nlags,12,2)


# Save Output
savename = "%sCESM1_PIC-FULL_SST_autocorrelation_thres0.npz" %  outpath

np.savez(savename,**{
    'class_count' : count_final,
    'acs' : acs_final,
    'cfs' : cfs_final,
    'thresholds' : thresholds,
    'lon' : lon,
    'lat' : lat,
    'lags':lags
    },allow_pickle=True)

#%% Try to plot a point

savename = "%sCESM1_PIC-FULL_SST_autocorrelation_thres%s.npz" %  (outpatt,nthres)
ld = np.load(savename,allow_pickle=True)

count_final = ld['class_count']
acs_final  = ld['acs']
lon = ld['lon']
lat = ld['lat']


lonf = -30
latf = 50

klon,klat = proc.find_latlon(lonf,latf,lon,lat)

kmonth = 2

fig,ax =plt.subplots(1,1)

ax,ax2 = viz.init_acplot(kmonth,np.arange(0,38,2),lags,ax=ax)
ax.plot(lags,acs_final[klon,klat,kmonth,0,:,kmonth],label="Cold Anomalies (%i)" % (count_final[klon,klat,kmonth,0]),color='b')
ax.plot(lags,acs_final[klon,klat,kmonth,1,:,kmonth],label="Warm Anomalies (%i)" % (count_final[klon,klat,kmonth,1]),color='r')
ax.legend()


#%%

"""
Just realized it might not be possible to neatly vectorize this.

This is because at each point, there will be a different # of points within
each threshold, so it does not fit neatly into a 2d maxtrix...

"""

# Split into negative or positive anomalies


sst_classes = proc.make_classes_nd(sst_valid,thresholds,dim=1,debug=True)

# Now compute the autocorrelation for each lag, and for each case (positive/negative)
sst_acs = np.zeros(npts_valid,12,nthres+1,nlags) # [point,basemonth,threshold,lag]
for th in range(nthres+1): #Loop for each threshold

    sst_mask = (sst_classes == th)
    sst_mask = np.where(sst_classes == th)
    
    for im in range(12):
        
        insst = np.take_along_axis(sst_valid,sst_mask,1)
        
        #st_valid[np.where(sst_mask)]
        
        # [mon x time]
        sst_acs[:,im,th,:] = proc.calc_lagcovar_nd(insst,insst,lags,im+1,0)
        
        #autocorrm[m,:,:] = proc.calc_lagcovar_nd(oksst,oksst,lags,m+1,0)
    
    
    
        
        






#%%

thresholds = [-1,0,1]
y = sst_valid
