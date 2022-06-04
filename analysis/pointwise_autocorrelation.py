#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CESM or stochastic model outputs
Support separate calculation for warm and cold anomalies

Based on postprocess_autocorrelation.py
Uses data preprocessed by reemergence/preprocess_data.py

Created on Thu Mar 17 17:09:18 2022

@author: gliu
"""
import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack = 0 # Set to True to run on stormtrack, False for local run

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "HadISST" #["PIC-FULL","HTR-FULL","PIC_SLAB","HadISST","ERSST"]
runid      = 9
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SST" # ["TS","SSS","SST]

# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]
#%% Set Paths for Input (need to update to generalize for variable name)


if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    if mconfig == "SM":
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    else:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname
        
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220324/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    if mconfig == "SM":
        datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    elif "PIC" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    elif "HTR" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    elif mconfig in ["HadISST","ERSST"]:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
    
# Import modules
from amv import proc,viz
import scm

# Set Input Names
# ---------------
if mconfig == "SM": # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    fnames      = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0.npz" %i for i in range(10)]
    mnames      = ["constant h","vary h","entraining"] 
elif "PIC" in mconfig:
    # Postproess Continuous CESM Run
    # ------------------------------
    print("WARNING! Not set up for stormtrack yet.")
    fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
    mnames     = ["FULL","SLAB"] 
elif "HTR" in mconfig:
    # CESM1LE Historical Runs
    # ------------------------------
    fnames     = ["%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname,]
    mnames     = ["FULL",]
elif mconfig == "HadISST":
    # HadISST Data
    # ------------
    fnames = ["HadISST_detrend2_startyr1870.npz",]
    mnames     = ["HadISST",]
elif mconfig == "ERSST":
    fnames = ["ERSST_detrend2_startyr1900_endyr2016.npz"]
    
    

# Set Output Directory
# --------------------
proc.makedir(figpath)
savename   = "%s%s_%s_autocorrelation_%s_%s.npz" %  (outpath,mconfig,varname,thresname,lagname)
if mconfig == "SM":
    savename = proc.addstrtoext(savename,"_runid2%02i" % (runid))

print("Output will save to %s" % savename)

#%% Read in the data (Need to update for variable name)
st = time.time()

if mconfig == "PIC-FULL":
    sst_fn = fnames[0]
elif mconfig == "PIC-SLAB":
    sst_fn = fnames[1]
elif "SM" in mconfig:
    sst_fn = fnames[runid]
else:
    sst_fn = fnames[0]
print("Processing: " + sst_fn)

if ("PIC" in mconfig) or ("SM" in mconfig):
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
        
        # Transpose to [lon x lat x time x otherdims]
        sst = sst.transpose(1,2,3,0)
        
    elif 'nc' in sst_fn:
        print("Loading netCDF")
        ds  = xr.open_dataset(datpath+sst_fn)
        
        ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
            
        lon = ds.lon.values
        lat = ds.lat.values
        sst = ds[varname].values # [lon x lat x time]
        
elif "HTR" in mconfig:
    
    ds  = xr.open_dataset(datpath+fnames[0])
    ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
    lon = ds.lon.values
    lat = ds.lat.values
    sst = ds[varname].values # [ENS x Time x Z x LAT x LON]
    sst = sst[:,840:,...].squeeze() # Select 1920 onwards
    sst = sst.transpose(3,2,1,0) # [LON x LAT x Time x ENS]
    
elif mconfig == "HadISST":
    
    # Load the data
    sst,lat,lon=scm.load_hadisst(datpath,startyr=1900) # [lon x lat x time]
    
    # Slice to region
    sst,lon,lat = proc.sel_region(sst,lon,lat,bboxlim)
    
elif mconfig == "ERSST":
    
    # Load the data
    sst,lat,lon=scm.load_ersst(datpath,startyr=1900)
    
    # Fliip the longitude
    lon,sst = proc.lon360to180(lon,sst)
    
    # Slice to region
    sst,lon,lat = proc.sel_region(sst,lon,lat,bboxlim)
    
    
print("Loaded data in %.2fs"% (time.time()-st))

# Apply land/ice mask if needed
if loadmask:
    print("Applying mask loaded from %s!"%loadmask)
    # Load the mask
    msk  = np.load(loadmask) # Lon x Lat (global)
    glon = np.load(glonpath)
    glat = np.load(glatpath)
    
    # Restrict to Region
    bbox = [lon[0],lon[-1],lat[0],lat[-1]]
    rmsk,_,_ = proc.sel_region(msk,glon,glat,bbox)
        
    # Apply to variable
    if "HTR" in mconfig:
        sst *= rmsk[:,:,None,None]
    else:
        sst *= rmsk[:,:,None]

#%% Do the calculations
"""
Inputs are:
    1) variable [lon x lat x time x otherdims]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    
"""
# First things first, combine lat/lon/otherdims, remove nan points

# Get Dimensions
if len(sst.shape) > 3:
    
    print("%s has more than 3 dimensions. Combining." % varname)
    nlon,nlat,ntime,notherdims = sst.shape
    sst = sst.transpose(0,1,3,2) # [nlon,nlat,otherdims,time]
    npts = nlon*nlat*notherdims # combine ensemble and points
    
else:
    notherdims      = 0
    nlon,nlat,ntime = sst.shape
    npts            = nlon*nlat

nyr             = int(ntime/12)
nlags           = len(lags)
nthres          = len(thresholds)

# Combine space, remove NaN points

sstrs                = sst.reshape(npts,ntime)
if varname == "SSS":
    sstrs[:,219]     = 0 # There is something wrong with this timestep?
sst_valid,knan,okpts = proc.find_nan(sstrs,1) # [finepoints,time]
npts_valid           = sst_valid.shape[0] 

# Split to Year x Month
sst_valid = sst_valid.reshape(npts_valid,nyr,12)

# Preallocate (nthres + 1 (for all thresholds), and last is all data)
class_count = np.zeros((npts_valid,12,nthres+2)) # [pt x eventmonth x threshold]
sst_acs     = np.zeros((npts_valid,12,nthres+2,nlags))  # [pt x eventmonth x threshold x lag]
sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]

# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    print(im)
    
    # For that month, determine which years fall into which thresholds [pts,years]
    sst_mon = sst_valid[:,:,im] # [pts x yr]
    sst_mon_classes = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)
    
    for th in range(nthres+2): # Loop for each threshold
    
        if th < nthres + 1: # Calculate/Loop for all points
            for pt in tqdm(range(npts_valid)): 
                
                # Get years which fulfill criteria
                yr_mask     = np.where(sst_mon_classes[pt,:] == th)[0] # Indices of valid years
                
                
                #sst_in      = sst_valid[pt,yr_mask,:] # [year,month]
                #sst_in      = sst_in.T
                #class_count[pt,im,th] = len(yr_mask) # Record # of events 
                #ac = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,0) # [lags]
                
                # Compute the lagcovariance (with detrending)
                sst_in = sst_valid[pt,:,:].T # transpose to [month x year]
                ac,yr_count = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,0,yr_mask=yr_mask,debug=False)
                cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
                
                # Save to larger variable
                class_count[pt,im,th] = yr_count
                sst_acs[pt,im,th,:] = ac.copy()
                sst_cfs[pt,im,th,:,:]  = cf.copy()
                # End Loop Point -----------------------------
        
        
        else: # Use all Data
            print("Now computing for all data on loop %i"%th)
            # Reshape to [month x yr x npts]
            sst_in    = sst_valid.transpose(2,1,0)
            acs = proc.calc_lagcovar_nd(sst_in,sst_in,lags,im+1,1) # [lag, npts]
            cfs = proc.calc_conflag(acs,conf,tails,nyr) # [lag x conf x npts]
            
            # Save to larger variable
            sst_acs[:,im,th,:] = acs.T.copy()
            sst_cfs[:,im,th,:,:]  = cfs.transpose(2,0,1).copy()
            class_count[:,im,th]   = nyr
        # End Loop Threshold -----------------------------
        
    # End Loop Event Month -----------------------------

#% Now Replace into original matrices
# Preallocate
count_final = np.zeros((npts,12,nthres+2)) * np.nan
acs_final   = np.zeros((npts,12,nthres+2,nlags)) * np.nan
cfs_final   = np.zeros((npts,12,nthres+2,nlags,2)) * np.nan

# Replace
count_final[okpts,...] = class_count
acs_final[okpts,...]   = sst_acs
cfs_final[okpts,...]   = sst_cfs

# Reshape output
if notherdims == 0:
    count_final = count_final.reshape(nlon,nlat,12,nthres+2)
    acs_final   = acs_final.reshape(nlon,nlat,12,nthres+2,nlags)
    cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres+2,nlags,2)
else:
    count_final = count_final.reshape(nlon,nlat,notherdims,12,nthres+2)
    acs_final   = acs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags)
    cfs_final   = cfs_final.reshape(nlon,nlat,notherdims,12,nthres+2,nlags,2)

# Get Threshold Labels
threslabs   = []
if nthres == 1:
    threslabs.append("$T'$ <= %i"% thresholds[0])
    threslabs.append("$T'$ > %i" % thresholds[0])
else:
    for th in range(nthres):
        thval= thresholds[th]
        
        if thval != 0:
            sig = ""
        else:
            sig = "$\sigma$"
        
        if th == 0:
            tstr = "$T'$ <= %i %s" % (thval,sig)
        elif th == nthres:
            tstr = "$T'$ > %i %s" % (thval,sig)
        else:
            tstr = "%i < $T'$ =< %i %s" % (thresholds[th-1],thval,sig)
        threslabs.append(th)
threslabs.append("ALL")

#% Save Output
np.savez(savename,**{
    'class_count' : count_final,
    'acs' : acs_final,
    'cfs' : cfs_final,
    'thresholds' : thresholds,
    'lon' : lon,
    'lat' : lat,
    'lags': lags,
    'threslabs' : threslabs
    },allow_pickle=True)

print("Script ran in %.2fs!"%(time.time()-st))
print("Output saved to %s."% (savename))