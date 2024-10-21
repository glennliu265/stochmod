#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CESM or stochastic model outputs
Support separate calculation for warm and cold anomalies

Based on postprocess_autocorrelation.py
Uses data preprocessed by reemergence/preprocess_data.py

Threshold variables also processed via preprocess_data.py

Created on Thu Mar 17 17:09:18 2022

@author: gliu
"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#%% Select dataset to postprocess

# Set Machine
# -----------
stormtrack  = 1 # Set to True to run on stormtrack, False for local run

# Autocorrelation parameters
# --------------------------
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

# For Stochastic Model Output, indicate SM_[runname], with runnames indicated in stochmod_params
mconfig    = "SM_Tddamp"# "PIC-FULL"#"HadISST" #["PIC-FULL","HTR-FULL","PIC_SLAB","HadISST","ERSST"]
runid      = 9
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SST" # ["TS","SSS","SST]

# Set to False to not apply a mask (otherwise specify path to mask)
loadmask   = False #"/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/limask180_FULL-HTR.npy"
glonpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lon180.npy"
glatpath   = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/CESM1_lat.npy"

# Load another variable to compare thresholds (might need to manually correct)
thresvar      = False #
thresvar_name = "HMXL"  
if stormtrack:
    thresvar_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"
else:
    thresvar_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/thresholdvar/HMXL_FULL_PIC_lon-80to0_lat0to65_DTNone.nc"

if thresvar is True:
    loadvar = xr.open_dataset(thresvar_path)
    loadvar = loadvar[thresvar_name].values.squeeze() # [time x lat x lon]
    
    # Adjust dimensions to [lon x lat x time x (otherdims)]
    loadvar = loadvar.transpose(2,1,0)#[...,None]

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
bboxlim  = [-80,0,0,65]

debug = False # Debug section below script (set to True to run)
#%% Set Paths for Input (need to update to generalize for variable name)

if stormtrack:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    # Input Paths 
    if "SM" in mconfig:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    else:
        datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/%s/" % varname
        
    # Output Paths
    figpath = "/stormtrack/data3/glliu/02_Figures/20220622/"
    outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/03_reemergence/proc/"
    
else:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")

    # Input Paths 
    if "SM" in mconfig:
        datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    elif "PIC" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    elif "HTR" in mconfig:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    elif mconfig in ["HadISST","ERSST"]:
        datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    
    
    # Output Paths
    figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220930/'
    outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'

cwd = os.getcwd()
sys.path.append(cwd+"/../")
import stochmod_params as sparams
   
# Import modules
from amv import proc,viz
import scm
import stochmod_params as sparams

#%% Set Input Names
# ---------------
if "SM" in mconfig: # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    smname = mconfig[3:] # Crop out "SM_"
    print("WARNING! Not set up for stormtrack yet.")
    
    fnames = sparams.rundicts[smname]
    fnames = ["stoch_output_%s.npz" % f for f in fnames]
    
    if "Qek" in mconfig:
        mnames      = ["entraining"] 
    else:
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
if "SM" in mconfig:
    savename = proc.addstrtoext(savename,"_runid2%02i" % (runid))
if thresvar is True:
    savename = proc.addstrtoext(savename,"_thresvar%s" % (thresvar_name))

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
    
    # Flip the longitude
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
    6) loadvar(optional) [lon x lat x time x otherdims] (thresholding variable)
    
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
    
if thresvar: # Only analyze where both threshold variable and target var are non-NaN
    ntimeldvar    = loadvar.shape[2]
    loadvarrs     = loadvar.reshape(nlat*nlon,ntimeldvar)
    _,knan,okpts  = proc.find_nan(sstrs*loadvarrs,1) # [finepoints,time]
    sst_valid     = sstrs[okpts,:]
    loadvar_valid = loadvarrs[okpts,:]
    
else:
    sst_valid,knan,okpts = proc.find_nan(sstrs,1) # [finepoints,time]
npts_valid           = sst_valid.shape[0] 


# Split to Year x Month
sst_valid = sst_valid.reshape(npts_valid,nyr,12)
if thresvar: # Select non-NaN points for thresholding variable
    loadvar_valid = loadvar_valid.reshape(npts_valid,nyr,12)

# Preallocate (nthres + 1 (for all thresholds), and last is all data)
class_count = np.zeros((npts_valid,12,nthres+2)) # [pt x eventmonth x threshold]
sst_acs     = np.zeros((npts_valid,12,nthres+2,nlags))  # [pt x eventmonth x threshold x lag]
sst_cfs     = np.zeros((npts_valid,12,nthres+2,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]

# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    #print(im)
    
    # For that month, determine which years fall into which thresholds [pts,years]
    sst_mon = sst_valid[:,:,im] # [pts x yr]
    if thresvar:
        loadvar_mon = loadvar_valid[:,:,im]
        sst_mon_classes = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)
    else:
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


#%% Debugging Corner

if debug:
    
    """
    Section one, examine subsetting at a point...
    """
    nlon = len(lon)
    nlat = len(lat)
    kpt  = np.ravel_multi_index(np.array(([40],[53])),(nlon,nlat))
    
    
    # Get Point Variable and reshape to yr x mon
    sstpt  = sstrs[kpt,:]
    mldpt  = loadvarrs[kpt,:]
    sst_in = sstpt.reshape(int(sstpt.shape[1]/12),12) # []
    mld_in = mldpt.reshape(sst_in.shape)
    
    # Calculate autocorrelation (no mask)
    acs    = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=None,debug=False)
    plt.plot(acs)
    
    # Calculate autocorrelation with mask
    loadvar_mon = loadvar_valid[:,:,im]
    sst_mon     = sst_valid[:,:,im]
    
    sst_class = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)[kpt,:]
    mld_class = proc.make_classes_nd(loadvar_mon,thresholds,dim=1,debug=False)[kpt,:]
    
    mask_sst     = np.where(sst_class.squeeze() == th)[0] # Indices of valid years
    mask_mld     = np.where(mld_class.squeeze() == th)[0] # Indices of valid years
    
    acs_sst,yr_count_sst = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_sst,debug=False)
    acs_mld,yr_count_mld = proc.calc_lagcovar(sst_in.T,sst_in.T,lags,im+1,0,yr_mask=mask_mld,debug=False)
    
    fig,ax=plt.subplots(1,1)
    ax.plot(lags,acs_sst,label="SST Threshold, count=%i" % yr_count_sst,color='k')
    ax.plot(lags,acs_mld,label="MLD Threshold, count=%i" % yr_count_mld,color='b')
    ax.legend()
    
    fig,ax = plt.subplots(1,1)
    ax.plot(sst_in_actual[0,:],label="actual SST")
    ax.plot(sst_in[:,0])
    ax.legend()