#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:26:26 2020

@author: gliu

Read in 5 deg CESM1LENS data and create correlation plots
"""


from scipy.io import loadmat
import numpy as np
import time
import matplotlib.pyplot as plt


def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
    import numpy as np
    from scipy import signal
    from scipy import stats
    
    
    
    debug = 0
    
    if debug == 1:
        basemonth = kmonth
        lags = lags
        var1 = temps
        var2 = temps
        detrendopt = 1
    
    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    totyr = var1.shape[1]
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.ceil(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.ceil(len(np.where(lags > 0)[0])/12))
    
    
    # Detrend variables if option is set
    if detrendopt == 1:
        var1 = signal.detrend(var1,1,type='linear')
        var2 = signal.detrend(var2,1,type='linear')
    
    # Get base timeseries to perform the autocorrelation on
    base_ts = np.arange(0+leadsize,totyr-lagsize)
    varbase = var1[basemonth-1,base_ts]
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    
    for i in lags:
        
        
        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
            
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(varbase)+nxtyr)
        varlag = var2[lagm-1,lag_ts]
        
        # Calculate correlation
        corr_ts[i] = stats.pearsonr(varbase,varlag)[0]
            
        if lagm == 3:
            print(i)
            print(corr_ts[i])
            
            
    return corr_ts



projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

ensnum = np.arange(1,41,1)       # Ensemble members to process
varnames = ("HMXL",'SST','SSS')  # Variables to load

# Set Region Bounding Coordinates
region = 'NEA'


if region == 'NEA':
    bounds = [40, 50, -35, -10]

elif region == 'SAR':
    bounds = [25, 35, -65, -45]
    
elif region == 'TRA':
    bounds = [-20, 40, -80, 0]
    
elif region == 'NAT':
    bounds = [0, 80, -80, 0]


latS = bounds[0]
latN = bounds[1]
lonW = bounds[2]
lonE = bounds[3]
if lonW < 0:
    lonW += 360
if lonE < 0:
    lonE += 360


# Lag options
lags = np.arange(-25,26,1)

# Compute some sizes
nens = len(ensnum)
nlags = len(lags)
# ----------------------------------
# Load in variables
# ----------------------------------
vardict = {}
# Load in each mat file and loop
for v in varnames:
    start = time.time()
    for n in ensnum:
        
        matname = "cesmle_htr_5deg/HTR_5deg_%s_ensnum%03d.mat" % (v,n)
        
        varmat  = loadmat(datpath+matname)   
        
        # [time(1032) x lon (73) x lat (37)]
        vn = varmat['varnew']
        
        # Make array from lat lon on loop 1 and adjust time period
        if n == 1:
            
            # Load lat/lon coordinates
            lat5 = varmat['lat5']
            lon5 = varmat['lon5']
            
            # Select starting from 1920
            vn = vn[840:1872,:,:]
            
            
            # Preallocate
            var_allens = np.zeros((vn.shape+ensnum.shape))
        
        # Assign array to here
        var_allens[:,:,:,n-1] = np.copy(vn)
        
        msg = "\rLoading %02d of %i members for %s" % (n,ensnum[-1],v)
        print(msg,end="\r",flush=True)
    
    ## Add variable to dictionary
    #if v == varnames[0]:
        
    vardict[v] = np.copy(var_allens)
    print("\n\tLoaded in %fs!" % (time.time()-start))
        

# ----------------------------------
# Process Data
# ----------------------------------

# Copy data over
MLD = np.copy(vardict['HMXL'])/100 # Convert to meters
SST = np.copy(vardict['SST'])
SSS = np.copy(vardict['SSS'])

# ------------------------
# Remove ensemble average
# ------------------------
#MLD = MLD - np.nanmean(MLD,axis=3)[:,:,:,None]
SST = SST - np.nanmean(SST,axis=3)[:,:,:,None]
SSS = SSS - np.nanmean(SSS,axis=3)[:,:,:,None]

# ------------------------
# Take regional average
# ------------------------
# Find Region coordinates
if lonW > lonE:
    # First dimension is empty, so take from second
    lonf = np.where((lon5 >= lonE) & (lon5 <=lonW))[1]
elif lonW < lonE:
    lonf = np.where((lon5 <= lonE) & (lon5 >=lonW))[1]
latf = np.where((lat5 >= latS)&(lat5 <= latN))[1]

# Restrict to region and take average (NOTE still need to verify this is select the right things)
mldr = np.nanmean(MLD[:,lonf[:,None],latf[None,:],:],axis=(1,2))
sstr = np.nanmean(SST[:,lonf[:,None],latf[None,:],:],axis=(1,2))
sssr = np.nanmean(SSS[:,lonf[:,None],latf[None,:],:],axis=(1,2))   

# -----------------------------------------------
# Find month with the maximum climatological MLD
# -----------------------------------------------

# Reshape time dimension to year x month, and take the mean along the year dimension
yearxmon = (int(mldr.shape[0]/12),12,mldr.shape[1])
mldclim = np.nanmean(np.reshape(mldr,yearxmon),axis=0)



# Locate the maximum month for each ensemble member
kmonth = [np.where(mldclim[:,e] == mldclim[:,e].max())[0][0]+1 for e in ensnum-1]

# -------------------------------
# Remove seasonal average
# -------------------------------

# Reshape the other two variables to year x mon and remove climatological mean
sstr = np.reshape(sstr,yearxmon)
sssr = np.reshape(sssr,yearxmon)

# Remove climatology
sstnc = sstr - np.nanmean(sstr,axis=0)
sssnc = sssr - np.nanmean(sssr,axis=0)


# ----------
# Detrend??
# ----------


# ----------------
# Calculate lag/lead
# ----------------
lag_sss = np.zeros((nlags,nens))
lag_sst = np.zeros((nlags,nens))
for e in range(0,nens):
    
    # Get base month for regression for that member
    basemonth = kmonth[e]
    
    # Retrieve time series data
    insss = np.copy(sssnc[:,:,e])
    insst = np.copy(sstnc[:,:,e])
    
    # Compute lag/lead
    lag_sss[:,e] = calc_lagcovar(insss,insss,lags,basemonth,1)
    lag_sst[:,e] = calc_lagcovar(insst,insst,lags,basemonth,1)