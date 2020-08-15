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
from scipy import signal
from scipy import stats
"""
Quick Function to linearly detrend, Ignore Nan

Assumes that the input variable has uniform step sizes, and is already ordered

Dependencies
    numpy as np
    scipy.stats
    
Taken from : https://stackoverflow.com/questions/44779315/detrending-data-with-nan-value-in-scipy-signal
    
"""
def detrendlin(var_in):
    debug = 0
    if debug == 1:
        var_in = sssr[:,32]
    
    x = np.arange(0,len(var_in))
    
    # Limit to non-nan values
    inotnan = ~np.isnan(var_in)
    
    # Perform Regression
    m,b,r_val,p_val,std_err=stats.linregress(x[inotnan],var_in[inotnan])
    
    # Detrend
    var_detrend = var_in - (m*x +b)
    
    return var_detrend



"""
Multidimensional detrend along each dimension, ignoring nans

Detrend is performed along first dimension

Input:
    1. var_in: N-D array with dim to detrend along on axis=0

Dependencies:
    numpy as np
    detrendlin function

"""
def detrendlin_nd(var_in):
    
    
    # Reshape to combine all other dimensions
    alldims = var_in.shape[1:]
    combinedims = 1
    for ele in alldims:
        combinedims *= ele
    var_rs     = np.reshape(var_in,(var_in.shape[0],combinedims))
    var_dt = np.zeros(var_rs.shape)
    
    
    # Loop over each dimension
    for i in range(0,combinedims):
        
        # Select timeseries for that point
        vloop = np.copy(var_rs[:,i])
        
        # Skip if all values are nan
        if np.all(np.isnan(vloop)):
            
            # Assign values to nan
            var_dt[:,i] = np.ones(vloop.shape)*np.nan
            
        else:
            
            # Detrend using 1d function
            var_dt[:,i] = detrendlin(vloop)
            
    
    var_dt = np.reshape(var_dt,var_in.shape)
    
    return var_dt
            
        
"""
var1 = mon x yr
"""
def calc_lagcovar(var1,var2,lags,basemonth,detrendopt):
    import numpy as np
    from scipy import signal
    from scipy import stats
    
    
    
    debug = 0
    
    if debug == 1:
        basemonth = 2
        lags = lags
        var1 = insss
        var2 = insss
        detrendopt = 1
    
    # Commented out since nan values are removed in detrending and corr calculation steps
    # # Remove nan points for calculation
    # if np.any(np.isnan(var1)):
    #     print("WARNING: Ignoring NaNs at "+ str(np.where(np.isnan(var1))))
    #     # Not really understanding the notation for this...
    #     # https://stackoverflow.com/questions/11620914/removing-nan-values-from-an-array
    #     var1 = np.where(np.isfinite(var1),var1,99) # Replace value with zero? need a better solution
    #     var1 = np.where(np.isfinite(var1),var1)
       
    #     dim1ok = np.where(np.isfinite(var1))[0]
    #     dim2ok = np.where(np.isfinite(var1))[1]
    
    # if np.any(np.isnan(var2)):
    #     print("WARNING: Ignoring NaNs at "+ str(np.where(np.isnan(var2))))
    #     var2 = np.where(np.isfinite(var2),var2,0)
        
    

    # Get total number of lags
    lagdim = len(lags)
    
    # Get timeseries length
    tot1 = var1.shape[1]
    tot2 = var2.shape[1]
    totyr = min(tot1,tot2)
    
    # Get lag and lead sizes (in years)
    leadsize = int(np.floor(len(np.where(lags < 0)[0])/12))
    lagsize = int(np.floor(len(np.where(lags > 0)[0])/12))
    
    
    # Detrend variables if option is set
    if detrendopt == 1:
        
        if np.any(np.isnan(var1)):
            # Move year dimension to the front
            var1 = np.transpose(var1,(1,0))
            var1 = detrendlin_nd(var1)
            var1 = np.transpose(var1,(1,0))
        else:
            var1 = signal.detrend(var1,1,type='linear')
        
        if np.any(np.isnan(var2)):
            var2 = np.transpose(var2,(1,0))
            var2 = detrendlin_nd(var2)
            var2 = np.transpose(var2,(1,0))
        else:
            var2 = signal.detrend(var2,1,type='linear') 
    

    
    # Get base timeseries to perform the autocorrelation on
    base_ts = np.arange(0+leadsize,totyr-lagsize)
    varbase = np.copy(var1[basemonth-1,base_ts])
        
    # Preallocate Variable to store correlations
    corr_ts = np.zeros(lagdim)
    
    # Set some counters
    nxtyr = 0
    addyr = 0
    modswitch = 0
    ilag = 0
    
    for i in lags:
        
        

        lagm = (basemonth + i)%12
        
        if lagm == 0:
            lagm = 12
            addyr = 1         # Flag to add to nxtyr
            modswitch = i+1   # Add year on lag = modswitch
            
        if addyr == 1 and i == modswitch:
            if debug == 1:
                print('adding year on '+ str(i))
            addyr = 0         # Reset counter
            nxtyr = nxtyr + 1 # Shift window forward
        
        # Index the other variable
        lag_ts = np.arange(0+nxtyr,len(base_ts)+nxtyr)
        varlag = np.copy(var2[lagm-1,lag_ts])
        
        # Re-copy over original variable to account for previous deletions
        varbase = np.copy(var1[basemonth-1,base_ts])
        
        # Calculate correlation
        # Remove nan values prior to calculation
        if np.any(np.isnan(varlag)):
            if debug == 1:
                print("NaN value found in varlag for i = %i" % i)
            
            # Find nonnan values
            nanidx = np.where(np.isnan(varlag))
            
            # Find corresponding year in base timeseries
            nanbase = np.where(base_ts == lag_ts[nanidx])
            
            # Remove from each variable
            varlag = varlag[np.where(~np.isnan(varlag))]
            varbase = np.delete(varbase,nanbase)
   
            
        if np.any(np.isnan(varbase)):
            if debug == 1:
                print("NaN value found in varbase for i = %i" % i)
                
            # Find non nan values
            nanidx = np.where(np.isnan(varbase))
            
            # Find corresponding year in base timeseries
            nanlag = np.where(lag_ts == base_ts[nanidx])
            
            # Remove from each variable
            varbase = varbase[nanidx]
            varlag = np.delete(varlag,nanlag)
        
        
        corr_ts[ilag] = stats.pearsonr(varbase,varlag)[0]
        ilag += 1
            
        if lagm == basemonth & debug == 1:
            print(i)
            print(corr_ts[i])
            
            
    return corr_ts



projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"
outpath = projpath + '02_Figures/20200629/'
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
lags = np.arange(0,37,1)

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
# Detrend
# ------------------------
detrend = 0
if detrend == 1:
    
    SSS = detrendlin_nd(SSS)
    SST = detrendlin_nd(SST)

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


# ----------
# Detrend??
# ----------


# -------------------------------
# Remove seasonal average
# -------------------------------

# Reshape the other two variables to year x mon and remove climatological mean
sstr = np.reshape(sstr,yearxmon)
sssr = np.reshape(sssr,yearxmon)

# Remove climatology
sstnc = sstr - np.nanmean(sstr,axis=0)
sssnc = sssr - np.nanmean(sssr,axis=0)


# ----------------
# Calculate lag/lead
# ----------------
#lags = np.arange(-25,26,1)
#nlags = len(lags)

lagvars ={}
lag_sss = np.zeros((nlags,nens))
lag_sst = np.zeros((nlags,nens))

for e in range(0,nens):
    
    # Get base month for regression for that member
    basemonth = kmonth[e]
    
    # Retrieve time series data
    insss = np.copy(sssnc[:,:,e])
    insst = np.copy(sstnc[:,:,e])
    
    # Permute to mon x year (to match calc_lagcovar settings)
    insss = np.transpose(insss,(1,0))
    insst = np.transpose(insst,(1,0))
    
    # Compute lag/lead
    lag_sss[:,e] = calc_lagcovar(insss,insss,lags,basemonth,1)
    lag_sst[:,e] = calc_lagcovar(insst,insst,lags,basemonth,1)
    

# Calculate ensemble average
lag_sss_ensavg = np.mean(lag_sss,1)
lag_sst_ensavg = np.mean(lag_sst,1)


lagvars['SSS'] = lag_sss
lagvars['SST'] = lag_sst



# -------------------------------
# Make some Figures
# -------------------------------

# Autocorrelation plots

# Start Plot
f1,axs = plt.subplots(2,1)
plt.style.use('seaborn')


n = 0
for vn in ('SST','SSS'):
    
    # Plot SSS Autocorrelation on Bottom
    # Plot each ensemble member, then 1 for labeling
    for e in range(1,nens):
        #print(e)
        axs[n].plot(lags,lagvars[vn][:,e],linewidth=0.75,color=(.75,.75,.75))
    ln0 = axs[n].plot(lags,lagvars[vn][:,-1],linewidth=0.75,color=(.75,.75,.75),label=r'Indv. Member')
    
    
    # Calculate and plot ensemble average
    ensavg = np.mean(lagvars[vn],1)
    ln1 = axs[n].plot(lags,ensavg,color='k',marker='o',markersize=5,label=r'Ens. Avg.')
        
    # Set legend location and fontsize
    #ax.legend(loc='best',)
    #plt.rc('legend', fontsize=10)    # legend fontsize
    lns = ln0 + ln1 
    labs = [l.get_label() for l in lns]
    axs[n].legend(lns,labs)
    
    # if n == 1:
    #     axs[n].set_xlabel('Lag (Months)')
        
    #axs[n].set_ylabel('Correlation')
    axs[n].set_title(vn,pad=1)
    axs[n].set_xticks(np.arange(0,42,6))

    n += 1
f1.suptitle('Reemergence for '+region+': 1920-2005 (CESM1-LENS)',fontsize=20)
f1.text(0.05,0.5,"Correlation",fontsize=12,ha="center", va="center", rotation=90)
f1.text(0.5,0.05,"Lag (Months)",fontsize=12,ha="center", va="center")



# Print figure
outname = outpath+"Reemergence_"+region+".png"
plt.savefig(outname, bbox_inches="tight",dpi=200)