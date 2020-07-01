#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 21:39:04 2020

@author: gliu
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt, detrend
from scipy import stats
from scipy.io import loadmat

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% Functions
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


def area_avg(data,bbox,lon,lat,wgt):
    """
    Function to find the area average of data [lon x lat x otherdim]
    
    Given a bounding box [lonW, lonE, latS, latN]
    
    and corresponding lon/lat variables

    """
    
    # If wgt == 1, apply area-weighting 
    if wgt == 1:
        wgt = np.cos(np.radians(lat))
        
        data = data * wgt[None,:,None]
    
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # Take average over lon and lat
    data_aa = np.nanmean(sel_data,(0,1))
    
    return data_aa


# From here: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



# Here: https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html
def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]


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
    var_detrend = var_in - (m * x +b)
    
    return var_detrend



def calc_AMV(lon,lat,sst,bbox,order,cutofftime):
    """
    

    Parameters
    ----------
    lon : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.
    sst : TYPE
        DESCRIPTION.
    bbox : TYPE
        DESCRIPTION.
    order : TYPE
        DESCRIPTION.
    cutofftime : TYPE
        DESCRIPTION.

    Returns
    -------
    amv

    """
    
    """
    
    # Dependencies
    functions: area_avg, detrendlin
    
    numpy as np
    from scipy.signal import butter,filtfilt
    """
    
    
    # Take the weighted area average
    aa_sst = area_avg(sst,bbox,lon,lat,1)
    
    # Linearly detrend the data
    aa_sst = detrendlin(aa_sst)
    
    # Normalize the data
    sstmean = np.nanmean(aa_sst)
    sststd  = np.nanstd(aa_sst)
    sstanom = (aa_sst - sstmean) / sststd
    
    # Design Butterworth Lowpass Filter
    filtfreq = len(aa_sst)/cutofftime
    nyquist  = len(aa_sst)/2
    cutoff = filtfreq/nyquist
    b,a    = butter(order,cutoff,btype="lowpass")
    
    # Compute AMV Index
    amv = filtfilt(b,a,sstanom)
    
    return amv,aa_sst
    
    
def plot_AMV(amv,ax=None):
    
    """
    
    Dependencies:
        
    matplotlib.pyplot as plt
    numpy as np
    """
    if ax is None:
        ax = plt.gca()
    
    
    htimefull = np.arange(len(amv))
    
    ax.plot(htimefull,amv,color='k')
    ax.fill_between(htimefull,0,amv,where=amv>0,facecolor='red',interpolate=True,alpha=0.5)
    ax.fill_between(htimefull,0,amv,where=amv<0,facecolor='blue',interpolate=True,alpha=0.5)

    return ax



def regress2ts(var,ts,normalizeall,method):
    
    
    # Anomalize and normalize the data (time series is assumed to have been normalized)
    if normalizeall == 1:
        varmean = np.nanmean(var,2)
        varstd  = np.nanstd(var,2)
        var = (var - varmean[:,:,None]) /varstd[:,:,None]
        
    # Get variable shapes
    londim = var.shape[0]
    latdim = var.shape[1]
    
    



    
    
    # 1st method is matrix multiplication
    if method == 1:
        
        # Combine the spatial dimensions 

        var = np.reshape(var,(londim*latdim,var.shape[2]))
        
        # Perform regression
        var_reg = np.matmul(np.ma.anomalies(var,axis=1),np.ma.anomalies(ts,axis=0))/len(ts)
        
        # Reshape to match lon x lat dim
        var_reg = np.reshape(var_reg,(londim,latdim))
    
    # 2nd method is looping point by poin  
    elif method == 2:
        
        
        # Preallocate       
        var_reg = np.zeros((londim,latdim))
        
        # Loop lat and long
        for o in range(londim):
            for a in range(latdim):
                
                # Get time series for that period
                vartime = np.squeeze(var[o,a,:])
                
                # Skip nan points
                if any(np.isnan(vartime)):
                    var_reg[o,a]=np.nan
                    continue
                
                # Perform regression 
                r = np.polyfit(ts,vartime,1)
                #r=stats.linregress(vartime,ts)
                var_reg[o,a] = r[0]
                #var_reg[o,a]=stats.pearsonr(vartime,ts)[0]
    
    return var_reg
        

#%% ----------------------------------------------------------------------------
# Set data paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200629/'

# Path to SST data from obsv
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"


# User settings (note, need to select better region that ignores stuff in the pacific)
bbox = [-75,5,0,65]

# # Filter settings: https://iescoders.com/time-series-analysis-filtering-or-smoothing-the-data/
# order   = 5
# fs      = 1/(30*24*3600) # sampling frequency in Hz
# nyquist = fs/2           # Nyquist is half thesampling frequency; 
# timecut = 120            # in Months
# cutoff  = 1/timecut/2     # desired cutoff frequency of the filter, Hz


# More filter settings rewritten (based on old matlab code)
order = 5
cutofftime = 120

#print('cutoff = ',1/cutoff*nyquist*30*24*3600,' months')




# ----------------------------------------------------------------------------
#%% Script Start 


hvarmode = np.arange(0,3,1)
hvarnames = ("Fixed (50m)","Maximum","Seasonal")



# Load in forcing and lat/lon
F        = np.load(datpath+"stoch_output_1000yr_Forcing.npy")
lon      = np.load(datpath+"lon.npy")
lat      = np.load(datpath+"lat.npy")




#%% Calculate AMV Index
sst = {}
amv = {}
aa = {}
# Load in dat for each mode and compute amv index
for mode in hvarmode:
    
    sst[mode] = np.load(datpath+"stoch_output_1000yr_entrain0_hvar%i.npy" % mode)


    # Calculate AMV Index
    amv[mode],aa[mode] = calc_AMV(lon,lat,sst[mode],bbox,order,cutofftime)

# Repeat, but for forcing
lpforcing,aaforcing = calc_AMV(lon,lat,F,bbox,order,cutofftime)
    
    
    
    

#%% Regress back to SST
regr_meth1 = {}
regr_meth2 = {}
# Perform regression for each mode
for mode in hvarmode:
    regr_meth1[mode]=regress2ts(sst[mode],amv[mode],0,1)
    
    regr_meth2[mode]=regress2ts(sst[mode],amv[mode],0,2)
 
    
# ----------------------------------------
# %% For Comparison, Repeat for Observations
# ----------------------------------------
# Path to SST data from obsv
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")
hlat = np.squeeze(obvhad['LAT'])
hlon = np.squeeze(obvhad['LON'])
hyr  = obvhad['YR']
hsst = obvhad['SST']

# Change hsst to lon x lat x time
hsst = np.transpose(hsst,(2,1,0))

# Take the set time period
startyr = 1920
monstart = (1920+1-hyr[0,0])*12
hsst = hsst[:,:,monstart::]


# For hsst, flip the latitude axis
# currently it is arranged 90:-90, need to flip to south first

# Find north and south latitude points
hsouth = np.where(hlat <= 0)
hnorth = np.where(hlat > 0)

# Find corresponding points in data
hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]

# Stitch things together, reversing the order 
hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)


# Take average from amv
h_amv,aa_hsst = calc_AMV(hlon,hlatnew,hsstnew,bbox,order,cutofftime)  

# Restrict to time domain

# Regress back to SST (Note: Normalizing seems to remove canonical AMV pattern)
h_regr=regress2ts(hsstnew,h_amv,0,2)

#%%
# -------------------------------
# Plot AMV from the 3 experiments
# -------------------------------
htimeall = np.arange(0,len(amv[1]))
hticks = np.arange(0,1100,100)
fig,ax = plt.subplots(3,1,sharex=True,sharey=True,figsize=(14,8))
for mode in hvarmode:
    plt.style.use("ggplot")
    
    plt.subplot(4,1,mode+1)
    
    #plt.plot(htimeall,aa[mode],color=[0.75,0.75,0.75])
    plot_AMV(amv[mode])
    
    
    
    plt.ylim([-2,2])
    plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
    plt.yticks(np.arange(-2,2.5,0.5),fontsize=16)
    plt.title("AMV Index for MLD %s" % hvarnames[mode],fontsize=20)
    #ax[mode+1].tick_params(labelsize=16)

    

    
    
plt.ylabel("AMV Index",fontsize=20)
plt.xlabel("Years",fontsize=20)
plt.tight_layout()
outname = outpath+'AMV_hvar_noentraing.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)


    
    
# -------------------------------
# %%Compare AMV to forcing
# -------------------------------
# Add forcing as first plot
fig,ax = plt.subplots(2,1,sharex=True,sharey=False,figsize=(14,6))
plt.style.use("ggplot")
plt.subplot(2,1,1)
plot_AMV(lpforcing)
plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
plt.yticks(np.arange(-0.25,0.50,0.25),fontsize=16)
plt.title("Forcing",fontsize=20)
plt.ylabel("Forcing Term K/s",fontsize=16)


plt.subplot(2,1,2)
model = 2
plot_AMV(amv[2])
plt.ylim([-2,2])
plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
plt.yticks(np.arange(-2,2.5,0.5),fontsize=16)
plt.title("AMV Index for MLD %s" % hvarnames[mode],fontsize=20)
plt.ylabel("AMV Index",fontsize=16)
plt.tight_layout()
outname = outpath+'AMV_hvar_forcing_comparison.png'
plt.savefig(outname, bbox_inches="tight",dpi=200)

    
    
    
# -------------------------------
#%% AMV Spatial Pattern Plots
# -------------------------------


def plot_AMV_spatial(var,lon,lat,bbox,cmap,cint=[0,],clab=[0,],ax=None):

    
    if ax is None:
        ax = plt.gca()
        
        
        
    # Add cyclic point to avoid the gap
    var,lon1 = add_cyclic_point(var,coord=lon)
    

    # Set up projections and extent
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    if len(cint) == 1:
        # Draw contours
        cs = ax.contourf(lon1,lat,var,cmap=cmap)
    
    else:
        # Draw contours
        cs = ax.contourf(lon1,lat,var,cint,cmap=cmap)
    
    
    
        # Negative contours
        cln = ax.contour(lon1,lat,var,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())
    
        # Positive Contours
        clp = ax.contour(lon1,lat,var,
                    cint[cint>=0],
                    colors='k',
                    linewidths=0.5,
                    transform=ccrs.PlateCarree())    
                      
        #ax.clabel(cln,colors=None)
                                
                
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if len(clab) == 1:
        bc = plt.colorbar(cs)
    else:
        bc = plt.colorbar(cs,ticks=clab)
    
    
    return ax
   # ax.set_title("AMV-related SST Pattern from HadISST, %i-%i"%(startyr,hyr[0,-1]))


# Set up plot
# Set colormaps and contour intervals
cmap = cmocean.cm.balance
cint = np.arange(-4,4.5,0.5)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
#cint = np.arange(-2,2.5,0.5)

clab = cint
for mode in hvarmode:
    print(mode)
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    plt.style.use("ggplot")
    
    varin = np.transpose(regr_meth2[mode],(1,0))
    
    plt.subplot(1,3,mode+1)
    plot_AMV_spatial(varin,lon,lat,bbox,cmap,cint,clab)
    plt.title("AMV Spatial Pattern \n MLD %s" % hvarnames[mode],fontsize=14)
    outname = outpath+'AMVpattern_hvar%i.png' % mode
    plt.savefig(outname, bbox_inches="tight",dpi=200)
    


#%% Plot AMV FFor HADLISST
# Same plot, but for HadlISST
cmap = cmocean.cm.balance
#cint = np.arange(-3.5,3.25,0.25)
cint = np.arange(-4,4.5,0.5)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
clab = cint

fig,ax = plt.subplots(1,1,figsize=(8,4))
plt.style.use("ggplot")

varin = np.transpose(h_regr,(1,0))

plt.subplot(1,1,1)
plot_AMV_spatial(varin,hlon,hlatnew,bbox,cmap,cint,clab)
plt.title("AMV-related SST Pattern from HadISST, %i-%i"%(startyr,hyr[0,-1]),fontsize=14)
outname = outpath+'AMVpattern_HADLISST.png' 
plt.savefig(outname, bbox_inches="tight",dpi=200)




#%% Load Matlab Data for Comparison
# Load matlab data for debugging
matlabver = loadmat(datpath+"test30W50N.mat")
SSTpt = matlabver['SSTpt']
MLamv = matlabver['AnomSSTfilt_DT']
fig,ax = plt.subplots(1,1)
#ax.plot(MLamv)
#ax.plot(amv)
ax.plot(SSTpt,color='k')
ax.plot(vartime,color='b')