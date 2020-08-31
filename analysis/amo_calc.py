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
from scipy import signal
import xarray as xr

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
import matplotlib.gridspec as gridspec

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz


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
        
    # Find lat/lon indices 
    kw = np.abs(lon - bbox[0]).argmin()
    ke = np.abs(lon - bbox[1]).argmin()
    ks = np.abs(lat - bbox[2]).argmin()
    kn = np.abs(lat - bbox[3]).argmin()
    
        
    # Select the region
    sel_data = data[kw:ke+1,ks:kn+1,:]
    
    # If wgt == 1, apply area-weighting 
    if wgt == 1:
        
        # Make Meshgrid
        _,yy = np.meshgrid(lon[kw:ke+1],lat[ks:kn+1])
        
        
        # Calculate Area Weights (cosine of latitude)
        wgta = np.cos(np.radians(yy)).T
        
        # Remove nanpts from weight
        nansearch = np.sum(sel_data,2) # Sum along time
        wgta[np.isnan(nansearch)] = 0
        
        
        # Apply area weights
        #data = data * wgtm[None,:,None]
        sel_data  = sel_data * wgta[:,:,None]

    
    # Take average over lon and lat
    if wgt ==1:

        # Sum weights to get total area
        sel_lat  = np.sum(wgta,(0,1))
        
        # Sum weighted values
        data_aa = np.nansum(sel_data/sel_lat,axis=(0,1))
    else:
        # Take explicit average
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


    

        

# Functions
def regress_2d(A,B):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    # Compute using nan functions (slower)
    if np.any(np.isnan(A)) or np.any(np.isnan(B)):
        print("NaN Values Detected...")
    
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
            Banom = B - np.nanmean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.nanmean(A,axis=a_axis)
            Banom = B - np.nanmean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.nansum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.nansum(B,axis=b_axis) - beta * np.nansum(A,axis=a_axis))/A.shape[a_axis]
    else:
        # 2D Matrix is in A [MxN]
        if len(A.shape) > len(B.shape):
            
            # Tranpose A so that A = [MxN]
            if A.shape[1] != B.shape[0]:
                A = A.T
            
            
            # Set axis for summing/averaging
            a_axis = 1
            b_axis = 0
            
            # Compute anomalies along appropriate axis
            Aanom = A - np.mean(A,axis=a_axis)[:,None]
            Banom = B - np.mean(B,axis=b_axis)
            
        
            
        # 2D matrix is B [N x M]
        elif len(A.shape) < len(B.shape):
            
            # Tranpose B so that it is [N x M]
            if B.shape[0] != A.shape[0]:
                B = B.T
            
            # Set axis for summing/averaging
            a_axis = 0
            b_axis = 0
            
            # Compute anomalies along appropriate axis        
            Aanom = A - np.mean(A,axis=a_axis)
            Banom = B - np.mean(B,axis=b_axis)[None,:]
        
        # Calculate denominator, summing over N
        Aanom2 = np.power(Aanom,2)
        denom = np.sum(Aanom2,axis=a_axis)    
        
        # Calculate Beta
        beta = Aanom @ Banom / denom
            
        
        b = (np.sum(B,axis=b_axis) - beta * np.sum(A,axis=a_axis))/A.shape[a_axis]
    
    
    return beta,b

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

def find_nan(data,dim):
    """
    For a 2D array, remove any point if there is a nan in dimension [dim]
    
    Inputs:
        1) data: 2d array, which will be summed along last dimension
        2) dim: dimension to search along. 0 or 1.
    Outputs:
        1) okdata: data with nan points removed
        2) knan: boolean array with indices of nan points
        

    """
    
    # Sum along select dimension
    datasum = np.sum(data,axis=dim)
    
    
    # Find non nan pts
    knan  = np.isnan(datasum)
    okpts = np.invert(knan)
    
    if dim == 0:
        okdata = data[:,okpts]
    elif dim == 1:    
        okdata = data[okpts,:]
    
    return okdata,knan,okpts
    
    
def year2mon(ts):
    """
    Separate mon x year from a 1D timeseries of monthly data
    """
    ts = np.reshape(ts,(int(np.ceil(ts.size/12)),12))
    ts = ts.T
    return ts
    
def ann_avg(ts,dim):
    """
    # Take Annual Average of a monthly time series
    where time is axis "dim"
    
    """
    tsshape = ts.shape
    ntime   = ts.shape[dim] 
    newshape =    tsshape[:dim:] +(int(ntime/12),12) + tsshape[dim+1::]
    annavg = np.reshape(ts,newshape)
    annavg = np.nanmean(annavg,axis=dim+1)
    return annavg

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

    


def array_nan_equal(a, b):
    """
    Check if two arrays are equal ignoring nans
    source: https://stackoverflow.com/questions/49245963/how-to-compare-numpy-arrays-ignoring-nans
    """
    m = np.isfinite(a) & np.isfinite(b)
    return np.array_equal(a[m], b[m])


def detrend_dim(invar,dim):
    
    """
    Detrends n-dimensional variable [invar] at each point along axis [dim].
    Performs appropriate reshaping and NaN removal, and returns
    variable in the same shape+order. Assumes equal spacing along [dim] for 
    detrending
    
    Also outputs linear model and coefficients.
    
    Dependencies: 
        numpy as np
        find_nan (function)
        regress_2d (function)
    
    Inputs:
        1) invar: variable to detrend
        2) dim: dimension of axis to detrend along
        
    Outputs:
        1) dtvar: detrended variable
        2) linmod: computed trend at each point
        3) beta: regression coefficient (slope) at each point
        4) interept: y intercept at each point
    
    
    """
    
    # Reshape variable
    varshape = invar.shape
    
    # Reshape to move time to first dim
    newshape = np.hstack([dim,np.arange(0,dim,1),np.arange(dim+1,len(varshape),1)])
    newvar = np.transpose(invar,newshape)
    
    # Combine all other dims and reshape to [time x otherdims]
    tdim = newvar.shape[0]
    otherdims = newvar.shape[1::]
    proddims = np.prod(otherdims)
    newvar = np.reshape(newvar,(tdim,proddims))
    
    # Find non nan points
    varok,knan,okpts = find_nan(newvar,0)
    
    # Ordinary Least Squares Regression
    tper = np.arange(0,tdim)
    m,b = regress_2d(tper,varok)
    
    # Detrend
    ymod = (m[:,None]*tper + b[:,None]).T
    dtvarok = varok - ymod
    
    # Replace into variable of original size
    dtvar  = np.zeros(newvar.shape) * np.nan
    linmod = np.copy(dtvar)
    beta   = np.zeros(okpts.shape) * np.nan
    intercept = np.copy(beta)
    
    dtvar[:,okpts] = dtvarok
    linmod[:,okpts] = ymod
    beta[okpts] = m
    intercept[okpts] = b
    
    # Reshape to original size
    dtvar  = np.reshape(dtvar,((tdim,)+otherdims))
    linmod = np.reshape(linmod,((tdim,)+otherdims))
    beta = np.reshape(beta,(otherdims))
    intercept = np.reshape(beta,(otherdims))
    
    # Tranpose to original order
    oldshape = [dtvar.shape.index(x) for x in varshape]
    dtvar = np.transpose(dtvar,oldshape)
    linmod = np.transpose(linmod,oldshape)
    
    return dtvar,linmod,beta,intercept
    
def init_map(bbox,ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    return ax

#%% ----------------------------------------------------------------------------
# Set data paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'  
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200818/'

# Path to SST data from obsv
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"


forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$")


# User settings (note, need to select better region that ignores stuff in the pacific)
bbox  = [-80,0,0,60]

# Mapping Box size
mbbox = [-100,0,0,90]

# # Filter settings: https://iescoders.com/time-series-analysis-filtering-or-smoothing-the-data/
# order   = 5
# fs      = 1/(30*24*3600) # sampling frequency in Hz
# nyquist = fs/2           # Nyquist is half thesampling frequency; 
# timecut = 120            # in Months
# cutoff  = 1/timecut/2     # desired cutoff frequency of the filter, Hz


# More filter settings rewritten (based on old matlab code)
order = 5
cutofftime = 10

#print('cutoff = ',1/cutoff*nyquist*30*24*3600,' months')


# Experiment Options
funiform = 0
runid = "001"
nyrs = 1000


# ----------------------------------------------------------------------------
#%% Script Start 


hvarmode = np.arange(0,4,1)
hvarnames = ("$h_{fixed}$","$h_{max}$","$h_{clim}$","Entrain")





# Load in Model SST [ lon x lat x time]
sst    = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s.npy"%(nyrs,funiform,runid),allow_pickle=True).item()
sst[3] = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s.npy"%(nyrs,funiform,runid))


# Load in Lat.Lon
lon      = np.load(datpath+"lon.npy")
lat      = np.load(datpath+"lat.npy")

# Load in forcing and lat/lon
# if funiform == 2:
#     F        = np.load(datpath+"stoch_output_%iyr_funiform%i_run%s_Forcing.npy"%(nyrs,funiform,runid),allow_pickle=True).item()
# else:
#     F        = np.load(datpath+"stoch_output_%iyr_funiform%i_run%s_Forcing.npy"%(nyrs,funiform,runid),allow_pickle=True)






#%% Calculate AMV Index

amv = {}
aa = {}
annsst = {}

# Load in dat for each mode and compute amv index
for mode in hvarmode:
    print(mode)
    #sst[mode] = np.load(datpath+"stoch_output_1000yr_funiform%i_entrain0_hvar%i.npy" % (funiform,mode))
    annsst[mode] = proc.ann_avg(sst[mode],2)

    # Calculate AMV Index
    amv[mode],aa[mode] = proc.calc_AMV(lon,lat,annsst[mode],bbox,order,cutofftime,1)


# # Repeat, but for forcing
# if funiform == 2:
#     lpforcing = {}
#     aaforcing = {}
#     for mode in hvarmode:
        
#         annforcing = ann_avg(F[mode],2)
#         lpforcing[mode],aaforcing[mode] = calc_AMV(lon,lat,annforcing,bbox,order,cutofftime,1)
# else:
#     annforcing = ann_avg(F,2)
#     lpforcing,aaforcing = calc_AMV(lon,lat,annforcing,bbox,order,cutofftime,1)

#%% Regress back to SST
regr_meth2 = {}

# Perform regression for each mode
for mode in hvarmode:
    
    regr_meth2[mode]=proc.regress2ts(annsst[mode],amv[mode]/np.nanstd(amv[mode]),0,1)


#----------------------------------------
# %% For Comparison, Repeat for Observations
# ----------------------------------------
# Path to SST data from obsv
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")

hyr  = obvhad['YR']





#----------------------------------------
# Load in data (calculated with hadisst proc)
hadnc = xr.open_dataset(datpath2+"HadISST_Detrended_Deanomalized_1920_2018.nc")
hlon = hadnc.lon.values
hlat = hadnc.lat.values
hsst = hadnc.SST.values

# Take annual average
annhsst = proc.ann_avg(hsst,2)


# Take average from amv
#h_amv,aa_hsst = proc.calc_AMV(hlon,hlat,annhsst,bbox,order,cutofftime,1)  

#h_regr=proc.regress2ts(annhsst,h_amv/np.nanstd(h_amv),0,1)


h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)

# %% Perform psd


# User defined settings
fs = 1/(3600*24*30) # Sampling Frequency (1 month in seconds0
#xtk     = [fs/1200,fs/120,fs/12,fs,fs*30]
#xtklabel = ['century','decade','year','mon',"day"]

xtk     = [fs/1200,fs/120,fs/12,fs]
xtklabel = ['century','decade','year','mon']

pxx = {}
freqs = {}
for mode in hvarmode:
    freqs[mode],pxx[mode] = signal.periodogram(aa[mode],fs)


freq,pxxentrain = signal.periodogram(aaentrain,fs)

freqhad,pxxhad = signal.periodogram(aa_hsst,fs)


freqforce,pxxforce = signal.periodogram(aaforcing,fs)


# Plot for HadlISST
fig,ax = plt.subplots(figsize=(6,3))
plt.style.use("ggplot")
ax.plot(freqhad,pxxhad)
ax.set_xlabel('Period [s]')
ax.set_ylabel('PSD [degC**2/Hz')
ax.set_xscale('log')
ax.set_ylim([1e-1,1e11])
ax.set_yscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtklabel)
ax.set_title("HadlISST Area-Average SST Anomaly Periodogram")

# Make Secondary xAxis
secax = ax.secondary_xaxis('top')
secax.set_xscale('log')
secax.set_xlabel('Frequency [Hz]')
secax.set_xticks(xtk)
secax.set_xticklabels(xtk)

plt.tight_layout()

plt.savefig("%sPeriodogram_Hadlisst.png"%outpath,dpi=200)




# Plot for no entrain (hvar2), entrain, and forcing
fig,ax = plt.subplots(figsize=(6,3))
plt.style.use("ggplot")
l1 = ax.plot(freqforce,pxxforce,label="Forcing")
l2 = ax.plot(freq,pxxentrain,label="Entrain")
l3 = ax.plot(freqs[2],pxx[2],label="No-Entrain")

ax.set_xlabel('Period [s]')
ax.set_ylabel('PSD [degC**2/Hz')
ax.set_xscale('log')
ax.set_ylim([1e-1,1e11])
ax.set_yscale('log')
ax.set_xticks(xtk)
ax.set_xticklabels(xtklabel)
ax.set_title("Stochastic Model Area-Average SST Anomaly Periodogram")
plt.legend()

# Make Secondary xAxis
secax = ax.secondary_xaxis('top')
secax.set_xscale('log')
secax.set_xlabel('Frequency [Hz]')
secax.set_xticks(xtk)
secax.set_xticklabels(xtk)

plt.tight_layout()
plt.savefig("%sPeriodogram_Stochmod_NoEntrain.png"%outpath,dpi=200)



#%%
# -------------------------------
# Plot AMV from the 4 experiments
# -------------------------------

xrange = [100,300]
htimeall = np.arange(0,len(amv[1]))
hticks = np.arange(0,1100,100)
fig,ax = plt.subplots(4,1,sharex=True,sharey=True,figsize=(4,8))
for mode in range(4):
    plt.style.use("ggplot")
    
    plt.subplot(4,1,mode+1)
    
    #plt.plot(htimeall,aa[mode],color=[0.75,0.75,0.75])
    viz.plot_AMV(amv[mode])
    
    
    
    #plt.ylim([-1.5,1.5])
    #plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
    #plt.yticks(np.arange(-1.5,2.0,0.5),fontsize=16)
    plt.title("AMV Index for %s" % hvarnames[mode],fontsize=12)
    plt.xlim(xrange)
    maxval = np.max(np.abs(amv[mode]))
    plt.ylim([maxval*-1,maxval])
    #ax[mode+1].tick_params(labelsize=16)

    

    

fig.text(0.04, 0.5, 'AMV Index for %s' % forcingname[funiform], va='center', rotation='vertical',fontsize=12)
plt.xlabel("Years",fontsize=12)
plt.tight_layout(rect=[.05,.025,0.95,0.95])
outname = outpath+'AMV_Index_uniform%i.png'%(funiform)
plt.savefig(outname, bbox_inches="tight",dpi=200)


    
    
# -------------------------------
# %%Compare AMV to forcing
# -------------------------------



# # Add forcing as first plot
# fig,ax = plt.subplots(2,1,sharex=True,sharey=False,figsize=(14,6))
# plt.style.use("ggplot")
# plt.subplot(2,1,1)
# if funiform == 2:
#     plot_AMV(lpforcing[2])
# else:
#     plot_AMV(lpforcing)
# #plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
# #plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
# #plt.yticks(np.arange(-1.5,2.0,0.5),fontsize=16)
# plt.title("Forcing (LP-Filtered)",fontsize=20)
# plt.ylabel("Forcing Term K/s",fontsize=16)


# plt.subplot(2,1,2)
# model = 2
# plot_AMV(amv[2])
# #plt.ylim([-1,1])
# #plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
# #plt.yticks(np.arange(-1,1.5,0.5),fontsize=16)
# plt.title("AMV Index for MLD %s" % hvarnames[mode],fontsize=20)
# plt.ylabel("AMV Index",fontsize=16)
# plt.tight_layout()
# outname = outpath+'AMV_hvar_forcing_comparison_funiform%i.png' % (funiform)
# plt.savefig(outname, bbox_inches="tight",dpi=200)

    
    
    
# -------------------------------
#%% AMV Spatial Pattern Plots
# -------------------------------



   # ax.set_title("AMV-related SST Pattern from HadISST, %i-%i"%(startyr,hyr[0,-1]))


# Set up plot
# Set colormaps and contour intervals
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
#cint = np.arange(-1,1.1,0.1)
#cint = np.arange(-10,10,1)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
#cint = np.arange(-2,2.5,0.5)
#cint = np.arange(-1,)

#cint = np.arange(-2,2,0.2)
#clab = cint
for mode in hvarmode:
    print(mode)
    fig,ax = plt.subplots(1,1,figsize=(3,1.5))
    plt.style.use("ggplot")
    
    varin = np.transpose(regr_meth2[mode],(1,0))
    
    plt.subplot(1,4,mode+1)
    viz.plot_AMV_spatial(varin,lon,lat,bbox,cmap,pcolor=0)
    #viz.plot_AMV_spatial(varin,lon,lat,mbbox,cmap,cint,clab)
    #plt.title("AMV Spatial Pattern \n Model %s" % hvarnames[mode],fontsize=10)
    plt.title("Model %s" % hvarnames[mode],fontsize=12)
    outname = outpath+'AMVpattern_funiform%i_hvar%i_run%s.png' % (funiform,mode,runid)
    plt.savefig(outname, bbox_inches="tight",dpi=200)
    


#%% Plot AMV FFor HADLISST

startyr = 1900
# Same plot, but for HadlISST
cmap = cmocean.cm.balance
#cint = np.arange(-3.5,3.25,0.25)
cint = np.arange(-0.8,1.0,0.2)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
clab = cint

fig,ax = plt.subplots(1,1,figsize=(3,1.5))
plt.style.use("ggplot")

varin = np.transpose(h_regr,(1,0))

plt.subplot(1,1,1)
viz.plot_AMV_spatial(varin,hlon,hlat,mbbox,cmap,cint,clab)
plt.title("AMV-related SST Pattern from HadISST, %i-%i"%(startyr,hyr[0,-1]),fontsize=8)
outname = outpath+'AMVpattern_HADLISST.png' 
plt.savefig(outname, bbox_inches="tight",dpi=200)


#%% Plot observation AMV time series




startyr = 1900


fig,ax = plt.subplots(1,1,figsize=(3,1.5))
plt.style.use("ggplot")

varin = np.transpose(h_regr,(1,0))

plt.subplot(1,1,1)
viz.plot_AMV(h_amv)
plt.title("AMVIndex (HadISST), %i-%i"%(startyr,hyr[0,-1]),fontsize=8)
plt.xlabel('Years')
plt.ylabel('AMV Index')
outname = outpath+'AMVidx_HADLISST.png' 
plt.savefig(outname, bbox_inches="tight",dpi=200)


#%% Plot AMV FFor entrain case


cmap = cmocean.cm.balance
#cint = np.arange(-3.5,3.25,0.25)
#cint = np.arange(-3,3.25,0.25)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
cint=np.arange(-1,1.1,0.1)
#cint = [0]
clab = cint


fig,ax = plt.subplots(1,1,figsize=(8,4))
plt.style.use("ggplot")

varin =regr_meth2[3].T

plt.subplot(1,1,1)
plot_AMV_spatial(varin,lon,lat,mbbox,cmap,cint,clab)
plt.title("AMV-related SST Pattern, Entrain Case")
outname = outpath+'AMVpattern_funiform%i_entrain_run%s.png'  % (funiform,runid)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%% Spatial AMV Pattern and AMV Time Series Plots...

fig = plt.figure(constrained_layout=True,figsize=(8,6))
gs  = fig.add_gridspec(3,3)

# Make AMV Spatial Pattern Map
ax1 = fig.add_subplot(gs[0:2,:],projection=ccrs.PlateCarree())
plot_AMV_spatial(varin,lon,lat,mbbox,cmap,cint,clab,ax=ax1)
#ax1.set_title("AMV-related SST Pattern, Entrain Case")


ax2 = fig.add_subplot(gs[2,:])
plot_AMV(amv[3],ax=ax2)
ax2.set_title("AMV Index")
ax2.set_xlabel("Years")



#%% Havent Fixed Things Below this line..


#%% Try to animate a forcing map


from matplotlib.animation import FuncAnimation

import matplotlib.animation as animation



# Prepare variables
invar = np.transpose(np.copy(sstentrain),(1,0,2))
frames = 100 #Indicate number of frames

# Define Figure to create base map for plotting
def make_figure(bbox):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    
    return fig,ax
    
    

fig,ax = make_figure(bbox) # Make the basemap



def draw(lon,lat,invar,frame,add_colorbar):
    ax = plt.gca()
    plotvar = invar[...,frame] # Assume dims [lonxlatxtime]
    pcm     = plt.pcolormesh(lon,lat,plotvar,cmap=cmocean.cm.balance)
    title   = "t = %i" % frame
    ax.set_title(title)
    if add_colorbar==True:
        cbar  = plt.colorbar(pcm,ax=ax,orientation='horizontal')
    return pcm
    

# # Indicate initial conditions
def drawinit():
    return draw(lon,lat,invar,0,add_colorbar=True)

# Indicate other conditions
def animate(frame):
    return draw(lon,lat,invar,frame,add_colorbar=False)

ani = animation.FuncAnimation(fig, animate,frames,interval=0.1,blit=False,init_func=drawinit,repeat=False)

ani.save("%stest_anim_sstentrain.mp4"%(outpath),writer=animation.FFMpegWriter(fps=8))
plt.close(fig)


#%% Annual Average Plots


def ann_avg(ts):
    """
    # Take Annual Average of a time series
    where time is the first dimension (in months)
    
    """
    nyrs   = ts.shape[0] 
    annavg = np.reshape(ts,(int(nyrs/12),12))
    #annavg = np.reshape(ts,(12,int(nyrs/12)))
    annavg = np.nanmean(annavg,axis=1)
    return annavg


anav_entrain      = ann_avg(aaentrain)
anav_forcing      = ann_avg(aaforcing)
anav_noentrain_h2 = ann_avg(aa[2])
pltyr = np.arange(0,anav_entrain.shape[0])


# Add forcing as first plot
fig,ax = plt.subplots(3,1,sharex=True,sharey=False,figsize=(14,8))
plt.style.use("ggplot")
plt.subplot(3,1,1)
plt.plot(pltyr,anav_forcing)
plt.xlim(0,1000)
#plt.xticks(np.arange(0,),hticks,fontsize=16)
#plt.xticks(np.arange(0,len(amv[1])+1200,1200),hticks,fontsize=16)
#plt.yticks(np.arange(-0.25,0.50,0.25),fontsize=16)
plt.title("Forcing, Ann. Avg",fontsize=20)
plt.ylabel("Forcing Term K/s",fontsize=16)


plt.subplot(3,1,2)
model = 2
plt.plot(pltyr,anav_noentrain_h2)
plt.ylim([-5,5])
plt.xlim(0,1000)
plt.title("No Entrain, Seasonally Varying h",fontsize=20)
plt.ylabel("degC",fontsize=16)



plt.subplot(3,1,3)
model = 2
plt.plot(pltyr,anav_entrain)
plt.ylim([-5,5])
plt.xlim(0,1000)
plt.title("Entrain",fontsize=20)
plt.ylabel("degC",fontsize=16)


plt.tight_layout()
outname = outpath+"AnnAvgComparison_funiform%i.png" % (funiform)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%% Mean Spatial Plots

#entrainmean =np.nanmean(sst[2],axis=2).T # Stochmod Hvar
entrainmean = np.nanmean(sstentrain,axis=2).T
emmax = np.nanmax(np.abs(entrainmean))


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())

# Set extent
ax.set_extent(bbox)

# Add filled coastline
ax.add_feature(cfeature.COASTLINE,facecolor='k')

# Add Gridlines
gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')

gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

pcm     = plt.pcolormesh(lon,lat,entrainmean,cmap=cmocean.cm.balance,vmax=emmax,vmin=-1*emmax)
plt.colorbar(pcm)
plt.title("Mean SST Anomaly from Entrain Model")
outname = outpath+"AvgSST_Entrain_funiform%i.png" % (funiform)
plt.savefig(outname, bbox_inches="tight",dpi=200)



#%% Plot HadISST AMV with bounding boxes

startyr = 1900
# Same plot, but for HadlISST
cmap = cmocean.cm.balance
#cint = np.arange(-3.5,3.25,0.25)
cint = np.arange(-0.8,1.0,0.2)
#clab = np.arange(-0.50,0.60,0.10)
#cint = np.arange(-1,1.2,0.2)
clab = cint

fig,ax = plt.subplots(1,1,figsize=(3,1.5))
plt.style.use("ggplot")

varin = np.transpose(h_regr,(1,0))

plt.subplot(1,1,1)
viz.plot_AMV_spatial(varin,hlon,hlat,mbbox,cmap,cint,clab)
plt.title("AMV-related SST Pattern from HadISST, %i-%i"%(startyr,hyr[0,-1]),fontsize=8)
outname = outpath+'AMVpattern_HADLISST.png' 
plt.savefig(outname, bbox_inches="tight",dpi=200)