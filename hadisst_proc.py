#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:04:03 2020

@author: gliu
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% Functions
def find_latlon(lonf,latf,lon,lat):
    """
    Find lat and lon indices
    """
    if((np.any(np.where(lon>180)) & (lonf < 0)) or (np.any(np.where(lon<0)) & (lonf > 180))):
        print("Potential mis-match detected between lonf and longitude coordinates")
    
    klon = np.abs(lon - lonf).argmin()
    klat = np.abs(lat - latf).argmin()
    
    msg1 = "Closest lon to %.2f was %.2f" % (lonf,lon[klon])
    msg2 = "Closest lat to %.2f was %.2f" % (latf,lat[klat])
    print(msg1)
    print(msg2)
    
    return klon,klat

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
    
    
    return beta,b


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
    

#%%


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


#%% For hsst, flip the latitude axis
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


# Reshape to [Time x Space] and remove NaN Points
hsstnew = np.reshape(hsstnew,(360*180,1176)).T
hsstok,knan,okpts = find_nan(hsstnew,0)


#%% Perform Linear Detrend on SST

start= time.time()
tper = np.arange(0,hsstok.shape[0])
beta,b = regress_2d(tper,hsstok) # Perform regression

# Detrend
dt_hsst = hsstnew[:,okpts] - (beta[:,None] * tper + b[:,None]).T

# Replace NaN vaues back into the system
hsstall = np.zeros(hsstnew.shape) * np.nan
hsstall[:,okpts] = dt_hsst

# Also save the linear model
ymodall = np.zeros(hsstnew.shape) * np.nan
ymodall[:,okpts] = (beta[:,None] * tper + b[:,None]).T

# Reshape again
dt_hsst = np.reshape(hsstall.T,(360,180,1176))
hsstnew = np.reshape(hsstnew.T,(360,180,1176))
ymodall = np.reshape(ymodall.T,(360,180,1176))
print("Detrended in %.2fs" % (time.time()-start))

#%% Visualize detrending for point [lonf,latf]


lonf = -30
latf = 64
klon,klat = find_latlon(lonf,latf,hlon,hlatnew)

tempts = hsstnew[klon,klat,:]
dtts = dt_hsst[klon,klat,:]
ymodts = ymodall[klon,klat,:]

#% Plot Detrended and undetrended lines
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,tempts,color='k',label="raw")
ax.plot(tper,dtts,color='b',label="detrended")
plt.legend()

#% Plot Scatter and fitted model...
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.scatter(tper,tempts,color='k',label="raw")
ax.plot(tper,ymodts,color='b',label="linear model")
plt.legend()

#%% Remove Seasonal Cycle


# Deseasonalize [lon x lat x yr x mon]
ahsst = np.reshape(dt_hsst,(360,180,int(dt_hsst.shape[2]/12),12))
ahsst = ahsst - np.mean(ahsst,axis=3)[:,:,:,None]
ahsst = np.reshape(ahsst,(360,180,dt_hsst.shape[2]))


#%% Plot climatological removal
ats = ahsst[klon,klat,:]

fig,ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(tper,tempts,color='k',label="Raw")
ax.plot(tper,dtts,color='b',label="Detrended")
plt.plot(tper,ats,color='r',label="Detrended,Deseasonalized")
plt.legend()


#%% Save Data

timecft = xr.cftime_range(start="1920-01-01",end="2017-12-01",freq="MS") 

da = xr.DataArray(ahsst,
                  dims=["lon","lat","time"],
                  coords={"lat":hlatnew,"lon":hlon,"time":timecft}
                 )
da.name = 'SST'
da.to_netcdf("%sHadISST_Detrended_Deanomalized_1920_2018.nc" % (datpath2))

aa = xr.open_dataset("%sHadISST_Detrended_Deanomalized_1920_2018.nc" % (datpath2))