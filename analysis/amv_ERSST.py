#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate/Preprocess SST from ERSST Dataset
Created on Thu May 27 21:21:36 2021

@author: gliu
"""

from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs
import xarray as xr


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%% User Input

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"
ncname  = "ERSST5.nc"
datpathout = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"

# Detrending Options
method  = 2
tstart = '1900-01-01'
tend   = '2016-12-31'


#def slice_time()

#%% Load data, slice time

# Open Dataset
ds = xr.open_dataset(datpath+ncname) # [jan 1854 to nov 2018]
dstime = ds.sel(time=slice(tstart,tend))

# Load in NumPy Arrays
sst   = dstime['sst'].values
lat   = dstime['lat'].values
lon   = dstime['lon'].values
times = dstime['time'].values 

# Get dimensions
nmon,nlat,nlon = sst.shape
nyrs = int(nmon/12)
#%% Preprocess the data

# Flip array
sstflip = np.flip(sst,axis=1)
latnew  = np.flip(lat)

# def flip_lat(hlat,hsst):
#     """
#     Flip latitude of variable goes from 90 to -90

#     Parameters
#     ----------
#     hlat : TYPE
#         DESCRIPTION.
#     hsst : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     hlatnew : TYPE
#         DESCRIPTION.
#     hsstnew : TYPE
#         DESCRIPTION.

#     """
#     # Find north and south latitude points
#     hsouth = np.where(hlat <= 0)
#     hnorth = np.where(hlat > 0)
    
#     # Find corresponding points in data
#     hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
#     hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]
    
#     # Stitch things together, reversing the order 
#     hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
#     hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)
    
#     return hlatnew,hsstnew


#%% Remove Seasonal Cycle first and plot
debug = True

# Flip to lon x lat x time
hsstnew = sstflip.transpose(2,1,0)

# Deseason by computing monthly averages
dsfirst = np.reshape(hsstnew,(nlon,nlat,nyrs,12))
scycle  = np.mean(dsfirst,axis=2)[:,:,None,:]
dsfirst = dsfirst -scycle
dsfirst = np.reshape(dsfirst,(nlon,nlat,nmon))

# Try to also deseason using sinusoid fit
hcopy = hsstnew.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(hcopy,1) # [Space x time]
x,E = proc.remove_ss_sinusoid(okdata.T) # {Time x Space}
ss  = E@x
okdata_ds = (okdata.T - ss).T
dssinu = np.zeros((nlon*nlat,nmon))*np.nan
dssinu[okpts,:] = okdata_ds
dssinu          = dssinu.reshape(nlon,nlat,nmon)

if debug:
    klon,klat = proc.find_latlon(330,55,lon,lat)
    
    sstpt = [dsfirst[klon,klat,:],
             dssinu[klon,klat,:]]
    
    nsmooths = [1,1]
    pct      = 0.10
    specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(sstpt,nsmooths,pct)
    
    
    xlm = [1e-2,5e0]
    xper = np.array([200,100,75,50,25,10,5,1,0.5]) # number of years
    xtks = 1/xper
    xlm  = [xtks[0],xtks[-1]]
    dt   = 3600*24*365
    
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    
    mnames = ["Mon-Anom","Sinusoid"]
    mcols  = ['b','r']
    msty   = ["solid",'dashed']
    for i in range(2):
        ax.plot(freqs[i]*dt,specs[i]/dt,label=mnames[i],color=mcols[i],ls=msty[i])
    ax.set_xlim(xlm)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    
    #viz.plot_freqlin(specs[::-1],freqs[::-1],["Mon-Anom","Sinusoid"],["blue","r"],ax=ax,xtick=xtks,xlm=xlm)
    #fig,ax = plt.subplots(1,1)
    #ax.scatter(np.arange(0,nmon),hsstnew[klon,klat,:] - (hsstnew.mean(-1))[:,:,None],alpha=0.5,color='gray')
    
    

#%%
# Detrend
start= time.time()
indata = dsfirst.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(indata,1)
x = np.arange(0,nmon,1)

if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T

# Replace back into dataset
dtdata = np.zeros((nlon*nlat,nmon))*np.nan
dtdata[okpts,:] = okdt
dtdata = dtdata.reshape(nlon,nlat,nmon)
print("Detrended in %.2fs" % (time.time()-start))



# # Plot Seasonal Cycle Removal and Detrended
lonf = -30+360
latf = 64
tper = np.arange(0,nmon)
klon,klat = proc.find_latlon(lonf,latf,lon,latnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))

ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdata[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

hlat = latnew
hsst = dtdata
hlon = lon
hyr  = times

# Save data (MONTHLY)
hadname  = "%sERSST_detrend%i_startyr%s_endyr%s.npz" % (datpathout,method,tstart[:4],tend[:4])
np.savez(hadname,**{
    'sst':dtdata,
    'lat':latnew,
    'lon':lon,
    'yr':hyr},allow_pickle=True)


#%% Calculate and plot AMV
bbox = [-80,0 ,0,65]
for i in range(2):
    if bbox[i] <= 0:
        bbox[i] += 360

h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)


