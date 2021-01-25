#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:30:52 2021

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
from dask.distributed import Client,progress
import dask

import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature

import matplotlib.colors as mc
from tqdm import tqdm

#%% User Edits

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_clim/"
lonpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat"
mskpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
bbox    = [-100,20,-20,90]

snames = ["Variance","Std. Dev.","Mean"]
mons3  = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
#%%

def quickloader(vname,mconfig,datpath):
    ld = np.load("%s%s_%s_stats.npz"%(datpath,vname,mconfig),allow_pickle=True)
    #print(ld.files)
    vvar = ld['var']
    vstd = ld['std']
    vavg = ld['avg']
    return vvar,vstd,vavg

#%% Script Start

# Load Lat/Lon
lm = loadmat(lonpath)
lat = lm['LAT'].squeeze()
lon = lm['LON'].squeeze()

# Load Land/ice Mask
msk = np.load(mskpath)

# Load Data
vname   = 'NHFLX'
mconfig = 'SLAB'
vstats = quickloader(vname,mconfig,datpath)

# For each variable, flip the longitude
vstatreg  = []
vstatglob = []
vmaxmon   = []
maxmonid  = []
maxmonidglob  = [] 
for v in vstats:
    # Find month of maximum
    kmon   = np.argmax(v,axis=0).flatten()
    maxmonidglob.append(kmon)
    
    # Preprocess (flip longitude, cut to region)
    v = v.transpose(2,1,0) * (msk.T)[:,:,None]
    lon1,v1 = proc.lon360to180(lon,v)
    vrr,lonr,latr = proc.sel_region(v1,lon1,lat,bbox=bbox)
    
    # Do the same for the mask
    _,msk180 = proc.lon360to180(lon,msk.T)
    mskr,_,_ = proc.sel_region(msk180,lon1,lat,bbox=bbox)
    
    # Find the month of maximum
    vrrmax = np.argmax(vrr,axis=2).T+1 * mskr.T
    kmonr = np.argmax(vrr,axis=2).flatten()
    
    vstatglob.append(v1)
    vstatreg.append(vrr)
    vmaxmon.append(vrrmax)
    maxmonid.append(kmonr)
    
#%% Save Variable as input into stochastic Model

nhflxstd = vstatglob[1]
np.save("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy",nhflxstd)

#%%

# Sample Global plot
i = 1
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax =viz.init_map([-180,180,-90,90],ax=ax)
pcm=ax.contourf(lon1,lat,np.argmax(vstatglob[i],axis=2).T+1*msk180.T,levels=np.arange(0,13,1),cmap='twilight')
#pcm=ax.pcolormesh(lonr,latr,(vstatreg[1][:,:,0]).T,cmap='twilight')
plt.colorbar(pcm,ax=ax,orientation='horizontal')
ax.set_title("Month of Maximum %s (%s)" % (snames[i],vname))
plt.savefig(datpath+"MonMax_%s_%s_%s_global.png"% (vname,mconfig,snames[i]) ,dpi=200)

# Sample regional plot
i = 1
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.init_map(bbox,ax=ax)
pcm=ax.contourf(lonr,latr,vmaxmon[1],levels=np.arange(0,13,1),cmap='twilight')
#pcm=ax.pcolormesh(lonr,latr,(vstatreg[1][:,:,0]).T,cmap='twilight')
plt.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.05)
ax.set_title("Month of Maximum %s (%s)" % (snames[i],vname))
plt.savefig(datpath+"MonMax_%s_%s_%s_NAtl.png"% (vname,mconfig,snames[i]) ,dpi=200)

#%%
# Load autocorrelation from slab model
autocorr            = np.load(datpath+"TS_SLAB_Autocorrelation.npy") # mon x lag x lat x lon
nmon,nlag,nlat,nlon = autocorr.shape

# Preprocess
autocorr1 = autocorr.transpose(3,2,1,0) # lon x lat x lag x mon
autocorr1 = autocorr1.reshape(nlon,nlat,nmon*nlag)
_,autocorr1 = proc.lon360to180(lon,autocorr1)
autocorr1,_,_ = proc.sel_region(autocorr1,lon1,lat,bbox)
nlonr,nlatr,_ = autocorr1.shape
autocorr1 = autocorr1.reshape(nlonr,nlatr,nlag,nmon) * mskr[:,:,None,None]
autocorr1 = autocorr1.reshape(nlonr*nlatr,nlag,nmon)
autocorr1 = autocorr1.transpose(2,1,0)

zz = 0       # Couter to make sure you covered all the points
klagmon = 1  # Index for month at lag 0
i    = 1     # Stats variable index (see snames)
xtk2 = np.arange(0,37,2) # Lags
lags = np.arange(0,37,1)
for kmon in tqdm(np.arange(0,12,1)):
    
    # Get indices (kmon=month of maximum forcing)
    indices  = np.where(maxmonid[i]==kmon)
    acmon    = autocorr1[klagmon,:,indices].squeeze()
    
    # autocorr = autocorr.reshape(nmon,nlag,nlat*nlon)
    # indices  = np.where(maxmonid[i]==kmon)
    # acmon = autocorr[kmon,:,indices].squeeze()
    
    # Autocorrelation plot
    fig,ax = plt.subplots(1,1)
    ax,ax2 = viz.init_acplot(klagmon,xtk2,lags,ax=ax)
    for n in range(acmon.shape[0]):
        ax.plot(lags,acmon[n,:37],alpha=0.1,color='k',label="")
    ax.plot(lags,acmon.mean(0)[:37],color='r',label="Average")
    ax.set_title("SST Autocorrelation for Points \n with Max %s (%s) in %s (%i points),\n Lag 0 = %s" % (snames[i],vname,mons3[kmon],acmon.shape[0],mons3[klagmon]))
    
    #ax.grid(True,ls='dotted')
    #ax.set_xlim(0,36)
    #ax.set_xticks(np.arange(0,36,3))
    plt.tight_layout()
    plt.savefig(datpath+"NAtl_AC_%s_%s_%s_maxmon%i.png"% (vname,mconfig,snames[i],kmon) ,dpi=200)
    
    
    # Points Plot
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.init_map(bbox,ax=ax)
    cl=ax.contour(lonr,latr,vmaxmon[1],levels=np.arange(kmon+1,kmon+2,1),colors='k',linestyles='dashed')
    #cl = ax.contour(lonr,latr,vmaxmon[1],levels=np.arange(0,13,1),colors='k',linestyles='solid')
    ax.clabel(cl,fontsize=8,fmt="%i")
        
    pcm1=ax.contourf(lonr,latr,vmaxmon[1],levels=np.arange(0,14,1),cmap='twilight')
    #pcm=ax.pcolormesh(lonr,latr,(vstatreg[1][:,:,0]).T,cmap='twilight')
    plt.colorbar(pcm1,ax=ax,orientation='vertical',fraction=0.05)
    ax.set_title("Month of Maximum %s (%s), (Month %i outlined)" % (snames[i],vname,kmon+1))
    #plt.tight_layout()
    plt.savefig(datpath+"NAtl_Pointloc_%s_%s_%s_maxmon%i.png"% (vname,mconfig,snames[i],kmon) ,dpi=200)

    #print(kmon)
    zz += acmon.shape[0]




#%%
# Take Along Axis (Incorrect Method? I just needed to index specific months..)
indices = np.tile(maxmonid[i][:,None],nlag)
#monid = maxmonid[i][:,None]
acsel = np.take_along_axis(autocorr,indices.T[None,:,:],axis=2)

#autocor = autocor.transpose(0,1,3,2)
#indices = maxmonid[i]

