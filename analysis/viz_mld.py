#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize and Compare Mixed Layer Depths from CESM

Created on Sun Apr 25 21:18:32 2021

@author: gliu
"""


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx

from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
from scipy import signal
from tqdm import tqdm

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import scm
import time
import cmocean


#%% User Edits

projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20210509/'
input_path  = datpath + 'model_input/'


mconfig = "FULL_PIC"
if mconfig == "FULL_PIC":
    configname = "Fully-Coupled PiC"

bboxplot  = [-100,20,-10,80]

# ------------
#%% Load Data

# Lat and Lon
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp       = loadmat(input_path+dampmat)
lon            = np.squeeze(loaddamp['LON1'])
lat            = np.squeeze(loaddamp['LAT'])

# Stochastic Model Input
if "PIC" in mconfig: # Load Mixed layer variables (preprocessed in prep_mld.py)
    hclim = np.load(input_path+"FULL_PIC_HMXL_hclim.npy")
    kprevall    = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
else: # Load Historical
    hclim         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
    kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# Load Slab MLD
hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    

# Load full MLD field
ds = xr.open_dataset(input_path+"HMXL_PIC.nc")
mldfullpic = ds.HMXL.values

# Reshape to separate month and season
nlat,nlon,ntime = mldfullpic.shape
nyr = int(ntime/12)
mldfullpic = mldfullpic.reshape(nlat,nlon,nyr,12)

# --------------------------------
#%% Plot Maximum Mixed layer depth
# --------------------------------

vtype = "Max"
invar = hclim

if vtype == "Max":
    invar = invar.max(2)
elif vtype == "Min":
    invar = invar.min(2)
elif vtype == "Range":
    invar = invar.max(2) - invar.min(2)

cint1 = [0,10,25,50,100,500]
cint2 = [1000,1500,2000]
#cint = [0,25,50,100,250,500,750,1000,1250,1500]
clevs = np.arange(0,1600,100)
cmap = cmocean.cm.dense

# Old Figsize was 5 x 4
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,5))
ax = viz.add_coast_grid(ax,bbox=bboxplot)

pcm = ax.pcolormesh(lon,lat,invar.T,vmin=clevs[0],vmax=clevs[-1],cmap=cmap)
pcm1 = ax.contourf(lon,lat,invar.T,levels=clevs,cmap=cmap)

cl1 = ax.contour(lon,lat,invar.T,cint1,colors="k",linewidths = .5)
ax.clabel(cl1,fmt="%i",fontsize=10)

# cl2 = ax.contour(lon,lat,invar.T,cint2,colors="w",linewidths = .5)
# ax.clabel(cl2,fmt="%i",fontsize=9)

#ax.add_feature(cfeature.LAND,color='gray')
#ax.set_title("$MLD_{max} - MLD_{min}$" + "\n 40-member Ensemble Average",fontsize=12)
ax.set_title("%s MLD of Mean Annual Cycle " % vtype + "\nContour Interval = %i m"%(clevs[1]-clevs[0]),fontsize=12)
fig.colorbar(pcm1,ax=ax)
#plt.colorbar(pcm1,ax=ax,orientation='horizontal',fraction=0.040, pad=0.10)
#plt.tight_layout()
plt.savefig(outpath+"MLD_%s_%s.png"%(mconfig,vtype),dpi=200)

# -------------------------------------------------------------------------------
# %% Calculate the Mixed Layer Depth variability over a particular set of seasons
# -------------------------------------------------------------------------------

mldfullwint = mldfullpic[:,:,:,[0,1,2,11]]

mldfullwint = mldfullwint.reshape(nlat,nlon,nyr*4)

wintstd = mldfullwint.std(2)/100


clevs = np.arange(0,650,50)
cint1 = [0,50,100,200]

# Old Figsize was 5 x 4
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,5))
ax = viz.add_coast_grid(ax,bbox=bboxplot)

pcm  = ax.pcolormesh(lon,lat,wintstd.T,cmap=cmocean.cm.dense)
pcm1 = ax.contourf(lon,lat,wintstd.T,levels=clevs,cmap=cmocean.cm.dense)
fig.colorbar(pcm1,ax=ax)

cl1 = ax.contour(lon,lat,wintstd.T,cint1,colors="k",linewidths = .5)
ax.clabel(cl1,fmt="%i",fontsize=9)

ax.set_title("1$\sigma$ MLD (DJFM, %s)"% configname + "\nContour Interval = %i m"%(clevs[1]-clevs[0]),fontsize=12)
plt.savefig(outpath+"MLD_DJFM_std_%s.png"%(mconfig),dpi=200)


# ----------------------------------
#%% Load and map heat flux feedback
# ----------------------------------

damping = np.load(input_path+"SLAB_PIC"+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")

vtype = "Max"
invar = damping

if vtype == "Max":
    invar = invar.max(2)
    cint1 = [0,10,20,30,40]#[0,10,25,50,100,500]
elif vtype == "Min":
    invar = invar.min(2)
elif vtype == "Range":
    invar = invar.max(2) - invar.min(2)
    
    cint1 = [0,5,10]#[0,10,25,50,100,500]
    cint2 = [0]#[1000,1500,2000]

#cint1 = [0,5,10]#[0,10,25,50,100,500]
#cint2 = [0]#[1000,1500,2000]
clevs = np.arange(-50,55,5)
cmap = cmocean.cm.balance

# Old Figsize was 5 x 4
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,5))
ax = viz.add_coast_grid(ax,bbox=bboxplot)

pcm = ax.pcolormesh(lon,lat,invar.T,vmin=clevs[0],vmax=clevs[-1],cmap=cmap)
pcm1 = ax.contourf(lon,lat,invar.T,levels=clevs,cmap=cmap)

cl1 = ax.contour(lon,lat,invar.T,cint1,colors="k",linewidths = .5)
ax.clabel(cl1,fmt="%i",fontsize=10)

# cl2 = ax.contour(lon,lat,invar.T,cint2,colors="w",linewidths = .5)
# ax.clabel(cl2,fmt="%i",fontsize=9)

#ax.add_feature(cfeature.LAND,color='gray')
#ax.set_title("$MLD_{max} - MLD_{min}$" + "\n 40-member Ensemble Average",fontsize=12)
ax.set_title("%s Heat Flux Damping" % vtype + "\nContour Interval = %i $W/m^{2}/^{\circ}C$"%(clevs[1]-clevs[0]),fontsize=12)
fig.colorbar(pcm1,ax=ax)
#plt.colorbar(pcm1,ax=ax,orientation='horizontal',fraction=0.040, pad=0.10)
#plt.tight_layout()
plt.savefig(outpath+"Damping_%s_%s.png"%(mconfig,vtype),dpi=200)


##
#%% Load and Map Forcing
#

forcing = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")

vtype = "Max"
invar = forcing

if vtype == "Max":
    invar = invar.max(2)
    cint1 = [25,50,75,100]#[0,10,25,50,100,500]
    clevs = np.arange(0,75,5)
elif vtype == "Min":
    invar = invar.min(2)
    
    cint1 = [5,10,15,20,25]
    clevs = np.arange(0,31,3)
elif vtype == "Range":
    
    invar = invar.max(2) - invar.min(2)
    cint1 = [20,40]#[0,10,25,50,100,500]
    clevs = np.arange(0,55,5)

#cint1 = [20,40]#[0,10,25,50,100,500]
#cint2 = [0]#[1000,1500,2000]
#clevs = np.arange(0,55,5)
cmap = cmocean.cm.thermal

# Old Figsize was 5 x 4
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,5))
ax = viz.add_coast_grid(ax,bbox=bboxplot)

pcm = ax.pcolormesh(lon,lat,invar.T,vmin=clevs[0],vmax=clevs[-1],cmap=cmap)
pcm1 = ax.contourf(lon,lat,invar.T,levels=clevs,cmap=cmap)

cl1 = ax.contour(lon,lat,invar.T,cint1,colors="k",linewidths = .5)
ax.clabel(cl1,fmt="%i",fontsize=10)

# cl2 = ax.contour(lon,lat,invar.T,cint2,colors="w",linewidths = .5)
# ax.clabel(cl2,fmt="%i",fontsize=9)

#ax.add_feature(cfeature.LAND,color='gray')
#ax.set_title("$MLD_{max} - MLD_{min}$" + "\n 40-member Ensemble Average",fontsize=12)
ax.set_title("%s Stochastic Forcing Amplitude" % vtype + "\nContour Interval = %i $W/m^{2}$"%(clevs[1]-clevs[0]),fontsize=12)
fig.colorbar(pcm1,ax=ax)
#plt.colorbar(pcm1,ax=ax,orientation='horizontal',fraction=0.040, pad=0.10)
#plt.tight_layout()
plt.savefig(outpath+"Forcing_%s_%s.png"%(mconfig,vtype),dpi=200)



#%% Tinker Ranges



#%% Plot all 3 together


invars = [damping,forcing,hclim]
vtype  = "Range"
cmaps  = [cmocean.cm.balance,cmocean.cm.thermal,cmocean.cm.dense]
cstep  = [5,5,100]
cints  = []


fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(24,5))

for i in range(3):
    
    # Get Variable
    invar = invars[i]
    if vtype == "Max":
        invar = invar.max(2)
    elif vtype == "Min":
        invar = invar.min(2)
    elif vtype == "Range":
        invar = invar.max(2) - invar.min(2)
    
    # Get Plotting Params
    cmap = cmaps[i]
    
    # Start Plotting
    ax = axs.flatten()[i]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
    
    
    
    



