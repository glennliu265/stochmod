#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:58:36 2020
viz_tau -> visualize the e-folding timescale in the same manner as Li et al 2020

@author: gliu
"""
from scipy.io import loadmat
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import viz,proc

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%%

#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20200823/'

# Path to damping data
damppath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"

# Set constants
rho = 1025
cp = 3850
dt = 3600*24*30

#%% 



#%% Load in TAU files and process
tauname = "tauall_1940-2015_40ens_lag3.nc"
dstau = xr.open_dataarray(datpath+tauname)
tlon = dstau['lon'].values
tlat = dstau['lat'].values
tau  = dstau.values #[mon x lat x lon x ens]

# Take ensemble mean and flip longitude coordinates
tauavg = np.nanmean(tau,3)
tauavg = np.transpose(tauavg,(2,1,0)) #lon x lat x mon
tlon1,tauavg = proc.lon360to180(tlon,tauavg) 
tauavg = np.transpose(tauavg,(2,1,0)) # back to mon x lat x lon
tauavg[~np.isfinite(tauavg)] = np.nan

#%% Load in MLD and Damping data...

# Load Damping Data
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])
damping = loaddamp['ensavg'] #[lon x lat x mon]
#damping = np.transpose(damping,(2,1,0)) # [mon x lat x lon]

# Load MLD Data (preprocessed in prep_mld.py)
mld         = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD
#mld = np.transpose(mld,(2,1,0)) # [mon x lat x lon]
#kprevall    = np.load(datpath+"HMXL_kprev.npy") # Entraining Month

#%% Restrict to Region
bbox = [tlon1[0], tlon1[-1], tlat[0], tlat[-1]]
klat = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
klon = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]

mld = mld[klon[:,None],klat[None,:],:]
damping = damping[klon[:,None],klat[None,:],:]

#mld = np.squeeze(mld[:,klat[None,:,None],klon[None,None,:]])
#damping = np.squeeze(damping[:,klat[None,:,None],klon[None,None,:]])


mld = np.transpose(mld,(2,1,0))
damping = np.transpose(damping,(2,1,0))

#%% Do some calculation


# Take annual means
tauann = np.nanmean(tauavg,0)
mldann = np.nanmean(mld,0)
dampann = np.nanmean(damping,0)

# Calculate Tau
# 0 = all, 1 = basin average h, 2 = basin average damping 
calctau0      = (mldann*rho*cp)/(dampann*dt)
calctau1      = (np.nanmean(mldann)*rho*cp)/(dampann*dt)
calctau2      = (mldann*rho*cp)/(np.nanmean(dampann)*dt)
#plt.pcolormesh(tlon1,tlat,tauann)



# %% Plot 3 different taus

# Set variables
cint = np.concatenate((np.arange(0,32,4),np.arange(30,60,10)))
#cmap = cmocean.cm.amp
cmap = cmocean.cm.amp
#cmap.set_bad(color='')

# Initialize Plot
fig,axs = plt.subplots(1,3,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,4))

ax = axs[0]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,calctau0,vmin=0,vmax=50,cmap=cmap)
pcm.set_zorder=0
#ax.contour(tlon1,tlat,calctau0,cint)
ax.set_title("All Vary")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

ax = axs[1]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,calctau1,vmin=0,vmax=50,cmap=cmap)
#ax.contour(tlon1,tlat,calctau1,cint,cmap=cmap)
ax.set_title("use $MLD_{N. Atl_Avg}$")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

ax = axs[2]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,calctau2,vmin=0,vmax=50,cmap=cmap)
#ax.contour(tlon1,tlat,calctau2,cint,cmap=cmap)
ax.set_title("use $Damping_{N. Atl_Avg}$")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

plt.suptitle("Damping Timescale $(1/month)$")
#plt.tight_layout()


plt.savefig(outpath+"Damping_Timescale_Plot.png",dpi=200)


#%% Plot difference in estimated and calculated tau

difftau = calctau0-tauann

cmbal = cmocean.cm.balance
cint = np.arange(0,60,10)

# Initialize Plot
fig,axs = plt.subplots(1,3,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,4))

ax = axs[0]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,calctau0,vmin=0,vmax=50,cmap=cmap)
pcm.set_zorder=0
#ax.contour(tlon1,tlat,calctau0,cint)
ax.set_title(r"Calculated $\tau$")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

ax = axs[1]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,tauann,vmin=0,vmax=50,cmap=cmap)
#ax.contour(tlon1,tlat,tauann,cint,cmap=cmap)
ax.set_title(r"Estimated $\tau$")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

ax = axs[2]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,difftau,vmin=-50,vmax=50,cmap=cmbal)
#ax.contour(tlon1,tlat,calctau2,cint,cmap=cmap)
ax.set_title("Calculated-Estimated")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

plt.suptitle("Damping Timescale $(1/month)$")
#plt.tight_layout()
plt.savefig(outpath+"Damping_Timescale_Comparison_Plot.png",dpi=200)


#%% Plot MLD and TAU


cmbal = cmocean.cm.balance
cmdens = cmocean.cm.dense
lamcint = np.arange(-50,60,10)
mldcint = np.arange(0,600,100)


# Initialize Plot
fig,axs = plt.subplots(1,3,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,4))

ax = axs[0]
ax=viz.init_map(bbox,ax=ax)
pcm = ax.pcolormesh(tlon1,tlat,calctau0,vmin=0,vmax=50,cmap=cmap)
pcm.set_zorder=0
cl = ax.contour(tlon1,tlat,calctau0,np.arange(0,60,10),colors='k',linewidths=0.5)  # Plot vaulues below 50 as black, above 50 as white
ax.clabel(cl,fmt="%i",fontsize=10)
ax.contour(tlon1,tlat,calctau0,np.arange(50,110,10),colors='w',linewidths=0.5)
ax.set_title(r"Calculated $\tau$ (mons)")
plt.colorbar(pcm,ax=ax,orientation='horizontal')

ax = axs[1]
ax=viz.init_map(bbox,ax=ax)
#pcm = ax.pcolormesh(tlon1,tlat,mldann,vmin=0,cmap=cmdens)
ctf = ax.contourf(tlon1,tlat,mldann,mldcint,cmap=cmdens)
cl = ax.contour(tlon1,tlat,mldann,mldcint,colors='k',linewidths=0.5)
ax.clabel(cl,fmt="%i",fontsize=10)
ax.set_title(r"Mixed Layer Depth (m)")
plt.colorbar(ctf,ax=ax,orientation='horizontal')

ax = axs[2]
ax=viz.init_map(bbox,ax=ax)
#pcm = ax.pcolormesh(tlon1,tlat,dampann,vmin=-50,vmax=50,cmap=cmbal)
ctf = ax.contourf(tlon1,tlat,dampann,lamcint,cmap=cmap)
cl = ax.contour(tlon1,tlat,dampann,lamcint,colors='k',linewidths=0.5)
ax.clabel(cl,fmt="%i",fontsize=10)
ax.set_title(r"$\lambda_{a}$")
plt.colorbar(ctf,ax=ax,orientation='horizontal')

plt.suptitle("Ann. Averaged Fields")
#plt.tight_layout()
plt.savefig(outpath+"Timescale_MLD_Lambda_Plot.png",dpi=200)