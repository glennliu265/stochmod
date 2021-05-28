#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Variance Maps

Created on Thu May 27 22:49:53 2021

@author: gliu
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import sys
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath + '/proc/'
   
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")




#%% Experiment parameters

# Options to determine the experiment ID
fscale  = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
applyfac  = 2
mconfig   = "SLAB_PIC"

# Do a stormtrackloop
runid = "303"
funiform = 1.5

# Experiment ID
expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)

# Read in Stochmod SST Data
sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")      

#%% Load Data

# # HadISST Inforation
# detrend = 2
# startyr = 1870

# # ERSST Information
# detrende = 2
# startyre = 1854

# # Load HadISST
# hsst = scm.load_hadisst(datpath+"../",method=detrend,startyr=startyr)

# # Load ERSST
# ersst = scm.load_ersst(datpath+"../",method=detrende,startyr=startyre)

# Load CESM 
fullpt,slabpt = scm.load_cesm_pt(datpath+"../",loadname='both')

# Load Mixed Layer Depths and select point
input_path    = datpath + "../" + "model_input/" 
mld           = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD

# Load lat/lon
lon,lat  = scm.load_latlon(lon360=True)

# Load mask
msk  = np.load(datpath+"../"+"landicemask_enssum.npy")

#%% Some Preprocessing

sstvars = []
for i in range(4):
    sstin = sst[i]
    sstvar = sstin.var(2)
    sstvars.append(sstvar)

# Flip the longitude
fullvar = fullpt.var(0)
slabvar = slabpt.var(0)

# Apply the mask
fullvar *= msk
slabvar *= msk


# Flip the longitude
lon180,fullvar1 = proc.lon360to180(lon,fullvar.T[:,:,None])
_,slabvar1 = proc.lon360to180(lon,slabvar.T[:,:,None])
fullvar1 = np.squeeze(fullvar1)
slabvar1 = np.squeeze(slabvar1)

# Select region
bbox = [lonr[0],lonr[-1],latr[0],latr[-1]]

# Limit to region
fullr,_,_ = proc.sel_region(fullvar1,lon180,lat,bbox)
slabr,_,_ = proc.sel_region(slabvar1,lon180,lat,bbox)

#%% Take the ratios

entr  = sstvars[3]
nentr = sstvars[1]

fullratio = entr/fullr
slabratio = nentr/slabr

proc.sel_region(fullvar1,lon180,lat,bbox)


bboxplot = [-80,0 ,0,65]

cmap = cmocean.cm.balance
cmap.set_bad(alpha=0)

# Ratio Contouring
vlms = [0,2]
cintsneg = np.arange(0,1.2,.2)
cintspos = np.arange(1,2.2,.2)

# Set up plot params
plt.style.use('default')
cmap = cmocean.cm.balance

# Next plot the results
fig,axs = plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,4.5))


ax = axs[0]
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm = ax.pcolormesh(lonr,latr,fullratio.T,vmin=vlms[0],vmax=vlms[-1],cmap=cmap)
#pcm = ax.contourf(lonr,latr,fullratio.T,levels=cints,cmap=cmocean.cm.balance)
clp = ax.contour(lonr,latr,fullratio.T,levels=cintspos,colors="k",linewidths=0.75)
ax.clabel(clp,fmt="%.1f")
cln = ax.contour(lonr,latr,fullratio.T,levels=cintsneg,colors="k",linestyles=":",linewidths=0.75)
ax.clabel(cln,fmt="%.1f")
fig.colorbar(pcm,ax=ax,fraction=0.046)
ax.set_title("Entraining Stochastic Model / CESM1 FULL")

ax = axs[1]
cintsneg = np.arange(0,1.1,.1)
cintspos = np.arange(1,2.1,.1)
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm = ax.pcolormesh(lonr,latr,slabratio.T,vmin=vlms[0],vmax=vlms[-1],cmap=cmap)
clp = ax.contour(lonr,latr,slabratio.T,levels=cintspos,colors="k",linewidths=0.75)
ax.clabel(clp,fmt="%.1f")
cln = ax.contour(lonr,latr,slabratio.T,levels=cintsneg,colors="k",linestyles=":",linewidths=0.75)
ax.clabel(cln,fmt="%.1f")
fig.colorbar(pcm,ax=ax,fraction=0.046)
ax.set_title("Non-Entraining Stochastic Model  / CESM1 SLAB")

plt.suptitle("Ratio of SST Variance",fontsize=14)
plt.savefig("")

