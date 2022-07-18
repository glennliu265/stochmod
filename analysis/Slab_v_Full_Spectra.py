#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slab v Full Spectra Analysis

Look at Qnet Ratios/Spectral Spope for Slab vs. Full

Created on Fri Jul 15 16:22:39 2022

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import numpy as np
import xarray as xr
from tqdm import tqdm 
import time

import cartopy.crs as ccrs

#%%

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/Spectra_Comparison/"
fns     = ("NHFLX_spectra_smooth030-taper010_PIC-FULL.nc",
           "NHFLX_spectra_smooth030-taper010_PIC-SLAB.nc")

mconfigs = ("FULL","SLAB")
mcolors  = ('k','gray')
#% PLOTTING PARAMETERS

dtplot = 3600*24*365
xper   = np.array([100,50,25,10,5,2.5])
xtks   = [1/x for x in xper]
xlms   = [xtks[0],xtks[-1]]

bboxplot = [-80,0,0,60]
#%% Functions

def get_specid(findyr,freq,dt=3600*24*365,verbose=True):
    # Convert to years
    per  =  1/(freq*dt)
    
    # Find nearest
    k = np.argmin(np.abs(per-findyr))
    if verbose:
        print("Nearest value to %.2f was %.2f" % (findyr,per[k],))
    return k

#%% Load the data

specs = []
freqs = []
r1s   = []
dofs  = []
for mc in range(2):
    
    ds = xr.open_dataset(datpath+fns[mc])
    
    specs.append(ds.spectra.values)
    freqs.append(ds.frequency.values)
    r1s.append(ds.r1s.values)
    dofs.append(ds.dofs.values)
    
    if mc == 0:
        lat = ds.lat.values
        lon = ds.lon.values
        

#%% TEST PLOT AT SPG

klon,klat = proc.find_latlon(-30,50,lon,lat)

fig,ax = plt.subplots(1,1,figsize=(10,4))

for mc in range(2):
    ax.plot(freqs[mc]*dtplot,specs[mc][0,klat,klon,:]/dtplot,color=mcolors[mc])
ax.legend()
ax.set_xticks(xtks)
ax.set_xlim(xlms)
ax.set_title("Volume")
ax.grid(True,ls="dotted")


#%% Plot power at a specific frequency

sel_per = 100
_,nlat,nlon,_=specs[mc].shape

sel_power = np.zeros((2,nlat,nlon)) # [model x lat x lon]


for mc in range(2):
    
    kper                = get_specid(sel_per,freqs[mc],dt=dtplot)
    sel_power[mc,:,:] = specs[mc][0,:,:,kper]


#%% Visualize it

use_pcm = True

fig,axs = plt.subplots(1,2,figsize=(10,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})


clvl = np.arange(0,1005,5)

for mc in range(2):
    
    ax = axs[mc]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="k")
    if use_pcm:
        pcm = ax.pcolormesh(lon,lat,sel_power[mc,:,:].T/dtplot,vmin=clvl[0],vmax=clvl[-1],)
    else:
        pcm = ax.contourf(lon,lat,sel_power[mc,:,:].T/dtplot,levels=clvl,extend='both')
    ax.set_title(mconfigs[mc])


fig.colorbar(pcm,ax=axs.flatten(),fraction=0.05)
plt.suptitle("$Q_{net}$ Spectra Power @ %i yrs" % (sel_per))


#%% Calculate Slope between two points


lowper = 100
hiper  = 5







# Endpoint Approach (can also try to fit a line)
boundf = np.zeros((2,2,nlat,nlon)) # [model x lo/hi x lat x lon]
boundP = boundf.copy()

for a in tqdm(range(nlat)):
    for o in range(nlon):
        for mc in range(2):
            
            # Get indices of high and low
            klo = get_specid(lowper,freqs[mc],dt=dtplot,verbose=False)
            khi = get_specid(hiper,freqs[mc],dt=dtplot,verbose=False)
            
            #inspec = specs[mc][0,a,o,:]
            #loval = inspec[klo]
            #hival = inspec[khi]
            
            # Save variable
            boundf[mc,0,a,o] = freqs[mc][klo]
            boundf[mc,1,a,o] = freqs[mc][khi]
            
            # Save power
            boundP[mc,0,a,o] = specs[mc][0,a,o,klo]
            boundP[mc,1,a,o] = specs[mc][0,a,o,khi]


boundf *= dtplot
boundP /= dtplot

df     = boundf[:,1,:,:] - boundf[:,0,:,:] # Hifreq - Lowfreq
dP     = boundP[:,1,:,:] - boundP[:,0,:,:]
Pslope = dP/df
    
    

