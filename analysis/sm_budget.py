#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:49:29 2022

@author: gliu
"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm

from time import time
#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220210/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)

#%% Load the Raw Stochastic Model Output

expname  = "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_Qek.npz"
npz      = np.load(datpath+expname)
print(npz.files)

lon = npz['lon']
lat = npz['lat']

#%% Load each component

"""

Eqn.  ::  T = -lambda*T + [ F' + Qek + we/h*Td' ] * FAC

Term  ::  (0)   (1)        (2)   (3)     (4)        (5)

"""

termkeys = ("sst","damping_term","forcing_term","ekman_term","entrain_term")
termkeys_title = ("T", "$\lambda T$","$F'$","$F_{ek}$","$w_{e} h^{-1}  T_{d}'$")

# Preallocate and load
# --------------------
nlon,nlat,ntime = npz[termkeys[0]].shape # [lon x lat x time x term]
tcomp           = np.zeros((nlon,nlat,ntime,len(termkeys))) * np.nan
for k,key in tqdm(enumerate(termkeys)):
    tcomp[...,k] = npz[key]

# Multiply Forcing Terms
# ----------------------
FAC    = npz['FAC']
FACrep = np.tile(FAC,int(ntime/12))

# Check if it is equivalent (Note it doesn't seem to be)
# ------------------------------------------------------
#            damping       +               (Fek+F) * FAC            + entrain_term
sum_terms = tcomp[:,:,:,1] + np.sum(tcomp[:,:,:,[2,3]],3) * FACrep + tcomp[:,:,:,4]
chk       = np.nanmax(np.abs( sum_terms-tcomp[...,0]).flatten())

maxdiff   = sum_terms-tcomp[...,0]
plt.pcolormesh(np.max(np.abs( sum_terms-tcomp[...,0]),-1).T),plt.colorbar()
plt.pcolormesh((sum_terms/tcomp[...,0])[...,0].T),plt.colorbar()
#%% Check the variance of each term
bboxplot = [-80,0,10,65]


vmaxes = [2.5,1.5,1,0.3,0.08]
set_vmax = True

fig,axs  = plt.subplots(1,len(termkeys),subplot_kw={'projection':ccrs.PlateCarree()},
                       figsize=(16,6),constrained_layout=True)

for k in range(len(termkeys)):
    ax = axs[k]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray')
    if set_vmax: # Optionally set maximum Values
        pcm = ax.pcolormesh(lon,lat,np.var(tcomp[...,k],-1).T,
                            cmap='cmo.ice',vmin=0,vmax=vmaxes[k])
    else:
        pcm = ax.pcolormesh(lon,lat,np.var(tcomp[...,k],-1).T,cmap='cmo.ice')
    fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction= 0.045)
    ax.set_title(termkeys_title[k])
    
    
    
    


