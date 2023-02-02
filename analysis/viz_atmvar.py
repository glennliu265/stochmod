#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize vertical gradients for files generated with crop_lens_atm.py

and the corresponding notebook on stormtrack (viz_atmvar_verticaldiff.ipynb)

Created on Wed Aug 17 18:44:47 2022

@author: gliu
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import glob
import sys
from tqdm import tqdm 


#%%

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/CESM-HTR-RCP85/30y_70to00/atmvar/"
vnames  = ("Q","T","Umod")
nc1     = "%s%sdiff_htr_1970to1999_ensavg.nc" #% (dp,vn)
nc2     = "%s%sdiff_rcp85_2070to2099_ensavg.nc" #% (dp,vn)
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/20220815/"


nvar = len(vnames)
v_sce = []
for sc in range(2):
    v_all = []
    if sc == 0:
        ncs = nc1 # Search nc, historical
    else:
        ncs = nc2 # rcp8.5
    for v in range(nvar):
        nc = ncs % (datpath,vnames[v])
        ds = xr.open_dataset(nc)
        v_all.append(ds)
    v_sce.append(v_all)

#%% Load the correct heat flux and append


vname = "LHFLX_damping"
dp1   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/CESM-HTR-RCP85/30y_70to00/%s/" % vname
nchtr = "htr/CESM1_htr_%s_ensorem1_detrend1_1970to2000_allens.nc" % vname
ncrcp = "rcp85/CESM1_rcp85_%s_ensorem1_detrend1_2070to2100_allens.nc" % vname


dshtr = xr.open_dataset(dp1+nchtr)[vname]
dsrcp = xr.open_dataset(dp1+ncrcp)[vname]

v_sce[0].append(dshtr)
v_sce[1].append(dsrcp)


#%% Now Plot the axes


ylm  = [-65,65]
ytks = np.arange(-60,75,15)
xlms = ([-.0012,.0012],[-2.0,2.0],[-.5,.5],[-2.5,2.5])
lcs  = ("cornflowerblue","red","magenta","k")
fig,axs = plt.subplots(1,nvar+1,sharey=True,figsize=(14,8))

for v in range(nvar+1):
    
    ax = axs[v]
    if v < nvar:
        plotvar = (v_sce[1][v] - v_sce[0][v]).mean('lon')
        ax.plot(plotvar[vnames[v]],ds.lat,color=lcs[v],lw=2)
        ax.set_xlabel(vnames[v])
    else:
        plotvar = (v_sce[1][v] - v_sce[0][v])
        plotvar = plotvar.isel(lag=0).mean('ens').mean('month').mean('lon')
        ax.plot(plotvar,ds.lat,color=lcs[v],lw=2)
        ax.set_xlabel(vname)
    ax.set_xlim(xlms[v])
    ax.axvline([0],ls='dashed',color="k")
    ax.set_ylim(ylm)
    ax.set_yticks(ytks)
    ax.grid(True,ls='dotted')
plt.suptitle("Zonally Averaged Differences in Vertical Gradient (RCP8.5 - Historical)",y=0.94)
plt.savefig("%sZonalMeanDiff.png"% (figpath),bbox_inches='tight')


