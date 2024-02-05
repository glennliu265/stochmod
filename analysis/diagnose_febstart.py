#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Diagnose Febstart

Check February start in CESM1 data and how that might be impacting calculations

Copied section from test_forcing_damping_shift
Created on Tue Jan 30 19:02:18 2024

@author: gliu
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx

#%%

lonf=330
latf=50

ncts1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/TS_anom_PIC_FULL.nc"
dsts1 = xr.open_dataset(ncts1)
dspt1 = dsts1.sel(lon=lonf,lat=latf,method='nearest')


## Compute a seasonal cycle using xarray
scyclex = dspt1.groupby('time.month').mean('time').TS.values

# Repeat but first fix the file
dsfix = proc.fix_febstart(dspt1)
scyclexfix = dsfix.groupby('time.month').mean('time').TS.values

## Compute with my function
sst_full       = dspt1.TS.values
scyclem        = proc.calc_clim(sst_full,0)
tsmetrics_full = scm.compute_sm_metrics([sst_full,])


## Compute again but with fixed
sst_full2       = dsfix.TS.values
scyclem2        = proc.calc_clim(sst_full2,0)
tsmetrics_full2 = scm.compute_sm_metrics([sst_full2,])



#%% Compare
mons3 = proc.get_monstr(nletters=3)

fig,ax=viz.init_monplot(1,1)
ax.plot(mons3,scyclex,label="Xarray Scycle")
ax.plot(mons3,scyclem,label="calc_clim scycle (numpy)")
ax.plot(mons3,scyclexfix,label="Xarray with startdate fixed",ls='dotted')
ax.legend()
plt.show()

#%% Compare Monthly Variances

fig,ax=viz.init_monplot(1,1)

ax.plot(mons3,tsmetrics_full['monvars'][0],label="calc_clim scycle (numpy)")
ax.plot(mons3,tsmetrics_full2['monvars'][0],label="Xarray with startdate fixed",ls='dotted')

ax.legend()
plt.show()

