#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check if variables were deseasoned properly.

Created on Wed Jan 31 18:15:58 2024

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

#%% Load other stuff


mons3 = proc.get_monstr(nletters=3)


#%% First Comparison: TS

# TS from reemergence script and stochmod scripts should be the same. Let me check at the SPG Point.
# The files were produced SEPT 24 2021 around when [calc_CESM_anom.py was last updated]
# Section below copied from test_forcing_damping_shift ------------------------

lonf = -30
latf = 50
# % Load SLAB SST at the point


ncts = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/TS_anom_PIC_SLAB.nc"
dsts = xr.open_dataset(ncts)
dspt = dsts.sel(lon=lonf+360,lat=latf,method='nearest')
sst_slab = dspt.TS.values
tsmetrics_slab = scm.compute_sm_metrics([sst_slab,])

#% Load FULL SST at a point

ncts1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/TS_anom_PIC_FULL.nc"
dsts1 = xr.open_dataset(ncts1)
dspt1 = dsts1.sel(lon=lonf+360,lat=latf,method='nearest')

sst_full       = dspt1.TS.values
tsmetrics_full = scm.compute_sm_metrics([sst_full,])

#%% Get Full TS? Another file processed August 04 2021

ncts2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/TS_PIC_FULL.nc"
ds2= xr.open_dataset(ncts2) # [yr  x mon x lat x lon]
ds2 = ds2.sel(lon=lonf+360,lat=latf,method='nearest').TS.values.flatten()

# Deseason 
ds2a = proc.deseason(ds2,).flatten()


tsmetrics2 = scm.compute_sm_metrics([ds2a,],)

# -----------------------------------------------------------------------------
#%% Next, load data from re-emergence

fpr     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"
fnr     = fpr + "CESM1_htr_SST.nc" # Note this starts in Feb!
dsr     = xr.open_dataset(fnr) # ens x time
sst_htr = dsr.SST.values

# Remove ens mean
ssta_htr = sst_htr - sst_htr.mean(0)[None,:]
nens,ntime=ssta_htr.shape
nyrs = int(ntime/12)

# Deseason
ssta_htr = proc.deseason(ssta_htr,dim=1).reshape(nens,nyrs*12)


# Prepare for processing
ssta_htr    = [ssta_htr[e,:] for e in range(nens)]
tsmetricshtr = scm.compute_sm_metrics(ssta_htr,)


monvar_htr = np.array(tsmetricshtr['monvars']) # [Ens x 12]



#%% Plot monthly variance

fig,ax=viz.init_monplot(1,1,)
ax.plot(mons3,tsmetrics_slab['monvars'][0],label="SLAB (TS_Anom)",color="gray")
ax.plot(mons3,tsmetrics_full['monvars'][0],label="FULL (TS_Anom)",color='k')

ax.plot(mons3,tsmetrics2['monvars'][0],label="FULL (TS_PIC)",color='cyan',ls='dashed')


for e in range(nens):
    ax.plot(mons3,monvar_htr[e,:],label="",color='red',alpha=0.1)
ax.plot(mons3,monvar_htr.mean(0),label="FULL HTR (rem)",color='red',alpha=1)

ax.legend()
plt.show()


#%% Now Check the input parameters. Look first at the HFF

#%% Ok, now try loading the damping
"""

"New" descriptes the settings in sm_rewrite_loop where:
    mode    = 5
    ensostr = ""
    lag     = lag1
    
"""


input_path = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/'
# Load Lat/Lon for plotting
lon,lat = scm.load_latlon()
lonf,latf=-30,50
klon,klat = proc.find_latlon(lonf,latf,lon,lat)

# Assuming new is default lagstr1, ensolag is removed
# Method 5 (not sure what this is again?)

dampfn = [
    
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy",
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode5_lag1.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode4.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode5_lag1.npy",
    
    ]

dampnames = [
    
    "SLAB PIC (old)",
    "SLAB PIC (new)",
    "FULL PIC (old)",
    "FULL PIC (new)",
    
    ]

# Load and print shape
dampload = [np.load(input_path+fn) for fn in dampfn]
[print(fl.shape) for fl in dampload]

# Restrict to Point
dpt   = [fl[klon,klat,:] for fl in dampload]


#%% Load Damping from the reemergence analysis (copied from stochmod_point.py)
datpathpt   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/ptdata/lon330_lat50/"
lbda_re     = np.load(datpathpt+"CESM1_htr_lbda_reestimate.npz",allow_pickle=True)['qnet'] # [ Ens x Mon x Lag]


#%% Plot Damping to Compare


fig,ax = viz.init_monplot(1,1)

for ff in range(len(dpt)):
    plotvar = dpt[ff]
    ax.plot(mons3,plotvar,label=dampnames[ff],marker="d")
    
    
#Plot Lags fo reach of the new ones
for l in range(3):
    
    ax.plot(lbda_re[:,:,l].mean(0)*-1,label="lag %i" % (l+1))
    

#ax.plot(mons3,np.sqrt(fprimestd[klon,klat,:]),color='gray',ls='dashed',marker="x",label="std(Fprime) (SLAB)")
ax.legend()
plt.show()


