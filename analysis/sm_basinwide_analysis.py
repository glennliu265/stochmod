#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare stochastic model output throughout the whole basin

Created on Mon Aug 16 18:23:33 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import time
import time
from tqdm import tqdm
#%% Set Paths

stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210818/"
    
    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    
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

#%% Settings Part 1 (Comparison)


## Compare Effect of including ekman advection
# Names of the experiments
# expids    = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
#              "stoch_output_forcingflxeof_qek_50eofs_SLAB-PIC_1000yr_run003.npz"
#             ]
# exnames   = ["SM (no $Q_{ek}$)",
#              "SM (with $Q_{ek}$)"
#             ]
# excolors  = ["blue",
#               "orange"]
#exoutnameraw = "Q-ek_comparison_50EOF"

## Compare before/after correctio,n and inclusion of Amp
expids = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run005.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz"
          ]
exnames = ["Before (Incl. Lat-weight)",
           "After (no q-corr)",
           "After (with q-corr)",
           "90% Variance Threshold (no-qcorr)"]
excolors = ["blue",
            "orange",
            "magenta",
            "red"]
exoutnameraw = "latweightfix_qamp_comparison_50EOF"

## Examine zffect of including AMP using variance-based threshold
expids = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run007.npz"]
exnames = ["no q-corr",
           "with q-corr (monthly)",
           "with q-corr (avg)"]
excolors = ["blue",
            "orange",
            "magenta"]
exoutnameraw = "90pctvariance_qamp_comparison"
#%% Settings Part 2

# Experiment information
bboxsim    = [-100,20,-20,90] # Simulation Box


# CESM Names
cnames    = ["CESM-FULL","CESM-SLAB",]
ccolors   = ["black","gray"]
cpath     = datpath + "../"

# Combine Names
enames    = np.hstack([cnames,exnames])
ecolors   = np.hstack([ccolors,excolors])


# Load lat/lon
lon180,lat  = scm.load_latlon(rawpath)
lon360,_    = scm.load_latlon(rawpath,lon360=True)
lonr        = np.load(datpath+"lon.npy")
latr        = np.load(datpath+"lat.npy")

# Model to conduct analysis on
model       = 0
modelnames  = ["h constant","h vary","entraining"]
exoutname  = exoutnameraw +  "_model%i" % model
# Autocorrelation parameters
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')


#%%




#%% Load CESM and Stochastic model data

st   = time.time()
ssts = scm.load_cesm_pt(cpath)

for i in tqdm(range(len(expids))):
    ld  = np.load(datpath+expids[i])
    sst = ld['sst']
    ssts.append(sst[model,...])

print("Loaded data in %.2fs"%(time.time()-st))


#%% Crop CESM-data to the same region

# Apply land/ice mask
msk = np.load(lipath)
    
def preprocess_CESM(sst,lon360,lat,msk,bbox):
    """
    Takes sst [time x lat x lon360]
     1. apply land ice mask
     2. tranpose to lon x lat x time
     3. flip to lon180
     4. cut to region
    
    """
    # 1. Apply mask
    sstin = sst.copy()
    sstin *= msk[None,:,:] 
    
    # 2. transpose
    sstin = sstin.transpose(2,1,0)
    
    # 3. flip longitude
    lon180,sstin = proc.lon360to180(lon360,sstin)
    
    # 4. select region
    sstr,lonr,latr = proc.sel_region(sstin,lon180,lat,bbox)
    return sstr

for i in tqdm(range(2)):
    ssts[i] = preprocess_CESM(ssts[i],lon360,lat,msk,bboxsim)


#%% Examine the variance ratio between both locations



sstvar = []
for i in range(len(ssts)):
    sstvar.append(ssts[i].var(2))
    
    
#%% Plot the difference in variance (all three)

bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
vlm = [0,2]
clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)


fig,axs = plt.subplots(1,3,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})


for expid in range(len(exnames)):
    ax = axs.flatten()[expid]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
    if model < 1: # Non-entraining, compare with CESM-SLAB
        comparison = sstvar[1]
        comparename = enames[1]
    else: # entraining, compare with CESM-FULL
        comparison = sstvar[0]
        comparename = enames[0]
    #pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
    pcm = ax.contourf(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clvl,cmap="RdBu_r")
    cl = ax.contour(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clab,colors="k",linewidths=0.75)
    ax.clabel(cl)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("Ratio of SST Variance \n (%s/%s)" % (enames[expid+2],comparename))
plt.savefig("%s%s_VarianceRatio_expnum%i.png" % (figpath,exoutname,expid),bbox_inches='tight',dpi=200)
#%%


expid = 2

bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
#vlm = [0.5,1.5]
vlm = [-1,3]
#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)

if model < 1: # Non-entraining, compare with CESM-SLAB
    comparison = sstvar[1]
    comparename = enames[1]
else: # entraining, compare with CESM-FULL
    comparison = sstvar[0]
    comparename = enames[0]
pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
#pcm = ax.contourf(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clvl,cmap="RdBu_r")
#cl = ax.contour(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax)
ax.set_title("Ratio of SST Variance \n (%s/%s)" % (enames[expid+2],comparename))

plt.savefig("%s%s_expnum%i.png" % (figpath,exoutname,expid),bbox_inches='tight',dpi=200)

#%% Compare this with a plot of the actual theoertical NHFLX underestimate

# lon x lat x mon
lbd_a    = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM-Slab-PIC_lbd_a_Natl_mon.npy")

# Plot the theoretical underestimate
a        = 1-lbd_a.mean(2)[...,None]
underest = 2*a**2 / (1+a)
underest = np.sqrt(underest.squeeze())

bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
vlm = [0.5,1.5]

#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)

if model < 1: # Non-entraining, compare with CESM-SLAB
    comparison = sstvar[1]
    comparename = enames[1]
else: # entraining, compare with CESM-FULL
    comparison = sstvar[0]
    comparename = enames[0]
#pcm = ax.pcolormesh(lonr,latr,underest.T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
pcm = ax.contourf(lonr,latr,underest.T,levels=clvl,cmap="RdBu_r")
cl = ax.contour(lonr,latr,underest.T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax)
ax.set_title("Theoretical NHFLX Underestimate (var(Q)/var(q))")
plt.savefig("%s%s_Theoretical_Underest_AnnAvg.png" % (figpath,exoutname),bbox_inches='tight',dpi=200)





#%% Load mixed layer depths to identify the base month

mld    = np.load(rawpath + "FULL_HTR_HMXL_hclim.npy")
kmonth = np.argmax(mld[klon,klat]) 