#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare impact of adding Td Damping (or any experiment) with the
default used for the Stochastic model paper

Works with experiments specified in stochmod_params
Compare sections of plot_temporal_region.py

Created on Thu Nov 16 17:51:04 2023

@author: gliu
"""



import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from tqdm import tqdm 
import time
import cartopy.crs as ccrs
import os
import sys

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20231117/"
   
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


#%% Import Parameters for Stochmod


cwd = os.getcwd()
sys.path.append(cwd+"/../")
import stochmod_params as sparams


#%% Indicate Other User Selections

continuous   = True


# Set some data paths
datpath_pointwise = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/"
runid             = 9 # Currently only supports one, upgrade later to load all..

#datpath      = ""

mons3 = proc.get_monstr(3)



#%% Indicate Experiments to Compare

runnames = ["default","Tddamp"]
runlists = [sparams.rundicts[name] for name in runnames]
nruns    = [len(r) for r in runlists]


#%% Specify Loading Functions

def unpack_smdict(indict):
    """
    Takes a dict of [run][region][models][OTHERDIMS] and unpacks it into
    an array [unpackaged]

    """
    # Get "Outer Shell" dimensions
    nrun    = len(indict)
    nregion = len(indict[0])
    nmodels = len(indict[0][0])
    
    # For Autocorrelation
    otherdims = indict[0][0][0].shape
    print("Found... Runs (%i) Regions (%i) ; Models (%i) ; Otherdims (%s)" % (nrun,nregion,nmodels,str(otherdims)))
    
    # Preallocate
    newshape = np.concatenate([[nrun,nregion,nmodels],otherdims])
    unpacked = np.zeros(newshape) * np.nan
    
    # Loop thru dict
    for run in range(nrun):
        for reg in range(nregion):
            for mod in range(nmodels):
                unpacked[run,reg,mod,:] = indict[run][reg][mod]
    return unpacked

def repack_smdict(inarr,nregion,nmodels):
    """
    Repackages a numpy array of inarr[region x model x otherdims] to 
    outdict{region}{model}
    """
    outdict = {}
    for reg in range(nregion):
        indict = {}
        for mod in range(nmodels):
            indict[mod] = inarr[reg,mod,:]
        outdict[reg] = indict.copy()
    return outdict

#%% Load Autocorrelation for Stochastic Model ( Copied and modified from plot_temporal_region.py)
nexps = len(runnames)

ac_exp      = [] # [exp][run][region][model][lag]
kmonths_exp = []

for exp in range(nexps):
    
    
    fnames = runlists[exp]
    print("Loading data for [%s]" % (runnames[exp]))
    
    
    # Load for stochastic model experiments
    # -------------------------------------
    if continuous: # Load for each run
        
        sstac   = [] # [run][region][model][lag]
        kmonths = [] 
        for f,fname in enumerate(fnames):
            rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,fname)
            ld = np.load(rsst_fn,allow_pickle=True)#.item()
            sstac.append(ld['autocorr_region'].item()) # I think its [region][model][lag]
            kmonths.append(ld['kmonths'].item())
        kmonths = kmonths[0] # Just take the first
        
        
        # Extract the region and take the average
        nrun    = len(sstac)
        nregion = len(sstac[0])
        nmodels = len(sstac[0][0])
        nlags   = len(sstac[0][0][0])
        sstac_rearr = unpack_smdict(sstac)
        sstac_avg   = sstac_rearr.mean(0) # [region x model x lag]

        # Repack as dict
        sstac       = repack_smdict(sstac_avg,nregion,nmodels)
        
    else:
        rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,fname)
        ld = np.load(rsst_fn,allow_pickle=True)#.item()
        sstac   = ld['autocorr_region'].item() # I think its [region][model][lag]
        kmonths  = ld['kmonths'].item()
        
    ac_exp.append(sstac)
    kmonths_exp.append(kmonths)

#%% Load Autocorrelation for CESM

# Load data for CESM1-PIC
# -----------------------
cesmacs    = []
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_Region_Autocorrelation_%s_ensorem0.npz" % (datpath,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)
cesmacs    = ldc['autocorr_region'].item() # [Region] x [Model]

#%% Plot Some Autocorrelation Differences (Regional)


# -------------------------------------
#%% Pointwise autocorrelation - Load (Stochastic Model)
# ------------------------------------

debug = False

# Load file from path
acfs_pointwise_sm = [] # [lon x lat x month x x lag]
t2_sm = []
for exp in range(nexps):
    
    expname  = runnames[exp]
    print("Loading ACFs Pointwise for %s" % (expname))
    loadname = "%sSM_%s_SST_autocorrelation_thres0_lag00to60_runid2%02d.npz" % (datpath_pointwise,expname,runid) 
    ld       = np.load(loadname,allow_pickle=True)
    if debug:
        print(ld.files)
    if exp == 0:
        lon  = ld['lon']
        lat  = ld['lat']
        lags = ld['lags']
    acfs     = ld['acs'][:,:,2,:,2,:] # [lon x lat x hconfig x month x threshold x lag], Just grab the  "ALL" ACF and Entraining Config
    acfs_pointwise_sm.append(acfs)
    t2_sm.append(proc.calc_T2(acfs,axis=3))


# Do the same for CESM
fn_pic_acf = datpath_pointwise + "PIC-FULL_SST_autocorrelation_thres0_lag00to60.npz"
ld         = np.load(fn_pic_acf,allow_pickle=True)
acf_cesm   = ld['acs'][:,:,:,2,:] # [lon x lat x month x 61]
t2_cesm    = proc.calc_T2(acf_cesm,axis=3) # [lon x lat x month]

# Calculate a quick landmask
landmask = t2_cesm[:,:,0].copy()
landmask[landmask == 1.] = np.nan
landmask[~np.isnan(landmask)] = 1
plt.pcolormesh(landmask)

#%% Debuggin part (CEMS)
# NVM, just didnt update variable. delete this. ## Note, not sure why CESM has hconfigs... confused...
# klon,klat=proc.find_latlon(-30,50,lon,lat)
# test = acf_cesm[klon,klat,:,1,:] # [Mystery x Lag]
# fig,ax = plt.subplots(1,1)
# for ii in range(3):
#     ax.plot(test[ii,:],label=ii)
# ax.legend()
# acf_cesm = acf_cesm[:,:,2,:,:]


#%% Check ACF for different simulations at apoint
lonf   =-30
latf   = 50
kmonth = 1

bboxinset = [-80,0,0,65]
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(3,3),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bboxinset,fill_color="k")
ax.plot(lonf,latf,marker="x",markersize=24)

klon,klat=proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle=proc.make_locstring(lonf,latf)

fig,ax = plt.subplots(1,1)
for exp in range(nexps):
    ax.plot(lags,acfs_pointwise_sm[exp][klon,klat,kmonth],label=runnames[exp])
ax.plot(lags,acf_cesm[klon,klat,kmonth,:],color="k")
ax.legend()

#%% Compare T2


bboxplot = [-80,0,0,64]

plot_diffs = True
kmonth     = 1
fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(18,6),subplot_kw={'projection':ccrs.PlateCarree()})
vlms    = [0,36]

clvls   = [6,12,18,24,36]

fsz_title = 20
fsz_label = 16
fsz_ticks = 16

cmap    = 'cmo.matter_r'

for a in range(3):
    ax = axs[a]
    
    
    blabels = [0,0,0,1]
    if a == 0:
        plotvar = t2_cesm[:,:,kmonth]
        title="CESM1-PIC"
        blabels[0] = 1
    elif a == 1:
        plotvar = t2_sm[0][:,:,kmonth]
        title="Entraining Stochastic Model"
    elif a == 2:
        plotvar = t2_sm[1][:,:,kmonth]
        title="Add $T_d'$ Damping"
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabels,fontsize=fsz_ticks)
    ax.set_title(title,fontsize=fsz_title)
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,(plotvar*landmask).T,cmap=cmap)
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')
    else:
        pcm = ax.pcolormesh(lon,lat,(plotvar*landmask).T,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    cl = ax.contour(lon,lat,(plotvar*landmask).T,levels=clvls,colors="k",linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_ticks)
if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.012,pad=0.01)
    cb.set_label("%s T2 (Months)" % (mons3[kmonth]),fontsize=fsz_label)
    
    
figname = "%sT2_ByModel_Tddamp_Comparison.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')
    
#%% Plot Change in Td


fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(18,6),subplot_kw={'projection':ccrs.PlateCarree()})
vlms    = [-25,25]

clvls   =np.arange(-36,42,6)

fsz_title = 20
fsz_label = 16
fsz_ticks = 16

cmap    = 'cmo.balance'

for a in range(3):
    ax = axs[a]
    
    blabels = [0,0,0,1]
    if a == 0:
        plotvar = t2_cesm[:,:,kmonth] - t2_sm[1][:,:,kmonth]
        title="CESM1-PIC - SM ($T_d'$ Damping)"
        blabels[0] = 1

    elif a == 1:
        plotvar = t2_sm[0][:,:,kmonth] - t2_sm[1][:,:,kmonth]
        title="SM - SM (with $T_d'$ damping)"

    elif a == 2:
        plotvar = t2_cesm[:,:,kmonth] - t2_sm[0][:,:,kmonth]
        title="CESM1-PIC - SM"

        
    ax = viz.add_coast_grid(ax,bboxplot,fill_color="k",blabels=blabels,fontsize=fsz_ticks)
    ax.set_title(title,fontsize=fsz_title)
    if vlms is None:
        pcm = ax.pcolormesh(lon,lat,(plotvar*landmask).T,cmap=cmap)
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')
    else:
        pcm = ax.pcolormesh(lon,lat,(plotvar*landmask).T,vmin=vlms[0],vmax=vlms[1],cmap=cmap)
    cl = ax.contour(lon,lat,(plotvar*landmask).T,levels=clvls,colors="k",linewidths=0.75)
    ax.clabel(cl,fontsize=fsz_ticks)
if vlms is not None:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.012,pad=0.01)
    cb.set_label("Difference in %s T2 (Months)" % (mons3[kmonth]),fontsize=fsz_label)

figname = "%sT2_ByModel_Tddamp_Differences.png" % (figpath)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%%

# # Calculate Confidence internals -----------------------------------------

# # Stochastic Model
# cfstoch = np.zeros([len(regions),3,len(lags),2]) # [Region x Model x Lag x Upper/Lower]
# n       = 1000
# for rid in range(len(regions)): # Loop by Region
#     for mid in range(3): # Loop by Model
#         inac                   = sstac[rid][mid]
#         cfs                    = proc.calc_conflag(inac,conf,tails,n)
#         cfstoch[rid,mid,:,:] = cfs.copy()
        
# # CESM1
# cfcesm = np.zeros((len(regions),2,len(lags),2)) # [Region x Model x Lag x Upper/Lower]
# ns     = [1798,898]
# for rid in range(len(regions)):
#     for mid in range(2):
#         inac                = cesmacs[rid][mid]
#         cfs                 = proc.calc_conflag(inac,conf,tails,ns[mid])
#         cfcesm[rid,mid,:,:] = cfs.copy()

    




