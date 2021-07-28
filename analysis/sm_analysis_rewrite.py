#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Re-written Analysis Script for Stochastic Model Ouput

Created on Sat Jul 24 20:22:54 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean


#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210726/"
   
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

#%% User Edits

# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over
runid     = "002"


# Analysis (7/26/2021, comparing 80% variance threshold and 5 or 3 EOFs)
#fnames      = ["flxeof_080pct_SLAB-PIC","flxeof_5eofs_SLAB-PIC","flxeof_3eofs_SLAB-PIC"]
#frcnamelong = ("80% Variance Threshold","5 EOFs","3 EOFs")


# Analysis: Trying different number of EOFs
neofs       = [1,2,3,5,10]
fnames      = ["flxeof_%ieofs_SLAB-PIC" % x for x in neofs]
frcnamelong = ["%i EOFs" % x for x in neofs]

#%% Post Process Outputs (Calculate AMV, Autocorrelation)
for frcname in fnames:
    expid = "forcing%s_%iyr_run%s" % (frcname,nyrs,runid) 
    scm.postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags)
    print("Completed Post-processing for Experiment: %s" % expid)
    
#%% Visualize AMV

# Regional Analysis Settings
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")        # Region Names
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
cint   = np.arange(-0.3,0.33,0.03) # Used this for 7/26/2021 Meeting
cl_int = np.arange(-0.3,0.4,0.1)
bboxplot = [-100,20,0,80]
modelnames  = ("Constant h","Vary h","Entraining")

# Experiment names


# -- Select Experiment -- 
fid   = 4
frcname = fnames[fid]
expid = "forcing%s_%iyr_run%s" % (frcname,nyrs,runid) 
regid = 0

# Load lat/lon regional
lon = np.load(datpath+"lon.npy")
lat = np.load(datpath+"lat.npy")

# Load post-propcssed output
# AMV_Region_forcingflxeof_5eofs_SLAB-PIC_1000yr_run001.npz
ldpath = datpath + "proc/AMV_Region_%s.npz" % expid
ld = np.load(ldpath,allow_pickle=True)

# Load. Things are organized by region, then by model
amvidx = ld['amvidx_region'].item()[regid]
amvpat = ld['amvpat_region'].item()[regid]


fig,axs = plt.subplots(1,3,figsize=(10,5),subplot_kw={'projection':ccrs.PlateCarree()})
for p in range(len(amvpat)):
    ax = axs.flatten()[p]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
    #cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
    #ax.clabel(cl,levels=cl_int)
    #pcm = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance)
    #ax.set_title(modelnames[p])
    #fig.colorbar(pcm,ax=ax,fraction=0.036)
    ax.set_title(modelnames[p])
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='horizontal',shrink=0.60)#,pad=0.015)
plt.suptitle("%s AMV Pattern ($\circ C$ per $\sigma_{AMV}$, Forcing: %s)"%(regions[regid],frcnamelong[fid]),y=0.80,fontsize=14)
#plt.tight_layout()
plt.savefig("%sAMV_Pattern_%s_region%s.png"%(figpath,expid,regions[regid]),dpi=200,bbox_tight='inches')




#%% Individual AMV Plots (rather than panel based)


for p in range(len(amvpat)):
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
    cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fmt="%.2f",fontsize=8)
    
    #pcm = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance)
    #ax.set_title(modelnames[p])
    #fig.colorbar(pcm,ax=ax,fraction=0.036)
    
    ax.set_title(modelnames[p] + " ($\circ C$ per $\sigma_{AMV}$) \n Forcing: %s" % frcnamelong[fid])
    fig.colorbar(pcm,ax=ax,orientation='horizontal',shrink=0.75)#,pad=0.015)
    plt.savefig("%sAMV_Pattern_%s_region%s_model%s.png"%(figpath,expid,regions[regid],modelnames[p]),dpi=200,bbox_tight='inches')


#%% Load in regional SSTs, and perform spectral analysis

# Stored in nested dict...
# Outer dict: here keys are 0-3, for each region
# Inner dict: keys are 0-2, for each model

# Load in SSTs for each region
sstdicts = []
for f in range(len(neofs)):
    # Load the dictionary
    expid = "forcing%s_%iyr_run%s" % (fnames[f],nyrs,runid) # Get experiment name
    rsst_fn = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,expid)
    sst = np.load(rsst_fn,allow_pickle=True).item()
    sstdicts.append(sst)


# Identify the variance for each region and load in numpy arrays
sstall  = np.zeros((len(fnames),len(regions),len(modelnames),nyrs*12)) # Forcing x Region x Model x Time
sstvars = np.zeros((len(fnames),len(regions),len(modelnames))) # Forcing x Region x Model
# An unfortunate nested loop... 
for fid in range(len(fnames)):
    for rid in range(len(regions)):
        for model in range(len(modelnames)):
            sstin  = sstdicts[fid][rid][model]
            sstvar = np.var(sstin)
            print("Variance for forcing %s, region %s, model %s is %f" % (fnames[fid],regions[rid],modelnames[model],sstvar))
            
            sstall[fid,rid,model,:] = sstin.copy()
            sstvars[fid,rid,model]   = sstvar

# Make a plot of the variance... compare for each forcing scenario
elabels = ["%i EOFs"% x for x in neofs]
for rid in range(len(regions)):
    fig,ax  = plt.subplots(1,1)
    for model in range(len(modelnames)):
        ax.plot(np.arange(0,len(neofs),1),sstvars[:,rid,model],marker="o",label=modelnames[model])
    ax.legend()
    ax.set_xticks(np.arange(0,len(neofs),1))
    ax.set_xticklabels(elabels)
    ax.set_ylabel("$(\circ C)^{2}$")
    ax.set_xlabel("Number of EOF Patterns")
    ax.set_title("%s AMV Index Variance vs. Number of EOF Patterns \n used to force the stochastic model (%s)" % (regions[rid],modelnames[model]))
    ax.grid(True,ls='dotted')
    plt.savefig("%svariance_vs_nEOF_by_model_runid%s_nyr%i_region%s.png"%(figpath,runid,nyrs,regions[rid]),dpi=150)
