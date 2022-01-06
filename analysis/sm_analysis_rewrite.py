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
from tqdm import tqdm
#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20211018/"
   
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
#%% User Edits

# Visualization options
viz_AMV=False

# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over
runid     = "006"
savesep   = False # Set to True if output was saved separately
useslab   = False # Set to True if you only used slab output for all simulations

# Additional Analysis Options
mask_damping = True # Set to True to mask out damping points that failed the T-Test

# Analysis (7/26/2021, comparing 80% variance threshold and 5 or 3 EOFs)
#fnames      = ["flxeof_080pct_SLAB-PIC","flxeof_5eofs_SLAB-PIC","flxeof_3eofs_SLAB-PIC"]
#frcnamelong = ("80% Variance Threshold","5 EOFs","3 EOFs")

# Analysis: Trying different number of EOFs
#neofs       = [1,2,50]#[1,2,3,5,10,25,50]
neofs = [90]
#fnames      = ["flxeof_qek_%ieofs_SLAB-PIC" % x for x in neofs]
#fnames      = ["flxeof_qek_%ieofs_SLAB-PIC_JJA" % x for x in neofs]
#fnames = ["flxeof_090pct_SLAB-PIC_eofcorr1"]
fnames = ["forcingflxeof_qek_50eofs_SLAB-PIC_1000yr_run005",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run007",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run004",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006"
          ]


# ## Compare types of forcing and effect of applying ampq
fnames = ["forcingflxstd_SLAB-PIC_1000yr_run006_ampq0",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
          "forcingflxstd_SLAB-PIC_1000yr_run006_ampq1",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1"]
#fnames      = ["flxeof_%ieofs_SLAB-PIC" % x for x in neofs]
#frcnamelong = ["%i EOFs" % x for x in neofs]
frcnamelong = ["50 EOFs (with $Q_{ek}$ and q-corr)",
               "90% Variance (ann avg. q-corr)",
               "90% Variance (monthly q-corr)",
               "90% Variance (no q-corr)"]

frcnamelong = ["var(Q)-based",
               "90% Variance",
               "var(Q)-based (q-corr)",
               "90% Variance (q-corr)"]
    #"90% Threshold (no-qcorr)"]


fnames = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
          "forcingflxeof_q-ek_090pct_SLAB-PIC_eofcorr1_1000yr_run009_ampq0",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
          "forcingflxeof_q-ek_090pct_SLAB-PIC_eofcorr1_1000yr_run009_ampq1"]


frcnamelong = ["90% Variance",
               "90% Variance (q-ek)",
               "90% Variance (q-corr)",
               "90% Variance (q-ek and q-corr)"]

## Same as above, but now correcting locally for eof variance
fnames  = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq0",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1"
          ]

frcnamelong = ["Basinwide Correction",
           "Local Correction",
           "Basinwide Correction (with q-corr)",
           "Local Correction (with q-corr)"
            ]

## New Variance Correction Method (derivation)
fnames  = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run008_ampq3",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run008_ampq3",
          ]

frcnamelong = ["Basinwide Correction (Old)",
           "Local Correction (Old)",
           "Basinwide Correction (New)",
           "Local Correction (New)"
            ]
exoutnameraw = "new_v_old_q-correction"

## Seasonal Variance
fnames = ('forcingflxeof_090pct_SLAB-PIC_eofcorr2_DJF_1000yr_run009_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run009_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_JJA_1000yr_run009_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_SON_1000yr_run009_ampq3')

## By Number of EOFs (local correction)
fnames = (
            "forcingflxeof_50eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_25eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_10eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_5eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_3eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_2eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3",
            "forcingflxeof_1eofs_SLAB-PIC_eofcorr2_1000yr_runtest009_ampq3"
            )
fnames = ("forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run009_ampq3",)

## NAO and EAP
fnames = ("forcingflxeof_EOF1_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
          "forcingflxeof_EOF2_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3")


## By Number of EOFs (basinwide correction)
fnames = (
            "forcingflxeof_50eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_25eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_10eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_5eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_3eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_2eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3",
            "forcingflxeof_1eofs_SLAB-PIC_eofcorr1_1000yr_runtest009_ampq3"
            )

## By Number of EOFs (no correction)
fnames = (
            "forcingflxeof_50eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_25eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_10eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_5eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_3eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_2eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
            "forcingflxeof_1eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3"
            )

## NAO and EAP
fnames = ("forcingflxeof_EOF1_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3",
          "forcingflxeof_EOF2_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3")

## Rewritten (with slab/full forcing/damping) (run010) ---




# Seasonal Analysis
#stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run010_ampq3.npz
fnames   = ('forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_DJF_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_JJA_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_SON_1000yr_run011_ampq3')

## NAO and EAP
fnames = ("forcingflxeof_EOF1_SLAB-PIC_eofcorr0_1000yr_run011_ampq3",
          "forcingflxeof_EOF2_SLAB-PIC_eofcorr0_1000yr_run011_ampq3")

# Test witha single output
fnames = ("forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method4_dmp0",)

# Seasonal Analysis
#stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run010_ampq3.npz
fnames   = ('forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_DJF_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_JJA_1000yr_run011_ampq3',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_SON_1000yr_run011_ampq3')

## NAO and EAP
fnames = ("forcingflxeof_EOF1_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
          "forcingflxeof_EOF2_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0")


## By Number of EOFs (no correction)
fnames = (
            "forcingflxeof_50eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_25eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_10eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_5eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_3eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_2eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0",
            "forcingflxeof_1eofs_SLAB-PIC_eofcorr0_1000yr_run011_ampq3_method4_dmp0"
            )

# Seasonal Analysis
#stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run010_ampq3.npz
fnames   = ('forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method4_dmp0',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_DJF_1000yr_run011_ampq3_method4_dmp0',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_MAM_1000yr_run011_ampq3_method4_dmp0',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_JJA_1000yr_run011_ampq3_method4_dmp0',
            'forcingflxeof_090pct_SLAB-PIC_eofcorr2_SON_1000yr_run011_ampq3_method4_dmp0')

# 90% Variance forcing with Ekman Forcing, needs n_models=1 as an additional argument
fnames   = ('forcingflxeof_090pct_SLAB-PIC_eofcorr2_Qek',)

print("Now processing the following files: \n ")
print(*fnames, sep='\n')


n_models=1 # SET TO 1 for EKMAN FORCING< None for all others!!!!
#%% Post Process Outputs (Calculate AMV, Autocorrelation)
for frcname in tqdm(fnames):
    expid = frcname
    
    if "Qek" in expid:
        print("Processing just entrain model for Qek")
        n_models = 1
    else:
        n_models = None
        
        
    scm.postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags,mask_pacific=True,
                                savesep=savesep,useslab=useslab,mask_damping=mask_damping,n_models=n_models)
    print("Completed Post-processing for Experiment: %s" % expid)
#%% Visualize AMV
if viz_AMV:
    # Regional Analysis Settings
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]
    bbox_NNA = [-80,0 ,10,65]
    regions = ("SPG","STG","TRO","NAT","NNAT")        # Region Names
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NNA) # Bounding Boxes
    cint   = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
    cl_int = np.arange(-0.45,0.50,0.05)
    bboxplot = [-100,20,0,80]
    modelnames  = ("Constant h","Vary h","Entraining")
    
    #%Experiment names
    # -- SelectExperiment -- 
    fid   = 0
    expid = fnames[fid]
    regid = 4

    
    # Load post-propcssed output
    ldpath = datpath + "proc/AMV_Region_%s.npz" % expid
    ld = np.load(ldpath,allow_pickle=True)
    
    # Load. Things are organized by region, then by model
    amvidx = ld['amvidx_region'].item()[regid]
    amvpat = ld['amvpat_region'].item()[regid]
    
    # Load lat/lon regional
    ld = np.load(datpath+"stoch_output_"+expid+".npz",allow_pickle=True)
    lon = ld['lon']#np.load(datpath+"lon.npy")
    lat = ld['lat']#np.load(datpath+"lat.npy")
    
    
    fig,axs = plt.subplots(1,3,figsize=(12,3.5),subplot_kw={'projection':ccrs.PlateCarree()})
    for p in range(len(amvpat)):
        ax = axs.flatten()[p]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
        ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
        cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
        ax.clabel(cl,levels=cl_int,fontsize=8)
        #pcm = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance)
        #ax.set_title(modelnames[p])
        #fig.colorbar(pcm,ax=ax,fraction=0.036)
        ax.set_title(modelnames[p])
    fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.75,pad=0.01)#,pad=0.015)
    plt.suptitle("%s AMV Pattern ($\circ C$ per $\sigma_{AMV}$, Forcing: %s)"%(regions[regid],frcnamelong[fid]),y=0.90,fontsize=14)
    plt.savefig("%sAMV_Pattern_%s_region%s.png"%(figpath,expid,regions[regid]),dpi=200,bbox_inches = 'tight')
    
    
    
    
    #%% Individual AMV Plots (rather than panel based)
    
    for p in range(len(amvpat)):
        fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
        cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
        ax.clabel(cl,levels=cl_int,fmt="%.2f",fontsize=8)
        
        ax.set_title("%s AMV " % (regions[regid]) + modelnames[p] + " ($\circ C$ per $\sigma_{AMV}$) \n Forcing: %s" % frcnamelong[fid])
        fig.colorbar(pcm,ax=ax,orientation='horizontal',shrink=0.75)#,pad=0.015)
        plt.savefig("%sAMV_Pattern_%s_region%s_model%s.png"%(figpath,expid,regions[regid],modelnames[p]),dpi=200,bbox_inches='tight')
    
    #%% End AMV Visualization
    
    
    #%% Load in regional SSTs, and perform spectral analysis
    
    # Stored in nested dict...
    # Outer dict: here keys are 0-3, for each region
    # Inner dict: keys are 0-2, for each model
    
    # Load in SSTs for each region
    sstdicts = []
    for f in range(len(fnames)):
        # Load the dictionary
        expid = fnames[f]
        #expid = "forcing%s_%iyr_run%s" % (fnames[f],nyrs,runids[f]) # Get experiment name
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
    
    
    #%% Load corresponding CESM Data
    
    expid      = "CESM1-PIC"
    rsst_fn    = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,expid)
    sstcesm    = np.load(rsst_fn,allow_pickle=True).item()
    cesmname   =  ["CESM-FULL","CESM-SLAB"]
    
    # Identify the variance for each region and load in numpy arrays
    #sstallcesm  = np.zeros((len(regions),2,nyrs*12)) # Forcing x Region x Model x Time
    sstvarscesm = np.zeros((len(regions),2)) # Forcing x Region x Model
    
    for rid in range(len(regions)-1):
        for model in range(len(cesmname)):
            sstin  = sstcesm[rid][model]
            sstvar = np.var(sstin)
            print("Variance for region %s, model %s is %f" % (regions[rid],cesmname[model],sstvar))
            sstvarscesm[rid,model]   = sstvar
            
    #%% Make a plot of the variance... compare for each forcing scenario
    
    elabels = ["%i EOFs"% x for x in neofs]
    eofaxis = np.arange(0,51,1)
    #xtk = [1,2,3,5,10,25,50]
    xtk = [1,2,5,10,25,50]
    #xtkplot = 
    for rid in range(len(regions)):
        fig,ax  = plt.subplots(1,1,figsize=(8,4))
        for model in range(len(modelnames)):
            ax.plot(xtk,sstvars[:,rid,model],marker="o",label=modelnames[model])
        
        ax.axhline(sstvarscesm[rid,0],color="k",label="CESM-FULL (%.3f $degC^{2}$)"%(sstvarscesm[rid,0]),ls='dashed')
        ax.axhline(sstvarscesm[rid,1],color="gray",label="CESM-SLAB (%.3f $degC^{2}$)" %(sstvarscesm[rid,1]),ls='dashed')
        ax.legend(ncol=2)
        ax.set_xticks(xtk)
        #ax.set_xticklabels(elabels)
        ax.set_ylabel("$(\circ C)^{2}$")
        ax.set_xlabel("Number of EOF Patterns")
        ax.set_title("%s AMV Index Variance vs. Number of EOF Patterns \n used to force the stochastic model" % (regions[rid]))
        ax.grid(True,ls='dotted')
        plt.savefig("%svariance_vs_nEOF_by_model_runid%s_nyr%i_region%s.png"%(figpath,runid,nyrs,regions[rid]),dpi=150)
    
    #%% Do some spectral analysis
    nsmooth = 100
    pct     = 0.10
    
    rid     = 2
    fid     = 3
    dofs    = [1000,1000,898,1798] # In number of years
    
    # Unpack and stack data
    insst = []
    
    for model in range(len(modelnames)):
        insst.append(sstall[fid,rid,model,:]) # Append each stochastic model result
        #print(np.var(sstall[fid,rid,model,:]))
    insst.append(sstcesm[rid][0]) # Append CESM-FULL
    insst.append(sstcesm[rid][1]) # Append CESM-SLAB 
    
    insstvars = []
    for s in insst:
        insstvars.append(np.var(s))
        #print(np.var(s))
    
    # Calculate Spectra and confidence Intervals
    specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct)
    alpha = 0.05
    bnds = []
    for nu in dofs:
        lower,upper = tbx.confid(alpha,nu*2)
        bnds.append([lower,upper])
    
    
    #print(insstvars)
    
    #%% Plot the full spectra (Frequency x Power)
    
    
    
    #%% Make the plot (Frequency x Power)
    
    
    timemax = None
    xlms = [0,0.2]
    xtks = [0,0.02,0.04,0.1,0.2]
    xtkl = 1/np.array(xtks)
    dt   = 3600*24*30
    speccolors = ["b","r","m","k","gray"]
    specnames  = np.hstack([modelnames,cesmname])
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],insstvars[i]) for i in range(len(insst)) ]
    
    
    plottitle  = "%s AMV Index Spectra for %s \n nsmooth=%i, taper=%.2f"%(regions[rid],frcnamelong[fid],nsmooth,pct)
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    ax = viz.plot_freqxpower(specs,freqs,speclabels,speccolors,
                         ax=ax,plotconf=CCs,plottitle=plottitle)
    plt.savefig("%s%s%sspectra_nsmooth%i_taper%03d_freqxpower.png"% (figpath,fnames[fid],regions[rid],nsmooth,pct*100),dpi=200,bbox_inches='tight')
    
    
    #%% Linear-Linear Multidecadal Plot focusing on this band
    
    plottitle  = "%s AMV Index Spectra for %s \n nsmooth=%i, taper=%.2f"%(regions[rid],frcnamelong[fid],nsmooth,pct)
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    ax = viz.plot_freqlin(specs,freqs,speclabels,speccolors,
                         ax=ax,plotconf=CCs,plottitle=plottitle)
    plt.savefig("%s%s%sspectra_nsmooth%i_taper%03d_linlin.png"% (figpath,fnames[fid],regions[rid],nsmooth,pct*100),dpi=200,bbox_inches='tight')
    
    
    #%% Log-log spectral plot
    
    plottitle  = "%s AMV Index Spectra for %s \n nsmooth=%i, taper=%.2f"%(regions[rid],frcnamelong[fid],nsmooth,pct)
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    ax = viz.plot_freqlog(specs,freqs,speclabels,speccolors,
                         ax=ax,plotconf=CCs,plottitle=plottitle)
    plt.savefig("%s%s%sspectra_nsmooth%i_taper%03d_loglog.png"% (figpath,fnames[fid],regions[rid],nsmooth,pct*100),dpi=200,bbox_inches='tight')

