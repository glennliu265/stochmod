#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:19:12 2020

Forcing sensitivity analysis (fscale) with stochmod_region output


@author: gliu
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

import time
import cmocean

from scipy.io import loadmat

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz
#%% 

# Set run parameters to test
runid    = "002"
fscales   = [1,10,100]
funiform = 5   
nyrs     = 1000

# Point to Plot
lonf = -30
latf = 65

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200820/'

# Autocorrelation Options
lags = np.arange(0,61,1)

# String Names Set

modelname = ("Fixed","Max", "Seasonal", "Entrain")
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$",
               "$NAO_{DJFM}  &  NHFLX_{Mon}$",
               "$(NAO  &  NHFLX)_{Mon}$",
               "$EAP_{DJFM}$",
               "$(NAO+EAP)_{DJFM}$")

# Set regions for analysis
bbox_SP = [-60,20,40,60]
bbox_ST = [-80,20,20,40]
bbox_TR = [-75,20,0,20]
bbox_NA = [-80,20 ,0,60]#[-75,20,0,90]

#regions = ("SPG","STG","TRO","NAT")
#bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
regions = ("SPG","STG","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_NA)

# AMV Calculation Box
amvbox  = [-80,0,0,60]

#%% Script Start
loadstart= time.time()

# Read in Stochmod SST Data

lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Load MLD Data
mld = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD

# Read in damping data for the coordinates
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])



# Make some strings
print("Data Loaded in %.2fs"%(time.time()-loadstart))

bbm = [-100,40,-20,90] # Mapping bbox
#%% Plot Bounding Box

    
    
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize =(4,4))
ax = viz.init_map([-100,40,-20,90],ax=ax)
lwb = 1.5

ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb)
ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
ax,l4 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb)
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb)


ax.legend([l1,l2,l3],labels=regions)
plt.savefig(outpath+"bboxes_viz_fscale.png",dpi=200)

#%% Load the data in


# Loop for each forcing...
for f in range(len(fscales)):

    
    fscale = fscales[f]
    expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
    
    
    
    # Load data
    loadstart= time.time()
    sst = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s_fscale%03d.npy"%(nyrs,funiform,runid,fscale),allow_pickle=True).item()
    sst[3] = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s_fscale%03d.npy"%(nyrs,funiform,runid,fscale))
    print("Data Loaded in %.2fs"%(time.time()-loadstart))      
    
    
    
    # Get regional data and take averages
    sstregion = {}
    for r in range(3):
        bbox = bboxes[r]
        sstr = {}
        for model in range(4):
            tsmodel = sst[model]
            tsmodel,_,_=proc.sel_region(tsmodel,lonr,latr,bbox)
            sstr[model] = np.nanmean(tsmodel,(0,1)) 
        sstregion[r] = sstr
    
    #% Create a line plot for each region
    # for model in range(4):
        
    #     fig,axs = plt.subplots(3,1,figsize=(6,4))
    #     for r in range(3):
    #         tsmodel = sstregion[r][model]
            
    #         figtitle = regions[r]
    #         axs[r] = viz.plot_annavg(tsmodel,"degC",figtitle,ax=axs[r])
    #         #axs[r].set_ylim([-1,1])
            
    #     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.suptitle("SST Timeseries for %s model, fscale %ix" % (modelname[model],fscale))
    #     plt.savefig("%sForcingSense_Timeseries_%s_model%s.png"%(outpath,expid,modelname[model]))
    
    #% Create lineplot just for the north atlantic (r=2)
    for model in range(4):  
       fig,ax = plt.subplots(1,1,figsize=(5,2.5))
       tsmodel = sstregion[2][model]
       figtitle = "SST (NAtl. Avg) Timeseries, fscale %ix \n" % (fscale)
       ax = viz.plot_annavg(tsmodel,"degC",figtitle,ax=ax)
       fig.tight_layout(rect=[0, 0.03, 1, 0.95])
       plt.savefig("%sForcingSense_Timeseries_%s_NAtl_Avg_model%s.png"%(outpath,expid,modelname[model]))
    
    
    
    #% Calculate AMV Index
    amvtime = time.time()
    amvidx = {}
    amvpat = {}
    for model in range(4):
        amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbox)
    print("Calculated AMV variables in %.2f" % (time.time()-amvtime))
    
    
    
    #% Make AMV Spatial Plots
    cmap = cmocean.cm.balance
    cmap.set_bad(color='yellow')
    #cint = np.arange(-1,1.1,0.1)
    #clab = cint
    fig,axs = plt.subplots(1,4,figsize=(12,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
    for mode in range(4):
        varin = np.transpose(amvpat[mode],(1,0))
        viz.plot_AMV_spatial(varin,lonr,latr,bbm,cmap,pcolor=0,ax=axs[mode])
        axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)   
    #plt.suptitle("AMV Pattern | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale),ha='center')
    #fig.tight_layout(rect=[0, 0.03, .75, .95])
    outname = outpath+'AMVpattern_%s_allmodels.png' % (expid)
    plt.savefig(outname, bbox_inches="tight",dpi=200)
    
    
    #%Make AMV Time Plots
    xlm = [24,240]
    ylm = [-0.5,0.5]
    #xtk = np.arange(xlm[0],xlm[1]+20,20)
    fig,axs = plt.subplots(1,4,figsize=(12,1.5))
    for mode in range(4): 
        viz.plot_AMV(amvidx[mode],ax=axs[mode])
        axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)
        axs[mode].set_xlim(xlm)
        #axs[mode].set_xticks(xtk)
        axs[mode].set_xlabel('Year')
        #axs[mode].set_ylim(ylm)
    axs[0].set_ylabel('AMV Index')
    #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    outname = outpath+'AMVIndex_%s_allmodels.png' % (expid)
    plt.savefig(outname, bbox_inches="tight",dpi=200)
    
    #% Compute Autocorrelatioin for each region
    start = time.time()
    kmonths = {}
    autocorr_region = {}
    for r in range(3):
        bbox = bboxes[r]
        
        autocorr = {}
        for model in range(4):
            
            # Get sst and havg
            havg,_,_= proc.sel_region(mld,lon,lat,bbox)
            
            # Find kmonth
            havg = np.nanmean(havg,(0,1))
            kmonth = havg.argmax()+1
            kmonths[r] = kmonth
            
            
            # Take regional average
            tsmodel = sstregion[r][model]
            tsmodel = proc.year2mon(tsmodel) # mon x year
            
            
            
            # Deseason (No Seasonal Cycle to Remove)
            tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
            
            # Calculate Autocorrelation
            autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
        
        autocorr_region[r] = autocorr.copy()
    print("Calculated regional autocorrelation in %.2f" % (time.time()-start))
    
    
    
    #% Plot autocorrelation
    xlim = [0,36]
    xtks = np.arange(0,39,3)
    ylim = [-0.25,1]
    ytks = np.arange(-.25,1.25,0.25)
    
    regioncolor= ('b','r','k')
    
    fig,axs = plt.subplots(1,4,figsize=(12,2))
    plt.style.use("seaborn")
    plt.style.use("seaborn-bright")
    for model in range(4):
        ax = axs[model]
        
        for r in range(3):
            label =  " %s basemonth %i" % (regions[r],kmonths[r])
            ax.plot(lags,autocorr_region[r][model],label=label,color=regioncolor[r])
            
        ax.set_title("%s" % (modelname[model]))
        
        #plt.xticks(xtk)
        if model == 0:
            ax.legend(prop={'size':8})
        plt.grid(True)
        ax.set_xlim(xlim)
        ax.set_xticks(xtks)
        ax.set_ylim(ylim)
        ax.set_yticks(ytks)
        
    plt.suptitle("SST Autocorrelation, Forcing %s" % forcingname[funiform])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath+"Region_SST_Autocorrelation_modelALL_%s.png"%(expid),dpi=200)
    
    
    # Calculate different AMVs for each region
    amvbboxes = [bbox_SP,bbox_ST,amvbox]
    amvidx_region = {}
    amvpat_region = {}
    for region in range(3):
        #% Calculate AMV Index
        amvtime = time.time()
        amvidx = {}
        amvpat = {}
        for model in range(4):
            amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbboxes[region])
        print("Calculated AMV variables for region %s in %.2f" % (regions[region],time.time()-amvtime))
        
        amvidx_region[region] = amvidx
        amvpat_region[region] = amvpat
    
    
    
        #% Make AMV Spatial Plots
        cmap = cmocean.cm.balance
        cmap.set_bad(color='yellow')
        #cint = np.arange(-1,1.1,0.1)
        #clab = cint
        fig,axs = plt.subplots(1,4,figsize=(12,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
        for mode in range(4):
            varin = np.transpose(amvpat[mode],(1,0))
            viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,pcolor=0,ax=axs[mode])
            axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)   
        #plt.suptitle("AMV Pattern | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale),ha='center')
        #fig.tight_layout(rect=[0, 0.03, .75, .95])
        outname = outpath+'AMVpattern_%s_allmodels_region%s.png' % (expid,regions[region])
        plt.savefig(outname, bbox_inches="tight",dpi=200)
        
        
        #%Make AMV Time Plots
        xlm = [24,240]
        ylm = [-0.5,0.5]
        #xtk = np.arange(xlm[0],xlm[1]+20,20)
        fig,axs = plt.subplots(1,4,figsize=(12,1.5))
        for mode in range(4): 
            viz.plot_AMV(amvidx[mode],ax=axs[mode])
            axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)
            axs[mode].set_xlim(xlm)
            #axs[mode].set_xticks(xtk)
            axs[mode].set_xlabel('Year')
            #axs[mode].set_ylim(ylm)
        axs[0].set_ylabel('AMV Index')
        #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        outname = outpath+'AMVIndex_%s_allmodels_region%s.png' % (expid,regions[region])
        plt.savefig(outname, bbox_inches="tight",dpi=200)
