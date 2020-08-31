#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:42:50 2020

@author: gliu

# Average or combine stochastic model outputs

"""

import numpy as np
import sys

import cmocean
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + 'proc/'
    outpathfig  = projpath + '02_Figures/20200823/'
   
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
naoscale  = 10 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over

# Set experiment variables to read
runids = ("001","002","003","004","005")
funiforms = (0,1,2,5,6)

# Set region variables
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")
bboxes  = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
rcol    = ('b','r',[0,1,0],'k')
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]

# Set model variables
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")

# Set forcing sttle
fnames  = ["Random","Uniform","NAO (DJFM)","EAP (DJFM)","NAO+EAP (DJFM)"]
fcolors = ["teal","mediumseagreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']

# Load CESM Data
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()
#%%

# Load lat/lon coordinates
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Preallocate--Goal [runid x funiform x region x model x variable]
nruns   = len(runids)
nftypes = len(funiforms)
nregions = len(regions)
nmod     = len(modelname)
amvsp = np.zeros((nruns,nftypes,nregions,nmod,97,117)) * np.nan # ... x lat x lon
amvid = np.zeros((nruns,nftypes,nregions,nmod,1000)) * np.nan # ... x year
ssta  = np.zeros((nruns,nftypes,nregions,nmod,37)) # ... x lag
kmon  = np.zeros((nruns,nftypes,nregions)) 
sstr  = np.zeros((nruns,nftypes,nregions,nmod,12000)) # ... x month

# Load all data into arrays
for rid in range(nruns):    # Loop for exp id

    for f in range(nftypes):# Loop for forcing
        runid    = runids[rid]
        funiform = funiforms[f]
        if funiform < 2:
            fscale = 1
        else:
            fscale = naoscale
        
        # Set experiment ID
        expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
        
        # Load AMV
        amvload  = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)
        amvsp_in = amvload['amvpat_region'].item() # Contains [funiform][region][model][97x117]
        amvid_in = amvload['amvidx_region'].item() # "" [1000]
        
        # Load SST Autocorrelation
        acload  = np.load("%sSST_Region_Autocorrelation_%s.npz"%(outpathdat,expid),allow_pickle=True)
        ssta_in = acload['autocorr_region'].item() # "" [37] 
        kmon_in = acload['kmonths'].item() # "" [0] [model] = scalar
        
        # Load Regional SST Data
        sstr_in =np.load("%sSST_RegionAvg_%s.npy"%(outpathdat,expid),allow_pickle=True).item() # "" [12000]
    
        for r in range(nregions): # Loop by region
            for model in range(4): # Loop by model
                amvsp[rid,f,r,model,:,:] = amvsp_in[r][model].copy()
                amvid[rid,f,r,model,:]   = amvid_in[r][model].copy()
                ssta[rid,f,r,model,:]    = ssta_in[r][model].copy()
                kmon[rid,f,r]            = kmon_in[r].copy()
                sstr[rid,f,r,model,:]    = sstr_in[r][model].copy()
                
    
        print("Loaded in post-processed data for %s" % expid)
        
# Get experiment averaged values
amvsp_avg = amvsp.mean(0)
amvid_avg = amvid.mean(0)
ssta_avg  = ssta.mean(0)
sstr_avg  = sstr.mean(0)


#%% Get a sense of how the parameters fluctuate by experiment

modid = 0 # Selection model id
r   = 3 # Select region id
fid   = 4 # Forcing index

testid = "funiform%i_%s_model%i" % (funiforms[fid],regions[r],modid)
bbox = [-80,0,0,80]

# Plot AMV Pattern
invar = amvsp[:,fid,r,modid,:,:]
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-1,1.1,0.1)
#clab = cint
fig,axs = plt.subplots(1,5,figsize=(15,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
for rid in range(nruns):
    print("Plotting for run %s" % runids[rid])
    
    varplot = invar[rid,:,:].T
    
    viz.plot_AMV_spatial(varplot,lonr,latr,bbox,cmap,pcolor=0,cint=cint,ax=axs[rid],fmt="%.1f")
    axs[rid].set_title("run%s" % runids[rid],fontsize=12)   

outname = outpathfig+'AMVpattern_runcomparison_%s.png' % (testid)
plt.savefig(outname, bbox_inches="tight",dpi=200)

# Plot autocorrelation for each forcing
xtk = np.arange(0,39,3)
for r in range(nregions):
    for modid in range(4):
        fig,ax=plt.subplots(1,1,figsize=(6,4))
        accesm = cesmauto[r]
        ax,ln = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,plotindv=False,returnlegend=False,returnline=True)
        lns = [ln]
        for fid in range(nftypes):
        
            varplot=ssta[:,fid,r,modid,:]
            ax,ln = viz.ensemble_plot(varplot,0,ax=ax,color=fcolors[fid],returnlegend=False,returnline=True,plotindv=False)

            lns = lns + [ln]
            
        labs = ["CESMLE"] + fnames
        ax.legend(lns,labs,loc=0,ncol=2)
        ax.set_ylim([-0.25,1])
        ax.set_xlim([0,36])
        ax.set_xticks(xtk)
        ax.set_title("%s Average SST Autocorrelation for Model %s, Month %i" % (regions[r],modelname[modid],kmon[rid,fid,r]))         
        plt.savefig("%sAutocorr_runcomparison_model%i_region%s.png"%(outpathfig,modid,regions[r]),dpi=200)                   
    
#%% Do autocorrelation plots, but just the ensemble average
fillrange=0
# Plot autocorrelation for each forcing
lags = np.arange(0,37,1)
xtk = np.arange(0,39,3)
for r in range(nregions):
    for modid in range(4):
        fig,ax=plt.subplots(1,1,figsize=(6,4))
        accesm = cesmauto[r]
        ax,ln = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,plotindv=True,ialpha=0.05,returnlegend=False,returnline=True)
        lns = [ln]
        
        if fillrange == 1:
            # Plot the ranges
            for fid in range(nftypes):
                varplot=ssta[:,fid,r,modid,:]
                
                varmax  = varplot.max(0)
                varmin  = varplot.min(0)
                plt.fill_between(lags,varmin,varmax,color=fcolors[fid],alpha=0.3)
                
            
        # Plot the ensaverages
        for fid in range(nftypes):
            varplot=ssta[:,fid,r,modid,:]
            varmean = varplot.mean(0)
            ln = ax.plot(lags,varmean,color=fcolors[fid],ls=fstyles[fid])
            #ax,ln = viz.ensemble_plot(varplot,0,ax=ax,color=fcolors[fid],returnlegend=False,returnline=True,plotindv=False)

            lns = lns + ln
            
        labs = ["CESMLE"] + fnames
        ax.legend(lns,labs,loc=0,ncol=2)
        ax.set_ylim([-0.4,1])
        ax.set_xlim([0,36])
        ax.set_xticks(xtk)
        ax.set_title("%s Average SST Autocorrelation for Model %s, Month %i" % (regions[r],modelname[modid],kmon[rid,fid,r]))         
        #plt.savefig("%sAutocorr_runcomparison_model%i_region%s.png"%(outpathfig,modid,regions[r]),dpi=200) 
        plt.savefig("%sSST_Autocorrelation_ForcingCompare_%s_model%i_fillrange%i.png" % (outpathfig,regions[r],modid,fillrange),dpi=200)
        
#%% Create AMV Plots for ensemble average


invar = amvsp.mean(0) #[5 x 4 x 4 x 97 x 117]
invar2 = amvid.mean(0)
bbox = [-100,0,0,90]
runid = runids[0]
cint = np.arange(-1,1.1,0.1)

for f in range(len(funiforms)):
    funiform = funiforms[f]
    
    expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
    
    #amvidx = amvid[funiform]
    #amvpat = amvsp[funiform]
            
    for region in range(4):
    
            #% Make AMV Spatial Plots
            cmap = cmocean.cm.balance
            cmap.set_bad(color='yellow')
            #cint = np.arange(-1,1.1,0.1)
            #clab = cint
            fig,axs = plt.subplots(1,4,figsize=(12,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
            
            for mode in range(4):
                print("Now on mode %i region %i f %i"% (mode,region,f))
                
                varin = invar[f,region,mode,:,:].T#  np.transpose(amvpat[region][mode],(1,0))
                viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,pcolor=0,ax=axs[mode],fmt="%.2f",cint=cint,fontsize=8)
                axs[mode].set_title("%s" % modelname[mode],fontsize=12)   
            #plt.suptitle("AMV Pattern | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale),ha='center')
            #fig.tight_layout(rect=[0, 0.03, .75, .95])
            outname = outpathfig+'AMVpattern_%s_allmodels_region%s.png' % (expid,regions[region])
            plt.savefig(outname, bbox_inches="tight",dpi=200)
            
            
            #%Make AMV Time Plots
            xlm = [24,240]
            ylm = [-0.5,0.5]
            
            #xtk = np.arange(xlm[0],xlm[1]+20,20)
            fig,axs = plt.subplots(1,4,figsize=(12,1.5))
            for mode in range(4): 
                
                viz.plot_AMV(invar2[f,region,mode,:],ax=axs[mode])
                axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)
                axs[mode].set_xlim(xlm)
                #axs[mode].set_xticks(xtk)
                axs[mode].set_xlabel('Year')
                #axs[mode].set_ylim(ylm)
            axs[0].set_ylabel('AMV Index')
            #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            outname = outpathfig+'AMVIndex_%s_allmodels_region%s.png' % (expid,regions[region])
            plt.savefig(outname, bbox_inches="tight",dpi=200)
#%% Create individual AMV Plots

invar = amvsp.mean(0) #[5 x 4 x 4 x 97 x 117]
invar2 = amvid.mean(0)
bbox = [-100,0,0,90]
runid = runids[0]

cint = np.arange(-1,1.1,0.1)
for f in range(nftypes):
    funiform = funiforms[f]
    
    expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
    
    #amvidx = amvid[funiform]
    #amvpat = amvsp[funiform]
            
    for region in range(4):
    
            #% Make AMV Spatial Plots
            cmap = cmocean.cm.balance
            cmap.set_bad(color='yellow')
            #cint = np.arange(-1,1.1,0.1)
            #clab = cint
            
            for mode in range(4):
                print("Now on mode %i region %i f %i"% (mode,region,f))
                
                fig,axs = plt.subplots(1,1,figsize=(4,3),subplot_kw={'projection':ccrs.PlateCarree()})
                varin = invar[f,region,mode,:,:].T#  np.transpose(amvpat[region][mode],(1,0))
                viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,pcolor=0,ax=axs,fmt="%.2f",fontsize=9,clabelBG=True,cint=cint)
                axs.set_title("%s AMV Pattern \n %s; Forcing: %s" % (regions[region],modelname[mode],fnames[f]),fontsize=12)  
                
                
                #plt.suptitle("AMV Pattern | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale),ha='center')
                #fig.tight_layout(rect=[0, 0.03, .75, .95])
                outname = outpathfig+'AMVpattern_%s_%s_%s_funiform%s.png' % (expid,regions[region],modelname[mode],fnames[f])
                plt.savefig(outname, bbox_inches="tight",dpi=200)
            
            
            # #%Make AMV Time Plots
            # xlm = [24,240]
            # ylm = [-0.5,0.5]
            
            # #xtk = np.arange(xlm[0],xlm[1]+20,20)
            # fig,axs = plt.subplots(1,4,figsize=(12,1.5))
            # for mode in range(4): 
                
            #     viz.plot_AMV(invar2[f,region,mode,:],ax=axs[mode])
            #     axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)
            #     axs[mode].set_xlim(xlm)
            #     #axs[mode].set_xticks(xtk)
            #     axs[mode].set_xlabel('Year')
            #     #axs[mode].set_ylim(ylm)
            # axs[0].set_ylabel('AMV Index')
            # #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
            # #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            # outname = outpathfig+'AMVIndex_%s_allmodels_region%s.png' % (expid,regions[region])
            # plt.savefig(outname, bbox_inches="tight",dpi=200)