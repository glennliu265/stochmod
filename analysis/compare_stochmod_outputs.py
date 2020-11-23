#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make some comparison plots
Created on Mon Aug 24 06:32:54 2020

@author: gliu
"""

import numpy as np
import sys

import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs

#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
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

#%% Set some parameters





# Lags for labeling
lags = np.arange(0,61,1)

# Options to determine the experiment ID
naoscale  = 10 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
# Do a stormtrackloop
#runids = ("003","004","005")
#funiforms = (0,1,2,5,6)

runids=['006']


# Set region variables
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
rcol = ('b','r',[0,1,0],'k')
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]


# Set Forcing Names and colors
funiforms=[0,1,2,5,6]
fnames  = ["Random","Uniform","NAO (DJFM)","EAP (DJFM)","NAO+EAP (DJFM)"]
fcolors = ["teal","springgreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']

# Set Model Names
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")



#%% Load in the data

# Load lat/lon coordinates
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Load dictionaries to store outputs as [forcing][region][model]
amvsp = {} # AMV Pattern
amvid = {} # AMV Index
sstr  = {} # SST Regional
ssta  = {} # Autocorrelation
kmon  = {} # Entrainment Month

for runid in runids:
    for funiform in funiforms:
        
        if funiform < 2:
            fscale = 1
        else:
            fscale = naoscale
            
        # Set experiment ID
        expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
        
        # Load AMV
        amvload = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)
        amvsp[funiform] = amvload['amvpat_region'].item() # Contains [region][model]
        amvid[funiform] = amvload['amvidx_region'].item()
        
        # Load SST Autocorrelation
        acload = np.load("%sSST_Region_Autocorrelation_%s.npz"%(outpathdat,expid),allow_pickle=True)
        ssta[funiform] = acload['autocorr_region'].item()
        kmon[funiform] = acload['kmonths'].item()
        
        # Load Regional SST Data
        sstr[funiform]=np.load("%sSST_RegionAvg_%s.npy"%(outpathdat,expid),allow_pickle=True).item()
        
        print("Loaded in post-processed data for %s" % expid)
        
# Load CESM Data
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()


        
#%% Make Autoorrelation Plots for each model, each region

rid = 3
xlm = [0,36]
xtk = np.arange(0,39,3)


nregions=4

for rid in range(nregions):
    accesm = cesmauto[rid]
    
    # Make a plot for each model
    for model in range(4):
        
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        plt.style.use('seaborn')
        # Plot CESM Ensemble Data
        ax = viz.ensemble_plot(accesm,0,ax=ax,color=rcol[rid],ysymmetric=0,ialpha=0.05)
        
        
        for f in range(len(funiforms)):
            
            funiform = funiforms[f]
            
            # Get Autocorrelation data to plot
            acplot = ssta[funiform][rid][model]
            ax.plot(lags,acplot,color=fcolors[f],ls=fstyles[f],label=fnames[f])
        
        plt.legend(ncol=3)
        plt.xticks(xtk)
        plt.xlim(xlm)
        plt.title("%s SST Autocorrelation for Month %02d, %s" % (regions[rid],kmon[funiform][rid],modelname[model]))
        plt.xlabel("Lags (Months)")
        plt.ylabel("Correlation")
        plt.savefig("%s%s_SST_Autocorrelation_ForcingCompare_model%i.png" % (outpathfig,regions[rid],model),dpi=200)
    


#%% Make AMV Plots

bbox = [-100,40,-20,90]
runid = runids[0]
cint = np.arange(-.5,.6,.1)
clabs = np.arange(-.5,.75,.25)
for f in range(len(funiforms)):
    funiform = funiforms[f]
    
    expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
    
    amvidx = amvid[funiform]
    amvpat = amvsp[funiform]
            
    for region in range(4):
    
            #% Make AMV Spatial Plots
            cmap = cmocean.cm.balance
            cmap.set_bad(color='yellow')
            #cint = np.arange(-1,1.1,0.1)
            #clab = cint
            fig,axs = plt.subplots(1,4,figsize=(12,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
            
            
            for mode in range(4):
                print("Now on mode %i region %i f %i"% (mode,region,f))
                
                varin = np.transpose(amvpat[region][mode],(1,0))
                viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,cint=cint,clab=clabs,pcolor=0,ax=axs[mode],fmt="%.1f",)
                axs[mode].set_title("%s" % modelname[mode],fontsize=12)   
            #plt.suptitle("AMV Pattern | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale),ha='center')
            #fig.tight_layout(rect=[0, 0.03, .75, .95])
            outname = outpathfig+'%s_AMVpattern_%s_allmodels.png' % (regions[region],expid)
            plt.savefig(outname, bbox_inches="tight",dpi=200)
            
            
            #%Make AMV Time Plots
            xlm = [24,240]
            ylm = [-0.5,0.5]
            
            #xtk = np.arange(xlm[0],xlm[1]+20,20)
            fig,axs = plt.subplots(1,4,figsize=(12,1.5))
            for mode in range(4): 
                
                viz.plot_AMV(amvidx[region][mode],ax=axs[mode])
                axs[mode].set_title("%s" % modelname[mode],fontsize=12)
                axs[mode].set_xlim(xlm)
                #axs[mode].set_xticks(xtk)
                axs[mode].set_xlabel('Year')
                #axs[mode].set_ylim(ylm)
            axs[0].set_ylabel('AMV Index')
            #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            outname = outpathfig+'%s_AMVIndex_%s_allmodels.png' % (regions[region],expid)
            plt.savefig(outname, bbox_inches="tight",dpi=200)



