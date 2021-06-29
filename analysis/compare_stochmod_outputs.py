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
    outpathfig  = projpath + '02_Figures/20210322_AMVTeleconf/'
    
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
lags = np.arange(0,37,1)

# Options to determine the experiment ID
naoscale  = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
# Do a stormtrackloop
#runids = ("003","004","005")
#funiforms = (0,1,2,5,6)

runids=['303']
funiforms=[1.5,3,5.5]

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
#funiforms=[0,1,2,5,6]
funiforms=[3]
#funiforms=[0,1,3,5.5,7,]
#fnames=["NAO+EAP (DJFM-MON)"]
#fnames  = ["Random","Uniform","NAO","EAP","NAO+EAP"]
fnames = ["NAO (DJFM-MON)"]
fcolors = ["teal","springgreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']
mconfig = "SLAB_PIC"
applyfac = 2

# Set Model Names
modelname = ("MLD Fixed","MLD Max", "MLD Seasonal", "MLD Entrain")


fscale = naoscale
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
        
        # Set experiment ID
        expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)
        
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
if mconfig == "FULL_HTR":
    cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()
elif mconfig == "SLAB_PIC":
    cesmauto = np.load(projpath + "01_Data/CESM-SLAB_PIC_autocorrelation_Regions.npy")

        
#%% Make Autoorrelation Plots for each model, each region

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monname=('January','February','March','April','May','June','July','August','September','October','November','December')




rid = 3
xlm = [0,36]
xtk = np.arange(0,39,3)
xtk2 =  np.arange(xlm[0],xlm[1],3)

nregions=4

for rid in range(nregions):
    
    if mconfig == "FULL_HTR":
        accesm = cesmauto[rid]
    elif mconfig == "SLAB_PIC":
        accesm = cesmauto[:,rid]
    
    # Make a plot for each model
    for model in range(4):
        # Get Month Index
        kmonth = kmon[funiform][rid]
        mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
        mons3tile = np.roll(mons3tile,-kmonth)
        
        fig,ax = plt.subplots(1,1,figsize=(6,4))
        plt.style.use('seaborn')
        
        ax2 = ax.twiny()
        ax2.set_xlim(xlm)
        ax2.set_xticks(xtk2)
        ax2.set_xticklabels(mons3tile[xtk2], rotation = 45)
        ax2.set_axisbelow(True)
        ax2.grid(zorder=0,alpha=0)
        # Plot CESM Ensemble Data
        if mconfig == "FULL_HTR":
            ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)
        elif mconfig == "SLAB_PIC":
            ax.plot(lags,accesm,label="CESM-SLAB (PIC)",color='k')
        
        for f in range(len(funiforms)):
            
            funiform = funiforms[f]
            
            # Get Autocorrelation data to plot
            acplot = ssta[funiform][rid][model]
            ax.plot(lags,acplot,color=fcolors[f],ls=fstyles[f],label=fnames[f])
        

        
        ax.legend(ncol=3)
        ax.set_xticks(xtk)
        ax.set_xlim(xlm)
        ax.set_title("%s SST Autocorrelation for %s, %s" % (regions[rid],monname[kmonth],modelname[model]))
        ax.set_xlabel("Lags (Months)")
        ax.set_ylabel("Correlation")
        plt.tight_layout()
        plt.savefig("%s%s_SST_Autocorrelation_ForcingCompare_model%i_%s_applyfac%i.png" % (outpathfig,regions[rid],model,mconfig,applyfac),dpi=200)
    



#%% Make AMV Plots

bbox = [-100,40,0,90]
runid = runids[0]
cint = np.arange(-.5,.55,.05)
clabs = np.arange(-.5,.75,.25)
vscale = 10
for f in range(len(funiforms)):
    funiform = funiforms[f]
    
    # Set experiment ID
    expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)  
    
    amvidx = amvid[funiform]
    amvpat = amvsp[funiform]
            
    for region in range(4):

            #% Make AMV Spatial Plots
            cmap = cmocean.cm.balance
            cmap.set_bad(color='yellow')
            fig,axs = plt.subplots(1,4,figsize=(12,1.5),subplot_kw={'projection':ccrs.PlateCarree()})
            for mode in range(4):
                
                print("Now on mode %i region %i f %i"% (mode,region,f))
                
                varin = np.transpose(amvpat[region][mode],(1,0))*vscale

                #viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,cint=cint,clab=clabs,pcolor=0,ax=axs[mode],fmt="%.1f",)
                viz.plot_AMV_spatial(varin,lonr,latr,bbox,cmap,cint=cint,labels=False,pcolor=0,ax=axs[mode],fmt="%.1f",)
                #axs[mode].set_title("MLD %s" % modelname[mode],fontsize=12)
                axs[mode].set_title("%s" % modelname[mode],fontsize=12)

            outname = outpathfig+'%s_AMVpattern_%s_allmodels_vscale%i.png' % (regions[region],expid,vscale)
            plt.savefig(outname, bbox_inches="tight",dpi=200)
            
            
            #%Make AMV Time Plots
            xlm = [24,240]
            ylm = [-0.5,0.5]
            
            #xtk = np.arange(xlm[0],xlm[1]+20,20)
            fig,axs = plt.subplots(1,4,figsize=(12,1.5))
            for mode in range(4): 
                
                viz.plot_AMV(amvidx[region][mode]*vscale,ax=axs[mode])
                axs[mode].set_title("%s" % modelname[mode],fontsize=12)
                axs[mode].set_xlim(xlm)
                #axs[mode].set_xticks(xtk)
                axs[mode].set_xlabel('Year')
                axs[mode].set_ylim(ylm)
            axs[0].set_ylabel('AMV Index')
            #plt.suptitle("AMV Index | Forcing: %s; fscale: %ix" % (forcingname[funiform],fscale))
            #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# <<<<<<< Updated upstream
#             outname = outpathfig+'%s_AMVIndex_%s_allmodels_region%s.png' % (regions[region],expid)
# =======
            #plt.tight_layout()
            outname = outpathfig+'%s_AMVIndex_%s_allmodels_vscale%i.png' % (regions[region],expid,vscale)
# >>>>>>> Stashed changes
            plt.savefig(outname, bbox_inches="tight",dpi=200)



