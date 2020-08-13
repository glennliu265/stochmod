#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:28:11 2020

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time
import hvplot.xarray

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

from time import time
#%%
# User Input Variables

# Set Point
lonf     = -30
latf     = 50

# Experiment Settings
entrain  = 1     # 0 = no entrain; 1 = entrain
hvarmode = 2     # 0 = fixed mld ; 1 = max mld; 2 = clim mld 
funiform = 0     # 0 = nonuniform; 1 = uniform; 2 = NAO-like; 3= NAO-monthly
nyrs     = 1000  # Number of years the experiment was run
runid    = "001" # Run ID for white noise sequence
fscale   = 1

# Autocorrelation Parameters
kmon       = 2                 # Lag 0 base month
lags       = np.arange(0,61,1) # Number of lags to include
detrendopt = 0                 # Detrend before autocorrelation
Li_etal    = 0        


# Bounding Box for visualization
lonW = -80
lonE = 20
latS = -20
latN = 90
bbox = [lonW,lonE,latS,latN]

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200813/'

# Text for plotting
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$")


if Li_etal == 1:
    # Set Bounding Boxes and Regions
    bbox_EA = [-75,10,40,65]
    bbox_TA = [-75,20 ,5,20]
    bbox_NA = [-80,20 ,0,65]#[-75,20,0,90]
    
    
    regions = ("EA","TA","NA")
    bboxes  = (bbox_EA,bbox_TA,bbox_NA)
else:

    bbox_SP = [-60,20,50,70]
    bbox_ST = [-80,20,10,45]
    bbox_NA = [-80,20 ,0,65]#[-75,20,0,90]
    regions = ("SPG","STG","NAT")
    bboxes = (bbox_SP,bbox_ST,bbox_NA)



#%% Load Data
start = time.time()

# Read in damping data for the coordinates
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])

# Load MLD Data
mld      = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD

# Read in Stochmod SST Data
sst = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain0_run%s.npy"%(nyrs,funiform,runid),allow_pickle=True).item()
sst[3] = np.load(datpath+"stoch_output_%iyr_funiform%i_entrain1_run%s.npy"%(nyrs,funiform,runid))
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")


#%% Get Regional Data

sstregion = {}
for r in range(3):
    bbox = bboxes[r]
    
    sstr = {}
    for model in range(4):
        tsmodel = sst[model]
        sstr[model],_,_=proc.sel_region(tsmodel,lonr,latr,bbox)
        
    
    sstregion[r] = sstr


#%% Calculate autocorrelation for a given region



kmonths = {}
autocorr_region = {}
for r in range(3):
    bbox = bboxes[r]
    
    autocorr = {}
    for model in range(4):
        
        # Get sst and havg
        tsmodel = sstregion[r][model]
        havg,_,_= proc.sel_region(mld,lon,lat,bbox)
        
        # Find kmonth
        havg = np.nanmean(havg,(0,1))
        kmonth = havg.argmax()+1
        kmonths[r] = kmonth
        
        
        # Take regional average
        tsmodel = np.nanmean(tsmodel,(0,1))
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        
        
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Plot
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
    
    autocorr_region[r] = autocorr.copy()
    
#%% Make the plots (individual)

xlim = [0,36]
xtks = np.arange(0,38,2)

for model in range(4):
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use("seaborn")
    plt.style.use("seaborn-bright")
    
    for r in range(3):
        label =  " %s basemonth %i" % (regions[r],kmonths[r])
        ax.plot(lags,autocorr_region[r][model],label=label)
        
    ax.set_title("SST Autocorrelation for model %s \n Forcing %s" % (modelname[model],forcingname[funiform]))
    
    #plt.xticks(xtk)
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.xticks(xtks)

    plt.savefig(outpath+"Region_SST_Autocorrelation_run%s_model%s_funiform%i_fscale%i.png"%(runid,modelname[model],funiform,fscale),dpi=200)


#%% Make the plots (row)
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
plt.savefig(outpath+"Region_SST_Autocorrelation_run%s_modelALL_funiform%i_fscale%i.png"%(runid,funiform,fscale),dpi=200)


#%% Make same autocorrelation plot but using data from CESM
cesmauto = np.load(datpath+"Autocorrelation_Region.npy",allow_pickle=True).item()
cesmauto

# Set colors
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           [.75,.75,.75]]

fig,ax = plt.subplots(1,1,figsize = (3,2))
                      
for r in range(3): 
    
    rname = regions[r]
    
    for e in range(42):
   
        ax.plot(lags,cesmauto[r][e,:],color=rcolmem[r],alpha=0.5)
    
ln1 = ax.plot(lags,np.nanmean(cesmauto[0],0),color=regioncolor[0],label=regions[0])
ln2 = ax.plot(lags,np.nanmean(cesmauto[1],0),color=regioncolor[1],label=regions[1])
ln3 = ax.plot(lags,np.nanmean(cesmauto[2],0),color=regioncolor[2],label=regions[2])

plt.grid(True)
ax.set_xlim(xlim)
ax.set_xticks(xtks)
ax.set_ylim(ylim)
ax.set_yticks(ytks)
lns = ln1 + ln2 + ln3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,prop={'size':8})
ax.set_title("CESM Autocorrelation")
plt.savefig(outpath+"Region_SST_Autocorrelation_CESM.png",dpi=200)








