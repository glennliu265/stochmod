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

import time
#%%
# User Input Variables

# Set Point
lonf     = -30
latf     = 50

# Experiment Settings
entrain  = 1     # 0 = no entrain; 1 = entrain
hvarmode = 2     # 0 = fixed mld ; 1 = max mld; 2 = clim mld 
funiform = 6     # 0 = nonuniform; 1 = uniform; 2 = NAO-like; 3= NAO-monthly
nyrs     = 1000  # Number of years the experiment was run
runid    = "001" # Run ID for white noise sequence
fscale   = 10

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

outpathdat = projpath + '01_Data/model_output/proc/'
outpathfig = projpath + '02_Figures/20200823/'

# Text for plotting
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$",
               "$NAO_{DJFM}  &  NHFLX_{Mon}$",
               "$(NAO  &  NHFLX)_{Mon}$",
               "$EAP_{DJFM}$",
               "$(NAO+EAP)_{DJFM}$")

if Li_etal == 1:
    # Set Bounding Boxes and Regions
    bbox_EA = [-75,10,40,65]
    bbox_TA = [-75,20 ,5,20]
    bbox_NA = [-80,20 ,0,65]#[-75,20,0,90]
    
    
    regions = ("EA","TA","NA")
    bboxes  = (bbox_EA,bbox_TA,bbox_NA)
else:
    bbox_SP = [-60,-15,40,65]
    bbox_ST = [-80,-10,20,40]
    bbox_TR = [-75,-15,0,20]
    bbox_NA = [-80,0 ,0,65]

    regions = ("SPG","STG","TRO","NAT")
    bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)


rcol = ('b','r',[0,1,0],'k')

#%% Load Data
start = time.time()

# Read in damping data for the coordinates
datpath = projpath + '01_Data/model_input/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(datpath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])

# Load MLD Data
mld      = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD

# Read in Stochmod SST Data
datpath = projpath + '01_Data/model_output/'
expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Load CESM1LE Verification Data
cesmsst = np.load(projpath + "01_Data/SST_Timeseries_RegionAvg.npy",allow_pickle=True).item()
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()

print("Data loaded in %.2fs"% (time.time()-start))

#%% Get Regional Data

nregion = len(regions)
sstregion = {}
for r in range(nregion):
    bbox = bboxes[r]
    
    sstr = {}
    for model in range(4):
        tsmodel = sst[model]
        sstr[model],_,_=proc.sel_region(tsmodel,lonr,latr,bbox)
        
    
    sstregion[r] = sstr

#%% Calculate autocorrelation and SST averaged time series for a given region

kmonths = {}
autocorr_region = {}
sstavg_region   = {}
for r in range(nregion):
    bbox = bboxes[r]
    
    autocorr = {}
    sstavg = {}
    for model in range(4):
        
        # Get sst and havg
        tsmodel = sstregion[r][model]
        havg,_,_= proc.sel_region(mld,lon,lat,bbox)
        
        # Find kmonth
        havg = np.nanmean(havg,(0,1))
        kmonth     = havg.argmax()+1
        kmonths[r] = kmonth
        
        
        # Take regional average 
        tsmodel = np.nanmean(tsmodel,(0,1))
        sstavg[model] = np.copy(tsmodel)
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Compute autocorrelation and save data for region
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
        
    
    autocorr_region[r] = autocorr.copy()
    sstavg_region[r] = sstavg.copy()
    
# Save Regional Autocorrelation Data
np.savez("%sSST_Region_Autocorrelation_%s.npz"%(outpathdat,expid),autocorr_region=autocorr_region,kmonths=kmonths)

# Save Regional Average SST 
np.save("%sSST_RegionAvg_%s.npy"%(outpathdat,expid),sstavg_region)


#%% Calculate AMV Index and Spatial Pattern for each region

# Calculate different AMVs for each region
amvbboxes = bboxes
amvidx_region = {}
amvpat_region = {}
for region in range(nregion):
    
    #% Calculate AMV Index
    amvtime = time.time()
    amvidx = {}
    amvpat = {}
    
    for model in range(4):
        amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,amvbboxes[region])
    print("Calculated AMV variables for region %s in %.2f" % (regions[region],time.time()-amvtime))
    
    amvidx_region[region] = amvidx
    amvpat_region[region] = amvpat
    
# Save Regional Autocorrelation Data
np.savez("%sAMV_Region_%s.npz"%(outpathdat,expid),amvidx_region=amvidx_region,amvpat_region=amvpat_region)



#%% Plot Individual Autocorrelation Plots

xlim = [0,36]
xtks = np.arange(0,38,2)

for model in range(4):
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use("seaborn")
    plt.style.use("seaborn-bright")
    
    for r in range(nregion):
        label =  " %s basemonth %i" % (regions[r],kmonths[r])
        ax.plot(lags,autocorr_region[r][model],color=rcol[r],label=label)
        
    ax.set_title("SST Autocorrelation for model %s \n Forcing %s" % (modelname[model],forcingname[funiform]))
    
    #plt.xticks(xtk)
    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)
    plt.xticks(xtks)

    plt.savefig(outpathfig+"Region_SST_Autocorrelation_run%s_model%s_funiform%i_fscale%i.png"%(runid,modelname[model],funiform,fscale),dpi=200)


#%% Make the Autoorrelation plots (row)

xlim = [0,36]
xtks = np.arange(0,39,3)
ylim = [-0.25,1]
ytks = np.arange(-.25,1.25,0.25)

regioncolor= rcol

fig,axs = plt.subplots(1,4,figsize=(12,2))
plt.style.use("seaborn")
plt.style.use("seaborn-bright")
for model in range(4):
    ax = axs[model]
    
    for r in range(nregion):
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
plt.savefig(outpathfig+"Region_SST_Autocorrelation_run%s_modelALL_funiform%i_fscale%i.png"%(runid,funiform,fscale),dpi=200)




#%% Make a referece CESM plot of region-averaged SST time series

units = 'degC'

fig,axs = plt.subplots(4,1,figsize=(6,6)) 
#plt.style.use('ggplot')
plt.style.use('seaborn')
for r in range(nregion):
    
    sstin = cesmsst[r]
    ax = axs[r]
    titlestr = "%s %s" % (regions[r],viz.quickstatslabel(sstin))
    ax = viz.ensemble_plot(sstin,0,ax=ax,color=rcol[r],ysymmetric=1,ialpha=0.05)
    ax.set_title(titlestr)    
plt.tight_layout()
plt.savefig("%sCESM1LE_RegionalSSTStats.png"%(outpathfig),dpi=200)

#%% Make time seris plots to compare to cesm

xrange = np.arange(0,12000,1)
units = "degC"
for r in range(nregion):
    
    fig,axs = plt.subplots(4,1,figsize=(6,6)) 
    #plt.style.use('ggplot')
    plt.style.use('seaborn')
    
    for m in range(4):
        
        sstin = sstavg_region[r][m][xrange]
        
        ax = axs[m]
        figtitle = "%s %s" % (regions[r],modelname[m])

        viz.plot_annavg(sstin,units,figtitle,ax=ax)
    
    plt.tight_layout()
    plt.savefig("%sSST_%s_%s.png"%(outpathfig,expid,regions[r]),dpi=200)


#%%


