#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make visualizations and comparisons with CESM1LE Data

Created on Sun Aug 23 18:51:37 2020

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
from dask.distributed import Client,progress
import dask


#%%#Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpathfig = projpath + '02_Figures/20200823/'

bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]

regions     = ("SPG","STG","TRO","NAT")
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
regioncolor = ('b','r',[0,1,0],'k')

nregion = len(regions)
lags = np.arange(0,61,1)


#%% Check STDEV of Forcing
cesmforce = xr.open_dataset(datpath+'NHFLX_ForcingStats_Converted_NotMonthly_hvarmode2_ensavg.nc')

fstd = cesmforce.fstd.values
clon = cesmforce.lon.values
clat = cesmforce.lat.values

clon1,fstd=proc.lon360to180(clon,fstd.transpose(1,0))

# Get values  for different regions
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]

regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)

stdboxes = []
stdval = []
for r in range(4):
    
    # Select data from region
    bbox = bboxes[r]
    datr,_,_ =proc.sel_region(fstd,clon1,clat,bbox)
    
    # Make Data 1D and remove NaN points
    datr = datr.flatten()
    datr,_,_ = proc.find_nan(datr,0)
    
    # Append data
    stdboxes.append(datr)
    stdval.append(np.mean(datr))

# Create Plot
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn")
bp = ax.boxplot(stdboxes,0,'',labels=regions) # Note Outlier POints are not shown
ax.set_xlabel("Region")
ax.set_ylabel("Standard Deviation")
ax.set_title("$\sigma_{Forcing}$ CESM1LE 42-member Average")
ax.set_ylim([0,0.8])
plt.savefig("%sForcing_Stdev_Regional.png"%(outpathfig),dpi=200)

#%% Make same autocorrelation plot but using data from CESM

xlim = [0,36]
xtks = np.arange(0,39,3)
ylim = [-0.25,1]
ytks = np.arange(-.25,1.25,0.25)

# Load CESM1LE Verification Data
cesmsst = np.load(projpath + "01_Data/SST_Timeseries_RegionAvg.npy",allow_pickle=True).item()
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()

# Set colors
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]

fig,ax = plt.subplots(1,1,figsize = (3,2))

for r in range(nregion): 
    
    rname = regions[r]
    
    for e in range(42):
   
        ax.plot(lags,cesmauto[r][e,:],color=rcolmem[r],alpha=0.5)
    
ln1 = ax.plot(lags,np.nanmean(cesmauto[0],0),color=regioncolor[0],label=regions[0])
ln2 = ax.plot(lags,np.nanmean(cesmauto[1],0),color=regioncolor[1],label=regions[1])
ln3 = ax.plot(lags,np.nanmean(cesmauto[2],0),color=regioncolor[2],label=regions[2])
ln4 =  ax.plot(lags,np.nanmean(cesmauto[3],0),color=regioncolor[3],label=regions[3])


plt.grid(True)
ax.set_xlim(xlim)
ax.set_xticks(xtks)
ax.set_ylim(ylim)
ax.set_yticks(ytks)
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,prop={'size':8})
ax.set_title("CESM Autocorrelation")

plt.savefig(outpathfig+"Region_SST_Autocorrelation_CESM.png",dpi=200)

#%% Make same autocorrelation plot but using data from CESM (Hi-Res)

xlim = [0,36]
xtks = np.arange(0,39,3)
ylim = [-0.25,1]
ytks = np.arange(-.25,1.25,0.25)

# Load CESM1LE Verification Data
cesmsst = np.load(projpath + "01_Data/SST_Timeseries_RegionAvg.npy",allow_pickle=True).item()
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()

# Set colors
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]

fig,ax = plt.subplots(1,1,figsize = (6,4))

for r in range(nregion): 
    
    rname = regions[r]
    
    for e in range(42):
   
        ax.plot(lags,cesmauto[r][e,:],color=rcolmem[r],alpha=0.25)
    
ln1 = ax.plot(lags,np.nanmean(cesmauto[0],0),color=regioncolor[0],label=regions[0])
ln2 = ax.plot(lags,np.nanmean(cesmauto[1],0),color=regioncolor[1],label=regions[1])
ln3 = ax.plot(lags,np.nanmean(cesmauto[2],0),color=regioncolor[2],label=regions[2])
ln4 =  ax.plot(lags,np.nanmean(cesmauto[3],0),color=regioncolor[3],label=regions[3])


plt.grid(True)
ax.set_xlim(xlim)
ax.set_xticks(xtks)
ax.set_ylim(ylim)
ax.set_yticks(ytks)
lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,prop={'size':12})
ax.set_title("CESM1LE Autocorrelation (1920-2005")

plt.savefig(outpathfig+"Region_SST_Autocorrelation_CESM_HiRes.png",dpi=200)

#%% Autocorrelation Plot, but just for the NAT

xlim = [0,36]
xtks = np.arange(0,39,3)
ylim = [-0.4,1]
ytks = np.arange(-.25,1.25,0.25)

# Load CESM1LE Verification Data
cesmsst = np.load(projpath + "01_Data/SST_Timeseries_RegionAvg.npy",allow_pickle=True).item()
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()

# Set colors
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]

fig,ax = plt.subplots(1,1,figsize = (6,4))

r = 3
    
rname = regions[r]

# for e in range(42):
   
#     ax.plot(lags,cesmauto[r][e,:],color=rcolmem[r],alpha=0.25)
    
# ln4 =  ax.plot(lags,np.nanmean(cesmauto[3],0),color=regioncolor[3],label=regions[3])

ax = viz.ensemble_plot(cesmauto[3],0,ax=ax,color='k',ysymmetric=0,ialpha=0.10)

plt.grid(True)
ax.set_xlim(xlim)
ax.set_xticks(xtks)
ax.set_ylim(ylim)
ax.set_yticks(ytks)
# lns = ln3
# labs = [l.get_label() for l in lns]
# ax.legend(lns,labs,prop={'size':12})
ax.set_title("CESM1LE Autocorrelation, NAT (1920-2005")

plt.savefig(outpathfig+"NAT_SST_Autocorrelation_CESM_HiRes.png",dpi=200)

#%% Make a referece CESM plot of region-averaged SST time series

units = 'degC'


fig,axs = plt.subplots(4,1,figsize=(6,6)) 
#plt.style.use('ggplot')
plt.style.use('seaborn')
for r in range(nregion):
    
    sstin = cesmsst[r]
    ax = axs[r]
    titlestr = "%s %s" % (regions[r],viz.quickstatslabel(sstin))
    ax = viz.ensemble_plot(sstin,0,ax=ax,color=regioncolor[r],ysymmetric=1,ialpha=0.05)
    ax.set_title(titlestr)    
plt.tight_layout()
plt.savefig("%sCESM1LE_RegionalSSTStats.png"%(outpathfig),dpi=200)