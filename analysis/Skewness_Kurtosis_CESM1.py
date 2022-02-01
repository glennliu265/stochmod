#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Variance, Skewness and Kurtosis for CESM1 Pre-industrial Control 
Simulations

Created on Tue Feb  1 11:22:13 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220201/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/06_School/06_Fall2021/12860/materials_2020/Assignment5/HW5/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
from scipy import stats
import scm
import tbx
from tqdm import tqdm

import importlib
importlib.reload(viz)
proc.makedir(figpath)
#%% User Edits

# CESM1 Data Loading Options
bbox = [-100,20,-20,90] 
mconfigs = ("SLAB","FULL")
cdatpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"


# Toggles
debug = True

print("Simulation bounding Box is %s "% (str(bbox)))
#%% Load the Data

# Additionally Load the CESM1-Data (Monthly Anomalies)
sst_cesm = []
for mconfig in tqdm(mconfigs):
    fname   = "%sCESM1_%s_postprocessed_NAtl.nc" % (cdatpath,mconfig)
    ds      = xr.open_dataset(fname)
    dsreg   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    sst_cesm.append(dsreg.SST.values)

# Load lat/lon
lon = dsreg.lon.values
lat = dsreg.lat.values

# Get Dimensions
nlon,nlat,ntime = sst_cesm[0].shape 
#%% Do a Quick Visualizations
if debug:
    viz.qv_seasonal(lon,lat,sst_cesm[0],bbox=bbox)

#%% First, make a plot with monthly data


def calc_stat(invar,axis,statname):
    # Calculate x'
    meanvar = np.mean(invar,2)
    anomvar = invar - meanvar[...,None]
    if statname == "Variance":
        # Mean of squared differences from mean
        outvar = np.mean(anomvar**2,2)
    elif statname == "Skewness":
        # S = [ mean(T'3)/mean(T'2) ] ^ (3/2)
        outvar = np.mean(anomvar**3,2) / (np.mean(anomvar**2,2))**(3/2)
    elif statname == "Kurtosis":
        # Excess Kurtosis = [ mean(T'4)/mean(T'2) ] ^ (2) -3
        outvar = np.mean(anomvar**4,2) / (np.mean(anomvar**2,2))**(2) - 3
        
    return outvar
    

def sum_stat(invars,axis,manual=False):
    nmod = len(invars)
    nlon,nlat,_ = invars[0].shape
    # Preallocate [Lat x Lon x Model x Stat]
    sumvars     = np.zeros((nlon,nlat,nmod,3)) * np.nan
    for m in range(nmod):
        invar = invars[m]
        
            
        if manual:
            for s in tqdm(range(3)):
                # Manual Calc
                # -----------
                statnames   = ["Variance","Skewness","Kurtosis"]
                # Calculate Variance
                sumvars[:,:,m,s] = calc_stat(invar,axis,statnames[s]) 
                # Calculate Skewness
                sumvars[:,:,m,s] = calc_stat(invar,axis,statnames[s]) 
                # Calculate Kurtosis
                sumvars[:,:,m,s] = calc_stat(invar,axis,statnames[s]) 
        else:
            # Using Scipy
            # -----------
            # Calculate Variance
            sumvars[:,:,m,0] = np.var(invar,axis=axis) 
            # Calculate Skewness
            sumvars[:,:,m,1] = stats.skew(invar,axis=axis)
            # Calculate Kurtosis
            sumvars[:,:,m,2] = stats.kurtosis(invar,axis=axis)
    return sumvars
#%%

statnames   = ["Variance","Skewness","Kurtosis"]
sumvars     = np.zeros((nlon,nlat,2,3)) * np.nan # [Lat x Lon x Model x Stat]
sumvars_man = sumvars.copy() # manual calculation

for m in range(2):
    
    invar = sst_cesm[m]
    
    
    # Using Scipy
    # -----------
    # Calculate Variance
    sumvars[:,:,m,0] = np.var(invar,axis=2) 
    # Calculate Skewness
    sumvars[:,:,m,1] = stats.skew(invar,axis=2)
    # Calculate Kurtosis
    sumvars[:,:,m,2] = stats.kurtosis(invar,axis=2)
    
    for s in tqdm(range(3)):
        

        
        # Manual Calc
        # -----------
        # Calculate Variance
        sumvars_man[:,:,m,s] = calc_stat(invar,2,statnames[s]) 
        # Calculate Skewness
        sumvars_man[:,:,m,s] = calc_stat(invar,2,statnames[s]) 
        # Calculate Kurtosis
        sumvars_man[:,:,m,s] = calc_stat(invar,2,statnames[s]) 

#%% Above, but written as a function

sumvars     = sum_stat(sst_cesm,2,manual=False)  # [16s (slab), 33s (full)]
sumvars_man = sum_stat(sst_cesm,2,manual=True)   # [20s (slab), 39s (full)]



#%% Do Some Quick Comparisons

# See maximum Absolute Difference
for m in range(2):
    for s in tqdm(range(3)):
        test = np.abs(sumvars[:,:,m,s] - sumvars_man[:,:,m,s])
        print(np.nanmax(test))
        
        plt.figure()
        plt.title("Model %s, Stat %s"% (mconfigs[m],statnames[s]))
        plt.pcolormesh(test.T,vmin=-2,vmax=2,cmap="RdBu_r")
        plt.colorbar()
        plt.show()

# Ok it seems the methods are equivalent
#%%

"""
Quick Notes On Interpretation

Skewness -- "Asymmetry between the extremes"
S = 0 --> Symmetric distribution
S > 0 --> Positive Tail is longer
S < 0 --> Negative Tail is longer

Kurtosis -- "extremity of the tails"
-------------------------------------
Gaussian Kurtosis = 3, Excess Kurtosis = K-3
(+) --> Distribution tends to be around mean or at tails, less in mid-range
    >> Extreme Events occur more frequently relative to Gaussian Distr.
(-) --> >> Extreme Events occur less frequently 
** Note necessarily true for both tails if skewness is non-zero **

"""


#%% Make Some Plots of Kurtosis, Skewness
import cmocean as cmo

plotm          = 1 # 0=SLAB, 1=FULL
refm           = 0
bboxplot       = [-85,0,0,65]
slims          = (2,1,1.5)
slims_ratio    = (2,4,4)
statnames_lb   = ["Variance","Skewness","Excess Kurtosis (K-3)"]

cmaps          = ['cmo.thermal','cmo.balance','cmo.balance']

cf = False

plotlog  = True
invarplot = sumvars#[lon x lat x model x stat]

fig,axs = plt.subplots(2,3,constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()},
                       figsize=(12,6))


cmap = cmo.cm.balance.copy()
cmap.set_bad(color="w")

for r in range(2):
    
    if r == 0: # Plot Just for plotm
        
        for s in range(3):
            ax   = axs[r,s]
            blabel = [0,0,0,0]
            if s == 0:
                blabel = [1,0,0,0]
                ylab = "CESM1-%s" % mconfigs[plotm] 
                ax.text(-0.15, 0.55, ylab, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
                
            ax   = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                      fill_color='gray')
            slim = slims[s]
            
            if cf:
                cint = np.arange(-slim,slim+0.05,0.05)#np.linspace(-slim,slim,30)
                pcm = ax.contourf(lon,lat,invarplot[:,:,plotm,s].T,
                                    levels=cint,cmap=cmaps[s])
            else:
                if s == 0:
                    
                    pcm = ax.pcolormesh(lon,lat,invarplot[:,:,plotm,s].T,
                                        vmin=0,vmax=slim,cmap=cmaps[s])
                else:
                    pcm = ax.pcolormesh(lon,lat,invarplot[:,:,plotm,s].T,
                                        vmin=-slim,vmax=slim,cmap=cmaps[s])
                
            fig.colorbar(pcm,ax=ax,fraction=0.042)
            ax.set_title(statnames_lb[s])
            
    if r == 1:
        
        for s in range(3):
            
            ax   = axs[r,s]
            blabel = [0,0,0,1]
            if s == 0:
                blabel = [1,0,0,1]
                if plotlog:
                    ylab = "log(%s/%s)" % (mconfigs[plotm],mconfigs[refm])
                else:
                    ylab = "%s/%s" % (mconfigs[plotm],mconfigs[refm])

                ax.text(-0.15, 0.55, ylab, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            ax   = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                      fill_color='gray')
            slim = slims[s]
            
            plotvar = invarplot[...,plotm,s]/invarplot[...,refm,s]
            
            if plotlog:
                signs    = np.sign(plotvar)
                logratio = np.log(np.abs(plotvar))
                
                pcm = ax.pcolormesh(lon,lat,logratio.T,cmap=cmap,
                                    vmin=-slims_ratio[s],vmax=slims_ratio[s])
            else:
                pcm = ax.pcolormesh(lon,lat,plotvar.T,
                                    vmin=0,vmax=2,cmap=cmap)
            #if s == 2:
            fig.colorbar(pcm,ax=ax,fraction=0.045)
plt.savefig("%sMonthlyKurtosis_%s_Ratio.png"% (figpath,mconfigs[plotm]),dpi=200,bbox_inches='tight')


#%% Plot SLAB and FULL separately

plotm          = 0 # 0=SLAB, 1=FULL
refm           = 1
bboxplot       = [-85,0,0,65]
slims          = (2,1,1.5)
slims_ratio    = (2,4,4)
statnames_lb   = ["Variance","Skewness","Excess Kurtosis (K-3)"]

cmaps          = ['cmo.thermal','cmo.balance','cmo.balance']

cf = False

plotlog  = True
invarplot = sumvars#[lon x lat x model x stat]

fig,axs = plt.subplots(2,3,constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()},
                       figsize=(12,6))


cmap = cmo.cm.balance.copy()
cmap.set_bad(color="w")

for plotm in range(2):
        
    for s in range(3):
        ax   = axs[plotm,s]
        blabel = [0,0,0,0]
        if s == 0:
            blabel = [1,0,0,0]
            ylab = "CESM1-%s" % mconfigs[plotm] 
            ax.text(-0.15, 0.55, ylab, va='bottom', ha='center',rotation='vertical',
                    rotation_mode='anchor',transform=ax.transAxes)
            
        ax   = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                  fill_color='gray')
        slim = slims[s]
        
        if cf:
            cint = np.arange(-slim,slim+0.05,0.05)#np.linspace(-slim,slim,30)
            pcm = ax.contourf(lon,lat,invarplot[:,:,plotm,s].T,
                                levels=cint,cmap=cmaps[s])
        else:
            if s == 0:
                
                pcm = ax.pcolormesh(lon,lat,invarplot[:,:,plotm,s].T,
                                    vmin=0,vmax=slim,cmap=cmaps[s])
            else:
                pcm = ax.pcolormesh(lon,lat,invarplot[:,:,plotm,s].T,
                                    vmin=-slim,vmax=slim,cmap=cmaps[s])
            
        fig.colorbar(pcm,ax=ax,fraction=0.042)
        if plotm == 0:
            ax.set_title(statnames_lb[s])
            
    
plt.savefig("%sMonthlyKurtosis_SLAB_FULL.png"%figpath,dpi=200,bbox_inches='tight')

#%% Repeat this with Seasonal Data


snames = ("DJF"   ,"MAM"  ,"JJA"  ,"SON")
sids   = ([11,0,1],[2,3,4],[5,6,7],[8,9,10])
sdata = [] # [model][season]


for m in tqdm(range(2)):
    ntime = sst_cesm[m].shape[-1]
    sst_in = sst_cesm[m].reshape(nlon,nlat,int(ntime/12),12)
    s_sst  = []
    for s in range(4):
        # Select data for month and flatten
        sst_seas = sst_in[:,:,:,sids[s]].reshape(nlon,nlat,int(ntime/12)*3)
        s_sst.append(sst_seas)
    sdata.append(s_sst)



# Compute summary Statistics for each season
sumslab = sum_stat(sdata[0],2,manual=False) # [lon x lat x season x stat]
sumfull = sum_stat(sdata[1],2,manual=False)
sumdats = [sumslab,sumfull]

# %% Plot seasonal cycle in each parameter



for stat in range(3):
    
    fig,axs = plt.subplots(3,4,constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()},
                           figsize=(12,7.5))
    
    for r in range(3):
        
        for s in range(4):
            
            blabel = [0,0,0,0]
            if r == 2:
                blabel[3] = 1
            if s == 0:
                blabel[0] = 1
            
            ax = axs[r,s]
            
            if r <2:
                plotvar = sumdats[r][:,:,s,stat]
                mtitle = mconfigs[r]
                slim = slims[stat]
            else:
                inratio = sumfull[:,:,s,stat]/sumslab[:,:,s,stat]
                plotvar = np.sign(inratio)* np.log(np.abs(inratio))
                mtitle = "log(FULL/SLAB)"
                slim = slims_ratio[stat]
            
            ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel)
            if stat == 0 and r < 2:
                pcm = ax.pcolormesh(lon,lat,plotvar.T,
                                    cmap='cmo.thermal',
                                    vmin = 0,vmax=slim)
            else:
                pcm = ax.pcolormesh(lon,lat,plotvar.T,
                                    cmap='cmo.balance',
                                    vmin = -slim,vmax=slim)
            if r == 0:
                ax.set_title("%s" % (snames[s]))
            if s == 0:
                ax.text(-0.20, 0.55, mtitle, va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
            
        fig.colorbar(pcm,ax=axs[r,:].flatten(),fraction=0.045)
    plt.savefig("%sSeasonality_Plot_%s.png" % (figpath,statnames[stat]),dpi=200,
                                               bbox_inches='tight')
            
            
        
        
    
    
    
    





