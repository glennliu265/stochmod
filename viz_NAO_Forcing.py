#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize NAO Forcing
Created on Wed Jul 29 11:19:02 2020

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

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc

#%% Functions

def init_map(bbox,ax=None):
    """
    Quickly initialize a map for plotting
    """
    # Create Figure/axes
    #fig = plt.gcf() 
    
    #ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    if ax is None:
        ax = plt.gca()
    #ax = plt.axes(projection=ccrs.PlateCarree())
        
    
    ax.set_extent(bbox)
    
    # Add Filled Coastline
    ax.add_feature(cfeature.COASTLINE,facecolor='k')
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='gray',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    return ax
    
def div_cint(var):
    """
    Automatically Set diverging colorbar based on min/max value
    """
    valmax = np.around(np.nanmax(var))
    valmin = np.around(np.nanmin(var))
    
    if np.abs(valmax) > np.abs (valmin):
        cint = np.linspace(-1*valmax,valmax,20)
    elif np.abs(valmax) < np.abs (valmin):
        cint = np.linspace(valmin,-1*valmin,20)
    
    return cint


#%% User Inputs
# Set data paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200730/'

# Path to SST data from obsv
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

#%%  Load/Process Data

# Load NAO Forcing DJFM (Monthyl)
naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
NAO1 = np.nanmean(naoforcing,axis=0) # Take PC1, Ens Mean and Transpose
NAO1 = np.transpose(NAO1,(2,1,0))

# Convert Longitude from degrees East
lon360 =  np.load(datpath+"CESM_lon360.npy")
lon180,NAO1 = proc.lon360to180(lon360,NAO1)


# Load NAO Forcing Seasonally Varying
naomon = np.load(datpath+"NAO_Monthly_Regression_PC.npz")
naomon = naomon['eofall']
naomon = np.nanmean(naomon,0) # Take ens mean
naomon = np.transpose(naomon,(2,1,0))
_,naomon = proc.lon360to180(lon360,naomon)

# Create quick nan mask and apply to NAO forcing
nanmask = np.sum(naomon,2)
nanmask = np.where(~np.isnan(nanmask),np.ones(nanmask.shape),np.nan)
NAO1 = NAO1 * nanmask[:,:,None]


# Load lat/lon arrays for plotting, from damping dataset
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])





#%% Create Seasonal Plots of winter DJFM Forcing

# Set plotting options
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
bbox= [-100,20,-20,90]

# Set colormap limits
value = [-40,40]
cints = {key: value for key in ['winter','spring','summer','fall']}
#cints = {'winter':[-60,60],'spring':[-40,40],'summer':[-10,10],'fall':[-10,10]}
#cints = {'winter':[-60,60],'spring':[-60,60],'summer':[-60,60],'fall':[-60,60]}

# Set strings
outname = "%sNHFLX-NAO_DJFM_regression" % outpath
figtitle = "NAO-NHFLX Regression (DJFM, $W m^{-2} \sigma_{NAO} ^{-1}$)"


def seasonalplots(var,cints,bbox,cmap,figtitle,outname):
    """
    Makes seasonal plots of var [lon x lat x month]
    
    Dependencies: init_map
    """

    # Set up seasons and indexing
    sids = {'winter':[11,0,1],'spring':[2,3,4],'summer':[5,6,7],'fall':[8,9,10]}
    seasons = ['winter','spring','summer','fall']
    
    
    for season in seasons:
        
        seasonid = sids[season]  # Set season name
        cintv = cints[season]    # Get colorbar max/min
        vmax = cintv[1]
        vmin = cintv[0]
        
        # Initialize Figure
        fig1,axs = plt.subplots(1,3,figsize=(12,8),subplot_kw={'projection': ccrs.PlateCarree()})
        #fig1.subplots_adjust(top=0.95)
        
        
        # Loop by month
        i = 0
        for s in seasonid:
            ax = axs[i]
            axs[0] = init_map(bbox,ax=ax)
    
            pcm = ax.pcolormesh(lon,lat,var[:,:,s].T,cmap=cmap,vmax=vmax,vmin=vmin)
            ax.set_title("Month %i" % (s+1),fontsize=12)
            fig1.colorbar(pcm,ax=ax,orientation="horizontal",fraction=0.046,pad=0.04)
            i+= 1
        
        plt.suptitle(figtitle,Y=0.60)
        #plt.tight_layout()
        plt.savefig("%s_%s.png"%(outname,season),dpi=200,bbox_inches='tight')
        
    
seasonalplots(NAO1,cints,bbox,cmap,figtitle,outname)


#%% Create the same plots,but for the other 

cmap = cmocean.cm.balance
bbox= [-100,20,-20,90]
#cints = {'winter':[-60,60],'spring':[-60,60],'summer':[-60,60],'fall':[-60,60]}
outname = "%sNHFLX-NAO_Monthly_regression" % outpath
figtitle = "NAO-NHFLX Regression (Monthly Varying, $W/m^{2}/\sigma_{NAO}$)"
seasonalplots(naomon,cints,bbox,cmap,figtitle,outname)