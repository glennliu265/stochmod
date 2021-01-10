#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script to prepare NAO and EAP forcing patterns

Currently works with the output for calc_NAO_eof

[Manual_EOF_Calc_NAO.npz]


Created on Thu Aug 20 12:07:35 2020

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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from cartopy import config
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import LongitudeFormatter,LatitudeFormatter
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% User Edits

# Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20200823/'

# File Name
datfile = "NAO_Monthly_Regression_PC123.npz"

# Mapping box
bbox = [260,20,0,90]

# Ensemble members where EAP is on PC3
use3 = [2,5,11,29]


# # Mapping box for forcing
# [-100,40,-20,90]
# Load latlon
ll = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")
lon = np.squeeze(ll['LON'])
lat = np.squeeze(ll['LAT'])


#%% Load Necessary Data

# Load Lat Lon
mat1 = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")
lon = np.squeeze(mat1["LON"])
lat = np.squeeze(mat1["LAT"])

# Load the data
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO.npz")
eofall    = npzdata['eofall']
pcall     = npzdata['pcall']
varexpall = npzdata['varexpall'] 

# Swap EOF 2 for EOF3 where indicated
for e in use3:
    ie = e-1
    eofall[ie,:,:,1] = eofall[ie,:,:,2]
    pcall[ie,:,1]    = pcall[ie,:,2]
    varexpall[ie,1]  = varexpall[ie,2]
    print("Swaped EOF2 for EOF3 for ens %i" % e)

#%% Plot results to check


def plot_regression(varin,lon,lat,cint=[0],ax=None,bbox=[-90,40,0,80],latlonlab=True,cbar=True,cbloc='horizontal'):
    """
    # Create regression plots for NAO and related EOFs (SLP)
    
    Inputs
        1) N (int) - PC index
        2) e (int) - ensemble index
        3) m (int) - month index
        4) varin (array: ens x mon x lat x lon x pc) - Regression Pattern
        5) lon (array: lon) - longitudes
        6) lat (array: lat) - latitudes
        7) varexp (array: ens x mon x pc) - Variance explained
    
        -OPTIONAL-
        8) cint (array: levels) - contour intervals
        9) ax (geoaxes object) - axis to plot on
        10) bbox (array: lonW,lonE,latS,latN) - plot extent
        11) latlonlab (bool) - True to include lat/lon labels
        12) cbar (bool) - True to include colorbar
    
    
    """

    # Get current axis if none is assigned
    if ax is None:
        ax = plt.gca()
    

    # Set colormap
    cmap = cmocean.cm.balance
    cmap.set_under(color='yellow')
    
    #Set Plotting boundaries
    ax.set_extent(bbox)
    
    # Add cyclic point to remove blank meridian
    var1,lon1 = add_cyclic_point(varin,lon) 
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,linewidth=0.75,linestyle=":")
    
    # Add contours
    if len(cint) == 1:
        
        # Plot without specificing range
        cs = ax.contourf(lon1,lat,var1,cmap=cmap,extend="both",transform=ccrs.PlateCarree())
        
    else:
        
        
        cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,extend="both",transform=ccrs.PlateCarree())
        
        # Negative contours
        cln = ax.contour(lon1,lat,var1,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())
        
        # Positive Contours
        clp = ax.contour(lon1,lat,var1,
                    cint[cint>=0],
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())   
        # Add Label
        plt.clabel(cln,fmt='%.1f',fontsize=8)
        plt.clabel(clp,fmt='%.1f',fontsize=8)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle=':')
    if latlonlab is True:
        gl.top_labels = gl.right_labels = False
        gl.xformatter = LongitudeFormatter(degree_symbol='')
        gl.yformatter = LatitudeFormatter(degree_symbol='')
    else:
        gl.top_labels = gl.right_labels = gl.left_labels = gl_bottom_labels=False
    
    # Create colorbar
    if cbar == True:
        plt.colorbar(cs,ax=ax,fraction=0.046,pad=0.10,orientation=cbloc)
        
    #ax.set_title("PC%i Regression ENS %s Mon%i \n Variance Explained: %03d" %(N+1,estring,m+1,varexp*100)+r"%",fontsize=16)
    return ax

oleof = eofall.copy()
olexp = varexpall.copy()
patname="SLP"

meanname="DJFM-CORR"
eapbox = [-30,15,35,60]
ebx    = [320,15,35,60]
#cint = np.arange(-2.4,2.6,0.2)
cint = np.arange(-5,5.5,0.5)
N = 1

# Initialize figure
fig,axs = plt.subplots(7,6,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12))

rowid = 0
colid = 0

for e in range(42):
    
    # Restrict to Ensemble,PC,month
    varplot = oleof[e,...,N]
    
    varexp1  = olexp[e,N]
   
    # Decide on Row or Column
    if e%6 == 0:
        colid = 0
    else:
        colid += 1
    rowid = int(np.fix(e/6))
    
    # Create Plot
    ax = axs[rowid,colid]
    ax = plot_regression(varplot,lon,lat,ax=ax,cint=cint,cbar=False)
    ax = viz.plot_box(eapbox,ax=ax)
    
    # Sum values to check for sign (transpose to lon x lat for format)
    #rsum = proc.sel_region(varplot.T,lon,lat,ebx,reg_sum=1)
    
    #asum,lor,lar = proc.sel_region(varplot.T,lon,lat,ebx,reg_sum=0)
    #chk  = rsum > 0 # set to 1 if it needs to be flipped
    
    #print("row %i col %i e%i"%(rowid,colid,e+1)) d

    ax.set_title("Ens%02d %.2f"%(e+1,varexp1*100) + r"%")
    #ax.set_title("Ens%02d %i"%(e+1,chk))
    print("\rPlotted ens %02d of 42" % (e+1),end='/r',flush=True)
plt.tight_layout()
plt.savefig("%s%sPattern_PC%i_%sAvg_OLDDJFM-CORR.png" % (outpath,patname,N+1,meanname),dpi=200)
print("Finished PC %i" % (N+1))

#%%Save Data

np.savez(datpath+"Manual_EOF_Calc_NAO_corr.npz",eofall=eofall,pcall=pcall,varexpall=varexpall)

    
    

