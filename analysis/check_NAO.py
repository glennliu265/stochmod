#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:22:46 2020

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
from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LongitudeFormatter,LatitudeFormatter
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% User Edits

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200823/'
naotype = 0
# File Name
datfile = "NAO_Monthly_Regression_PC123_naotype%i.npz" % naotype

# Mapping box
bbox = [260,20,0,90]
# # Mapping box for forcing
# [-100,40,-20,90]

# Load latlon
ll = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")
lon = np.squeeze(ll['LON'])
lat = np.squeeze(ll['LAT'])
#%%% Load the data

# Load in data (naodat.files to print keys)
naodat = np.load(datpath+datfile,allow_pickle=True)

varexp  = naodat['varexpall'] # variance explained [ens x mon x pc]
pcall   = naodat['pcall'] # PC [ens x mon x yr x pc]
eof     = naodat['psleofall'] # SLP EOFs [ens x mon x lat x lon x pc]
pattern = naodat['flxpattern'] # NHFLX Pattern  [ens x mon x lat x lon x pc]

#%% Plotting functions

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


#%% Create plots for all 42 ensemble members, separately for each month

varin = np.copy(eof)
N = 1   

eapbox = []
# Make some ensemble plots
for m in range(12):
    
    fig,axs = plt.subplots(7,6,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12))
    
    rowid = 0
    colid = 0
    # Loop for each ensemble member
    for e in range(42):
        
        # Restrict to Ensemble,PC,month
        if e == "avg":
            # Assumed [ens, mon, lat, lon, pc]
            varplot = np.nanmean(varin,axis=0)[m,...,N]
            varexp1 = np.nanmean(varexp,axis=0)[m,N]
        else:
            varplot = varin[e,m,:,:,N]
            varexp1 = varexp[e,m,N]
        
        # Decide on Row or Column
        if e%6 == 0:
            colid = 0
        else:
            colid += 1
        rowid = int(np.fix(e/6))
        
        # Create Plot
        ax = axs[rowid,colid]
        ax = plot_regression(varplot,lon,lat,ax=ax)
        #ax = viz.plot_box(eapbox,ax=ax)
        #print("row %i col %i e%i"%(rowid,colid,e+1))

        ax.set_title("Ens%02d %.2f"%(e+1,varexp1*100) + r"%")
    plt.tight_layout()
    plt.savefig("%sSLPPattern_PC%i_mon%02d.png" % (outpath,N+1,m+1),dpi=200)
    print("Finished PC %i mon %i" % (N+1,m+1))

#%% Create an ensemble and lag average of each mode (for EAP identificaton)

eapbox = [-30,15,35,60]
ebx    = [320,15,35,60]
cint = np.arange(-2.4,2.6,0.2)
if naotype > 1:
    # Take annual mean (OR DJFM, as currently indicated)
    varin = np.nanmean(np.copy(eof[:,[11,0,1],:,:,:]),1) # [ens x lat x lon x pc]
else:
    varin = np.copy(eof)
patname = "SLP"
meanname = 'DJFM'

for N in range(3):
    
    # Initialize figure
    fig,axs = plt.subplots(7,6,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12))
    
    
    
    rowid = 0
    colid = 0
    
    for e in range(42):
        
        # Restrict to Ensemble,PC,month
        varplot = varin[e,:,:,N]
        
        varexp1  = np.nanmean(varexp[e,...,N])
       
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
    plt.tight_layout()
    plt.savefig("%s%sPattern_PC%i_%sAvg_naotype%i.png" % (outpath,patname,N+1,meanname,naotype),dpi=200)
    print("Finished PC %i" % (N+1))
        
        

#plt.figure(1,1)
# Print colorbar sample
fig,ax = plt.subplots(1,1,figsize=(12,12))
cf = ax.contourf(lon,lat,varplot,cint,cmap=cmocean.cm.balance)
plt.colorbar(cf,ax=ax)
plt.savefig(outpath+"colorbarscrap.png",dpi=200)

#%% Make same plots (DJFM, all ens)m but for the older data [made by calc_NAO_eof.py]

load2 = np.load(datpath+"Manual_EOF_Calc_NAO.npz")
oleof = load2['eofall'] # ens x lat x lon x pc
olexp = load2['varexpall'] # ens x pc
patname="SLP"

meanname="DJFM-OLD"
eapbox = [-30,15,35,60]
ebx    = [320,15,35,60]
cint = np.arange(-2.4,2.6,0.2)
for N in range(3):
    
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
    plt.tight_layout()
    plt.savefig("%s%sPattern_PC%i_%sAvg_OLDDJFM.png" % (outpath,patname,N+1,meanname),dpi=200)
    print("Finished PC %i" % (N+1))


#%% Take the ensemble average, and plot one for each month

# Take the ensemble average
varin = np.nanmean(np.copy(eof),0) #[mon x lat x lon x pc]
expavg = np.nanmean(varexp,0) # [mon x pc]


# Make seasonal plots, and EOF 2 and EOF 3...
# ---- Winter plots ----


mons = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
seasonname = ('DJF','MAM','JJA','OND')
inpc=[1,2]
cint = np.arange(-0.8,0.9,0.1)



for s in range(4):
    fig,axs = plt.subplots(2,3,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,8))
    for iN in range(2):
        N = inpc[iN]
        
        for im in range(3):
            m = mons[s][im]
            ax = axs[iN,im]
            ax = plot_regression(varin[m,:,:,N],lon,lat,cint=cint,ax=ax)
            ax.set_title("Mon%i VarExpl %.2f" % (m+1,expavg[m,N]*100)+"%",fontsize=10)
    #plt.tight_layout(rect=[0,0,0.75,1])
    plt.savefig("%sSLPPattern_Ensavg_EOF23_%s.png"%(outpath,seasonname[s]),dpi=200)


#%% Plot Forcing and SLP contours together

m = 0
N = 0
cint1 = np.arange(-0.8,0.9,0.1)
mons = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
inpc=[0,1,2]
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cmap.set_under(color='yellow')
cint2 = np.arange(-6,7,1)


for iN in range(3):
    fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,8))
    N = inpc[iN]
    for s in range(4):
        mns = mons[s]
        for im in range(3):
            
            m = mns[im]
            ax = axs[s,im]
    
            varc = np.copy(eof[:,m,:,:,N].mean(0))
            varf = np.copy(pattern[:,m,:,:,N].mean(0))
            vare = np.copy(varexp[:,m,N].mean(0))
            varf,lon1 = add_cyclic_point(varf,lon) 

            ax  = viz.init_map(bbox,ax=ax)
            ctf = ax.contourf(lon1,lat,varf,cint2,cmap=cmap)
            ax  = viz.plot_contoursign(varc,lon,lat,cint=cint1,ax=ax,add_cyc=True)
            ax.set_title("Mon%i %.1f"%(m+1,vare*100)+r"%")
            plt.colorbar(ctf,ax=ax,pad=0.10,fraction=0.046)
            
        #End mon loop
    
    # end season loop
    plt.tight_layout()
    plt.savefig("%sEOF%i_SLP_NHFLX_Plots_Ensavg.png" % (outpath,N+1),dpi=200)
        