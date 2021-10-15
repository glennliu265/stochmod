#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Input Parameters (Seasonal Cycle) at a Point, and Basinwide!

Plots included...

[1] Comparing EOF-based forcing (SLAB and FULL) vs. Old Forcing (FLXSTD)
[2] Seasonal cycle of inputs (Damping, Forcing, MLD) from Generals

Created on Tue Apr 27 01:20:49 2021
Updated Oct 2021 ...

@author: gliu
"""

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx

from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
from scipy import signal
from tqdm import tqdm

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import calendar as cal

import scm
import time
import cmocean

#%% User Edits

projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20211018/'
input_path  = datpath + 'model_input/'
proc.makedir(outpath)


# Put slab version first, then the load_load func. searches in the same
# directory replace "SLAB_PIC" with "FULL_PIC"
frcname = "flxeof_090pct_SLAB-PIC_eofcorr2"

# Which point do you want to visualize conditions for?
lonf = -30
latf = 50

# Additional Plotting Parameters
bbox = [-80,0,10,65]
#mconfig = "FULL_PIC"
#if mconfig == "FULL_PIC":
#    configname = "Fully-Coupled PiC"

#bboxplot  = [-100,20,-10,80]


#mconfig = "FULL_PIC"

# # ------------
# #%% Load Data

# # Lat and Lon
# lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
# dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
# loaddamp       = loadmat(input_path+dampmat)
# lon            = np.squeeze(loaddamp['LON1'])
# lat            = np.squeeze(loaddamp['LAT'])

# # Stochastic Model Input
# if "PIC" in mconfig: # Load Mixed layer variables (preprocessed in prep_mld.py)
#     hclim = np.load(input_path+"FULL_PIC_HMXL_hclim.npy")
#     kprevall    = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
# else: # Load Historical
#     hclim         = np.load(input_path+"HMXL_hclim.npy") # Climatological MLD
#     kprevall    = np.load(input_path+"HMXL_kprev.npy") # Entraining Month

# # Load Slab MLD
# hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    

# # Load full MLD field
# ds = xr.open_dataset(input_path+"HMXL_PIC.nc")
# mldfullpic = ds.HMXL.values

# # Reshape to separate month and season
# nlat,nlon,ntime = mldfullpic.shape
# nyr = int(ntime/12)
# mldfullpic = mldfullpic.reshape(nlat,nlon,nyr,12)

# Load Old Forcing
flxstd = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")


# Get mons3 from calendar function
mons3 = [cal.month_abbr[i] for i in np.arange(1,13,1)]
#%% Load All Inputs (Basinwide)

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs

innames = ["Longitude","Latitude",
           "Mixed-Layer Depth","Detraining Month",
           "Damping (SLAB)","Damping (FULL)",
           "Forcing (SLAB)","Forcing (FULL)"]

# Set some strings
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring      = "lon%i_lat%i" % (lonf,latf)
locstringtitle = "Lon: %.1f Lat: %.1f" % (lonf,latf)

# -------------------------------------
#%% Retrieve data for point comparisons
# -------------------------------------

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
#scm.get_data(1,[lonf,latf],lat,lon,)
inputs_pt = []
for i in range(len(inputs)):
    invar = inputs[i]
    print(inputs[i].shape)
    
    if len(invar.shape) > 1:
        inputs_pt.append(invar[klon,klat,...])

hpt,kprev,lbd_a,lbd_af,Fpt,Fpt_f = inputs_pt

#%% manuallly load forcing for debugging
load_forcing_manual=False

if load_forcing_manual:
    ffull = np.load(input_path+"flxeof_090pct_FULL-PIC_eofcorr2.npy")
    fslab = np.load(input_path+"flxeof_090pct_SLAB-PIC_eofcorr2.npy")
    Fpt = fslab[klon,klat,:]
    Fpt_f = ffull[klon,klat,:]

#%% [1] Compare New/Old Forcing

Fptstd = np.sqrt((Fpt**2).sum(0))
Fptstd2 = np.std(Fpt,0) # Note, this was not equivalent
Fptstdf = np.sqrt((Fpt_f**2).sum(0))

fig,ax = plt.subplots(1,1)
ax.plot(mons3,flxstd[klon,klat,...],label="$\sigma(Q_{net}'$)",color='k',marker="o")
ax.plot(mons3,Fptstd,label="EOF forcing (SLAB)",color="r",marker="X")
ax.plot(mons3,Fptstdf,label="EOF forcing (FULL)",color="cornflowerblue",marker="d")
#ax.plot(mons3,Fptstd2,label="np.std(x)")

ax.legend()
ax.grid(True,ls='dotted')
ax.set_title("Changes in Forcing at SPG Test Point (50N,30W)")
ax.set_ylabel("Forcing ($W/m^2$)")
ax.set_xlabel("Month")
ax.set_xlim([0,11])

#%% Generals-style visualization of MLD,LBD_A,Forcing at the point
# This was taken from [synth_stochmod_spectra.py]

def make_patch_spines_invisible(ax):
    #Source: https://matplotlib.org/2.0.2/examples/pylab_examples/multiple_yaxis_with_spines.html
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# plotting specs
plotylab = ("Mixed-Layer Depth ($m$)",
            "$Forcing \, Amplitude \, (W/m^2)$",
            "$\lambda_{a} (Wm^{-2} \, ^{\circ} C^{-1})$"
            )
plotlab  = ("h",r"$ \alpha $",r"$\lambda_a$")
plotmarker =("o","d","x")
plotcolor  = ("mediumblue","orangered","limegreen")


for m in range(2): # Loop for SLAB, and FULL

    # Setup based on model
    if m == 0:
        # Set variables to plot
        plotvar = [hpt,Fptstd,lbd_a]
        mcf = "CESM-SLAB"
    elif m == 1:
        plotvar = [hpt,Fptstdf,lbd_af]
        mcf = "CESM-FULL"
    title="Seasonal Cycle of Inputs (%s)" % (mcf)
    
    # Initialize figure, axes -------------
    fig,ax1 = plt.subplots(1,1,figsize=(5,3))
    fig.subplots_adjust(right=0.75)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    axs     = [ax1,ax2,ax3]
    tkw = dict(size=4, width=1.5)
    
    ls = []
    for i in range(3): # Plot each axis
        
        ax = axs[i]
        
        if i == 1:
            # Offset the right spine of par2.  The ticks and label have already been
            # placed on the right by twinx above.
            ax.spines["right"].set_position(("axes", 1.25))
            
            # Having been created by twinx, par2 has its frame off, so the line of its
            # detached spine is invisible.  First, activate the frame but make the patch
            # and spines invisible.
            make_patch_spines_invisible(ax)
            
            # Second, show the right spine.
            ax.spines["right"].set_visible(True)
            
            
        # Plot axis, then label and color
        p, = ax.plot(mons3,plotvar[i],color=plotcolor[i],label=plotlab[i],lw=0.75,marker=plotmarker[i],markersize=4)
        ax.set_ylabel(plotylab[i],fontsize=10)
        ax.yaxis.label.set_color(p.get_color())
        ax.tick_params(axis='y', colors=p.get_color(), **tkw)
        ls.append(p)
    
    # Additional Settings, Save Figure
    ax1.set_xticklabels(mons3, rotation = 45, ha="right")
    ax1.grid(True,ls='dotted')
    ax1.legend(ls, [l.get_label() for l in ls],loc='upper center')
    ax1.set_title(title)
    plt.savefig(outpath+"Scycle_MLD_Forcing_%s_Triaxis_%s.png"% (locstring,mcf),dpi=150,bbox_inches='tight')

# ****************************************************************************
#%% Some Basinwide Plots...

# Try to visualize the Forcing
cnames = ["SLAB","FULL"]
# Square, sum, sqrt along EOF dimension
alphas = [alpha,alpha_full]
alphas2 = []
alphasum = []
for i in range(2):
    a2 = np.sqrt(np.nansum(alphas[i]**2,2))
    asum = np.nansum(alphas[i],2)
    alphas2.append(a2)
    alphasum.append(asum)
    
    


#%% Annual Average Plot (Squared Sum)
clvl=np.arange(0,75,2.5)
fig,axs =  plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()})
for i in range(2):
    ax = axs[i]
    ax = viz.add_coast_grid(ax=ax,bbox=bbox)
    pcm=ax.contourf(lon,lat,np.nanmean(alphas2[i],2).T,levels=clvl,cmap=cmocean.cm.balance)
    ax.set_title(cnames[i])
    
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal')
plt.suptitle("Annual Mean Sqrt(Sum($EOF^2$))",y=1.05)
#alpha2 = np.sqrt(np.nansum(alpha**2,2))
#alphafull2 = 

#%% Annual Average sums

clvl=np.arange(-70,75,5)
fig,axs =  plt.subplots(1,2,subplot_kw={'projection':ccrs.PlateCarree()})
for i in range(2):
    ax = axs[i]
    ax = viz.add_coast_grid(ax=ax,bbox=bbox)
    pcm=ax.contourf(lon,lat,np.nanmean(alphasum[i],2).T,levels=clvl,cmap=cmocean.cm.balance)
    ax.set_title(cnames[i])
    
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal')
plt.suptitle("Sum(EOF^2)",y=1.05)



diff = np.nanmean(alphasum[1],2) - np.nanmean(alphasum[0],2)


