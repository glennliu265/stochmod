#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Input Parameters (Seasonal Cycle) at a Point, and Basinwide!

Plots included...

[1] Comparing EOF-based forcing (SLAB and FULL) vs. Old Forcing (FLXSTD)
[2] Seasonal cycle of inputs (Damping, Forcing, MLD) from Generals

[3] Stochastic Model Inputs (seasonal averages, FULL)


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
outpath    = projpath + '02_Figures/20211214/'
input_path  = datpath + 'model_input/'
proc.makedir(outpath)


# Put slab version first, then the load_load func. searches in the same
# directory replace "SLAB_PIC" with "FULL_PIC"
frcname = "flxeof_090pct_FULL-PIC_eofcorr2"

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
hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    

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


# -------------------
#%% Fancy Kprev Plot
# -------------------

monstr_kprv = np.append(mons3,'Jan')
fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
viz.viz_kprev(hpt,kprev,locstring="50$\degree$N, 30$\degree$W",
              ax=ax,msize=50,mstyle="x",lw=2.5)
ax.grid(True,ls='dotted')
ax.set_xticklabels(monstr_kprv)
plt.savefig(outpath+"MLD_Detrainment_month_SPGPoint.png",dpi=200)



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

# --------------------- --------------------- --------------------- ---------------------
#%% Compute seasonal averages

# calculate seasonal averages for the forcing (CESM-FULL) dataset 
alphaavg,snames = proc.calc_savg(alphas2[-1],debug=True,return_str=True)
dampingavg,snames = proc.calc_savg(dampingfull,debug=True,return_str=True)
havg,snames = proc.calc_savg(h,debug=True,return_str=True)

# Calculate seasonal averages for CESm-SLAB
alphaavgslab,snames = proc.calc_savg(alphas2[0],debug=True,return_str=True)
dampingavgslab,snames = proc.calc_savg(dampingslab,debug=True,return_str=True)
havgslab,snames = proc.calc_savg(hblt,debug=True,return_str=True)

#%% Plot the Forcing Patterns

clvl=np.arange(0,105,5)
fig,axs =  plt.subplots(1,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

for i in range(4):
    ax = axs[i]
    pcm=ax.contourf(lon,lat,alphaavg[i].T,levels=clvl,cmap='hot',extend='both')
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
    #fig.colorbar(pcm,ax=ax)
    ax.set_title(snames[i])

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009)
cb.set_label("Total Forcing Amplitude ($W/m^2$)")

#%% Plot the Damping Patterns


clvl=np.arange(-60,65,5)
fig,axs =  plt.subplots(1,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

for i in range(4):
    ax = axs[i]
    pcm=ax.contourf(lon,lat,dampingavg[i].T,levels=clvl,cmap=cmocean.cm.balance,extend='both')
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
    #fig.colorbar(pcm,ax=ax)
    ax.set_title(snames[i])

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009)
cb.set_label("Atmospheric Damping ($W/m^2$)")


#%% Plot the Mixed Layer Depth


vlms = [0,500]
#clvl=np.arange(-,65,5)

fig,axs =  plt.subplots(1,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

for i in range(4):
    ax = axs[i]
    pcm=ax.pcolormesh(lon,lat,havg[i].T,vmin=vlms[0],vmax=vlms[1],cmap='cmo.deep')
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
    #fig.colorbar(pcm,ax=ax)
    ax.set_title(snames[i])

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009)
cb.set_label("Mixed-Layer Depth ($m$)")



#%% Now Plot all 3 Together

# Set Inputs
invars = (alphaavg,dampingavg,havg)
cblabs = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contour = 50 $m$"
          )
vnames = (r"Total Forcing Amplitude ($\alpha$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")
cblabs2 = (u"Contour = 5 $Wm^{-2}$",u"Contour = 5 $Wm^{-2} \degree C^{-1}$",u"Contour = 50 $m$")

cints  = (np.arange(0,105,5),np.arange(0,65,5),np.arange(0,1050,50)
          )

cmaps  = ('hot','cmo.thermal','cmo.dense') 

snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')

fig,axs = plt.subplots(3,4,figsize=(12,8),subplot_kw={'projection':ccrs.PlateCarree()})
for v in range(3):
    
    invar = invars[v]
    cblab = cblabs[v]
    cint  = cints[v]
    cmap  = cmaps[v]
    
    for s in range(4):
        ax = axs[v,s]
        
        # Set Lat/Lon Labels
        blabel = [0,0,0,0]
        if v == 2:
            blabel[-1] = 1 # Add Bottom Label
        if s == 0:
            blabel[0]  = 1 # Add Left Label
        
        # Set Title (First Row Only)
        if v == 0:
            ax.set_title(snamesl[s],fontsize=14)
        
        pcm=ax.contourf(lon,lat,invar[s].T,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray')
        
    cb = fig.colorbar(pcm,ax=axs[v,:].flatten(),orientation='vertical',fraction=0.009)
    cb.set_label(cblab,fontsize=12)
        
    
#%% Try Subfigures Method
import matplotlib as mpl

#mpl.rcParams['font.sans-serif'] = "Avenir"#"stix"
#mpl.rcParams['font.family'] = "sans-serif"#"STIXGeneral"
#mpl.rcParams["text.usetex"] = True


mpl.rcParams['font.sans-serif'] = "stix"
mpl.rcParams['font.family'] = "STIXGeneral"


mpl.rcParams.update(mpl.rcParamsDefault)


cblabs2 = (u"Contours: 5 $Wm^{-2}$",
           u"Contours: 5 $Wm^{-2} \degree C^{-1}$",
           u"Contours: 50 $m$")


fig = plt.figure(constrained_layout=True,figsize=(12,8))
fig.suptitle("Stochastic Model Inputs (CESM1-FULL, Seasonal Average)",fontsize=20)

# Create 3x1 subfigs
subfigs = fig.subfigures(nrows=3,ncols=1)
for row,subfig in enumerate(subfigs):
    subfig.suptitle(vnames[row])
    
    v = row
    
    invar = invars[v]
    cblab = cblabs[v]
    cint  = cints[v]
    cmap  = cmaps[v]
    
    # Create 1x4 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=4,subplot_kw={'projection':ccrs.PlateCarree()})
    
    for s, ax in enumerate(axs):
        
        # Set Lat/Lon Labels
        blabel = [0,0,0,0]
        if v == 2:
            blabel[-1] = 1 # Add Bottom Label
        if s == 0:
            blabel[0]  = 1 # Add Left Label
        
        # Set Title (First Row Only)
        if v == 0:
            ax.set_title(snamesl[s],fontsize=14)
        
        pcm=ax.contourf(lon,lat,invar[s].T,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray')
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009,pad=.010)
    cb.set_label(cblabs2[v],fontsize=12)
    
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_CESM-FULL.png",dpi=200,bbox_inches='tight')

#%% Plot CESM1-SLAB Forcing and Damping


invars = (alphaavgslab,dampingavgslab,havgslab)
cblabs = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contour = 50 $m$"
          )
vnames = (r"Total Forcing Amplitude ($\alpha$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")
cblabs2 = (u"Contour = 5 $Wm^{-2}$",u"Contour = 5 $Wm^{-2} \degree C^{-1}$",u"Contour = 50 $m$")
cints  = (np.arange(0,105,5),np.arange(0,65,5),np.arange(0,1050,50)
          )
cmaps  = ('hot','cmo.thermal','cmo.dense') 
snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')


fig = plt.figure(constrained_layout=True,figsize=(12,8))
fig.suptitle("Stochastic Model Inputs (CESM1-SLAB, Seasonal Average)",fontsize=20)

# Create 3x1 subfigs
subfigs = fig.subfigures(nrows=3,ncols=1)
for row,subfig in enumerate(subfigs):
    subfig.suptitle(vnames[row])
    
    v = row
    
    invar = invars[v]
    cblab = cblabs[v]
    cint  = cints[v]
    cmap  = cmaps[v]
    
    # Create 1x4 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=4,subplot_kw={'projection':ccrs.PlateCarree()})
    
    for s, ax in enumerate(axs):
        
        # Set Lat/Lon Labels
        blabel = [0,0,0,0]
        if v == 2:
            blabel[-1] = 1 # Add Bottom Label
        if s == 0:
            blabel[0]  = 1 # Add Left Label
        
        # Set Title (First Row Only)
        if v == 0:
            ax.set_title(snamesl[s],fontsize=14)
        
        pcm=ax.contourf(lon,lat,invar[s].T,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray')
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009,pad=.010)
    cb.set_label(cblabs2[v],fontsize=12)
    
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_CESM-SLAB.png",dpi=200,bbox_inches='tight')


# --------------------------------------
#%% Compare CESM1 FULL MLD with Levitus
# --------------------------------------
# Load WOA
levpath = "/Users/gliu/Downloads/06_School/01_Fall2019/12860/12860/TermProject/tp_p1/Data/mld/"
levname = "mldpd.mnltm.nc"
ds      = xr.open_dataset(levpath+levname)
h_woa   = ds.mldpd.values
lat_woa = ds.lat.values
lon_woa = ds.lon.values

# Transpose [mon x lat x lon] --> [lon x lat x mon]
print(h_woa.shape)
h_woa = h_woa.transpose(2,1,0)
print(h_woa.shape)

# Convert lon to -180
lon1_woa,h_woa = proc.lon360to180(lon_woa,h_woa)

# Flip latitude
h_woa = np.flip(h_woa,1)
lat_woa = np.flip(lat_woa)

# Do interpolation, Compute Differences (Assume lat/lon are midpoints
x_tol = (lon[1] - lon[0])
y_tol = (lat[1] - lat[0])

nlon,nlat,_ = h.shape
mld_diff = np.zeros(h.shape)*np.nan
woa_avg  = mld_diff.copy()
for o in range(nlon):
    lonc = lon[o]
    
    for a in range(nlat):
        latc = lat[a]
        
        klat = np.where((lat_woa >= latc - y_tol) & (lat_woa <=  latc + y_tol))[0]
        klon = np.where((lon1_woa >= lonc - x_tol) & (lon1_woa <=  lonc + x_tol))[0]
        
        if np.any(np.array(klat[0].shape)==0) or np.any(np.array(klon[0].shape)==0):
            print("Nothing found for lon %f lat %f" % (lonc,latc))
            continue
        h_values = h_woa[klon[:,None,None],klat[None,:,None],:]
        if len(h_values.shape)<3:
            print("Warning size is under 3!")
            break
        if np.any(np.array(h_values.shape)==0): # Points where nothing is found
            print("Nothing found for lon %f lat %f" % (lonc,latc))
            continue
        woa_avg[o,a,:]  = np.nanmean(h_values,(0,1))

# Compute Differences, Then seasonal Average
mld_diff = h-woa_avg
hdiff_savg,monstrs=proc.calc_savg(mld_diff,return_str=True,debug=True)

#%% Make the Plot
fig,axs = plt.subplots(2,2,figsize=(7,7),
                      subplot_kw={'projection':ccrs.PlateCarree()})

for s,ax in enumerate(axs.flatten()):
    print(s)
    pcm = ax.pcolormesh(lon,lat,hdiff_savg[s].T,vmin=-250,vmax=250,cmap='cmo.balance')
    
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,fill_color='gray')
    ax.set_title(monstrs[s])
    #fig.colorbar(pcm,ax=ax)
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.05)
plt.suptitle("Seasonal Mean Mixed Layer Depth Differences in meters \n (CESM1 - WOA 1994)",fontsize=14,y=.94)

plt.savefig("%sMLD_Differences-CESM1_WOA1994_Savg.png" %(outpath),dpi=200,bbox_inches='tight')






