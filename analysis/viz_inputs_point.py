#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Input Parameters (Seasonal Cycle) at a Point, and Basinwide

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
import colorcet as cc

import scm
import time
import cmocean

#%% User Edits

projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20220414/'
input_path  = datpath + 'model_input/'
proc.makedir(outpath)


# Put slab version first, then the load_load func. searches in the same
# directory replace "SLAB_PIC" with "FULL_PIC"
frcname = "flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0"
#frcname = "flxeof_090pct_FULL-PIC_eofcorr2"
#frcname = "Qek_eof_090pct_FULL_PIC_eofcorr0"

# Which point do you want to visualize conditions for?
lonf = -30#-55#
latf = 50#11 #50
flocstring = "lon%i_lat%i" % (lonf,latf)
locstring = "%i$\degree$N, %i$\degree$W" % (latf,np.abs(lonf))

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

# Load limask
limask = np.load(datpath+"model_input/limask180_FULL-HTR.npy")

# # Regional Analysis Settings (NEW, with STG Split)
# Regional Analysis Settings
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic","Subtropical (East)","Subtropical (West)",)
bbcol       = ["Blue","Red","Yellow","Black","Black"]
bbcol       = ["Blue","Red","Yellow","Black","Black","magenta","red"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted"]

method = 5
lagstr = 'lag1'

brew_cat8 = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']

#%% Load All Inputs (Basinwide)

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
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


# Calculate entrainment related damping
beta = scm.calc_beta(h)

# -------------------
#%% Fancy Kprev Plot
# -------------------

monstr_kprv = np.append(mons3,'Jan')
fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
viz.viz_kprev(hpt,kprev,locstring=locstring,
              ax=ax,msize=10,mstyle="X",lw=2.5,txtalpha=.65,usetitle=False)
ax.grid(True,ls='dotted')
ax.set_xticklabels(monstr_kprv) 
ax.set_ylim([10,150])
ax.set_yticks(np.arange(10,170,20))

ax.minorticks_on()
ax.xaxis.set_tick_params(which='minor', bottom=False)
plt.savefig(outpath+"MLD_Detrainment_month_%s.png" % (flocstring),dpi=200)
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
# Also used in SM Paper Draft

useC    = True
notitle = True


def make_patch_spines_invisible(ax):
    #Source: https://matplotlib.org/2.0.2/examples/pylab_examples/multiple_yaxis_with_spines.html
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# plotting specs
if useC:
    plotylab = ("Mixed-Layer Depth ($m$)",
                "Forcing Amplitude $(Wm^{-2})$",
                "Heat Flux Feedback $(Wm^{-2} \, \degree C^{-1})$"
                )
    
else:
    plotylab = ("Mixed-Layer Depth ($m$)",
                "Forcing Amplitude $(Wm^{-2})$",
                "Heat Flux Feedback $(Wm^{-2} \, K^{-1})$"
                )
plotlab  = ("h",r"$ F'$",r"$\lambda_a$")
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
    if notitle is False:
        ax1.set_title(title)
    plt.savefig(outpath+"Scycle_MLD_Forcing_%s_Triaxis_%s.png"% (flocstring,mcf),dpi=150,bbox_inches='tight')

# ****************************************************************************
#%% Some Basinwide Plots...

# Try to visualize the Forcing
cnames = ["SLAB","FULL"]
# Square, sum, sqrt along EOF dimension
alphas = [alpha,alpha_full]
alphas2 = []
alphasum = []
for i in range(2):
    a2 = np.sqrt(np.nansum(alphas[i]**2,2)) # Sqrt of sum of squares
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
betaavg,_ = proc.calc_savg(beta,debug=True,return_str=True)

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
plt.savefig("%sForcing_SeasonaAvg_%s.png"%(outpath,frcname),dpi=200)
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


#%% Plot Beta

fig,axs =  plt.subplots(1,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})
clvl = np.linspace(0,1.5,20)

for i in range(4):
    ax = axs[i]
    pcm=ax.contourf(lon,lat,betaavg[i].T,levels=clvl,cmap=cmocean.cm.balance,extend='both')
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
    #fig.colorbar(pcm,ax=ax)
    ax.set_title(snames[i])
fig.colorbar(pcm,ax=axs.flatten())

#%% Plot "Cumulative beta"



betacumu = beta.sum(-1)
fig,ax =  plt.subplots(1,1,figsize=(7,7),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax=ax,bbox=bbox,fill_color='gray')
pcm = ax.contourf(lon,lat,betacumu.T)
cb=fig.colorbar(pcm,ax=ax)


#%% Plot max MLD variations

bboxplot    = [-80,0,5,60]
hrange= np.nanmax(h,axis=2) - np.nanmin(h,axis=2)


fig,ax =  plt.subplots(1,1,figsize=(8,4),subplot_kw={'projection':ccrs.PlateCarree()})

pcm=ax.pcolormesh(lon,lat,hrange.T,vmin=0,vmax=125,cmap=cmocean.cm.dense)
ax = viz.add_coast_grid(ax=ax,bbox=bboxplot,fill_color='gray')
fig.colorbar(pcm,ax=ax)



#%% Experiment with nonlinear colormap

#https://stackoverflow.com/questions/8461605/making-small-values-visible-on-matplotlib-colorbar-in-python
from matplotlib import colors
import cmocean
#import matplotlib as mpl
# cdict = {'red':   [(0.0,  0.0, 0.0),
#                    (0.5,  1.0, 1.0),
#                    (1.0,  1.0, 1.0)],

#          'green': [(0.0,  0.0, 0.0),
#                    (0.25, 0.0, 0.0),
#                    (0.75, 1.0, 1.0),
#                    (1.0,  1.0, 1.0)],

#          'blue':  [(0.0,  0.0, 0.0),
#                    (0.5,  0.0, 0.0),
#                    (1.0,  1.0, 1.0)]}


# Try out different gammas > 1.0. Gamma >1 increases sensitivity in the lower part of scale
#cmapmld = colors.LinearSegmentedColormap.from_list('mldmap',['darkred','ivory','darkblue'],gamma=0.275)
#cmapmld = colors.LinearSegmentedColormap.from_list('mldmap',['w','crimson','ivory','darkblue'],gamma=0.3)
cmapmld = 'cmo.topo'
#cmap = 'jet'
cintmld = [0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,200,
           300,400,500,600,700,800,900,1000,1100,1200]
#cintmld = np.arange(0,1250,10)
#cintmld = np.concatenate([np.arange(0,200,5),np.arange(200,1300,100)])



    
colors1  = plt.cm.Purples(np.linspace(0,1.,128))
colors1  = zip(np.linspace(0,0.5,128),colors1) 
colors2  = cmocean.cm.deep(np.linspace(0,1.,128))
colors2  = zip(np.linspace(0.5,1.,128),colors2) 
colorsf = list(colors1) + list(colors2)
cmapmld = colors.LinearSegmentedColormap.from_list('mldmap', colorsf,gamma=0.4)

#https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps

#%% Plot the Mixed Layer Depth

vlms = [0,1000] 
#clvl=np.arange(-,65,5)

fig,axs =  plt.subplots(1,4,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

for i in range(4):
    
    ax = axs[i]
    #pcm=ax.pcolormesh(lon,lat,havg[i].T,vmin=vlms[0],vmax=vlms[1],cmap='cmo.dense')
    pcm = ax.contourf(lon,lat,havg[i].T,levels=cintmld,cmap=cmapmld)
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
    #fig.colorbar(pcm,ax=ax)
    ax.set_title(snames[i])
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009)
cb.set_label("Mixed-Layer Depth ($m$)")


#%% Check maximum MLD depth in the summer
bboxs = [-80,0,0,65]
havgr,lonr,latr = proc.sel_region(havg[2][...,None],lon,lat,bboxs)

fig,ax =  plt.subplots(1,1,figsize=(6,6),subplot_kw={'projection':ccrs.PlateCarree()})

pcm = ax.pcolormesh(lon,lat,havg[2].T,vmin=0,vmax=100,cmap="cmo.dense")
ax  = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=[0,0,0,0],fill_color='gray')
fig.colorbar(pcm,ax=ax)
#%% Now Plot all 3 Together (OLD)

# Set Inputs
cintmld = [0,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500,750,1000,1250]
cmapmld = colors.LinearSegmentedColormap.from_list('mldmap',['fuchsia','w','darkblue'],gamma=0.30)
cmapdamp = colors.LinearSegmentedColormap.from_list('mldmap',['darkgreen','mintcream'],gamma=1)

invars = (alphaavg,dampingavg,havg)
cblabs = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contours = 10-250 $m$"
          )
vnames = (r"Total Forcing Amplitude ($\alpha$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")
cblabs2 = (u"Contour = 5 $Wm^{-2}$",u"Contour = 5 $Wm^{-2} \degree C^{-1}$",u"Contours = 10 to 250 $m$")

cints  = (np.arange(0,105,5),np.arange(0,45,5),cintmld
          )


cmaps  = ('hot',cmapdamp,cmapmld) 

snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')


sp_id = 0

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
        if v < 2:
            plotvar = (invar[s] * limask).T
        else:
            plotvar = invar[s].T
        
        pcm=ax.contourf(lon,lat,plotvar,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray')
        
        ax = viz.label_sp(sp_id,ax=ax,fontsize=14,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
        sp_id += 1
        
    cb = fig.colorbar(pcm,ax=axs[v,:].flatten(),orientation='vertical',fraction=0.009)
    cb.set_label(cblab,fontsize=12)
        
    
#%% Try Subfigures Method (Outdated, for SM Draft 2)
import matplotlib as mpl

notitle   = True
nocblabel = True
leftlabel = True

#mpl.rcParams['font.sans-serif'] = "Avenir"#"stix"
#mpl.rcParams['font.family'] = "sans-serif"#"STIXGeneral"
#mpl.rcParams["text.usetex"] = True


mpl.rcParams['font.sans-serif'] = "stix"
mpl.rcParams['font.family'] = "STIXGeneral"

mpl.rcParams.update(mpl.rcParamsDefault)

cblabs2 = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contours = 10-250 $m$"
          )

# cblabs2 = (u"Contours: 5 $Wm^{-2}$",
#            u"Contours: 5 $Wm^{-2} \degree C^{-1}$",
#            u"Contours = 10-250 $m$")


fig = plt.figure(constrained_layout=True,figsize=(14,8))
#fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=0, hspace=0, wspace=0)
if notitle is False:
    fig.suptitle("Stochastic Model Inputs (CESM1-FULL, Seasonal Average)",fontsize=20)

# Create 3x1 subfigs
sp_id = 0
subfigs = fig.subfigures(nrows=3,ncols=1)
for row,subfig in enumerate(subfigs):
    if notitle is False:
        subfig.suptitle(vnames[row])
    
    v = row
    
    invar = invars[v]
    cblab = cblabs[v]
    cint  = cints[v]
    cmap  = cmaps[v]
    
    # Create 1x4 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=4,
                          subplot_kw={'projection':ccrs.PlateCarree()})
    
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
        if v < 2:
            plotvar = (invar[s] * limask).T
        else:
            plotvar = invar[s].T
        
        pcm=ax.contourf(lon,lat,plotvar,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray',ignore_error=True)
        
        ax = viz.label_sp(sp_id,ax=ax,fontsize=18,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
        sp_id += 1
        
        if leftlabel:
            if s == 0:
                ax.text(-0.15, 0.55, vnames[v], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)
                #ax.set_ylabel(vnames[s])
                #ax.text(x = 0, y = 0, s = vnames[s], rotation = 90, va = "top",ha="left",fontsize=8)
                
            
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009,pad=.010)
    #cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.075,pad=.1)
    if nocblabel is False:
        cb.set_label(cblabs2[v],fontsize=12)
        
#fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_CESM-FULL.png",dpi=200,bbox_inches='tight')
#%% Try Subfigures Method (FIXED CONSTRAINED LAYOUT, for SM Draft 2)
import matplotlib as mpl

# Set Inputs
#cintmld = np.concatenate([np.arange(0,105,5)np.arange(125,225,25),np.arange()])
cintmld = [0,10,20,30,40,50,60,70,80,90,100,125,150,175,200,300,400,500,600,700,750,1000,1250]
#cmapmld = colors.LinearSegmentedColormap.from_list('mldmap',['w','indigo','k'],gamma=0.37)
cmapmld = colors.LinearSegmentedColormap.from_list('mldmap',['w','indigo','k'],gamma=1)
cmapdamp = colors.LinearSegmentedColormap.from_list('mldmap',['darkgreen','mintcream'],gamma=1)

invars = (alphaavg,dampingavg,havg)
cblabs = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contours = 10-250 $m$"
          )
vnames = (r"Total Forcing Amplitude ($F'$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")

cints  = (np.arange(0,105,5),np.arange(0,45,5),cintmld
          )


cmaps  = ('gist_heat_r','cmo.algae',cmapmld) 

snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')


notitle   = True
nocblabel = True
leftlabel = True

mpl.rcParams['font.sans-serif'] = "stix"
mpl.rcParams['font.family'] = "STIXGeneral"
mpl.rcParams.update(mpl.rcParamsDefault)


fig,axs = plt.subplots(3,4,constrained_layout=True,
                       figsize=(14,8),subplot_kw={'projection':ccrs.PlateCarree()})

if notitle is False:
    fig.suptitle("Stochastic Model Inputs (CESM1-FULL, Seasonal Average)",fontsize=20)

# Create 3x1 subfigs
sp_id = 0
for row in range(3):
    
    v = row
    
    invar = invars[v]
    cblab = cblabs[v]
    cint  = cints[v]
    cmap  = cmaps[v]

    for s  in range(4):
        
        ax = axs[row,s]
        
        # Set Lat/Lon Labels
        blabel = [0,0,0,0]
        if v == 2:
            blabel[-1] = 1 # Add Bottom Label
        if s == 0:
            blabel[0]  = 1 # Add Left Label
        
        # Set Title (First Row Only)
        if v == 0:
            ax.set_title(snamesl[s],fontsize=14)
        if v < 3:
            plotvar = (invar[s] * limask).T
        # else:
        #     plotvar = invar[s].T
        
        if v == 2:
            pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=200,cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=200))
            #pcm = ax.pcolormesh(lon,lat,plotvar,vmin=0,vmax=200,cmap=cmap)
            cf = ax.contour(lon,lat,plotvar,levels=np.arange(300,1050,150),colors='w',linewidths=0.75)
            ax.clabel(cf,levels=[300,600,900],fontsize=10,inline_spacing=2)
        else:
            
            pcm = ax.pcolormesh(lon,lat,plotvar,vmin=cint[0],vmax=cint[-1],cmap=cmap)
            #pcm = ax.contourf(lon,lat,plotvar,levels=cint,extend='both',cmap=cmap)
            
        #print(ax)
        ax  = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray',ignore_error=True)
        ax = viz.label_sp(sp_id,ax=ax,fontsize=18,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
        sp_id += 1
        
        if leftlabel:
            if s == 0:
                ax.text(-0.15, 0.55, vnames[v], va='bottom', ha='center',rotation='vertical',
                        rotation_mode='anchor',transform=ax.transAxes)

    cb = fig.colorbar(pcm,ax=axs[row,:],orientation='vertical',fraction=0.009,pad=.010)
    #cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.075,pad=.1)
    if nocblabel is False:
        cb.set_label(cblabs2[v],fontsize=12)
        
#fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_CESM-FULL.png",dpi=200,bbox_inches='tight')

#%% Plot CESM1-SLAB Forcing and Damping


invars = (alphaavgslab,dampingavgslab,havgslab)

vnames = (r"Total Forcing Amplitude ($\alpha$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")

cblabs2 = ("Total Forcing Amplitude \n Contour = 5 $W/m^2$",
          "Atmospheric Damping \n Contour = 5 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contours = 10-250 $m$"
          )



# Draft 1
# cints  = (np.arange(0,105,5),np.arange(0,65,5),np.arange(0,1050,50)
#           )
# cmaps  = ('hot','cmo.thermal','cmo.dense') 
cints  = (np.arange(0,105,5),np.arange(0,65,5),cintmld
          )
cmaps  = ('hot',cmapdamp,cmapmld) 

snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')


fig = plt.figure(constrained_layout=True,figsize=(12,8))
fig.suptitle("Stochastic Model Inputs (CESM1-SLAB, Seasonal Average)",fontsize=20)
sp_id = 0
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
        
        ax = viz.label_sp(sp_id,ax=ax,fontsize=18,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
        sp_id += 1
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009,pad=.010)
    cb.set_label(cblabs2[v],fontsize=12)
    

    
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_CESM-SLAB.png",dpi=200,bbox_inches='tight')

# ----------------------------------------
# %% Plot Differences in Heat Flux Feedback
# ----------------------------------------

notitle = True

fig,axs = plt.subplots(2,2,figsize=(7,7),constrained_layout=True,
                      subplot_kw={'projection':ccrs.PlateCarree()})

sp_id = 0

cints = np.arange(-24,26,2)
snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')

for s,ax in enumerate(axs.flatten()):
    
    blabel = [0,0,0,0]
    if s%2 == 0:
        blabel[0] = 1
    if s>1:
        blabel[-1]=1    
    
    plotvar = dampingavg[s]-dampingavgslab[s]
    
    print(s)
    pcm = ax.contourf(lon,lat,plotvar.T,levels=cints,cmap='cmo.balance')
    
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,fill_color='gray',blabels=blabel)
    ax.set_title(snamesl[s])
    #fig.colorbar(pcm,ax=ax)
    ax = viz.label_sp(sp_id,ax=ax,fontsize=14,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
    sp_id += 1
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.05)
cb.set_label("Atmospheric Heat Flux Feedback ($Wm^{-2}K^{-1}$)")
if notitle is False:
    plt.suptitle("Seasonal Mean Mixed Layer Depth Differences in meters \n (CESM1 - WOA 1994)",fontsize=14,y=.94)
plt.savefig("%sHFLX_Differences_FULL-SLAB_Savg.png" %(outpath),dpi=200,bbox_inches='tight')



# ----------------------------------------------------------
# %% Plot the differences between CESM1-FULL and CESM1-SLAB
# ----------------------------------------------------------

# Declare the lists
invarsslab = (alphaavgslab,dampingavgslab,havgslab)
invarsfull = (alphaavg,dampingavg,havg)
    
# Set the Labels
notitle = True
cblabs2 = ("Total Forcing Amplitude \n Contour = 1 $W/m^2$",
          "Atmospheric Damping \n Contour = 2 $W/m^2 / \degree C$",
          "Mixed-Layer Depth \n Contours = 20 $m$"
          )
vnames = (r"Total Forcing Amplitude ($\alpha$)",
          r"Atmospheric Damping ($\lambda_a$)",
          r"Mixed-Layer Depth ($h$)")
cblabs2 = (u"Contour = 5 $Wm^{-2}$",u"Contour = 5 $Wm^{-2} \degree C^{-1}$",u"Contours = 10-250 $m$")

# Draft 2
cints  = (np.arange(-10,11,1),np.arange(-24,26,2),np.arange(-500,520,20))
          
cmaps  = ('PiYG','RdBu_r','PuOr') 
snamesl = ('Winter (DJF)','Spring (MAM)','Summer (JJA)','Fall (SON)')


fig = plt.figure(constrained_layout=True,figsize=(12,8))
#fig.suptitle("Stochastic Model Inputs (CESM1-SLAB, Seasonal Average)",fontsize=20)
sp_id = 0
# Create 3x1 subfigs
subfigs = fig.subfigures(nrows=3,ncols=1)
for row,subfig in enumerate(subfigs):
    if notitle is False:
        subfig.suptitle(vnames[row])
    
    v = row
    
    #invar = invars[v]
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
            
        plotvar = (invarsslab[v][s] - invarsfull[v][s]).T
        
        pcm=ax.contourf(lon,lat,plotvar,levels=cint,extend='both',cmap=cmap)
        ax = viz.add_coast_grid(ax=ax,bbox=bbox,blabels=blabel,fill_color='gray')
        
        ax = viz.label_sp(sp_id,ax=ax,fontsize=18,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
        sp_id += 1
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.009,pad=.010)
    cb.set_label(cblabs2[v],fontsize=12)
    

    
#plt.show()
plt.savefig(outpath+"Seasonal_Inputs_SLAB-Minus-FULL.png",dpi=200,bbox_inches='tight')


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

notitle = True

fig,axs = plt.subplots(2,2,figsize=(7,7),constrained_layout=True,
                      subplot_kw={'projection':ccrs.PlateCarree()})

sp_id = 0
for s,ax in enumerate(axs.flatten()):
    
    blabel = [0,0,0,0]
    if s%2 == 0:
        blabel[0] = 1
    if s>1:
        blabel[-1]=1    
    
    print(s)
    pcm = ax.pcolormesh(lon,lat,hdiff_savg[s].T,vmin=-250,vmax=250,cmap='cmo.balance')
    
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,fill_color='gray',blabels=blabel)
    ax.set_title(monstrs[s])
    #fig.colorbar(pcm,ax=ax)
    ax = viz.label_sp(sp_id,ax=ax,fontsize=14,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
    sp_id += 1
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.05)
cb.set_label("Mixed-layer depth (m)")
if notitle is False:
    plt.suptitle("Seasonal Mean Mixed Layer Depth Differences in meters \n (CESM1 - WOA 1994)",fontsize=14,y=.94)
plt.savefig("%sMLD_Differences-CESM1_WOA1994_Savg.png" %(outpath),dpi=200,bbox_inches='tight')

# -------------------------
#%% Compare CESM with MIMOC
# -------------------------
import glob

recalc = False


if recalc:
    mldpath = datpath + "MIMOC_ML_v2.2_PT_S/"
    #testpath = mldpath + "MIMOC_ML_v2.2_PT_S_MLP_month01.nc"
    nclist = glob.glob(mldpath+"*.nc")
    nclist.sort()
    print(nclist)
    
    # Read in and concatenate by month variable
    ds_all = []
    for nc in nclist:
        ds = xr.open_dataset(nc)
        print(ds)
        ds_all.append(ds.DEPTH_MIXED_LAYER)
    ds_all = xr.concat(ds_all,dim="month")
    
    
    # Get dimensions, read to numpy array
    lenlon = len(ds_all[0].LONG)
    lenlat = len(ds_all[0].LAT)
    mmlon = np.linspace(0,360,lenlon)
    mmlat = np.linspace(-90,90,lenlat)
    mm_mld = np.zeros((12,lenlat,lenlon))*np.nan
    for d,ds in enumerate(ds_all):
        mm_mld[d,:,:] = ds.values
        
    # Now Interpolate
    deg = 1 #Placeholder
    tol = 1
    lon360,_ = scm.load_latlon(lon360=True)
    outvar,lat5,lon5 = proc.coarsen_byavg(mm_mld,mmlat,mmlon,deg,tol,newlatlon=[lon360,lat],usenan=True,latweight=False)

    # Save netcdf
    savenetcdf = datpath+"MIMOC_ML_v2.2_regriddedCESM_noweight.nc"
    ds = proc.numpy_to_da(outvar,np.arange(0,12,1),lat5,lon5,"MLD",savenetcdf=savenetcdf)
else:
    savenetcdf = datpath+"MIMOC_ML_v2.2_regriddedCESM_noweight.nc"
    ds = xr.open_dataset(savenetcdf)

# Reload and regrid
mm_mld = ds.MLD.values # {month x lat x lon}
lon360 = ds.lon.values
mm_mld = mm_mld.transpose(2,1,0)
_,mm_mld = proc.lon360to180(lon360,mm_mld)

# Compute Differences, Then seasonal Average
mld_diff = h-mm_mld
hdiff_savg,monstrs=proc.calc_savg(mld_diff,return_str=True,debug=True)
mm_savg,monstrs = proc.calc_savg(mm_mld,return_str=True,debug=True)
#%% Make the Plot

notitle = True
plotmld = False # Tur to just plot mimoc seasonal avg

fig,axs = plt.subplots(2,2,figsize=(7,7),constrained_layout=True,
                      subplot_kw={'projection':ccrs.PlateCarree()})

sp_id = 0
for s,ax in enumerate(axs.flatten()):
    
    blabel = [0,0,0,0]
    if s%2 == 0:
        blabel[0] = 1
    if s>1:
        blabel[-1]=1    
    
    print(s)
    if plotmld:
        pcm = ax.pcolormesh(lon,lat,mm_savg[s].T,vmin=-250,vmax=250,cmap='cmo.balance')
        fig.colorbar(pcm,ax=ax)
    else:
        pcm = ax.pcolormesh(lon,lat,hdiff_savg[s].T,vmin=-250,vmax=250,cmap='cmo.balance')
    
    
    ax = viz.add_coast_grid(ax=ax,bbox=bbox,fill_color='gray',blabels=blabel)
    ax.set_title(monstrs[s])
    
    ax = viz.label_sp(sp_id,ax=ax,fontsize=14,fig=fig,labelstyle="(%s)",case='lower',alpha=.75)
    sp_id += 1
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.05)
cb.set_label("Mixed-layer depth (m)")
if notitle is False:
    plt.suptitle("Seasonal Mean Mixed Layer Depth Differences in meters \n (CESM1 - MIMOC)",fontsize=14,y=.94)
plt.savefig("%sMLD_Differences-CESM1_MIMOC_Savg.png" %(outpath),dpi=200,bbox_inches='tight')





# -------------------------------------------------------
#%% Plot Seasonal Variation in parameters for each region
# -------------------------------------------------------

# Make Selection
ridsel = [0,2,6,5] # SPG, STGw, STGe
inparams   = [alphas2[1],dampingfull,h] # [lon x lat x month]
param_name = ("forcing","damping","mld")

# Make shared li mask
allmask = np.ones((288,192,12))
for v in range(3):
    
    allmask[np.isnan(inparams[v])]=np.nan
    

# Store in List of Dicts by region number
rparams = [] # Regional Parameters [region_id]
for r in ridsel:
    paramdict = {}
    for v in range(3):
        
        # Select parameter for that region
        vreg,_,_ = proc.sel_region(inparams[v]*allmask,lon,lat,bboxes[r])
        
        # Remove Any NaNs
        vshape = vreg.shape
        vreg   = vreg.reshape(np.prod(vshape[:2]),vshape[2])
        okdata,knan,okpts = proc.find_nan(vreg,1)
        print(okdata.shape)
        
        #vreg   = vreg.reshape(np.prod(vshape[:2]),vshape[2])
        paramdict[param_name[v]]=okdata
        
    rparams.append(paramdict)

#%% Make a Monthly Box Plot/Stdev plot

sepfig = True
rid    = 0

fig,axs = plt.subplots(3,1,figsize=(6,6))

for r,rid in enumerate(ridsel):
    for v in range(3):
        
        ax = axs.flatten()[v]
        
        print(ax)
        
        plotvar = rparams[r][param_name[v]]
        print(plotvar.shape)
        vmean = np.nanmean(plotvar,0)
        vstd  = 1*np.nanstd(plotvar,0)
        
        ax.set_title(param_name[v])
        
        ax.plot(mons3,vmean,color=bbcol[rid],label=regions[rid])
        
        ax.fill_between(mons3,vmean-vstd,vmean+vstd,color=bbcol[rid],label="",alpha=0.05)
        #ax.plot(mons3,vmean)
        
        #ax.boxplot(plotvar,linecolor=bbox_col[rid])
        ax.set_xticks(np.arange(0,12,1))
        ax.set_xticklabels(mons3)
        ax.set_xlim([0,11])
        
        if v == 0:
            ax.legend()
        
        ax.grid(True,ls='dotted')
    
plt.tight_layout()
    

#%% Make a Scatterplot of the different space

rho = 1026
cp0 = 3996
dt  = 3600*24*30
reciprocal = True
samelims   = True


if reciprocal is not True:
    xlim = [0,1]
    ylim = [0,2]
    #xlim = [0,2]
    #ylim = [0,2]
else:
    xlim = [0,35]
    ylim = [0,1]
    

fig,axs = plt.subplots(4,3,figsize=(12,12),constrained_layout=True)

for m in range(12):
    
    
    ax = axs.flatten()[m]
    for r,rid in enumerate(ridsel):
        
        hreg = rparams[r]['mld'][...,m]
        dreg = rparams[r]['damping'][...,m] / (rho*cp0) * dt
        freg = rparams[r]['forcing'][...,m] / (rho*cp0) * dt
        
        if reciprocal:
            
            ax.scatter(1/(dreg/hreg),freg/hreg,5,color=bbcol[rid],alpha=0.3,label=regions[rid])
        else:
            ax.scatter(dreg/hreg,freg/hreg,5,color=bbcol[rid],alpha=0.3,label=regions[rid])
        
        ax.set_title(mons3[m])
        
    if m == 0:
        ax.legend(ncol=1)
        
    ax.grid(True,ls='dotted')
    if samelims:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

#fig.text(-.02,0.39,"Forcing/MLD ($F'/ h$)",fontsize=22,rotation='vertical')
fig.text(-.02,0.30,"Atmospheric Forcing" + r" ($\frac{F'}{\rho \, c_p \, h}$" + ": K/month)",fontsize=22,rotation='vertical')
if reciprocal:
    #fig.text(0.39,-0.02,"Damping/MLD$^{-1}$ ($h/ \lambda$)",fontsize=22)
    #fig.text(0.39,-0.02,"Damping$^{-1}$ (months)",fontsize=22)
    
    fig.text(0.30,-0.02, "Damping$^{-1}$ " + r" ($\frac{\rho \, c_p \, h}{\lambda}$" + ": months)",fontsize=22)
else:
    
    #fig.text(0.39,-0.02,"Damping/MLD ($\lambda$ \h)",fontsize=22)
    fig.text(0.30,-0.02, "Damping$^{-1}$ " + r" ($\frac{\lambda}{\rho \, c_p \, h}$" + ": months)",fontsize=22)
# fig.text(-0.07, 0.55, 'latitude', va='bottom', ha='center',
#         rotation='vertical', rotation_mode='anchor',
#         transform=ax.transAxes)
# fig.text(0.5, -0.2, 'longitude', va='bottom', ha='center',
#         rotation='horizontal', rotation_mode='anchor',
#         transform=ax.transAxes)

plt.tight_layout()
plt.savefig("%sScatter_DampingvForcing.png"%(outpath),dpi=150,bbox_inches='tight')

#%% Make Mean Plot of Above

if reciprocal is not True:
    xlim = [0,1]
    ylim = [0,2]
    #xlim = [0,2]
    #ylim = [0,2]
else:
    xlim = [0,35]
    ylim = [0,1]
    

fig,ax = plt.subplots(1,1,figsize=(6,6))


for r,rid in enumerate(ridsel):
    
    hreg = rparams[r]['mld'].mean(-1)
    dreg = rparams[r]['damping'].mean(-1) / (rho*cp0) * dt
    freg = rparams[r]['forcing'].mean(-1) / (rho*cp0) * dt
    
    if reciprocal:
        
        ax.scatter(1/(dreg/hreg),freg/hreg,5,color=bbcol[rid],alpha=0.3,label=regions[rid])
    else:
        ax.scatter(dreg/hreg,freg/hreg,5,color=bbcol[rid],alpha=0.3,label=regions[rid])
    
    ax.set_title("Annual Mean Parameters")
    

ax.legend(ncol=1)
    
ax.grid(True,ls='dotted')
if samelims:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

ax.set_ylabel("Atmospheric Forcing" + r" ($\frac{F'}{\rho \, c_p \, h}$" + ": K/month)",fontsize=14,rotation='vertical')
if reciprocal:
    ax.set_xlabel("Damping$^{-1}$ " + r" ($\frac{\rho \, c_p \, h}{\lambda}$" + ": months)",fontsize=14)
else:

    ax.set_xlabel(0.30,-0.02, "Damping$^{-1}$ " + r" ($\frac{\lambda}{\rho \, c_p \, h}$" + ": months)",fontsize=22)

plt.tight_layout()
plt.savefig("%sScatter_DampingvForcing_mean.png"%(outpath),dpi=150,bbox_inches='tight')
#%% Scrap Section

plotvar = h.mean(2)



levels = 10
levels = [0,100,150,200,250,300]
levels = np.arange(0,1000,1)

fig,ax = plt.subplots(1,1)
cf = ax.contourf(lon,lat,plotvar.T,levels=levels)
cb = fig.colorbar(cf,ax=ax)







