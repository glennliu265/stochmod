#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Model Pointwise Analysis

Created on Tue Aug 10 16:38:51 2021

Takes output of sm_rewrite.py and visualizes some relevant quantities at a given point

- Calculates Autocorrelation for each model, compares with CESM
- Spectral Analysis, testing sensitivity to smoothing

@author: gliu

"""
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import time
import time
from tqdm import tqdm
#%% Set Paths

stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210810/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

#%% Settings Part 1 (Comparison)


## Compare Effect of including ekman advection
# Names of the experiments
# expids    = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
#              "stoch_output_forcingflxeof_qek_50eofs_SLAB-PIC_1000yr_run003.npz"
#             ]
# exnames   = ["SM (no $Q_{ek}$)",
#              "SM (with $Q_{ek}$)"
#             ]
# excolors  = ["blue",
#               "orange"]
#exoutnameraw = "Q-ek_comparison_50EOF"

## Compare before/after correctio,n and inclusion of Amp
expids = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run005.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz"
          ]
exnames = ["Before (Incl. Lat-weight)",
           "After (no q-corr)",
           "After (with q-corr)",
           "90% Variance Threshold (no-qcorr)"]
excolors = ["blue",
            "orange",
            "magenta",
            "red"]
exoutnameraw = "latweightfix_qamp_comparison_50EOF"

## Examine zffect of including AMP using variance-based threshold
expids = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run007.npz"]
exnames = ["no q-corr",
           "with q-corr (monthly)",
           "with q-corr (avg)"]
excolors = ["blue",
            "orange",
            "magenta"]
exoutnameraw = "90pctvariance_qamp_comparison"
#%% Settings Part 2
# CESM Names
cnames    = ["CESM-FULL","CESM-SLAB",]
ccolors   = ["black","gray"]
cpath     = datpath + "../"

# Combine Names
enames    = np.hstack([cnames,exnames])
ecolors   = np.hstack([ccolors,excolors])

# Point to find
lonf      = -30
latf      = 50
locstring = "Lon: %i; Lat: %i" % (lonf,latf)
locfn     = "lon%i_lat%i" % (lonf,latf)

# Check point
fig,ax    = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
ax        = viz.add_coast_grid(ax)
ax.scatter(lonf,latf,marker="x",color='r')
ax.set_title("X is %s" % locstring)

# Load lat/lon
lon180,lat  = scm.load_latlon(rawpath)
lonr        = np.load(datpath+"lon.npy")
latr        = np.load(datpath+"lat.npy")
klonr,klatr = proc.find_latlon(lonf,latf,lonr,latr)
klon,klat   = proc.find_latlon(lonf,latf,lon180,lat)

# Model to conduct analysis on
model       = 0
modelnames  = ["h constant","h vary","entraining"]
exoutname  = exoutnameraw +  "_model%i" % model
# Autocorrelation parameters
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

#%% Load CESM and Stochastic model data

st   = time.time()
ssts = scm.load_cesm_pt(cpath,grabpoint=[lonf,latf])

for i in tqdm(range(len(expids))):
    ld  = np.load(datpath+expids[i])
    sst = ld['sst']
    ssts.append(sst[model,klonr,klatr,:])

print("Loaded data in %.2fs"%(time.time()-st))


#%% Load mixed layer depths to identify the base month

mld    = np.load(rawpath + "FULL_HTR_HMXL_hclim.npy")
kmonth = np.argmax(mld[klon,klat]) 

#%% Autocorrelation Analysis

# Parameter Choices
# -----------------
lags        = np.arange(0,37,1)
conf        = 0.95
tails       = 2
xtk2        = np.arange(0,37,2)


# Calculate Autocorrelation
acs    = scm.calc_autocorr(ssts,lags,kmonth+1)

# Calculate autocorrelation and confidence intervals
nlags   = len(lags)
cfac = np.zeros([len(enames),nlags,2])
for m in range(len(enames)):
    inac = acs[m]
    n = int(len(ssts[m])/12)
    cfs = proc.calc_conflag(inac,conf,tails,n)
    cfac[m,:,:] = cfs

# Plut Autocorrelation
fig,ax     = plt.subplots(1,1,figsize=(6,4))
title      = "SST Autocorrelation (Lag 0 = %s)" % (mons3[kmonth])
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(len(enames)):
    ax.plot(lags,acs[i],label=enames[i],color=ecolors[i])
    ax.fill_between(lags,cfac[i,:,0],cfac[i,:,1],color=ecolors[i],alpha=0.15)
ax.legend()
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(figpath+"Autocorrelation_%s.png"%(exoutname),dpi=200)

#%% Spectra Analysis (Test Smoothing)

# Parameter Choices
# -----------------
pct      = 0
nsmooths = [1,10,25,50,100,500]
plotdt   = 365*3600*24 # Annual Intervals
plotconf = True

# Estimate spectra with different smoothing
outparams = []
for n in tqdm(range(len(nsmooths))):
    nsmooth = np.ones(len(enames))*nsmooths[n]
    
    # Calculate Spectra
    params=scm.quick_spectrum(ssts,nsmooth,pct)
    #specs,freqs,CCs,dofs,r1s=params
    
    outparams.append(params)

# Full Range Plot (Linear-Linear)
# -------------------------------
xtick    = [1/100,1/10,1/2,1,2,4]
xlm      = [5e-3,5]
fig,axs = plt.subplots(2,3,figsize=(20,12))
for n in range(len(nsmooths)):
    
    # Retrieve information
    ax = axs.flatten()[n]
    nsmooth = nsmooths[n]
    specs,freqs,CCs,dofs,r1s=outparams[n]
    
    # Plot spectra
    for m in range(len(enames)):
        ax.plot(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])
    
        if plotconf:
             ax.plot(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")
    
    # Set Labels
    ax.set_ylabel("Power ($(degC)^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    ax.legend(fontsize=10,ncol=2)
    ax.set_title("Smoothing %i bands" % nsmooth,fontsize=20)
savename = "%sSST_Spectra_Smoothingtest_%s.png" % (figpath,exoutname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# Decadal Plot (Linear-Linear)
# -------------------------------
xtick    = np.array([0,0.02,0.04,0.1,0.2])
xtkl     =  1/np.array(xtick)
xlm      = [0,0.2]
fig,axs  = plt.subplots(2,3,figsize=(20,12))
for n in range(len(nsmooths)):
    
    # Retrieve information
    ax = axs.flatten()[n]
    nsmooth = nsmooths[n]
    specs,freqs,CCs,dofs,r1s=outparams[n]
    
    # Plot spectra
    for m in range(len(enames)):
        ax.plot(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])
    
        if plotconf:
             ax.plot(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")
    
    # Set Labels
    ax.set_ylabel("Power ($(degC)^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = xtkl
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    ax.legend(fontsize=10,ncol=2)
    ax.set_title("Smoothing %i bands" % nsmooth,fontsize=20)
savename = "%sSST_Spectra_Smoothingtest_%s_decadal-lin-lin.png" % (figpath,exoutname)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# Freq x Power (Linear-Log, full range)
# --------------------------------------------
xtick  = [float(10)**(x) for x in np.arange(-4,2)]
xlm    = [5e-4,10]
fig,axs  = plt.subplots(2,3,figsize=(20,12))
for n in range(len(nsmooths)):
    
    # Retrieve information
    ax = axs.flatten()[n]
    nsmooth = nsmooths[n]
    specs,freqs,CCs,dofs,r1s=outparams[n]
    
    # Plot spectra
    for m in range(len(enames)):
        ax.semilogx(freqs[m]*plotdt,specs[m]*freqs[m],color=ecolors[m],label=enames[m])
    
        if plotconf:
             ax.semilogx(freqs[m]*plotdt,CCs[m][:,1]*freqs[m],label="",color=ecolors[m],ls="dashed")
    
    # Set Labels
    ax.set_ylabel("Frequency x Power ($(degC)^{2}$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    ax.legend(fontsize=10,ncol=2)
    ax.set_title("Smoothing %i bands" % nsmooth,fontsize=20)
savename = "%sSST_Spectra_Smoothingtest_%s_freqxpower_log-lin.png" % (figpath,exoutname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# Log x Log Plot (full range)
# -------------
xtick  = [float(10)**(x) for x in np.arange(-4,2)]
xlm    = [5e-4,10]
fig,axs  = plt.subplots(2,3,figsize=(20,12))
for n in range(len(nsmooths)):
    
    # Retrieve information
    ax = axs.flatten()[n]
    nsmooth = nsmooths[n]
    specs,freqs,CCs,dofs,r1s=outparams[n]
    
    # Plot spectra
    for m in range(len(enames)):
        ax.loglog(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])
    
        if plotconf:
             ax.loglog(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")
    
    # Set Labels
    ax.set_ylabel("Frequency x Power ($(degC)^{2}$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick,include_title=False)
    
    # Set upper x-axis ticks
    xtick2 = htax.get_xticks()
    xtkl   = ["%.1f" % (1/x) for x in xtick2]
    htax.set_xticklabels(xtkl)
    
    # Set axis limits
    ax.set_xlim(xlm)
    htax.set_xlim(xlm)  
    ax.legend(fontsize=10,ncol=2)
    ax.set_title("Smoothing %i bands" % nsmooth,fontsize=20)
savename = "%sSST_Spectra_Smoothingtest_%s_freqxpower_log-log.png" % (figpath,exoutname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Plot for single smoothing option

# Parameter Choices
# -----------------
pct      = 0.10
nsmooth  = 10
plotdt   = 365*3600*24 # Annual Intervals
plotconf = True
specname = "nsmooth%i_taper%03i" % (nsmooth,pct*100)


# Estimate spectra
params=scm.quick_spectrum(ssts,nsmooth,pct)
specs,freqs,CCs,dofs,r1s=params


# Full Range Plot (Linear-Linear)
# -------------------------------
xtick    = [1/100,1/10,1/2,1,2,4]
xlm      = [5e-3,5]
fig,ax = plt.subplots(1,1,figsize=(6,4))
# Plot spectra
for m in range(len(enames)):
    ax.plot(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])
    if plotconf:
         ax.plot(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")
# Set Labels
ax.set_ylabel("Power ($(degC)^{2} / cpy$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick,include_title=False)
# Set upper x-axis ticks
xtick2 = htax.get_xticks()
xtkl   = ["%.1f" % (1/x) for x in xtick2]
htax.set_xticklabels(xtkl)
# Set axis limits
ax.set_xlim(xlm)
htax.set_xlim(xlm)  
ax.legend(fontsize=10,ncol=2)
ax.set_title("SST Spectra for %s Stochastic Model (%s) \n smoothing = %i bands, taper = %.2d " % (modelnames[model],locstring,nsmooth,pct),fontsize=14)
savename = "%sSST_Spectra_Smoothingtest_%s_%s.png" % (figpath,exoutname,specname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# Decadal Plot (Linear-Linear)
# ----------------------------
xtick    = np.array([0,0.02,0.04,0.1,0.2])
xtkl     =  1/np.array(xtick)
xlm      = [0,0.2]
fig,ax = plt.subplots(1,1,figsize=(6,4))
# Plot spectra
for m in range(len(enames)):
    ax.plot(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])

    if plotconf:
         ax.plot(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")

# Set Labels
ax.set_ylabel("Power ($(degC)^{2} / cpy$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick,include_title=False)

# Set upper x-axis ticks
xtick2 = htax.get_xticks()
xtkl   = xtkl
htax.set_xticklabels(xtkl)

# Set axis limits
ax.set_xlim(xlm)
htax.set_xlim(xlm)  
ax.legend(fontsize=10,ncol=2)
ax.set_title("SST Spectra for %s Stochastic Model (%s) \n smoothing = %i bands, taper = %.2d " % (modelnames[model],locstring,nsmooth,pct),fontsize=14)
savename = "%sSST_Spectra_Smoothingtest_%s_decadal-lin-lin_%s.png" % (figpath,exoutname,specname)
plt.savefig(savename,dpi=150,bbox_inches='tight')



# Freq x Power (Linear-Log, full range)
# -------------------------------------
xtick  = [float(10)**(x) for x in np.arange(-4,2)]
xlm    = [5e-4,6]
#xlm    = [5e-3,1]
fig,ax = plt.subplots(1,1,figsize=(6,4))
# Plot spectra
for m in range(len(enames)):
    ax.semilogx(freqs[m]*plotdt,specs[m]*freqs[m],color=ecolors[m],label=enames[m])

    if plotconf:
         ax.semilogx(freqs[m]*plotdt,CCs[m][:,1]*freqs[m],label="",color=ecolors[m],ls="dashed")

# Set Labels
ax.set_ylabel("Frequency x Power ($(degC)^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick,include_title=False)

# Set upper x-axis ticks
xtick2 = htax.get_xticks()
xtkl   = ["%.1f" % (1/x) for x in xtick2]
htax.set_xticklabels(xtkl)

# Set axis limits
ax.set_xlim(xlm)
htax.set_xlim(xlm)  
ax.legend(fontsize=10,ncol=2)
ax.set_title("SST Spectra for %s Stochastic Model (%s) \n smoothing = %i bands, taper = %03i " % (modelnames[model],locstring,nsmooth,pct*100),fontsize=14)
savename = "%sSST_Spectra_Smoothingtest_%s_freqxpower_log-lin_%s.png" % (figpath,exoutname,specname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


# Log x Log Plot (full range)
# ---------------------------
xtick  = [float(10)**(x) for x in np.arange(-4,2)]
xlm    = [5e-4,10]
fig,ax = plt.subplots(1,1,figsize=(6,4))
# Plot spectra
for m in range(len(enames)):
    ax.loglog(freqs[m]*plotdt,specs[m]/plotdt,color=ecolors[m],label=enames[m])

    if plotconf:
         ax.loglog(freqs[m]*plotdt,CCs[m][:,1]/plotdt,label="",color=ecolors[m],ls="dashed")
# Set Labels
ax.set_ylabel("Frequency x Power ($(degC)^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick,include_title=False)

# Set upper x-axis ticks
xtick2 = htax.get_xticks()
xtkl   = ["%.1f" % (1/x) for x in xtick2]
htax.set_xticklabels(xtkl)

# Set axis limits
ax.set_xlim(xlm)
htax.set_xlim(xlm)  
ax.legend(fontsize=10,ncol=2)
ax.set_title("SST Spectra for %s Stochastic Model (%s) \n smoothing = %i bands, taper = %03i " % (modelnames[model],locstring,nsmooth,pct*100),fontsize=14)
savename = "%sSST_Spectra_Smoothingtest_%s_freqxpower_log-log_%s.png" % (figpath,exoutname,specname)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% State the variance of each point

for e in range(len(enames)):
    print("Variance of %s is %f (degC)^2" % (enames[e],np.var(ssts[e])))

#%% Plot parameters for that point

ld = np.load(rawpath+"")






