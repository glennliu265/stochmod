#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare CESM and HadISST (Point)
(Autocorrelation and Spectral Estimates)

@author: gliu
"""

import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs
from scipy import signal

#%% User Edits

# HadISST Inforation
detrend = 2
startyr = 1900

# ERSST Information
detrende = 2
startyre = 1900

# Query Point
query = [-30,50] # [lon,lat]

# Set Path
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath     = projpath + '02_Figures/20210527/'
proc.makedir(outpath)

# Plotting INformation
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# Autocorrelation Settings
conf  = 0.95
tails = 2
lags  = np.arange(0,37,1)
xtk2  = np.arange(0,37,2)

# Spectrum options
enames   = ("HadISST","ERSST","CESM1 FULL","CESM1 SLAB")
ecolors  = ('b','r','k','gray')
plotdt   = 3600*24*365
nsmooths = [50,50,500,250]
pct      = 0.10


def calc_r1(ts):
    return np.corrcoef(ts[:-1],ts[1:])[0,1]

#%% Load Data

# Load Lat/lon
lon,lat = scm.load_latlon()

# Information for point
lonf,latf = query
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])

# Load HadISST
hsstpt = scm.load_hadisst(datpath,method=detrend,startyr=startyr,grabpoint=query)

# Load ERSST
ersstpt = scm.load_ersst(datpath,method=detrende,startyr=startyre,grabpoint=query)

# Load CESM 
fullpt,slabpt = scm.load_cesm_pt(datpath,loadname='both',grabpoint=query)

# Load Mixed Layer Depths and select point
input_path    = datpath + 'model_input/'
mld           = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
mldpt         = mld[klon,klat,:] 
kmonth        = mldpt.argmax()
#%% Calculate Autocorrelation and plot

# Calculate Autocorrelation
ssts = [hsstpt,ersstpt,fullpt,slabpt]
acs  = scm.calc_autocorr(ssts,lags,kmonth+1)
hadac = acs[0]
ersac = acs[1]
fullac = acs[2]
slabac = acs[3]

# Calculate confidence intervals
cfslab = proc.calc_conflag(slabac,conf,tails,898)
cffull = proc.calc_conflag(fullac,conf,tails,1798)
cfhad  = proc.calc_conflag(hadac,conf,tails,hsstpt.shape[0]/12)
cfers  = proc.calc_conflag(ersac,conf,tails,ersstpt.shape[0]/12)
cfs = [cfhad,cfers,cffull,cfslab]

# Plot Things
fig,ax     = plt.subplots(1,1,figsize=(6,4))
title = "SST Autocorrelation at %s (Reanalysis vs. CESM1) \n Lag 0 = %s" % (locstringtitle,mons3[kmonth])
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(3):
    #ax.plot(lags,acs[i],label=enames[i],color=ecolors[i],marker="o",markersize=3)
    ax.fill_between(lags,cfs[i][lags,0],cfs[i][lags,1],color=ecolors[i],alpha=0.20,zorder=-1)
for i in range(3):
    ax.plot(lags,acs[i],label=enames[i],color=ecolors[i],marker="o",markersize=3)
ax.legend(ncol=2)
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
plt.savefig(outpath+"Autocorrelation_CompareHadISST_%s.png" % locstring,dpi=200)


#%% Autocorrelation over a particular region


#%%  Spectral Estimate Calculation

# Calculate spectra
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(ssts,nsmooths,pct)

#%% Variance Preserving PLots

# Plot Spectra


fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(4):
    ax.semilogx(freqs[i]*plotdt,specs[i]*freqs[i],color=ecolors[i],label=enames[i]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(ssts[i])))
    ax.semilogx(freqs[i]*plotdt,CCs[i][:,1]*freqs[i],color=ecolors[i],alpha=0.5,ls='dashed')
    ax.semilogx(freqs[i]*plotdt,CCs[i][:,0]*freqs[i],color=ecolors[i],alpha=0.5,ls='dotted')

# Set x limits
xtick = ax.get_xticks()

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-lin',xtick=xtick)


xlm = [5e-4,10]
ax.set_xlim(xlm)
htax.set_xlim(xlm)
ylm = [-.01,.4]

# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtick]
htax.set_xticklabels(xtkl)

ax.set_title("SST Spectral Estimates at %s"%(locstringtitle))
plt.tight_layout()
plt.savefig(outpath+"Spectrum_VariancePres_CompareHadISST_%s.png" % locstring,dpi=200)

#%% Linear Linear PLots


# def lin_quickformat(ax,plotdt,freq):
#     # Set tickparams and clone
#     xtick = np.arange(0,1.7,.2)
#     ax.set_xticks(xtick)
#     ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
#     ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
#     htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
    
#     # Set xtick labels
#     xtkl = ["%.1f" % (1/x) for x in xtick]
#     htax.set_xticklabels(xtkl)
    
    
#     # Set some key lines
#     ax = viz.add_yrlines(ax,dt=plotdt)
    
#     ax.legend(fontsize=10)
#     return ax,htax

# Plot Spectra
fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(4):
    ax.plot(freqs[i]*plotdt,specs[i]/plotdt,color=ecolors[i],label=enames[i]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(ssts[i])))
    
    ax.plot(freqs[i]*plotdt,CCs[i][:,1]/plotdt,color=ecolors[i],alpha=0.5,ls='dashed')
    ax.plot(freqs[i]*plotdt,CCs[i][:,0]/plotdt,color=ecolors[i],alpha=0.5,ls='dotted')

    

# Set x limits
xtick = np.arange(0,1.7,.2)
ax.set_xticks(xtick)

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick)

ax = viz.add_yrlines(ax,dt=plotdt)

#ylm = [-.01,.4]
# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtick]
htax.set_xticklabels(xtkl)
ax.legend()
ax.set_title("SST Spectral Estimates at %s"%(locstringtitle))
plt.tight_layout()
plt.savefig(outpath+"Spectrum_Linear_CompareHadISST_%s.png" % locstring,dpi=200)

#%% Log Log Plots

# Plot Spectra
fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(4):
    ax.loglog(freqs[i]*plotdt,specs[i]/plotdt,color=ecolors[i],label=enames[i]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(ssts[i])))
    
    ax.loglog(freqs[i]*plotdt,CCs[i][:,1]/plotdt,color=ecolors[i],alpha=0.5,ls='dashed')
    ax.loglog(freqs[i]*plotdt,CCs[i][:,0]/plotdt,color=ecolors[i],alpha=0.5,ls='dotted')

# Set x limits\
xtick = ax.get_xticks()
xtkl = ["%.1f" % (1/x) for x in xtick]


# Set Labels
ax.set_ylabel("Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='log-log',xtick=xtick)
htax.set_xticklabels(xtkl)


#xlm = [1e-2,10]
#ax.set_xlim(xlm)
#htax.set_xlim(xlm)

#ylm = [-.01,.4]
# Set xtick labels


ax.set_title("SST Spectral Estimates at %s"%(locstringtitle))
plt.tight_layout()
plt.savefig(outpath+"Spectrum_LogLog_CompareHadISST_%s.png" % locstring,dpi=200)


#%% Reformulate R


