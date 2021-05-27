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
startyr = 1870

# Query Point
query = [-30,50] # [lon,lat]

# Set Paths
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

# Load CESM 
fullpt,slabpt = scm.load_cesm_pt(datpath,loadname='both',grabpoint=query)

# Load Mixed Layer Depths and select point
input_path    = datpath + 'model_input/'
mld           = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
mldpt         = mld[klon,klat,:] 
kmonth        = mldpt.argmax()
#%% Calculate Autocorrelation and plot

# Calculate Autocorrelation
ssts = [hsstpt,fullpt,slabpt]
acs  = scm.calc_autocorr(ssts,lags,kmonth+1)
hadac = acs[0]
fullac = acs[1]
slabac = acs[2]

# Calculate confidence intervals
cfslab = proc.calc_conflag(slabac,conf,tails,898)
cffull = proc.calc_conflag(fullac,conf,tails,1798)
cfhad  = proc.calc_conflag(hadac,conf,tails,hsstpt.shape[0]/12)

# Plot Things
fig,ax     = plt.subplots(1,1,figsize=(6,4))
title = "SST Autocorrelation at %s \n Lag 0 = %s" % (locstringtitle,mons3[kmonth])
#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)


# Plot CESM Data
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,slabac[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=3)
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,fullac,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

# Plot HadISST Data
ax.plot(lags,hadac,label="HadISST",color="b",marker="x",markersize=3)
ax.fill_between(lags,cfhad[:,0],cfhad[:,1],color='b',alpha=0.10)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
plt.savefig(outpath+"Autocorrelation_CompareHadISST_%s.png" % locstring,dpi=200)

#%% Calculate Autospectrum




nsmooths = [50,500,250]
pct      = 0.10

# Calculate spectra
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(ssts,nsmooths,pct)

# Reassign some names
CLs = CCs[1:]



# Plot Spectra
plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))


enames=("HadISST","CESM1 FULL","CESM1 SLAB")
ecolors = ('b','k','gray')

for i in range(3):
    ax.semilogx(freqs[i]*plotdt,specs[i]*freqs[i],color=ecolors[i],label=enames[i]+"$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(ssts[i])))
    
    ax.semilogx(freqs[i]*plotdt,CCs[i][:,1]*freqs[i],color=ecolors[i],alpha=0.5,ls='dashed')
    ax.semilogx(freqs[i]*plotdt,CCs[i][:,0]*freqs[i],color=ecolors[i],alpha=0.5,ls='dotted')

    

# Set x limits
#xlm = [1/(plotdt*),1/(plotdt*1)]

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

ax.set_xlabel("")
ax.set_title("SST Spectral Estimates (CESM1 vs. HadISST)")
plt.tight_layout()


    
    
    




