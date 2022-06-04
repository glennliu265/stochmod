#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze the F' Variable calculated from CESM1

Created on Wed May 18 17:49:05 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import time

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import proc,viz
import scm
import yo_box as ybx

#%%
datpath  = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/'
fnames   = "Fprime_PIC_%s_rolln0.nc"
mconfigs = ["FULL","SLAB"]
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220526/"
proc.makedir(figpath)

fnames2  = "CESM_proc/NHFLX_PIC_%s.nc"
#%% Load whole Dataset

i      = 1
ds     = xr.open_dataset(datpath+fnames % (mconfigs[i]))
st     = time.time()
fprime = ds.Fprime.values
times  = ds.time.values
lat    = ds.lat.values
lon    = ds.lon.values
print("Loaded in %.2fs"%(time.time()-st))

#%%
# Just Load  A Point ())

lonf = 330
latf = 50

fprimes = []
qnets   = []

for i in range(2):
    
    # Load Fprime
    ds     = xr.open_dataset(datpath+fnames % (mconfigs[i]))
    ds     = ds.sel(lon=lonf,lat=latf,method='nearest')
    st     = time.time()
    fprime = ds.Fprime.values
    times  = ds.time.values
    lat    = ds.lat.values
    lon    = ds.lon.values
    fprimes.append(fprime)
    
    ds     = xr.open_dataset(datpath+fnames2 % (mconfigs[i]))
    ds     = ds.sel(lon=lonf,lat=latf,method='nearest')
    
    # Remove the mean scycle
    ds     = ds - ds.mean('yr')
    
    # Load out the variables
    st     = time.time()
    qnet = ds.NHFLX.values.flatten()
    lat    = ds.lat.values
    lon    = ds.lon.values
    qnets.append(qnet)
    
    print("Loaded in %.2fs"%(time.time()-st))

    
#%% Anomalize (or check anomalization)


i = 1

scycle,ts_monyr = proc.calc_clim(fprimes[i],0,returnts=1)

scycleq,ts_monyr = proc.calc_clim(fprimes[i],0,returnts=1)
    
#%% Now Do Spectra Analysis

from tqdm import tqdm

def make_wn(ntime,stdev,nsamples):
    wn_ts_all = []#np.zeros((nsamples,ntime)) # [Sample][Time]
    for m in tqdm(range(nsamples)):
        
        # Make and append timeseries
        ts = np.random.normal(0,stdev,ntime)
        wn_ts_all.append(ts)
    return wn_ts_all


def mc_wn_spectra(ts,nsamples,nsmooth,pct,return_pcts="ALL",tails="both"):
    
    # Make the white noise timeseries
    stdev     = np.std(ts)
    ntime     = len(ts)
    wn_ts_all = make_wn(ntime,stdev,nsamples)
    
    # Compute the spectra
    specs,freqs,_,_,_=scm.quick_spectrum(wn_ts_all,nsmooth,pct)
    
    # Calculate the spectra
    specs = np.array(specs) # [sample, freq]
    freq  = freqs[0]
    
    # Sort by order and select top N %
    sortspec = np.sort(specs,axis=0)
    if isinstance(return_pcts,str): # Return all values
        return sortspec,freq
    else:
        nthres = len(return_pcts)
        cfspec = np.zeros([2,nthres,len(freq)]) # [lower(0)/upper(1) x thres x freq]
        for i in range(2):
            if i == 0: #Lower
                pcnts = np.array(return_pcts)
            elif i == 1: #Upper
                pcnts = 1 - np.array(return_pcts)
            # Select rank
            idsel         = np.floor(pcnts*nsamples).astype(int)
            selection     = sortspec[idsel,:]
            cfspec[i,:,:] = selection
        if tails == 'upper':
            cfspec = cfspec[1,...]
        if tails == 'lower':
            cfspec = cfspec[0,...]
        return cfspec,freq


# Calculate Spectra
nsmooth = 100
pct     = 0.10
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(fprimes,nsmooth,pct)
dtplot  = 3600*24*365

qspecs,qfreqs,qCCs,qdofs,qr1s = scm.quick_spectrum(qnets,nsmooth,pct)


#%% Compute White Noise confidence
nsamples = 1000

pcts     = [0.025,0.05,0.5]
cfspecs = []
for i in tqdm(range(2)):
    cfspec,freq = mc_wn_spectra(fprimes[i],nsamples,nsmooth,pct,return_pcts=pcts,tails="both")
    cfspecs.append(cfspec)
    
qcfspecs = []
for i in tqdm(range(2)):
    cfspec,freq = mc_wn_spectra(qnets[i],nsamples,nsmooth,pct,return_pcts=pcts,tails="both")
    qcfspecs.append(cfspec)
    
    
    

#%%

# Settings
xper    = np.array([100,20,10,5,2])
xtk     = 1/xper
xlm     = (xtk[0],xtk[-1])

pctid   = 0

pcolors  = ["red","orange",]
pcolors2 = ["blue","violet"]

twosp = True

# Plot Results
if twosp:
    fig,axs = plt.subplots(1,2,figsize=(16,4),constrained_layout=True,sharey=True)
else:
    fig,ax = plt.subplots(1,1)
    
for mc in range(2):
    
    if twosp:
        ax = axs[mc]
    
    ax.plot(freqs[mc]*dtplot,specs[mc]/dtplot,label="F' " + mconfigs[mc],color=pcolors[mc],zorder=1)
    ax.plot(qfreqs[mc]*dtplot,qspecs[mc]/dtplot,label="$Q'_{net}$ "+mconfigs[mc],color=pcolors2[mc],zorder=1)
    
    
    for q in range(2):
        if q == 0:
            incfspecs = cfspecs
            infreqs   = freqs
            incolors  = pcolors
        if q == 1:
            incfspecs = qcfspecs
            infreqs   = qfreqs
            incolors  = pcolors2
            
        # Plot the confidence
        for i in [0,2]:
            
            if i == 0:
                ls = 'dotted'
            elif i == 1:
                ls = 'dashed'
            elif i == 2:
                ls = 'solid'
                
            if i < 2:
                ax.fill_between(infreqs[mc]*dtplot,incfspecs[mc][0,i,:]/dtplot,
                                incfspecs[mc][1,i,:]/dtplot,alpha=0.05,color=incolors[mc],zorder=9)
            
            ax.plot(infreqs[mc]*dtplot,incfspecs[mc][0,i,:]/dtplot,color=incolors[mc],ls=ls,alpha=0.3)
            ax.plot(infreqs[mc]*dtplot,incfspecs[mc][1,i,:]/dtplot,color=incolors[mc],ls=ls,alpha=0.3)


if twosp:
    loopax = axs
else:
    loopax = [ax,]
for a,ax in enumerate(loopax):
    ax.legend()
    #ax.set_xticks(xtk)
    ax.set_xlim(xlm)
    ax2 = ax.twiny()
    ax2.set_xticks(xtk)
    ax2.set_xlim(xlm)
    ax2.set_xticklabels(xper)
    ax2.grid(True,ls='dotted')
    if a == 0:
        ax.set_ylabel("Power ($Wm^{-1}cpy^{-1}$)")
    ax2.set_xlabel("Period (Years)")
    ax.set_xlabel("Frequency ($Years^{-1}$)")
    
    
    
plt.savefig("%sFprime_Qnet_Spectra_Comparison_twosp%i.png" % (figpath,twosp),dpi=150)

#%% Make two subplots to be ore comparable


#%%

#viz.plot_freqlin()
