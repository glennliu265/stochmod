#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Examine ENSO-removal in the Tropical Atlantic


"Was there incomplete removal of ENSO in the CESM-FULL simulations?"

Created on Thu Feb 24 21:37:26 2022

@author: gliu
"""

import sys
import cartopy.crs as ccrsp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import sys
import cmocean
from tqdm import tqdm

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220225/"
   
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

proc.makedir(figpath)

#%% Set path to the data


mconfigs = ["SLAB","FULL"]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
#%% PART 1: Compare ENSO Component in FULL/SLAB

vname      = "TS"
units      = "$K \sigma_{ENSO}^{-1}$"
vname_long = "SST"

ensopats = []
for m,mconfig in enumerate(mconfigs):
    ensofile = datpath + "../ENSO/%s_PIC_ensocomp/ENSOREM_%s_lag1_pcs2_monwin3_ensocomponent.npz" % (mconfig,vname)
    ld       = np.load(ensofile,allow_pickle=True)
    ensopats.append(ld['ensopattern'])
    lon = ld['lon']
    lat = ld['lat']
#%% Plot for a single month/mode/mconfig

im   = 0
N    = 0
mid  = 1

bboxplot  = [-100,10,-1,55] 
cints     = np.arange(-.3,.325,.025)
plotvar = ensopats[mid][im,:,:,N]
cblab = "%s (%s)" % (vname_long,units)
ptitle = "CESM-%s %s ENSO Pattern (EOF %i)" % (mconfigs[mid],mons3[im],N+1)

def plot_enso(lon,lat,plotvar,title,cblab,ax=None
              ,bboxplot=[-100,10,-1,55],
              cints=np.arange(-.3,.325,.025),
              plot_cb=True):
    if ax is None:
        ax = plt.gca()
    ax      = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm     = ax.contourf(lon,lat,plotvar,levels=cints,cmap='cmo.balance',extend='both')
    ax.set_title(title)
    if plot_cb:
        cb      = fig.colorbar(pcm,ax=ax)
        cb.set_label(cblab)
    return ax,pcm
    

fig,ax  = plt.subplots(1,1,constrained_layout=True,subplot_kw={'projection':ccrs.PlateCarree()})
ax,pcm      = plot_enso(lon,lat,plotvar,ptitle,cblab,ax=ax,bboxplot=bboxplot,cints=cints,plot_cb=True)



#%% Monthly Plots

loopm = np.concatenate([[11,],np.arange(0,11,1)])



for mid in range(2):
    for N in range(2):
        fig,axs  = plt.subplots(4,3,constrained_layout=True,figsize=(12,12),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        for i in tqdm(range(12)):
            
            im      = loopm[i]
            ax      = axs.flatten()[i]
            plotvar = ensopats[mid][im,:,:,N]
            ptitle  = "%s" % (mons3[im])#@"CESM-%s %s ENSO Pattern (EOF %i)" % (mconfigs[mid],mons3[im],N+1)
            ax,pcm  = plot_enso(lon,lat,plotvar,ptitle,cblab,ax=ax,bboxplot=bboxplot,cints=cints,plot_cb=False)
        plt.suptitle("ENSO Pattern (%s, EOF %i)"% (mconfigs[mid],N+1))
        savename = "%sENSO_Pattern_Monthly_%s_EOF%i" % (figpath,mconfigs[mid],N+1)
        plt.savefig(savename,dpi=150)


# Plot the colorbar
ornt = 'v'
if ornt == 'h':
    fig,ax = plt.subplots(1,1,figsize=(8,4),constrained_layout=True)
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')
elif ornt == 'v':
    fig,ax = plt.subplots(1,1,figsize=(4,8),constrained_layout=True)
    cb = fig.colorbar(pcm,ax=ax,orientation='vertical')
cb.set_label(cblab)
savename = "%sENSO_Pattern_Monthly_colorbar.png" % (figpath)
plt.savefig(savename,dpi=150)

#%% Now check the TS variable itself
# ----------------------------------

# name of TS file with ENSO removed
noenso_name = datpath + "../ENSO/ENSOREM_TS_lag1_pcs2_monwin3_FULL_PIC.npz"
ld          = np.load(noenso_name,allow_pickle=True)
ts_noenso   = ld['TS']
nolon       = ld['lon']
nolat       = ld['lat']
ntime,nlat,nlon =ts_noenso.shape
tsa_noenso   = ts_noenso.reshape((int(ntime/12),12,nlat,nlon)) # Already anomalized

# Load data with ENSO present
enso_name   = datpath + "../CESM_proc/TS_PIC_FULL.nc"
ds          = xr.open_dataset(enso_name)
ts_enso     = ds.TS.values
nyr,_,nlat,nlon = ts_enso.shape
tsa_enso    = ts_enso - ts_enso.mean(0)[None,...] # Deseason

#%% Let's check a particular point
lonf      = -55+360
latf      = 11
# lonf      = -32+360
# latf      = 14
# lonf = -30+360
# latf      = 60
klon,klat = proc.find_latlon(lonf,latf,nolon,nolat)
flocstring = "lon%i_lat%i" % (lonf,latf)
locstring = "%i$\degree$N, %i$\degree$W" % (latf,np.abs(lonf))

# Get values
tsnames  = ["ENSO Removed","Raw"]
tspts    = [tsa_noenso[...,klat,klon].flatten(),tsa_enso[...,klat,klon].flatten()]
tscolors = ["red","black"]

# Lets try computing the spectra
nsmooth = 50
pct     = 0.10
dtplot  = 3600*24*365 
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(tspts,nsmooth,pct)

xtks = [1/50,1/25,1/10,1/5,1/2,1/1]
xper = [int(1/x) for x in xtks]
fig,ax = plt.subplots(1,1,figsize=(10,3))
for i in range(2):
    ax.plot(freqs[i]*dtplot,specs[i]/dtplot,label=tsnames[i],color=tscolors[i])
    ax.plot(freqs[i]*dtplot,CCs[i][:,1]/dtplot,label="",color=tscolors[i],ls='dotted',alpha=1)
ax.legend()
#ax.set_xticks(xtks)
#ax.set_xticklabels(xper)
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xlabel("Period (Years)")
ax.set_ylabel("Power ($K^2 cpy^{-1}$)")
ax.grid(True)

ax2 = ax.twiny()
ax2.set_xlim([xtks[0],xtks[-1]])
ax2.set_xticks(xtks)
ax2.set_xticklabels(xper)
ax2.grid(True,ls='dotted',color='gray')
ax2.set_xlabel("Period (Years)")
ax.set_title("SST Spectrum for CESM-FULL @ %s"%locstring)
plt.savefig("%sSST_Spectra_ENSO_Removal_%s.png"%(figpath,flocstring),dpi=100,bbox_inches='tight')

#% Compute the Autocorrelation
lags      = np.arange(0,37)
xtk2      = lags[::2]
basemonth = 2 # (not the index but actual month!!)
autocorr,confs = scm.calc_autocorr(tspts,lags,basemonth,calc_conf=True)

fig,ax = plt.subplots(1,1,figsize=(10,3))
ptitle="%s SST  Autocorrelation for CESM-%s @ %s" % (mons3[basemonth-1],mconfigs[mid],locstring)
ax,ax2 = viz.init_acplot(basemonth-1,xtk2,lags,ax=ax,title=ptitle)

for i in range(2):
    ax.plot(lags,autocorr[i],label=tsnames[i],color=tscolors[i])
    ax.fill_between(lags,confs[i][:,0],confs[i][:,1],color=tscolors[i],alpha=0.25)

ax.legend()
plt.savefig("%sSST_Autocorrelation_ENSO_Removal_%s.png"% (figpath,flocstring),dpi=100)



