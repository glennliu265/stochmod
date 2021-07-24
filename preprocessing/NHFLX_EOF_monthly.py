#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate EOFs for Surface Heat Flux Anomalies


Created on Tue Jul 20 11:49:09 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm


#%%
stormtrack = 0
mconfig    = "SLAB_FULL" 



if stormtrack == 1:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
elif stormtrack == 0:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    #datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210726/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"

from amv import proc,viz
import scm


def sel_regionxr(ds,bbox):
    """
    Select region from xr.dataset with 'lon' (degrees East), 'lat'.
    Current supports crossing the prime meridian

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    bbox : TYPE
        DESCRIPTION.

    Returns
    -------
    dsreg : TYPE
        DESCRIPTION.

    """
    
    # Select lon (lon360)
    if bbox[0] > bbox[1]: # Crossing Prime Meridian
        print("Crossing Prime Meridian!")
        dsreg = ds.isel(lon =(ds.lon < bbox[1]) + (ds.lon>bbox[0]))
        dsreg = dsreg.sel(lat=slice(bbox[2],bbox[3]))
    else:
        dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    return dsreg

#%%
mconfig = "PIC_SLAB"
bbox    = [260,20,0,65]
bboxeof = [280,20,0,65]

debug = True

lonf = -30
latf = 50

# EOF parameters
N_mode = 100

# Plotting params
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
blabels=[0,0,0,0]

#%% Open the dataset

# Open the dataset
ds = xr.open_dataset("%sNHFLX_PIC_SLAB_raw.nc" % datpath)

# Apply land/ice mask
msk = np.load(lipath)
ds *= msk[None,:,:]

# Select Region
dsreg = sel_regionxr(ds,bboxeof)

# Read out data
st     = time.time()
flxglob = ds.NHFLX.values
flxreg = dsreg.NHFLX.values
lonr    = dsreg.lon.values
latr    = dsreg.lat.values
print("Loaded data in %.2fs"%(time.time()-st))

# Preprocess
ntime,nlat,nlon = flxreg.shape

# Detrend
flxa,linmod,beta,intercept = proc.detrend_dim(flxreg,0)
# Plot Spatial Map
if debug: # Appears to be positive into the ocean
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.pcolormesh(lonr,latr,flxreg[0,:,:],vmin=-500,vmax=500,cmap="RdBu_r")
    fig.colorbar(pcm,ax=ax)


# Plot Detrending
klon,klat = proc.find_latlon(lonf+360,latf,lonr,latr)
if debug: 
    fig,ax = plt.subplots(1,1)
    t = np.arange(0,ntime)
    ax.scatter(t,flxreg[:,latf,lonf],label="Raw")
    ax.plot(t,linmod[:,latf,lonf],label="Model")
    ax.scatter(t,flxa[:,latf,lonf],label="Detrended")
    ax.legend()

# Apply Area Weight
wgt = np.sqrt(np.cos(np.radians(latr)))
#plt.plot(wgt)
flxwgt = flxa * wgt[None,:,None]

# Remove NaN Points [time x npts]
flxwgt = flxwgt.reshape((ntime,nlat*nlon))
okdata,knan,okpts = proc.find_nan(flxwgt,0)
npts = okdata.shape[1]

# Calculate Monthly Anomalies, change to [yr x mon x npts]
nyr = int(ntime/12)
okdata = okdata.reshape((nyr,12,npts))
okdata = okdata - okdata.mean(0)[None,:,:]


# Prepare for eof anaylsis
eofall    = np.zeros((N_mode,12,nlat*nlon)) * np.nan
pcall     = np.zeros((N_mode,12,nyr)) * np.nan
varexpall = np.zeros((N_mode,12)) * np.nan
# Looping for each month
for m in tqdm(range(12)):
    
    # Calculate EOF
    datain = okdata[:,m,:].T # [space x time]
    eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
    
    # Standardize PCs
    pcstd = pcs / pcs.std(0)[None,:]
    
    # Regress back to dataset
    eof,b = proc.regress_2d(pcstd.T,datain.T)
    
    if debug:
        # Check to make sure both regress_2d methods are the same
        # (comparing looping by PC, and using A= [P x N])
        eof1 = np.zeros((N_mode,npts))
        b1  = np.zeros(eof1.shape)
        # Regress back to the dataset
        for n in range(N_mode):
            eof1[n,:],b1[n,:] = proc.regress_2d(pcstd[:,n],datain)
        print("max diff for eof (matrix vs loop) is %f"%(np.nanmax(np.abs(eof-eof1))))
        print("max diff for b (matrix vs loop) is %f"%(np.nanmax(np.abs(b-b1))))
        
    # Save the data
    eofall[:,m,okpts] = eof
    pcall[:,m,:] = pcs.T
    varexpall[:,m] = varexp

# Flip longitude
eofall = eofall.reshape(N_mode,12,nlat,nlon)
eofall = eofall.transpose(3,2,1,0) # [lon x lat x mon x N]
lonr180,eofall1 = proc.lon360to180(lonr,eofall.reshape(nlon,nlat,N_mode*12))
eofall = eofall.reshape(nlon,nlat,12,N_mode)

#%% Save the results
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%iEOFsPCs_%s.npz" % (datpath,N_mode,bboxtext)

np.savez(savename,**{
    "eofall":eofall,
    "pcall":pcall,
    "varexpall":varexpall,
    'lon':lonr180,
    'lat':latr},allow_pickle=True)

#%% Load the data
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%iEOFsPCs_%s.npz" % (datpath,N_mode,bboxtext)
ld = np.load(savename,allow_pickle=True)

eofall    = ld['eofall']
pcall     = ld['pcall']
varexpall = ld['varexpall']

lonr = ld['lon']
latr = ld['lat']

#%% Calculate/plot cumulative variance explained

# Calculate cumulative variance at each EOF
cvarall = np.zeros(varexpall.shape)
for i in range(N_mode):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Plot Params
N_modeplot = 10
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)


for m in range(12):
    plt.plot(modes,cvarall[:,m]*100,label="Month %i"% (m+1),marker="o",markersize=3)
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
plt.savefig("%sSLAB-PIC_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,N_modeplot,bboxtext),dpi=150)


#%% Find index of variance threshold

vthres  = 0.80
thresid = np.argmax(cvarall>vthres,axis=0)

for i in range(12):
    print("Before")
    print(cvarall[thresid[i]-1,i])
    print("After")
    print(cvarall[thresid[i],i])


ytk = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax.bar(mons3,thresid,color=[0.56,0.90,0.70],alpha=0.80)
ax.set_title("Number of EOFs required \n to explain %i"%(vthres*100)+"% of the NHFLX variance")
ax.set_yticks(ytk)
ax.set_ylabel("# EOFs")
ax.grid(True,ls='dotted')
plt.savefig("%sSLAB-PIC_NHFLX_EOFs%i_%s_NumEOFs_%ipctvar_bymon.png"%(outpath,N_mode,bboxtext,vthres*100),dpi=150)

#%% Save outptut as forcing for stochastic model

eofforce = eofall.copy()

for i in range(12):
    eofforce




#%% Do some analysis/visualization


# Plot Mode vs. % Variance Exp (for each Month)
xtk = np.arange(1,11,1)
fig,ax = plt.subplots(1,1)
for m in range(12):
    plt.plot(xtk,varexpall[:,m]*100,label="Month %i"% (m+1),marker="o")
ax.legend()
ax.set_ylabel("% Variance Explained")
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Perc. Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xticks(xtk)
plt.savefig("%sSLAB-PIC_NHFLX_EOFs_%s_ModevVariance_bymon.png"%(outpath,bboxtext),dpi=150)

# Same as above, but cumulative plot
cvarall = np.zeros(varexpall.shape)
for i in range(10):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Plot Month vs. Total % Variance Exp for 10 EOFs)
fig,ax = plt.subplots(1,1)
ax.bar(mons3,varexpall.sum(0)*100,color='cornflowerblue',alpha=0.7)
ax.set_title("Total % Variance Explained by first 10 EOFs")
ax.set_ylabel("% Variance Explained")
ax.set_ylim([0,100])
ax.set_yticks(np.arange(0,110,10))
ax.grid(True,ls='dotted')
plt.savefig("%sSLAB-PIC_NHFLX_EOFs_%s_TotalVarianceEOFs_bymon.png"%(outpath,bboxtext),dpi=150)

#%% Plot Net Heat Flux EOF Patterns

vlim = [-20,20]

bboxplot = bbox.copy()
for i in range(2):
    if bbox[i] > 180:
        bboxplot[i] -= 360

for n in tqdm(range(N_mode)):
    plotord = np.roll(np.arange(0,12,1),1)
    fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,8))
    for p,m in enumerate(plotord):
        ax = axs.flatten()[p] # Plot Index
        ax.set_title("%s (%.1f" % (mons3[m],varexpall[n,m]*100) + "%)",fontsize=10)
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabels)
        pcm = ax.pcolormesh(lonr180,latr,eofall[:,:,m,n].T,vmin=vlim[0],vmax=vlim[-1],cmap="RdBu_r")
        fig.colorbar(pcm,ax=ax,fraction=0.035)
    plt.suptitle("NHFLX EOF %i (CESM1-SLAB) (W/$m^2$ per $\sigma_{PC}$)" % (n+1),fontsize=14)
    fig.subplots_adjust(top=0.95)
    plt.savefig("%sSLAB-PIC_NHFLX_EOFs_%s_EOF%iPattern_bymon.png"%(outpath,bboxtext,n+1),dpi=150)

#%% Plot PCs and Spectra

dt  = 365*24*3600
pct = 0.10
nsmooth = 50

tsin = []
for n in range(N_mode):
    for m in range(12):
        tsin.append(pcall[n,m,:])
        
freqs = []
specs = []

specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(tsin,nsmooth,pct,opt=1,dt=dt,clvl=[.95])
