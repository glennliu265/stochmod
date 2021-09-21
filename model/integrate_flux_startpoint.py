#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 02:56:15 2021

@author: gliu
"""


import numpy as np
import xarray as xr
from tqdm import tqdm
import time
import cmocean
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.io import loadmat

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
import tbx
import scm
from scipy import signal
#%% User Edits
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210225/"

debug     = True
useanom   = False
pointmode = True
lonf     = -30
latf     = 50
rho      = 1026
cp0      = 3996
dt       = 3600*24*30
quadrature = False
lags       = np.arange(0,37,1)
mons3      = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

# testlam = array([169.94615638, 174.40679466, 187.23575848, 188.840277  ,
#        190.71626866, 183.75317341, 177.30749037, 176.80800517,
#        165.06422115, 169.41851357, 178.10587955, 159.28876965])



cmbal = cmocean.cm.balance
#%% Load MLD, TS, Fnet

# ----------------
# Load Fnet and TS
# ----------------
if useanom:
    # NHFLX, LON x LAT x TIME
    st = time.time()
    dsflx = xr.open_dataset(datpath+"NHFLX_PIC.nc") # [288 x 192 x 10766], lon180
    flx = dsflx.NHFLX.values
    lon = dsflx.lon.values
    lat = dsflx.lat.values
    print("Loaded NHFLX in %.2fs"% (time.time()-st))
    
    
    #%Load in TS from slab
    #fn1 = datpath + "../TS_SLAB_withENSO.npy"
    fn2 = datpath + "../ENSOREM_TS_lag1_pcs2_monwin3.npz"
    #tsenso1 = np.load(fn1)
    ld2 = np.load(fn2)
    tsenso0 = ld2['TS']
    lon360  = ld2['lon']
    lat     = ld2['lat']
    
    # Remap lon360 --> 180
    nmon,nlat,nlon = tsenso0.shape
    tsenso0 = tsenso0.transpose(2,1,0) # lon x lat x mon
    lon1,tsenso180 = proc.lon360to180(lon360,tsenso0) # Flip longitudes

else:
    # NHFLX, LON x LAT x TIME
    st = time.time()
    dsflx = xr.open_dataset(datpath+"NHFLX_PIC_SLAB_raw.nc") # [288 x 192 x 10766], lon180
    flx = dsflx.NHFLX.values # [10812, 192, 288]
    lon360 = dsflx.lon.values
    lat = dsflx.lat.values
    
    # Remap lon360 --> 180
    flx = flx.transpose(2,1,0)
    lon1,flx = proc.lon360to180(lon360,flx)
    print("Loaded NHFLX in %.2fs"% (time.time()-st))
    
    # Load TS
    st     = time.time()
    dssst  = xr.open_dataset(datpath+"TS_PIC_SLAB_raw.nc") # [288 x 192 x 10766], lon180
    ts     = dssst.TS.values # [10812, 192, 288]
    
    # Remap lon360 --> 180
    ts = ts.transpose(2,1,0)
    lon1,ts = proc.lon360to180(lon360,ts)
    print("Loaded TS in %.2fs"% (time.time()-st))

# -----------------------
# Load Mixed Layer Depths
# -----------------------
st = time.time()
dsmld = xr.open_dataset(datpath+"HMXL_PIC.nc")
mld = dsmld.HMXL.values/100 # Convert to meters
lon = dsmld.lon.values
lat = dsmld.lat.values
print("Loaded MLD in %.2fs"% (time.time()-st))

# -----------------------
# Close Datasets
# -----------------------
dsmld.close()
dsflx.close()
dssst.close()

# -------------------------------
#%% Load Qcorrection and Slab MLD
# -------------------------------

hblt  = np.load(datpath+"../SLAB_PIC_hblt.npy")
qdp   = np.load(datpath+"../SLAB_PIC_qdp.npy") 

qdpa = -1*(qdp -qdp.mean(2)[:,:,None])

# --------------------------------------------------------------
#%% Load in Snow Contribution for single point
# --------------------------------------------------------------


# Load Data
precsc = np.load(datpath+"../PRECSC_PIC_SLAB_lon330_lat50.npy")
precsl = np.load(datpath+"../PRECSL_PIC_SLAB_lon330_lat50.npy")


# Convert to W/m2
lhf_snow = 3.33e5 # J/kg, Latent heat of fusion of snow
rho_fw   = 1e3 # kg/m3, Density of Fresh Water
precsc = precsc * (lhf_snow*rho_fw)
precsl = precsl * (lhf_snow*rho_fw)

fig,axs=plt.subplots(2,1)
ax = axs[0]
ax.plot(mons3,precsc.reshape(901,12).T,label="Convective",color='k',alpha=0.3)
ax.set_title("Convective snow rate (water equivalent) (W/m2)")
ax = axs[1]
ax.plot(mons3,precsl.reshape(901,12).T,label="Large Scale",color='r',alpha=0.3)
ax.set_title("Large-scale (stable) snow rate (water equivalent) (W/m2)")
plt.tight_layout()
plt.savefig(outfigpath+"SNOW_Contribution_50N_330E.png",dpi=200)
#ax.legend()


# Plot cumulative contribution
acontrib = (precsc+precsl).reshape(901,12)
mcontrib = acontrib.mean(0)


xtklsnow = ["%s \n (%.2f) " % (mons3[i],mcontrib[i]) for i in range(12)]

fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.plot(mons3,acontrib.T,alpha=0.25,color='k')
ax.plot(mons3,acontrib.mean(0),color='r',label="Mean")
ax.legend()
ax.grid(True,ls='dotted')
ax.set_xticklabels(xtklsnow)
ax.set_title(r"$Q_{snow}$")
ax.set_ylabel("W/m2")
plt.savefig(outfigpath+"MeanContrib.png",dpi=200)

print(acontrib.mean(0))

snowcorr =acontrib


#%%% Integrate the model

klon,klat = proc.find_latlon(lonf,latf,lon1,lat)
loctitle  = "Lon %.2f Lat %.2f" % (lon1[klon],lat[klat])
locfn     = "lon%i_lat%i" % (lonf,latf)
print("Running model for %.2f lon,  %.2f lat" % (lon[klon],lat[klat]))


# Get point for inputs
flxin   = flx[klon,klat,:] # [-180,180]
qdpin   = -qdp[klon,klat,:] # [-180,180]
mldin   = hblt.mean(2)[klon,klat] # [-180,180]
tsin    = ts[klon,klat,:] #[-180,180]
snowin  = precsl+precsc

def run_model(flxin,qdpin,mldin,tsin,snowin,klon,klat,startyr,endyr,
              dt,rho,cp0):
    
    # Get starting indices
    kstart  = 12*startyr
    kend    = 12*endyr - 1
    tsteps  = (kend - kstart)+1 # Get number of timesteps
    
    # Preallocate
    sst     = np.zeros(tsteps) 
    sst[0]  = tsin[kstart]
    
    for t in tqdm(range(1,tsteps)): # Integrate starting from Feb
        m = (t)%12
        
        sst[t] = sst[t-1]  + (flxin[t] + qdpin[m] - snowin[t]) * (dt / (rho*cp0*mldin))
    
    return sst,tsin[kstart:kend+1]

#%% Testing Integration ------ (test for 1 start yr)


    
startyr = 100
endyr = 901
sst,tsin = run_model(flxin,qdpin,mldin,tsin,snowin,klon,klat,startyr,endyr,dt,rho,cp0)

# Quick test Output Test output
fig,ax=plt.subplots(1,1)
ax.set_title("Temperature Integration Output (Undetrended)")
ax.plot(sst,label="Integration Output")
ax.plot(tsin,label="CESM",color='k')
ax.legend()
ax.grid(True,ls='dotted')
#plt.savefig(outfigpath+"Undetrended_Output_1dIntegrationlast300.png",dpi=200)




# -----------------------
#%% Post Processing
# -----------------------
def postproc_sst(sst,manom=True,detrend=True):
    sst = sst-sst.mean(0) # Anomalize
    
    if detrend:
        # Detrend Data
        sst = signal.detrend(sst)
    
    if manom:
        # Calculate Monthly Anomalies
        ntime = sst.shape[0]
        sst    = sst.reshape(int(ntime/12),12)
        sst = sst - sst.mean(0)[None,:]
    return sst.flatten()

# Set Experiment Colors/Parameters
expcolors = ['k','b']
expnames  = ('$SST_{CESM}$',
             r"$SST_{Integration}$")
expls = ('solid','solid')
sstin = [tsin,sst]

# postprocess
sstproc = []
for sstr in sstin:
    sstproc.append(postproc_sst(sstr,manom=True))
sstann = []
for sstr in sstproc:
    nyr = sstr.shape[0]
    sstr = sstr.reshape(int(nyr/12),12)
    sstann.append(sstr.mean(1))
diffs = []
diffsann=[]
for s,sstr in enumerate(sstproc):
    diff = (sstproc[0]-sstr)**2
    diffa =(sstann[0]-sstann[s])**2
    diffs.append(diff)
    diffsann.append(diffa)
    print("RMSE For %i is %.3f"% (s,np.mean(diff)))
kmonth = 1#mld[klon,klat,...].mean().argmax()
print(kmonth)
acs  =scm.calc_autocorr(sstproc,lags,kmonth+1)


# Visualize Results
fig,ax=plt.subplots(1,1)
ax.set_title("Temperature Integration Output (Undetrended)")
for i in range(2):
    ax.plot(sstann[i],label=expnames[i],color=expcolors[i],ls=expls[i])
ax.legend()
ax.grid(True,ls='dotted')


# Plot Result
xtk2 = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

fig,ax = plt.subplots(1,1)
title = r"$Q_{net}$ Integration SST Autocorrelation,"+ " Lag 0 = %s,\n %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(2):
    ax.plot(lags,acs[i],color=expcolors[i],label=expnames[i],ls=expls[i])
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()
#plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s" % (locfn),dpi=200)

#
# %% Try rolling start year (Every 100 years)
#
ssts,tsins,expnames = [],[],[]
for i in range(9):
    startyr= i*100
    endyr  =901
    expnames.append("start=y%i"%startyr)
    print("Startyr %i"%startyr)
    isst,its = run_model(flxin,qdpin,mldin,tsin,snowin,klon,klat,startyr,endyr,dt,rho,cp0)
    ssts.append(isst)
    tsins.append(its)

sstin = ssts
sstin.insert(0,tsin)
tsins.insert(0,tsin)
expnames.insert(0,'$SST_{CESM}$')

# postprocess
sstproc = []
tsproc  = []
for s,sstr in enumerate(sstin):
    
    tsproc.append(postproc_sst(tsins[s]))
    sstproc.append(postproc_sst(sstr,manom=True))
    
sstann = []
tsann = []
for s,sstr in enumerate(sstproc):
    
    nyr = sstr.shape[0]
    sstr = sstr.reshape(int(nyr/12),12)
    tsann.append(tsproc[s].reshape(int(len(tsins[s])/12),12).mean(1))
    sstann.append(sstr.mean(1))
    
diffs = []
diffsann=[]
for s,sstr in enumerate(sstproc):
    
    diff  = (tsproc[s]-sstr)**2
    diffa = (tsann[s]-sstann[s])**2
    
    # if s == 0:
    #     diff = (tsins[s]-sstr)**2
    #     diffa =(sstann[0][s*100:10811]-sstann[s])**2
    # else:
    #     diff = (tsins[s]-sstr)**2
    #     diffa =(sstann[0][(s-1)*100:10811]-sstann[s])**2
    diffs.append(diff)
    diffsann.append(diffa)
    print("RMSE For %i is %.3f"% (s,np.mean(diff)))
kmonth = 1#mld[klon,klat,...].mean().argmax()
acs  =scm.calc_autocorr(sstproc,lags,kmonth+1)

# Plot Result (Autocorrelation)
xtk2 = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
fig,ax = plt.subplots(1,1)
title = r"$Q_{net}$ Integration SST Autocorrelation,"+ " Lag 0 = %s,\n %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
for i in range(len(acs)):
    if i == 0:
        expcol ='k'
        ealpha =1
    else:
        expcol = 'g'
        ealpha = i/len(acs)
    ax.plot(lags,acs[i],color=expcol,alpha=ealpha,label=expnames[i])
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend(ncol=3,fontsize=8)
plt.savefig(outfigpath + "CESM_SLAB_Auto_startyr%s" % (locfn),dpi=200)

# Plot Result (Differences)
for i in range(9):
    yrs = np.arange(i*100,901,1)
    fig,axs = plt.subplots(2,1)
        
    ax = axs[0]
    ax.plot(yrs,tsann[i+1],label="CESM",color='k')
    ax.plot(yrs,sstann[i+1],label="Integration",color='g')
    ax.legend()
    ax.set_title("%s"%(expnames[i+1]))
    ax.grid(True,ls='dotted')
    
    ax = axs[1]
    ax.plot(yrs,diffsann[i+1],label="Differences",color='k')
    ax.set_title("Differences (Squared error)")
    plt.tight_layout()
    ax.grid(True,ls='dotted')
    
    plt.savefig("%sDifferences_Startyr%i.png"%(outfigpath,i*100),dpi=150)
        

