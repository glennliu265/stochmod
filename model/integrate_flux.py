#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Model where T(t+1) = T(t) + F(t)/(rho*cp0*h)

Created on Wed Feb  3 14:11:57 2021
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
import tbx
import scm
from scipy import signal
#%% User Edits
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
outfigpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210211/"

debug = True
lonf  = -30
latf  = 50
rho   = 1026
cp0   = 3996
dt    = 3600*24*30
quadrature = False
lags = np.arange(0,37,1)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

#%% Load Data

# NHFLX, LON x LAT x TIME
st = time.time()
dsflx = xr.open_dataset(datpath+"NHFLX_PIC.nc")
flx = dsflx.NHFLX.values
lon = dsflx.lon.values
lat = dsflx.lat.values
print("Loaded NHFLX in %.2fs"% (time.time()-st))

# Mixed Layer Depths
st = time.time()
dsmld = xr.open_dataset(datpath+"HMXL_PIC.nc")
mld = dsmld.HMXL.values/100 # Convert to meters
lon = dsmld.lon.values
lat = dsmld.lat.values
print("Loaded MLD in %.2fs"% (time.time()-st))

#%% Preprocessing

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
loctitle = "Lon %.2f Lat %.2f" % (lon[klon],lat[klat])
locfn    = "lon%i_lat%i" % (lonf,latf)

# Calculate 1000 year mean and seasonal MLDs
nlon,nlat,ntimef = mld.shape
hclim      = mld.reshape(nlon,nlat,int(ntimef/12),12)
mldcycle   = hclim.mean(2) 
mld_1kyr   = mld[:,:,:1000]
mld_1kmean = mld_1kyr.mean(2)

print("1000 yr mean mld is : %.3f m" % (mld_1kmean[klon,klat]))
print("Mean of scycle is   : %.3f m" % (mldcycle.mean(2)[klon,klat]))
#hclim = np.zeros((nlon,nlat,12))*np.nan

if debug:
    fig,ax  = plt.subplots(1,1)
    for i in range(10):
        ax.plot(hclim[klon,klat,i,:],label='year %i'%i,alpha=0.5)
    ax.plot(mldcycle[klon,klat,...],label='mean cycle',color='k')
    ax.set_title("Seasonal MLD Cycle at %s"%loctitle)
    ax.grid(True,ls='dotted')
    ax.legend()

#%% Load qflux correction and mixed layer depth

cmbal = cmocean.cm.balance
hblt = np.load(datpath+"../SLAB_PIC_hblt.npy")
qdp  = np.load(datpath+"../SLAB_PIC_qdp.npy") 

qdpa = -1*(qdp -qdp.mean(2)[:,:,None])

if debug:
    fig,axs = plt.subplots(2,2)
    
    axlist = axs.flatten()
    
    ax = axlist[0]
    pcm = ax.pcolormesh(mld_1kmean.T,vmin=0,vmax=500)
    ax.set_title("MLD (Mean, yr400-1400)")
    fig.colorbar(pcm,ax=ax)
    
    
    ax = axlist[1]
    pcm = ax.pcolormesh(flx[:,:,0].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_title("Qnet (400-01)")
    fig.colorbar(pcm,ax=ax)
    
    ax = axlist[2]
    pcm = ax.pcolormesh(hblt[:,:,0].T,vmin=0,vmax=500)
    ax.set_title("hblt (Jan)")
    fig.colorbar(pcm,ax=ax)
    
    ax = axlist[3]
    pcm = ax.pcolormesh(qdpa[:,:,0].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_title("qdp (Jan)")
    fig.colorbar(pcm,ax=ax)
    
    plt.tight_layout()
    
    plt.savefig(outfigpath+"QIntegration_Inputs.png",dpi=200)

#%% Plot Qflx Correction over north atlantic

#fig,axs= plt.subplots(4,3,subplot_kw={'projection': ccrs.PlateCarree()})
fig,axs= plt.subplots(4,3,figsize=(8,8))

for i in range(12):
    ax  = axs.flatten()[i]
    pcm = ax.pcolormesh(lon,lat,qdpa[:,:,i].T,vmin=-500,vmax=500,cmap=cmbal)
    ax.set_xlim([-80,20])
    ax.set_ylim([0,80])
    ax.set_title(mons3[i])
    fig.colorbar(pcm,ax=ax)

    plt.tight_layout()
plt.savefig(outfigpath+"qdp_anomaly.png",dpi=200)

#%% Plot NHFLX at that point

flxmon = flx[klon,klat,...].reshape(int(flx.shape[2]/12),12)
fig,ax = plt.subplots(1,1)
ax.plot(mons3,flxmon.T,color='gray',alpha=0.1,label="")
ax.plot(mons3,flxmon.std(0),color='k',ls='dashed',label=r"$\pm 1 \sigma$")
#ax.plot(mons3,flxmon.mean(0),color='k',ls='dotted')
ax.plot(mons3,-1*flxmon.std(0),color='k',ls='dashed',label="")
ax.plot(mons3,qdpa[klon,klat,:],color='r',ls='solid',label="Qflx Correction")
#ax.set_ylim([-80,80])
ax.grid(True,ls='dotted')

ax.set_title("NHFLX Seasonal Cycle at %s"%loctitle)
plt.savefig("%sNHFLX_Scycle_SLAB_PIC_%s.png"%(outfigpath,locfn),dpi=200)



#%% Plot result after correction

flxmon = (flx[klon,klat,...]+np.tile(qdpa[klon,klat,...],int(10776/12))).reshape(int(flx.shape[2]/12),12)

fig,ax = plt.subplots(1,1)
ax.plot(mons3,flxmon.T,color='gray',alpha=0.1,label="")
ax.plot(mons3,flxmon.std(0)+qdpa[klon,klat,:],color='k',ls='dashed',label=r"$\pm 1 \sigma$")
#ax.plot(mons3,flxmon.mean(0),color='k',ls='dotted')
ax.plot(mons3,-1*flxmon.std(0)+qdpa[klon,klat,:],color='k',ls='dashed',label="")
ax.plot(mons3,(qdpa[klon,klat,:]),color='r',ls='solid',label="Qflx Correction")
ax.set_ylim([-200,200])
ax.grid(True,ls='dotted')
ax.legend()

ax.set_title("NHFLX Seasonal Cycle at %s (Corrected)"%loctitle)
plt.savefig("%sNHFLX_Scycle_SLAB_PIC_%s_corr.png"%(outfigpath,locfn),dpi=200)



#%% If quadrature is true, interpolate values



print("Interpolating Values")
flxquad = np.zeros(flx.shape)
for i in tqdm(np.arange(1,flx.shape[2]-1)):
    flxquad[:,:,i] = (flx[:,:,i-1] + flx[:,:,i]) / 2 # Take mean of 2 values
    #print("%f is between %f and %f"% (flxquad[222,111,i-1],flx[222,111,i-1],flx[222,111,i]))

flxquad[:,:,0] = flx[:,:,0]
#%%
# Plot Differences
fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.plot(np.arange(0.5,13.5,1),flx[222,111,:13],label="Original Timeseries",color='k',zorder=-2,marker="d")

ax.scatter(np.arange(1,13,1),flxquad[222,111,1:13],label="Interpolated Timeseries",marker="o",color='red')
ax.plot(np.arange(1,13,1),flxquad[222,111,1:13],label="",marker="o",color='red',ls='dashed')
ax.set_xticks(np.arange(0,13,1))
ax.set_xticklabels(np.hstack([mons3,mons3[0]]))
ax.grid(True,ls='dotted')
ax.legend()
ax.set_xlabel("Month")
ax.set_ylabel("Forcing (W/m2)")
ax.set_title("Interpolated Qnet")
plt.savefig(outfigpath+"Interpolated_Qnet.png",dpi=200)
    #ax.plot(np.arange(1,24,1),flx[22,111,1:25])
    #tsteps = 

#%% Visualize differences in flux


fig,ax = plt.subplots(1,1)
ax.plot(hblt[klon,klat,m],color='k')
#ax.plot(mld[klon,klat,:])
ax.set_ylim([51,53])


#%% Load in mean SST from slab

fn1 = datpath + "../TS_SLAB_withENSO.npy"
fn2 = datpath + "../ENSOREM_TS_lag1_pcs2_monwin3.npz"

tsenso1 = np.load(fn1)
ld2 = np.load(fn2)
tsenso0 = ld2['TS']
lon360 = ld2['lon']
lat = ld2['lat']

# Reshape to [yr x mon x lat x lon]
nmon,nlat,nlon = tsenso0.shape

tsenso0 = tsenso0.transpose(2,1,0) # lon x lat x mon
lon1,tsenso180 = proc.lon360to180(lon360,tsenso0) # Flip longitudes


#tsenso0 = tsenso0.reshape(int(nmon/12),12,nlat,nlon)


#%% Integrate the model
flxin = flx
qdpin= qdpa
if quadrature:
    flxin = flxnew
    qdpin = np.roll(qdpa,1,)



mldmean = hblt.mean(2)#mld_1kmean
_,_,tsteps = flx.shape
sstq  = np.zeros((nlon,nlat,tsteps))
sstq[:,:,0] = tsenso180[:,:,0]
sstn = sstq.copy()
ssts = sstq.copy()
for t in tqdm(range(1,tsteps)):
    
    m = (t)%12
    
    sstq[:,:,t] = sstq[:,:,t-1]  + (flxin[:,:,t] + qdpin[:,:,m]) * (dt / (rho*cp0*mldmean))
    sstn[:,:,t] = sstn[:,:,t-1]  + flxin[:,:,t]                  * (dt / (rho*cp0*mldmean))
    
    ssts[:,:,t] = ssts[:,:,t-1]  + flxquad[:,:,t] * (dt / (rho*cp0*mldmean))

# Plot results
fig,ax = plt.subplots(1,1)
ax.plot(sstq[klon,klat,:],label="With Correction")
ax.plot(sstn[klon,klat,:],label="Without Correction")
ax.legend()

#ax.set_xlim([0,100])
#%% Calculate Autocorrelation
kmonth = mldcycle[klon,klat,...].argmax()
print(kmonth)
ssts = [sst[klon,klat,...],sst1[klon,klat,...]]
acs =scm.calc_autocorr(ssts,lags,kmonth+1)

#%% Load CESM Slab Autocorrelation
cesmslabac     = np.load(datpath+"../CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
klon360,_      = proc.find_latlon(lonf+360,latf,lon360,lat)

slabac = cesmslabac[kmonth,lags,klat,klon360]

#%% Try detrending SST

# #sstin = sst[klon,klat,...] #- sst1.mean(2)[klon,klat,None]
# #sstdt = signal.detrend(sstin,axis=0,type='linear')
# #x,Cxx,sstdt1,yhat,ybnds = tbx.detrend_poly_LSE(sstin,1)

# # Quickly plot output sst
# fig,ax = plt.subplots(1,1,figsize=(8,4))
# ax.plot(sstin,lw=0.5,label='Raw Output',color='b',alpha=0.4)
# #ax.plot(sstdt,lw=0.5,color='r',label="Detrended")
# ax.plot(sstdt1,lw=0.5,color='orange',label="Detrended")
# ax.plot(yhat,lw=0.5,color='k',label="Removed Trend")
# #ax.plot(yhat,lw=0.5,color='k',label="Trend (y=%.2fx + %.2f)"% (x[1],x[0]))
# ax.set_xlabel("Time (Months)")
# ax.set_ylabel("SST (degC)")
# ax.grid(True,ls='dotted')
# ax.legend(fontsize=8,ncol=3)
# ax.set_ylim([-4,4])

# plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_Model_Output_detrend_10x.png" % (locfn),dpi=200)





#%%
lws = .75
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tsenso0[klon360,klat,:],label="SLAB SST",color='k',lw=lws)
ax.plot(sstn[klon,klat,:],label="Flux Integration",color='b',lw=lws)
ax.plot(ssts[klon,klat,:],label="Flux, Shifted Forcing",color='g',lw=lws)
#ax.plot(sstq[klon,klat,:],label="Flux Integration with Q Correction",color='orange',lw=lws,alpha=0.90)
#ax.set_xlim([10276,10776])
ax.set_xlabel("Time (Months)")
ax.set_ylabel("SST (degC)")
ax.grid(True,ls='dotted')
#ax.set_ylim([-2,2])
#ax.set_xlim([0,100])
ax.legend(fontsize=8,ncol=1)
#ax.set_title("SST Integration Comparison \n Location: %s" % (loctitle))
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegrationComp_Point%s_Model_Output_detrend_Quad_all.png" % (locfn),dpi=200)

#%% Do the same as above, but for annual averages


lws = 1
sstqann  = proc.ann_avg(sstq[klon,klat,:],0)
sstnann  = proc.ann_avg(sstn[klon,klat,:],0)
sstsann  = proc.ann_avg(ssts[klon,klat,:],0)
sstslab  = proc.ann_avg(tsenso0[klon360,klat,:],0)

fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(sstslab,label="SLAB SST",color='k',lw=lws)
ax.plot(sstnann,label="Flux Integration ",color='b',lw=lws,alpha=0.90)
ax.plot(sstsann,label="Flux Integration, Shifted Forcing",color='g',alpha=0.9)
#ax.plot(sstqann,label="Flux Integration with Q Correction",color='orange',lw=lws,alpha=0.90)
ax.set_xlabel("Time (Years)")
ax.set_xticks(np.arange(0,1000,50))
ax.set_ylabel("SST (degC)")
#ax.set_xlim([800,900])
ax.grid(True,ls='dotted')
#ax.legend(fontsize=8,ncol=3)
#ax.set_title("SST (Ann Average),\n Location: %s" % (loctitle))
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegrationComp_Point%s_Model_Output_detrend_Ann_quad.png" % (locfn),dpi=200)


#%% Visualize error with year

errorn = np.abs(tsenso0[klon360,klat]-sstn[klon,klat,:])
errorq = np.abs(tsenso0[klon360,klat]-sstq[klon,klat,:])


errorn = errorn.reshape(int(len(errorn)/12),12)
fig,ax = plt.subplots(1,1)
#ax.plot(mons3,errorn.T)
ax.plot(mons3,errorn.std(0))
#ax.set_xlim([0,100])
#ax.plot(errorq)

#%% Plot SST and NHFLX

flxpt = flx[klon,klat,:]
flxann = proc.ann_avg(flxpt,0)

fig,ax = plt.subplots(1,1,figsize=(4,2))
ax.plot(sstslab,color='b',lw=0.5)

ax2 = ax.twinx()

ax2.plot(flxann,color='gray',lw=0.5,alpha=0.5)


#%% Examining forcing ar particular points

of1 = -30
af1 = 50

flxr = flx.reshape(nlon,nlat,int(10776/12),12)

o,a = proc.find_latlon(of1,af1,lon,lat)
flxpt = flxr[o,a,...]
flxstd = flxpt.std(0)

fig,ax=plt.subplots(1,1)

ax.plot(mons3,flxpt.T,color='gray',alpha=0.1)
ax.plot(mons3,flxstd,color='k',ls='dashed')
ax.plot(mons3,-1*flxstd,color='k',ls='dashed')
ax.set_title("NHFLX Anomalies Lon %i, Lat %i"%(of1,af1))
ax.set_ylim([-80,80])





#%% Create Comparitive plot (testing month of forcing)
xtk2 = np.arange(0,37,2)


fig,ax = plt.subplots(1,1)
title = "Autocorrelation, Lag 0 = %s,\n Location: %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,acs[0],label="NHFLX Forcing (same month)")
ax.plot(lags,acs[1],label="NHFLX Forcing (last month)")
ax.plot(lags,slabac,label="CESM-SLAB, Actual Correlation",color='k')
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()

#%% Plot ac and slab comparison
xtk2 = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

fig,ax = plt.subplots(1,1)
title = "Autocorrelation, Lag 0 = %s,\n Location: %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
#ax.plot(lags,acs[0],label="NHFLX Forcing (same month)")
ax.plot(lags,acs[1],label="NHFLX Forcing")
ax.plot(lags,ac[1],label="Stochastic Model Output")
ax.plot(lags,slabac,label="CESM-SLAB, Actual Correlation",color='k')
#ax.plot(lags,cesmslabac[:,lags,klat,klon360].T,label="CESM-SLAB, Actual Correlation")
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s" % (locfn),dpi=200)


#%% Visualize forcing at this point


flxpt = flx[klon,klat,:]
flxpt = flxpt.reshape(int(tsteps/12),12)

fig,ax = plt.subplots(1,1)

ax.plot(flxpt.T,color='gray',alpha=0.1)
ax.plot(flxpt.std(0),color='k',ls='dashed')

 -
fig,ax= plt.subplots(1,1)
# ax.plot(testa,label="Orifile")
# ax.plot(testb,label="newfile")
ax.plot(testa-testb)
ax.legend()





#%% Try Plotting Autocorrelation of detrended SSts

sstall = [sstq[klon,klat,...],sstn[klon,klat,...],ssts[klon,klat,...]]
kmonth = mldcycle[klon,klat,...].argmax()
print(kmonth)
acs =scm.calc_autocorr(sstall,lags,kmonth+1)
xtk2 = np.arange(0,37,2)
# Plot
fig,ax = plt.subplots(1,1)
title = "Autocorrelation, Lag 0 = %s,\n Location: %s" % (mons3[kmonth],loctitle)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,acs[0],lw=0.85,label="Flux Integration with Q Corretion",color='orange')
ax.plot(lags,acs[1],ls='dashed',lw=0.85,label="Flux Integration",color='b')
ax.plot(lags,acs[2],ls='dashed',lw=0.85,label="Flux Integration, Shifted Forcing",color='g')
ax.plot(lags,slabac,label="CESM-SLAB, Actual Correlation",color='k')
ax.plot(lags,acnoenso,label="CESM-SLAB,ENSO Removed",color='r')
ax.set_ylabel('Correlation')
ax.set_xlabel('Lags (months)')
ax.legend()
plt.savefig(outfigpath + "CESM_SLAB_FluxIntegration_Point%s_detrend_10x.png" % (locfn),dpi=200)

#%% Calculate Autocorrelation but for point without enso

# Get variable
tspoint = tsenso0[klon360,klat,:]
tspoint = tspoint.reshape(int(10776/12),12).T
tspoint = tspoint-tspoint.mean(1)[:,None]

# Calculate Lag correlation
acnoenso = proc.calc_lagcovar(tspoint,tspoint,lags,kmonth+1,0)



