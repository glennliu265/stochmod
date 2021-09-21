#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Solve by inversion

Created on Sat Feb 20 17:42:15 2021

@author: gliu
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
import time
from tqdm import tqdm

# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import proc,viz
import tbx
from dask.distributed import Client,progress
import dask

import cartopy.crs as ccrs
import cmocean
import cartopy
import xarray as xr
import cartopy.feature as cfeature

import matplotlib.colors as mc

#%%
lonf  = -30+360
latf  = 50
debug = True

dt  = 3600*24*30
rho = 1026
cp0 = 3996 



# Load CESM Data
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
fn2     = datpath + "../ENSOREM_TS_lag1_pcs2_monwin3.npz"
ld2     = np.load(fn2)
tscesm  = ld2['TS']
lon360  = ld2['lon']
lat     = ld2['lat']

genrand = True


#  Load mized layer depth values
hblt = np.load(datpath+"../SLAB_PIC_hblt.npy")
#%% Find Point and visualize

klon360,klat = proc.find_latlon(lonf,latf,lon360,lat)
lon180,_ = proc.lon360to180(lon360,tscesm.transpose(2,1,0))
klon180,_ = proc.find_latlon(lonf-360,latf,lon180,lat)
if debug:
    fig,ax = plt.subplots(1,1)
    pcm = ax.pcolormesh(lon360,lat,tscesm[0,:,:])
    ax.scatter(lon360[klon360],lat[klat],s=200,marker='x')
tspt  = tscesm[:,klat,klon360]
h     = hblt[klon180,klat,:].mean()
ntime = tscesm.shape[0]
nyrs  = int(ntime/12)

#%%
from scipy import optimize

# Reshape to separate month and year
tspt = tspt.reshape(nyrs,12)
y   = tspt[1:,:]
ym1 = tspt[:-1,:]


# Iterate [nitr] times and compute best lambda and forcing
nitr = 10
Fall = np.zeros((ntime,nitr))
lambdas = np.zeros((12,nitr))
alphas  = np.zeros(lambdas.shape)
tsest = np.zeros((nyrs-1,12,nitr))
for it in tqdm(range(nitr)):
    # Generate white noise
    F    = np.random.normal(0,1,ntime)
    Fall[:,it] = F
    F    = F.reshape(nyrs,12)
    Winv = np.ones((nyrs-1,nyrs-1))*1/(nyrs-1)  #NOTE THIS IS WRONG, need to fix
    
    # For each month
    for m in range(12):
        b = tspt[1:,[m]] # nyr x 1
        A = np.hstack([tspt[:-1,[m]],F[1:,[m]]]) # nyr x 2
        
        x,r = optimize.nnls(A,b.squeeze())
        
        tsest[:,m,it] = A@x
        #x,_= tbx.LSE(A,Winv,b,np.zeros(Winv.shape))
        lambdas[m,it] = x[0]
        alphas[m,it]=x[1]


# Convert Results
lbda = -1*np.log(lambdas)/dt*(rho*cp0*h)
alphac = alphas/dt*(rho*cp0*h)
plt.plot(lbda.mean(1))
plt.plot(alphac.mean(1))

# Recreate timeseries
it = 1

ts_int = np.zeros(ntime)
ts_int[0] = tspt[0,0]
for t in range(1,ntime):
    m = (t+1)%12
    #ts_int[t] = lambdas[m,it]*ts_int[t-1] + alphas[m,it] * Fall[t,it]
    #ts_int[t] = lambdas.mean(1)[m]*ts_int[t-1] + alphas.mean(1)[m] * Fall[t,it]
    ts_int[t] = np.exp(-lbdset[m])*ts_int[t-1] + alphaset[m] * Fall[t,it]
    
fig,ax = plt.subplots(1,1)
ax.plot(tspt.flatten()[1:],label="CESM",color='k')
ax.plot(tsest[:,:,it].flatten(),label="Estlimate",color='r')
ax.plot(ts_int,label="Integrated",color='orange')
ax.legend()

# Calculate Autocorrelation
tsintac = ts_int.reshape(nyrs,12).T
kmonth  = 2
lags   = np.arange(0,37,1)
#acest  = proc.calc_lagcovar(tsest[...,it].T,tsest[...,it].T,lags,kmonth+1,1)
accesm = proc.calc_lagcovar(tspt.T,tspt.T,lags,kmonth+1,1)
acint  = proc.calc_lagcovar(tsintac,tsintac,lags,kmonth+1,1)




# Plot some differences
fig,ax = plt.subplots(1,1)
xtk2 = np.arange(0,37,2)
title      = "SST Autocorrelation using estimated Damping (Lag 0 = Feb)" 
kmonth=1
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=damppt1,title=title)
ax.plot(lags,accesm,label="CESM-SLAB",color='k')
ax.plot(lags,acint,label="Using Estimated Damping",color='orange')
ax.legend()
#ax.grid(True,ls='dotted')
ax2.set_ylabel("Damping (W/m2)")
plt.savefig(outpath+"Autocorrelation_Estimate.png",dpi=200)



#%% Perform case where you already know the forcing, and just ned to solve for the damping
Fpt = np.array([55.278503, 53.68089 , 42.456623, 33.448967, 22.954145, 22.506973,
       22.151728, 24.135042, 33.337887, 40.91648 , 44.905064, 51.132706])

damppt = np.array([16.99182759, 17.46026415, 18.72853076, 18.86257056, 19.09411826,
       18.37147339, 17.76045915, 17.68516372, 16.50739147, 16.97239387,
       17.79174395, 15.93981852])

damppt1 = np.array([17.18319618, 17.58090563, 18.78515961, 18.88929534, 19.10085634,
        18.38692996, 17.79925571, 17.70014925, 16.53779173, 17.01315418,
        17.91329541, 16.04075808])
alphaset = Fpt/rho/cp0/h*dt
lbdset  = damppt1/rho/cp0/h*dt



# Iterate [nitr] times and compute best lambda and forcing
nitr    = 10000
Fall    = np.zeros((ntime,nitr))
lambdas1 = np.zeros((12,nitr))
#alphas  = np.zeros(lambdas.shape)
tsest1 = np.zeros((nyrs-1,12,nitr))

for it in tqdm(range(nitr)):
    # Generate white noise
    F    = np.random.normal(0,1,ntime)
    Fall[:,it] = F
    #F = Fall[:,it]
    F    = F.reshape(nyrs,12)
    Winv = np.ones((nyrs-1,nyrs-1))*1/(nyrs-1) 
    
    # For each month
    for m in range(12):
        b = tspt[1:,[m]] - F[1:,[m]]*alphaset[m] # nyr x 1
        A = tspt[:-1,[m]] # nyr x 1
        
        x,r = optimize.nnls(A,b.squeeze())
        
        tsest1[:,m,it] = A@x
        #x,_= tbx.LSE(A,Winv,b,np.zeros(Winv.shape))
        lambdas1[m,it] = x[0]
        
lbda1 = -1*np.log(lambdas1)/dt*(rho*cp0*h)

np.save("lba_estimate_10k.npy",lbda1)

# 10k best
array([17.18319618, 17.58090563, 18.78515961, 18.88929534, 19.10085634,
        18.38692996, 17.79925571, 17.70014925, 16.53779173, 17.01315418,
        17.91329541, 16.04075808])


fig,ax = plt.subplots(1,1)
ax.plot(mons3,(lbda1/10),label="",color='k',alpha=0.10)
ax.plot(mons3,lbda1.mean(1)/10,color='w',label='mean')
ax.plot(mons3,damppt,color='r',label="CESM-Estimate")
ax.legend()
ax.set_ylabel("Damping (W/m2)")
plt.savefig(outpath+"damping10estimates.png",dpi=200)''


# Recreate stochastic model
stocht = np.zeros((nitr,ntime))
for it in tqdm(range(nitr)):
    ts_int = np.zeros(ntime)
    ts_int[0] = tspt[0,0]
    for t in range(1,ntime):
        m = (t+1)%12
        #ts_int[t] = lambdas[m,it]*ts_int[t-1] + alphas[m,it] * Fall[t,it]
        #ts_int[t] = lambdas.mean(1)[m]*ts_int[t-1] + alphas.mean(1)[m] * Fall[t,it]
        ts_int[t] = lambdas1[m,it]*ts_int[t-1] + alphaset[m] * Fall[t,it]
    stocht[it,:] = ts_int

# Calculate autocorrelation
acs10k = np.zeros((nitr,37))
for it in tqdm(range(nitr)):
    stp = stocht[it,...]
    stp = stp.reshape(int(stp.shape[0]/12),12)
    acstp = proc.calc_lagcovar(stp.T,stp.T,lags,kmonth+1,1)
    acs10k[it,:] = acstp
    
    
    
diffs = acs10k - accesm[None,:]

bestid = np.argmin(np.sum(diffs**2,1))


fig,ax = plt.subplots(1,1)
ax.plot(lags,accesm,color='k')
ax.plot(lags,acs10k[bestid,:],color='orange')