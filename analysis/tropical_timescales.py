#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare lag-correlation of atmospheric and TS variables

Created on Wed Sep  1 04:40:06 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import xarray as xr
import time
from tqdm import tqdm

# %%
stormtrack = 0
if stormtrack == 0:
    projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath = projpath + '01_Data/model_output/'
    rawpath = projpath + '01_Data/model_input/'
    outpathdat = datpath + '/proc/'
    figpath = projpath + "02_Figures/20220720/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"

    sys.path.append(
        "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append(
        "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat = datpath + '/proc/'

    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append(
        "/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc, viz
import tbx
import scm
# %% Functions

def calc_dx_dy(longitude, latitude):
    ''' This definition calculates the distance between grid points that are in
        a latitude/longitude format.

        Equations from:
        http://andrew.hedges.name/experiments/haversine/

        dy should be close to 55600 m
        dx at pole should be 0 m
        dx at equator should be close to 55600 m

        Accepts, 1D arrays for latitude and longitude

        Returns: dx, dy; 2D arrays of distances between grid points 
                                    in the x and y direction in meters 
    '''
    dlat = np.abs(lat[1]-lat[0])*np.pi/180
    dy = 2*(np.arctan2(np.sqrt((np.sin(dlat/2))**2),
            np.sqrt(1-(np.sin(dlat/2))**2)))*6371000
    dy = np.ones((latitude.shape[0], longitude.shape[0]))*dy

    dx = np.empty((latitude.shape))
    dlon = np.abs(longitude[1] - longitude[0])*np.pi/180
    for i in range(latitude.shape[0]):
        a = (np.cos(latitude[i]*np.pi/180) *
             np.cos(latitude[i]*np.pi/180)*np.sin(dlon/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        dx[i] = c * 6371000
    dx = np.repeat(dx[:, np.newaxis], longitude.shape, axis=1)
    return dx, dy


# %% Set constants

omega = 7.2921e-5  # rad/sec
rho = 1026      # kg/m3
cp0 = 3996      # [J/(kg*C)]
mons3 = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


# Load lat/lon
lon360 = np.load(rawpath+"CESM1_lon360.npy")
lon180 = np.load(rawpath+"CESM1_lon180.npy")
lat = np.load(rawpath+"CESM1_lat.npy")

# Load Land/ice mask
msk = np.load(lipath)
#ds *= msk[None,:,:]

# Get distances relating to the grid
dx, dy = calc_dx_dy(lon360, lat)
xx, yy = np.meshgrid(lon360, lat)
# Set southern hemisphere distances to zero (?) since y = 0 at the equator..
#dy[yy<0] *= -1

#%% Load TS/SST

st = time.time()
ds1 = xr.open_dataset(rawpath+"../CESM_proc/TS_PIC_FULL.nc")
ts = ds1.TS.values
#ts -= 273.15 #(convert to celsius)
print("Completed in %.2fs"%(time.time()-st))

# Calculate the mean gradient for each month
ts_monmean = ts.mean(0)

# Calculate anomaly
ts_anom  = ts - ts_monmean[None,:,:,:]

nyr,_,nlat,nlon = ts_anom.shape
ts_anom = ts_anom.reshape(nyr*12,nlat*nlon)
#%% Load TAUx, TAUy

# Load each wind stress component [yr mon lat lon]
st   = time.time()
dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUX_PIC_FULL.nc")
taux = dsx.TAUX.values
dsx  = xr.open_dataset(rawpath+"../CESM_proc/TAUY_PIC_FULL.nc")
tauy = dsx.TAUY.values
print("Loaded wind stress data in %.2fs"%(time.time()-st))


# Convert stress from stress on OCN on ATM --> ATM on OCN
taux*= -1
tauy*= -1


# Remove monthly means
taux_monmean = taux.mean(0)
tauy_monmean = tauy.mean(0)
taux_anom = taux - taux_monmean[None,...]
tauy_anom = tauy - tauy_monmean[None,...]

# Comebine mon.year
taux_anom = taux_anom.reshape(ts_anom.shape)
tauy_anom = tauy_anom.reshape(ts_anom.shape)
#%% Now compute the correlation coefficients
lagmax = 37

lags = np.arange(1,lagmax,1)

corrcs = np.zeros((3,len(lags),nlat*nlon))
invars = [ts_anom,taux_anom,tauy_anom]
for v in range(len(invars)):
    invar = invars[v]
    
    for i in tqdm(range(len(lags))):
        l = lags[i]
        test = proc.pearsonr_2d(invar[l:,:],invar[:-l,:],dim=0)
        corrcs[v,i,:] = test
corrcs = corrcs.reshape(3,len(lags),nlat,nlon)


vnames = ["SST","TAUX","TAUY"]

savename = "%s../CESM_FULL-PIC_1-%ilag_correlation_SST_TAU.npz" % (datpath,lagmax)
np.savez(savename,**{
    "lag_corr":corrcs,
    "lags":lags,
    "vnames":vnames,
    "lon360":lon360,
    "lat":lat
    })
    

#%% Load the Data

ldname = "%s../CESM_FULL-PIC_1-3lag_correlation_SST_TAU.npz" % (datpath)
ld     = np.load(ldname,allow_pickle=True)

corrcs = ld['lag_corr']
lags   = ld['lags']
vnames = ld['vnames']
lon360 = ld['lon360']
lat    = ld['lat']

#%% Lets make some plots
bboxplot = [-90,0,0,75]

fig,axs = plt.subplots(3,3,constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()},figsize=(18,18))

levels = np.arange(1,1,0.1)
vlm = [-1,1]
levels = np.arange(-1,1.05,0.05)
for v in tqdm(range(len(invars))):
    
    for i in range(len(lags)):
        ax = axs[v,i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        ax.set_title("%s Lag %i"%(vnames[v],lags[i]))
        
        pcm = ax.pcolormesh(lon360,lat,corrcs[v,i,:,:]*msk,
                               vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.balance)
        
        cl = ax.contour(lon360,lat,corrcs[v,i,:,:]*msk,
                                levels=levels,colors="k",linewidths=0.75)
        # pcm = ax.contourf(lon360,lat,corrcs[v,i,:,:]*msk,
        #                        levels=levels,cmap='inferno')
        ax.clabel(cl)
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.01,anchor=(1.5,0.7))
plt.savefig("%sLagCorrelationMaps.png"%(figpath),dpi=150,bbox_inches='tight')
#%%

timescale = 1/(1-corrcs)

bboxplot = [-90,0,0,75]

fig,axs = plt.subplots(3,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(18,18))

#levels = np.arange(1,1,0.1)
vlm = [0,12]
levels = np.arange(0,13,1)
for v in tqdm(range(3)):
    
    for i in range(len(lags)):
        ax = axs[v,i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        ax.set_title("%s Lag %i"%(vnames[v],lags[i]))
        
        pcm = ax.pcolormesh(lon360,lat,timescale[v,i,:,:]*msk,
                               vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.thermal)
        
        cl = ax.contour(lon360,lat,timescale[v,i,:,:]*msk,
                                levels=levels,colors="k",linewidths=0.75)
        # pcm = ax.contourf(lon360,lat,corrcs[v,i,:,:]*msk,
        #                        levels=levels,cmap='inferno')
        ax.clabel(cl)
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.01,anchor=(1.5,0.7))
plt.savefig("%sTimescaleMaps.png"%(figpath),dpi=150,bbox_inches='tight')


#%% Plota ratio


ratio = corrcs[1,:,:]/corrcs[0,:,:]
rname = "TAUX_SST"

bboxplot = [-90,0,0,75]

fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(18,18))

levels = np.arange(0,1,0.1)
vlm = [0,1]
#levels = np.arange(0,13,1)
    
for i in range(len(lags)):
    ax = axs[i]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    ax.set_title("%s Lag %i"%(rname,lags[i]))
    
    pcm = ax.pcolormesh(lon360,lat,ratio[i,:,:]*msk,
                           vmin=vlm[0],vmax=vlm[-1],cmap=cmocean.cm.thermal)
    
    cl = ax.contour(lon360,lat,ratio[i,:,:]*msk,
                            levels=levels,colors="k",linewidths=0.75)
    # pcm = ax.contourf(lon360,lat,corrcs[v,i,:,:]*msk,
    #                        levels=levels,cmap='inferno')
    ax.clabel(cl)
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.01,anchor=(1.5,0.7))
plt.savefig("%sRatioMaps_%s.png"%(figpath,rname),dpi=150,bbox_inches='tight')


    
    # if plot_contours:
    #     pcm1 = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
    #                         cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
    #     pcm = ax.contourf(lon180,lat,thresperc[:,:,im].T,
    #                       levels=levels,cmap='inferno')
    #     cl = ax.contour(lon180,lat,thresperc[:,:,im].T,
    #                       levels=levels,colors='k',linewidths=0.50)
    #     ax.clabel(cl)

    # else:
    #     pcm = ax.pcolormesh(lon180,lat,thresperc[:,:,im].T,
    #                         cmap='inferno',vmin=vlm[0],vmax=vlm[-1])
    # ax.set_title("%s" % (mons3[im]))
# fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.01,anchor=(1.5,0.7))
# plt.savefig("%s%s_NHFLX_EOFs_vratio_%03ipercEOFs_allmon.png"%(outpath,mcname,vthres*100),dpi=150,bbox_inches='tight')



# l = 1


# test = test.reshape(nlat,nlon)

# # The I'm tired loop approach
# for i in range(nlat*nlon):



# test = tqdm(test.reshape(nlat,nlon))
    
#test = np.corrcoef(ts_anom[l:,...],ts_anom[:-l,...])

#%% Find e-folding timescale








