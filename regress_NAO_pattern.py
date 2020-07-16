#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:38:34 2020

@author: gliu
"""

import xarray as xr
from scipy import signal
import numpy as np
import time

import matplotlib.pyplot as plt
from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point



# Functions
def regress_2d(A,B):
    """
    Regresses A (independent variable) onto B (dependent variable), where
    either A or B can be a timeseries [N-dimensions] or a space x time matrix 
    [N x M]. Script automatically detects this and permutes to allow for matrix
    multiplication.
    
    Returns the slope (beta) for each point, array of size [M]
    
    
    """
    # Determine if A or B is 2D and find anomalies
    
    
    # 2D Matrix is in A [MxN]
    if len(A.shape) > len(B.shape):
        
        # Tranpose A so that A = [MxN]
        if A.shape[1] != B.shape[0]:
            A = A.T
        
        
        # Set axis for summing/averaging
        a_axis = 1
        b_axis = 0
        
        # Compute anomalies along appropriate axis
        Aanom = A - np.nanmean(A,axis=a_axis)[:,None]
        Banom = B - np.nanmean(B,axis=b_axis)
        

        
    # 2D matrix is B [N x M]
    elif len(A.shape) < len(B.shape):
        
        # Tranpose B so that it is [N x M]
        if B.shape[0] != A.shape[0]:
            B = B.T
        
        # Set axis for summing/averaging
        a_axis = 0
        b_axis = 0
        
        # Compute anomalies along appropriate axis        
        Aanom = A - np.nanmean(A,axis=a_axis)
        Banom = B - np.nanmean(B,axis=b_axis)[None,:]
    
    # Calculate denominator, summing over N
    Aanom2 = np.power(Aanom,2)
    denom = np.sum(Aanom2,axis=a_axis)    
    
    # Calculate Beta
    beta = Aanom @ Banom / denom
    
        
    return beta

    



# Path to data
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath  = projpath + '01_Data/'
outpath = projpath+'/02_Figures/20200716/'
flux = 'NHFLX' # [ 'FLNS','FSNS','LHFLX','SHFLX','RHFLX','THFLX','NHFLX','PSL']

#ncname = "NHFLX_NAOproc.nc"

#




# %% Load data


# Load Data (NAO Index)
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO.npz")
eofall    = npzdata['eofall']
pcall     = npzdata['pcall']
varexpall = npzdata['varexpall'] 

# Load Data base on flux (Combine separately for RHFLX and THFLX)
if flux == 'THFLX':
    
    # Load Latent Heat Fluxes
    ncname1 = 'LHFLX_NAOproc.nc'
    ds1 = xr.open_dataset(datpath+ncname1)
    lhflx = ds1['LHFLX']
    
    # Load Sensible Heat Fluxes
    ncname2 = 'SHFLX_NAOproc.nc'
    ds2 = xr.open_dataset(datpath+ncname2)
    shflx = ds2['SHFLX']
    
    # Sum the two fluxes
    flx = lhflx+shflx
    
elif flux == 'RHFLX':
    
    # Load Shortwave 
    ncname1 = 'FSNS_NAOproc.nc'
    ds1 = xr.open_dataset(datpath+ncname1)
    fsns = ds1['FSNS']
    
    # Load Sensible Heat Fluxes
    ncname2 = 'FLNS_NAOproc.nc'
    ds2 = xr.open_dataset(datpath+ncname2)
    flns = ds2['FLNS']
    
    # Sum the two fluxes
    flx = fsns+flns
    
else:
    
    ncname = "%s_NAOproc.nc" % flux
    ds = xr.open_dataset(datpath+ncname)
    flx = ds[flux]
    

# Read in Coordinate values
lon = flx['lon'].values
lat = flx['lat'].values
time = flx['year'].values
flx = flx.values

# %% Prepare data

var = np.copy(flx)*-1 # Note, multiply by negative 1 to convert to upwards negative

# Get dimension sizes
nens,nyr,nlat,nlon = var.shape
npc = pcall.shape[2]

# Combine lat and lon dimensions
var = np.reshape(var,(nens,nyr,nlat*nlon))

# Regress for each mode and ensemble member
varpattern = np.zeros((npc,nens,nlat*nlon))
for n in range(npc):
    for e in range(nens):
        
        pcin = pcall[e,:,n]
        datain = var[e,...]

        varpattern[n,e,:] = regress_2d(pcin,datain)
        
        msg = '\rCompleted Regression for PC %02i/%02i, ENS %02i/%02i' % (n+1,npc,e+1,nens)
        print(msg,end="\r",flush=True)

# Reshape variable [pc, ens, lat, lon]
varout = np.reshape(varpattern,(npc,nens,nlat,nlon))

F = np.copy(varout)

outname = datpath+"NAO_%s_Forcing.npy" % flux 
np.save(outname,F)

#%%
# Try Plotting ensemble average
def plot_regression_slp(N,e,varin,slp,lon,lat,cint=[0]):
    # Restrict to Ensemble and PC
    if e == "avg":
        # Assumed [pc, ens, lat, lon]
        varplot = np.nanmean(varin,axis=1)[N,...]
        slpplt = np.nanmean(slp,axis=0)[:,:,N]
        estring = "AVG"
    else:
        varplot = varin[N,e,:,:]
        slpplt = slp[e,:,:,N]
        estring = "%02i"%e
    
    cmap = cmocean.cm.balance
    #Plotting boundaries
    lonW = -90
    lonE = 40
    latS = 0
    latN = 80

    # if N == 0:
    #     cint = np.arange(-3,3.5,0.5)
    # elif N == 1:
    #     cint = np.arange(-1,1.2,0.2)
    # elif N == 2:
    #     cint = np.arange(-2,2.1,0.1)
    
    cintp = np.arange(-5,6,1)
    # Plot the EOF
    slp1,lon1 = add_cyclic_point(slpplt,lon)
    var1,lon1 = add_cyclic_point(varplot,lon) 
    
    plt.style.use('ggplot')
    fig,ax= plt.subplots(1,1,figsize=(6,4))
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lonW,lonE,latS,latN])
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,linewidth=0.75,linestyle=":")
    
    # Add contours
    if len(cint) == 1:
        cs = ax.contourf(lon1,lat,var1,cmap=cmap,transform=ccrs.PlateCarree())
    else:
        cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,transform=ccrs.PlateCarree())
        
        # Negative contours
        cln = ax.contour(lon1,lat,slp1,
                    cintp[cintp<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())
        plt.clabel(cln,levels=[-2,-4],fmt='%i',fontsize=8)
        
        # Positive Contours
        clp = ax.contour(lon1,lat,slp1,
                    cintp[cintp>=0],
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())   
        # Add Label
        
        plt.clabel(clp,levels=[0,2,4],fmt='%i',fontsize=8)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    

    bc = plt.colorbar(cs,orientation='horizontal')
    ax.set_title("%s - PC%i Regression, ENS %s, " %(flux,N+1,estring),fontsize=16)


    return fig
N =0
e = 'avg'
if flux == 'RHFLX':
    cint = np.arange(-10,11,1)
else:
    cint = np.arange(-50,55,5)
estring = str(e)



fig = plot_regression_slp(N,e,varout,eofall,lon,lat,cint=cint)
plt.savefig(outpath+"%s_NAO_EOF%i_Ens%s.png"%(flux,N+1,estring), bbox_inches="tight",dpi=200)


#
#%%Loop for each ensemble member
#cint = np.arange(-50,55,5)

for e in range(42):
    estring = str(e+1)
    fig = plot_regression_slp(0,e,varout,eofall,lon,lat,cint=cint)
    plt.savefig(outpath+"%s_NAO_EOF%i_Ens%s.png"%(flux,N+1,estring), bbox_inches="tight",dpi=200)

#%%

varin = np.copy(varout)

# Restrict to Ensemble and PC
if e == "avg":
    # Assumed [pc, ens, lat, lon]
    varplot = np.nanmean(varin,axis=1)[N,...]
    estring = "AVG"
else:
    varplot = varin[N,e,:,:]
    estring = "%02i"%e

cmap = cmocean.cm.balance
#Plotting boundaries
lonW = -90
lonE = 40
latS = 20
latN = 80

# if N == 0:
#     cint = np.arange(-3,3.5,0.5)
# elif N == 1:
#     cint = np.arange(-1,1.2,0.2)
# elif N == 2:
#     cint = np.arange(-2,2.1,0.1)


# Plot the EOF
var1,lon1 = add_cyclic_point(varplot,lon) 

plt.style.use('ggplot')
fig,ax= plt.subplots(1,1,figsize=(6,4))

p = ccrs.LambertConformal(central_longitude=-20,central_latitude=45,cutoff=0)

ax = plt.axes(projection=p)
ax.set_extent([lonW,lonE,latS,latN],crs=ccrs.PlateCarree())

# Add filled coastline
ax.coastlines()

# Add contours
if len(cint) == 1:
    cs = ax.contourf(lon1,lat,var1,cmap=cmap,transform=ccrs.PlateCarree())
else:
    cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,transform=ccrs.PlateCarree())
    # Negative contours
    cln = ax.contour(lon1,lat,var1,
                cint[cint<0],
                linestyles='dashed',
                colors='k',
                linewidths=0.5,
                transform=ccrs.PlateCarree())
    
    # Positive Contours
    clp = ax.contour(lon1,lat,var1,
                cint[cint>=0],
                colors='k',
                linewidths=0.5,
                transform=ccrs.PlateCarree()
                )  

# Add Gridlines
gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='black',linestyle=':')

gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
bc = plt.colorbar(cs,orientation='horizontal')
ax.set_title("NHFLX - PC%i Regression, ENS %s, " %(N+1,estring),fontsize=16)



#%% Plots just considering heat flux...

# Try Plotting ensemble average
def plot_regression(N,e,varin,lon,lat,cint=[0]):
    # Restrict to Ensemble and PC
    if e == "avg":
        # Assumed [pc, ens, lat, lon]
        varplot = np.nanmean(varin,axis=1)[N,...]
        estring = "AVG"
    else:
        varplot = varin[N,e,:,:]
        estring = "%02i"% (e+1)
    
    cmap = cmocean.cm.balance
    #Plotting boundaries
    lonW = -90
    lonE = 40
    latS = 0
    latN = 80

    # if N == 0:
    #     cint = np.arange(-3,3.5,0.5)
    # elif N == 1:
    #     cint = np.arange(-1,1.2,0.2)
    # elif N == 2:
    #     cint = np.arange(-2,2.1,0.1)
    
    
    # Plot the EOF
    var1,lon1 = add_cyclic_point(varplot,lon) 
    
    plt.style.use('ggplot')
    fig,ax= plt.subplots(1,1,figsize=(6,4))
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lonW,lonE,latS,latN])
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,linewidth=0.75,linestyle=":")
    
    # Add contours
    if len(cint) == 1:
        cs = ax.contourf(lon1,lat,var1,cmap=cmap,transform=ccrs.PlateCarree())
    else:
        cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,transform=ccrs.PlateCarree())
        # Negative contours
        cln = ax.contour(lon1,lat,var1,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())
        
        # Positive Contours
        clp = ax.contour(lon1,lat,var1,
                    cint[cint>=0],
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())   
        # Add Label
        plt.clabel(cln,fmt='%i',fontsize=8)
        plt.clabel(clp,fmt='%i',fontsize=8)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    

    bc = plt.colorbar(cs,orientation='horizontal')
    ax.set_title("%s - PC%i Regression, ENS %s, " %(flux,N+1,estring),fontsize=16)


    return fig
N =0
e = 'avg'
cint = np.arange(-50,55,5)
estring = str(e)



fig = plot_regression(N,e,varout,lon,lat,cint=cint)
plt.savefig(outpath+"%s_NAO_EOF%i_Ens%s.png"%(flux,N+1,estring), bbox_inches="tight",dpi=200)


#%% Ensemble and annual average of nhflux


nhflxavg = np.squeeze(nhflx[0,0,...])

varplot = np.copy(nhflxavg)

cmap = cmocean.cm.balance
#Plotting boundaries
lonW = -90
lonE = 40
latS = 20
latN = 80


# Plot the EOF
var1,lon1 = add_cyclic_point(varplot,lon) 

plt.style.use('ggplot')
fig,ax= plt.subplots(1,1,figsize=(6,4))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lonW,lonE,latS,latN])

# Add filled coastline
ax.add_feature(cfeature.COASTLINE,linewidth=0.75,linestyle=":")

# Add contours
if len(cint) == 1:
    cs = ax.contourf(lon1,lat,var1,cmap=cmap,transform=ccrs.PlateCarree())
else:
    cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,transform=ccrs.PlateCarree())
    # Negative contours
    cln = ax.contour(lon1,lat,var1,
                cint[cint<0],
                linestyles='dashed',
                colors='k',
                linewidths=1,
                transform=ccrs.PlateCarree())
    
    # Positive Contours
    clp = ax.contour(lon1,lat,var1,
                cint[cint>=0],
                colors='k',
                linewidths=1,
                transform=ccrs.PlateCarree())   
    # Add Label
    plt.clabel(cln,fmt='%i',fontsize=8)
    plt.clabel(clp,fmt='%i',fontsize=8)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


bc = plt.colorbar(cs,orientation='horizontal')
ax.set_title("NHFLX - PC%i Regression, ENS %s, " %(N+1,estring),fontsize=16)



#%% Testing with another projection

# Restrict to Ensemble and PC
    if e == "avg":
        # Assumed [pc, ens, lat, lon]
        varplot = np.nanmean(varin,axis=1)[N,...]
        estring = "AVG"
    else:
        varplot = varin[N,e,:,:]
        estring = "%02i"%e
    
    cmap = cmocean.cm.balance
    #Plotting boundaries
    lonW = -90
    lonE = 40
    latS = 0
    latN = 80

    # if N == 0:
    #     cint = np.arange(-3,3.5,0.5)
    # elif N == 1:
    #     cint = np.arange(-1,1.2,0.2)
    # elif N == 2:
    #     cint = np.arange(-2,2.1,0.1)
    
    
    # Plot the EOF
    var1,lon1 = add_cyclic_point(varplot,lon) 
    
    plt.style.use('ggplot')
    fig,ax= plt.subplots(1,1,figsize=(6,4))
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lonW,lonE,latS,latN])
    
    # Add filled coastline
    ax.add_feature(cfeature.COASTLINE,linewidth=0.75,linestyle=":")
    
    # Add contours
    if len(cint) == 1:
        cs = ax.contourf(lon1,lat,var1,cmap=cmap,transform=ccrs.PlateCarree())
    else:
        cs = ax.contourf(lon1,lat,var1,cint,cmap=cmap,transform=ccrs.PlateCarree())
        # Negative contours
        cln = ax.contour(lon1,lat,var1,
                    cint[cint<0],
                    linestyles='dashed',
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())
        
        # Positive Contours
        clp = ax.contour(lon1,lat,var1,
                    cint[cint>=0],
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree())   
        # Add Label
        plt.clabel(cln,fmt='%i',fontsize=8)
        plt.clabel(clp,fmt='%i',fontsize=8)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle=':')
    gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    

    bc = plt.colorbar(cs,orientation='horizontal')
    ax.set_title("NHFLX - PC%i Regression, ENS %s, " %(N+1,estring),fontsize=16)
