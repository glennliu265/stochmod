#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Visualizes the output of regress_NAO_Pattern.py

Created on Wed Sep  2 11:29:20 2020

@author: gliu
"""

from scipy import signal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point

#%% User Edits

# Indicate flux type
flux = 'NHFLX' # [ 'FLNS','FSNS','LHFLX','SHFLX','RHFLX','THFLX','NHFLX','PSL']

# Path to data
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath  = projpath + '01_Data/'
outpath  = projpath+'/02_Figures/Scrap/'



#%% Functions

def plot_regression_slp(N,e,varin,slp,lon,lat,cint=[0]):
    # Restrict to Ensemble and PC
    if e == "avg":
        # Assumed [pc, ens, lat, lon]
        varplot = np.mean(varin,axis=1)[N,...]
        slpplt = np.mean(slp,axis=0)[:,:,N]
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
    fig,ax= plt.subplots(1,1,figsize=(4,3))
    
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
    

    bc = plt.colorbar(cs,orientation='vertical',fraction=0.046,pad=0.05)
    ax.set_title("%s - PC%i Regression, ENS %s, " %(flux,N+1,estring),fontsize=12)


    return fig

def plot_regression(N,e,varin,lon,lat,cint=[0]):
    # Restrict to Ensemble and PC
    if e == "avg":
        # Assumed [pc, ens, lat, lon]
        varplot = np.mean(varin,axis=1)[N,...]
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

#%% Load Data

# Load Results from EOF Analysis
npzdata = np.load(datpath+"Manual_EOF_Calc_NAO_corr.npz")
eofall    = npzdata['eofall']    # [ens x lat x lon x pc]
pcall     = npzdata['pcall']     # [ens x yr x pc]
varexpall = npzdata['varexpall'] # [ens x pc]


# Load resultant NHFLX patterns calculated in regress_NAO_pattern.py
outname = datpath+"model_input/NAO_EAP_%s_ForcingDJFM.npy" % flux 
varout = np.load(outname)


# Read in Coordinate values
ncname = "%s_NAOproc.nc" % flux
ds = xr.open_dataset(datpath+ncname)
flx = ds[flux]
lon = flx['lon'].values
lat = flx['lat'].values
time = flx['year'].values
flx = flx.values


#%% Plot NHFLX Forcing [Colors] and SLP EOF [Contours] (ensemble average)

N =1
e = 'avg'
if flux == 'RHFLX':
    cint = np.arange(-10,11,1)
else:
    cint = np.arange(-50,55,5)
estring = str(e)

fig = plot_regression_slp(N,e,varout,eofall,lon,lat,cint=cint)
plt.savefig(outpath+"%s_NAO_EOF%i_Ens%s.png"%(flux,N+1,estring), bbox_inches="tight",dpi=200)

#%% Plot NHFLX Forcing [Colors] and SLP EOF [Contours] for each ensemble member
#cint = np.arange(-50,55,5)

for e in range(42):
    estring = str(e+1)
    fig = plot_regression_slp(0,e,varout,eofall,lon,lat,cint=cint)
    plt.savefig(outpath+"%s_NAO_EOF%i_Ens%s.png"%(flux,N+1,estring), bbox_inches="tight",dpi=200)

#%% Plot NHFLX Forcing (colors and contours), but using a different projection...

varin = np.copy(varout)

# Restrict to Ensemble and PC
if e == "avg":
    # Assumed [pc, ens, lat, lon]
    varplot = np.mean(varin,axis=1)[N,...]
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

#%% Plots just considering heat flux regression result...

# Try Plotting ensemble average
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
        varplot = np.mean(varin,axis=1)[N,...]
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