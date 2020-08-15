#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:05:06 2020

# Script to visualize stochastic model output
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from cartopy import config
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import matplotlib.ticker as mticker
from cartopy.util import add_cyclic_point


#%% Functions



#%% User Edits

# Set Point
lonf    = -30
latf    = 50

# Autocorrelation Parameters
kmon    = 2                 # Lag 0 base month
lags    = np.arange(0,61,1) # Number of lags to include


# Paths ---




#%% Svript Start





#%% Find and calculate autocorrelation at a single point



# Load data if it hasnt been
loaddata = 1
if loaddata == 1:
    dataname = datpath+"stoch_output_1000yr_entrain1_hvar2.npy"
    noentrain = np.load(dataname)

# Find Lat/Lon (can write this into a function)
klon = np.abs(lonr - lonf).argmin()
klat = np.abs(latr - latf).argmin()
msg1 = "For Longitude %.02f, I found %.02f" % (lonf,lonr[klon])
msg2 = "For Latitude %.02f, I found %.02f" % (latf,latr[klat])
print(msg1)
print(msg2)

#Get data for a single point
hcycle_pt = hclim[klon,klat,:]

temp_ts = noentrain[klon,klat,:]
temp_ts = np.reshape(temp_ts,(int(np.ceil(len(temp_ts)/12)),12))
temp_ts = np.transpose(temp_ts,(1,0))

# Calculate Lag Autocorrelation
corr_ts = calc_lagcovar(temp_ts,temp_ts,lags,kmon,0)
    
    
# Plot Correlation
f1 = plt.figure()
ax = plt.axes()
plt.style.use('seaborn')
ax.plot(lags,corr_ts,'c',lw=3,label='Stochastic (No-Entrain)')
ax.legend(prop=dict(size=16))
titlestr = 'SST Anomaly Autocorrelation; \n Month: '+monsfull[kmon-1] + ' | Lon: ' + \
    str(lonf) + ' | Lat: '+ str(latf)
    
ax.set_ylabel('Correlation',fontsize=14)
ax.set_ylim(-0.2,1.1)
ax.set_xlabel('Lag (months)',fontsize=14)
ax.set_title(titlestr,fontsize=20)

# Plot Seasonal MLD
fmx = plt.figure()
ax = plt.axes()
ax.plot(range(1,13),hcycle_pt)
ax.legend()
ax.set(xlim=(1,12),ylim=(0,200),xlabel='Months',ylabel='MLD(m)')



# Plot a map of the climatological MLD
mon = 12
cmap = cmocean.cm.balance
bbox = [-75,5,0,65]
hplot = hclim[:,:,mon-1]
var = np.transpose(np.copy(hplot),(1,0))


# Add cyclic point to avoid the gap
var,lon1 = add_cyclic_point(var,coord=lonr)


# Set up projections and extent
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(bbox)

# Add filled coastline
ax.add_feature(cfeature.COASTLINE,facecolor='k')


# Draw contours
cs = ax.contourf(lon1,latr,var,cmap=cmap)            
# Add Gridlines
gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')
gl.xlabels_top = gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
bc = plt.colorbar(cs)



#%% Attempt at animation....

import matplotlib.animation as animation

animationname = outpath + "test.mp4"


fig,ax = make_figure()

frames    = t_end # Number of frames
min_value =# Lowest Value
max_value = # Highest Value

def draw(frame,add_colorbar):
    grid = 

    