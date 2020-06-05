#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test to Visualize MLD Data from nc file
Created on Fri Jun  5 12:18:32 2020

@author: gliu
"""


import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Plotting information
bbox = [-100,20,-20,80]
imon  = 12
clevs = np.concatenate([np.arange(0,200,50),np.arange(200,550,50)])
dsource = 'cesm_hmxl'



projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
ncpath   = projpath + "01_Data/"
outpath = projpath + "02_Figures/20200605/'"

if dsource == 'argo':
    # Argo MLD Climatology: http://mixedlayer.ucsd.edu/
    ncname = "Argo_mixedlayers_monthlyclim_12112019.nc"
    mldname = "mld_da_mean"
    lonname = "lon"
    latname = "lat"
    monname = "iMONTH"
elif dsource == 'cesm_hmxl':
    ncname  = "HMXL_HTR_clim.nc"
    mldname = "HMXL"
    lonname = "TLONG"
    latname = "TLAT"
    monname = "month"
elif dsource == 'cesm_xmxl':
    ncname  = "XMXL_HTR_clim.nc"
    mldname = "XMXL"
    lonname = "TLONG"
    latname = "TLAT"
    monname = "month"
    


# -----------------------------------------
# Open dataset and read in variables
ncdat    = xr.open_dataset(ncpath+ncname)

# Read in variables
lon = ncdat[lonname]
lat = ncdat[latname]
mld = ncdat[mldname]

if dsource == 'cesm_hmxl' or dsource == 'cesm_xmxl':
    lon = lon.isel(ensemble=0)
    lat = lat.isel(ensemble=0)
    mld = mld.mean('ensemble')/100
    #mld = np.transpose(mld,(2,0,1))


# Make Plot
fig = plt.figure(figsize=(15,5))
ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())

# Need to somehow generalize this
if dsource == 'cesm_hmxl' or dsource == 'cesm_xmxl':
    cf = ax.contourf(lon,lat,mld.isel(month=imon-1),
                        levels=clevs,
                        cmap='ocean',
                        transform=ccrs.PlateCarree())
    # cl = ax.contour(lon,lat,mld.isel(month=imon-1),
    #                 levels=clevs,
    #                 linewidth=0.5,
    #                 colors=["black"])
elif dsource == 'argo':
    cf = ax.contourf(lon,lat,mld.isel(iMONTH=imon-1),
            levels=clevs,
            cmap='ocean',       
            transform=ccrs.PlateCarree())
    # cl = ax.contour(lon,lat,mld.isel(iMONTH=imon-1),
    #         levels=clevs,           
    #         colors=["black"])

# Add coastlines
#ax.coastlines()
land = ax.add_feature(
    cfeature.NaturalEarthFeature(
        'physical','land','110m',facecolor='black'))

# Format gridlines
gl = ax.gridlines(draw_labels='true',
                  linewidth=1,
                  color='gray',
                  alpha=1
                  )
gl.xlabels_top  =0
gl.ylabels_right=0          
             

ax.set_extent(bbox)

# Add colorbar
cb = plt.colorbar(cf,ticks=clevs,orientation="vertical")
cb.ax.set_title("Mixed Layer Depth (m)")

# Set labels
ax.set_title("Climatological MLD ("+ dsource +") \n Mon"+str(imon))
ax.set(xlabel="Longitude",ylabel="Latitude")

plt.gcf()

outname = 'Climatological_MLD_' + dsource + ".png"
plt.savefig(outpath+outname, bbox_inches="tight",dpi=200)
#plt.show()


