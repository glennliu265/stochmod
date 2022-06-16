#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize Mixed Layer Depth Variability(and Barotropic Streamfunction)

Created on Wed Jun  8 19:19:05 2022

@author: gliu

"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
import time
import sys

#%% Import my modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import proc,viz
import scm

import importlib
importlib.reload(viz)

#%% Examine Interannual MLD Variability

# Paths
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220609/"

ncname = "HMXL_FULL_PIC_bilinear.nc"


months = proc.get_monstr()

# Set Selection BBOX
bbox           = [-100,20,0,75]
bboxplot       = [-80,0,0,67]

#%% Load the data in...
ds = xr.open_dataset(datpath+ncname)

lon   = ds.lon.values
lat   = ds.lat.values
mld   = ds.HMXL.values
times = ds.time.values



#%%


varr,lonr,latr = proc.sel_region(mld.transpose(2,1,0),lon,lat,bbox) # [lon x lat x time]

nlon,nlat,ntime = varr.shape
nyr             = int(ntime/12)
varr            = varr.reshape(nlon,nlat,nyr,12)

stdvar          = np.std(varr,2)



#%%


im       = 6
vlms     = [0,100]
clins    = np.arange(100,1100,100)

for im in range(12):
    fig,ax   = plt.subplots(1,1,constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,8))
    ax       = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    if vlms is None:
        
        pcm      = ax.pcolormesh(lonr,latr,stdvar[:,:,im].T/100,cmap="cmo.deep")
    else:
        pcm      = ax.pcolormesh(lonr,latr,stdvar[:,:,im].T/100,cmap="cmo.deep",
                                 vmin=vlms[0],vmax=vlms[1])
        
        cl= ax.contour(lonr,latr,stdvar[:,:,im].T/100,levels=clins,linewidths=0.5,colors="w")
        ax.clabel(cl)
    cb = fig.colorbar(pcm,ax=ax)
    cb.set_label("1$\sigma_{MLD}$ (m)",fontsize=14)
    ax.set_title("%s Interannual MLD Variability" % months[im],fontsize=22)
    savename = "%sInterannMLDVar_mon%02i.png" % (figpath,im+1)
    plt.savefig(savename,dpi=150,bbox_inches="tight")

#%%

fig,ax = plt.subplots(4,3,constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCaree()},figsize=(16,8))
# -------------------
#%% Next, Try for BSF
# -------------------
st = time.time()

# Select the region
vname       = "HMXL"

if vname == "BSF":
    cints_clim  = np.arange(-60,65,5)
    cints_std   = np.arange(0,8.5,0.5)
    cmap_clim   = "cmo.curl"
    cmap_std    = "inferno"
elif vname == "HMXL":
    cints_clim  = np.arange(0,1100,100)
    cints_std   = np.arange(0,620,20)
    cmap_clim   = "cmo.dense"
    cmap_std    = "cmo.deep"

ncname = "%s_FULL_PIC_bilinear.nc" % vname
ds     = xr.open_dataset(datpath+ncname)
dsreg  = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))

# Read out the files
vareg    = dsreg[vname].values
lonr   = dsreg.lon.values
latr   = dsreg.lat.values

# Calculate and remove climatology
vbar,tsmonyr = proc.calc_clim(vareg,0,returnts=1)
vprime      = tsmonyr - vbar[None,:,:,:]
print("Computed Anomalies in %.2fs" % (time.time()-st))

# Compute Interannual Variability
vstd = vprime.std(0)

#%% Convert cm --> m
if vname == "HMXL":
    vbar /= 100
    vstd /= 100

#%% 

# Plot the Seasonal Cycle
ax = viz.qv_seasonal(lonr,latr,vbar.transpose(2,1,0),cmap=cmap_clim,
                     bbox=bboxplot,contour=True,cints = cints_clim)
plt.suptitle("%s Climatological Monthly Mean" % (vname),fontsize=18)
plt.savefig("%s%s_ClimSCycle.png"%(figpath,vname),dpi=150,bbox_inches="tight")

#%%

# Plot the Interannual Variability
ax = viz.qv_seasonal(lonr,latr,vstd.transpose(2,1,0),cmap=cmap_std,
                     bbox=bboxplot,contour=True,cints=cints_std)
plt.suptitle("%s Interannual Variability ($\sigma_{%s}$)" % (vname,vname),fontsize=18)
plt.savefig("%s%s_InterAnnVar.png"%(figpath,vname),dpi=150,bbox_inches="tight")


#%% Generate A Sequence of plots

# Seasonal Cycle
for im in range(12):
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                          constrained_layout=True,figsize=(12,8))
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    pcm    = ax.contourf(lonr,latr,vbar[im,:,:],levels = cints_clim,cmap=cmap_clim)
    ax     = viz.label_sp("Mon%02i"%(im+1),usenumber=True,labelstyle="%s",
                          ax=ax,alpha=0.8,fontsize=32)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("Climatological Monthly Mean",fontsize=28)
    plt.savefig("%s%s_ClimSCycle_mon%02i.png"%(figpath,vname,im+1),dpi=150,bbox_inches="tight")

#%% Interann Var
for im in range(12):
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                          constrained_layout=True,figsize=(12,8))
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    pcm    = ax.contourf(lonr,latr,vstd[im,:,:],levels = cints_std,cmap=cmap_std)
    ax     = viz.label_sp("Mon%02i"%(im+1),usenumber=True,labelstyle="%s",
                          ax=ax,alpha=0.8,fontsize=32)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("%s Interannual Variability ($\sigma_{%s}$)" % (vname,vname),fontsize=28)
    plt.savefig("%s%s_InterAnnVar_mon%02i.png"%(figpath,vname,im+1),dpi=150,bbox_inches="tight")
    

#%% Save the files, if desired....

savename = "%s../CESM1_PiC_%s_Clim_Stdev.nc" % (datpath,vname)

varnames  = ("clim_mean","stdev")
varlnames = ("Climatological Monthly Mean","Standard Deviation")

dims     = {'month':np.arange(1,13,1),
              "lat"  :latr,
              "lon"  :lonr}

outvars  = [vbar,vstd]

das = []
for v,name in enumerate(varnames):

    attr_dict = {'long_name':varlnames[v],}
    da = xr.DataArray(outvars[v],
                dims=dims,
                coords=dims,
                name = name,
                attrs=attr_dict
                )
    if v == 0:
        ds = da.to_dataset() # Convert to dataset
    else:
        ds = ds.merge(da) # Merge other datasets
        
    # Append to list if I want to save separate dataarrays
    das.append(ds)

#% Save as netCDF
# ---------------
st = time.time()
encoding_dict = {name : {'zlib': True} for name in varnames} 
print("Saving as " + savename)
ds.to_netcdf(savename,
         encoding=encoding_dict)
print("Saved in %.2fs" % (time.time()-st))
