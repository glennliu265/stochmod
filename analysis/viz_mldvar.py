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
figpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20230519/"
proc.makedir(figpath)
ncname = "HMXL_FULL_PIC_bilinear.nc"


months = proc.get_monstr()

# Set Selection BBOX
bbox           = [-100,20,0,75]
bboxplot       = [-80,0,0,67]

#%% Load the data in...
ds    = xr.open_dataset(datpath+ncname).load()
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

scycle          = varr.mean(2)


#%% Look at the wintertime variability

# Get the winter component
varr_winter = varr[:,:,:,[10,11,0]].reshape(nlon,nlat,nyr*3)

# Compute wintertime standard deivation
stdvar_winter = np.std(stdvar,2)


#%%

plotvar = stdvar_winter[:,:].T/100
vlms          = [0,300]
clins         = np.arange(100,1100,100)
plot_bbox_spg = [-65,-5,42,68]

fig,ax   = plt.subplots(1,1,constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,4))
ax       = viz.add_coast_grid(ax,bbox=plot_bbox_spg,fill_color="gray")
pcm      = ax.pcolormesh(lonr,latr,plotvar,cmap="cmo.deep",vmin=vlms[0],vmax=vlms[1])
cl= ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w")
ax.clabel(cl)
fig.colorbar(pcm,ax=ax,fraction=0.021,pad=0.01)
ax.set_title("1$\sigma$ Wintertime Mean MLD (m)")
savename = "%sCESM1_PIC_Mean_wintertime_mld.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Multi Panel Wintertime Mean
plot_months = [9,10,11,0,1,2,3,4]

vlms          = [0,500]
clins         = np.arange(100,1100,100)

fig,axs     = plt.subplots(4,2,constrained_layout=True,
                       subplot_kw={'projection': ccrs.PlateCarree()},figsize=(12,12))

for ia in range(len(plot_months)):
    
    ax = axs.flatten()[ia]
    im = plot_months[ia]
    plotvar= stdvar[:,:,im].T/100
    
    blabel = [0,0,0,0]
    if ia%2 == 0:
        blabel[0] = 1
    if ia>5:
        blabel[-1]=1
    
    ax       = viz.add_coast_grid(ax,bbox=plot_bbox_spg,fill_color="gray",blabels=blabel)
    pcm      = ax.pcolormesh(lonr,latr,plotvar,cmap="cmo.deep",vmin=vlms[0],vmax=vlms[1])
    cl= ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w")
    ax.clabel(cl)
    
    #ax.set_title("Month %i" % (im+1))
    ax = viz.label_sp(mons3[im],ax=ax,alpha=0.95,labelstyle="%s",usenumber=True,fontsize=14)
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
cb.set_label("$1\sigma$ Mixed Layer Depth (m)")
    


savename = "%sInterannMLDVar_selected_months_%ito%i.png" % (figpath,plot_months[0],plot_months[-1])
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Look at the standard variation versus the seasonal cycle

# Get the maximum interannual variable and month for each point
stdvar_max     = np.nanmax(stdvar,2)
stdvar_max_mon = np.argmax(stdvar,2)+1


# Compute the range of the seasonal cycle
scycle_range   = np.ptp(scycle,axis=2)



# Set up the plot
blabel     = [1,0,0,1]
def init_spg_plot(bbox_spg=[-65, -5, 42, 68],blabel=[1,0,0,1]):
    fig,ax     = plt.subplots(1,1,constrained_layout=True,
                           subplot_kw={'projection': ccrs.PlateCarree()},figsize=(8,3.5))
    ax         = viz.add_coast_grid(ax,bbox=bbox_spg,fill_color="gray",blabels=blabel)
    return fig,ax


# Plot the maximum interannual variability
plotvar = stdvar_max.T/100
cmap    = "cmo.deep"
clins   = [100,200,300,400,500,600]
fig,ax  = init_spg_plot()
ax.set_title("Interannual Variability of MLD (m)")
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,vmin=0,vmax=500)
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
cl= ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w")
ax.clabel(cl)
savename = "%sMLD_Interann_Var.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches="tight")

# Check just wintertime MLD variability
plotvar = stdvar_winter.T/100
cmap    = "cmo.deep"
clins   = [100,200,300,400,500,600]
fig,ax  = init_spg_plot()
ax.set_title("Interannual Variability of MLD$_{winter}$ (m)")
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,vmin=0,vmax=500)
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
cl= ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w")
ax.clabel(cl)
savename = "%sMLD_Interann_Var_winter.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches="tight")

# Plot month of max variability
plotvar = stdvar_max_mon.T
clvls   = np.arange(0,13,1)
import matplotlib as mpl
cmap    = plt.cm.gist_ncar
norm = mpl.colors.BoundaryNorm(clvls, cmap.N)
fig,ax  = init_spg_plot()
ax.set_title("Month of max Interannual Variability")
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,norm=norm)
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
savename = "%sMLD_Interann_Var_Mon.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches="tight")

# Plot the maximum scycle
plotvar = scycle_range.T/100
cmap    = "cmo.dense"
clins   = [100,200,300,400,500,600,700,800,900,1000,1200]
fig,ax  = init_spg_plot()
ax.set_title("Max Seasonal Range in MLD (m)")
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,vmin=0,vmax=1500)
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
cl= ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w")
ax.clabel(cl)
savename = "%sMLD_AnnRange.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches="tight")

# Plot the ratio
plotlog = False
if plotlog:
    plotvar = np.log(stdvar_max/scycle_range).T
    vlms = [-1,1]
    title = "$log_{10}$(Interannual Variability / Seasonal Range)"
    clins = [0,1]
else:
    plotvar = (stdvar_max/scycle_range).T
    vlms = [0,2]
    title = "Interannual Variability / Seasonal Range"
    clins = [1,2]
cmap    = "cmo.balance"
#clins   = [100,200,300,400,500,600,700,800,900,1000,1200]
fig,ax  = init_spg_plot()
ax.set_title(title)
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,vmin=vlms[0],vmax=vlms[1])
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
cl      = ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="k")
ax.clabel(cl)
savename = "%sMLD_VarRatio_plotlog%i.png" % (figpath,plotlog)
plt.savefig(savename,dpi=150,bbox_inches="tight")

#%% Compare with HBLT

hblt_slab = np.load(datpath+"../SLAB_PIC_hblt.npy")
hblt_reg,_,_  = proc.sel_region(hblt_slab,lon,lat,bbox)

hmxl_annmean = scycle.mean(2)
hblt_annmean = np.nanmean(hblt_reg,2)

# Compare the values for the subpolar gyre


plotvar = (hmxl_annmean/100 - hblt_annmean).T
cmap    = "cmo.balance"
#clins   = [100,200,300,400,500,600]
fig,ax  = init_spg_plot()
ax.set_title("HBLT (SLAB) - HMXL (Ann. Mean, FULL) (m)")
pcm     = ax.pcolormesh(lonr,latr,plotvar,cmap=cmap,vmin=-150,vmax=150)
cb      = fig.colorbar(pcm,ax=ax,fraction=0.035,pad=0.02)
cl      = ax.contour(lonr,latr,plotvar,levels=clins,linewidths=0.5,colors="w",vmin=-400,vmax=400)
ax.clabel(cl)
savename = "%sMLD_HMXL_minus_HBLT.png" % figpath
plt.savefig(savename,dpi=150,bbox_inches="tight")



#%% Examine characteristics from observational MLD

# Load MIMOC data (from viz_inputs_point.py)
mldpath = datpath + "../MIMOC_ML_v2.2_PT_S/"
#testpath = mldpath + "MIMOC_ML_v2.2_PT_S_MLP_month01.nc"
nclist = glob.glob(mldpath+"*.nc")
nclist.sort()
print(nclist)

# Read in and concatenate by month variable
ds_all = []
for nc in nclist:
    ds = xr.open_dataset(nc)
    print(ds)
    ds_all.append(ds.DEPTH_MIXED_LAYER)
ds_all = xr.concat(ds_all,dim="month")



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


#%% Get h' and make a histogram for a  point

nbins     = 20
lonf      = -30
latf      = 50
selmons   = [11,0,1]
klon,klat = proc.find_latlon(lonf,latf,lonr,latr)
locfn,loctitle = proc.make_locstring(lonf,latf)

monstr = "mons"
for m in selmons:
    monstr += "%s-" % (m+1)
monstr = monstr[:-1]
print(monstr)

varpt    = varr[klon,klat,:,selmons]
varprime = (varpt - varpt.mean(0)[None,:])/100

fig,ax   = plt.subplots(1,1)
ax.hist(varprime.flatten(),bins=nbins,alpha=0.5,edgecolor='k')
ax.set_xlim([-1e2,1e2])
ax.set_xlabel("$h'$ (meters)")
ax.set_ylabel("Count")
ax.grid(True,ls='dotted')
ax.set_title("Histogram of MLD Anomalies in CESM1-PiC (%i bins) \n @ %s Months=%s"% (nbins,loctitle,str(np.array(selmons)+1)))
savename = "%sHistogram_hprime_CESM1-PiC_%s_nbins%i_%s.png" % (figpath,locfn,nbins,monstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot entrainment velocity

nbins = 30

selmons = [11,0,1,6]

varpt    = varr[klon,klat,:,:]
we       = (varpt.flatten() - np.roll(varpt.flatten(),1))/100
we       = we.reshape(nyr,12)
we_bar   = we.mean(0)

id_detrain = np.where(we_bar<0)[0]

we_prime = we - we_bar[None,:]
we_prime[:,id_detrain] = np.nan
we_prime = we_prime[:,selmons]

fig,ax = plt.subplots(1,1)
ax.hist(we_prime.flatten(),bins=nbins,alpha=0.5,edgecolor='k',color='green')
ax.set_xlim([-1.5e2,1.5e2])
ax.set_xlabel("$w_e' \frac{dh}{dt}'$ (meters/month)")
ax.set_ylabel("Count")
ax.grid(True,ls='dotted')

ax.set_title("Histogram of Entrainment Velocity Anomalies in CESM1-PiC (%i bins) \n @ %s Months=%s"% (nbins,loctitle,str(np.array(selmons)+1)))
savename = "%sHistogram_we_CESM1-PiC_%s_nbins%i_%s.png" % (figpath,locfn,nbins,monstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

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
