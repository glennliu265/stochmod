#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute Monthly Variance over a particular region

Copied upper section from pointwise autocorrelation script\
Copied experiment loading sections from compare_Tddamp

Copied sections from viz_inputs_point


Created on Tue Nov 28 21:04:54 2023

@author: gliu

"""

import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cartopy.crs as ccrs

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20231208/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)
#%% Import Parameters for Stochmod

cwd = os.getcwd()
sys.path.append(cwd+"/../")
import stochmod_params as sparams

#%% User Edits

# Get the list of files
expname      = "default" # Indicate Experiment name (see stochmod_params)
flist        = sparams.rundicts[expname]
print(flist)
continuous   = True


#%% Load the files, processing by runid

#bbox_sel = [-45,-10,50,60]
bbox_sel = [-31,-30,49,50]
debug    = False

f        = 0
ssts     = []
ssts_reg = []
for f in range(len(flist)):
    # Load Information for the run
    expid   = flist[f]
    ld      = np.load(datpath+"stoch_output_%s.npz"%(expid),allow_pickle=True)
    sst     = ld['sst'] # [hconfig x lon x lat x time]
    lon     = ld['lon']
    lat     = ld['lat']
    
    # Select region
    sst      = sst.transpose(1,2,3,0) # [lon z lat x time x hconfig]
    nlon,nlat,ntime,nmodels=sst.shape
    sst      = sst.reshape(nlon,nlat,ntime*nmodels) 
    sst_ravg = proc.sel_region(sst,lon,lat,bbox_sel,reg_avg=True,awgt=1)
    sst_ravg = sst_ravg.reshape(ntime,nmodels)
    ssts.append(sst_ravg)
    
    
    #sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
    #ssts_reg.append(sstr)
    
    # Check 
    if debug:
        sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
        ax = viz.add_coast_grid(ax,fill_color="k")
        ax.set_extent([-80,0,0,65])
        pcm = ax.pcolormesh(lonr,latr,sstr[:,:,0].T)

ssts     = np.concatenate(ssts,axis=0) # [time x hconfig]
#ssts_reg = np.concatenate(ssts_reg,axis=2) # [Lon x Lat x Hconfig]

#%% Load SOM and FCM (use version processed by viz_AMV_CESM.py)

fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FCM","SOM"] 
mcolors    = ["k","gray"]

#datpathc    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
sstscesm =[]
for f in range(2):
    varname = "SST"
    sst_fn = fnames[f]
    ds   = xr.open_dataset(datpath+"../"+sst_fn)
    ds   = ds.sel(lon=slice(bbox_sel[0],bbox_sel[1]),lat=slice(bbox_sel[2],bbox_sel[3]))
    lonc = ds.lon.values
    latc = ds.lat.values
    sstc = ds[varname].values # [lon x lat x time]
    sstscesm.append(sstc)


# Compute the mean
sstscesm = [proc.area_avg(s,bbox_sel,lonc,latc,1) for s in sstscesm]

# Reshape
#nlonc,nlatc,_ = sstscesm[0].shape
ntimes_cesm = [s.shape[0] for s in sstscesm]
nyrs_cesm   = [int(t/12) for t in ntimes_cesm]
sstscesm = [sstscesm[ii].reshape(nyrs_cesm[ii],12) for ii in range(2)]
#%% Load non-processed version of CESM1 data (no land ice mask)

# Load Raw SST
sstsc     = []
mconfigs = ["FULL","SLAB"]
for i in range(2):
    # Open the file
    ds = xr.open_dataset(datpath+"../CESM_proc/"+"TS_anom_PIC_%s.nc"%(mconfigs[i]))
    if i == 0:
        ds = ds.sel(time=slice('0800-02-01','2201-01-01'))
        
        
    # Select the region
    ds = proc.format_ds(ds,)
    ds = proc.sel_region_xr(ds,bbox_sel)
    sst = ds.TS.values
    
    sstsc.append(sst)


# Do preproc
sstsc_proc = []
for sst in sstsc:
    
    # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
    sst = sst.transpose(2,1,0)
    
    # Flip longitude
    # st = time.time()
    # lon180,sst = proc.lon360to180(lon360,sst)
    # print("Flipped Longitude in %.2fs"%(time.time()-st))
    
    # Remove monthly anomalies
    st = time.time()
    nlon,nlat,ntime = sst.shape
    sst = sst.reshape(nlon,nlat,int(ntime/12),12)
    ssta = sst - sst.mean(2)[:,:,None,:]
    print("Deseasoned in %.2fs"%(time.time()-st))
    print("Mean was %e" % (np.nanmax(ssta.mean(2))))
    ssta = ssta.reshape(nlon,nlat,int(ntime/12)*12)
    
    
    sstsc_proc.append(ssta)

lonrc = ds.lon.values
latrc = ds.lat.values


# Remove Points if option is set
remove_points=True
if remove_points:
    
    sstsc_proc_in = []
    for mc in range(2):
        
        #sstr,lonrc,latrc = proc.sel_region(sstsc_proc[mc],lon180,latglob,bbox_sel,)
        sst_rem = sstsc_proc[mc]
        # for o in range(2):
        #     sst_rem[o,-1,:] = np.nan
        sstsc_proc_in.append(sst_rem)
        
    # Compute the mean over the rgion
    # Compute the mean
    sstscesm = [proc.area_avg(s,bbox_sel,lonrc,latrc,0) for s in sstsc_proc_in]
    
    
else:
    
    sstsc_proc_in = sstsc_proc


    # Compute the mean over the rgion
    # Compute the mean
    sstscesm = [proc.area_avg(s,bbox_sel,lon180,latglob,1) for s in sstsc_proc_in]



if debug:
    # Check Region
    #sstr,lonrc,latrc = proc.sel_region(sstsc_proc[0],lonrc,latrc,bbox_sel,)
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                          constrained_layout=True,figsize=(12,4))
    ax = viz.add_coast_grid(ax,fill_color="k")
    ax.set_extent([-80,0,40,65])
    pcm = ax.pcolormesh(lonrc,latrc,np.std(sstsc_proc[0],2).T,cmap="inferno")
    fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01)
    
    #viz.plot_box(bbox_sel,color="y")
    
    for xx in range(4):
        ax.scatter(lonrc[xx],latrc[-1],marker="x",color="k")


ntimes_cesm = [s.shape[0] for s in sstscesm]
nyrs_cesm   = [int(t/12) for t in ntimes_cesm]
sstscesm = [sstscesm[ii].reshape(nyrs_cesm[ii],12) for ii in range(2)]

#%% Compute the monthly variance

# Remove Seasonal Cycle
scycle,ssts = proc.calc_clim(ssts,0,returnts=1) # [Year x Mon x Hconfig]
sstsa = ssts - scycle[None,:,:]

# Remove Seasonal Cycle for CESM1

# nlonr,nlatr,_ = ssts_reg.shape
# ssts_re
# scycler,ssts_reg = proc.calc_clim(ssts_reg,2,returnts=1)# [lon x lat x time x 12]

#%% Plot the monthly variance!

mons3  = proc.get_monstr(3)
fig,ax = plt.subplots(1,1,figsize=(6,4.5),constrained_layout=True)

# Plot CESM
cesmvars = []
for cc in range(2):
    plotvar = np.var(sstscesm[cc],0)
    ax.plot(mons3,plotvar,c=mcolors[cc],label=mnames[cc],marker="o")
    cesmvars.append(plotvar)
    
sstvars = []
for hc in range(nmodels):
    
    plotvar = np.nanvar(ssts,0)[:,hc]
    ax.plot(mons3,plotvar,c=sparams.mcolors[hc],label=sparams.modelnames[hc],marker="o")
    sstvars.append(plotvar)
ax.legend()
ax.set_ylabel("Monthly Variance ($\degree C^2$)")

ax.set_title("Seasonal Distribution of Variance (Stochastic Model)",fontsize=14)
#ax = viz.add_ticks()
ax.grid(True,ls='dotted')
#ax.set_ylim([0,0.8])


savename = "%sStochastic_Model_Hierarchy_Monthly_Variance.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')


#%% Look At timeseries

fig,ax = plt.subplots(1,1)

ax.plot(sstscesm[:,0])
ax

#%% Save the monthly variance

sstvars = np.array(sstvars)


coords_dict = {'hierarchy_level': list(sparams.modelnames),
               'mon'            : mons3}


da_monvar = xr.DataArray(sstvars,coords=coords_dict,dims=coords_dict,name="monthly_variance")
#edict     = proc.make_encoding_dict(da_monvar)
savename  = "%sMonthly_Variance_Stochastic_Model.nc" % figpath
da_monvar.to_netcdf(savename)

"#%% Investigate Forcing/Damping/MLD Terms over this same region
input_path  = datpath + '../model_input/'
frcname     = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
method      = 4
lagstr      = 'lag1'

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs



#%% Save for when plot is a single point ()






vars_pt = np.concatenate([np.array(cesmvars),np.array(sstvars)],axis=0) # [model x 12]
names   = np.hstack([mnames,sparams.modelnames])
savename_pt = "%sSPG_Point_Monthly_Variance_PiC.npz" % (figpath)
    np.savez(savename_pt,**{
        'monvar':vars_pt,
        'names' : names,
        'mons3' : mons3,
        'lon'   : lonr,
        'lat'   : latr
        },allow_pickle=True)



#%% Restrict to Region and Take the Regional Average

invars      = [h,dampingslab,dampingfull,np.sum(alpha**2,2),np.sum(alpha_full**2,2)]

invars_ravg = [proc.sel_region(v,lon,lat,bbox_sel,reg_avg=True,awgt=1) for v in invars]


#%% Plot the monthly variance for each of the terms

fig,axs = plt.subplots(3,1,figsize=(6,8),constrained_layout=True)

for v in range(3):
    
    ax = axs[v]
    
    if v == 0:
        ax.plot(mons3,invars_ravg[0],color="magenta")
        ylab = "Mixed layer Depth \n (m)"
        
    elif v == 1:
        ax.plot(mons3,invars_ravg[1],label="Damping (Slab)",color='blue')
        ax.plot(mons3,invars_ravg[2],label="Damping (Full)",color='red')
        ylab = "Damping \n ($W m^{-2}  \degree C^{-1}$)"
        
    elif v == 2:
        ax.plot(mons3,invars_ravg[3],label="Forcing (Slab)",color='blue')
        ax.plot(mons3,invars_ravg[4],label="Forcing (Full)",color='red')
        ylab = "Forcing Amplitude \n $(W m^{-2})^2$"
    ax.legend()
    ax.set_ylabel(ylab)
    ax.grid(True,ls='dotted')

savename = "%sStochastic_Model_Hierarchy_Monthly_Variance_Inputs_Separate.png" % figpath
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%%


