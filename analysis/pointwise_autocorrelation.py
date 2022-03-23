#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compute pointwise autocorrelation for CESM or stochastic model outputs
Support separate calculation for warm and cold anomalies

Based on pointwise_autocorrelation.py

Created on Thu Mar 17 17:09:18 2022

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

import numpy as np
import xarray as xr
from tqdm import tqdm 
import time

import cartopy.crs as ccrs
#%% Select dataset to postprocess

# Autocorrelation parameters
lags   = np.arange(0,37)
lagname = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

# Set Output Directory
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220317/'
proc.makedir(figpath)


# Postprocess Continuous SM  Run
# ------------------------------
datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
outpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
onames      = ["spectra_%s_Fprime_rolln0_ampq0_method5_dmp0_run2%02d.nc" % (lagname,i) for i in range(10)]
mnames      = ["constant h","vary h","entraining"] 

# Postproess Continuous CESM Run
# ------------------------------
datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
outpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/proc/"
fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FULL","SLAB"] 
onames     = ["spectra_%s_PIC-%s.nc" % (lagname,mnames[i]) for i in range(2)]

# Other Params
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

#%% Read in the data

sst_fn = fnames[0]
print("Processing: " + sst_fn)

# Load in SST [model x lon x lat x time] Depending on the file format
if 'npy' in sst_fn:
    print("Loading .npy")
    sst = np.load(datpath+sst_fn)
    # NOTE: Need to write lat/lon loader
elif 'npz' in sst_fn:
    print("Loading .npz")
    ld  = np.load(datpath+sst_fn,allow_pickle=True)
    lon = ld['lon']
    lat = ld['lat']
    sst = ld['sst'] # [model x lon x lat x time]
elif 'nc' in sst_fn:
    print("Loading netCDF")
    ds  = xr.open_dataset(datpath+sst_fn)
    
    ds  = ds.sel(lon=slice(-80,0),lat=slice(0,65))
    
    
    lon = ds.lon.values
    lat = ds.lat.values
    sst = ds.SST.values


#%% Set some more things...

thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])

varname = "SST"
mconfig = "PIC-FULL"
savename   = "%sCESM1_%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)

#%% Do the calculations
"""
Inputs are:
    1) variable [lon x lat x time]
    2) lon      [lon]
    3) lat      [lat]
    4) thresholds [Numeric] (Standard Deviations)
    5) savename [str] Full path to output file
    
"""
# First things first, combine lat/lon, remove nan points
# Get Dimensions
nlon,nlat,ntime = sst.shape
nyr             = int(ntime/12)
npts            = nlon*nlat
nlags           = len(lags)
nthres          = len(thresholds)

# Combine space, remove NaN points
sst                  = sst.reshape(npts,ntime)
sst_valid,knan,okpts = proc.find_nan(sst,1) # [finepoints,time]
npts_valid           = sst_valid.shape[0] 

# Split to Year x Month
sst_valid = sst_valid.reshape(npts_valid,nyr,12)

# Preallocate
class_count = np.zeros((npts_valid,12,nthres+1)) # [pt x eventmonth x threshold]
sst_acs     = np.zeros((npts_valid,12,nthres+1,nlags))  # [pt x eventmonth x threshold x lag]
sst_cfs     = np.zeros((npts_valid,12,nthres+1,nlags,2))  # [pt x eventmonth x threshold x lag x bounds]

# A pretty ugly loop....
# Now loop for each month
for im in range(12):
    print(im)
    
    # For that month, determine which years fall into which thresholds [pts,years]
    sst_mon = sst_valid[:,:,im] # [pts x yr]
    sst_mon_classes = proc.make_classes_nd(sst_mon,thresholds,dim=1,debug=False)
    
    for th in range(nthres+1): # Loop for each threshold
        for pt in tqdm(range(npts_valid)): # Loop for each point
            
            # Get years which fulfill criteria
            yr_mask     = np.where(sst_mon_classes[pt,:] == th)[0]
            sst_in      = sst_valid[pt,yr_mask,:] # [year,month]
            sst_in      = sst_in.T
            class_count[pt,im,th] = len(yr_mask) # Record # of events 
            
            # Compute the lagcovariance (with detrending)
            ac = proc.calc_lagcovar(sst_in,sst_in,lags,im+1,1) # [lags]
            cf = proc.calc_conflag(ac,conf,tails,len(yr_mask)) # [lags, cf]
            
            # Save to larger variable
            sst_acs[pt,im,th,:] = ac.copy()
            sst_cfs[pt,im,th,:,:]  = cf.copy()
            # End Loop Point
            
        # End Loop Threshold
    # End Loop Event Month

# Now Replace into original matrices

# Preallocate
count_final = np.zeros((npts,12,nthres+1)) * np.nan
acs_final   = np.zeros((npts,12,nthres+1,nlags)) * np.nan
cfs_final   = np.zeros((npts,12,nthres+1,nlags,2)) * np.nan

# Replace
count_final[okpts,...] = class_count
acs_final[okpts,...]   = sst_acs
cfs_final[okpts,...]   = sst_cfs

# Reshape output
count_final = count_final.reshape(nlon,nlat,12,nthres+1)
acs_final   = acs_final.reshape(nlon,nlat,12,nthres+1,nlags)
cfs_final   = cfs_final.reshape(nlon,nlat,12,nthres+1,nlags,2)

# Save Output
np.savez(savename,**{
    'class_count' : count_final,
    'acs' : acs_final,
    'cfs' : cfs_final,
    'thresholds' : thresholds,
    'lon' : lon,
    'lat' : lat,
    'lags': lags
    },allow_pickle=True)


#%% Try to plot a point

savename = "%sCESM1_PIC-FULL_SST_autocorrelation_thres%s.npz" %  (outpath,nthres)
ld = np.load(savename,allow_pickle=True)


colors = ['b','r']

count_final = ld['class_count']
acs_final   = ld['acs']
lon         = ld['lon']
lat         = ld['lat']

thresholds  = ld['thresholds']
thresname   = []
if nthres == 1:
    thresname.append("$T'$ <= %i"% thresholds[0])
    thresname.append("$T'$ > %i" % thresholds[0])
else:
    for th in range(nthres):
        thval= thresholds[th]
        
        if thval != 0:
            sig = ""
        else:
            sig = "$\sigma$"
        
        if th == 0:
            tstr = "$T'$ <= %i %s" % (thval,sig)
        elif th == nthres:
            tstr = "$T'$ > %i %s" % (thval,sig)
        else:
            tstr = "%i < $T'$ =< %i %s" % (thresholds[th-1],thval,sig)
        thresname.append(th)
        

lonf = -30
latf = 50

klon,klat = proc.find_latlon(lonf,latf,lon,lat)

kmonth = 4

fig,ax =plt.subplots(1,1)

ax,ax2 = viz.init_acplot(kmonth,np.arange(0,38,2),lags,ax=ax)
for th in range(nthres+1):
    ax.plot(lags,acs_final[klon,klat,kmonth,th,:,kmonth],marker="o",
            color=colors[th],lw=2,
            label="%s (n=%i)" % (thresname[th],count_final[klon,klat,kmonth,th]))
    #ax.plot(lags,acs_final[klon,klat,kmonth,0,:,kmonth],label="Cold Anomalies (%i)" % (count_final[klon,klat,kmonth,0]),color='b')
    #ax.plot(lags,acs_final[klon,klat,kmonth,1,:,kmonth],label="Warm Anomalies (%i)" % (count_final[klon,klat,kmonth,th]),color='r')
    #ax.legend()

ax.legend()

plt.savefig("%sAutocorrelation_WarmCold.png"%figpath,dpi=150)
#%% load data for MLD

ksel = 2#"max"

# Use the function used for sm_rewrite.py
frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
input_path = datpath + "model_input/"
lagstr     = "lag1"
method     = 5 # Damping Correction Method
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
long,latg,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt   = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt   = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]

# Compute Specific Region
reg_sel  = [lon[0],lon[-1],lat[0],lat[-1]]
#reg_sel   = [-80,0,30,65]
inputs = [h,kprevall,dampingslab,dampingfull,alpha,alpha_full,hblt]
outputs,lonr2,latr2 = scm.cut_regions(inputs,long,latg,reg_sel,0)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs


if ksel == 'max':
    
    # Get indices of kprev
    hmax = np.argmax(h,axis=2)
    
    # Select autocorrelation function of maximum month
    acmax = np.zeros((nlon,nlat,nthres+1,nlags))*np.nan
    for o in tqdm(range(nlon)):
        for a in range(nlat):
            kpt = hmax[o,a,]
            acmax[o,a,...] = acs_final[o,a,kpt,:,:,kpt]
else:
    acmax = acs_final[:,:,ksel,:,:,ksel]

#% Integrate
integr_ac = np.trapz(acmax,x=lags,axis=-1)

#%% Visualize (Works for warm/cold)
bboxplot = [-80,0,0,60]

intitles = ("Cold","Warm")

fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True,figsize=(14,6))
for th in range(3):
    ax = axs[th]
    
    if th < nthres+1:
        plotac = integr_ac[:,:,th].T
        title  = "%s Anomalies (%s K)" % (intitles[th],thresname[th])
        cmap   = "cmo.amp"
        vlm    = [0,10]
    else:
        plotac = (integr_ac[:,:,1] - integr_ac[:,:,0]).T
        title  = "Warm - Cold Anomalies"
        cmap   = "cmo.balance"
        vlm    = [-3,3]
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    ax.set_title(title)
    pcm = ax.pcolormesh(lon,lat,plotac,cmap=cmap,
                        vmin=vlm[0],vmax=vlm[1])
    
    if th == 2:
        cl = ax.contour(lon,lat,plotac,levels=[0,],colors="k",linewidths=.75)
        
    else:
        cl = ax.contour(lon,lat,plotac,levels=np.arange(0,14,2),colors="w",linewidths=.75)
    ax.clabel(cl,)
    
    if th == 0:
        cb = fig.colorbar(pcm,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.04)
        cb.set_label("Integrated Timescale (Months)")
    elif th == 2:
        cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04,pad=0.01)
        cb.set_label("Difference (Months)")
plt.suptitle("Integrated ACF (to 36 months) for CESM1-FULL PiC",fontsize=14,y=.88)
plt.savefig("%sIntegratedACF_Cold_Warm_Anomalies_CESM_PiC_ksel%i.png"%(figpath,ksel),dpi=150)
#%%

"""
Just realized it might not be possible to neatly vectorize this.

This is because at each point, there will be a different # of points within
each threshold, so it does not fit neatly into a 2d maxtrix...

"""

# Split into negative or positive anomalies


sst_classes = proc.make_classes_nd(sst_valid,thresholds,dim=1,debug=True)

# Now compute the autocorrelation for each lag, and for each case (positive/negative)
sst_acs = np.zeros(npts_valid,12,nthres+1,nlags) # [point,basemonth,threshold,lag]
for th in range(nthres+1): #Loop for each threshold

    sst_mask = (sst_classes == th)
    sst_mask = np.where(sst_classes == th)
    
    for im in range(12):
        
        insst = np.take_along_axis(sst_valid,sst_mask,1)
        
        #st_valid[np.where(sst_mask)]
        
        # [mon x time]
        sst_acs[:,im,th,:] = proc.calc_lagcovar_nd(insst,insst,lags,im+1,0)
        
        #autocorrm[m,:,:] = proc.calc_lagcovar_nd(oksst,oksst,lags,m+1,0)
    
    
    
        
        






#%%

thresholds = [-1,0,1]
y = sst_valid
