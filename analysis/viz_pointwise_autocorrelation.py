#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize the output of pointwise_autocorrelation.py


Created on Thu Mar 24 15:23:20 2022

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
# --------------------------
lags        = np.arange(0,37)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2

mconfig    = "HTR-FULL" # #"PIC-FULL"

thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SSS"

# Set Output Directory
# --------------------
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220325/'
proc.makedir(figpath)
outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
savename   = "%sCESM1_%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)
print("Output will save to %s" % savename)

# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]

#%% Set Paths for Input (need to update to generalize for variable name)

if mconfig == "SM": # Stochastic model
    # Postprocess Continuous SM  Run
    # ------------------------------
    datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
    fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
    mnames      = ["constant h","vary h","entraining"] 
elif "PIC" in mconfig:
    # Postproess Continuous CESM Run
    # ------------------------------
    datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
    mnames     = ["FULL","SLAB"] 
elif "HTR" in mconfig:
    # CESM1LE Historical Runs
    # ------------------------------
    datpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM_proc/"
    fnames     = ["%s_FULL_HTR_lon-80to0_lat0to65_DTEnsAvg.nc" % varname,]
    mnames     = ["FULL",]
    

#%% Try to plot a point

#
ld = np.load(savename,allow_pickle=True)

count_final = ld['class_count']
acs_final   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]
lon         = ld['lon']
lat         = ld['lat']

thresholds  = ld['thresholds']
threslabs   = ld['threslabs']
#%% Plot autocorrelation at a point

# Select a Point
lonf   = -30
latf   = 50
kmonth = 0

# Get Indices
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstr,locfn = proc.make_locstring(lonf,latf)

# Make the plot
title    = "%s Autocorrelation @ Lon: %i, Lat : %i" % (varname,lonf,latf)
fig,ax   = plt.subplots(1,1)
ax,ax2   = viz.init_acplot(kmonth,np.arange(0,38,2),lags,ax=ax,title=title)

if "PIC" in mconfig: # Just plot the one timeseries
    for th in range(nthres+2):
        ax.plot(lags,acs_final[klon,klat,kmonth,th,:],marker="o",
                color=colors[th],lw=2,
                label="%s (n=%i)" % (threslabs[th],count_final[klon,klat,kmonth,th]))
        #ax.plot(lags,acs_final[klon,klat,kmonth,0,:,kmonth],label="Cold Anomalies (%i)" % (count_final[klon,klat,kmonth,0]),color='b')
        #ax.plot(lags,acs_final[klon,klat,kmonth,1,:,kmonth],label="Warm Anomalies (%i)" % (count_final[klon,klat,kmonth,th]),color='r')
        #ax.legend()
else:
    
    for th in range(nthres+2):
        plotac    = acs_final[klon,klat,:,kmonth,th,:]
        plotcount = count_final[klon,klat,:,kmonth,th].sum()
        ax.plot(lags,plotac.mean(0),marker="o",color=colors[th],lw=2,
                label="%s (n=%i)" % (threslabs[th],plotcount))
        
        ax.fill_between(lags,plotac.min(0),plotac.max(0),
                        color=colors[th],alpha=0.25,zorder=-1,label="")
    

ax.legend()
plt.savefig("%sAutocorrelation_WarmCold_%s_%s_month%i.png"% (figpath,mconfig,locstr,kmonth+1),dpi=150)
#%% load data for MLD

ksel = 'max' #2 #"max"

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
