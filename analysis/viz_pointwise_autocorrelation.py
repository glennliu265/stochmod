#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize the output of pointwise_autocorrelation.py


Includes:
    Reference Month vs. Lag plots for a single point

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
varname    = "SSS" #"SST"

# Set Output Directory
# --------------------
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220502/'
proc.makedir(figpath)
outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
savename   = "%s%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)
print("Loading the following dataset: %s" % savename)

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
elif mconfig == "HadISST":
    # HadISST Data
    # ------------
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    fnames  = ["HadISST_detrend2_startyr1870.npz",]
    mnames     = ["HadISST",]
elif mconfig == "ERSST":
    fnames  = ["ERSST_detrend2_startyr1900_endyr2016.npz"]
    
    
#%% Load in the data
st          = time.time()
ld          = np.load(savename,allow_pickle=True)
count_final = ld['class_count']
acs_final   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]
lon         = ld['lon']
lat         = ld['lat']
thresholds  = ld['thresholds']
threslabs   = ld['threslabs']

nthres      = len(thresholds)
if "HTR" in mconfig:
    lens=True
    nlon,nlat,nens,nmon,_,nlags = acs_final.shape
else:
    lens=False
    nlon,nlat,nmon,_,nlags = acs_final.shape
print("Data loaded in %.2fs"% (time.time()-st))
#%% Plot autocorrelation at a point

# Select a Point
lonf   = -30
latf   = 50
kmonth = 1

# Get Indices
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locfn,locstr = proc.make_locstring(lonf,latf)

# Make the plot
title    = "%s Autocorrelation @ Lon: %i, Lat : %i (%s)" % (varname,lonf,latf,mconfig)
fig,ax   = plt.subplots(1,1)
ax,ax2   = viz.init_acplot(kmonth,np.arange(0,38,2),lags,ax=ax,title=title)

if "HTR" not in mconfig: # Just plot the one timeseries
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

#%% Contour the Autocorrelation (Similar to in Park et al. 2006)
# (Need to add significance levels)

usecontourf = False
clvls       = np.arange(-.1,1.05,0.05)
cmap        = 'cmo.dense'
appendjan   = True

yvals       = np.arange(1,13,1)
# Compute the significance


fig,axs   = plt.subplots(1,3,figsize=(16,4),constrained_layout=True)

for th in range(3):
    
    ax = axs.flatten()[th]
    
    if "HTR" in mconfig:
        plotac    = acs_final[klon,klat,:,:,th,:].mean(0)
        plotcount = count_final[klon,klat,:,:,th].mean(0)
    else:
        plotac    = acs_final[klon,klat,:,th,:]
        plotcount = count_final[klon,klat,:,th]
    
    rhocrit   = proc.ttest_rho(0.05,1,plotcount) # [month]
    sigmask   = plotac > rhocrit[:,None]
    
    #sigmask2 = plotac.copy()
    #sigmask2[~sigmask] = 0
    
    if appendjan: # Add extra january
        yvals       = np.arange(1,14,1)
        mons3       = [viz.return_mon_label(m%12,nletters=3) for m in np.arange(1,14)]
        
        sigmask     = np.concatenate([sigmask,sigmask[[0],:]],axis=0)
        plotac      = np.concatenate([plotac,plotac[[0],:]],axis=0)
        
        
    else:
        yvals       = np.arange(1,13,1)
        mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
    
        
    
    # # Plot it
    if usecontourf:
        cf = ax.contourf(lags,yvals,plotac,levels=clvls,cmap=cmap)
    else:
        cf = ax.pcolormesh(lags,yvals,plotac,vmin=clvls[0],vmax=clvls[-1],
                            shading='auto',cmap=cmap)
        
    # Plot the mask
    # Hatching mask
    # ax.contourf(lags,yvals,sigmask,colors='gray',
    #             hatches=["-",],levels=[-1,0],alpha=0,edgecolor='gray')
    
    # Masking Function with Stippling
    msk = viz.plot_mask(lags,yvals,sigmask.T,
                        ax=ax,markersize=1.0,color="gray",reverse=False)
    
    # Contour Line Plot
    #cl = ax.contour(lags,yvals,sigmask2,colors="k",levels=clvls)
    #ax.clabel(cl)
    
    #ax.pcolormesh(lags,mons3,sigmask,shading='auto')
    
    ax.grid(True,ls='dotted')
    ax.set_xticks(np.arange(0,39,3))
    ax.set_yticks(yvals)
    ax.set_yticklabels(mons3)
    
    if th == 0:
        ax.set_ylabel("Reference Month")
    if th == 1:
        ax.set_xlabel("Lag (Months)")
        
    ax.set_title(threslabs[th])
    
    ax.invert_yaxis()
    
cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.065)
cb.set_label("Correlation")
plt.suptitle("%s Autocorrelation @ %s (%s)"% (varname,locstr,mconfig),fontsize=14,y=1.05)
plt.savefig("%s%sAutocorrelation_Contours_WarmCold_%s_%s.png"% (figpath,varname,mconfig,locfn),dpi=150,bbox_inches='tight')

#%% Same plot as above, but flip the axes

usecontourf = False
clvls       = np.arange(-.1,1.05,0.05)
cmap        = 'cmo.dense'
appendjan   = True
referencex  = True # Flip so that reference month is on the x-axis

# Compute the significance
p     = 0.05
tails = 1 

yvals       = np.arange(1,13,1)



fig,axs   = plt.subplots(1,3,figsize=(10,10),constrained_layout=True)

for th in range(3):
    
    ax = axs.flatten()[th]
    
    if "HTR" in mconfig:
        plotac    = acs_final[klon,klat,:,:,th,:].mean(0)
        plotcount = count_final[klon,klat,:,:,th].mean(0)
    else:
        plotac    = acs_final[klon,klat,:,th,:]
        plotcount = count_final[klon,klat,:,th]
    
    rhocrit   = proc.ttest_rho(p,tails,plotcount) # [month]
    sigstr    = "%i" % ((p)*100) + "%" + r" Sig. (%i-tail): $\rho$ > %.2f (n=%i)" % (tails,rhocrit.mean(),plotcount.mean()) 
    sigmask   = plotac > rhocrit[:,None]
    
    #sigmask2 = plotac.copy()
    #sigmask2[~sigmask] = 0
    
    if appendjan: # Add extra january
        yvals       = np.arange(1,14,1)
        mons3       = [viz.return_mon_label(m%12,nletters=3) for m in np.arange(1,14)]
        
        sigmask     = np.concatenate([sigmask,sigmask[[0],:]],axis=0)
        plotac      = np.concatenate([plotac,plotac[[0],:]],axis=0)
        
    else:
        yvals       = np.arange(1,13,1)
        mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
    
        
    
    # # Plot it
    if usecontourf:
        cf = ax.contourf(yvals,lags,plotac.T,levels=clvls,cmap=cmap)
    else:
        cf = ax.pcolormesh(yvals,lags,plotac.T,vmin=clvls[0],vmax=clvls[-1],
                            shading='auto',cmap=cmap)
        
    # Plot the mask
    # Hatching mask
    # ax.contourf(lags,yvals,sigmask,colors='gray',
    #             hatches=["-",],levels=[-1,0],alpha=0,edgecolor='gray')
    
    # Masking Function with Stippling
    msk = viz.plot_mask(yvals,lags,sigmask,
                        ax=ax,markersize=2.0,color="k",geoaxes=False)
    
    # Contour Line Plot
    #cl = ax.contour(lags,yvals,sigmask2,colors="k",levels=clvls)
    #ax.clabel(cl)
    
    #ax.pcolormesh(lags,mons3,sigmask,shading='auto')
    
    ax.grid(True,ls='dotted')
    ax.set_yticks(np.arange(0,39,3))
    ax.set_xticks(yvals)
    ax.set_xticklabels(mons3,rotation=45)
    
    if th == 1:
        ax.set_xlabel("Reference Month")
    if th == 0:
        ax.set_ylabel("Lag (Months)")
        
    ax.set_title(threslabs[th] + "\n" + sigstr)
    
    #ax.invert_xaxis()
    
cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
cb.set_label("Correlation")
plt.suptitle("%s Autocorrelation @ %s (%s), Ensemble Avg."% (varname,locstr,mconfig),fontsize=14,y=1.05)
plt.savefig("%s%sAutocorrelation_Contours_WarmCold_%s_%s_flip.png"% (figpath,varname,mconfig,locfn),dpi=150,bbox_inches='tight')






#%% load data for MLD


# Use the function used for sm_rewrite.py
frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
lagstr     = "lag1"
method     = 5 # Damping Correction Method
inputs     = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
long,latg,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt       = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt       = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]

# Compute Specific Region
reg_sel  = [lon[0],lon[-1],lat[0],lat[-1]]
inputs   = [h,kprevall,dampingslab,dampingfull,alpha,alpha_full,hblt]
outputs,lonr2,latr2 = scm.cut_regions(inputs,long,latg,reg_sel,0)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs

#%% Select which basemonth to plot

# Select which mixed layer depth setting to evaluate
ksel = 'max' #2 #"max"

if ksel == 'max':
    # Get indices of kprev
    hmax = np.argmax(h,axis=2)
    
    # Select autocorrelation function of maximum month
    newshape = [d for d in acs_final.shape if d != 12]
    acmax = np.zeros(newshape)*np.nan
    for o in tqdm(range(nlon)):
        for a in range(nlat):
            kpt            = hmax[o,a,]
            acmax[o,a,...] = acs_final[o,a,...,kpt,:,:]
else:
    acmax = acs_final[...,ksel,:,:]

#% Integrate
integr_ac = np.trapz(acmax,x=lags,axis=-1) # [lon x lat x (ens) x thres]
#%%
#%% Visualize (Works for warm/cold)
bboxplot = [-80,0,0,60]

if varname in ["SST","TS"]:
    intitles = ("Cold","Warm","All")
elif varname == "SSS":
    intitles = ("Fresh","Salty","All")
    
plotdiff = False # Set to True to plot differences/ Otherwise plots all 3
fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True,figsize=(14,6))
for th in range(3):
    ax = axs[th]
    
    if th < nthres+1 or not plotdiff:
        if lens:
            plotac = integr_ac[:,:,:,th].mean(-1).T
        else:
            plotac = integr_ac[:,:,th].T
        title  = "%s Anomalies (%s K)" % (intitles[th],thresname[th])
        cmap   = "cmo.amp"
        vlm    = [0,10]
    else:
        if lens:
            plotac = (integr_ac[:,:,:,1].mean(-1) - integr_ac[:,:,:,0].mean(-1)).T
        else:
            plotac = (integr_ac[:,:,1] - integr_ac[:,:,0]).T
        title  = "Warm - Cold Anomalies"
        cmap   = "cmo.balance"
        vlm    = [-3,3]
        
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    ax.set_title(title)
    pcm = ax.pcolormesh(lon,lat,plotac,cmap=cmap,
                        vmin=vlm[0],vmax=vlm[1])
    
    if th == 2 and plotdiff:
        cl = ax.contour(lon,lat,plotac,levels=[0,],colors="k",linewidths=.75)
        
    else:
        cl = ax.contour(lon,lat,plotac,levels=np.arange(0,14,2),colors="w",linewidths=.75)
    ax.clabel(cl,)
    
    if plotdiff:
        if th == 0:
            cb = fig.colorbar(pcm,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.04)
            cb.set_label("Integrated Timescale (Months)")
        elif th == 2:
            cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04,pad=0.01)
            cb.set_label("Difference (Months)")
if not plotdiff:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.04)
    cb.set_label("Integrated Timescale (Months)")

plt.suptitle("Integrated %s ACF (to 36 months) for CESM1 (%s)" % (varname,mconfig),fontsize=14,y=.88)
plt.savefig("%sIntegrated%sACF_Cold_Warm_Anomalies_CESM_%s_ksel%s_plotdiff%i.png"%(figpath,varname,mconfig,str(ksel),plotdiff)
            ,dpi=150,bbox_inches='tight')

#%% Visualize the standard deviation for large ensemble

plotdiff = False # Set to True to plot differences/ Otherwise plots all 3
fig,axs  = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True,figsize=(14,6))
vstep    = 1

for th in range(3):
    ax = axs[th]
    
    if th < nthres+1 or not plotdiff:
        plotac = integr_ac[:,:,:,th].std(-1).T
        title  = "%s Anomalies (%s K)" % (intitles[th],thresname[th])
        cmap   = "cmo.amp"
        vlm    = [0,5]
    else:

        plotac = (integr_ac[:,:,:,1].std(-1) - integr_ac[:,:,:,0].std(-1)).T
        title  = "Warm - Cold Anomalies"
        cmap   = "cmo.balance"
        vlm    = [-3,3]
        
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    ax.set_title(title)
    pcm = ax.pcolormesh(lon,lat,plotac,cmap=cmap,
                        vmin=vlm[0],vmax=vlm[1])
    
    if th == 2 and plotdiff:
        cl = ax.contour(lon,lat,plotac,levels=[0,],colors="k",linewidths=.75)
        
    else:
        cl = ax.contour(lon,lat,plotac,levels=np.arange(0,vlm[-1]+vstep,vstep),colors="w",linewidths=.75)
    ax.clabel(cl,)
    
    if plotdiff:
        if th == 0:
            cb = fig.colorbar(pcm,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.04)
            cb.set_label("1$\sigma$ of Integrated Timescale (Months)")
        elif th == 2:
            cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04,pad=0.01)
            cb.set_label("Difference (Months)")
if not plotdiff:
    cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.04)
    cb.set_label("1$\sigma$ Integrated Timescale (Months)")

plt.suptitle("1$\sigma$ (Ensemble) Integrated %s ACF (to 36 months)" % (varname),fontsize=14,y=.88)
plt.savefig("%sIntegrated%sACF_Cold_Warm_Anomalies_CESM_%s_ksel%s_plotdiff%i_stdev.png"%(figpath,varname,mconfig,str(ksel),plotdiff)
            ,dpi=150,bbox_inches='tight')


#%% Check how this looks like, separately for each ensemble member
th      = 1
fig,axs = plt.subplots(7,6,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True,figsize=(20,16))
for e in tqdm(range(42)):
    ax = axs.flatten()[e]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray",blabels=[0,0,0,0])
    

    plotac = integr_ac[:,:,e,th].T

    title  = "%s Anomalies (%s K)" % (intitles[th],thresname[th])
    cmap   = "cmo.amp"
    vlm    = [0,10]
    
    pcm = ax.pcolormesh(lon,lat,plotac,cmap=cmap,
                        vmin=vlm[0],vmax=vlm[1])
    cl = ax.contour(lon,lat,plotac,levels=np.arange(0,14,2),colors="w",linewidths=.20)
    ax.clabel(cl,fontsize=12)
    ax = viz.label_sp(e+1,case='lower',usenumber=True,ax=ax,labelstyle="%s",alpha=0.65,fontsize=14)
    
plt.savefig("%sIntegrated%sACF_Cold_Warm_Anomalies_CESM_%s_ksel%s_allens_%s.png"%(figpath,varname,mconfig,str(ksel),th),dpi=150)

#%% Scrap from the old script....

# """
# Just realized it might not be possible to neatly vectorize this.

# This is because at each point, there will be a different # of points within
# each threshold, so it does not fit neatly into a 2d maxtrix...

# """

# # Split into negative or positive anomalies


# sst_classes = proc.make_classes_nd(sst_valid,thresholds,dim=1,debug=True)

# # Now compute the autocorrelation for each lag, and for each case (positive/negative)
# sst_acs = np.zeros(npts_valid,12,nthres+1,nlags) # [point,basemonth,threshold,lag]
# for th in range(nthres+1): #Loop for each threshold

#     sst_mask = (sst_classes == th)
#     sst_mask = np.where(sst_classes == th)
    
#     for im in range(12):
        
#         insst = np.take_along_axis(sst_valid,sst_mask,1)
        
#         #st_valid[np.where(sst_mask)]
        
#         # [mon x time]
#         sst_acs[:,im,th,:] = proc.calc_lagcovar_nd(insst,insst,lags,im+1,0)
        
#         #autocorrm[m,:,:] = proc.calc_lagcovar_nd(oksst,oksst,lags,m+1,0)
# #%%

# thresholds = [-1,0,1]
# y = sst_valid
