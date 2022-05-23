#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare T2 for CESM and Stochastic Model

First section is copied from viz_pointiwse_autocorrelation


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
lags        = np.arange(0,61)
lagname     = "lag%02ito%02i" % (lags[0],lags[-1]) 
thresholds  = [0,] # Standard Deviations
conf        = 0.95
tails       = 2
mconfig     = "SM"
runid       = 3
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SST" #"SST"



# Set Output Directory
# --------------------
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220526/'
proc.makedir(figpath)
outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
savename   = "%s%s_%s_autocorrelation_%s.npz" %  (outpath,mconfig,varname,thresname)
if "SM" in mconfig:
    savename = proc.addstrtoext(savename,"_runid2%02i" % (runid))
print("Loading the following dataset: %s" % savename)


# Plotting Params
# ---------------
colors   = ['b','r','k']
bboxplot = [-80,0,0,60]
mons3    = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
modelnames  = ("Constant h","Vary h","Entraining") # From sm_stylesheet
mcolors     = ["red","magenta","orange"] 

#%% Set Paths for Input (need to update to generalize for variable name)

# Postprocess Continuous SM  Run
# ------------------------------
datpath     = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"
fnames      = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
mnames      = ["constant h","vary h","entraining"] 

    
    
#%% Load in the stochastic model data
st          = time.time()
ld          = np.load(savename,allow_pickle=True)
count_final = ld['class_count']
acs_final   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]
lon         = ld['lon']
lat         = ld['lat']
thresholds  = ld['thresholds']
threslabs   = ld['threslabs']

nthres      = len(thresholds)
if "HTR" in mconfig or "SM" in mconfig:
    lens=True
    nlon,nlat,nens,nmon,_,nlags = acs_final.shape
else:
    lens=False
    nlon,nlat,nmon,_,nlags = acs_final.shape
print("Data loaded in %.2fs"% (time.time()-st))


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

#%% Select which basemonth to plot and calculate T2 (SM)
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

#%% Load  and compare with CESM-PiC

mconfigs = ("SLAB","FULL")

cesm_acs = []
for mc in range(2):
    
    mc_in  = mconfigs[mc]
    ncname = "%sPIC-%s_SST_autocorrelation_thres0_lag60.npz" % (outpath,mc_in)
    ld     = np.load(ncname)
    
    c_ac   = ld['acs'] # [lon x lat x mon x thres x lag]
    cesm_acs.append(c_ac)
    
cesm_acs = np.array(cesm_acs) # [model x ...]
cesm_acs = cesm_acs.transpose(1,2,0,3,4,5) # [lon x lat x (model) x month x thres x lag]

if ksel == 'max':
    # Get indices of kprev
    hmax = np.argmax(h,axis=2)
    
    # Select autocorrelation function of maximum month
    newshape = [d for d in cesm_acs.shape if d != 12]
    acmax_cesm = np.zeros(newshape)*np.nan
    for o in tqdm(range(nlon)):
        for a in range(nlat):
            kpt            = hmax[o,a,]
            acmax_cesm[o,a,...] = cesm_acs[o,a,...,kpt,:,:]
else:
    acmax_cesm = cesm_acs[...,ksel,:,:]

#% Integrate
integr_ac_cesm = np.trapz(acmax_cesm,x=lags,axis=-1) # [lon x lat x (model) x thres]

#%% Rename variables for ease

t2all    = np.concatenate([integr_ac_cesm,integr_ac],axis=2) # [lon x lat x model x thres]
t2names  = np.concatenate([mconfigs,mnames])
t2cols   = np.concatenate([['gray','k'],mcolors]) 
acmaxall = np.concatenate([acmax_cesm,acmax],axis=2) # [lon x lat x model x thres x lag]

acall    = np.concatenate([cesm_acs,acs_final],axis=2)


#%% Visualize avg correlation at last N lags
ithres  = 2
maxcorr = np.nanmax(acmaxall[...,[-1]],axis=-1)[...,ithres] # [lon x lat x model]


axsorder = [3,4,0,1,2]
vmin = -.2
vmax = .2
fig,axs = viz.init_2rowodd(3,proj=ccrs.PlateCarree(),figsize=(12,8))

for mc in range(5):
    
    ax = axs[axsorder[mc]]
    pcm = ax.pcolormesh(lon,lat,maxcorr[...,mc].T,vmin=vmin,vmax=vmax,cmap='cmo.balance')
    fig.colorbar(pcm,ax=ax,fraction=0.025)
    ax.set_title(t2names[mc])


#%% Visualize Lag correlation at a selected point

# Select a Point
ithres  = 2
lonf    = -30
latf    = 50
kmonth  = 2

# Get Indices
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locfn,locstr = proc.make_locstring(lonf,latf)

# Make the plot
title    = "%s Autocorrelation @ Lon: %i, Lat : %i (%s)" % (varname,lonf,latf,mconfig)
fig,ax   = plt.subplots(1,1)
ax,ax2   = viz.init_acplot(kmonth,np.arange(0,66,6),lags,ax=ax,title=title)

for mc in range(len(t2names)):
    
    ax.plot(lags,acall[klon,klat,mc,kmonth,ithres,:],marker="o",markersize=2,
            color=t2cols[mc],lw=2,
            label="%s" % (t2names[mc]))
ax.legend()
#ax.set_xlim([0,60])

#%% Plot autocorrelation at a point

imodel = 2

# Get Indices
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locfn,locstr = proc.make_locstring(lonf,latf)

# Make the plot
title    = "%s Autocorrelation @ Lon: %i, Lat : %i (%s)" % (varname,lonf,latf,mconfig)
fig,ax   = plt.subplots(1,1)
ax,ax2   = viz.init_acplot(kmonth,np.arange(0,38,2),lags,ax=ax,title=title)

if "HTR" in mconfig: 
    for th in range(nthres+2): # Just plot the one timeseries
        plotac    = acs_final[klon,klat,:,kmonth,th,:]
        plotcount = count_final[klon,klat,:,kmonth,th].sum()
        ax.plot(lags,plotac.mean(0),marker="o",color=colors[th],lw=2,
                label="%s (n=%i)" % (threslabs[th],plotcount))
        
        ax.fill_between(lags,plotac.min(0),plotac.max(0),
                        color=colors[th],alpha=0.25,zorder=-1,label="")
elif "SM" in mconfig:
    
    for th in range(nthres+2):
        ax.plot(lags,acs_final[klon,klat,imodel,kmonth,th,:],marker="o",
                color=colors[th],lw=2,
                label="%s (n=%i)" % (threslabs[th],count_final[klon,klat,imodel,kmonth,th]))
else: 
    for th in range(nthres+2):
        ax.plot(lags,acs_final[klon,klat,kmonth,th,:],marker="o",
                color=colors[th],lw=2,
                label="%s (n=%i)" % (threslabs[th],count_final[klon,klat,kmonth,th]))
        #ax.plot(lags,acs_final[klon,klat,kmonth,0,:,kmonth],label="Cold Anomalies (%i)" % (count_final[klon,klat,kmonth,0]),color='b')
        #ax.plot(lags,acs_final[klon,klat,kmonth,1,:,kmonth],label="Warm Anomalies (%i)" % (count_final[klon,klat,kmonth,th]),color='r')
        #ax.legend()
    

ax.legend()
plt.savefig("%sAutocorrelation_WarmCold_%s_%s_month%i.png"% (figpath,mconfig,locstr,kmonth+1),dpi=150)



#%%

ithres = 2

clvl   = np.arange(-10,12,2)

# FULL - SLAB
diffocean_slab = t2all[:,:,1,:] - t2all[:,:,0,:]

# FULL - const h
diffocean_h0   = t2all[:,:,1,:] - t2all[:,:,2,:]

# FULL - vary h
diffocean_ml   = t2all[:,:,1,:] - t2all[:,:,3,:]

# FULL - entrain
diffentrain    = t2all[:,:,1,:] - t2all[:,:,4,:]

diffs     = [diffocean_slab,diffocean_h0,diffocean_ml,diffentrain]
diffsname = ("FULL - SLAB","FULL - h const","FULL - h vary","FULL - entrain") 

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(12,8),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for d in range(4):
    ax = axs.flatten()[d]
    ax.set_title(diffsname[d])
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray')
    
    pcm = ax.pcolormesh(lon,lat,diffs[d][...,ithres].T,vmin=-10,vmax=10,cmap='cmo.balance')
    
    cl = ax.contour(lon,lat,diffs[d][...,ithres].T,levels=clvl,colors="k",linewidths=0.5)
    ax.clabel(cl,clvl[::2])
    
    
cb=fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045)
cb.set_label("$T_2$ Difference")

plt.savefig("%sT2_Difference_SM_vs_CESMFULL_kmax_thres%i.png"%(figpath,ithres),dpi=150,bbox_inches='tight')
#%% Do some regional analysis




#%%  Plot autocorrelation for a given region
# Use acmax (for selected month) # [lon x lat x model x thres x lag]

ithres = 2

xlim = [0,36]

bbsel = [-35,-15,38,48] #[-60,-40,35,45]#

locfn,loctitle = proc.make_locstring_bbox(bbsel)
acr,lonr,latr  = proc.sel_region(acmaxall,lon,lat,bbsel,autoreshape=True)

nlonr,nlatr,_,_,nlag = acr.shape
npts                 = nlonr*nlatr

# fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(16,4),
#                        sharex=True,sharey=True)

fig,axs = viz.init_2rowodd(3,figsize=(16,6))

for i in range(5):
    
    ax = axs[i]
    
    acplot = acr[:,:,i,ithres,:].reshape(npts,nlag)
    
    for p in range(npts):
        ax.plot(lags,acplot[p,:],color=t2cols[i],alpha=0.05)
        
    ax.plot(lags,acplot.mean(0),color=t2cols[i])
    
    ax.set_xlim(xlim)
    ax.grid(True,ls='dotted')
    ax.set_title(t2names[i],color=t2cols[i])
    if i >2:
        ax.set_xlabel("Lag (Months)")
    if i == 0 or i == 3:
        ax.set_ylabel("Lagged SST Autocorrelation")


zoomrng = 5
bbin    = bbsel
bbplot = bbplot = [-80,0,0,65]#[bbin[0]-zoomrng,bbin[1]+zoomrng,bbin[2]-zoomrng,bbin[3]+zoomrng]
left, bottom, width, height = [0.75, 0.12, 0.18, 0.25]
axin = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
axin = viz.add_coast_grid(axin,bbox=bbplot,fill_color='gray',
                          fix_lon=[bbin[0],bbin[1]],
                          fix_lat=[bbin[2],bbin[3]])
axin.set_facecolor('lightblue')
axin = viz.plot_box(bbin,ax=axin,color="red",proj=ccrs.PlateCarree(),
                    linewidth=3,linestyle='solid',)


plt.suptitle("Stochastic Model Autocorrelation for %s" % (loctitle),fontsize=14,y=.95)
savename = "%sIntegrated_ACF_SM_comparison_thres%i_%s.png" % (figpath,ithres,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% # Visulize lag to lag differences (mse)



lagrange = np.arange(0,61,1)

lagrngstr   = "lags%02ito%02i" % (lagrange[0],lagrange[-1])

clvl = np.arange(0,0.11,0.01)

# FULL - SLAB
diffocean_slab = acmaxall[:,:,1,ithres,lagrange] - acmaxall[:,:,0,ithres,lagrange]

# FULL - const h
diffocean_h0   = acmaxall[:,:,1,ithres,lagrange] - acmaxall[:,:,2,ithres,lagrange]

# FULL - vary h
diffocean_ml   = acmaxall[:,:,1,ithres,lagrange] - acmaxall[:,:,3,ithres,lagrange]

# FULL - entrain
diffentrain    = acmaxall[:,:,1,ithres,lagrange] - acmaxall[:,:,4,ithres,lagrange]

diffs     = [diffocean_slab,diffocean_h0,diffocean_ml,diffentrain]



fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(12,8),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for d in range(4):
    
    ax = axs.flatten()[d]
    ax.set_title(diffsname[d])
    
    plotmse = ((diffs[d])**2).mean(-1)
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray')
    
    pcm = ax.pcolormesh(lon,lat,plotmse.T,vmin=clvl[0],vmax=clvl[-1],cmap='cmo.ice')
    
    cl = ax.contour(lon,lat,plotmse.T,levels=clvl,colors="w",linewidths=0.5)
    ax.clabel(cl,clvl[::2])
    
    
cb=fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045)
cb.set_label("$ACF$ MSE (Lags %i to %i)" % (lagrange[0],lagrange[-1]))

plt.savefig("%sACF_DifferenceMSE_SM_vs_CESMFULL_kmax_thres%i_%s.png"%(figpath,ithres,lagrngstr),dpi=150,bbox_inches='tight')

#%% Find top N points


d       = -1
plotmse = ((diffs[d])**2).mean(-1)

proc


# -----------------------------------------------------------------------------
#%% SCRAP BELOW

#%% Examine a specific region
ithres = 2


cmap='cmo.balance'
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                      constrained_layout=True,figsize=(12,12))


plotac = (integr_ac[:,:,2,ithres] - integr_ac[:,:,0,ithres]).T
# Set up plot
ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
ax.set_title(title)
pcm = ax.pcolormesh(lon,lat,plotac,cmap=cmap,
                    vmin=vlm[0],vmax=vlm[1])

# Plot the contours
if modeln == 2:
    cl = ax.contour(lon,lat,plotac,levels=diffstep,colors="k",linewidths=.88)
    
else:
    cl = ax.contour(lon,lat,plotac,levels=np.arange(0,vlm[-1]+vstep,vstep),colors="w",linewidths=.75)
ax.clabel(cl,)
cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.04,pad=0.01)
cb.set_label("Difference (Months)")

ax.grid(True,ls='dotted')

viz.plot_box([-60,-40,35,45],ax=ax,linewidth=2)
viz.plot_box([-35,-15,38,48],ax=ax,linewidth=2,color='green')
viz.plot_box([-38,-20,56,62],ax=ax,linewidth=2,color='purple')

#
#Some figures specific to analysis of stochastic model output





#%% Rename variables for processing below

t2_sm   = integr_ac
t2_cesm = integr_



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
