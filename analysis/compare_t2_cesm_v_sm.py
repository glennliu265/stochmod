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
allruns     = True # Set to True to load all runs, then avg.
thresholds = [0,]
thresname  = "thres" + "to".join(["%i" % i for i in thresholds])
varname    = "SSS" #"SST"

# Set Output Directory
# --------------------
figpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220817/'
proc.makedir(figpath)
outpath     = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/03_reemergence/01_Data/proc/'
savename   = "%s%s_%s_autocorrelation_%s_%s.npz" %  (outpath,mconfig,varname,thresname,lagname)
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

if "SM" in mconfig and allruns:
    
    print("Loading all runs!")
    count_final = []
    acs_final   = []
    for i in range(10):
        
        # Get Name
        savename   = "%s%s_%s_autocorrelation_%s_%s.npz" %  (outpath,mconfig,varname,thresname,lagname)
        savename = proc.addstrtoext(savename,"_runid2%02i" % (i))
            
        # Load Data
        ld          = np.load(savename,allow_pickle=True)
        count = ld['class_count']
        acs   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]
        
        acs_final.append(acs)
        count_final.append(count)
    
    acs_final = np.array(acs_final).mean(0)
    count_final = np.array(count_final).mean(0)
        
else:
    
    ld          = np.load(savename,allow_pickle=True)
    count_final = ld['class_count']
    acs_final   = ld['acs'] # [lon x lat x (ens) x month x thres x lag]

# Load Labels
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
ksel = 1 #2 #"max"

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
integr_ac = 1 + 2*np.trapz(acmax**2,x=lags,axis=-1)

#np.trapz(acmax,x=lags,axis=-1) # [lon x lat x (ens) x thres]

#%% Also calculate the re-emergence index

rkmonth                  = 1
maxmincorr,maxids,minids = proc.calc_remidx_simple(acs_final,rkmonth,monthdim=3,lagdim=-1,debug=True)
remidx                   = maxmincorr[1,...] - maxmincorr[0,...] # [year x lat x lon x (ens) x thres]

#%% Debugging plot for testing the re-emergence detection

imodel = 2
ithres = -1

lonf = -50
latf = 30
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locfn,loctitle = proc.make_locstring(lon[klon],lat[klat])

xtks2 = np.arange(0,63,3)
monlabs = viz.prep_monlag_labels(rkmonth,xtks2,label_interval=2,useblank=True)

plotac     = acs_final[klon,klat,imodel,rkmonth,ithres,:]
plotremidx = remidx[:,klon,klat,imodel,ithres]
plotmaxmin = maxmincorr[:,:,klon,klat,imodel,ithres]

fig,ax = plt.subplots(1,1,figsize=(10,4),constrained_layout=True)
ax.plot(lags,plotac,color="k",marker=".",markersize=5,zorder=-1)

for y in range(remidx.shape[0]):
    
    # Plot the search points
    ax.scatter(maxids[y],plotac[maxids[y]],marker="x",color="r")
    ax.scatter(minids[y],plotac[minids[y]],marker="+",color="b")
    
    
    ilagmax = np.where(plotac == plotmaxmin[1,y])[0][0]
    ilagmin = np.where(plotac == plotmaxmin[0,y])[0][0]
    
    ax.scatter(ilagmax,plotmaxmin[1,y],marker="d",color="cyan")
    ax.scatter(ilagmin,plotmaxmin[0,y],marker="o",color="cyan")
    ax.axvline(ilagmax,color="r",ls="solid",lw=0.5)
    ax.axvline(ilagmin,color="b",ls='dashed',lw=0.5)

ax.set_xticks(xtks2)
ax.set_xticklabels(monlabs)
ax.grid(True,ls='dotted')
ax.set_xlim([lags[0],lags[-1]])
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (Months) from %s" % mons3[rkmonth])
ax.set_title("Lagged Autocorrelation for %s, Threshold %s @ %s" % 
             (modelnames[imodel],threslabs[ithres],loctitle))

savename = "%sStochasticModel_Idx_%s_mon%02i_thres%i_%s.png" % (figpath,locfn,rkmonth+1,ithres,modelnames[imodel])
plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Quick visualization of computed re-emergence patterns

ithres = 2
bboxplot = [-80,0,0,65]

clvls = np.arange(-.5,.55,.05)
fig,axs = plt.subplots(3,5,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(16,9))
for imodel in range(3):
    
    for y in range(5):
        
        ax = axs[imodel,y]
        
        blabel = [0,0,0,0]
        if imodel == 2:
            blabel[-1] = 1
        if y == 0:
            blabel[0] = 1
            
            ax.text(-0.25, 0.50, modelnames[imodel], va='bottom', ha='center',rotation='horizontal',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        if imodel == 0:
            ax.set_title("Year %i" % (y+1))
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                fill_color="gray",ignore_error=True)
        
        plotac = remidx[y,:,:,imodel,ithres]
        
        cf = ax.pcolormesh(lon,lat,plotac.T,cmap='cmo.balance',vmin=-.5,vmax=.5)
        cl = ax.contour(lon,lat,plotac.T,levels=clvls,colors="k",linewidths=0.5)
        ax.clabel(cl,fontsize=10)
cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
cb.set_label("Re-emergence Index (Max - Min Correlation)")
plt.suptitle("%s Re-emergence Index, Stochastic Model, Threshold: %s" % (mons3[rkmonth],threslabs[ithres]))
savename = "%sStochasticModel_REMIdx_mon%02i_thres%i.png" % (figpath,rkmonth+1,ithres)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -------------------------
#%% Quickly visualize h-max
# -------------------------

hmax = np.argmax(h,axis=2)

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
#cf     = ax.pcolormesh(lon,lat,hmax.T+1,cmap='twilight_shifted')
cf = ax.contourf(lon,lat,hmax.T+1,levels=np.arange(0,13,1),cmap='twilight_shifted')
cl = ax.contour(lon,lat,hmax.T+1,levels=np.arange(0,13,1),colors="w",linewidths=0.5)
ax.clabel(cl)
cb = fig.colorbar(cf,ax=ax)
cb.set_label("Month")
ax.set_title("Month of Max MLD in CESM1-PiC")
plt.savefig("%sMax_MLD_Mon.png"%figpath,dpi=150)

#%% Trying to predict where REM will be largest

dhdt  = (np.roll(h,-1,axis=2) - h).max(2)
hdiff = h.max(2) - h.min(2) 

bboxplot = [-80,0,20,65]


fig,axs = plt.subplots(1,2,figsize=(16,6),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)

for a in range(2):
    ax     = axs.flatten()[a]
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    
    if a == 0:
        plotvar = hdiff
        cblbl   = "MLD (m)"
        cmap    = 'cmo.dense'
        clvls   = np.arange(0,850,50)
        cl_lvls = np.arange(500,1400,200)
        lcol    = "w"
        title   = "Max Climatological MLD"
    else:
        plotvar = dhdt
        cblbl   = r"$w_e=\frac{dh}{dt}$ (m/mon)"
        cmap    = 'cmo.deep'
        clvls   = np.arange(0,325,25)
        cl_lvls = np.arange(300,800,100)
        lcol    = "w"
        title   = "Max Entrainment Velocity"
    
    pcm = ax.pcolormesh(lon,lat,plotvar.T,cmap=cmap,vmin=clvls[0],vmax=clvls[-1])
    cl  = ax.contour(lon,lat,plotvar.T,colors=lcol,levels=cl_lvls,linewidths=0.5)
    ax.clabel(cl,fontsize=12)
    cb = fig.colorbar(pcm,ax=ax,fraction=0.044,orientation='horizontal')
    cb.set_label(cblbl)
    ax.set_title(title)
plt.savefig("%sMLD_we_maxclim.png"%figpath,dpi=150,bbox_inches='tight')

#%% Load  and compare with CESM-PiC

mconfigs = ("SLAB","FULL")

cesm_acs = []
for mc in range(2):
    
    mc_in  = mconfigs[mc]
    ncname = "%sPIC-%s_SST_autocorrelation_thres0_lag00to60.npz" % (outpath,mc_in)
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
integr_ac_cesm = 1 + 2 * np.trapz(acmax_cesm**2,x=lags,axis=-1) # [lon x lat x (model) x thres]


#%% Compute REM-Index for CESM

cesm_maxmincorr,maxids,minids = proc.calc_remidx_simple(cesm_acs,rkmonth,monthdim=3,lagdim=-1,debug=True)
cesm_remidx                   = cesm_maxmincorr[1,...] - cesm_maxmincorr[0,...] # [year lat lon model thres]


#%% Rename variables for ease

t2all     = np.concatenate([integr_ac_cesm,integr_ac],axis=2) # [lon x lat x model x thres]
t2names   = np.concatenate([mconfigs,mnames])
t2cols    = np.concatenate([['gray','k'],mcolors]) 
acmaxall  = np.concatenate([acmax_cesm,acmax],axis=2) # [lon x lat x model x thres x lag]

acall     = np.concatenate([cesm_acs,acs_final],axis=2)

remidxall = np.concatenate([cesm_remidx,remidx],axis=3) # [yr x lat x lon x model x thres]

# ----------------------------------
# %% Quick Check how ksel impacts T2
# ----------------------------------
bboxplot = [-80,0,5,67]
ithres   = -1
t2seas   = 1 + 2*np.trapz(acall[:,:,:,:,ithres,:]**2,x=lags,axis=-1) # [lon x lat x (model) x month]

imodel   = 0

clm      = [0,24]
clvl     = [6,12,18]
cextras  = [24,30,36,]
plotmons = np.roll(np.arange(0,12),1)

# Make plot for each model
for imodel in range(5):
    fig,axs  = plt.subplots(4,3,figsize=(10,10),
                           subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
    for ia in tqdm(range(12)):
        ax = axs.flatten()[ia]
        im = plotmons[ia]
        blabel=[0,0,0,0]
        if ia%3 == 0:
            blabel[0] = 1
        if ia > 8:
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray",blabels=blabel)
        ax = viz.label_sp(mons3[im],usenumber=True,ax=ax,labelstyle="%s",fontsize=16,alpha=0.75)
        
        pcm = ax.pcolormesh(lon,lat,t2seas[:,:,imodel,im].T,vmin=clm[0],vmax=clm[-1],cmap='cmo.deep')
        
        cl = ax.contour(lon,lat,t2seas[:,:,imodel,im].T,levels=cextras,colors='w',lw=.55)
        ax.clabel(cl)
        
        clint = ax.contour(lon,lat,t2seas[:,:,imodel,im].T,levels=clvl,colors='k',lw=.55)
        ax.clabel(clint)
        
        
        
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.035,pad=0.01)
    cb.set_label("$T_2$ (Months)")
    plt.suptitle("$T_2$ by Reference Month (%i Lags) for %s"%(lags[-1],t2names[imodel]))
        
    savename = "%sT2bymon_%ilags_%s.png" % (figpath,lags[-1],t2names[imodel])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Visualize avg correlation at last N lags
ithres  = 2
maxcorr = np.nanmax(acmaxall[...,[-1]],axis=-1)[...,ithres] # [lon x lat x model]

axsorder = [3,4,0,1,2]
vmin = -.2
vmax = .2
fig,axs = viz.init_2rowodd(3,proj=ccrs.PlateCarree(),figsize=(14,8))

for mc in range(5):
    
    ax = axs[axsorder[mc]]
    pcm = ax.pcolormesh(lon,lat,maxcorr[...,mc].T,vmin=vmin,vmax=vmax,cmap='cmo.balance')
    fig.colorbar(pcm,ax=ax,fraction=0.025)
    ax.set_title(t2names[mc])


#%% Visualize Lag correlation at a selected point

# Select a Point
ithres  = 2
lonf    = -30#-15
latf    = 50 #61
kmonth  = 2

# Get Indices
klon,klat    = proc.find_latlon(lonf,latf,lon,lat)
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

#%% Compare WARM/COLD for each Model to examine asymmetry/symmetry

kmonth = 1

threscols = ("b","r","k")
plotorder = [2,3,4,0,1]


fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(14,6))

# Make the plot
for mc in range(5):
    imc = plotorder[mc]
    
    title    = "%s" % (t2names[imc])
    if mc == 4:
        ax = axs.flatten()[-1]
    else:
        ax = axs.flatten()[mc]
    ax,ax2   = viz.init_acplot(kmonth,np.arange(0,66,6),lags,ax=ax,title=title)
    
    for ithres in range(3):
        
        ax.plot(lags,acall[klon,klat,imc,kmonth,ithres,:],marker="o",markersize=2,
                color=threscols[ithres],lw=2,
                label="%s" % (threslabs[ithres]))
        ax.legend()
        
plt.suptitle("%s Autocorrelation @ Lon: %i, Lat : %i" % (varname,lonf,latf),y=1.01)

axs[1,1].axis('off')
plt.savefig("%sHotColdACDifferences_%s.png"%(figpath,locfn),dpi=150,bbox_inches="tight")

#%% Visualize the pcolor lag autocorrelation
ithres = 2

p     = 0.01
tails = 1

def add_refline_monvlag(refmon,nlags,appendjan=True,c='k',lw=.75,
                        ls='solid',ax=ax,adj=0):
    if ax is None:
        ax = plt.gca()
    if appendjan:
        refmons = np.arange(1,14,1)
    else:
        refmons = np.arange(1,13,1)
    
    lags = np.arange(0,nlags)
    
    # Row is lags, col is reference month
    ref_col,lag_row = np.meshgrid(refmons,lags)
    monmatrix       = lag_row + ref_col 
    
    # Select multiples of remon
    maxyr = np.floor(monmatrix.flatten().max()/12)
    # findvalues = np.arange(refmon,maxyr*12+12,12,)
    # maskmatrix = np.isin(monmatrix,findvalues)
    # xx,yy = np.where(maskmatrix)
    
    # Get left and right y indices
    lagval_left  = np.arange(refmon,maxyr*12+12,12,)
    lagval_right = lagval_left - 12
    
    #fig,ax = plt.subplots(1,1)
    #pcm = ax.pcolormesh(refmons,lags,monmatrix,vmin=0,vmax=12,shading='nearest')
    for i in range(len(lagval_left)):
        ax.plot([1,13+adj],[lagval_left[i],lagval_right[i]-adj],
                color=c,lw=lw,ls=ls)
    #ax.plot(np.zeros(len(lagval_left)),lagval_left)
    # for i in range(int(maxyr)):
    #     ax.plot(yy[i*12:(i+1)*12]+1,xx[i*12:(i+1)*12]+1)
    #ax.set_xlim([1,14])
    #ax.set_ylim([0,36])
    #fig.colorbar(pcm,ax=ax)
    #ax.set_yticks(np.arange(0,37,1))
    #ax.set_xticks(np.arange(1,14,1))
    return ax
    
    
    
    
    
def plot_ac_monvlag(plotac,plotcount,clvls,lags,ax=None,
                    p=0.05,tails=1,appendjan=True,usecontourf=False,
                    cmap='cmo.dense'):
    
    
    # Calculate Critical Rho Value
    rhocrit   = proc.ttest_rho(p,tails,plotcount) # [month]
    sigmask   = plotac > rhocrit[:,None]
    
    # Add extra january
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
        
    # Masking Function with Stippling
    msk = viz.plot_mask(yvals,lags,sigmask,
                        ax=ax,markersize=2.0,color="k",geoaxes=False)
    
    ax.grid(True,ls='dotted')
    ax.set_yticks(np.arange(lags[0],lags[-1],3))
    ax.set_xticks(yvals)
    ax.set_xticklabels(mons3,rotation=45)
    return ax,rhocrit

#%% Plot all models

clvls = np.arange(-.1,1.05,0.05)
fig,axs = plt.subplots(1,5,figsize=(18,8))
for a in range(5):
    
    ax =axs.flatten()[a]
    
    
    plotac = acall[klon,klat,a,:,ithres,:]
    if a == 0:
        plotcount = np.ones(12) * 901
    elif a == 1:
        plotcount = np.ones(12) * 1801
    else:
        plotcount = np.ones(12) * 1000
        
    
    ax,rhocrit = plot_ac_monvlag(plotac,plotcount,clvls,lags,ax=ax,
                        p=p,tails=tails,appendjan=True,usecontourf=False,
                        cmap='cmo.dense')
    
    ax = add_refline_monvlag(12,plotac.shape[1],ax=ax,adj=1)
    
    sigstr    = "%i" % ((p)*100) + "%" + r" (%i-tail): $\rho$ > %.2f (n=%i)" % (tails,rhocrit.mean(),plotcount.mean())
    title = "%s \n %s" % (t2names[a],sigstr)
    ax.set_title(title)
    
    ax.set_ylim([lags[0],lags[-1]])
    ax.set_xlim([0,14])
    
    #$pcm = ax.pcolormesh(lags,)
plt.savefig("%sACF_LagvMon_%s.png"% (figpath,locfn),dpi=150,bbox_inches='tight')

#%% Focus on just 2 models

lonf    = -40 #-15
latf    = 0   #61

# Get Indices
klon,klat    = proc.find_latlon(lonf,latf,lon,lat)
locfn,locstr = proc.make_locstring(lonf,latf)

modelsel   = [1,4]
ithres     = 2
linerefmon = 12


clvls = np.arange(-.1,1.05,0.05)
fig,axs = plt.subplots(1,2,figsize=(10,8))
for a in range(2):
    
    ax     = axs.flatten()[a]
    imodel = modelsel[a]
    
    plotac = acall[klon,klat,imodel,:,ithres,:]
    if a == 0:
        plotcount = np.ones(12) * 901
    elif a == 1:
        plotcount = np.ones(12) * 1801
    else:
        plotcount = np.ones(12) * 1000
    
    ax,rhocrit = plot_ac_monvlag(plotac,plotcount,clvls,lags,ax=ax,
                        p=p,tails=tails,appendjan=True,usecontourf=False,
                        cmap='cmo.dense')
    
    ax = add_refline_monvlag(linerefmon,plotac.shape[1],ax=ax,adj=0)
    
    sigstr    = "%i" % ((p)*100) + "%" + r" (%i-tail): $\rho$ > %.2f (n=%i)" % (tails,rhocrit.mean(),plotcount.mean())
    title = "%s \n %s" % (t2names[imodel],sigstr)
    ax.set_title(title)
    
    ax.set_ylim([lags[0],lags[-1]])
    ax.set_xlim([0,14])
    
    #$pcm = ax.pcolormesh(lags,)
plt.savefig("%sACF_LagvMon_%s_%sv%s.png"% (figpath,locfn,t2names[modelsel[0]],t2names[modelsel[1]]),dpi=150,bbox_inches='tight')


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


# ----------------------------------------------------------------
#%% Visualize the T2 Differences to isolate role of Ocean dynamics
# ----------------------------------------------------------------

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

# entrain - varyh
effentrain     = t2all[:,:,4,:] - t2all[:,:,3,:]

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

plt.savefig("%sT2_Difference_SM_vs_CESMFULL_k%s_thres%i.png"%(figpath,ksel,ithres),dpi=150,bbox_inches='tight')

# ----------------------------------------------------------------
#%% Visualize the MSE Differences to see fit to stochastic model
# ----------------------------------------------------------------

lagrange    = np.arange(0,61,1)

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


# entrain - varyh
#effentrain    = acmaxall[:,:,4,ithres,lagrange] - acmaxall[:,:,3,ithres,lagrange]


diffs_mse     = [diffocean_slab,diffocean_h0,diffocean_ml,diffentrain]

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(12,8),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for d in range(4):
    
    ax = axs.flatten()[d]
    ax.set_title(diffsname[d])
    
    plotmse = ((diffs_mse[d])**2).mean(-1)
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray')
    
    pcm = ax.pcolormesh(lon,lat,plotmse.T,vmin=clvl[0],vmax=clvl[-1],cmap='cmo.ice')
    
    cl = ax.contour(lon,lat,plotmse.T,levels=clvl,colors="w",linewidths=0.5)
    ax.clabel(cl,clvl[::2])
    
    
cb=fig.colorbar(pcm,ax=axs.flatten(),fraction=0.045)
cb.set_label("$ACF$ MSE (Lags %i to %i)" % (lagrange[0],lagrange[-1]))

plt.savefig("%sACF_DifferenceMSE_SM_vs_CESMFULL_k%s_thres%i_%s.png"%(figpath,ksel,ithres,lagrngstr),dpi=150,bbox_inches='tight')

# -----------------------------------
# %% AMV Teleconf Plot: T2 Comparison
# -----------------------------------

clbls_T2 = np.arange(-12,36,6) #np.arange(0,27,3)
clbls_dT = np.arange(-12,30,6)

fig,axs = plt.subplots(2,2,constrained_layout=True,figsize=(8,8),
                       subplot_kw={'projection':ccrs.PlateCarree()})

for ia in range(4):
    
    if ia == 0:
        title   = "Entrain"
        plotvar = t2all[:,:,4,ithres]
        clbl = clbls_T2
    elif ia == 1:
        title   = "Entrain - h vary"
        plotvar = effentrain[...,ithres]
        clbl = clbls_dT
    elif ia == 2:
        title   = "CESM-FULL"
        plotvar = t2all[:,:,1,ithres]
        cblabel = "$T_2$ (Months)"
        clbl = clbls_T2
    elif ia == 3:
        title = "FULL - Entrain"
        plotvar = diffentrain[...,ithres]
        cblabel = "$T_2$ Difference (Months)"
        clbl = clbls_dT
    
    ax = axs.flatten()[ia]
    blabel = [0,0,0,0]
    if ia%2==0:
        blabel[0]=1
        cmap   = 'cmo.deep'
        levels = np.arange(0,27,3)
        
    else:
        cmap = 'cmo.balance'
        levels = np.arange(-16,18,2)
    
    if ia>1:
        blabel[-1]=1
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,fill_color='gray',blabels=blabel)
    ax.set_title(title)
    pcm = ax.contourf(lon,lat,plotvar.T,levels=levels,cmap=cmap,extend='both')
    cl  = ax.contour(lon,lat,plotvar.T,levels=clbl,colors='w',linewidths=0.45)
    ax.clabel(cl)
    
    if ia>1:
        cb = fig.colorbar(pcm,ax=axs[:,ia%2].flatten(),orientation='horizontal',fraction=0.025)
        cb.set_label(cblabel)
plt.suptitle("$T_2$ Comparisons, Reference Month = %s"%(mons3[ksel]))
plt.savefig("%sT2Comparisons_AMVTeleconf_ksel%s.png"%(figpath,str(ksel)),dpi=150,bbox_inches="tight")

#%% Draw the locator

bboxes  = [[-65,-43,42,47],[-43,-25,50,56]]
bbcol   = ['k','darkviolet']

bboxis  = [-75,-5,30,66]
plotvar = diffentrain[...,ithres]
title   = "FULL - Entrain"
cblabel = "$T_2$ Difference (Months)"
clbl    = clbls_dT


fig,ax = plt.subplots(1,1,figsize=(8,4,),subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
ax     = viz.add_coast_grid(ax,bbox=bboxis,fill_color="gray")

pcm = ax.contourf(lon,lat,plotvar.T,levels=levels,cmap=cmap,extend='both')
cl  = ax.contour(lon,lat,plotvar.T,levels=clbl,colors='w',linewidths=0.45)
ax.clabel(cl)


for b,bb in enumerate(bboxes):
    viz.plot_box(bb,ax=ax,linewidth=4,color=bbcol[b])

cb = fig.colorbar(pcm,ax=ax,fraction=0.024)
cb.set_label(cblabel)
ax.set_title(title)
plt.savefig("%sAMVconf_Locator_T2diff.png"%figpath,dpi=150,bbox_inches='tight')



#%% Do some regional analysis

"""
Some scrap to decide on the bounding box/region

[-43,-32,48,60] --> [-41,-35,52,58] # SPGw
[-30,-20,49,55] --> [-30,-22,49,55] # SPGe
[-22,-12,42,60] --> [-20,-12,45,65] # NE Atlantic (horiz)

--> [-62,-42,40,48] # S. Grand Banks

--> [-70,-40,28,35] # STG Centre

--> [-75,-65,25,35] # Sargasso Sea Patch
"""

# Scrap for identifying regions



#%% Indicate Region of Analysis
ithres = 2
bbsel    = [-43,-25,50,56] #[-65,-43,42,47]#[-60,-40,35,45]# Indicate region section here
locfancy = "SPG Center" #"Transition Zone"
locfn,loctitle = proc.make_locstring_bbox(bbsel)

# Selecting plotting variable for insets
# Use acmax (for selected month) # [lon x lat x model x thres x lag]
plotdiff = diffs[-1][...,ithres].T

#%% Load in regional variables (copied block from  investigate_sm_regional.py)

method  = 5
lagstr  = 'lag1'
reg_sel = bbsel

frcname     = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
input_path = datpath + "../model_input/"

# Use the function used for sm_rewrite.py,
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
long,latg,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]
#klonf,klatf = proc.find_latlon(lonf,latf,lon,lat)

inputs       = [h,kprevall,dampingslab,dampingfull,alpha,alpha_full,hblt]
outputs,_,_  = scm.cut_regions(inputs,long,latg,reg_sel,0)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs

# Sqrt(Sum(EOF Coeffs)))
alpha_full2 = np.sqrt((alpha_full**2).sum(2))
alpha2      = np.sqrt((alpha**2).sum(2))


#%% Plot stochastic model inputs for that region

mcinput = "FULL"

if mcinput == "FULL":
    invars = [h,alpha_full2,dampingfull,]
else:
    invars = [hblt,alpha2,damping,]
    

# Combine spatial dims
npts   = np.prod(invars[0].shape[:2])
invars = [v.reshape(npts,nmon) for v in invars]

# Taken from viz_inputs_pt.py)
ylabs = ("MLD (m)",
         "Forcing ($Wm^{-2}$)",
        "Damping ($Wm^{-2}K^{-1}$)",
         )
vcolor  = ("mediumblue","orangered","limegreen")

fig,axs = plt.subplots(3,1,constrained_layout=True,sharex=True,figsize=(6,8))

for a in range(3):
    ax = axs[a]
    for n in range(npts):
        ax.plot(mons3,invars[a][n,:],label="",alpha=0.2,color=vcolor[a])
        
    # Plot mean
    ax.plot(mons3,np.nanmean(invars[a],0),label="Mean",alpha=1,color='k')
    
    # Plot 1 Std
    std1 = np.nanstd(invars[a],0)
    
    ax.plot(mons3,np.nanmean(invars[a],0)+std1,label="1$\sigma$",alpha=1,color='k',ls='dotted')
    ax.plot(mons3,np.nanmean(invars[a],0)-std1,label="",alpha=1,color='k',ls='dotted')
    
    if a == 0:
        ax.legend()
    ax.set_ylabel(ylabs[a])
    ax.grid(True,ls='dotted')
    
    
ax.set_xlim([0,11])
plt.suptitle("Parameter Values for %s (CESM-%s)"% (loctitle,mcinput))

plt.savefig("%sSM_ParameterValues_%s_%s.png"%(figpath,mcinput,locfn),dpi=150,bbox_inches='tight')

#%%  Plot autocorrelation for a given region for all models
xlim = [0,36]

acr,lonr,latr  = proc.sel_region(acmaxall,lon,lat,bbsel,autoreshape=True)

nlonr,nlatr,_,_,nlag = acr.shape
npts                 = nlonr*nlatr


# Plot Autocorrelations
# fig,axs = plt.subplots(2,3,constrained_layout=True,figsize=(16,4),
#                        sharex=True,sharey=True)

fig,axs = viz.init_2rowodd(3,figsize=(16,6))
for i in range(5):
    
    ax = axs[i]
    
    acplot = acr[:,:,i,ithres,:].reshape(npts,nlag)
    
    for p in range(npts):
        ax.plot(lags,acplot[p,:],color=t2cols[i],alpha=0.05)
        
    ax.plot(lags,np.nanmean(acplot,0),c="k",alpha=1)
    
    ax.set_xlim(xlim)
    ax.grid(True,ls='dotted')
    ax.set_title(t2names[i],color=t2cols[i])
    if i >2:
        ax.set_xlabel("Lag (Months)")
    if i == 0 or i == 3:
        ax.set_ylabel("Lagged SST Autocorrelation")


# Plot Inset
zoomrng = 5
bbin    = bbsel
bbplot = bbplot = [-80,0,0,65]#[bbin[0]-zoomrng,bbin[1]+zoomrng,bbin[2]-zoomrng,bbin[3]+zoomrng]
left, bottom, width, height = [0.75, 0.12, 0.18, 0.25]
axin = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
axin = viz.add_coast_grid(axin,bbox=bbplot,fill_color='gray',
                          fix_lon=[bbin[0],bbin[1]],
                          fix_lat=[bbin[2],bbin[3]])
#axin.set_facecolor('lightblue')
axin.contourf(lon,lat,plotdiff,levels=np.arange(-16,18,2),cmap='cmo.balance',extend='both')
axin = viz.plot_box(bbin,ax=axin,color="red",proj=ccrs.PlateCarree(),
                    linewidth=3,linestyle='solid',)


plt.suptitle("Stochastic Model Autocorrelation for %s" % (loctitle),fontsize=14,y=.95)
savename = "%sIntegrated_ACF_SM_comparison_k%s_thres%i_%s.png" % (figpath,ksel,ithres,locfn)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# ------------------------------------------------------------
#%% More condensed/summary plot with effective damping/forcing
# ------------------------------------------------------------
cp0     = 3996
rho     = 1026
dt      = 3600*24*30
imodels = [1,-1] # Models to plot ACF for 

xtks     = np.arange(0,66,6)
fig = plt.figure(figsize=(16,8))

gs = fig.add_gridspec(nrows=6, ncols=6, left=0.1, right=1,
                      hspace=.5, wspace=0.5)


# (0) Plot Effective Parameters
# --------------------------
ax0 = fig.add_subplot(gs[0:3, 0:3])

ax = ax0
veff_names = ["Effective Forcing","Effective Damping"]
for v in range(2):
    
    kvar    = v+1
    plotvar = invars[kvar]/invars[0] / (rho*cp0) * dt
    
    for n in range(npts):
        
        ax.plot(mons3,plotvar[n,:],label="",alpha=0.1,color=vcolor[kvar])
        
    # Plot mean
    ax.plot(mons3,np.nanmean(plotvar,0),label=veff_names[v],alpha=1,color=vcolor[kvar])
        
        
    ax.grid(True,ls='dotted')
    ax.set_xlim([0,11])
ax.set_ylabel("Effective Forcing/Damping ($\degree C$ $sec^{-1}$)")
ax.legend()

# (1) Plot Autocorrelations
# ---------------------
ax1 = fig.add_subplot(gs[3:, 0:3])
ax = ax1
acr_flatten = acr.reshape((npts,)+acr.shape[2:]) # [pt x model x thres xlag]

for i in range(2):
    imodel = imodels[i]
    
    for n in range(npts):
        ax.plot(lags,acr_flatten[n,imodel,ithres,:],color=t2cols[imodel],label="",alpha=0.05)
    ax.plot(lags,np.nanmean(acr_flatten[:,imodel,ithres,:],0),color=t2cols[imodel],label=t2names[imodel])
    
    
ax.set_xlim([lags[0],lags[-1]])
ax.grid(True,ls='dotted')
ax.set_xticks(xtks)
ax.set_ylabel("Autocorrelation (Lag 0 = $t_{h max}$)")
ax.set_xlabel("Lag (Months)")
ax.legend()
if isinstance(ksel,int):
    xtk_lbls = viz.prep_monlag_labels(ksel,xtks,1,useblank=True)
    ax.set_xticklabels(xtk_lbls)

# (2) Plot the Inset
# ------------------
axin = fig.add_subplot(gs[0:-1,3:5],projection=ccrs.PlateCarree())
bbin   = bbsel
bbplot = [-80,0,0,65] 
cint   = np.arange(-10,11,1)

axin = viz.add_coast_grid(axin,bbox=bbplot,fill_color='gray',
                          fix_lon=[bbin[0],bbin[1]],
                          fix_lat=[bbin[2],bbin[3]])

cf = axin.contourf(lon,lat,plotdiff,levels=cint,cmap='cmo.balance',extend='both')
cl = axin.contour(lon,lat,plotdiff,levels=cint[::2],colors="k",extend='both',linewidths=.55)
axin.clabel(cl)
axin = viz.plot_box(bbin,ax=axin,color="black",proj=ccrs.PlateCarree(),
                    linewidth=3,linestyle='solid',)
cb = fig.colorbar(cf,ax=axin,fraction=0.045,orientation='horizontal',pad=0.1)
cb.set_label("Months")
axin.set_title("$T_2$ Difference (CESM-FULL - Entrain)")

plt.suptitle("CESM-FULL vs. Entrain for %s" % loctitle,fontsize=16,y=0.92)

plt.savefig("%sFULLvEntrain_SummaryFig_%s_k%s_thres%i.png" % (figpath,locfn,ksel,ithres),dpi=150,bbox_inches="tight")

# -----------------------------------------------------------------------------
#%% Same Plot as Above (but without the locator) [For AMV Teleconf]
# -----------------------------------------------------------------------------

cp0=3996
rho=1026
dt =3600*24*30
imodels = [1,-1] # Models to plot ACF for 

include_params = False

xtks     = np.arange(0,66,6)
fig = plt.figure(figsize=(16,8))

if include_params:
    fig,axs = plt.subplots(2,1,figsize=(8,8))

    # (0) Plot Effective Parameters
    # --------------------------
    ax = axs[0]
    veff_names = ["Effective Forcing","Effective Damping"]
    vcoloreff  = ("","cornflowerblue","salmon")
    for v in range(2):
        
        kvar    = v+1
        plotvar = invars[kvar]/invars[0] / (rho*cp0) * dt
        
        for n in range(npts):
            
            ax.plot(mons3,plotvar[n,:],label="",alpha=0.1,color=vcoloreff[kvar])
            
        # Plot mean
        ax.plot(mons3,np.nanmean(plotvar,0),label=veff_names[v],alpha=1,color=vcoloreff[kvar])
        
        # Plot stdev
        #ax.plot(mons3,np.nanmean(plotvar,0)+np.nanstd(plotvar,0),ls='dashed',alpha=1,color=vcoloreff[kvar])
        #ax.plot(mons3,np.nanmean(plotvar,0)-np.nanstd(plotvar,0),ls='dashed',alpha=1,color=vcoloreff[kvar])
            
        ax.grid(True,ls='dotted')
        ax.set_xlim([0,11])
    ax.set_ylabel("Effective Forcing/Damping ($\degree C$ $sec^{-1}$)")
    ax.legend()

    # (1) Plot Autocorrelations
    # ---------------------
    ax = axs[1]
else:
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    
    
acr_flatten = acr.reshape((npts,)+acr.shape[2:]) # [pt x model x thres xlag]

for i in range(2):
    imodel = imodels[i]
    
    for n in range(npts):
        ax.plot(lags,acr_flatten[n,imodel,ithres,:],color=t2cols[imodel],label="",alpha=0.05)
    ax.plot(lags,np.nanmean(acr_flatten[:,imodel,ithres,:],0),color=t2cols[imodel],label=t2names[imodel])
    
    
ax.set_xlim([lags[0],lags[-1]])
ax.grid(True,ls='dotted')
ax.set_xticks(xtks)
ax.set_ylabel("Autocorrelation (Lag 0 = %s)"% mons3[ksel])
ax.set_xlabel("Lag (Months)")
ax.legend()
if isinstance(ksel,int):
    xtk_lbls = viz.prep_monlag_labels(ksel,xtks,1,useblank=True)
    ax.set_xticklabels(xtk_lbls)


if include_params:
    plt.suptitle("%s (%s)" % (locfancy,loctitle),fontsize=16,y=0.92)
else:
    ax.set_title("%s (%s)" % (locfancy,loctitle),fontsize=16)
plt.savefig("%sFULLvEntrain_SummaryFig_%s_k%s_thres%i_nolocator_inclparams%i.png" % (figpath,locfn,ksel,ithres,include_params),dpi=150,bbox_inches="tight")

# --------------------------------------------------
#%% Visualize Warm/Cold T2 Differences in each model
# --------------------------------------------------
mc    = 0


for mc in range(5):
    cmap  = 'cmo.balance'
    cints = np.arange(-5,5.5,0.5)
    
    
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                          constrained_layout=True,figsize=(12,8))
    
    ax     = viz.add_coast_grid(ax,bbox=bboxplot,fill_color="gray")
    ax.set_title("$T_2$ Difference (Warm - Cold, %s)" % t2names[mc],
                 fontsize=24,color=t2cols[mc])
    
    
    plotvar = t2all[:,:,mc,1] - t2all[:,:,mc,0]
    
    cf = ax.contourf(lon,lat,plotvar.T,levels=cints,cmap=cmap,extend='both')
    cl = ax.contour(lon,lat,plotvar.T,levels=cints,colors="k",linewidths=0.5)
    ax.clabel(cl,cints[::2])
    cb = fig.colorbar(cf,ax=ax)
    cb.set_label("$T_2$ Difference (Warm - Cold)",fontsize=14)
    
    plt.savefig("%sT2_Difference_%s.png"%(figpath,t2names[mc]),dpi=150,bbox_inches='tight')

#%% Do intercomparison of REM Index for CESM1 and Entraining SM

bboxplot = [-80,0,0,65]
modelsel = [0,1,4]
ithres   = 2

clvls = np.arange(-.5,.55,.05)
fig,axs = plt.subplots(3,5,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(18,10))

for im in range(len(modelsel)):
    
    imodel = modelsel[im]
    
    for y in range(5):
        
        ax = axs[im,y]
        
        blabel = [0,0,0,0]
        if im == 2:
            blabel[-1] = 1
        if (y == 0):
            blabel[0] = 1
            
            ax.text(-0.25, 0.50, t2names[imodel], va='bottom', ha='center',rotation='horizontal',
                    rotation_mode='anchor',transform=ax.transAxes)
        
        if im == 0:
            ax.set_title("Year %i" % (y+1))
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                fill_color="gray",ignore_error=True)
        
        plotac = remidxall[y,:,:,imodel,ithres]
        
        cf = ax.pcolormesh(lon,lat,plotac.T,cmap='cmo.balance',vmin=-.5,vmax=.5)
        cl = ax.contour(lon,lat,plotac.T,levels=clvls,colors="k",linewidths=0.5)
        ax.clabel(cl,fontsize=10)
cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
cb.set_label("Re-emergence Index (Max - Min Correlation)")
plt.suptitle("%s Re-emergence Index Threshold: %s" % (mons3[rkmonth],threslabs[ithres]))
savename = "%sREMIdx_Intercomparison_mon%02i_thres%i.png" % (figpath,rkmonth+1,ithres)
plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Just plot for 2 models and difference (Entrain, CESM-FULL)

bboxplot = [-80,0,0,65]
modelsel = [1,4]
ithres   = 2
imon     = 1
yr       = 0

for yr in range(5):
    clvls = np.arange(-.5,.55,.05)
    fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True,figsize=(16,6))
    
    
    
    for a in range(3):
        
        ax = axs.flatten()[a]
        
        blabel = [0,0,0,1]
        if (a == 0):
            blabel[0] = 1
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                fill_color="gray",ignore_error=True)
        
        
        if a <2:
            imodel = modelsel[a]
            plotvar = remidxall[yr,:,:,imodel,ithres]
            cmap    = 'inferno'
            clvls   = np.arange(0,0.55,0.05)
            cblbl   = "RE Index (Max - Min Correlation)"
            title   = "%s" % t2names[imodel]
        elif a == 2:
            plotvar = remidxall[yr,:,:,modelsel[0],ithres] - remidxall[yr,:,:,modelsel[1],ithres]
            cmap    = 'cmo.balance'
            clvls   = np.arange(-.4,.45,.05)
            cblbl   = "Difference In RE Index"
            title   = "%s - %s" % (t2names[modelsel[0]], t2names[modelsel[1]])
        
        cf = ax.pcolormesh(lon,lat,plotvar.T,cmap=cmap,vmin=clvls[0],vmax=clvls[-1])
        cl = ax.contour(lon,lat,plotvar.T,levels=clvls,colors="k",linewidths=0.5)
        ax.clabel(cl,fontsize=10)
        ax.set_title(title)
        
        if a == 1:
            cb = fig.colorbar(cf,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
            cb.set_label(cblbl)
        elif a == 2:
            cb = fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.035,pad=0.01)
            cb.set_label(cblbl)
    
    plt.suptitle("%s Year %i Re-emergence Index, Threshold: %s" % (mons3[rkmonth],yr,threslabs[ithres]),y=.95)
    savename = "%sREMIdx_Intercomparison_mon%02i_thres%i_%sv%s_Y%i.png" % (figpath,rkmonth+1,ithres,
                                                                       t2names[modelsel[0]],
                                                                       t2names[modelsel[1]],yr)
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Same as above, but differencing warm vs. cold anomalies for the selected model
# For Thesis Committee Meeting 8/17/2022

bboxplot = [-80,0,0,65]
modelsel = 1
#thres   = 2
imon     = 1
yr       = 0

for yr in range(5):
    clvls = np.arange(-.5,.55,.05)
    fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True,figsize=(16,6))
    
    for a in range(3):
        
        ax = axs.flatten()[a]
        
        blabel = [0,0,0,1]
        if (a == 0):
            blabel[0] = 1
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,
                                fill_color="gray",ignore_error=True)
        
        if a <2:
            plotvar = remidxall[yr,:,:,modelsel,a]
            cmap    = 'inferno'
            clvls   = np.arange(0,0.55,0.05)
            cblbl   = "RE Index (Max - Min Correlation)"
            title   = "%s" % threslabs[a]
        elif a == 2:
            plotvar = remidxall[yr,:,:,modelsel,1] - remidxall[yr,:,:,modelsel,0]
            cmap    = 'cmo.balance'
            clvls   = np.arange(-.4,.45,.05)
            cblbl   = "Difference In RE Index"
            title   = "Difference (Warm - Cold)"
        
        cf = ax.pcolormesh(lon,lat,plotvar.T,cmap=cmap,vmin=clvls[0],vmax=clvls[-1])
        cl = ax.contour(lon,lat,plotvar.T,levels=clvls,colors="k",linewidths=0.5)
        ax.clabel(cl,fontsize=10)
        ax.set_title(title)
        
        if a == 1:
            cb = fig.colorbar(cf,ax=axs[:2].flatten(),orientation='horizontal',fraction=0.035,pad=0.01)
            cb.set_label(cblbl)
        elif a == 2:
            cb = fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.035,pad=0.01)
            cb.set_label(cblbl)
    
    plt.suptitle("%s Year %i Re-emergence Index (%s)" % (mons3[rkmonth],yr+1,t2names[modelsel]),y=.95)
    savename = "%sREMIdx_Intercomparison_mon%02i_PosNeg_%s_Y%i.png" % (figpath,rkmonth+1,
                                                                       t2names[modelsel],yr)
    plt.savefig(savename,dpi=150,bbox_inches='tight')


    
#%% Try to figure out the year of the last re-emergence




# ----------------------------------------------------------------------------
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

# ------------------------------------------------------------
# Some figures specific to analysis of stochastic model output
# ------------------------------------------------------------


#%% Visualize amplitude of the seasonal re-emergence





#%% Plot the regional variables


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
