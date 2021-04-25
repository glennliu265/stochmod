#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calc Autocorrelation (stochastic model output)
new postprocessing to calculate autocorrelation at each gridpoint


/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/stoch_output_SLAB_PIC_1000yr_funiform1_run300_fscale001_applyfac2.npy

Created on Mon Apr  5 11:30:16 2021
@author: gliu
"""

import numpy as np
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cartopy.crs as ccrs
import cmocean
from pylab import cm
#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    outpathfig  = projpath + '02_Figures/20210406/'
   
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
import yo_box as ybx

#%% User Edits

# Experiment ID information
fscale    = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
applyfac  = 2
mconfig   = "SLAB_PIC"
runid     = "302"
funiform  = 1.5
bboxsim   =[-100,20,-20,90]


# Data Loading Options
recalc_auto = False
recalc_spectra = False

# Analysis Options
lags      = np.arange(0,37,1)
bbox_NA   = [-80,0 ,0,65]
lonf,latf =-30,50

# Plotting Options
bboxplot  = [-100,20,-10,80]

#%% Load Data

expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)
#% ---- Read in Data ----
#start = time.time()

# Read in Stochmod SST Data
sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Load MLD Data
if mconfig == "SLAB_PIC": # Output was mistakenly named slab_pic, which doesnt have variable mld
    mld = np.load(rawpath+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
else:
    mld = np.load(rawpath+mconfig+"_HMXL_hclim.npy") # Climatological MLD

# Load full lat/lon
dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp       = loadmat(rawpath+dampmat)
lon            = np.squeeze(loaddamp['LON1'])
lat            = np.squeeze(loaddamp['LAT'])
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()

# Load Autocorrelation
cesmslabac     = np.load(datpath+"../CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
cesmfullac      = np.load(datpath+"../CESM_clim/TS_FULL_Autocorrelation.npy")


# Load full sst data from model
ld  = np.load(datpath+"../FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
sstfull = ld['TS']
ld2 = np.load(datpath+"../SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
sstslab = ld2['TS']

query = [lonf,latf]
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
expcolors = ('blue','orange','magenta','red')

#%% Test for a single point

# Get data for point
klon,klat   = proc.find_latlon(lonf,latf,lon,lat) # Globa, 180 lon
klon360,_   = proc.find_latlon(lonf+360,latf,lon360,lat) # Globa, 360 lon
klonr,klatr = proc.find_latlon(lonf,latf,lonr,latr) # NAtl, 180lon



# Get MLD Information
mldpt=mld[klon,klat,:]
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)

# Get comparison data from CESM SLAB and Full Simulations
slabauto= cesmslabac[kmonth,lags,klat,klon360]
fullauto= cesmfullac[kmonth,lags,klon360,klat]

# Get point value for simulation output
sstpts = []
for i in range(4):
    sstpt = sst[i][klonr,klatr,...]
    sstpts.append(sstpt)
    
# Calculate Autocorrelation
ac = scm.calc_autocorr(sstpts,lags,kmonth+1)

# Make test Plot
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation (Stochastic Model, %i yr) \n %s (Lag 0 = %s); funiform=%i" % (nyrs,locstringtitle,mons3[mldpt.argmax()],funiform)
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,slabauto[lags],label="CESM SLAB",color='k')
ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')

for i in range(1,4):
    ax.plot(lags,ac[i],label=labels[i],color=expcolors[i])
    #ax.plot(lags,ac2[i,...],label=labels[i],color=expcolors[i])

ax.legend()
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpathfig+"Compare_Autocorrelation_CESM.png",dpi=200)
#%% Compute the autocorrelation matrix for all months, all lags

# Loop through each point. and calculate an autocorrelation curve
if recalc_auto:
    pointsize = len(lonr)*len(latr)
    startloop = time.time()
    
    
    nlonr,nlatr,nmon = sst[0].shape
    sstnp = np.zeros((4,nlonr,nlatr,nmon))*np.nan
    
    for i in range(4):
        sstnp[i,:,:,:] = sst[i]
    sstnp = sstnp.transpose(0,3,1,2) #[model x time x lon x lat]
    sstrs = sstnp.reshape(4,nmon,pointsize)
    
    
    # Preallocate 
    autocorr_all = np.ones((4,12,len(lags),nlonr,nlatr)) * np.nan
    for e in range(4):
    
        enstime = time.time()
    
    
        # Get ensemble [time x space]
        sstens = sstrs[e,:,:]
    
        # Isolate non-nan points, summing along dimension zero
        oksst,knan,okpts = proc.find_nan(sstens,0)
    
        # Get dimensions and reshape the time to [month x yr x space]
        timedim,spacedim = oksst.shape 
        oksst = np.reshape(oksst,(int(timedim/12),12,spacedim))
        oksst = np.transpose(oksst,(1,0,2))
        
        # Preallocate and loop for each month...
        autocorrm = np.ones((12,len(lags),spacedim)) * np.nan
    
        # Loop for the months
        for m in range(12):
    
            # Calculate autocorrelation for that month
            autocorrm[m,:,:] = proc.calc_lagcovar_nd(oksst,oksst,lags,m+1,0)
    
            msg = "Completed Mon %02d for ENS %02d (Elapsed: %.2fs)\r" % (m+1,e+1,time.time()-startloop)
            print(msg,end="\r",flush=True)
    
        #end month loop
    
        # Preallocate array to re-insert nan points
        autocorr = np.ones((12,len(lags),pointsize)) * np.nan
        autocorr[:,:,okpts] = autocorrm
    
        autocorr = np.reshape(autocorr,(12,len(lags),nlonr,nlatr))
        
        autocorr_all[e,...] = autocorr.copy()
        
    # Save output
    np.save(outpathdat+"Stochmod_Autocorrelation_%s.npy"%(expid),autocorr_all)
    #print("\nCompleted ENS %02d in %.2fs" % (e+1,time.time()-enstime))
    
    
    
    ac2 = autocorr_all[:,kmonth,:,klonr,klatr]
else:
    print("Loaded Old Data)")
    autocorr_all = np.load(outpathdat+"Stochmod_Autocorrelation_%s.npy"%(expid))
    

#%% Calculate RMSE for autocorrelation

debug = True

# -----------
# Make a mask
# -----------
sstcpy = sst[0]
sstcpy = sstcpy.sum(2)
sstcpy[~np.isnan(sstcpy)] = 1
plt.pcolormesh(sstcpy.T),plt.colorbar()
msk = sstcpy
nlonr,nlatr = msk.shape

# ----------------------------------------------
# First make all arrays into the same dimensions
# [Model (4) x Month (12) x Lag (37) x Lon (97) x Lat (117)]
# ----------------------------------------------

# Preprocess ACs
cesmslabac1 = cesmslabac[:,lags,:,:].transpose(0,1,3,2) # [mon x lag x lon x lat]
cesmacs = [cesmslabac1,cesmfullac]

cesmacproc = []
for ac in cesmacs:
    
    # Make into [lon x lat x otherdims]
    _,nlag,nlon,nlat = ac.shape
    ac = ac.reshape(12*nlag,nlon,nlat)
    ac = ac.transpose(1,2,0)
    
    # Flip longitude
    lon1,acflip=proc.lon360to180(lon360,ac)
    if debug:
        plt.pcolormesh(acflip[:,:,22].T)
        plt.show()
        
    # Restrict to region, apply mask
    acreg,_,_ = proc.sel_region(acflip,lon1,lat,bboxsim)
    acreg *= msk[:,:,None]
    
    
    # Make into [month x lag x lon x lat]
    acreg = acreg.transpose(2,0,1)
    acreg = acreg.reshape(12,nlag,nlonr,nlatr)
    if debug:
        plt.pcolormesh(acreg[1,11,:,:].T),plt.colorbar()
        plt.show()
        plt.plot(acreg[kmonth,:,klonr,klatr])
        
    # Append to new plot
    cesmacproc.append(acreg)
    
#%% Calculate RMSE with selected simulation

# Calculate squared errors
slaberror = (autocorr_all-cesmacproc[0][None,...])**2
fullerror = (autocorr_all-cesmacproc[1][None,...])**2

# Find maximum mixed later depths
mldreg,_,_= proc.sel_region(mld,lon1,lat,bboxsim)
mldreg *= msk[:,:,None]
plt.pcolormesh(mldreg[:,:,0].T)
hmaxreg = np.argmax(mldreg,2)

# Quickly select the relevant months
slaberrorh = np.zeros((4,nlag,nlonr,nlatr))*np.nan
fullerrorh = np.zeros(slaberrorh.shape)*np.nan
for a in tqdm(range(nlatr)):
    for o in range(nlonr):
        
        if np.any(np.isnan(mldreg[o,a,:])):
            continue
        
        # Get month of maximum MLD
        hid = hmaxreg[o,a]
        slaberrorh[:,:,o,a] = slaberror[:,hid,:,o,a]
        fullerrorh[:,:,o,a] = fullerror[:,hid,:,o,a]

#%% Test Plot
i = 3
cname  = ["SLAB_PIC","FULL_PIC"]
errorh = [slaberrorh,fullerrorh]
mc = 1


fig,ax = plt.subplots(1,1)
ax2 = ax.twinx()
ax2.bar(lags,errorh[mc][i,:,klonr,klatr],label="error",alpha=0.25,color='r')
#ax2.scatter(lags,fullerrorh[i,:,klonr,klatr],label="error",color='r')
ax2.set_ylabel("Squared Error")
ax2.set_xticks(lags[::3])


ax.plot(lags,cesmacproc[mc][kmonth,:,klonr,klatr],label="CESM %s"%cname[mc])
ax.plot(lags,autocorr_all[i,kmonth,:,klonr,klatr],label="Stochastic Model %s"%labels[i])
ax.grid(True)
ax.set_xticks(lags[::3])
ax.set_ylabel("Correlation")
ax.set_label("Lag (Months)")
ax.set_title("MSE=%f for Autocorrelation, 50N 30W"%(errorh[mc][i,:,klonr,klatr].mean()))
ax.legend()

plt.savefig("%sMSE_Compare_AC_%s_model%i_v_%s.png"%(outpathfig,expid,i,cname[mc]),dpi=200)
#%% Make maps of the squared error

errorh = [slaberrorh,fullerrorh]
cname  = ["SLAB_PIC","FULL_PIC"]

mc = 1
i  = 3


plotmse = errorh[mc].mean(1)[i,...]
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm=ax.contourf(lonr,latr,plotmse.T,levels=5,cmap="Blues")
ax.scatter(lonf,latf,100,color='red',marker="+")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Mean-squared-error in Autocorrelation \n Stochastic Model (%s) - CESM (%s)"%(labels[i],cname[mc]))
plt.savefig("%sMSE_Map_%s_model%i_v_%s.png"%(outpathfig,expid,i,cname[mc]),dpi=200)

#%% Make Histogram

smean   = np.nanmean(plotmse)
smedian = np.nanmedian(plotmse)
sstd    = np.nanstd(plotmse)
spt     = plotmse[klonr,klatr]

fig,ax= plt.subplots(1,1)
bins = 25#np.arange(0,0.155,0.005)
ax.hist(plotmse.flatten(),bins,edgecolor='w',lw=.75,alpha=1,color='cornflowerblue')
ax.set_title("Mean-squared-error in Autocorrelation \n Stochastic Model (%s) - CESM (%s)"%(labels[i],cname[mc]))


ax.axvline(smean,label=r"$\mu=%.3f$" % smean,color='k')
#ax.axvline(smean,label=r"$Median=%.3f$" % smedian,color='gray')
ax.vlines([smean-sstd,smean+sstd],ymin=0,ymax=750,color='k',ls='dashed',label=r"$\pm1\sigma=%.3f$" % sstd)
ax.vlines([spt],ymin=0,ymax=500,color='r',label="MSE for Lon %i Lat %i (%.3f)"%(lonf,latf,spt))
ax.legend()

ax.set_ylabel("Count")
ax.set_xlabel("MSE")
plt.savefig("%sMSE_Histogram_%s_model%i_v_%s.png"%(outpathfig,expid,i,cname[mc]),dpi=200)

#%% Identify "Best Model"

#allerr = fullerrorh.copy() # Copy full error
#allerr[:2,:,:,:] = slaberrorh[:2,:,:,:] # Copy slab error over
mc =1
allerr=errorh[mc]
include50 = False
# 
meanallerr = allerr.mean(1)
if include50 is False:
    meanallerr=meanallerr[1:,...]
    tklab = ['Mean','Seasonal','Entrain']
    ncat = 3
else:
    ncat=4
    tklab = ['50m','Mean','Seasonal','Entrain']
bestmod = np.argmin(meanallerr[:,...],0)
bestmod = np.array(bestmod).astype('float')
bestmod *= msk


cmap = cm.get_cmap('cmo.dense',ncat)
fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm = ax.pcolormesh(lonr,latr,bestmod.T,cmap=cmap)
ax.scatter(lonf,latf,100,color='red',marker="+")
#cbar=plt.colorbar(pcm,ax=ax,fraction=0.045)
cbar = fig.colorbar(pcm,ax=ax,aspect = 20, ticks=np.arange(0,4,1),fraction=0.035,orientation='horizontal')
cbar.set_ticklabels(tklab)
ax.set_title("Best Model (lowest MSE) \n Stochastic Model - CESM (%s)"%cname[mc])
#plt.tight_layout()
plt.savefig("%sMSE_map_%s_bestmod_CESM%s.png"%(outpathfig,expid,cname[mc]),dpi=200)



#%% Visualize a point

mc = 1
i  = 3

lonf = -40
latf = 25
klon

plotmse = errorh[mc].mean(1)[i,...]
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm=ax.contourf(lonr,latr,plotmse.T,levels=5,cmap="Blues")
ax.scatter(lonf,latf,100,color='red',marker="+")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Mean-squared-error in Autocorrelation \n Stochastic Model (%s) - CESM (%s)"%(labels[i],cname[mc]))
plt.savefig("%sMSE_Map_%s_model%i_v_%s.png"%(outpathfig,expid,i,cname[mc]),dpi=200)



#%% Visualize Results from a Point (Autocorrelation (Redone 04.16) -----------------


lonf = -30
latf = 50
locfn = "lon%i_lat%i"%(lonf,latf)
loctitle = "Lon: %i Lat: %i"%(lonf,latf)

# Reobtain the indices
klon,klat   = proc.find_latlon(lonf,latf,lon,lat) # Global, 180 lon
klon360,_   = proc.find_latlon(lonf+360,latf,lon360,lat) # Global, 360 lon
klonr,klatr = proc.find_latlon(lonf,latf,lonr,latr) # NAtl, 180lon

# Get MLD Information
mldpt=mld[klon,klat,:]
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)

# Get comparison data from CESM SLAB and Full Simulations
slabauto= cesmslabac[kmonth,lags,klat,klon360]
fullauto= cesmfullac[kmonth,lags,klon360,klat]
stochauto = autocorr_all[:,kmonth,lags,klonr,klatr] # [model, month, lag, lon, lat]

#%% Make maps of the squared error
#% 
bboxplot = [-100,20,-10,80]
errorh = [slaberrorh,fullerrorh]
cname  = ["SLAB_PIC","FULL_PIC"]
mc = 1
i  = 3
plotmse = errorh[mc].mean(1)[i,...]
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm=ax.contourf(lonr,latr,plotmse.T,levels=5,cmap="Blues")
ax.scatter(lonf,latf,100,color='red',marker="+")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Mean-squared-error in Autocorrelation \n Stochastic Model (%s) - CESM (%s)"%(labels[i],cname[mc]))
plt.savefig("%sMSE_Map_%s_model%i_v_%s_%s.png"%(outpathfig,expid,i,cname[mc],locfn),dpi=200)


# Plot Autocorrelation
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation (Stochastic Model, %i yr) \n %s (Lag 0 = %s)" % (nyrs,loctitle,mons3[mldpt.argmax()])
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,slabauto[lags],label="CESM SLAB",color='k')
ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')
for i in range(1,4):
    ax.plot(lags,stochauto[i],label=labels[i],color=expcolors[i])
ax.legend()
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpathfig+"Compare_Autocorrelation_CESM_%s_funiform%i.png"%(locfn,funiform),dpi=200)

#%% Postprocess Spectra

# Load Data

#%% Select data for point
sstfullpt = sstfull[:,klat,klon360]
sstslabpt = sstslab[:,klat,klon360]


# Select data from stochastic model
# -----------------------------------------------------------------
sststochpt = []
for m in range(4):
    sstpt  = sst[m][klonr,klatr,:]
    sststochpt.append(sstpt)
sststochpt = np.array(sststochpt) # {model x time}



# -----------------------------------------------------------------
#%% Plot Timeseries
# -----------------------------------------------------------------

nmean = 10


# -----------------------
# CESM time plot Model Output Time Plot
# -----------------------
fig,ax=plt.subplots(1,1,figsize=(8,3))
csst = [sstfullpt,sstslabpt]
ccol = ['k','gray']
clab = ["CESM-Full","CESM-Slab"]
for i in range(2):
    
    sstann = proc.ann_avg(csst[i],0)
    
    win = np.ones(nmean)/nmean
    sstann = np.convolve(sstann,win,mode='valid')
    
    yrs = np.arange(0,sstann.shape[0])
    
    plabel = clab[i] + r", 1$\sigma=%.2f$" % np.std(sstann)
    ax.plot(sstann,label=plabel,lw=0.5,color=ccol[i])
    
    print("Std for %s is %.2f"%(labels[i],np.std(sst[i])))
ax.legend(fontsize=8,ncol=3)
ax.set_xlabel("Years")
ax.set_ylabel("degC")
ax.set_title("CESM SST (Annual) at %s (%i-year Running Mean) "% (loctitle,nmean))
ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig("%sCESMSST_comparison_%s_funiform%i.png" % (outpathfig,locfn,funiform),dpi=150)




# -----------------------
# Stochastic Model Output Time Plot
# -----------------------

fig,ax=plt.subplots(1,1,figsize=(8,3))

for i in [1,2,3]:
    
    sstann = proc.ann_avg(sststochpt[i]*1,0)
    plabel = labels[i] + r", 1$\sigma=%.2f$" % np.std(sstann)
    win = np.ones(nmean)/nmean
    sstann = np.convolve(sstann,win,mode='valid')
    
    yrs = np.arange(0,sstann.shape[0])
    
    
    ax.plot(sstann,label=plabel,lw=0.5,color=expcolors[i])
    #ax.plot(sst[i],label=plabel,lw=0.5,color=expcolors[i])
    print("Std for %s is %.2f"%(labels[i],np.std(sststochpt[i])))
    print("Std for Ann mean %s is %.2f"%(labels[i],np.std(sstann)))
ax.legend(fontsize=8,ncol=3)
ax.set_xlabel("Years")
ax.set_ylabel("degC")
ax.set_title("Stochastic Model SST at %s (%i-year Running Mean) "% (loctitle,nmean))
ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig("%sStochasticModelSST_comparison_%s_funiform%i_nmean%i.png" % (outpathfig,locfn,funiform,int(nmean)),dpi=150)




#%% Spectral Calculations

# Parameters
pct     = 0.10
nsmooth = 100
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1
specnames = "nsmooth%i_taper%i" % (nsmooth,pct*100)

# -----------------------------------------------------------------
# First, calculate for CESM
# -----------------------------------------------------------------
nsmooths = [nsmooth,nsmooth*2]
freq1s,P1s,CLs = [],[],[]
for i,sstin in enumerate([sstfullpt,sstslabpt]):
    
    sps = ybx.yo_spec(sstin,opt,nsmooths[i],pct,debug=False)
    P,freq,dof,r1=sps
    
    CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
    #pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    
    P    = P*dt
    freq = freq/dt
    CC   = CC*dt
    
    P1s.append(P)
    freq1s.append(freq)
    CLs.append(CC)
    
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s
clfull,clslab = CLs

# All CESM Spectra
fig,ax = plt.subplots(1,1)
ax.set_ylabel("Frequency x Power",fontsize=13)
for i in [0,1]:
    ax.semilogx(freq1s[i],freq1s[i]*P1s[i],label=clab[i],color=ccol[i],lw=0.75)
xmin = 10**(np.floor(np.log10(np.min(freq))))
ax.set_xlim([xmin,0.5/dt])
ax.grid(True,ls='dotted')
freqtick = ax.get_xticks()
yy = ax.get_ylim()
# ax.set_xscale("log")
# ax.set_yscale("linear")
period =1/freq
htax   =ax.twiny()
htax.set_xscale("log")
htax.set_yscale("linear")
xtkl = ["%.1f"% s for s in np.fix(1/freqtick/dt)]
htax.set_xlim([xmin,0.5/dt])
htax.set_xlabel("Period (%s)"%tunit,fontsize=13)
ax,htax=viz.make_axtime(ax,htax)
ax.legend()
vlv = [1/(3600*24*365*100),1/(3600*24*365*10),1/(3600*24*365)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
ax.set_xlabel("Frequency (cycles/year)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum at %s \n"%loctitle + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectraCESM_%s_funiform%i_nsmooth%i_pct%03d_axopt%i.png"%(outpathfig,'COMPARISON',funiform,nsmooth,pct*100,axopt),dpi=200)



# -----------------------------------------------------------------
# Calculate and make individual plots for stochastic model output
# -----------------------------------------------------------------
specparams  = []
splotparams = []
specs = []
freqs = []
for i in range(4):
    sstin = sststochpt[i]
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    specs.append(P*dt)
    freqs.append(freq/dt)
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    fig,ax,h,hcl,htax,hleg = pps
    
    if i < 2:
        l1 =ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
        l2 =ax.semilogx(freqcesmslab,clslab[:,0]*freqcesmslab,label="CESM-SLAB (AR1)",color='red',lw=0.75,alpha=0.4)
        l3 =ax.semilogx(freqcesmslab,clslab[:,1]*freqcesmslab,label="CESM-SLAB (95%)",color='blue',lw=0.75,alpha=0.4)
    else:
        l1 =ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='gray',lw=0.75)
        l2 =ax.semilogx(freqcesmfull,clfull[:,0]*freqcesmfull,label="CESM-FULL (AR1)",color='red',lw=0.75,alpha=0.4)
        l3 =ax.semilogx(freqcesmfull,clfull[:,1]*freqcesmfull,label="CESM-FULL (95%)",color='blue',lw=0.75,alpha=0.4)

    if axopt != 1:
        dtin = 3600*24*365
        ax,htax=viz.make_axtime(ax,htax)
    
    vlv = [1/(100*365*24*3600),1/(10*365*24*3600),1/(365*24*3600)]
    vll = ["Century","Decade","Year"]
    for vv in vlv:
        ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    
    ax.grid(True,which='both',ls='dotted')
    ax.set_xlabel("Frequency (cycles/year)",fontsize=13)
    ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
    ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
    plt.tight_layout()
    plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i_%s_funiform%i.png"%(outpathfig,labels[i],nsmooth,pct*100,axopt,locfn,funiform),dpi=200)

# %%
#% All stochastic model spectra
# Make the plot ---
# Set up variance preserving plot
freq = freqs[0]
fig,ax = plt.subplots(1,1)
ax.set_ylabel("Frequency x Power",fontsize=13)
for i in [1,2,3]:
    ax.semilogx(freqs[i],freqs[i]*specs[i],label=labels[i],color=expcolors[i],lw=0.75)
    
ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='k',lw=0.75)
xmin = 10**(np.floor(np.log10(np.min(freq))))
ax.set_xlim([xmin,0.5/dt])
ax.grid(True,ls='dotted')
freqtick = ax.get_xticks()
yy = ax.get_ylim()
# ax.set_xscale("log")
# ax.set_yscale("linear")
period =1/freq
htax   =ax.twiny()
htax.set_xscale("log")
htax.set_yscale("linear")
xtkl = ["%.1f"% s for s in np.fix(1/freqtick/dt)]
htax.set_xlim([xmin,0.5/dt])
htax.set_xlabel("Period (%s)"%tunit,fontsize=13)
ax,htax=viz.make_axtime(ax,htax)
ax.legend()
vlv = [1/(3600*24*365*100),1/(3600*24*365*10),1/(3600*24*365)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
ax.set_xlabel("Frequency (cycles/year)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum at %s \n"%loctitle + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_COMPARISON_funiform%i_nsmooth%i_pct%03d_axopt%i.png"%(outpathfig,funiform,nsmooth,pct*100,axopt),dpi=200)


#%% Estimate Spectra for the whole basin (CESM)


# Load global mask
mskglob = np.load(datpath+"../landicemask_enssum.npy")


sstins = [sstfull,sstslab]
lens   = [10788,5388]
_,nlat,nlon = sstfull.shape

# Recalculate Spectra for CESM
if recalc_spectra:
    spectraest = []
    freqest = []
    #CCest = []
    for m in range(2):
        
        sstin = sstins[m]
        flen  = lens[m]
        
        mspec = np.zeros((flen,nlat,nlon)) * np.nan
        mfreq = np.zeros(mspec.shape) * np.nan
        #mCC  = np.zeros((flen,nlat,nlon,2)) * np.nan
        for o in tqdm(range(len(lon360))):
            
            for a in range(len(lat)):
                # Skip NaN
                if np.isnan(mskglob[a,o]):
                    continue
                
                sstptin = sstin[:,a,o]
                
                # Estimate Spectra
                sps = ybx.yo_spec(sstptin,opt,nsmooths[m],pct,debug=False)
                P,freq,dof,r1=sps
                #CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
                
                P    = P*dt
                freq = freq/dt
                #CC   = CC*dt
                
                #mCC[:,a,o,:] = CC
                mfreq[:,a,o] = freq
                mspec[:,a,o] = P
        # Completed model
        spectraest.append(mspec)
        freqest.append(mfreq)
        #CCest.append(mCC)
                    
    
    
       
        
    # Save data
    np.savez("%sCESM_PIC_Spectra_%s.npz"%(datpath,specnames),**{
        'freqslab':freqest[1],
        'freqfull':freqest[0],
        'specslab':spectraest[1],
        'specfull':spectraest[0]
        },allow_pickle=True)
else:
    
    ld = np.load("%sCESM_PIC_Spectra_%s.npz"%(datpath,specnames),allow_pickle=True)
    
    spectraest = []
    freqest = []
    
    freqest.append(ld['freqfull'])
    freqest.append(ld['freqslab'])
    
    spectraest.append(ld['specfull'])
    spectraest.append(ld['specslab'])

    
# Test Plot
klon360,klat = proc.find_latlon(330,50,lon360,lat)
fig,ax = plt.subplots(1,1)
ax.semilogx(freqest[0][:,klat,klon360],freqest[0][:,klat,klon360]*spectraest[0][:,klat,klon360],color='k',label="Full")
ax.semilogx(freqest[1][:,klat,klon360],freqest[1][:,klat,klon360]*spectraest[1][:,klat,klon360],color='gray',label="Slab")
     
#%% Make Difference Plots based on Power Spectra

plotdiff = False

# Set Names
plotname = 'Ratio (Full/Slab)'
plotnamesave = "Ratio_FulltoSlab"
if plotdiff:
    plotname = 'Difference (Full - Slab)'
    plotnamesave = 'Diff_FullminusSlab'


# Calculate Area Under Curve
fsums = []
for i in range(2):
    tfreq = freqest[i]
    tpwr  = spectraest[i]

    # Sum area under spectra
    df   = np.mean(tfreq[1:,...] - tfreq[:-1,...],0)
    fsum = np.sum(tpwr*df[None,:,:],0)
    fsums.append(fsum)



# Calculate variance difference directly from slab



# ---------------------------------
# First plot difference in variance
# ---------------------------------
if plotdiff: # Differences
    vardiff = np.var(sstfull,0) - np.var(sstslab,0)
    pdiff   = fsums[0] - fsums[1]
    
    vlm = [-.5,.5]
    clevs = [0]#np.arange(-.5,.6,.05)
    cflevs = np.arange(-.5,.5,.05)
    
else: # Ratio
    vardiff = np.var(sstfull,0)/np.var(sstslab,0)
    pdiff   = fsums[0] / fsums[1]
    vlm = [0,2]
    clevs = [0.5,1]
    cflevs = np.arange(0,2.05,.05)
    
    
vardiff *= mskglob
pdiff *= mskglob


fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm1=ax.pcolormesh(lon360,lat,vardiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
pcm = ax.contourf(lon360,lat,vardiff,levels=cflevs,cmap=cmocean.cm.balance)
cl = ax.contour(lon360,lat,vardiff,levels=clevs,colors='k',linewidths=.75)
ax.clabel(cl,fmt="%.2f")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Variance %s ($^{\circ}C^{2}$)"%(plotname))
plt.savefig("%sCESM_Variance%s.png"%(outpathfig,plotnamesave),dpi=200)

# -------------------------------------------
# Next, plot difference in spectral estimates
# -------------------------------------------
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm1=ax.pcolormesh(lon360,lat,pdiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
pcm = ax.contourf(lon360,lat,pdiff,levels=cflevs,cmap=cmocean.cm.balance)
cl = ax.contour(lon360,lat,pdiff,levels=clevs,colors='k',linewidths=.75)
ax.clabel(cl,fmt="%.2f")
#pcm=ax.contourf(lon360,lat,vardiff,levels=100,cmap="Blues")
#ax.scatter(lonf,latf,100,color='red',marker="+")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Spectra %s ($^{\circ}C^{2}$)"%(plotname))
plt.savefig("%sCESM_Spectrum%s.png"%(outpathfig,plotnamesave),dpi=200)

#%% Sum Spectra within certain ranges


vlv = [1/(3600*24*365*100),1/(3600*24*365*10),1/(3600*24*365)]

thresall = []
for i in range(2):
    tfreq = freqest[i]
    tpwr  = spectraest[i]

    freqq = tfreq[:,klat,klon360]
    df = np.mean(freqq[1:] - freqq[:-1])
    
    fsumthres = []
    for t in range(3):
        
        
        if t == 0: # Highest Frequencies (Less than 1 Year)
            kfreq = freqq >= vlv[-1]
        elif t == 1: # Interannual (1 year to 10 years)
            kfreq = (freqq < vlv[-1]) * (freqq > vlv[-2])
        elif t == 2: # Lowest Frequencies (>10 years)
            kfreq = freqq <= vlv[-2]
            
        
        fsum = np.sum(tpwr[kfreq,:,:]*df,0)
        fsumthres.append(fsum)
    thresall.append(fsumthres)
        

if plotdiff:
    anndiff  = thresall[0][0] - thresall[1][0]
    ianndiff = thresall[0][1] - thresall[1][1]
    decdiff  = thresall[0][2] - thresall[1][2]
    
    vlm = [-.25,.25]
    clevs = [0]#np.arange(-.5,.6,.05)

else:
    anndiff  = thresall[0][0] / thresall[1][0]
    ianndiff = thresall[0][1] / thresall[1][1]
    decdiff  = thresall[0][2] / thresall[1][2]



#%% Plot Differences by Spectral Period

# ------
# Annual
# ------
clevs = [0.5,1]
cflevs = np.arange(0,2.05,.05)
vlm = [cflevs[0],cflevs[-1]]

    
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm1 =ax.pcolormesh(lon360,lat,anndiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
pcm = ax.contourf(lon360,lat,anndiff,levels=cflevs,cmap=cmocean.cm.balance)
# First label (0-1)
cl = ax.contour(lon360,lat,anndiff,levels=clevs,colors='k',linewidths=.75)
ax.clabel(cl,fmt="%.2f")
# Second Label (>1)
cl2 = ax.contour(lon360,lat,anndiff,levels=[2,3],colors='w',linewidths=.75)
ax.clabel(cl2,fmt="%.2f")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Spectra %s ($^{\circ}C^{2}$) \n Periods Less than 1 Year"%(plotname))
plt.savefig("%sCESM_Spectrum%sAnn.png"%(outpathfig,plotnamesave),dpi=200)

# -----------
# Interannual
# -----------
if plotdiff:
    vlm = [-.5,.5]
    cflevs = np.arange(-.5,.55,.05)
    clevs = [0]#np.arange(-.5,.6,.05)
else:
    clevs = [0.25,0.5,1]
    cflevs = np.arange(0,2.05,.05)
    vlm = [cflevs[0],cflevs[-1]]
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
#pcm=ax.pcolormesh(lon360,lat,ianndiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
#cl = ax.contour(lon360,lat,ianndiff,levels=clevs,colors='k',linewidths=.75)
pcm1 =ax.pcolormesh(lon360,lat,ianndiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
pcm = ax.contourf(lon360,lat,ianndiff,levels=cflevs,cmap=cmocean.cm.balance)
cl = ax.contour(lon360,lat,ianndiff,levels=clevs,colors='k',linewidths=.75)
ax.clabel(cl,fmt="%.2f")
cl2 = ax.contour(lon360,lat,ianndiff,levels=[2,3],colors='w',linewidths=.75)
ax.clabel(cl2,fmt="%.2f")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Spectra %s ($^{\circ}C^{2}$) \n Annual to Decadal Periods"%(plotname))
plt.savefig("%sCESM_Spectrum%sIntAnn.png"%(outpathfig,plotnamesave),dpi=200)

# -------
# Decadal
# -------
if plotdiff:
    vlm = [-.25,.25]
    cflevs = np.arange(-.25,.26,.01)
    clevs = [0]#np.arange(-.5,.6,.05)
else:
    clevs = [0.25,0.5,1]
    cflevs = np.arange(0,2.05,.05)
    vlm = [cflevs[0],cflevs[-1]]
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection': ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bboxplot)
pcm1 =ax.pcolormesh(lon360,lat,decdiff,vmin=vlm[0],vmax=vlm[1],cmap=cmocean.cm.balance)
pcm = ax.contourf(lon360,lat,decdiff,levels=cflevs,cmap=cmocean.cm.balance)

cl = ax.contour(lon360,lat,decdiff,levels=clevs,colors='k',linewidths=.75)
ax.clabel(cl,fmt="%.2f")
cl2 = ax.contour(lon360,lat,decdiff,levels=[2,3],colors='w',linewidths=.75)
ax.clabel(cl2,fmt="%.2f")
fig.colorbar(pcm,ax=ax,fraction=0.045)
ax.set_title("Spectra %s ($^{\circ}C^{2}$) \n Multidecadal Periods"%(plotname))
plt.savefig("%sCESM_Spectrum%sDecadal.png"%(outpathfig,plotnamesave),dpi=200)




