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

#%% User Edits

# Experiment ID information
fscale    = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
applyfac  = 2
mconfig   = "SLAB_PIC"
runid     = "300"
funiform  = 1.5
bboxsim   =[-100,20,-20,90]

recalc_auto = False

# Analysis Options
lags = np.arange(0,37,1)
bbox_NA = [-80,0 ,0,65]
lonf,latf=-30,50

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

#plt.savefig(outpath+"Compare_Autocorrelation_CESM.png",dpi=200)

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
    
            msg = "Completed Mon %02d for ENS %02d (Elapsed: %.2fs)" % (m+1,e+1,time.time()-startloop)
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

# Test Plot
i = 3
fig,ax = plt.subplots(1,1)
ax2 = ax.twinx()
ax2.bar(lags,fullerrorh[i,:,klonr,klatr],label="error",alpha=0.25,color='r')
#ax2.scatter(lags,fullerrorh[i,:,klonr,klatr],label="error",color='r')
ax2.set_ylabel("Squared Error")
ax2.set_xticks(lags[::3])


ax.plot(lags,cesmacproc[1][kmonth,:,klonr,klatr],label="CESM")
ax.plot(lags,autocorr_all[i,kmonth,:,klonr,klatr],label="Stochastic Model")
ax.grid(True)
ax.set_xticks(lags[::3])
ax.set_title("MSE=%f"%(fullerrorh[i,:,klonr,klatr].mean()))
ax.legend()

#%% Make maps of the squared error
import cartopy.crs as ccrs
import cmocean
from pylab import cm

bboxplot = [-100,20,-10,80]
errorh = [slaberrorh,fullerrorh]
cname  = ["SLAB_PIC","FULL_PIC"]

mc = 1
i  = 2


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
