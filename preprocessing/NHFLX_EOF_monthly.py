#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate EOFs for Surface Heat Flux Anomalies


Created on Tue Jul 20 11:49:09 2021

@author: gliu
"""

import xarray as xr
import numpy as np
import glob
import time

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm


#%%
stormtrack = 0

if stormtrack == 1:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
elif stormtrack == 0:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    #datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20210804/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"

from amv import proc,viz
import scm


def sel_regionxr(ds,bbox):
    """
    Select region from xr.dataset with 'lon' (degrees East), 'lat'.
    Current supports crossing the prime meridian

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    bbox : TYPE
        DESCRIPTION.

    Returns
    -------
    dsreg : TYPE
        DESCRIPTION.

    """
    
    # Select lon (lon360)
    if bbox[0] > bbox[1]: # Crossing Prime Meridian
        print("Crossing Prime Meridian!")
        dsreg = ds.isel(lon =(ds.lon < bbox[1]) + (ds.lon>bbox[0]))
        dsreg = dsreg.sel(lat=slice(bbox[2],bbox[3]))
    else:
        dsreg = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    return dsreg

#%%
mconfig = "PIC_SLAB"
bbox    = [260,20,0,65]
bboxeof = [280,20,0,65]

debug = True

lonf = -30
latf = 50

# EOF parameters
N_mode = 200

# Plotting params
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
blabels=[0,0,0,0]

#%% Open the dataset

# Open the dataset
st      = time.time()
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
    
    # Open dataset
    ds = xr.open_dataset("%sNHFLX_PIC_SLAB_raw.nc" % datpath) # time x lat x lon

    # Apply land/ice mask
    msk = np.load(lipath)
    ds *= msk[None,:,:]
    
    # Select Region
    dsreg = sel_regionxr(ds,bboxeof)
    
    # Read out data
    flxglob = ds.NHFLX.values
    #flxreg = dsreg.NHFLX.values
    lon     = ds.lon.values
    lat     = ds.lat.values
    slpglob = np.load(datpath + "../CESM_proc/PSL_PIC_SLAB.npy")
    print("Loaded data in %.2fs"%(time.time()-st))
    
elif mconfig == "PIC_FULL":
    
    mcname = "FULL-PIC"
    
    # Load Data
    flxglob = np.load("%s../NHFLX_PIC_FULL.npy"%(datpath)) # yr x mon x lat x lon
    lon = np.load(datpath+"CESM1_lon360.npy")
    lon180 = np.load(datpath+"CESM1_lon180.npy")
    lat    = np.load(datpath+"CESM1_lat.npy") 
    
    # Combine time
    nmon,_,nlat,nlon = flxglob.shape
    flxglob = flxglob.reshape(nmon*12,nlat,nlon)
    
    # Apply a mask
    msk = np.load(lipath)
    flxglob *= msk[None,:,:]
    
    # Load SLP
    ds = xr.open_dataset("%s../CESM_proc/PSL_PIC_FULL.nc" % datpath)
    slpglob = ds.PSL.values
    print("Loaded data in %.2fs"%(time.time()-st))



#%% Preprocess, EOF Analysis
ntime,nlat,nlon = flxglob.shape

#% SLP reshape and apply mask ------------------------------------------------
slpglob = slpglob.reshape(ntime,nlat,nlon) # [yr x mon x lat x lon] to [time lat lon]
#slpglob *= msk[None,...]

# Detrend --------------------------------------------------------------------
flxa,linmod,beta,intercept = proc.detrend_dim(flxglob,0)
slpa,_,_,_ = proc.detrend_dim(slpglob,0)
# Plot Spatial Map
if debug: # Appears to be positive into the ocean
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.pcolormesh(lon,lat,flxglob[0,:,:],vmin=-500,vmax=500,cmap="RdBu_r")
    fig.colorbar(pcm,ax=ax)
# Plot Detrending
klon,klat = proc.find_latlon(lonf+360,latf,lon,lat)
if debug: 
    fig,ax = plt.subplots(1,1)
    t = np.arange(0,ntime)
    ax.scatter(t,flxglob[:,latf,lonf],label="Raw")
    ax.plot(t,linmod[:,latf,lonf],label="Model")
    ax.scatter(t,flxa[:,latf,lonf],label="Detrended")
    ax.legend()

# Apply Area Weight (to region) ----------------------------------------------
wgt = np.sqrt(np.cos(np.radians(lat)))

#plt.plot(wgt)

flxwgt = flxa * wgt[None,:,None]
#slpwgt = slpa * wgt[None,:,None] # Don't apply area-weight to regressed variable

# Select region --------------------------------------------------------------
flxreg,lonr,latr = proc.sel_region(flxwgt.transpose(2,1,0),lon,lat,bboxeof)
nlonr,nlatr,_ = flxreg.shape
flxreg = flxreg.transpose(2,1,0) # Back to time x lat x lon

# Remove NaN Points [time x npts] --------------------------------------------
flxwgt = flxa.reshape((ntime,nlat*nlon)) # Dont use weighted variable
okdata,knan,okpts = proc.find_nan(flxwgt,0)
npts = okdata.shape[1]

flxreg = flxreg.reshape((ntime,nlatr*nlonr)) # Use lat weights for EOF region
okdatar,knanr,okptsr = proc.find_nan(flxreg,0)
nptsr = okdatar.shape[1]

nptsall = nlat*nlon
#slpwgt = slpwgt.reshape(ntime,nptsall) # Repeat for slp 
slpwgt = slpa.reshape(ntime,nptsall) # Repeat for slp 
okslp  = slpwgt#[:,okpts]

# Calculate Monthly Anomalies, change to [yr x mon x npts] -------------------
nyr = int(ntime/12)
okdata = okdata.reshape((nyr,12,npts))
okdata = okdata - okdata.mean(0)[None,:,:]
okdatar = okdatar.reshape((nyr,12,nptsr)) # Repeat for region
okdatar = okdatar - okdatar.mean(0)[None,:,:]
okslp = okslp.reshape((nyr,12,nptsall))

# Prepare for eof anaylsis ---------------------------------------------------
eofall    = np.zeros((N_mode,12,nlat*nlon)) * np.nan
eofslp    = eofall.copy()
pcall     = np.zeros((N_mode,12,nyr)) * np.nan
varexpall = np.zeros((N_mode,12)) * np.nan
# Looping for each month
for m in tqdm(range(12)):
    
    # Calculate EOF
    datain = okdatar[:,m,:].T # [space x time]
    regrin = okdata[:,m,:].T
    slpin  = okslp[:,m,:].T
    
    eofs,pcs,varexp = proc.eof_simple(datain,N_mode,1)
    
    # Standardize PCs
    pcstd = pcs / pcs.std(0)[None,:]
    
    # Regress back to dataset
    eof,b = proc.regress_2d(pcstd.T,regrin.T)
    eof_s,_ = proc.regress_2d(pcstd.T,slpin.T)
    
    # if debug:
    #     # Check to make sure both regress_2d methods are the same
    #     # (comparing looping by PC, and using A= [P x N])
    #     eof1 = np.zeros((N_mode,npts))
    #     b1  = np.zeros(eof1.shape)
    #     # Regress back to the dataset
    #     for n in range(N_mode):
    #         eof1[n,:],b1[n,:] = proc.regress_2d(pcstd[:,n],regrin)
    #     print("max diff for eof (matrix vs loop) is %f"%(np.nanmax(np.abs(eof-eof1))))
    #     print("max diff for b (matrix vs loop) is %f"%(np.nanmax(np.abs(b-b1))))
        
    # Save the data
    eofall[:,m,okpts] = eof
    eofslp[:,m,:] = eof_s
    pcall[:,m,:] = pcs.T
    varexpall[:,m] = varexp

# Flip longitude ------------------------------------------------------------
eofall = eofall.reshape(N_mode,12,nlat,nlon)
eofall = eofall.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofall = proc.lon360to180(lon,eofall.reshape(nlon,nlat,N_mode*12))
eofall = eofall.reshape(nlon,nlat,12,N_mode)
# Repeat for SLP eofs
eofslp = eofslp.reshape(N_mode,12,nlat,nlon)
eofslp = eofslp.transpose(3,2,1,0) # [lon x lat x mon x N]
lon180,eofslp = proc.lon360to180(lon,eofslp.reshape(nlon,nlat,N_mode*12))
eofslp = eofslp.reshape(nlon,nlat,12,N_mode)
#%% Save the results
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)

np.savez(savename,**{
    "eofall":eofall,
    "eofslp":eofslp,
    "pcall":pcall,
    "varexpall":varexpall,
    'lon':lon180,
    'lat':lat},allow_pickle=True)




#%% -------Analysis Section Below

#%% Load the data
if mconfig == "PIC_SLAB":
    mcname = "SLAB-PIC"
elif mconfig == "PIC_FULL":
    mcname = "FULL-PIC"

bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
bboxstr  = "Lon %i to %i, Lat %i to %i" % (bbox[0],bbox[1],bbox[2],bbox[3])
savename = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (datpath,mcname,N_mode,bboxtext)
ld = np.load(savename,allow_pickle=True)

eofall    = ld['eofall']
eofslp    = ld['eofslp']
pcall     = ld['pcall']
varexpall = ld['varexpall']

lon = ld['lon']
lat = ld['lat']


#%%  Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG)

spgbox     = [-60,20,40,80]
N_modeplot = 5

for N in tqdm(range(N_modeplot)):
    for m in range(12):
        
        sumflx = proc.sel_region(eofall[:,:,[m],N],lon,lat,spgbox,reg_avg=True)
        sumslp = proc.sel_region(eofslp[:,:,[m],N],lon,lat,spgbox,reg_avg=True)
        
        if sumflx > 0:
            print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
            eofall[:,:,m,N]*=-1
            pcall[N,m,:] *= -1
        if sumslp > 0:
            print("Flipping sign for SLP, mode %i month %i" % (N+1,m+1))
            eofslp[:,:,m,N]*=-1




#%% Save a select number of EOFs

N_mode_choose = 1
eofforce      = eofall.copy()
eofforce      = eofforce.transpose(0,1,3,2) # lon x lat x pc x mon
eofforce      = eofforce[:,:,:N_mode_choose,:]
savenamefrc   = "%sflxeof_%ieofs_%s.npy" % (datpath,N_mode_choose,mcname)
np.save(savenamefrc,eofforce)
print("Saved data to "+savenamefrc)



#%% Check how much variance is explained by each EOF

#eofall.shape = (288, 192, 12, 200)
#pcall.shape = (200, 12, 901)


# Sum N number of modes
nlon,nlat,_,_ = eofall.shape
_,_,nyr = pcall.shape


N_recon = 100
#nhflx_recon = np.zeros((nlon,nlat,12,nyr,N_recon))
nhflx_recon = np.zeros((nlon,nlat,12,nyr))
for N in tqdm( range(N_recon)):
    
    # Get EOF and PC for corresponding mode
    eofN = eofall[:,:,:,N] # lon x lat x month
    pcN  = pcall[N,:,:] # month x time
    
    # Reconstruct the Net Heat Flux for that mode
    #nhflx_recon[:,:,:,:,N] = eofN[...,None] * pcN[None,None,...]
    nhflx_recon[:,:,:,:] += eofN[...,None] * pcN[None,None,...]
    
    


# Quick preprocess slpa for comparison
flxa *= msk[None,:,:]
lon180,flx180 = proc.lon360to180(lon,flxa.transpose(2,1,0))
flx180 = flx180.reshape(nlon,nlat,nyr,12).transpose(0,1,3,2) # lon x lat x mon x year


# Now check the variance percentange
lonf = -30
latf = 50
klon180,klat = proc.find_latlon(lonf,latf,lon180,lat)


vrecon = np.var(nhflx_recon[klon180,klat,:,:,],1)
vorig  = np.var(flx180[klon180,klat,:,:,],1)
vratio =  vrecon/vorig

plt.plot(vratio)

# Reconstruct net heat flux by multiplying EOFs by PCs (might be a bit intensive...)
#nhflx_reconstr = eofall[...,None] * pcall.transpose(1,0,2)[None,None,:,:,:]






#%% Calculate/plot cumulative variance explained

# Calculate cumulative variance at each EOF
cvarall = np.zeros(varexpall.shape)
for i in range(N_mode):
    cvarall[i,:] = varexpall[:i+1,:].sum(0)

# Plot Params
N_modeplot = 50
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)


for m in range(12):
    plt.plot(modes,cvarall[:,m]*100,label="Month %i"% (m+1),marker="o",markersize=4)
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
plt.savefig("%s%s_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,mcname,N_modeplot,bboxtext),dpi=150)


#%% Find index of variance threshold

vthres  = 0.90
thresid = np.argmax(cvarall>vthres,axis=0)
thresperc = []
for i in range(12):
    
    print("Before")
    print(cvarall[thresid[i]-1,i])
    print("After")
    print(cvarall[thresid[i],i])
    
    # Append percentage
    thresperc.append(cvarall[thresid[i],i])
thresperc = np.array(thresperc)


#ytk = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax.bar(mons3,thresid,color=[0.56,0.90,0.70],alpha=0.80)
ax.set_title("Number of EOFs required \n to explain %i"%(vthres*100)+"% of the NHFLX variance")
#ax.set_yticks(ytk)
ax.set_ylabel("# EOFs")
ax.grid(True,ls='dotted')




plt.savefig("%s%s_NHFLX_EOFs%i_%s_NumEOFs_%ipctvar_bymon.png"%(outpath,mcname,ArithmeticErrorN_mode,bboxtext,vthres*100),dpi=150)

#%% Save outptut as forcing for stochastic model, variance based threshold


# Calculate correction factor
eofcorr  = True
if eofcorr:
    ampfactor = 1/thresperc
else:
    ampfactor = 1



eofforce = eofall.copy() # [lon x lat x month x pc]
cvartest = cvarall.copy()
for i in range(12):
    # Set all points after crossing the variance threshold to zero
    stop_id = thresid[i]
    print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
    eofforce[:,:,i,stop_id+1:] = 0
    cvartest[stop_id+1:,i] = 0
eofforce = eofforce.transpose(0,1,3,2) # [lon x lat x pc x mon]

eofforce *= ampfactor[None,None,None,:]

# Cut to maximum EOF
nmax = thresid.max()
eofforce = eofforce[:,:,:nmax+1,:]

savenamefrc = "%sflxeof_%03ipct_%s_eofcorr%i.npy" % (datpath,vthres*100,mcname,eofcorr)
np.save(savenamefrc,eofforce)

# Test plot maps
i = 0
fig,ax = plt.subplots(2,1)
ax[0].pcolormesh(eofforce[:,:,stop_id,i].T),plt.colorbar()
ax[1].pcolormesh(eofforce[:,:,stop_id+1,i].T),plt.colorbar()

# Test Plot Cumulative % for indexing chec
N_modeplot = 20
modes = np.arange(1,N_mode+1)
xtk = np.arange(1,N_mode+1,1)
ytk = np.arange(15,105,5)
fig,ax = plt.subplots(1,1)
for m in range(12):
    plt.plot(modes,cvartest[:,m]*100,label="Month %i"% (m+1))
ax.legend(fontsize=8,ncol=2)
ax.set_ylabel("Cumulative % Variance Explained")
ax.set_yticks(ytk)
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Cumulative Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xlim([1,N_modeplot])
#ax.axhline(80)
#ax.set_xticks(xtk)
#plt.savefig("%sSLAB-PIC_NHFLX_EOFs%i_%s_ModevCumuVariance_bymon.png"%(outpath,N_modeplot,bboxtext),dpi=150)

#%% Do some analysis/visualization


# Plot Mode vs. % Variance Exp (for each Month)
xtk = np.arange(1,N_modeplot+1,1)
fig,ax = plt.subplots(1,1)
for m in range(12):
    plt.plot(xtk,varexpall[:N_modeplot,m]*100,label="Month %i"% (m+1),marker="o")
ax.legend()
ax.set_ylabel("% Variance Explained")
ax.set_xlabel("Mode")
ax.set_title("NHFLX EOFs, Perc. Variance Expl. vs. Mode \n %s"% bboxstr)
ax.grid(True,ls='dotted')
ax.set_xticks(xtk)
plt.savefig("%s%s_NHFLX_EOFs_%s_ModevVariance_bymon.png"%(outpath,mcname,bboxtext),dpi=150)


# Same as above, but cumulative plot
#cvarallplot = np.zeros(varexpall.shape)
#for i in range(N_modeplot):
#    cvarallplot[i,:] = varexpall[:i+1,:].sum(0)

#%% Plot Month vs. Total % Variance Exp for 10 EOFs)
N_modeplot=5
fig,ax = plt.subplots(1,1)
ax.bar(mons3,varexpall[:N_modeplot,:].sum(0)*100,color='cornflowerblue',alpha=0.7)
ax.set_title("Total Percent Variance Explained by first %i EOFs" % N_modeplot)
ax.set_ylabel("% Variance Explained")
ax.set_ylim([0,100])
ax.set_yticks(np.arange(0,110,10))
ax.grid(True,ls='dotted')
plt.savefig("%s%s_NHFLX_EOFs_%s_TotalVariance_First%iEOFs_bymon.png"%(outpath,mcname,bboxtext,N_modeplot),dpi=150)

#%% Plot Net Heat Flux EOF Patterns

vlim = [-30,30]
#slplevels = np.arange()
blabels=[1,0,0,1]
N_modeplot = 5
slp_int = np.arange(-400,450,50)
slp_lab = np.arange(-400,500,100)

bboxplot = bbox.copy()
for i in range(2):
    if bbox[i] > 180:
        bboxplot[i] -= 360

for n in tqdm(range(N_modeplot)):
    plotord = np.roll(np.arange(0,12,1),1)
    fig,axs = plt.subplots(4,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,12))
    for p,m in enumerate(plotord):
        ax = axs.flatten()[p] # Plot Index
        ax.set_title("%s (%.1f" % (mons3[m],varexpall[n,m]*100) + "%)",fontsize=10)
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabels)
        
        # Plot NHFLX
        pcm = ax.pcolormesh(lon,lat,eofall[:,:,m,n].T,vmin=vlim[0],vmax=vlim[-1],cmap="RdBu_r")
        
        # Plot SLP and Labels
        cl = ax.contour(lon,lat,eofslp[:,:,m,n].T,levels=slp_int,colors='k',linewidths=1)
        ax.clabel(cl,levels=slp_lab,fontsize=10)
        
        # Scrap
        #pcm = ax.contourf(lon,lat,eofslp[:,:,m,n].T,levels=slp_int,cmap=cmocean.cm.balance,linewidths=1.)
        #fig.colorbar(pcm,ax=ax,fraction=0.035)
    
    fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.85,pad=0.05,anchor=(1.75,0.7))
    plt.suptitle("NHFLX EOF %i (CESM1-SLAB) (W/$m^2$ per $\sigma_{PC}$) \n SLP Contour Interval: 50 mb" % (n+1),fontsize=14)
    fig.subplots_adjust(top=0.90)
    plt.savefig("%s%s_NHFLX_EOFs_%s_EOF%iPattern_bymon.png"%(outpath,mcname,bboxtext,n+1),dpi=150)

#%% Seasonally Averaged Plots


sid = [[11,0,1],[2,3,4],[5,6,7],[8,9,10]]
snm = ["DJF","MAM","JJA","SON"]
#blabels=[0,]

n = 0
fig,axs = plt.subplots(3,4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(14,8))
for n in range(3):
    for p in range(4):
        ks = sid[p]
        ax = axs[n,p]
        ax.set_title("EOF %i: %s (%.1f" % (n+1,snm[p],varexpall[n,ks].mean()*100) + "%)",fontsize=10)
        if p == 0:
            blabels=[1,0,0,0]
        elif n == 2:
            blabels=[0,0,0,1]
        elif (n == 2) and (p ==0):
            blabels=[1,0,0,1]
        else:
            blabels=[0,0,0,0]
        ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabels)
        
        # Plot NHFLX
        pcm = ax.pcolormesh(lon,lat,eofall[:,:,ks,n].mean(2).T,vmin=vlim[0],vmax=vlim[-1],cmap="RdBu_r")
        
        # Plot SLP and Labels
        cl = ax.contour(lon,lat,eofslp[:,:,ks,n].mean(2).T,levels=slp_int,colors='k',linewidths=1)
        ax.clabel(cl,levels=slp_lab,fontsize=10)
plt.savefig("%s%s_NHFLX_EOFs1-3_EOFPattern_%s_seasavg.png"%(outpath,mcname,bboxtext),dpi=150)




#%% Plot PCs and Spectra

dt  = 365*24*3600
pct = 0.10
nsmooth = 50

tsin = []
for n in range(N_mode):
    for m in range(12):
        tsin.append(pcall[n,m,:])
        
freqs = []
specs = []

specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(tsin,nsmooth,pct,opt=1,dt=dt,clvl=[.95])

#%% Calculate contribution of each EOF to AMV

# Regional Analysis Settings
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")        # Region Names
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')

rids = [0,1,3]

aavg_eof = np.zeros((N_mode,12,len(rids)))*np.nan
for r in range(len(rids)):
    rid = rids[r]
    rbbox = bboxes[rid]
    for m in range(12):
        ineof = eofall[:,:,m,:]
        
        eofaa = proc.sel_region(ineof,lon,lat,rbbox,reg_avg=1,awgt=1)
        aavg_eof[:,m,r] = eofaa.copy()
        




step = 1
ylm = [-30,30]
xlm = [0,20]
xtk = np.arange(xlm[0],xlm[-1]+step,step)

m = 0
for m in range(12):
    fig,ax = plt.subplots(1,1,figsize=(6,3))
    for r in range(len(rids)):
        ax.plot(np.arange(1,N_mode+1),aavg_eof[:,m,r],label="%s AMV Index"%(regions[r]),marker="o")
    ax.legend()
    ax.set_xlim(xlm)
    ax.set_ylim(ylm)
    ax.set_xticks(xtk)
    ax.grid(True)
    
    ax.set_ylabel("EOF Area Average (W/m2/$\sigma_{PC}$)")
    ax.set_xlabel("Mode")
    ax.set_title("%s, Area Average of EOF)"% (mons3[m]))
    plt.savefig("%sEOF_NHFLX_Area_Avg_Mon%02d.png"%(outpath,m+1),dpi=200)
    

    



