#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and Visualize AMV
from CESM PiC Runs

- Postprocesses output the same was as stochastic model output
- Load in ERSST and HadISST, and compare AMV PAtterns

Created on Mon May 24 22:55:19 2021


@author: gliu
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import linalg,stats,signal
from scipy.signal import butter,filtfilt
from scipy.io import loadmat
import sys
import cartopy.crs as ccrs

import cmocean

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm
import yo_box as ybx

#%% User Edits

# Path to data 
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20231127/'
proc.makedir(outpath)
datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"


bbox = [-80,0,10,65]
#bboxplot = 
runmean=True
ensorem = False # Set to True to use ENSO-removed data

# Use separate landice mask for each
limasks = (datpath+"CESM-FULL_landicemask360.npy",
           datpath+"CESM-SLAB_landicemask360.npy"
           )
#% ----------------------
#%% Load PiC Data
#% ----------------------
st = time.time()
# Load full sst data from model # [time x lat x lon]
if ensorem: # Load full field with ENSO removed
    ld  = np.load(datpath+"FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
    sstfull = ld['TS']
    ld2 = np.load(datpath+"SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
    sstslab = ld2['TS']
    remove_anom=True
    
else: # Load anomalies without ENSO removal (~82 sec)
    ssts     = []
    mconfigs = ["FULL","SLAB"]
    for i in range(2):
        ds = xr.open_dataset(datpath+"CESM_proc/"+"TS_anom_PIC_%s.nc"%(mconfigs[i]))
        sst = ds.TS.values
        ssts.append(sst)
    sstfull,sstslab = ssts
    remove_anom=False
    
# Load lat/lon
lat    = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LAT'].squeeze()
lon360 = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()

print("Loaded PiC Data in %.2fs"%(time.time()-st))
# --------------
#%% Preprocessing
# --------------
st = time.time()

def preproc_CESMPIC(sst,remove_anom=True,limask=None):
    # Apply Land/Ice Mask
    if limask is None:
        mask = np.load(datpath+"landicemask_enssum.npy")
    else:
        mask = np.load(limask)
    sst = sst * mask[None,:,:]
    
    # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
    sst = sst.transpose(2,1,0)
    
    # Flip longitude
    st = time.time()
    lon180,sst = proc.lon360to180(lon360,sst)
    print("Flipped Longitude in %.2fs"%(time.time()-st))
    
    # Remove monthly anomalies
    st = time.time()
    if remove_anom:
        nlon,nlat,ntime = sst.shape
        sst = sst.reshape(nlon,nlat,int(ntime/12),12)
        ssta = sst - sst.mean(2)[:,:,None,:]
        print("Deseasoned in %.2fs"%(time.time()-st))
        print("Mean was %e" % (np.nanmax(ssta.mean(2))))
        ssta = ssta.reshape(nlon,nlat,int(ntime/12)*12)
    else:
        ssta = sst
    return ssta,lon180

# Preprocess (Apply Land/ice Mask, Adjust Dimensions, Remove Anomalies)
sstas = []
for sst in [sstfull,sstslab]:
    ssta,lon180 = preproc_CESMPIC(sst,remove_anom=remove_anom,limask=limasks[i])
    sstas.append(ssta)

sstfulla,sstslaba = sstas
nlon,nlat,ntimef  = sstfulla.shape
_,_,ntimes        = sstslaba.shape
print("preprocessed PiC Data in %.2fs"%(time.time()-st))

# # # Apply Land/Ice Mask
# # mask = np.load(datpath+"landicemask_enssum.npy")
# # sstfull = sstfull * mask[None,:,:]
# # sstslab = sstslab * mask[None,:,:]

# # # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
# # sstfull = sstfull.transpose(2,1,0)
# # sstslab = sstslab.transpose(2,1,0)

# # Flip longitude
# st = time.time()
# lon180,sstfull = proc.lon360to180(lon360,sstfull)
# _,sstslab = proc.lon360to180(lon360,sstslab)
# print("Flipped Longitude in %.2fs"%(time.time()-st))

# # Remove monthly anomalies
# st = time.time()
# nlon,nlat,ntimef = sstfull.shape
# _,_,ntimes = sstslab.shape
# sstslab = sstslab.reshape(nlon,nlat,int(ntimes/12),12)
# sstfull = sstfull.reshape(nlon,nlat,int(ntimef/12),12)
# sstslaba = sstslab - sstslab.mean(2)[:,:,None,:]
# sstfulla = sstfull - sstfull.mean(2)[:,:,None,:]
# print("Deseasoned in %.2fs"%(time.time()-st))

# #
# #proc.sel_region(sstfull,lon360,

#%% Save the data above as netcdf files


tfull = xr.cftime_range(start="0400-01-01",periods=sstfull.shape[0],freq="MS")
tslab = xr.cftime_range(start="0200-01-01",periods=sstslab.shape[0],freq="MS")

ts = [tfull,tslab]
for s in tqdm(range(2)):
    
    # Create dimensions and attribute dict
    dims = {'lon':lon180,
            'lat':lat,
            'time':ts[s],
            }
    attr_dict = {'ocean_model_config':mconfigs[s]}
    
    # Make DataArray
    da = xr.DataArray(sstas[s],
                dims=dims,
                coords=dims,
                name = 'SST',
                attrs=attr_dict
                )
    
    # Make encoding dict and export
    encoding_dict = {"SST" : {'zlib': True}} 
    savename = "%sCESM1_%s_postprocessed_NAtl.nc" % (datpath,mconfigs[s])
    print("Saving to %s" % savename)
    da.to_netcdf(savename,
             encoding=encoding_dict)

# --------------------------------------------------------------
#%% Postprocess in manner similar to the stochastic model output
# --------------------------------------------------------------
# The chunk below was take from scm.postprocess_stochoutput()
# Added on 7/27/2021
# currently written with paths on local device (not stormtrack)

#% ---- Inputs
expid       = "CESM1-PIC_ensorem%i" % (ensorem)
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath1     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath1 + '/proc/'

#%% Set preloaded inputs, # of lags
preload = [lon180,lat,sstas]
lags    = np.arange(0,37,1)

# 
scm.postprocess_stochoutput(expid,datpath1,rawpath,outpathdat,lags,preload=preload,mask_pacific=True)



#%% Do the same thing for CESM1-LE


# Load in the CESM1-LE Data
sstle,lonh,lath = scm.load_cesm_le(preprocess=True) # [lat x llon x time x ensemble]

# Set up preload for sm style postprocessing
sstle_all = []
for e in range(sstle.shape[-1]):
    sstin = sstle[...,e].transpose(1,0,2) # Flip lon and lat
    sstle_all.append(sstin)

# Postprocess it
preload = [lonh,lath,sstle_all]
lags    = np.arange(0,37,1)
ensorem     = 0
expid       = "CESM1-LE_ensorem%i" % (ensorem)
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath1     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath1 + '/proc/'
scm.postprocess_stochoutput(expid,datpath1,rawpath,outpathdat,lags,preload=preload,mask_pacific=True)

#%% Additionally Calculate Seasonal AMV Patterns

snames = ("Ann","DJF","MAM","JJA","SON")
sids  = (np.arange(0,12,1),[11,0,1],[2,3,4],[5,6,7],[8,9,10])

samvpats = np.zeros((5,2,288,192))*np.nan
samvids  = []
for i in tqdm(range(5)):
    
    sname = snames[i]
    sid   = sids[i]
    
    #amvpats  = []
    amvids   = []
    dropedge=5
    for m in range(2):
        amvid,amvpat = proc.calc_AMVquick(sstas[m],lon180,lat,bbox,dropedge=5,monid=sid)
        #amvpats.append(samvpats)
        samvpats[i,m,:,:] = amvpat
        amvids.append(amvid)
    #samvpats.append(amvpats)
    samvids.append(amvids)
    
    
#%% Quick viz
clvl = np.arange(-.5,.525,.025)

fig,axs = plt.subplots(2,2,subplot_kw={'projection' : ccrs.PlateCarree()})
for i in range(4):
    
    ax = axs.flatten()[i]
    ax = viz.add_coast_grid(ax,bbox=bbox)
    ax.set_title(snames[i+1])
    pcm = ax.contourf(lon180,lat,samvpats[i+1,0,:,:].T,cmap=cmocean.cm.balance,levels=clvl)

#%% Save Seasonal Patterns for plotting later
savename = "%s%s_SeasonalAMVPattern.npy" % (outpathdat,expid)
print(savename)
np.save(savename,samvpats)


savename = "%s%s_SeasonalAMV.npz" % (outpathdat,expid)
print(savename)
np.savez(savename,**{
    'samvpats' : samvpats,
    'samvids'  : samvids,
    'lon':lon180,
    'lat':lat,
    'snames':snames,
    'mnames':["CESM-FULL","CESM-SLAB"]
    },allow_pickle=True)

#%% Load data preprocessed above

projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath1     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath1 + '/proc/'

# Load data for CESM1-PIC
cesmacs= []
expid      = "CESM1-PIC_ensorem%i" % (ensorem)
rsst_fn    = "%sAMV_Region_%s.npz" % (outpathdat,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)

cesmpat    = ldc['amvpat_region'].item()[4] # Just take North Atlantic
cesmidx    = ldc['amvidx_region'].item()[4] # Just take North Atlantic

# Load global lat/lon
clon,clat  = scm.load_latlon(rawpath)

#%% Load CESM-Historical data as well
cesmacs_le = []
expid      = "CESM1-LE_ensorem%i" % (ensorem)
rsst_fn    = "%sAMV_Region_%s.npz" % (outpathdat,expid)
ldc_le     = np.load(rsst_fn,allow_pickle=True)

# Load the CESM Patterns
cesmpat_le    = ldc_le['amvpat_region'].item()[4] # Just take North Atlantic
cesmidx_le    = ldc_le['amvidx_region'].item()[4] # Just take North Atlantic

# Load global lat/lon
clonh,clath  = scm.load_latlon(rawpath)

# ------------------------------------------
# %% Load and postprocess HadISST and ERSST
# ------------------------------------------

load_limopt = True
manual_calc = False


if load_limopt:
    nmon = 3
    
    # Load in LIM-opt dataset
    ssts,lons,lats,times = scm.load_limopt_sst()
    
    lons.append(clon)
    lons.append(clon)
    lats.append(clat)
    lats.append(clat)
    #ssts.append()
    for s in range(len(ssts)):
        ssts[s] = ssts[s][:,:,4:-4]
    
else:
    nmon = 12
    
    # Load in the Datasets **********************
    # Already detrended.HadISST
    st = time.time()
    datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
    hsst,hlat,hlon = scm.load_hadisst(datpath2,method=2,startyr=1900,grabpoint=None)
    print("Completed in %.2fs" % (time.time()-st))
    
    # Load the masks
    datpath3 = datpath2 + "lim-opt/"
    dsm = xr.open_dataset(datpath3+"HadISST.MSK.nc")
    hmsk = dsm.MSK.values
    hmsk[hmsk==0] = np.nan
    hsst *= hmsk.T[:,:,None]
    
    # Load ERSST
    st = time.time()
    esst,elat,elon = scm.load_ersst(datpath2,method=2,startyr=1900,grabpoint=None)
    elon360 = elon.copy()
    elon,esst = proc.lon360to180(elon,esst)
    print("Completed in %.2fs" % (time.time()-st))
    
    # Load the masks
    dsm = xr.open_dataset(datpath3+"ERSST.MSK.nc")
    emsk = dsm.MSK.values
    _,emsk = proc.lon360to180(elon360,emsk.T)
    emsk[emsk==0] = np.nan
    esst *= emsk[:,:,None]
    
    # Set up lists **********************
    lons = [hlon,elon,clon,clon]
    lats = [hlat,elat,clat,clat]
    ssts = [hsst,esst]

# Calculate AMV (North Atlantic)
if load_limopt and ~manual_calc: # Load pre-computed AMV Index
    opats,oidxs,lons,lats,_ = scm.load_limopt_amv()
    
    lons.append(clon)
    lons.append(clon)
    lats.append(clat)
    lats.append(clat)

else:
    oidxs = []
    opats = []
    for s in range(len(ssts)):
        amvidx,amvpattern = proc.calc_AMVquick(ssts[s][:,:,:],lons[s],lats[s],bbox,order=5,cutofftime=10,
                                               anndata=False,runmean=False,dropedge=5,nmon=nmon)
        oidxs.append(amvidx)
        opats.append(amvpattern)



#%% Set up arrays for plotting

if load_limopt:
    ossts = ssts
    amvpats = opats
    amvids  = oidxs
    for i in range(2):
        amvpats.append(cesmpat[i])
        amvids.append(cesmidx[i])
    mnames  = ["COBE","HadISST","ERSST","CESM-FULL","CESM-SLAB"]
    
else:
    ossts    = [hsst,esst]
    amvpats = [opats[0],opats[1],cesmpat[0],cesmpat[1]]
    amvids  = [oidxs[0],oidxs[1],cesmidx[0],cesmidx[1]]
    mnames  = ["HadISST","ERSST","CESM-FULL","CESM-SLAB"]


# Plotting Parameters
clim = .5 #0.025
cstp = 0.025
cmult = 2
cint = np.arange(-clim,clim+cstp,cstp)
cl_int = np.arange(-clim,clim+cstp*cmult,cstp*cmult)
bboxplot = [-80,0 ,0,55]
# ---------------------------------------------
#%% *** PLOT AMV PATTERNS (Obs. vs. CESM!!) ***
# ---------------------------------------------



square=True

if load_limopt:
    fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,8))
    for i in range(5):
        ax = axs.flatten()[i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
        cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
        ax.clabel(cl)
        ax.set_title("%s ($\sigma_{AMV}^2$=%.4f$\degree \, C^{2}$)"%(mnames[i],np.var(amvids[i])))
    
    cb = fig.colorbar(cf,ax=axs.flatten(),orientation="horizontal",fraction=0.04,pad=0.05)
    plt.suptitle("AMV Pattern (CESM vs. Obs)",y=0.9)   
    cb.set_label("AMV Pattern for SST; Contour Interval=%.3f ($\degree C \sigma_{AMV}^{-1}$)"%cstp)
    plt.savefig("%sAMV_Patterns_ObsLIMopt_v_CESM.png"% (outpath),dpi=200,bbox_inches='tight')
    
else:
    if square:
        fig,axs = plt.subplots(2,2,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,8))
    else:
        fig,axs = plt.subplots(1,4,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,6))
    for i in range(4):
        ax = axs.flatten()[i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
        cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
        ax.clabel(cl)
        ax.set_title("%s ($\sigma_{AMV}^2$=%.4f$\degree \, C^{2}$)"%(mnames[i],np.var(amvids[i])))
    if square:
        cb = fig.colorbar(cf,ax=axs.flatten(),orientation="horizontal",fraction=0.04,pad=0.05)
        plt.suptitle("AMV Pattern (CESM vs. Obs)",y=0.9)
    else:
        cb = fig.colorbar(cf,ax=axs.flatten(),orientation="vertical",fraction=0.010,pad=0.05)
        plt.suptitle("AMV Pattern (CESM vs. Obs)",y=0.9)
    cb.set_label("AMV Pattern for SST; Contour Interval=%.3f ($\degree C \sigma_{AMV}^{-1}$)"%cstp)
    
    
    plt.savefig("%sAMV_Patterns_Obs_v_CESM.png"% (outpath),dpi=200,bbox_inches='tight')

#amvids = [oid]
#ax = viz.add_coast_grid(ax,bbox=bboxplot)

#%% Try centering an odd number of plots


import matplotlib.gridspec as gridspec




def init_2rowodd(ncol,proj,figsize=(6,6),oddtop=False,debug=False):
    """
    Initialize a 2-row subplot where
    the bottom row has the smaller number of plots
    source: https://www.tutorialguruji.com/python/matplotlib-allign-uneven-number-of-subplots/

    Parameters
    ----------
    ncol : INT
        Number of columns (even). Last row will contain ncol-1 subplots

    proj : Cartopy Projection
        Projection to set the subplots as
        
    figsize : INT (Length x Height)
        Figure Size
        
    oddtop : BOOL
        Set to True to make odd row on top

    Returns
    -------
    axs : LIST of matplotlib axes
        Flattened list containing subplots

    """
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2,ncol*2)
    
    nodd = ncol*2-1
    
    axs = []
    for i in range(ncol*2-1):
        
        
        if oddtop: # Shorter row on top
            if i < ncol-1: 
                rowid   = 0     # Top row
                startid = i*2+1 # Start on 1
                stopid  = i*2+3 # Stop 2 subplots later
                msg = "for %i <= %i --> gs[0,%i:%i]" % (i,ncol-1,startid,stopid)
            else:
                rowid   = 1              # Bot Row
                startid = 2*(i-ncol)+2   # Start from 0 (+2 since i-ncol = -2)
                stopid  = 2*(i-ncol)+4   # End 2 plots later
                msg = "for %i > %i --> gs[1,%i:%i]" % (i,ncol,startid,stopid)
        else: # Shorter row on bottom
            if i < ncol:
                rowid = 0
                startid = 2 * i
                stopid  = 2 * i + 2
                msg = "for %i < %i --> gs[0,%i:%i]" % (i,ncol,startid,stopid)
            else:
                rowid = 1
                startid = 2 * i - nodd
                stopid  = 2 * i + 2 - nodd
                msg = "for %i >= %i --> gs[1,%i:%i]" % (i,ncol,startid,stopid)
        
        ax = plt.subplot(gs[rowid,startid:stopid],projection=proj)
        
        if debug:
            
            print(msg)
            
        axs.append(ax)
    return fig,axs

ncol    = 3
proj    = ccrs.PlateCarree()
figsize = (12,7.5)
fig,axs = init_2rowodd(ncol,proj,figsize=figsize,oddtop=False,debug=True)
for i in range(len(axs)):
    ax = axs[i]
    
    if i == 0:
        blb = [1,0,0,1]
    elif i == 3:
        blb = [1,0,0,1]
    else:
        blb = [0,0,0,1]
        
        
    
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blb)
    
    cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
    cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
    
    ax.clabel(cl)
    ax.set_title("%s ($\sigma_{AMV}^2$=%.4f$\degree \, C^{2}$)"%(mnames[i],np.var(amvids[i])))

cb = fig.colorbar(cf,ax=axs,orientation="horizontal",fraction=0.04,pad=0.05)
plt.suptitle("AMV SST Pattern (CESM vs. Obs)",y=0.95)   
cb.set_label("SST; Contour Interval=%.3f ($\degree C \sigma_{AMV}^{-1}$)"%cstp)
plt.savefig("%sAMV_Patterns_ObsLIMopt_v_CESM.png"% (outpath),dpi=200,bbox_inches='tight')

        
        


#%% Plot CESM-AMV Alone (for SM_Paper_Outline)
i = 2

# Prepare Tick labels
cb_lab = ["%.2f"%(n) for n in cl_int]

# Initialize Figure
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(6,4))
ax = viz.add_coast_grid(ax,bbox=bboxplot)

# Plot
cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
ax.clabel(cl)
ax.set_title("%s AMV Pattern" %(mnames[i]))

# Add Colorbar, Reduce tick labels
cb = fig.colorbar(cf,ax=ax,orientation="horizontal",fraction=0.050,pad=0.1)
cb.set_label("SST ($\degree C \sigma_{AMV}^{-1}$); Contour Interval=%.3f "%cstp)
cb.set_ticks(cl_int[::2],cb_lab[::2])

# Add a marker, save figure
ax.scatter(-30,50,200,marker="*",facecolor='yellow',zorder=9,edgecolor='k',linewidth=.5)
plt.savefig("%sAMV_Patterns_Indv_%s.png"% (outpath,mnames[i]),dpi=200,bbox_inches='tight')

#%% Calculate North Atlantic SST Spectra

awgt = 1
monsec = 3600*24*30 # ~Seconds in Month

fullsmth=75
slabsmth=65
obssmth=12


# Load regionally averaged SST
rsst_fn    = "%sSST_RegionAvg_%s.npy" % (outpathdat,expid)
ldc        = np.load(rsst_fn,allow_pickle=True).item()
rssts      = ldc[4] # Get NNAT index

bbox_NA_new = [-80,0,10,65]

# Calculate regionall averaged SST for reanalysis
nassti = []
for i in range(len(ssts)):
    aa_sst = proc.area_avg(ossts[i],bbox_NA_new,lons[i],lats[i],awgt)
    nassti.append(aa_sst)

# Calculate the spectra
if load_limopt:
    nasstis = [nassti[0],nassti[1],nassti[2],rssts[0],rssts[1]]
    nsmooths = [obssmth,obssmth,obssmth,fullsmth,slabsmth,]
    mcols = ["r","b","cyan","k","gray"]
    
    dts    = [monsec*3,monsec*3,monsec*3,monsec,monsec] # LIM-OPT
    
    mmark = ["o","d","*","x","+"]

else:
    nasstis = [nassti[0],nassti[1],rssts[0],rssts[1]]
    nsmooths = [obssmth,obssmth,fullsmth,slabsmth,]
    mcols = ["r","b","k","gray"]
    mmark = ["o","d","*","x"]
    dts = np.ones(len(nasstis))*monsec
    
pct = 0.10
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(nasstis,nsmooths,pct,dt=dts)



smoothname = "smth-obs%03i-full%02i-slab%02i" % (obssmth,fullsmth,slabsmth)
#%% Plot the North Atlantic Spectra

# Older Params
xlm = [1e-2,5e0]
xper = np.array([200,100,50,25,10,5,1,0.5]) # number of years
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
xlm = [1e-2,5e0]

# Newer Params
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0,0.3]

title = ("North Atlantic SST Spectra \n Smoothing (# bands): Reanalysis (%i), CESM-FULL (%i), CESM-SLAB (%i)" %  (obssmth,fullsmth,slabsmth))

speclabels = ["%s (%.4f$\degree \, C^{2}$)" % (mnames[i],np.var(nasstis[i])) for i in range(len(mnames))]
fig,ax = plt.subplots(1,1,figsize=(12,4))
ax,ax2 = viz.plot_freqlin(specs,freqs,speclabels,mcols,
                     ax=ax,plottitle=title,xtick=xtks,xlm=xlm,marker="",return_ax2=True,lw=4)

ax2.set_xlabel("Period (Years)")
plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)

ax.set_ylim(ylm)
#plt.suptitle("Regional AMV Index Spectra (unsmoothed, Forcing=%s)"%(frcnamelong[f]))
savename = "%sNASST_Spectra_Obs-v-CESM_LinearLinear_%s_Limopt%i.png" % (outpath,smoothname,load_limopt)
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% Calculate and plot autocorrelation

kmonth = 1
lags   = np.arange(0,37,1)
xtk2   = np.arange(0,38,2)
for i in range(len(nasstis)):
    nasstis[i] = signal.detrend(nasstis[i],type='linear')

# Calculate Autocorrelation
acs,confs    = scm.calc_autocorr(nasstis,lags,kmonth+1,calc_conf=True)

# Plot the autocorrelation
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="North Atlantic SST Autocorrelation")
for i in range(len(acs)):
    
    ax.fill_between(lags,confs[i][:,0],confs[i][:,1],color=mcols[i],alpha=0.15,zorder=-1)
    ax.plot(lags,acs[i],label=mnames[i],color=mcols[i],marker=mmark[i])
    

ax.legend(ncol=2)
plt.savefig("%sNASST_autocorr_Obs-v-CESM.png"%outpath,dpi=150)

#%% Wait... let's check the seasonal cycle

fig,axs = plt.subplots(4,1)

for i in range(len(nasstis)):
    ax = axs.flatten()[i]
    sstyrmon = proc.year2mon(nasstis[i]) # [mon x year]
    ax.plot(sstyrmon.mean(1))
    ax.set_title(mnames[i])


#%% Comparison Plot for SM Draft 2


fig = plt.figure(constrained_layout=True, facecolor='w',figsize=(12,8))

gs = fig.add_gridspec(nrows=3, ncols=3, left=.02, right=1,
                      hspace=.075, wspace=0.25)


# Create Top Row with AMV plots
axs = []
for i in range(3):
    
    blabel = [0,0,0,1]
    if i == 0:
        blabel[0] = 1
    
    
    ax  = fig.add_subplot(gs[0:2, i],projection=ccrs.PlateCarree())
    ax  = viz.add_coast_grid(ax,bbox=bbox,line_color='k',
                             ignore_error=True,fill_color='gray',
                             blabels=blabel)
    
    cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
    cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
    
    ax.clabel(cl)
    ax.set_title("%s ($\sigma_{AMV}^2$=%.4f$\degree \, C^{2}$)"%(mnames[i],np.var(amvids[i])))

    axs.append(ax)

ax3 = fig.add_subplot(gs[2, :])

ax3,ax2 = viz.plot_freqlin(specs,freqs,speclabels,mcols,
                     ax=ax3,plottitle=title,xtick=xtks,xlm=xlm,return_ax2=True,lw=4)
plt.tight_layout()

#%%  Just Plot the AMV Patterns (SM Draft 2)


fig,axs = plt.subplots(1,3,constrained_layout=True,figsize=(10,4),
                       subplot_kw={'projection':ccrs.PlateCarree()})

spid = 0
for i in range(3):
    
    blb = [0,0,0,1]
    if i == 0:
        blb[0] = 1
        
    ax = axs.flatten()[i]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blb,ignore_error=True,
                            fill_color='gray')
    
    cf = ax.contourf(lons[i],lats[i],amvpats[i].T,cmap=cmocean.cm.balance,levels=cint,extend='both')    
    cl = ax.contour(lons[i],lats[i],amvpats[i].T,levels=cl_int,colors='k',linewidths=0.5)
    
    ax.clabel(cl)
    ax.set_title("%s ($\sigma_{AMV}^2$=%.4f$\degree \, C^{2}$)"%(mnames[i],np.var(amvids[i])))
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
    spid += 1
    
cb = fig.colorbar(cf,ax=axs,orientation="horizontal",fraction=0.08,pad=0.05)
#plt.suptitle("AMV SST Pattern (CESM vs. Obs)",y=0.95)   
cb.set_label("SST ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Patterns_Obs.png"% (outpath),dpi=200,bbox_inches='tight')



#%% Old Scripts **************************************************************
# ****************************************************************************
# 
# -----------------
#%% Calculate AMV
# -----------------

# Calculate AMV Index
idxs = []
pats = []
for sst in sstas:
    
    amvidx,amvpattern=proc.calc_AMVquick(sst,lon180,lat,bbox,order=5,cutofftime=10,anndata=False,runmean=runmean)
    idxs.append(amvidx)
    pats.append(amvpattern)
    
# -----------------
#%% Plot AMV Patterns
# -----------------

# Plot AMV Patterns
def plot_AMV_generals(lat,lon,amvpattern,vscale=1):
    """
    Customized AMV Plot for Generals Presentation (for consistent plotting)
    
    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    lon : TYPE
        DESCRIPTION.
    amvpattern : [lon x alt]
        DESCRIPTION.
    vscale : INT
        Amt of times to scale AMV pattern by
    Returns
    -------
    None.
    """
    bbox = [-80,0 ,0,65]
    
    
    # Set up plot params
    plt.style.use('default')
    cmap = cmocean.cm.balance
    # Original Generals Values
    cints = np.arange(-.55,.60,.05)
    cintslb = np.arange(-.50,.6,.1)
    # 
    

    
    # Make the plot
    fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    ax,cb  = viz.plot_AMV_spatial(amvpattern.T*vscale,lon,lat,bbox,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True,
                                 fontsize=8)
    cb.set_ticks(cintslb)
    return fig,ax,cb

fig,ax,cb = plot_AMV_generals(lat,lon180,pats[0])
ax.set_title("AMV Pattern (CESM1-FULL; 400 to 2200) \n Contour Interval: 0.03 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"CESM1-FULL_AMV_Spatial_Pattern.png",bbox_inches='tight')

fig,ax,cb = plot_AMV_generals(lat,lon180,pats[-1])
ax.set_title("AMV Pattern (CESM1-SLAB; 101 to 1001 ) \n Contour Interval: 0.03 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"CESM1-SLAB_AMV_Spatial_Pattern.png",bbox_inches='tight')

#%% Plot AMV but in a different style (7/26/2021 AMV Project Meeting)

#cint = np.arange(-.30,.33,.03)
#cl_int = np.arange(-.30,.4,.1)
cint   = np.arange(-0.45,0.50,0.05) # Used this for 8/10/2021 Meeting
cl_int = np.arange(-0.45,0.50,0.05)
bboxplot = [-100,20,0,80]
lon = lon180
modelnames = ["CESM-FULL","CESM-SLAB"]
figpath = outpath

amvpat = pats

cmap1 = cmocean.cm.balance
cmap1.set_bad(color="w")
for p in range(len(amvpat)):
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm2 = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmap1)
    pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
    
    cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fmt="%.2f",fontsize=8)
    
    #ax.set_title(modelnames[p])
    #fig.colorbar(pcm,ax=ax,fraction=0.036)
    ax.set_title(modelnames[p] + " AMV Pattern ($\circ C$ per $\sigma_{AMV}$)")
    fig.colorbar(pcm,ax=ax,orientation='horizontal',shrink=0.75)#,pad=0.015)
    plt.savefig("%sAMV_Pattern_regionNAT_model%s.png"%(figpath,modelnames[p]),dpi=200,bbox_tight='inches')

#
# %% Calculate NASSTI
#
bbox_mid = [-80,0,40,60]

awgt = 1
nassti = []
enassti = []
for sst in sstas:
    
    aa_sst = proc.area_avg(sst,bbox,lon180,lat,awgt)
    nassti.append(aa_sst)

    aa_sst2 = proc.area_avg(sst,bbox_mid,lon180,lat,awgt)
    enassti.append(aa_sst2)
#
# Save NASSTI
#   
fn = datpath + "CESM-PIC_NASSTI.npz"
np.savez(fn,**{
         'nassti_slab': nassti[-1],
         'nassti_full': nassti[0],
         'enassti_slab': enassti[-1],
         'enassti_full': enassti[0]
         }
        )


#%% Calculate Index over AMV-mid (Garuba et al. 2018 comparison)
# 7/26/2021 meeting notes


#
bbox_mid = [-80,0,40,60]
# Calculate AMV Index
idxs_mid = []
pats_mid = []
for sst in sstas:
    
    amvid_mid,amvpat_mid=proc.calc_AMVquick(sst,lon180,lat,bbox_mid,order=5,cutofftime=10,anndata=False,runmean=runmean)

    idxs_mid.append(amvid_mid)
    pats_mid.append(amvpat_mid)

# Do spectral analysis




# ---------------------------
#%% Do some spectral analysis
# ---------------------------

# amvid = []
# for k in amvidx.keys():
#     amvid.append(amvidx[k])
# amvid = np.array(amvid)
    

enumvar = enassti # idxs


# # -------------------------------------------
# # First calculate for CESM1 (full and slab)
# # -------------------------------------------
# Key Params
plotcesm = True
cnames  = ["CESM1 FULL","CESM1 SLAB"]
nsmooths = [20,20] # Set Smothing
#nsmooths = [250,125]

timemax = None#250*12

# Other Params
pct     = 0.10
opt     = 1
dt      = 3600*24*365
tunit   = "Years"
clvl    = [0.95]
axopt   = 3
clopt   = 1

dtplot = 3600*24*365

# Calculate spectra
freq1s,P1s,CLs = [],[],[]
for i,sstin in enumerate(enumvar):
    
    # Limit to maximum time
    if timemax is None:
        sstin=sstin
    else:
        sstin = sstin[:timemax]
    
    # Calculate and Plot
    sps = ybx.yo_spec(sstin,opt,nsmooths[i],pct,debug=False)
    P,freq,dof,r1=sps
    
    # Plot if option is set
    if plotcesm:
        
        pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dtplot,clvl=clvl,axopt=axopt,clopt=clopt)
        print(r1)
        fig,ax,h,hcl,htax,hleg = pps
        #ax,htax = viz.make_axtime(ax,htax)
        #ax = viz.add_yrlines(ax)
        #ax.set_title("%s Spectral Estimate \n nsmooth=%i, taper = %.2f" % (cnames[i],nsmooths[i],pct*100) +r"%")
        #ax.grid(True,which='both',ls='dotted')
        #ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
        #plt.tight_layout()
        #plt.savefig("%sNASSTI_SpectralEstimate_%s_nsmooth%i_taper%i.png"%(outpath,cnames[i],nsmooths[i],pct*100),dpi=200)
    CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
    
    P    = P*dt
    freq = freq/dt
    CC   = CC*dt
    P1s.append(P)
    freq1s.append(freq)
    CLs.append(CC)

# Read outvariables
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s
clfull,clslab = CLs

#%% Remake the Plot(Linear)

def lin_quickformat(ax,plotdt,freq):
    # Set tickparams and clone
    xtick = np.arange(0,1.7,.2)
    ax.set_xticks(xtick)
    ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
    
    # Set xtick labels
    xtkl = ["%.1f" % (1/x) for x in xtick]
    htax.set_xticklabels(xtkl)
    
    
    # Set some key lines
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    return ax,htax


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
ax.plot(freqcesmslab*plotdt,CLs[1][:,0]/plotdt,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
ax.plot(freqcesmfull*plotdt,CLs[0][:,0]/plotdt,color='black',label="CESM1 FULL AR1",ls=':')
ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL)")
plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i.png"%(outpath,nsmooths[i],pct*100),dpi=200)

#%% Plot Linear, but over a particular range

xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
if timemax is None:
    timemax = 0

def lin_quickformat(ax,plotdt,freq):
    # Set tickparams and clone
    xtick = np.arange(0,1.7,.2)
    ax.set_xticks(xtick)
    ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
    
    # Set xtick labels
    xtkl = ["%.1f" % (1/x) for x in xtick]
    htax.set_xticklabels(xtkl)
    
    
    # Set some key lines
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    return ax,htax


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
#ax.plot(freqcesmslab*plotdt,CLs[1][:,0]/plotdt,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
#ax.plot(freqcesmfull*plotdt,CLs[0][:,0]/plotdt,color='black',label="CESM1 FULL AR1",ls=':')
#ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL) \n nsmooth=%i"%(nsmooths[0]))

ax.set_xlim(xlms)
ax.set_xticks(xtks)
xtick=np.array(xtks)
htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
# Set xtick labels
htax.set_xticklabels(xtkl)

ax.set_ylim([0,2])

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i_decadal_timemax%i.png"%(outpath,nsmooths[i],pct*100,timemax),dpi=200)



#%% Remake the plot (variance preserving)

def lin_quickformat(ax,plotdt,freq):
    # Set tickparams and clone
    xtick = np.arange(0,1.7,.2)
    ax.set_xticks(xtick)
    ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
    
    # Set xtick labels
    xtkl = ["%.1f" % (1/x) for x in xtick]
    htax.set_xticklabels(xtkl)
    
    
    # Set some key lines
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    return ax,htax


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.semilogx(freqcesmslab*plotdt,Pcesmslab*freqcesmslab,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.semilogx(freqcesmslab*plotdt,CLs[1][:,1]*freqcesmslab,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
ax.semilogx(freqcesmslab*plotdt,CLs[1][:,0]*freqcesmslab,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.semilogx(freqcesmfull*plotdt,Pcesmfull*freqcesmfull,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.semilogx(freqcesmfull*plotdt,CLs[0][:,1]*freqcesmfull,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
ax.semilogx(freqcesmfull*plotdt,CLs[0][:,0]*freqcesmfull,color='black',label="CESM1 FULL AR1",ls=':')

# Set x limits
#xlm = [1/(plotdt*),1/(plotdt*1)]
xlm = [5e-4,10]
ax.set_xlim(xlm)
ylm = [-.01,.4]

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqcesmfull,"Years",dtplot,mode='log-lin')



# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtk]
htax.set_xticklabels(xtkl)

#ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL),nsmooth=%i"%nsmooths[0])
plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i.png"%(outpath,nsmooths[i],pct*100),dpi=200)





#%% Load in some stochastic model results, and compare

datpath2  = projpath + '01_Data/model_output/'
fscale    = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
applyfac  = 2
mconfig   = "SLAB_PIC"
runid     = "303"
funiform  = 1.5
#expid     = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)

runid   = "002"
frcname = "flxeof_5eofs_SLAB-PIC"
expid   = "forcing%s_%iyr_run%s" % (frcname,nyrs,runid) 

if "forcing" in expid:
    sst = sst = np.load(datpath2+"stoch_output_%s.npz"%(expid),allow_pickle=True)['sst'].item()
else:
    sst = np.load(datpath2+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Select NAtl Region for each model


#% Calculate AMV Index
amvtime = time.time()
amvidx = {}
amvpat = {}

for model in range(4):
    amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,bbox,order=5,cutofftime=10,anndata=False,runmean=runmean)
    
print("Calculated AMV variables for region in %.2f" % (time.time()-amvtime))


#%% Plot AMV Pattern
modeln = ["MLD Fixed","MLD Mean", "MLD Seasonal", "Entraining"]
ecolors = ['blue','r','magenta','orange']

enames = ("Constant h","Vary h","Entraining")
ecolors = ['blue','r','magenta','orange']

#enames = modeln
for model in [1,2,3]:
    
    fig,ax,cb = plot_AMV_generals(latr,lonr,amvpat[model])
    ax.set_title("AMV Pattern (Stochastic Model %s) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$" % (modeln[model]))
    plt.savefig(outpath+"Stochmod_AMV_Spatial_Pattern_model%i.png"%model,bbox_inches='tight')
    
    
#%% Calculate Spectra


nsmooths2 = np.ones(4)* 1
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(amvidx,nsmooths2,pct)

# Plot Spectra
fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(4):
    ax.plot(freqs[i]*plotdt,specs[i]/plotdt,color=ecolors[i],label=enames[i])
    
    ax.plot(freqs[i]*plotdt,CCs[i][:,1]/plotdt,color=ecolors[i],alpha=0.5,ls='dashed')
    ax.plot(freqs[i]*plotdt,CCs[i][:,0]/plotdt,color=ecolors[i],alpha=0.5,ls='dotted')

    

# Set x limits
xtick = np.arange(0,1.7,.2)
ax.set_xticks(xtick)

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick)

ax = viz.add_yrlines(ax,dt=plotdt)

#ylm = [-.01,.4]
# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtick]
htax.set_xticklabels(xtkl)
ax.legend()
ax.set_title("AMV Idx Spectral Estimates (Stochastic Model)")
plt.tight_layout()
plt.savefig(outpath+"AMVIDX_Stochastic_Model.png",dpi=200)



#%% comparitive lin-lin plot


xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
if timemax is None:
    timemax = 0

def lin_quickformat(ax,plotdt,freq):
    # Set tickparams and clone
    xtick = np.arange(0,1.7,.2)
    ax.set_xticks(xtick)
    ax.set_ylabel("Power ($\degree C^{2} / cpy$)",fontsize=12)
    ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
    htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
    
    # Set xtick labels
    xtkl = ["%.1f" % (1/x) for x in xtick]
    htax.set_xticklabels(xtkl)
    
    
    # Set some key lines
    ax = viz.add_yrlines(ax,dt=plotdt)
    
    ax.legend(fontsize=10)
    return ax,htax


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))


# Plot CESM
i = 1
ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB")
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="",ls='dashed')
ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL")
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="",ls='dashed')

ax.set_xlabel("")
#ax.set_title("CESM1 NASSTI (SLAB vs. FULL) \n nsmooth=%i"%(nsmooths[0]))

ax.set_xlim(xlms)
ax.set_xticks(xtks)
xtick=np.array(xtks)
htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
# Set xtick labels
htax.set_xticklabels(xtkl)

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i_decadal_timemax%i.png"%(outpath,nsmooths[i],pct*100,timemax),dpi=200)







