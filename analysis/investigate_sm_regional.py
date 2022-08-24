#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate Properties of Stochastic Model SST over a selected region 

Created on Wed Mar  2 16:36:04 2022
@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
import xarray as xr

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220726/"
   
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
#%% User Edits

# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over

# Visualize Continuous run 200, Fprime
fnames   = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
exname   = "Fprime_amq0_method5_cont"

# # Visualize Continuous run 200, Qnet 
# fnames =["forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run2%02d_ampq3_method5_dmp0"%i for i in range(10)]
# frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
# exname ="Qnet_amq3_method5_cont"

# Plotting Params
darkmode = False
debug    = True 

#%% Functions
def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags

#%% User Edits

# Regional Analysis Settings (OLD)
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,10,20]
bbox_NA = [-80,0 ,0,65]
bbox_NNA = [-80,0 ,10,65]
regions = ("SPG","STG","TRO","NAT","NAT")#,"NNAT")        # Region Names
regionlong = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic (10N-65N)")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NNA) # Bounding Boxes
bbcol  = ["Blue","Red","Yellow","Black","Black"]
bbsty  = ["solid","dashed","solid","dotted","dotted"]

# # Regional Analysis Setting (NEW, STG SPLOIT)
# Regional Analysis Settings
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w  = [-80,-40,20,40]
bbox_ST_e  = [-40,-10,20,40]
regions = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw")        # Region Names
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w) # Bounding Boxes
regionlong = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic","Subtropical (East)","Subtropical (West)",)
bbcol  = ["Blue","Red","Yellow","Black","Black"]
bbcol      = ["Blue","Red","Yellow","Black","Black","magenta","red"]
bbsty      = ["solid","dashed","solid","dotted","dotted","dashed","dotted"]


# AMV Pattern Contours
cint        = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int      = np.arange(-0.45,0.50,0.05)
cmax  = 0.5
cstep = 0.025
lstep = 0.05
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]
bboxplot    = [-80,0,5,60]

modelnames  = ("Constant h","Vary h","Entraining")
mcolors     = ["red","magenta","orange"]

# CESM Names
cesmname   =  ["CESM-FULL","CESM-SLAB"]
cesmcolor  =  ["k","gray"]
cesmline   =  ["dashed","dotted"]

# Autocorrelation PLots
xtk2       = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
conf  = 0.95
tails = 2

proj  = ccrs.PlateCarree()
dfcol = "k"

#%% load some additional data

# Load lat/lon regional
# Get one of the outputs
ldname = "%sstoch_output_%s.npz" % (datpath,fnames[0])
if exname == "numEOFs":
    ldname = ldname.replace("50","2")
    

ld     = np.load(ldname,allow_pickle=True)
lon    = ld['lon']
lat    = ld['lat']

#lon = np.load(datpath+"lon.npy")
#lat = np.load(datpath+"lat.npy")

# Load global lat/lon
lon180g,latg  = scm.load_latlon(rawpath)

#%% For each model read in the data

# Select a particular region
reg_sel  =[-36,-20,44,60]
"""
Other Regions
North Atlantic [-80,0,10,20]
NNA Box (from JCLI review)

"""

reg_strf = "lon%ito%i_lat%ito%i" % (reg_sel[0],reg_sel[1],reg_sel[2],reg_sel[3])

# Load in data and take annual average
sst_all = []
for f,fname in tqdm(enumerate(fnames)):
    ld = np.load(datpath+"stoch_output_%s.npz"%fname,allow_pickle=True)
    ssts = ld['sst']
    if f == 0:
        lonr   = ld['lon']
        latr   = ld['lat']
        
    # Quick Transpose
    ssts = ssts.transpose(1,2,3,0) # --> [lon,lat,year,time,model]
    
    # Select Region
    ssts_reg,lonr2,latr2 = proc.sel_region(ssts,lonr,latr,reg_sel,autoreshape=True)
    sst_all.append(ssts_reg)
sst_all = np.concatenate(sst_all,axis=2) # [lon x lat x TIME x model]
nlonr,nlatr,ntime,nmod = sst_all.shape

if debug:
    mid = 2
    fig,ax = plt.subplots(1,1,figsize=(8,2),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=reg_sel,fill_color='gray')
    pcm = ax.pcolormesh(lonr2,latr2,sst_all[...,mid].var(-1).T,cmap='bone')
    fig.colorbar(pcm,ax=ax,orientation='horizontal')
    
#%% Load in CESM Data and limit to the same region

bbox = reg_sel
mconfigs = ("FULL","SLAB")
cdatpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
sst_cesm = []
for mconfig in mconfigs:
    fname   = "%sCESM1_%s_postprocessed_NAtl.nc" % (cdatpath,mconfig)
    ds      = xr.open_dataset(fname)
    dsreg   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    sst_cesm.append(dsreg.SST.values)
    
#%% Load damping masks

dmsks = scm.load_dmasks(bbox=bbox)
dmsks.append(dmsks[-1]) # Duplicate Last dmsk

#%% Load AMV Pattern for Reference

# Load for stochastic model experiments
amvpats  = []
amvids   = []
for f in range(len(fnames)):
    
    # Load the dictionary [h-const, h-vary, entrain]
    expid = fnames[f]
    rsst_fn = "%sproc/AMV_Region_%s.npz" % (datpath,expid)
    print("Loading %s" % rsst_fn)
    ld = np.load(rsst_fn,allow_pickle=True)#.item()
    
    amvidx = ld['amvidx_region'].item()
    amvpat = ld['amvpat_region'].item()
    
    amvpats.append(amvpat)
    amvids.append(amvidx)
    
#%% Select a point, and compute the spectra vs. CESM

lonf = -30
latf = 17

# Compute Average AMV Pattern
mid_sel = 2
amvpats_sel = np.stack([amvpat[4][mid_sel] for amvpat in amvpats])
amvpats_avg = amvpats_sel.mean(0)

# Get Point Indices
klon,klat    = proc.find_latlon(lonf,latf,lonr2,latr2)
locstring    = "Lon: %.2f, Lat: %.2f" % (lonr[klonf],latr[klat])
locfstring   = "lon%i_lat%i" % (lonf,latf)

inssts     = (sst_all[klon,klat,:,0],sst_all[klon,klat,:,1],
              sst_all[klon,klat,:,2],
              sst_cesm[0][klon,klat,:],sst_cesm[1][klon,klat,:])
expnames   = np.concatenate([modelnames,cesmname])#("Stochastic Model","CESM-SLAB","CESM-FULL")
expcolors  = np.concatenate([mcolors,cesmcolor])#('orange','gray','k')

# Lets try computing the spectra
nsmooth = np.concatenate([np.ones(3)*300,[100,75,]])
pct     = 0.10
dtplot  = 3600*24*365 
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooth,pct)

freqns = [spec.shape[0] for spec in specs] # Get number of frequencies for each dimension

#%% Plot the Spectra
bboxinset = [-70,-15,10,20]

xtks = [1/100,1/50,1/25,1/10,1/5,1/2,1/1]
xper = [int(1/x) for x in xtks]
fig,ax = plt.subplots(1,1,figsize=(10,3))
for i in range(len(inssts)):
    ax.plot(freqs[i]*dtplot,specs[i]/dtplot,label=expnames[i],color=expcolors[i],lw=3)
    ax.plot(freqs[i]*dtplot,CCs[i][:,1]/dtplot,label="",color=expcolors[i],ls='dotted',alpha=1)
ax.legend()
#ax.set_xticks(xtks)
#ax.set_xticklabels(xper)
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xlabel("Period (Years)")
ax.set_ylabel("Power ($K^2 cpy^{-1}$)")
ax.grid(True)

ax2 = ax.twiny()
ax2.set_xlim([xtks[0],xtks[-1]])
ax2.set_xticks(xtks)
ax2.set_xticklabels(xper)
ax2.grid(True,ls='dotted',color='gray')
ax2.set_xlabel("Period (Years)")
ax.set_title("SST Spectrum for CESM-FULL @ %s"%locstring)


# Add an inset
# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.35, 0.55, 0.30, 0.35]
ax2 = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
ax2 = viz.add_coast_grid(ax2,bbox=bboxinset,fill_color='gray',ignore_error=True)
pcm = ax2.contourf(lonr,latr,amvpats_avg.T,levels=cint,cmap='cmo.balance')
ax2.plot(lonr2[klon],latr2[klat],marker="x",color="k",markersize=10)
plt.savefig("%sSST_Spectra_Comparison_%s.png"%(figpath,locfstring),dpi=150,bbox_inches='tight')

#%% Repeat Analysis for all points within the region

# Do spectral analysis for stochastic model
spec_sm     = np.zeros((nmod,nlonr,nlatr,freqns[0]))*np.nan # [model x lon x lat x freq]

for mid in range(3):
    for o in tqdm(range(nlonr)):
        for a in range(nlatr):
            insst = sst_all[o,a,:,mid]
            if np.any(np.isnan(insst)):
                continue
            specs,_,_,_,_ = scm.quick_spectrum([insst,],[nsmooth[0]],pct,verbose=False)
            spec_sm[mid,o,a,:] = specs[0].copy()
            
#%% Repeat for cesm

spec_full = np.zeros((nlonr,nlatr,freqns[0+3]))*np.nan
spec_slab = np.zeros((nlonr,nlatr,freqns[1+3]))*np.nan

cspecs = [spec_full,spec_slab]
for cid in range(2):
    for o in tqdm(range(nlonr)):
        for a in range(nlatr):
            insst = sst_cesm[cid][o,a,...]
            if np.any(np.isnan(insst)):
                continue
            specs,_,_,_,_ = scm.quick_spectrum([insst,],[nsmooth[cid+3]],pct,verbose=False)
            cspecs[cid][o,a,:] = specs[0].copy() # [lon x lat x freq]

#%% Plot all the spectra for the stochastic model

fig,ax = plt.subplots(1,1,figsize=(10,3))
for o in tqdm(range(nlonr)):
    for a in range(nlatr):
        ax.plot(freqs[0]*dtplot,spec_sm[mid,o,a,:]/dtplot,label="",lw=1,alpha=0.2)
            
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xlabel("Period (Years)")
ax.set_ylabel("Power ($K^2 cpy^{-1}$)")
ax.grid(True)

ax2 = ax.twiny()
ax2.set_xlim([xtks[0],xtks[-1]])
ax2.set_xticks(xtks)
ax2.set_xticklabels(xper)
ax2.grid(True,ls='dotted',color='gray')
ax2.set_xlabel("Period (Years)")

#%% Determine some thresholds:
    
thresvals   = (1/20,1/10,1/5) # 1/yr

# Search for variance at lower frequencies for each threshold
thresids_sm = [(freqs[0]*dtplot)<= (thres) for thres in thresvals]

def calc_specvar(freq,spec,thresval,dtthres,droplast=True
                 ,upperthres=None,return_thresids=False):
    """
    Calculate variance of spectra BELOW a certain threshold
    
    Inputs:
        freq     [ARRAY]   : frequencies (1/sec)
        spec     [ARRAY]   : spectra (Power/cps) [otherdims ..., freq]
        thresval [FLOAT]   : Threshold frequency (in units of dtthres)
        dtthres  [FLOAT]   : Units of thresval (in seconds)
        droplast [BOOL]    : True,start from lowest freq (left riemann sum)
        upperthres [FLOAT] : Upper threshold (in units of dtthres)
        return_thresids [BOOL] : Set to True to just return the threshold indices
    """
    # Get indices of frequencies less than the threshold
    if upperthres is None:
        thresids = freq*dtthres <= thresval
    else:
        thresids = (freq*dtthres >= thresval) * (freq*dtthres <= thresval)
    if return_thresids:
        return thresids
        
    # Limit to values
    specthres = spec[...,thresids]
    freqthres = freq[thresids]
    
    # Compute the variance (specval*df)
    if droplast:
        specval    = specthres[...,:-1]#np.abs((specthres[1:] - specthres[:-1]))/dtthres
    else:
        specval    = specthres[...,1:]
    df       = ((freqthres[1:] - freqthres[:-1]).mean(0))
    return np.sum((specval*df),-1)

specvars = np.zeros([nlonr,nlatr,5,len(thresvals)])*np.nan # [nlon,nlat,model,thresval]
for mid in tqdm(range(5)):
    
    for t,thresval in enumerate(thresvals):
        if mid < 3:
            inspec = spec_sm[mid,...] # [lon x lat x freq]
        else:
            inspec = cspecs[mid-3]
        infreq = freqs[mid]
        specvars[:,:,mid,t] = calc_specvar(infreq,inspec,thresval,dtplot,droplast=True)
        
#%% Now make some plots to compare what is going on...
bboxinset = [-70,-12,10,18]

t       = 2
autocbar = False
vlms      = ([0,0.05],[0,0.10],[0,0.15])
for t in range(len(thresvals)):
    fig,axs = plt.subplots(5,1,subplot_kw={'projection':ccrs.PlateCarree()},
                           constrained_layout=True,figsize=(10,10))
    vlm = vlms[t]
    for mid in range(5):
        blabel = [1,0,0,0]
        if mid == 4:
            blabel[-1] = 1
        
        
        ax  = axs.flatten()[mid]
        ax  = viz.add_coast_grid(ax,bbox=bboxinset,fill_color='gray',blabels=blabel)
        if autocbar:
            pcm = ax.pcolormesh(lonr2,latr2,specvars[:,:,mid,t].T,cmap='cmo.thermal')
            fig.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.008)
        else:
            pcm = ax.pcolormesh(lonr2,latr2,specvars[:,:,mid,t].T,
                                cmap='cmo.thermal',vmin=vlm[0],vmax=vlm[1])
        ax.set_title(expnames[mid])
        
        ax.plot(lonf,latf,marker="x",markersize=20,color="k")
        
        # Plot Masks
        if mid < 3:
            viz.plot_mask(lonr2,latr2,dmsks[mid],ax=ax,markersize=2,color='k',marker="o")
    
    if autocbar is False:
        fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.015)
    plt.suptitle("SST Variance ($K^2$) at Periods > %i years" % (int(1/thresvals[t])))
    savename = "%sSST_Variance_%s_thresid%i.png" % (figpath,fnames[0][30:99],t)
    plt.savefig(savename,dpi=150)

#%% Select a specific Point

lonf = -29 
latf = 15.5
klon,klat    = proc.find_latlon(lonf,latf,lonr2,latr2)
locstring    = "Lon: %.2f, Lat: %.2f" % (lonr[klon],latr[klat])
locfstring   = "lon%i_lat%i" % (lonf,latf)

method = 5
lagstr = 'lag1'

frcname     = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
input_path = datpath + "../model_input/"

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True,method=method,lagstr=lagstr)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]
#klonf,klatf = proc.find_latlon(lonf,latf,lon,lat)
inputs = [h,kprevall,dampingslab,dampingfull,alpha,alpha_full,hblt]
outputs,_,_ = scm.cut_regions(inputs,lon,lat,reg_sel,0)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt = outputs

#outputs_pt = scm.cut_regions(inputs,lon,lat,reg_sel,1,points=[lonf,latf])
#h_pt,kprev_pt,damping_pt,dampingfull_pt,alpha_pt,alpha_full_pt,hblt_pt = outputs_pt

npts = np.prod(np.array(h.shape[:-1]))
# Plot variables

#%% Comparison Plot for a single point

# Plot for a specific point
# Get Point Indices

ylabs = ("MLD (m)",
         "Damping ($Wm^{-2}K^{-1}$)",
         "Forcing ($Wm^{-2}$)")

palpha = 0.1
# Just Plot Everything
xlims = [0,11]
fig,axs = plt.subplots(3,1,figsize=(8,8),constrained_layout=True,sharex=True)

# Plot MLD
ax = axs[0]
mlins = ax.plot(mons3,h.reshape(npts,12).T,alpha=palpha,color="k")
ax.set_ylabel(ylabs[0])
ax.plot(mons3,h[klon,klat,:],marker="x",color='yellow')

ax = axs[1]
dlins = ax.plot(mons3,damping.reshape(npts,12).T,alpha=palpha,color="r")
dlins2 = ax.plot(mons3,dampingfull.reshape(npts,12).T,alpha=palpha,color="b")
ax.set_ylabel(ylabs[1])
ax.plot(mons3,dampingfull[klon,klat,:],marker="x",color='yellow')

ax = axs[2]
flins = ax.plot(mons3,np.linalg.norm(alpha,axis=2).reshape(npts,12).T,alpha=palpha,color="r")
flins_2 = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12).T,alpha=palpha,color="b")
ax.set_xlim(xlims)
ax.set_ylabel(ylabs[2])
ax.plot(mons3,np.linalg.norm(alpha_full,axis=2)[klon,klat,:],marker="x",color='yellow')
plt.suptitle("Red (SLAB), Blue (FULL), Yellow (%s)"% (locstring))
plt.savefig("%sPoint_Comparison_%s.png"% (figpath,locfstring),dpi=150)

#%% Lets select a contour to plot

t        = 2
mid      = 2 
autocbar = False
vlms      = ([0,0.05],[0,0.10],[0,0.15])
testcontour = [0,0.08,0.1,0.12,0.14]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(10,5))
vlm = vlms[t]

ax  = viz.add_coast_grid(ax,bbox=bboxinset,fill_color='gray',blabels=blabel)
if autocbar:
    pcm = ax.pcolormesh(lonr2,latr2,specvars[:,:,mid,t].T,cmap='cmo.thermal')
    fig.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.008)
else:
    pcm = ax.pcolormesh(lonr2,latr2,specvars[:,:,mid,t].T,
                        cmap='cmo.thermal',vmin=vlm[0],vmax=vlm[1])

cl=ax.contour(lonr2,latr2,specvars[:,:,mid,t].T,levels=testcontour,colors="k")
ax.clabel(cl)

ax.set_title(expnames[mid])

ax.plot(lonf,latf,marker="x",markersize=20,color="k")

# Plot Masks
if mid < 3:
    viz.plot_mask(lonr2,latr2,dmsks[mid],ax=ax,markersize=2,color='k',marker="o")
if autocbar is False:
    fig.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.015)
    
plt.suptitle("SST Variance ($K^2$) at Periods > %i years" % (int(1/thresvals[t])))
savename = "%sSST_Variance_%s_thresid%i_CONTOUR.png" % (figpath,fnames[0][30:99],t)
plt.savefig(savename,dpi=150)

#%% Select a threshold

vthres_sel = 0.10
kthres     = specvars[:,:,mid,t] > vthres_sel
kthres = kthres.flatten()


#%% Now plot those points

# Plot for a specific point
# Get Point Indices

ylabs = ("MLD (m)",
         "Damping ($Wm^{-2}K^{-1}$)",
         "Forcing ($Wm^{-2}$)")

palpha = 0.05
salpha = 0.5 # Seleccted Alpha
# Just Plot Everything
xlims = [0,11]
fig,axs = plt.subplots(3,1,figsize=(8,8),constrained_layout=True,sharex=True)

# Plot MLD
ax = axs[0]
mlins = ax.plot(mons3,h.reshape(npts,12).T,alpha=palpha,color="k")
ax.set_ylabel(ylabs[0])
mlins_sel = ax.plot(mons3,h.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")

ax = axs[1]
dlins = ax.plot(mons3,damping.reshape(npts,12).T,alpha=palpha,color="r")
dlins2 = ax.plot(mons3,dampingfull.reshape(npts,12).T,alpha=palpha,color="b")
ax.set_ylabel(ylabs[1])
dlins2_sel = ax.plot(mons3,dampingfull.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")

ax = axs[2]
flins = ax.plot(mons3,np.linalg.norm(alpha,axis=2).reshape(npts,12).T,alpha=palpha,color="r")
flins_2 = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12).T,alpha=palpha,color="b")
ax.set_xlim(xlims)
ax.set_ylabel(ylabs[2])
flins_sel = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12)[kthres,:].T,
                    alpha=salpha,color="yellow")
plt.suptitle("Red (SLAB), Blue (FULL), Yellow ($var(SST_{%i year})$ >  %.2f $K^{2}$)"% (int(1/thresvals[t]),vthres_sel))
plt.savefig("%sPoint_Comparison_thres%iyrs_thresval%.2f.png"% (figpath,int(1/thresvals[t]),vthres_sel),dpi=150)


#%% Try Adding a 4th subplot with contour inset 


# Plot for a specific point
# Get Point Indices

t        = 0
mid      = 2 

vthreses = np.arange(0.01,0.09,0.01) # [0.08,0.10,0.12,0.14,0.16,0.18,0.20]

ylabs = ("MLD (m)",
         "Damping ($Wm^{-2}K^{-1}$)",
         "Forcing ($Wm^{-2}$)")


palpha = 0.05
salpha = 0.3 # Seleccted Alpha
# Just Plot Everything
xlims = [0,11]
vlm   = vlms[t]
for vthres_sel in tqdm(vthreses):

    kthres     = specvars[:,:,mid,t] > vthres_sel
    kthres = kthres.flatten()
    
    fig,axs = plt.subplots(4,1,figsize=(8,8),constrained_layout=True,sharex=True)
    
    # Plot MLD
    ax = axs[0]
    mlins = ax.plot(mons3,h.reshape(npts,12).T,alpha=palpha,color="k")
    ax.set_ylabel(ylabs[0])
    mlins_sel = ax.plot(mons3,h.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")
    
    ax = axs[1]
    dlins = ax.plot(mons3,damping.reshape(npts,12).T,alpha=palpha,color="r")
    dlins2 = ax.plot(mons3,dampingfull.reshape(npts,12).T,alpha=palpha,color="b")
    ax.set_ylabel(ylabs[1])
    dlins2_sel = ax.plot(mons3,dampingfull.reshape(npts,12)[kthres,:].T,alpha=salpha,color="yellow")
    
    ax = axs[2]
    flins = ax.plot(mons3,np.linalg.norm(alpha,axis=2).reshape(npts,12).T,alpha=palpha,color="r")
    flins_2 = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12).T,alpha=palpha,color="b")
    ax.set_xlim(xlims)
    ax.set_ylabel(ylabs[2])
    
    flins_sel = ax.plot(mons3,np.linalg.norm(alpha_full,axis=2).reshape(npts,12)[kthres,:].T,
                        alpha=salpha,color="yellow")
    
    ax = axs[3]
    ax.axis('off')
    
    # Add Locator
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    ax2 = axs[3]
    left, bottom, width, height = [0.05, -0.4, 0.95, 1]
    ax2 = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
    ax2 = viz.add_coast_grid(ax2,bbox=bboxinset,fill_color='gray',ignore_error=True)
    pcm = ax2.pcolormesh(lonr2,latr2,specvars[:,:,mid,t].T,
                        cmap='cmo.thermal',vmin=vlm[0],vmax=vlm[1])
    cl  = ax2.contour(lonr2,latr2,specvars[:,:,mid,t].T,levels=[vthres_sel,],colors="k")
    ax2.clabel(cl)
    viz.plot_mask(lonr2,latr2,dmsks[mid],ax=ax2,markersize=2,color='k',marker="o")
    #fig.colorbar(pcm,ax=ax2,fraction=0.002)
    
    plt.suptitle("Red (SLAB), Blue (FULL), Yellow ($\sigma^2_{SST}$ >  %.2f $K^{2}$)"% (vthres_sel))
    plt.savefig("%sPoint_Comparison_thres%iyrs_thresval%.2f_wmap.png"% (figpath,int(1/thresvals[t]),vthres_sel),dpi=150,bbox_inches='tight')
# add spectra?

#%% Do the same as Above, but this time for the SST Spectra
t        = 2
mid      = 2 

vthreses = [0.08,0.10,0.12,0.14,0.16,0.18,0.20]
palpha   = 0.05
salpha   = 0.3 # Seleccted Alpha

npts = len(lonr2)*len(latr2)

for v,vthres_sel in tqdm(enumerate(vthreses)):
    
    # Get Threshold
    kthres     = specvars[:,:,mid,t] > vthres_sel
    kthres = kthres.flatten()
    
    # Get specs
    specsplot = spec_sm[mid,...].reshape(npts,spec_sm.shape[-1])# points x freq
    
    # Make Figure
    fig,ax = plt.subplots(1,1,figsize=(10,3))
    ln1    = ax.plot(freqs[0]*dtplot,specsplot.T/dtplot,label="",lw=1,alpha=0.1,color='k')
    ln1_se = ax.plot(freqs[0]*dtplot,specsplot[kthres,:].T/dtplot,label="",lw=1,alpha=0.5,color='yellow')
    
    ax.axvline(thresvals[t],ls='dashed',lw=2,color="r")
    
    #ax.axhline(thresvals[t],ls='dashed',lw=2,color="r")
    
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.set_xlabel("Period (Years)")
    ax.set_ylabel("Power ($K^2 cpy^{-1}$)")
    ax.grid(True)

    ax2 = ax.twiny()
    ax2.set_xlim([xtks[0],xtks[-1]])
    ax2.set_xticks(xtks)
    ax2.set_xticklabels(xper,fontsize=8,rotation=45)
    ax2.grid(True,ls='dotted',color='gray')
    ax2.set_xlabel("Period (Years)")
    ax.set_title("%s Points with $\sigma_{SST}$ > %.2f $K^2$ at Periods > %i yrs" % (kthres.sum(),vthres_sel,1/thresvals[t]))

    plt.savefig("%sSST_Spectra_Comparison_%s_tthres%iyr_vthres%.2f.png"%(figpath,locfstring,1/thresvals[t],vthres_sel),dpi=150,bbox_inches='tight')

# -------------------------------------
#%% -- Make Some Parameter Scatterplots
# -------------------------------------

# Data Preparation ...

# Compute the variance (SM)
sst_all_var = np.var(sst_all,2) # [lon x lat x model]

# Compute variance (CESM)
sst_var_cesm = np.zeros([nlonr,nlatr,2])*np.nan # [lon x lat x model (FULL,SLAB)]
for i in range(2):
    sst_var_cesm[...,i] = sst_cesm[i].var(2)
    
# Set up input parameters
slabparam = [damping,hblt,np.linalg.norm(alpha,axis=2)]
fullparam = [dampingfull,h,np.linalg.norm(alpha_full,axis=2)]
smparams  = [fullparam,slabparam]
vnames  = ("Heat Flux Feedback $(Wm^{-2} K^{-1})$",
           "Mixed-Layer Depth ($m$)",
           "Forcing Amplitude $(W/m^2)$",
            )
vnamesf = ("lbd","mld","frc")

# Calculate "Effective" Parameters
cp0     = 3996
rho     = 1026
dt      = 3600*24*30
conv    = dt/(rho*cp0)
slabeff = [damping/hblt*conv,
           np.linalg.norm(alpha,axis=2)/hblt*conv]
fulleff = [dampingfull/h*conv,
           np.linalg.norm(alpha_full,axis=2)/h*conv]
smparamseff = [fulleff,slabeff]
vnameseff   = ("Effective Damping $K month^{-1}$",
               "Effective Forcing $K month^{-1}$")
vnamesfeff  = ("lbdeff","frceff")

#%% Scatterplot 1 (Mean Damping vs SST Variance)

fig,axs = plt.subplots(1,3,figsize=(12,8),sharex=True,sharey=True)

for mid,ax in enumerate(axs):
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(modelnames[mid])
    
    
    if mid == 0:
        xvar = damping.mean(2)
    else:
        xvar = dampingfull.mean(2)
        
    sigpts = (dmsks[mid] == 1).flatten()
    
    
    ax.scatter(xvar.flatten()[sigpts],sst_all_var[...,mid].flatten()[sigpts],color=mcolors[mid],marker="o",facecolors='none',alpha=0.75)
    ax.scatter(xvar.flatten()[~sigpts],sst_all_var[...,mid].flatten()[~sigpts],color='gray',marker="x",alpha=1)
    
    if mid == 0:
        ax.set_ylabel("SST Variance ($K^2$)")
    if mid == 1:
        ax.set_xlabel("Damping ($Wm^{-2}K^{-1}$)")
    ax.grid(True,ls='dotted',color='gray')
    

#%%

fig,axs = plt.subplots(1,3,figsize=(12,8),sharex=True,sharey=True)

for mid,ax in enumerate(axs):
    #ax.set_aspect('equal', adjustable='box')
    ax.set_title(modelnames[mid])
    
    
    if mid == 0:
        xvar = hblt.min(2)
    else:
        xvar = h.min(2)
        
    sigpts = (dmsks[mid] == 1).flatten()
    
    
    ax.scatter(xvar.flatten()[sigpts],sst_all_var[...,mid].flatten()[sigpts],color=mcolors[mid],marker="o",facecolors='none',alpha=0.75)
    ax.scatter(xvar.flatten()[~sigpts],sst_all_var[...,mid].flatten()[~sigpts],color='gray',marker="x",alpha=1)
    
    if mid == 0:
        ax.set_ylabel("SST Variance ($K^2$)")
    if mid == 1:
        ax.set_xlabel("MLD (m)")
    ax.grid(True,ls='dotted',color='gray')
    

#%% CESM vs. SM

vmethod = "Mean" # Mean,Max,Min


v         = 2


inparams  = smparamseff # smparams, or smparamseff
inlabels  =  vnameseff  # vnames  , or vnameseff
inlabelsf = vnamesfeff  # vnamesf , or vnamesfeff
vrange    = len(inparams)

cmap = 'plasma'
for vmethod in tqdm(["Mean","Max","Min"]):
    for v in range(vrange):
        fig,axs = plt.subplots(1,3,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
        for mid,ax in enumerate(axs):
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(modelnames[mid])
            
            # Set the input parameters
            if mid == 0:
                xvar = sst_var_cesm[...,1] # SLAB
                cvarin = inparams[1][v]
            else:
                xvar = sst_var_cesm[...,0] # FULL
                cvarin = inparams[0][v]
                
            # Make Adjustments
            if vmethod == "Mean":
                cvar = cvarin.mean(2)
            elif vmethod == "Max":
                cvar = cvarin.max(2)
            elif vmethod == "Min":
                cvar = cvarin.min(2)
            else:
                print("vmethod must be one of [Mean,Max,Min]")
                continue
                
            sigpts = (dmsks[mid] == 1).flatten()
            
            
            pts=ax.scatter(xvar.flatten()[sigpts],sst_all_var[...,mid].flatten()[sigpts],
                           c=cvar.flatten()[sigpts],
                           marker="o",alpha=0.55,cmap=cmap)
            pts=ax.scatter(xvar.flatten()[~sigpts],sst_all_var[...,mid].flatten()[~sigpts],
                           c=cvar.flatten()[~sigpts],
                           marker="x",alpha=1,cmap=cmap)
            
            ax.plot([0,1],[0,1],color="k",lw=0.75,ls='dotted')
            
            
            if mid == 0:
                ax.set_ylabel("Stochastic Model SST Variance ($K^2$)")
            if mid == 1:
                ax.set_xlabel("CESM SST Variance ($K^2$)")
            ax.grid(True,ls='dotted',color='gray')
        ax.set_xlim([0,0.6])
        ax.set_ylim([0,0.6])
        cb = fig.colorbar(pts,ax=axs.flatten(),orientation='vertical',fraction=0.015,pad=0.02)
        cb.set_label(vmethod + " " + inlabels[v])
        savename = "%sRegion_%s_Scatterplot_CESMvsSM_%s_%s.png" % (figpath,reg_strf,inlabelsf[v],vmethod)
        plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Quickly check the forcing
bboxinset = [-70,-15,10,18]
plotvars = [np.linalg.norm(alpha_full,axis=2),np.linalg.norm(alpha,axis=2),
            np.linalg.norm(alpha_full,axis=2)-np.linalg.norm(alpha,axis=2)]

fig,axs = plt.subplots(3,1,subplot_kw={'projection':ccrs.PlateCarree()})
for i in range(3):
    ax = axs[i]
    if i < 2:
        vlm = [10,25]
        cmap='cmo.solar'
        title = "%s %s" % (mconfigs[i],vnames[2])
    else:
        vlm = [-3,3]
        cmap='cmo.balance'
        title = "FULL - SLAB"
    ax = viz.add_coast_grid(ax,bbox=bboxinset,fill_color='gray')
    pcm = ax.pcolormesh(lonr2,latr2,plotvars[i].mean(-1).T
                        ,vmin=vlm[0],vmax=vlm[1],cmap=cmap)
    ax.set_title(title)
    fig.colorbar(pcm,ax=ax)

#%% Similar to Laifang's paper, plot the Effective Forcing vs Damping


fig,ax = plt.subplots()


#%% Responding to YO's comment, just plot the variance to see what is going on

bboxinset = [-70,-12,10,18]

t         = 2
vlm       = [0,0.4]

#vlms      = ([0,0.05],[0,0.10],[0,0.15])
autocbar  = False
fig,axs = plt.subplots(5,1,subplot_kw={'projection':ccrs.PlateCarree()},
                       constrained_layout=True,figsize=(10,10))
#vlm = vlms[t]


for mid in range(5):
    blabel = [1,0,0,0]
    if mid == 4:
        blabel[-1] = 1
    
    ax  = axs.flatten()[mid]
    ax  = viz.add_coast_grid(ax,bbox=bboxinset,fill_color='gray',blabels=blabel)
    
    if mid < 3:
        plotvar = np.var(sst_all[:,:,:,mid],-1)
    else:
        plotvar = np.var(sst_cesm[mid-3],-1)
    
    if autocbar:
        pcm = ax.pcolormesh(lonr2,latr2,plotvar.T,cmap='cmo.thermal',shading='nearest')
        fig.colorbar(pcm,ax=ax,orientation='vertical',fraction=0.008)
    else:
        pcm = ax.pcolormesh(lonr2,latr2,plotvar.T,
                            cmap='cmo.thermal',vmin=vlm[0],vmax=vlm[1],shading='nearest')
    ax.set_title(expnames[mid])
    
    # Plot Masks
    if mid < 3:
        viz.plot_mask(lonr2,latr2,dmsks[mid],ax=ax,markersize=2,color='k',marker="o")

if autocbar is False:
    fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.015)
plt.suptitle("SST Variance ($K^2$)" )
savename = "%sSST_Variance_%s.png" % (figpath,fnames[0][30:99])
plt.savefig(savename,dpi=150)



