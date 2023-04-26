#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare (Detrended Historical: HTR) and (Preindustrial Control: PiC) AMV/NASST

Question: Does external forcing modulate/impact timescales and
amplitude of internal variability?

Created on Thu Apr 20 13:46:00 2023

@author: gliu
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import sys
import time


import cartopy.crs as ccrs

#%% User Edits

figpath     = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/02_Figures/20230426/"
datpath_pic = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
datpath_htr = "/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/CESM_data/"
bbox        = [-80,0,0,65]

#%% Import Custom Packages

# Import general packages
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz

# Import stochastic model processing packages (PiC)
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
import scm

# Import AMV Prediction processing packages
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/04_Predict_AMV/03_Scripts/predict_amv/")
import amv_dataloader as dl
#%% Functions

def preprocess(sst,lat,lon,bbox):
    
    ntime,nlat,nlon = sst.shape
    
    # Remove mean climatological cycle
    st = time.time()
    climavg,sst_yrmon=proc.calc_clim(sst,0,returnts=True)
    sst_anom = sst_yrmon - climavg[None,:,:,:]
    sst_anom = sst_anom.reshape(sst.shape)
    print("Deseasoned in %.2fs" % (time.time()-st))
    
    # Flip dimensions
    sst_flipped = proc.flipdims(sst_anom) # {Lon x Lat x Time}
    
    # Flip Lat/lon
    if np.any(lon > 180):
        st = time.time()
        print("Flipping Longitude!")
        lon,sst_flipped=proc.lon360to180(lon,sst_flipped)
        print("Flipped lon in %.2fs" % (time.time()-st))
    
    # Crop to region
    print("Cropping Region!")
    sst_region,lonr,latr = proc.sel_region(sst_flipped,lon,lat,bbox)
    print("Cropped in %.2fs." % (time.time()-st))
    
    # Upflip dimensions
    sst_region = proc.flipdims(sst_region) # {Time x Lat x Lon}
    return sst_region,lonr,latr

#%% Load some data (Note, need to check preprocessing steps to make sure I am consistent...)

# Load CESM1 lat lon
lon,lat=scm.load_latlon(lon360=True)

# Load the CESM PiC Data
#st = time.time()
ssts_pic = scm.load_cesm_pt(datpath_pic,loadname='full',grabpoint=None,ensorem=0)
sst_pic  = ssts_pic[0] # Time x lat x lon
#print("Loaded PiC Data in %.2fs" % (time.time()-st))

# Preprocess and crop CESM PiC (deseason, flip dimension, crop to NAtl)
sst_region,lonr,latr = preprocess(sst_pic,lat,lon,bbox)
sst_anom_pic = sst_region

# Take annual average (can remove later if I can find my monthly anomalized Htr Data)
sst_anom_pic = proc.ann_avg(sst_anom_pic,0)

# Load the CESM HTR Data -----------------------------------------------------
sst_htr_norm,lat,lon   = dl.load_data_cesm(["SST",],bbox,detrend=0,return_latlon=True,datpath=datpath_htr)
nfactors               = dl.load_nfactors(["SST",],datpath=datpath_htr,detrend=0,regrid=None)[0]

# Unnormalize
sst_anom_htr = sst_htr_norm * nfactors['stdev'] + nfactors['mean'] # (1, 42, 86, 69, 65)
sst_anom_htr = sst_anom_htr.squeeze()  # (42, 86, 69, 65)

# Remove ensemble mean
sst_anom_htr = sst_anom_htr - sst_anom_htr.mean(0)[None,...]



# Get dimensions
nens,ntime_htr,nlat,nlon = sst_anom_htr.shape
ntime_pic,nlat,nlon = sst_anom_pic.shape

#%%

# Get SSTs
ssts           = [sst_anom_pic,sst_anom_htr]
scenario_names = ["PiControl","Historical"]

# Get consistent limasks and apply them to the data
limask_all = []
for sc in range(2):
    sst_in = ssts[sc]
    while len(sst_in.shape) > 2:
        sst_in = sst_in.sum(0)
    
    limask = sst_in.copy()
    limask[~np.isnan(sst_in)] = 1
    limask_all.append(limask)
limask_universal = np.array(limask_all).sum(0)/len(limask_all)
ssts[0] *= limask_universal[None,:,:]
ssts[1] *= limask_universal[None,None,:,:]


# Compute NASST Index
nasst_pic = proc.area_avg(proc.flipdims(ssts[0]),bbox,lonr,latr,1)
nasst_all = [nasst_pic,]
for e in range(nens):
    nasst_ens = proc.area_avg(proc.flipdims(ssts[1][e,...]),bbox,lonr,latr,1)
    nasst_all.append(nasst_ens)
#nasst_htr = np.array(nasst_htr)

# Compute AMV Index

#%% Do spectral analysis 

nsmooths  = (50,)+(4,)*nens
pct       = 0.10
dtyear    = 3600*24*365
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(nasst_all,nsmooths,pct,
                                              opt=1,dt=dtyear,clvl=[.95],verbose=False)

smoothstr = "Smoothing: PiControl (%i), Historical (%i) | Taper: %03i" % (nsmooths[0],nsmooths[1],pct*100) + "%"
smoothfn  = "smooth_PIC%i_HTR%i_taperpct%03i" % (nsmooths[0],nsmooths[1],pct*100)

#%% Plot the spectra

alw     = 1.25
dt      = dtyear


for ee in range(nens):
    title   = "North Atlantic SST Power Spectra (PiControl vs. Detrended Historical)\n%s" % smoothstr
    xticks  = np.array([1/100,1/50,1/25,1/10,1/5,1/2.5]) #* (1/dtyear)
    xlbls   = (1/xticks).astype(int).astype(str)
    
    
    fig,ax  = plt.subplots(1,1,figsize=(8,3),constrained_layout=True)
    for e in range(nens):
        if e == 0:
            lbl="Historical (Indv. Ens)"
        else:
            lbl=""
        ax.plot(freqs[1+e]*dt,specs[1+e]/dt,color="salmon",alpha=0.4,label=lbl)
    
    ax.plot(freqs[1+e]*dt,np.array(specs[1:]).mean(0)/dt,marker="d",
            label="Historical (Ens. Avg)",color="gray",alpha=1) 
    
    
    ax.plot(freqs[1+ee]*dt,specs[1+ee]/dt,color="darkblue",label="ens %i" %(ee+1))
    ax.plot(freqs[0][:-1]*dt,specs[0]/dt,label="PiControl",color="k",)
    
    ax.legend()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlbls)
    ax.set_xlabel("")
    ax.set_xlim([xticks[0],xticks[-1]])
    
    ax.axvline([1/86],ls="dashed",lw=0.75)
    ax.axvline([1/1801],ls="dashed",lw=0.75)
    
    ax.set_xlabel("Period (Years)")
    ax.set_ylabel("Power ($\degree C^2 cpy^{-1}$)")
    
    
    ax.set_title(title)
    savename = "%sNASST_PowerSpectra_PIC_v_HTR_%s_ensfocus%02i.png" % (figpath,smoothfn,ee+1)
    plt.savefig(savename,dpi=200,)
    #savename = "Power Spectra"

#%% Compute pointwise variance (normal and low-passed)


pic_var       = ssts[0].var(0)
htr_var_byens = ssts[1].var(1)/pic_var[None,:,:]


ssts_lp = []
for ii in range(2):
    if ii == 0:
        sst_lp = proc.lp_butter(ssts[ii],10,6)
        ssts_lp.append(sst_lp)
    elif ii == 1:
        sst_ens_lp = []
        for e in range(nens):
            sst_ens = proc.lp_butter(ssts[ii][e,...],10,6)
            sst_ens_lp.append(sst_ens)
            
        ssts_lp.append(np.array(sst_ens_lp))
    
        
            
        
        
        
    
    
pic_var_lp       = ssts_lp[0].var(0)
htr_var_byens_lp = ssts_lp[1].var(1)/pic_var_lp[None,:,:]

#proc.lp_butter()

#%% Plot Low pass filtered variance and ratios

bbox_plot = [-80,0,0,62]

lp_mode = False

fig,axs = plt.subplots(1,2,constrained_layout=True,
                       subplot_kw={"projection":ccrs.PlateCarree()},figsize=(8,4.5))


for aa in range(2):
    ax = axs[aa]
    
    if aa == 0:
        if lp_mode:
            plotvar = pic_var_lp
        else:
            plotvar = pic_var
        title   = "SST Variance \nPiControl"
        cblabel = "SST Variance ($\degree C^2$)"
        clims   = [0,0.5]
        cmap    = "inferno"
    else:
        
        if lp_mode:
            plotvar = np.log10(htr_var_byens.mean(0))
        else:
            plotvar = np.log10(htr_var_byens_lp.mean(0))
            
        
        title   = "Ens. Mean Ratio of Variance \nDetrended Historical / PiControl"
        cblabel = "$log_{10}(\sigma^2_{H} \, / \, \sigma^2_{P})$"
        clims    = [-0.30,0.30]
        cmap     = 'cmo.balance'

    
    
    blabel=[0,0,0,1]
    if aa == 0:
        blabel[0] = 1
    ax = viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="k",blabels=blabel)
        
    pcm = ax.pcolormesh(lon,lat,plotvar,cmap=cmap,vmin=clims[0],vmax=clims[1])
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05)
    cb.set_label(cblabel)
    ax.set_title(title)
        
    
savename = "%sVariance_Maps_SST_lp%i.png" % (figpath,lp_mode)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Compute the amv pattern

ssts_loop = [proc.flipdims(ssts[0])]
for e in range(nens):
    ssts_loop.append(proc.flipdims(ssts[1][e,...]))
    
    
amvids  = []
amvpats = []
for ii in range(len(ssts_loop)):
    var_in = ssts_loop[ii]
    amvidxout,amvpattern = proc.calc_AMVquick(var_in,lon,lat,bbox,order=6,cutofftime=10,anndata=True,
                      runmean=False,dropedge=0,monid=None,nmon=12,
                      mask=None)
    
    amvids.append(amvidxout)
    amvpats.append(amvpattern.T)


#%%

fig,axs = plt.subplots(1,2,constrained_layout=True,
                       subplot_kw={"projection":ccrs.PlateCarree()},figsize=(8,4))


cmap     = 'cmo.balance'
cblabel  = "AMV Pattern ($\degree C^2 \sigma^{-1}_{AMV}$)"
cints    = np.arange(-0.4,0.425,0.025)

for aa in range(2):
    ax = axs[aa]
    
    if aa == 0:
        plotvar = amvpats[0]
        title   = "PiControl"
        
    else:
        plotvar = np.array(amvpats[1:]).mean(0)
        

        
        title   = "Historical (Ens. Avg. Pattern)"
        clims    = [-0.30,0.30]
        

    
    blabel=[0,0,0,1]
    if aa == 0:
        blabel[0] = 1
    ax = viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="k",blabels=blabel)
        

    ax.set_title(title)
        
    pcm = ax.contourf(lon,lat,plotvar,cmap=cmap,levels=cints)
    cl = ax.contour(lon,lat,plotvar,levels=cints,colors="k",linewidths=0.55)
    ax.clabel(cl,levels=cints[::2])
    
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.05)
cb.set_label(cblabel)
    
savename = "%sAMVPats_SST_lp%i.png" % (figpath,lp_mode)
plt.savefig(savename,dpi=150,bbox_inches="tight")


#%% Plot for each ensemble member

for e in range(nens):
    
    fig,ax = plt.subplots(1,1,constrained_layout=True,
                           subplot_kw={"projection":ccrs.PlateCarree()},figsize=(6,4.5))
    ax = viz.add_coast_grid(ax,bbox=bbox_plot,fill_color="k")
    ax.set_title("Ens %02i" % (e+1))
    
    
    plotvar = np.array(amvpats[1:])[e,...]
    pcm = ax.contourf(lon,lat,plotvar,cmap=cmap,levels=cints)
    cl = ax.contour(lon,lat,plotvar,levels=cints,colors="k",linewidths=0.55)
    ax.clabel(cl,levels=cints[::2])
    
    cb = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.05)
    cb.set_label(cblabel)
    
    savename = "%sAMVPats_SST_ens%02i.png" % (figpath,e+1)
    plt.savefig(savename,dpi=150,bbox_inches="tight")
    





#%% Unwritten Bottom?
#%%

speccolors = ("k",) + ("r",)*nens
speclabels = ("CESM1_PiControl",) + ("",)*nens
ax,ax2 = viz.plot_freqlog(specs,freqs,speclabels,speccolors,lw=alw,
                     ax=ax,plottitle=title,
                     xlm=None,xtick=None,return_ax2=True,
                     plotids=None,legend=False)

# Plot the power spectra (bottom row)
for i in range(len(rids)+1):
    
    fig,ax = plt.subplots(1,1,figsize=(8,3),constrained_layout=True)
    
    #ax  = axs[1,i]
    
    if i == len(rids):
        rid = order[-1]
        legendflag = True
    else:
        rid = order[i]
        legendflag = False
    
    if plotar1:
        conf_in = Cfsall[rid]
    else:
        conf_in = None
    
    #speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
    nspecs = len(specsall[rid])
    speclabels= specnames
    
    
    if legendflag:
        
        plotnames  = ("Non-Entraining","Entraining","CESM-FULL")
        plotcolors = ('magenta','orange','w')
        for n in range(3):
            ax.plot(0,0,color=plotcolors[n],label=plotnames[n])
        ax.legend(ncol=3,fontsize=12)
        
    else:
        
        if plotlog:
            ax,ax2 = viz.plot_freqlog(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                                 ax=ax,plottitle=regionlong[rid],
                                 xlm=xlm,xtick=yrticks,return_ax2=True,
                                 plotids=plotidspec,legend=False,usegrid=usegrid)
        else:
                
            
            ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                                 ax=ax,plottitle=regionlong[rid],plotconf=conf_in,
                                 xlm=xlm,xtick=yrticks,return_ax2=True,
                                 plotids=plotidspec,legend=legendflag,linearx=linearx,usegrid=usegrid)
    
            
        # Turn off title and second axis labels
        if periodx: # Switch Frequency with Period for x-axis.
            ax2.set_xlabel("")
            sxtk2 = ax2.get_xticklabels()
            xtk2new = np.repeat("",len(sxtk2))
            ax2.set_xticklabels(sxtk2new)
            ax.set_xticklabels(1/xtks)
            
            # Move period labels to ax1
            ax.set_xticklabels(xper)
            ax.set_xlabel("Period (Years)")
        else:
            if i == 1:
                ax2.set_xlabel("Period (Years)")
        
        
        #ax.grid(False,ls='dotted',alpha=0.5)
        
        # Set Rotation of Period Labels
        if plotlog is False:
            rotation  =0
            xfontsize =8
        else:
            rotation  =0
            xfontsize =8
        
        if periodx:
            plt.setp(ax.get_xticklabels(), rotation=rotation,fontsize=xfontsize)
        else:
            plt.setp(ax2.get_xticklabels(), rotation=rotation,fontsize=xfontsize)
        
        if useC:
            ax.set_ylabel("Power Spectrum ($\degree C^2 /cpy$)",fontsize=axisfs)
        else:
            ax.set_ylabel("Power Spectrum ($K^2 /cpy$)",fontsize=axisfs)
                
        if i == 0:# Turn off y label except for leftmost plot
    
            ax.set_ylim(specylim_spg)
        else:
    
            ax.set_ylim(specylim_stg)
            
            
        if plotlog:
            ax.set_ylim([1e-2,1e0])
            
        ax.set_title(title,color=bbcol[rid],fontsize=16,fontweight="bold")
        ax.set_xlabel("Frequency (Cycles/Year)",fontsize=axisfs)
        ax2.set_xlabel("Period (Years)",fontsize=axisfs)
    
        #title = "%s Power Spectra" % (regions[rid])
        #title = ""
        #ax.set_title(region[rid],color=bbcol[rid],fontsize=12)
        
        ax = viz.label_sp(sp_id,fontsize=14,fig=fig,labelstyle="(%s)",case='lower',alpha=0.7)
        sp_id += 1
        
    if legendflag:
        plt.savefig("%sRegional_Autocorrelation_Spectra%s_%s_legend.png"%(figpath,smoothname,regions[rid]),
                    dpi=200,transparent=False)

    else:
        plt.savefig("%sRegional_Autocorrelation_Spectra%s_%s.png"%(figpath,smoothname,regions[rid]),
                    dpi=200,transparent=False)

plt.tight_layout()




        #%%
    
                




# Make landice mask consistent

# Compute the area-weighed average NASST






