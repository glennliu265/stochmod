#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare stochastic model output throughout the whole basin

Created on Mon Aug 16 18:23:33 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
import time
import time
from tqdm import tqdm
#%% Set Paths

stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20211001/"
    
    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    
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

#%% Settings Part 1 (Comparison)


## Compare Effect of including ekman advection
# Names of the experiments
# expids    = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
#              "stoch_output_forcingflxeof_qek_50eofs_SLAB-PIC_1000yr_run003.npz"
#             ]
# exnames   = ["SM (no $Q_{ek}$)",
#              "SM (with $Q_{ek}$)"
#             ]
# excolors  = ["blue",
#               "orange"]
#exoutnameraw = "Q-ek_comparison_50EOF"

## Compare before/after correctio,n and inclusion of Amp
expids = ["stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run003.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run005.npz",
          "stoch_output_forcingflxeof_50eofs_SLAB-PIC_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz"
          ]
exnames = ["Before (Incl. Lat-weight)",
           "After (no q-corr)",
           "After (with q-corr)",
           "90% Variance Threshold (no-qcorr)"]
excolors = ["blue",
            "orange",
            "magenta",
            "red"]
exoutnameraw = "latweightfix_qamp_comparison_50EOF"

## Examine zffect of including AMP using variance-based threshold
expids = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run004.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run007.npz"]
exnames = ["no q-corr",
           "with q-corr (monthly)",
           "with q-corr (avg)"]
excolors = ["blue",
            "orange",
            "magenta"]
exoutnameraw = "90pctvariance_qamp_comparison"


## Examine effect of using amp across 2 different types of foring
expids = ["stoch_output_forcingflxstd_SLAB-PIC_1000yr_run006_ampq0.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0.npz",
          "stoch_output_forcingflxstd_SLAB-PIC_1000yr_run006_ampq1.npz",
          "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1.npz"]

exnames = ["var(Q) forcing",
           "EOF-based forcing",
           "var(Q) forcing (qcorr)",
           "EOF-based forcing(qcorr)"]
           # "with q-corr (monthly)",
          
excolors = ["cyan",
            "blue",
            "magenta",
            "orange"]
            #"magenta"]
exoutnameraw = "old_vs_eof_fullcomparison"

## Same as above, but now correcting locally for eof variance
expids  = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq0.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1.npz"
          ]

exnames = ["Basinwide Correction",
           "Local Correction",
           "Basinwide Correction (with q-corr)",
           "Local Correction (with q-corr)"
            ]

excolors = ["cyan",
            "blue",
            "magenta",
            "orange"]

exoutnameraw = "basinwide_vs_local_correction"

## New Variance Correction Method
expids  = ["stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run008_ampq3.npz",
           "stoch_output_forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run008_ampq3.npz",
          ]

exnames = ["Basinwide Correction (Old)",
           "Local Correction (Old)",
           "Basinwide Correction (New)",
           "Local Correction (New)"
            ]
exoutnameraw = "new_v_old_q-correction"

## 
expids = (
    "stoch_output_forcingflxeof_EOF1_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3.npz",
    "stoch_output_forcingflxeof_EOF2_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3.npz",
    "stoch_output_forcingflxeof_2eofs_SLAB-PIC_eofcorr0_1000yr_runtest009_ampq3_local.npz",
    )
exnames = ("EOF1 (NAO)",
            "EOF2 (EAP)",
            "Both (NAO+EAP)",
            )
exoutnameraw = "NAO_EAP"

#%% Settings Part 2

# Experiment information
bboxsim    = [-100,20,-20,90] # Simulation Box


# CESM Names
cnames    = ["CESM-FULL","CESM-SLAB",]
ccolors   = ["black","gray"]
cpath     = datpath + "../"

# Combine Names
enames    = np.hstack([cnames,exnames])
ecolors   = np.hstack([ccolors,excolors])


# Load lat/lon
lon180,lat  = scm.load_latlon(rawpath)
lon360,_    = scm.load_latlon(rawpath,lon360=True)
lonr        = np.load(datpath+"lon.npy")
latr        = np.load(datpath+"lat.npy")

# Model to conduct analysis on
model       = 0
modelnames  = ["h constant","h vary","entraining"]

exoutname  = exoutnameraw +  "_model%i" % model
# Autocorrelation parameters
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')


#%%




#%% Load CESM and Stochastic model data

st   = time.time()
ssts = scm.load_cesm_pt(cpath)
Qs = []
for i in tqdm(range(len(expids))):
    ld  = np.load(datpath+expids[i])
    sst = ld['sst']
    Q   = ld['Q']
    ssts.append(sst[model,...])
    Qs.append(Q)

print("Loaded data in %.2fs"%(time.time()-st))


#%% Crop CESM-data to the same region

# Apply land/ice mask
msk = np.load(lipath)
    
def preprocess_CESM(sst,lon360,lat,msk,bbox):
    """
    Takes sst [time x lat x lon360]
     1. apply land ice mask
     2. tranpose to lon x lat x time
     3. flip to lon180
     4. cut to region
    
    """
    # 1. Apply mask
    sstin = sst.copy()
    sstin *= msk[None,:,:] 
    
    # 2. transpose
    sstin = sstin.transpose(2,1,0)
    
    # 3. flip longitude
    lon180,sstin = proc.lon360to180(lon360,sstin)
    
    # 4. select region
    sstr,lonr,latr = proc.sel_region(sstin,lon180,lat,bbox)
    return sstr

for i in tqdm(range(2)):
    ssts[i] = preprocess_CESM(ssts[i],lon360,lat,msk,bboxsim)

# <You can now skip down below to regional analysis section if needed>

#%% Examine the variance ratio between both locations

sstvar = []
for i in range(len(ssts)):
    sstvar.append(ssts[i].var(2))

Qvar = []
for i in np.arange(2,len(sstvar)):
    Qvar.append(Qs[i-2].var(2))
    
#%% Plot the difference in variance (all three)

bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
vlm = [0,2]
clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)
fig,axs = plt.subplots(1,3,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

for expid in range(len(exnames)):
    
    ax = axs.flatten()[expid]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
    if model < 1: # Non-entraining, compare with CESM-SLAB
        comparison = sstvar[1]
        comparename = enames[1]
    else: # entraining, compare with CESM-FULL
        comparison = sstvar[0]
        comparename = enames[0]
    #pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
    pcm = ax.contourf(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clvl,cmap="RdBu_r")
    cl = ax.contour(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clab,colors="k",linewidths=0.75)
    ax.clabel(cl)
    fig.colorbar(pcm,ax=ax)
    ax.set_title("Ratio of SST Variance \n (%s/%s)" % (enames[expid+2],comparename))
plt.savefig("%s%s_VarianceRatio_expnum%i.png" % (figpath,exoutname,expid),bbox_inches='tight',dpi=200)

#%% Plot selected ratio,compare with CESM-SLAB


plotnum = 4
bboxplot = [-90,0,0,75]

vlm = [0,2]
clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)
comparison  = sstvar[plotnum]/sstvar[1]
#comparison = sstvar[2]/sstvar[3]
comparename = "SST Variance Ratio (%s/CESM-SLAB)" %enames[plotnum]



#pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
pcm1 = ax.pcolormesh(lonr,latr,comparison.T,cmap="RdBu_r",vmin=clvl[0],vmax=clvl[-1])
pcm = ax.contourf(lonr,latr,comparison.T,levels=clvl,cmap="RdBu_r")
cl = ax.contour(lonr,latr,comparison.T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax,pad=0.01)

ax.set_title("%s" % (comparename))
plt.savefig("%s%s_expnum%i_sstratio_plotnum%s.png" % (figpath,exoutname,expid,plotnum),bbox_inches='tight',dpi=200)

#%% Plot the ratio of Qnet (corrected and uncorrected)
# OR plot the ratio of SST


# vlm  = [0,2]
# clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
# clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)
plot_sstvar = True
plotnum = 0

bboxplot = [-90,0,0,75]

vlm = [0,2]
clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)

if plotnum == 0: #var(Q)-based
    comparename = "%s"%(enames[2])
    if plot_sstvar:
        comparison = sstvar[2]/sstvar[4]
    else:
        comparison  = Qvar[0]/Qvar[2]
    
else: # EOF-based
    comparename = "%s"%(enames[3])
    if plot_sstvar:
        comparison = sstvar[3]/sstvar[5]
    else:
        comparison = Qvar[1]/Qvar[3]

#pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
#pcm = ax.pcolormesh(lonr,latr,comparison.T,cmap="RdBu_r")
pcm = ax.contourf(lonr,latr,comparison.T,levels=clvl,cmap="RdBu_r")
cl = ax.contour(lonr,latr,comparison.T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax,pad=0.01)
if plot_sstvar:
    ax.set_title("Ratio of $SST_{no q-corr}/SST_{q-corr}$ \n (%s)" % (comparename))
    plt.savefig("%s%s_expnum%i_sstratio_plotnum%s.png" % (figpath,exoutname,expid,plotnum),bbox_inches='tight',dpi=200)
else:
    ax.set_title("Ratio of $Q_{uncorrected}/Q_{corrected}$ \n (%s)" % (comparename))
    plt.savefig("%s%s_expnum%i_Qratio_plotnum%s.png" % (figpath,exoutname,expid,plotnum),bbox_inches='tight',dpi=200)


#%% 

#expid = 
bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
#vlm = [0.5,1.5]
#vlm = [-1,3]
vlm = [-1,3]
clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)

if model < 1: # Non-entraining, compare with CESM-SLAB
    comparison = sstvar[1]
    comparename = enames[1]
else: # entraining, compare with CESM-FULL
    comparison = sstvar[0]
    comparename = enames[0]
pcm = ax.pcolormesh(lonr,latr,(sstvar[expid+2]/comparison).T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
pcm = ax.contourf(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clvl,cmap="RdBu_r")
cl = ax.contour(lonr,latr,(sstvar[expid+2]/comparison).T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax,pad=0.01)
ax.set_title("Ratio of SST Variance \n (%s/%s)" % (enames[expid+2],comparename))

plt.savefig("%s%s_expnum%i.png" % (figpath,exoutname,expid),bbox_inches='tight',dpi=200)

#%% Compare this with a plot of the actual theoertical NHFLX underestimate

# lon x lat x mon
lbd_a    = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/CESM-Slab-PIC_lbd_a_Natl_mon.npy")

# Plot the theoretical underestimate
a        = 1-lbd_a.mean(2)[...,None]
underest = 2*a**2 / (1+a)
underest = np.sqrt(underest.squeeze())

bboxplot = [-90,0,0,75]
#vlm = [.90,1.10]
vlm = [0.5,1.5]

#clvl = np.arange(vlm[0],vlm[-1]+.05,0.05)
#clab = np.arange(vlm[0],vlm[-1]+0.1,0.1)

fig,ax = plt.subplots(1,1,figsize=(12,4),subplot_kw={'projection':ccrs.PlateCarree()})

ax = viz.add_coast_grid(ax,bbox=bboxplot)

if model < 1: # Non-entraining, compare with CESM-SLAB
    comparison = sstvar[1]
    comparename = enames[1]
else: # entraining, compare with CESM-FULL
    comparison = sstvar[0]
    comparename = enames[0]
#pcm = ax.pcolormesh(lonr,latr,underest.T,vmin=vlm[0],vmax=vlm[-1],cmap="RdBu_r")
pcm = ax.contourf(lonr,latr,underest.T,levels=clvl,cmap="RdBu_r")
cl = ax.contour(lonr,latr,underest.T,levels=clab,colors="k",linewidths=0.75)
ax.clabel(cl)
fig.colorbar(pcm,ax=ax)
ax.set_title("Theoretical NHFLX Underestimate (var(Q)/var(q))")
plt.savefig("%s%s_Theoretical_Underest_AnnAvg.png" % (figpath,exoutname),bbox_inches='tight',dpi=200)





#%% Load mixed layer depths to identify the base month

mld    = np.load(rawpath + "FULL_HTR_HMXL_hclim.npy")
#kmonth = np.argmax(mld[klon,klat]) 

#%% <Regional Analysis> %%




#%% Select the bounding boxes
bbox_SP    = [-60,-15,40,65]
bbox_ST_e  = [-80,-40,20,40]
bbox_ST_w  = [-40,-10,20,40]
bbox_TR    = [-75,-15,10,20]
bbox_NA    = [-80,0 ,10,65]
bboxes     = (bbox_SP,bbox_ST_w,bbox_ST_e,bbox_TR,bbox_NA) # Bounding Boxes
regions    = ("SPG","STGw","STGe","TRO","NAT","NAT")#,"NNAT")        # Region Names
regionlong = ("Subpolar","Subtropical (West)","Subtropical (East)","Tropical","North Atlantic (10N-65N)")
bbcol      = ["Blue","Red","cornflowerblue","Yellow","Black"]
bbsty      = ["solid","dashed","dotted","solid","dotted"]

#%% Test Plot
bboxplot = [-100,20,0,80]
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(5,5))
ax = viz.add_coast_grid(ax,bboxplot)

# Plot the amv pattern
#pcm = ax.contourf(lon180g,latg,cesmpat[rid][cid].T,levels=cint,cmap=cmocean.cm.balance)
#ax.pcolormesh(lon180g,latg,cesmpat[rid][cid].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
#cl = ax.contour(lon180g,latg,cesmpat[rid][cid].T,levels=cl_int,colors="k",linewidths=0.5)
#ax.clabel(cl,levels=cl_int,fontsize=8)
#ax.set_title("Regional Analysis Bounding Boxes")
#cb = fig.colorbar(pcm,ax=ax,orientation='horizontal')
#cb.set_label("CESM-SLAB AMV ($\degree C$ per $\sigma_{AMV}$)")
#%%
# 
ls = []
for bb in [0,1,2,4]:
    ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                         color=bbcol[bb],linestyle=bbsty[bb],linewidth=2,return_line=True)
    ls.append(ll)
    
ax.legend(ncol=4)




#%% 
debug = 0
lags  = np.arange(0,37,1)
nsmooth = 100
pct     = 0.10

rssts    = []  # [region][model]
racs     = []  # Autocorrelations
raccfs   = []  # Confidence intervals (AC)
rspecs   = []  # Spectra
rfreqs   = []  # Frequencies
rkmonth  = []  # Deepest MLD month


for r in range(len(bboxes)):
    
    rsst  = [] # Containing regional SSTs for each model
    rac   = []
    rspec = []
    rfreq = []
    raccf = []
    
    for d in tqdm(range(len(ssts))):
        # Test
        if debug:
            tvar,tlon,tlat = proc.sel_region(ssts[d],lonr,latr,bbox=bboxes[r])
            plt.pcolormesh(tlon,tlat,tvar[:,:,0].T),plt.title(regionlong[r])
            
        # Calculate regional average
        ravg = proc.sel_region(ssts[d],lonr,latr,bboxes[r],reg_avg=1,awgt=1)
        rsst.append(ravg)
        
        # Calulate Autocorrelation, and confidence levels
        rmld   = proc.sel_region(mld,lon180,lat,bboxes[r],reg_avg=1,awgt=1)
        kmonth = np.argmax(rmld)
        rkmonth.append(kmonth)
        ac = scm.calc_autocorr([ravg],lags,kmonth+1)[0]
        rac.append(ac)
        
        # Calculate spectra
        specs,freqs,CCs,dofs,r1s = scm.quick_spectrum([ravg,],[nsmooth,],pct)
        rspec.append(specs[0])
        rfreq.append(freqs[0])
        
        # End region loop
        
    rssts.append(rsst)
    racs.append(rac)
    rspecs.append(rspec)
    rfreqs.append(rfreq)


#%%

rid = 0


timemax = None
xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
dt   = 3600*24*30
speccolors = ["k","gray","r","b","m",]
specnames  = np.hstack([cnames,modelnames])

exnames = ("EOF1 (NAO)",
            "EOF2 (EAP)",
            "Both (NAO+EAP)",
            )


fig,axs = plt.subplots(2,3,figsize=(16,10))

for rid in tqdm(range(5)):
    ax    = axs.flatten()[rid]
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],np.var(rssts[rid][i])) for i in range(len(speccolors)) ]
    ax = viz.plot_freqxpower(rspecs[rid],rfreqs[rid],speclabels,speccolors,
                         ax=ax,plottitle=regionlong[rid])
    
    if rid <2:
        ax.set_xlabel("")
    if rid%2 == 1:
        ax.set_ylabel("")
        
plt.suptitle("Regional AMV Index Spectra")
savename = "%sSST_Spectra_Comparison_Test.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')



