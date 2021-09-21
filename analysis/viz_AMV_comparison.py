#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize AMV (comparisons)

Compare selected stochastic model run with CESM1 Output
Works with output from sm_analysis_rewrite (scm.postprocess_output)

These will make rather dense plots, intended for the manuscript

Created on Mon Sep 13 12:24:53 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210920/"
   
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
#%% User Edits


# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over
runid     = "008"

# Analysis (7/26/2021, comparing 80% variance threshold and 5 or 3 EOFs)
#fnames      = ["flxeof_080pct_SLAB-PIC","flxeof_5eofs_SLAB-PIC","flxeof_3eofs_SLAB-PIC"]
#frcnamelong = ("80% Variance Threshold","5 EOFs","3 EOFs")

# Analysis: Trying different number of EOFs
#neofs       = [1,2,50]#[1,2,3,5,10,25,50]
neofs = [90]
#fnames      = ["flxeof_qek_%ieofs_SLAB-PIC" % x for x in neofs]
#fnames      = ["flxeof_qek_%ieofs_SLAB-PIC_JJA" % x for x in neofs]
#fnames = ["flxeof_090pct_SLAB-PIC_eofcorr1"]
fnames = ["forcingflxeof_qek_50eofs_SLAB-PIC_1000yr_run005",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run007",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run004",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006"
          ]


# ## Compare types of forcing and effect of applying ampq
fnames = ["forcingflxstd_SLAB-PIC_1000yr_run006_ampq0",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
          "forcingflxstd_SLAB-PIC_1000yr_run006_ampq1",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1"]
#fnames      = ["flxeof_%ieofs_SLAB-PIC" % x for x in neofs]
#frcnamelong = ["%i EOFs" % x for x in neofs]
frcnamelong = ["50 EOFs (with $Q_{ek}$ and q-corr)",
               "90% Variance (ann avg. q-corr)",
               "90% Variance (monthly q-corr)",
               "90% Variance (no q-corr)"]

frcnamelong = ["var(Q)-based",
               "90% Variance",
               "var(Q)-based (q-corr)",
               "90% Variance (q-corr)"]
    #"90% Threshold (no-qcorr)"]


fnames = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
          "forcingflxeof_q-ek_090pct_SLAB-PIC_eofcorr1_1000yr_run009_ampq0",
          "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
          "forcingflxeof_q-ek_090pct_SLAB-PIC_eofcorr1_1000yr_run009_ampq1"]


frcnamelong = ["90% Variance",
               "90% Variance (q-ek)",
               "90% Variance (q-corr)",
               "90% Variance (q-ek and q-corr)"]

## Same as above, but now correcting locally for eof variance
fnames  = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq0",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq0",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
           "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1"
          ]

frcnamelong = ["Basinwide Correction",
           "Local Correction",
           "Basinwide Correction (with q-corr)",
           "Local Correction (with q-corr)"
            ]

## New Variance Correction Method
fnames  = ["forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run006_ampq1",
            "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run006_ampq1",
            "forcingflxeof_090pct_SLAB-PIC_eofcorr1_1000yr_run008_ampq3",
            "forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run008_ampq3",
          ]

frcnamelong = ["Basinwide Correction (Old)",
            "Local Correction (Old)",
            "Basinwide Correction (New)",
            "Local Correction (New)"
            ]
exoutnameraw = "new_v_old_q-correction"

#%% Functions
def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags

#%% User Edits

# Regional Analysis Settings
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
bbox_NNA = [-80,0 ,10,65]
regions = ("SPG","STG","TRO","NAT")#,"NNAT")        # Region Names
regionlong = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic (10N-65N)")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NNA) # Bounding Boxes
cint   = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int = np.arange(-0.45,0.50,0.05)
bboxplot = [-100,20,0,80]
modelnames  = ("Constant h","Vary h","Entraining")
mcolors     = ["red","blue","magenta"]

# CESM Names
cesmname   =  ["CESM-FULL","CESM-SLAB"]
cesmcolor  =  ["k","gray"]
cesmline   =  ["dashed","dotted"]

# Autocorrelation PLots
xtk2       = np.arange(0,37,2)
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
conf  = 0.95
tails = 2

#%% load some additional data

# Load lat/lon regional
lon = np.load(datpath+"lon.npy")
lat = np.load(datpath+"lat.npy")

# Load global lat/lon
lon180g,latg  = scm.load_latlon(rawpath)

#%% Load Autocorrelation

# Load for stochastic model experiments
sstacs  = []
kmonths = []
for f in range(len(fnames)):
    # Load the dictionary [h-const, h-vary, entrain]
    expid = fnames[f]
    rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,expid)
    ld = np.load(rsst_fn,allow_pickle=True)#.item()
    sstac   = ld['autocorr_region'].item()
    kmonth  = ld['kmonths'].item()
    sstacs.append(sstac)
    kmonths.append(kmonth)

# Load data for CESM1-PIC
cesmacs= []
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_Region_Autocorrelation_%s.npz" % (datpath,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)
cesmacs    = ldc['autocorr_region'].item() # Forcing x Region x Model

## Calculate Confidence Intervals -----------

#  Stochastic Model 
cfstoch = np.zeros([len(fnames),len(regions),3,len(lags),2]) # [Forcing x Region x Model x Lag x Upper/Lower]
n       = 1000
for f in range(len(fnames)): # Loop ny forcing
    for rid in range(len(regions)): # Loop by Region
        for mid in range(3): # Loop by Model
            inac                   = sstacs[f][rid][mid]
            cfs                    = calc_conflag(inac,conf,tails,n)
            cfstoch[f,rid,mid,:,:] = cfs.copy()

# CESM1
cfcesm = np.zeros((4,2,len(lags),2)) # [Region x Model x Lag x Upper/Lower]
ns     = [1798,898]
for rid in range(len(regions)):
    for mid in range(2):
        inac                = cesmacs[rid][mid]
        cfs                 = calc_conflag(inac,conf,tails,ns[mid])
        cfcesm[rid,mid,:,:] = cfs.copy()
#%% Make 4-panel plot comparing regional autocorrelation

plt.style.use("default")

for f in range(len(frcnamelong)):
    fig,axs = plt.subplots(2,2,figsize=(12,6))
    
    for rid in range(4):
        
        ax = axs.flatten()[rid]
        
        kmonth = kmonths[f][rid]
        
        
        # Plot some differences
        title      = "%s (Lag 0 = %s)" % (regions[rid],mons3[kmonth])
        ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
        #ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
        #ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')
        
    
        # Plot Each Model
        for mid in range(3):
            ax.plot(lags,sstacs[f][rid][mid],color=mcolors[mid],label=modelnames[mid])
            ax.fill_between(lags,cfstoch[f,rid,mid,lags,0],cfstoch[f,rid,mid,lags,1],
                            color=mcolors[mid],alpha=0.10)
            
        # Plot CESM
        for cid in range(2):
            ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],ls=cesmline[cid])
            ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
                            color=cesmcolor[cid],alpha=0.10)
            
        if rid == 0:
            ax.legend()
        if rid%2 == 1:
            ax.set_ylabel("")
        if rid<2:
            ax.set_xlabel("")
    
    title = "SST Autocorrelation (Forcing = 90 perc. Variance [%s])" % (frcnamelong[f])
    plt.suptitle(title,y=1.02)
    savename = "%sSST_Autocorrelation_Comparison_%s.png" % (figpath,fnames[f])
    plt.savefig(savename,dpi=150,bbox_inches='tight')


#%% Next load/plot the AMV Patterns

# Load for stochastic model experiments
amvpats  = []
amvids   = []
for f in range(len(fnames)):
    # Load the dictionary [h-const, h-vary, entrain]
    expid = fnames[f]
    rsst_fn = "%sproc/AMV_Region_%s.npz" % (datpath,expid)
    ld = np.load(rsst_fn,allow_pickle=True)#.item()
    
    amvidx = ld['amvidx_region'].item()
    amvpat = ld['amvpat_region'].item()
    
    amvpats.append(amvpat)
    amvids.append(amvidx)

# Load data for CESM1-PIC
cesmacs= []
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/AMV_Region_%s.npz" % (datpath,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)
cesmpat = ldc['amvpat_region'].item()
cesmidx = ldc['amvidx_region'].item()

#%% Plot AMV Patterns
for f in tqdm(range(len(fnames))):
    for rid in range(4):
        fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,6))
        
        # Plot Stochastic Model Output
        for mid in range(3):
            ax = axs.flatten()[mid]
            
            # Set Labels, Axis, Coastline
            if mid == 0:
                blabel = [1,0,0,0]
            elif mid == 1:
                blabel = [0,0,0,1]
            else:
                blabel = [0,0,0,0]
            
            # Make the Plot
            ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel)
            pcm = ax.contourf(lon,lat,amvpats[f][rid][mid].T,levels=cint,cmap=cmocean.cm.balance)
            ax.pcolormesh(lon,lat,amvpats[f][rid][mid].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
            cl = ax.contour(lon,lat,amvpats[f][rid][mid].T,levels=cl_int,colors="k",linewidths=0.5)
            ax.clabel(cl,levels=cl_int,fontsize=8)
            ax.set_title("%s [var(AMV) = %f]"%(modelnames[mid],np.var(amvids[f][rid][mid])))
            
        # Plot CESM1
        axs[1,1].axis('off')
        
        for cid in range(2):
            if cid == 0:
                ax = axs[1,2]
                blabel = [0,0,0,1]
            else:
                ax = axs[1,0]
                blabel = [1,0,0,1]
                
            # Make the Plot
            ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel)
            pcm = ax.contourf(lon180g,latg,cesmpat[rid][cid].T,levels=cint,cmap=cmocean.cm.balance)
            ax.pcolormesh(lon180g,latg,cesmpat[rid][cid].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
            cl = ax.contour(lon180g,latg,cesmpat[rid][cid].T,levels=cl_int,colors="k",linewidths=0.5)
            ax.clabel(cl,levels=cl_int,fontsize=8)
            ax.set_title("%s [var(AMV) = %f]"%(cesmname[cid],np.var(cesmidx[rid][cid])))
        
        fig.colorbar(pcm,ax=axs[1,1],orientation='horizontal')
        plt.suptitle("%s AMV Pattern and Index Variance [Forcing = %s]" % (regionlong[rid],frcnamelong[f]),fontsize=14)
        savename = "%sSST_AMVPattern_Comparison_region%s_%s.png" % (figpath,regions[rid],fnames[f])
        plt.savefig(savename,dpi=150,bbox_inches='tight')



#%% Load Stochastic Model Output (Regional SSTs)

# Load in SSTs for each region
sstdicts = []
for f in range(len(fnames)):
    # Load the dictionary [h-const, h-vary, entrain]
    expid = fnames[f]
    rsst_fn = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,expid)
    sst = np.load(rsst_fn,allow_pickle=True).item()
    sstdicts.append(sst)

# Identify the variance for each region and load in numpy arrays
sstall  = np.zeros((len(fnames),len(regions),len(modelnames),nyrs*12)) # Forcing x Region x Model x Time
sstvars = np.zeros((len(fnames),len(regions),len(modelnames))) # Forcing x Region x Model

# An unfortunate nested loop... 
for fid in range(len(fnames)):
    for rid in range(len(regions)):
        for model in range(len(modelnames)):
            sstin  = sstdicts[fid][rid][model]
            sstvar = np.var(sstin)
            print("Variance for forcing %s, region %s, model %s is %f" % (fnames[fid],regions[rid],modelnames[model],sstvar))
            
            sstall[fid,rid,model,:] = sstin.copy()
            sstvars[fid,rid,model]   = sstvar

#%% Load corresponding CESM Data

expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,expid)
sstcesm    = np.load(rsst_fn,allow_pickle=True).item()
cesmname   =  ["CESM-FULL","CESM-SLAB"]

# Identify the variance for each region and load in numpy arrays
#sstallcesm  = np.zeros((len(regions),2,nyrs*12)) # Forcing x Region x Model x Time
sstvarscesm = np.zeros((len(regions),2)) # Forcing x Region x Model

for rid in range(len(regions)-1):
    for model in range(len(cesmname)):
        sstin  = sstcesm[rid][model]
        sstvar = np.var(sstin)
        print("Variance for region %s, model %s is %f" % (regions[rid],cesmname[model],sstvar))
        sstvarscesm[rid,model]   = sstvar

#%% Do some spectral analysis
nsmooth = 100
pct     = 0.10
rid     = 2
fid     = 3
dofs    = [1000,1000,898,1798] # In number of years

# Unpack and stack data
insst = []
for model in range(len(modelnames)):
    insst.append(sstall[fid,rid,model,:]) # Append each stochastic model result
    #print(np.var(sstall[fid,rid,model,:]))
insst.append(sstcesm[rid][0]) # Append CESM-FULL
insst.append(sstcesm[rid][1]) # Append CESM-SLAB 

insstvars = []
for s in insst:
    insstvars.append(np.var(s))
    #print(np.var(s))

# Calculate Spectra and confidence Intervals
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct)
alpha = 0.05
bnds = []
for nu in dofs:
    lower,upper = tbx.confid(alpha,nu*2)
    bnds.append([lower,upper])
    

# Loop for each region
specsall = [] # array[forcing][region][model]
freqsall = []
Cfsall   = []
bndsall  = []
sstvarall = []
for f in range(len(fnames)):
    ss = []
    ff = []
    cc = []
    bb = []
    vv = []
    for rid in range(len(regions)-1):
        
        # Get SSTs
        insst = []
        for mid in range(len(modelnames)):
            # Append each stochastic model result
            insst.append(sstall[f,rid,mid,:])
        insst.append(sstcesm[rid][0])
        insst.append(sstcesm[rid][1])
        
        # Calculate the variance
        insstvars = []
        for s in insst:
            insstvars.append(np.var(s))
        
        # Calculate Spectra and confidence Intervals
        specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct)
        alpha = 0.05
        bnds = []
        for nu in dofs:
            lower,upper = tbx.confid(alpha,nu*2)
            bnds.append([lower,upper])
        
        ss.append(specs)
        ff.append(freqs)
        cc.append(CCs)
        bb.append(bnds)
        vv.append(insstvars)
    
    specsall.append(ss)
    freqsall.append(ff)
    Cfsall.append(cc)
    bndsall.append(bb)
    sstvarall.append(vv)




#%% Make the plot (Frequency x Power)
timemax = None
xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
dt   = 3600*24*30
speccolors = ["r","b","m","k","gray"]
specnames  = np.hstack([modelnames,cesmname])
#speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[i]) for i in range(len(insst)) ]

for f in tqdm(range(len(frcnamelong))): 
        fig,axs = plt.subplots(2,2,figsize=(16,8))
        for rid in range(4):
            
            
            
            for model in range(len(modelnames)):
                speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[f][rid][model]) for i in range(len(modelnames)) ]
               
                print(speclabels)
            ax    = axs.flatten()[rid]
            ax = viz.plot_freqxpower(specsall[f][rid],freqsall[f][rid],speclabels,speccolors,
                                 ax=ax,plottitle=regionlong[rid])
            
            if rid <2:
                ax.set_xlabel("")
            if rid%2 == 1:
                ax.set_ylabel("")
        plt.suptitle("Regional AMV Index Spectra (unsmoothed, Forcing=%s)"%(frcnamelong[f]))
        savename = "%sSST_Spectra_Comparison_%s_model%i.png" % (figpath,fnames[f],mid)
        plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make plot (Linear-Linear Multidecadal)

for f in tqdm(range(len(frcnamelong))):
    fig,axs = plt.subplots(2,2,figsize=(16,8))
    for rid in range(4):
        
        
        speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[f][rid][i]) for i in range(len(insst)) ]
        ax    = axs.flatten()[rid]
        ax = viz.plot_freqlin(specsall[f][rid],freqsall[f][rid],speclabels,speccolors,
                             ax=ax,plottitle=regionlong[rid])
        
        if rid <2:
            ax.set_xlabel("")
        if rid%2 == 1:
            ax.set_ylabel("")
    plt.suptitle("Regional AMV Index Spectra (unsmoothed, Forcing=%s)"%(frcnamelong[f]))
    savename = "%sSST_Spectra_Comparison_%s_Linear-Decadal.png" % (figpath,fnames[f])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Make plot (Log-Log)

for f in tqdm(range(len(frcnamelong))):
    fig,axs = plt.subplots(2,2,figsize=(16,8))
    for rid in range(4):
        
        
        speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[f][rid][i]) for i in range(len(insst)) ]
        ax    = axs.flatten()[rid]
        ax = viz.plot_freqlog(specsall[f][rid],freqsall[f][rid],speclabels,speccolors,
                             ax=ax,plottitle=regionlong[rid])
        
        if rid <2:
            ax.set_xlabel("")
        if rid%2 == 1:
            ax.set_ylabel("")
    plt.suptitle("Regional AMV Index Spectra (unsmoothed, Forcing=%s)"%(frcnamelong[f]))
    savename = "%sSST_Spectra_Comparison_%s_Log-Log.png" % (figpath,fnames[f])
    plt.savefig(savename,dpi=150,bbox_inches='tight')
