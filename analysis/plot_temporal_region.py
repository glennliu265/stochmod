#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize temporal metrics (autocorrelation, power spectra)
for regions of a selected output. 

Modified from viz_AMV_comparison.py.

This includes plots that will be used for the SM manuscript

Created on Sat Dec 18 12:24:53 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
from time import time

import matplotlib.patheffects as PathEffects

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20230512/"
   
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

#%% Experimental Configurations

mconfig    = "SLAB_PIC"
nyrs       = 1000        # Number of years to integrate over
continuous = True # Set to True to Load a continuous run with the lines below


if continuous:
    # Do a continuous Run
    # -------------------
    fnames   = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
    frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
    exname   = "Fprime_amq0_method5_cont"
else:
    # Options to determine the experiment ID (Old Format)
    # --------------------------------------------------
    runid      = "011"
    
    # Indicate the experiment file name, display name (for plotting), and output save name
    fname       = 'forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method4_dmp0'
    frcnamelong = "EOF Forcing (90% Variance)"
    exname      ="90perc_variance"

#%% Other Settings

# Analysis Options
lags = np.arange(0,37,1)

# Set the Visualization mode
darkmode   = False
if darkmode:
    plt.style.use("dark_background")
    dfcol = "w"
else:
    plt.style.use("default")
    dfcol = "k"

# Calculate in place (or load output from sm_postprocess_output)
calc_inplace=False

pubready = False

#%% Labels and Plotting

# # Regional Analysis Setting (NEW, with STG Split)
# Regional Analysis Settings
bbox_SP     = [-60,-15,40,65]
bbox_ST     = [-80,-10,20,40]
bbox_TR     = [-75,-15,10,20]
bbox_NA     = [-80,0 ,0,65]
bbox_NA_new = [-80,0,10,65]
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
bbox_NA_et  = [-80,0,20,60]
regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw","eNAT")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w,bbox_NA_et) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic",
               "North Atlantic","Subtropical (East)","Subtropical (West)",
               "Extratropical North Atlantic")
bbcol       = ["Blue","Red","Yellow","Black","Black"]
bbcol       = ["cornflowerblue","Red","Yellow","Black","Black","limegreen","indigo","Black"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted",'dashed']

# AMV Contours
cint        = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int      = np.arange(-0.45,0.50,0.05)

# SM Names and colors
modelnames  = ("Vary $F'$ and $\lambda_a$",
                    "Vary $F'$, $h$, and $\lambda_a$",
                    "Entraining")
mcolors     = ["red","magenta","orange"]
mlines      = ["solid","dotted","dashed"]


# CESM Names
cesmname    =  ["CESM-FULL","CESM-SLAB"]
cesmcolor   =  [dfcol,"gray"]
cesmline    =  ["dashed","dotted"]

# Autocorrelation Plot parameters
xtk2        = np.arange(0,37,2)
mons3       = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
conf        = 0.95
tails       = 2
alw         = 3

# Power Spectra Options
speccolors = ["r","magenta","Orange","k","gray"]
specnames  = np.hstack([modelnames,cesmname])


# Linear-Power Spectra, < 2-yr
xlm   = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper  = np.array([100,50,20,10,5,2])
xtks  = 1/xper
xlm   = [xtks[0],xtks[-1]]


#%% Silly functions to repackage postprocessed output (need to stop using dicts)

def unpack_smdict(indict):
    """
    Takes a dict of [run][region][models][OTHERDIMS] and unpacks it into
    an array [unpackaged]

    """
    # Get "Outer Shell" dimensions
    nrun    = len(indict)
    nregion = len(indict[0])
    nmodels = len(indict[0][0])
    
    # For Autocorrelation
    otherdims = indict[0][0][0].shape
    print("Found... Runs (%i) Regions (%i) ; Models (%i) ; Otherdims (%s)" % (nrun,nregion,nmodels,str(otherdims)))
    
    # Preallocate
    newshape = np.concatenate([[nrun,nregion,nmodels],otherdims])
    unpacked = np.zeros(newshape) * np.nan
    
    # Loop thru dict
    for run in range(nrun):
        for reg in range(nregion):
            for mod in range(nmodels):
                unpacked[run,reg,mod,:] = indict[run][reg][mod]
    return unpacked

def repack_smdict(inarr,nregion,nmodels):
    """
    Repackages a numpy array of inarr[region x model x otherdims] to 
    outdict{region}{model}
    """
    outdict = {}
    for reg in range(nregion):
        indict = {}
        for mod in range(nmodels):
            indict[mod] = inarr[reg,mod,:]
        outdict[reg] = indict.copy()
    return outdict



#%% load some additional data

# Load lat/lon regional

# Get one of the outputs
if continuous:
    fname = fnames[0] # Take first one to load lat/lon
ldname = "%sstoch_output_%s.npz" % (datpath,fname)
if exname == "numEOFs":
    ldname = ldname.replace("50","2")
ld     = np.load(ldname,allow_pickle=True)
lon    = ld['lon']
lat    = ld['lat']


# Load global lat/lon
lon180g,latg  = scm.load_latlon(rawpath)

# -------------------------------------------
#%% Autocorrelation 
# -------------------------------------------

if ~calc_inplace:
    
    # Load for stochastic model experiments
    # -------------------------------------
    if continuous: # Load for each run
        
        sstac   = [] # [run][region][model][lag]
        kmonths = [] 
        for f,fname in enumerate(fnames):
            rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,fname)
            ld = np.load(rsst_fn,allow_pickle=True)#.item()
            sstac.append(ld['autocorr_region'].item()) # I think its [region][model][lag]
            kmonths.append(ld['kmonths'].item())
        kmonths = kmonths[0] # Just take the first
        
        
        # Extract the region and take the average
        nrun    = len(sstac)
        nregion = len(sstac[0])
        nmodels = len(sstac[0][0])
        nlags   = len(sstac[0][0][0])
        sstac_rearr = unpack_smdict(sstac)
        sstac_avg   = sstac_rearr.mean(0) # [region x model x lag]

        # Repack as dict
        sstac       = repack_smdict(sstac_avg,nregion,nmodels)
        
    else:
        rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,fname)
        ld = np.load(rsst_fn,allow_pickle=True)#.item()
        sstac   = ld['autocorr_region'].item() # I think its [region][model][lag]
        kmonths  = ld['kmonths'].item()
    
    # Load data for CESM1-PIC
    # -----------------------
    cesmacs    = []
    expid      = "CESM1-PIC"
    rsst_fn    = "%s/proc/SST_Region_Autocorrelation_%s_ensorem0.npz" % (datpath,expid)
    ldc        = np.load(rsst_fn,allow_pickle=True)
    cesmacs    = ldc['autocorr_region'].item() # [Region] x [Model]
    
    # Calculate Confidence internals -----------------------------------------
    
    
    # Stochastic Model
    cfstoch = np.zeros([len(regions),3,len(lags),2]) # [Region x Model x Lag x Upper/Lower]
    n       = 1000
    for rid in range(len(regions)): # Loop by Region
        for mid in range(3): # Loop by Model
            inac                   = sstac[rid][mid]
            cfs                    = proc.calc_conflag(inac,conf,tails,n)
            cfstoch[rid,mid,:,:] = cfs.copy()
            
    # CESM1
    cfcesm = np.zeros((len(regions),2,len(lags),2)) # [Region x Model x Lag x Upper/Lower]
    ns     = [1798,898]
    for rid in range(len(regions)):
        for mid in range(2):
            inac                = cesmacs[rid][mid]
            cfs                 = proc.calc_conflag(inac,conf,tails,ns[mid])
            cfcesm[rid,mid,:,:] = cfs.copy()

    
else:
    
    print("PLACEHOLDER: NEED TO WRITE CODE")
    
"""
Key Outputs ----

Autocorrelation:
sstac   [region][model][lag]
cesmacs [region][model][lag]

Deepest MLD Month Index:
kmonths [region]

Confidence Intervals:
cfstoch [Region x Model x Lag x Upper/Lower]
cfcesm  [Region x Model x Lag x Upper/Lower]
"""
    
#%% Plot Bounding Boxes (STG)Locator

cid = 0
rid = 4

bboxtemp = [-85,-5,15,45]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4.5,3))
ax = viz.add_coast_grid(ax,bboxtemp,fill_color='k')
fig.patch.set_alpha(1)  # solution

props = dict(boxstyle='square', facecolor='white', alpha=0.8)
ax.text(-63.5,48,"STG Bounding Boxes",ha='center',bbox=props,fontsize=12)
#ax.set_title("STG Bounding Boxes",fontsize=10)
# # 
ls = []
for bb in [5,6]:
    ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                          color=bbcol[bb],linestyle=bbsty[bb],linewidth=3,return_line=True)
    ls.append(ll)
    
ax.legend(ncol=2,fontsize=8)

plt.savefig("%sSTG_Locator.png"%figpath,dpi=100,bbox_inches='tight',transparent=True)


#%% Plot Bounding Boxes (SPG) over Locator

cid = 0
rid = 0

bboxtemp = [-90,5,0,65]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4.5,3))
ax = viz.add_coast_grid(ax,bboxtemp,fill_color='gray')
fig.patch.set_alpha(1)  # solution
# # Plot the amv pattern
props = dict(boxstyle='square', facecolor='white', alpha=0.8)


ax.text(-67,70,"Bounding Boxes",ha='center',bbox=props,fontsize=12) # (works for SPG Only)


# # 
ls = []
for bb in [0,4]:
    ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                          color=bbcol[bb],linestyle=bbsty[bb],linewidth=3,return_line=True)
    ls.append(ll)
    
ax.legend(ncol=2,fontsize=8)

plt.savefig("%sSPG-NAT_Locator.png"%figpath,dpi=100,bbox_inches='tight',transparent=True)

#%% Load Stochastic Model Output (Regional SSTs)

# set detrending option
detrend_cesm = True

# Load in SSTs for each region
# ----------------------------
if continuous: 
    # Load in regional SST for each run
    ssts = []
    for f,fname in enumerate(fnames):
        rsst_fn = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,fname)
        ssts.append(np.load(rsst_fn,allow_pickle=True).item())
    
    # Unpack it
    ssts_allrun = unpack_smdict(ssts) # [run x region x model x time]
    #sstall      = ssts_allrun.mean(0) # [region x model x time]
    #sstvar      = np.var(sstall,2)    # [region x model]
    
else:
    # Load the dictionary [h-const, h-vary, entrain]
    rsst_fn = "%s/proc/SST_RegionAvg_%s.npy" % (datpath,fname)
    sst = np.load(rsst_fn,allow_pickle=True).item()
    sstdict = sst
    
    # Identify the variance for each region and load in numpy arrays
    sstall  = np.zeros((len(regions),len(modelnames),nyrs*12)) # Region x Model x Time
    sstvars = np.zeros((len(regions),len(modelnames))) # Region x Model
    
    # An unfortunate nested loop... 
    for rid in range(len(regions)):
        for model in range(len(modelnames)):
            sstin  = sstdict[rid][model]
            sstvar = np.var(sstin)
            print("Variance for region %s, model %s is %f" % (regions[rid],modelnames[model],sstvar))
            
            sstall[rid,model,:] = sstin.copy()
            sstvars[rid,model]   = sstvar

#% Load corresponding CESM Data ------------------
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_RegionAvg_%s_ensorem0.npy" % (datpath,expid)
sstcesm    = np.load(rsst_fn,allow_pickle=True).item()
cesmname   =  ["CESM-FULL","CESM-SLAB"]

# Identify the variance for each region and load in numpy arrays
sstvarscesm = np.zeros((len(regions),2)) # Forcing x Region x Model

for rid in range(len(regions)):
    for model in range(len(cesmname)):
        sstin  = sstcesm[rid][model]
        if detrend_cesm:
            output,tsmodel,residual = proc.polyfit_1d(np.arange(sstin.shape[0]),sstin,4)
            sstin = sstin - tsmodel
        
        sstvar = np.var(sstin)
        print("Variance for region %s, model %s is %f" % (regions[rid],cesmname[model],sstvar))
        sstvarscesm[rid,model]   = sstvar
#%% Do some spectral analysis

ssmooth    = 30        # Stochastic Model Smoothing
cnsmooths  = [100,100] # CESM1 Smoothing
pct        = 0.10
alpha      = 0.05      # For CI Calculatuions

sdof       = 1000       # Degrees of freedom
cdofs      = [898,1798] #


use_ann    = False # Set to true to use annual data

smoothname = "smth-obs%03i-full%02i-slab%02i" % (ssmooth,cnsmooths[0],cnsmooths[1])


if continuous:

    # Loop for each region
    ss = []
    ff = []
    cc = []
    bb = []
    vv = []
    
    # Calculate for CESM and SM Separately
    for rid in range(len(regions)):
        
        # Preallocate arrays for average for each model
        sm_avgspec = []
        sm_avgCC   = []
        sm_avgr1   = [] 
        sm_avgvar  = []
        for mid in range(len(modelnames)):
            
            # First, Compute SM Spectra for each RUN
            # ---------------------------------------
            # Setup
            sst_sel = ssts_allrun[:,rid,mid,:] # [run x time (mon)]
            if use_ann:
                sst_sel = proc.ann_avg(sst_sel,1) # [run x year]
                dt = 3600*24*365 # Annual Data
            else:
                dt = None # Use Default
                
            insst = [sst_sel[run,:] for run in range(nrun)]
            
            # Calculate Spectra for all runs of a selected rid/model
            specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,ssmooth,pct,dt=dt)
            
            # Assign to temporary variable # smspecs_all [run,region,nmodels,nreqs]
            if rid == 0:
                nfreqs = specs[0].shape[0]
                smspecs_all = np.zeros((nrun,nregion,nmodels,nfreqs)) * np.nan
            for run in range(nrun):
                smspecs_all[:,rid,mid,:] = specs[run].copy()
            
            # Take average of SM
            # ------------------
            sm_avgspec.append(np.array(specs).mean(0))
            sm_avgCC.append(np.array(CCs).mean(0))
            sm_avgr1.append(np.mean(r1s))
            sm_avgvar.append(np.mean(np.var(sst_sel,1),0))
            
            
        # Compute for CESM
        # ----------------
        insst_cesm                    =  [sstcesm[rid][0],sstcesm[rid][1]]
        if detrend_cesm:
            for ii in range(2):
                sstin = insst_cesm[ii]
                output,tsmodel,residual = proc.polyfit_1d(np.arange(sstin.shape[0]),sstin,1)
                sst_dt = sstin - tsmodel
                insst_cesm[ii] = sst_dt
        if use_ann:
            insst_cesm = [proc.ann_avg(sst,0)[1:] for sst in insst_cesm] # Ann avg, drop 1st year b/c odd
        cspecs,cfreqs,cCCs,cdofs,cr1s = scm.quick_spectrum(insst_cesm,cnsmooths,pct,dt=dt)
        cvars = [np.var(csst) for csst in insst_cesm]
        
        
        # Combine with CESM
        # -----------------
        ss.append(sm_avgspec+cspecs)
        ff.append([freqs[0],freqs[0],freqs[0]]  +cfreqs)
        cc.append(sm_avgCC +cCCs)
        dofs.append([dofs[0],dofs[0],dofs[0]] +cdofs)
        r1s.append(sm_avgr1 + cr1s)
        vv.append(sm_avgvar + cvars)
        
        # Calculate Confidence Inervals
        # -----------------------------
        bnds = []
        for nu in dofs:
            lower,upper = tbx.confid(alpha,nu*2)
            bnds.append([lower,upper])
        bb.append(bnds)
else:
    nsmooth    = np.concatenate([np.ones(3)*ssmooth,np.ones(2)*cnsmooths])
    smoothname = "smth-obs%03i-full%02i-slab%02i" % (ssmooth,cnsmooths[0],cnsmooths[1])
    
    pct     = 0.10
    dofs    = [1000,1000,898,1798] # In number of years
    
    # Loop for each region
    ss = []
    ff = []
    cc = []
    bb = []
    vv = []
    for rid in range(len(regions)):
        
        # Get SSTs
        insst = []
        for mid in range(len(modelnames)):
            # Append each stochastic model result
            insst.append(sstall[rid,mid,:])
        insst.append(sstcesm[rid][0])
        insst.append(sstcesm[rid][1])
        if use_ann:
            insst = [proc.ann_avg(sst,0)[1:] for sst in insst]
            dt = 3600*24*365
        else:
            dt = None
        
        # Calculate the variance
        insstvars = []
        for s in insst:
            insstvars.append(np.var(s))
        
        # Calculate Spectra and confidence Intervals
        specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct,dt=None)
        
        
        
        bnds = []
        for nu in dofs:
            lower,upper = tbx.confid(alpha,nu*2)
            bnds.append([lower,upper])
        
        ss.append(specs)
        ff.append(freqs)
        cc.append(CCs)
        bb.append(bnds)
        vv.append(insstvars)
        
specsall  = ss # array[forcing][region][model]
freqsall  = ff
Cfsall    = cc
bndsall   = bb
sstvarall = vv

"""

Key Outputs

specsall  [region][model] spectra
freqsall  [region][model] frequencies
Cfsall    [region][model] confidence intervals (AR1 h0)
bndsall   [region][model] confidence bounds (chi2)
sstvarall [region][model] SST variances

"""

#%% Function to add subplot labels


#%% Now let's plot two regions side by side, with 
# Autocorr on top row
# Power Spectra on the bottom

alw = 3
exclude_consth = True # Set to true to NOT plot constant h model
notitle        = True

if continuous: # Number of spectra to plot
    nspecs = 5
else:
    nspec  = len(insst)


if exclude_consth:
    plotid = [1,2]
    plotidspec = [1,2,3,4] # Exclude constant h
else:
    plotid = [0,1,2]
    plotidspec = [0,1,2,3,4]

rid_L = 6 #  STGw
rid_R = 5 #  STGe
order = [rid_L,rid_R]
specylim = [0,0.40]

fig,axs =plt.subplots(2,2,figsize=(12,8))

sp_id = 0

# Plot the autocorrelation (top row)
for i in range(2):
    
    ax = axs[0,i]
    
    rid    = order[i] 
    kmonth = kmonths[rid]
    #title  = "%s Autocorrelation (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])

    title  = "%s Autocorrelation" % (regionlong[rid]) # No Month
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="")
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    # Plot Each Stochastic Model
    for mid in plotid:
        ax.plot(lags,sstac[rid][mid],color=mcolors[mid],label=modelnames[mid],lw=alw)
        ax.fill_between(lags,cfstoch[rid,mid,lags,0],cfstoch[rid,mid,lags,1],
                        color=mcolors[mid],alpha=0.10)
    
    # Plot CESM
    for cid in range(2):
        ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],ls=cesmline[cid],lw=alw)
        ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
                        color=cesmcolor[cid],alpha=0.10)
        
    if i == 1:
        ax.set_ylabel("")
        ax.legend(ncol=2,fontsize=12)
    
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1


# Plot the power spectra (bottom row)
for i in range(2):
    
    ax  = axs[1,i]
    rid = order[i]
    
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(nspecs) ]
    
    ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                         ax=ax,plottitle=regionlong[rid],
                         xlm=xlm,xtick=xtks,return_ax2=True,plotids=plotidspec)
    
    # Turn off title and second axis labels
    ax.set_title("")
    ax2.set_xlabel("")
    sxtk2 = ax2.get_xticklabels()
    sxtk2new = np.repeat("",len(sxtk2))
    ax2.set_xticklabels(sxtk2new)
    
    # Move period labels to ax1
    ax.set_xticklabels(1/xtks)
    ax.set_xlabel("Period (Years)")
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    if i == 1:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Power ($\degree C^2 /cpy$)")
    ax.set_ylim(specylim)
    
    title = "%s Power Spectra" % (regions[rid])
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1

plt.tight_layout()
plt.savefig("%sSTG_Autocorrelation_Spectra%s.png"%(figpath,smoothname),
            dpi=200,transparent=False)


#%% Same, but for North Atlantic
# Autocorr on top row
# Power Spectra on the bottom

alw = 3

rid_L = 0 #  SPG
rid_R = 4 #  NAT
order = [rid_L,rid_R]

nspecs = 2

fig,axs =plt.subplots(2,2,figsize=(12,8))

sp_id = 0

# Plot the autocorrelation (top row)
for i in range(2):
    
    ax = axs[0,i]
    
    rid    = order[i] 
    kmonth = kmonths[rid]
    #title  = "%s Autocorrelation (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])
    title  = "%s Autocorrelation" % (regionlong[rid]) # No Month
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="")
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    # Plot Each Stochastic Model
    for mid in range(3):
        ax.plot(lags,sstac[rid][mid],color=mcolors[mid],label=modelnames[mid],lw=alw)
        ax.fill_between(lags,cfstoch[rid,mid,lags,0],cfstoch[rid,mid,lags,1],
                        color=mcolors[mid],alpha=0.10)
    
    # Plot CESM
    for cid in range(2):
        ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],ls=cesmline[cid],lw=alw)
        ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
                        color=cesmcolor[cid],alpha=0.10)
        
    if i == 1:
        ax.set_ylabel("")
        ax.legend(ncol=2,fontsize=12)
        
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1


# Plot the power spectra (bottom row)
for i in range(2):
    
    ax  = axs[1,i]
    rid = order[i]
    
    if rid == 4:
        specylim = [0,0.30]
    elif rid == 0:
        specylim = [0,1] 
        
    
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(nspecs) ]
    
    
    ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                         ax=ax,plottitle=regionlong[rid],
                         xlm=xlm,xtick=xtks,return_ax2=True)
    
    # Turn off title and second axis labels
    ax.set_title("")
    ax2.set_xlabel("")
    sxtk2 = ax2.get_xticklabels()
    sxtk2new = np.repeat("",len(sxtk2))
    ax2.set_xticklabels(sxtk2new)
    
    # Move period labels to ax1
    ax.set_xticklabels(1/xtks)
    ax.set_xlabel("Period (Years)")
    plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
    if i == 1:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Power ($\degree C^2 /cpy$)")
    ax.set_ylim(specylim)
    
    
    
    title = "%s Power Spectra" % (regions[rid])
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1

plt.tight_layout()
plt.savefig("%sSPG-NAT_Autocorrelation_Spectra%s.png"%(figpath,smoothname),
            dpi=200,transparent=False)


#%% SM Draft 2, 3, Plot all Regionals in 1 Plot

"""
Updated for Revision 01 08/24/2022
"""

# Plotting Params
# ---------------
alw            = 3
exclude_consth = True # Set to true to NOT plot constant h model
notitle        = True
plotslab       = False # Set to True to plot the slab model simulation
plotlog        = False # Set to True to Plot in Log Scale
linearx        = 1 # Keep frequency axis linear, period axis marked (Works only with plotlog=False)
periodx        = False
useC           = True # Use Celsius in Labels
usegrid        = False
xtk2           = np.arange(0,39,3)
plotar1        = False
plotlegend     = True # Plot autocorrelation legend
yrticks        = [1/100,1/20,1/10,1/5]


axisfs         = 14 # Axis Label Fontsize

xtk2_labels = []
kmonth_seen = []
for t,tk in enumerate(xtk2):
    if tk%6 == 0:
        #print(tk)
        monlbl = [(kmonth+tk)%12]
        if monlbl in kmonth_seen:
            lbl = tk
        else:
            lbl = "%i\n %s" % (tk,mons3[(kmonth+tk)%12])
            #kmonth_seen.append(monlbl) # Uncomment this to only plot first feb/aug
        #print(lbl)
    else:
        lbl = ""
    xtk2_labels.append(lbl)

# Select what lines to plot, based on above toggles
if exclude_consth:
    plotid = [1,2]
    plotidspec = [1,2,3,4] # Exclude constant h
else:
    plotid = [0,1,2]
    plotidspec = [0,1,2,3,4]

if plotslab is False: # Remove slab simulation
    plotidspec.remove(4)
    specylim_spg = [0,0.42]
    specylim_stg = [0,0.175]
else:
    specylim_spg = [0,0.8]
    specylim_stg = [0,0.3]
rids  = [0,6,5,]
order = rids

# Do Plotting
# ---------------
if plotlegend:
    fig,axs =plt.subplots(2,3,figsize=(16,12))
else:
    fig,axs =plt.subplots(2,3,figsize=(16,8))
sp_id = 0
# Plot the autocorrelation (top row)
for i in range(len(rids)):
    
    ax = axs[0,i]
    
    rid    = order[i] 
    kmonth = kmonths[rid]
    #title  = "%s Autocorrelation (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])
    
    title  = "%s" % (regionlong[rid]) # No Month
    #ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="",usegrid=usegrid)
    ax.set_xlim([lags[0],lags[-1]])
    ax.set_xticks(xtk2)
    ax.set_xticklabels(xtk2_labels)
    
    ax.set_title(title,color=bbcol[rid],fontsize=16,fontweight="bold")
    # Plot Each Stochastic Model
    for mid in plotid:
        ax.plot(lags,sstac[rid][mid],color=mcolors[mid],label=modelnames[mid],lw=alw)
        ax.fill_between(lags,cfstoch[rid,mid,lags,0],cfstoch[rid,mid,lags,1],
                        color=mcolors[mid],alpha=0.10)
    
    # Plot CESM
    if plotslab:
        crange = 2
    else:
        crange = 1
    for cid in range(crange):
        ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],lw=alw)
        ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
                        color=cesmcolor[cid],alpha=0.10)
    
    if i >0: # Set ylabel to false for all plots except leftmost
        ax.set_ylabel("")
        ax.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel("Autocorrelation",fontsize=axisfs)
        
    if i == 1: # Add legend and x-label to middle plot
        if plotlegend:
            leg = ax.legend(ncol=3,fontsize=12,bbox_to_anchor=(0.7, -0.65, 0.5, 0.5),edgecolor="k")
        #ax.legend(ncol=2,fontsize=12,)
        ax.set_xlabel("Lags (Months)",fontsize=axisfs)
    else:
        ax.set_xlabel("")
    
    # x =-0.4, y=1.13
    ax = viz.label_sp(sp_id,ax=ax,fontsize=20,fig=fig,labelstyle="(%s)",case='lower',alpha=0)
    sp_id += 1
    


# Plot the power spectra (bottom row)
for i in range(len(rids)):
    
    ax  = axs[1,i]
    rid = order[i]
    
    if plotar1:
        conf_in = Cfsall[rid]
    else:
        conf_in = None
    
    #speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
    nspecs = len(specsall[rid])
    speclabels=["" for i in range(nspecs)]
    
    if plotlog:
        ax,ax2 = viz.plot_freqlog(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                             ax=ax,plottitle=regionlong[rid],
                             xlm=xlm,xtick=yrticks,return_ax2=True,
                             plotids=plotidspec,legend=False,usegrid=usegrid)
    else:
        ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                             ax=ax,plottitle=regionlong[rid],plotconf=conf_in,
                             xlm=xlm,xtick=yrticks,return_ax2=True,
                             plotids=plotidspec,legend=False,linearx=linearx,usegrid=usegrid)
        
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
    
    if i == 0:# Turn off y label except for leftmost plot
        if useC:
            ax.set_ylabel("Power Spectrum ($\degree C^2 /cpy$)",fontsize=axisfs)
        else:
            ax.set_ylabel("Power Spectrum ($K^2 /cpy$)",fontsize=axisfs)
        ax.set_ylim(specylim_spg)
    else:
        ax.set_ylabel("")
        ax.set_ylim(specylim_stg)
        
        
    if plotlog:
        ax.set_ylim([1e-2,1e0])
        if i == 1: # Turn off ylabels for middle plot
            ax.yaxis.set_ticklabels([])
        
    if i != 1:
        ax.set_xlabel("")
        ax2.set_xlabel("")
    else:
        ax.set_xlabel("Frequency (Cycles/Year)",fontsize=axisfs)
        ax2.set_xlabel("Period (Years)",fontsize=axisfs)
        
    if i == 2: # Just turn off for last STG plot
        ax.yaxis.set_ticklabels([])
    
    
    #title = "%s Power Spectra" % (regions[rid])
    title = ""
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1

plt.tight_layout()

if pubready:
    plt.savefig("%sFig07a_Regional_Autocorrelation_Spectra%s.png"%(figpath,smoothname),
                dpi=1200,transparent=False)
else:
    plt.savefig("%sRegional_Autocorrelation_Spectra%s.png"%(figpath,smoothname),
                dpi=200,transparent=False)

#%% Presentation for Model Hierarchies: Plot Each Region Separately

rids = [0, 6, 5]

nspecs = len(rids)


if darkmode:
    if "k" in speccolors:
        speccolors[speccolors.index('k')] = 'w'


sp_id = 0
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

#%% Plot the North Atlantic Power Spectra


rid = 7

# Plotting params
# ---------------
# Autocorrelation
xtk1 = np.arange(0,38,2)
xlm1 = [xtk1[0],xtk1[-1]]
xtk1_lbl = viz.prep_monlag_labels(1,xtk1,3,useblank=False)

# Spectra
dtplot = 3600*24*365
xper   = np.array([100,20,10,5,2])
xtk2   = 1/xper
xlm2   = [xtk2[0],xtk2[-1]]

# Select the Correct Spectra -------------------
inspec = specsall[rid]
infreq = freqsall[rid]
kmonth = kmonths[rid]
speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(inspec)) ]
mons   = proc.get_monstr()
bbin    = bboxes[rid]
bblabel = "Lon: %i to %i $\degree$E, Lat: %i to %i $\degree$N" %(bbin[0],bbin[1],bbin[2],bbin[3])

fig,axs = plt.subplots(2,1,figsize=(10,8),constrained_layout=True)

# ------------------------
# Plot the autocorrelation
ax = axs[0]

for s in range(3):
    ax.fill_between(lags,cfstoch[rid,s,lags,0],cfstoch[rid,s,lags,1],
                                     color=mcolors[s],alpha=0.10,zorder=9)
    ax.plot(sstac_avg[rid,s,:],color=mcolors[s],lw=3)
    
for c in range(2):
    ax.fill_between(lags,cfcesm[rid,c,lags,0],cfcesm[rid,c,lags,1],
                                     color=cesmcolor[c],alpha=0.10,zorder=9)
    ax.plot(cesmacs[rid][c],color=cesmcolor[c],lw=3)

ax.set_xlim(xlm1)
ax.set_xticks(xtk1)
ax.set_xlabel("Lag (Months)")
ax.set_xticklabels(xtk1_lbl)
ax.set_ylabel("Correlation")
ptitle = "%s Area-Averaged SST Autocorrelation \n %s Avg. (%s)" % (mons[kmonth],regions[rid],bblabel)
ax.set_title(ptitle)
ax.grid(True,ls='dotted')


# ----------------
# Plot the spectra
ax = axs[1]
for s in range(len(specnames)):
    ax.plot(infreq[s]*dtplot,inspec[s]/dtplot,color=speccolors[s],lw=3,label=speclabels[s])

#ax.set_xticks(xtk2)
ax.set_xlim(xlm2)
#ax2 = viz.twin_freqaxis(ax,infreq[s],"Years",dtplot,mode='lin-lin',xtick=xtk2)
ax2 = ax.twiny()
ax2.set_xticks(xtk2)
ax2.set_xticklabels(xper)
ax2.set_xlim(xlm2)
ax2.set_xlabel("Period (Years)")
ax.set_xlabel("Frequency ($year^{-1}$)")
ax.set_ylabel("Power ($\degree C^2 \, year^{-1}$)")
ax2.grid(True,ls='dotted')

ax.legend()

# Add an inset
# ------------
zoomrng = 10

bbplot = [bbin[0]-zoomrng,bbin[1]+zoomrng,bbin[2]-zoomrng,bbin[3]+zoomrng]
left, bottom, width, height = [0.75, 0.66, 0.20, 0.35]
axin = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
axin = viz.add_coast_grid(axin,bbox=bbplot,fill_color='gray',
                          fix_lon=[bbin[0],bbin[1]],
                          fix_lat=[bbin[2],bbin[3]])
axin.set_facecolor('lightblue')
axin = viz.plot_box(bbin,ax=axin,color=bbcol[rid],proj=ccrs.PlateCarree(),
                    linewidth=4,linestyle='dashed',)

#pcm = ax2.contourf(lonr,latr,amvpats_avg.T,levels=cint,cmap='cmo.balance')
#ax2.plot(lonr2[klon],latr2[klat],marker="x",color="k",markersize=10)

plt.savefig("%sCESM_v_SM_ac_spectra_%s.png"% (figpath,regions[rid]),dpi=150)

#%% Make the Corresponding Bounding Box


"""
NOTE: This is now outdated

See viz_inputs_point for the updated script, which uses the continuous AMV
Pattern!
"""

pointonly = False

cid      = 0
rids     = [0,6,5,]
bboxtemp = [-85,-5,15,68]
cint     = np.arange(-0.45,0.50,0.05)
plotamv  = True # Add AMV Plot as backdrop (False=WhiteBackdrop)

fix_lon  = [-80,-40,0]
fix_lat  = [20,40,65]

# Load the CESM1-FULL AMV Pattern
# Load data for CESM1-PIC
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/AMV_Region_%s_ensorem0.npz" % (datpath,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)
cesmpat = ldc['amvpat_region'].item()
cesmidx = ldc['amvidx_region'].item()
long,latg = scm.load_latlon()

# Start the PLot
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(3,2))
ax = viz.add_coast_grid(ax,bboxtemp,fill_color='gray',ignore_error=True,fix_lon=fix_lon,fix_lat=fix_lat)
if plotamv:
    pcm = ax.contourf(long,latg,cesmpat[4][0].T,cmap='cmo.balance',levels=cint)
fig.patch.set_alpha(1)  # solution
ax.plot(-30,50,marker="*",color='yellow',markersize=7.5,markeredgecolor='k',markeredgewidth=.4)
if pointonly: # Just plot the point

    

    plt.savefig("%sRegional_BBOX_Locator_pointonly.png"%figpath,dpi=100,bbox_inches='tight',transparent=True)
else:
    
    # # Plot the amv pattern
    props = dict(boxstyle='square', facecolor='white', alpha=0.8)
    
    # Add text and background highlights
    txtspg  = ax.text(-38,50,"SPG",ha='center',fontsize=15,weight='bold',color="k") 
    txtstgw = ax.text(-60,27,"STGw",ha='center',fontsize=15,weight='bold',color="k") 
    txtstge = ax.text(-25,27,"STGe",ha='center',fontsize=15,weight='bold',color="k") 
    for txt in [txtspg,txtstgw,txtstge]:
        txt.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='w')])
    
    # First Plot Solid lines below
    for bb in rids:
        ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                              color=bbcol[bb],linestyle="solid",linewidth=3,return_line=True)
    
    # Then plot dashed lines above
    ls = []
    for bb in rids:
        
        ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                              color=bbcol[bb],linestyle="dotted",linewidth=3,return_line=True)
        ls.append(ll)
    
    
        plt.savefig("%sRegional_BBOX_Locator.png"%figpath,dpi=100,bbox_inches='tight',transparent=True)

#%% Plot ratios of SLAB and FULL

recip   = True  # True for SLAB/FULL, False for FULL/SLAB
rid_sel = [0,5,6]
dtplot  = 3600*24*365
debug  = False
plotlog  = True

# Interpolate the data
freqslab   = freqsall[0][-1]
freqfull   = freqsall[0][-2]

if debug:
    interpfull = np.interp(freqslab,freqfull,specsall[0][-2])
    fig,ax = plt.subplots(1,1)
    ax.loglog(freqslab*dtplot,interpfull/dtplot,label="Interp")
    ax.loglog(freqfull*dtplot,specsall[0][-2]/dtplot,ls='dashed',label="Original")
    ax.set_xlim([1e-2,1e0])


fig,ax = plt.subplots(1,1)
for r in rid_sel:
    
    # Interpolate FULL to SLAB frequencies
    specfull        = specsall[r][-2]
    specslab        = specsall[r][-1]
    specfull_interp = np.interp(freqslab,freqfull,specfull)
    
    # Compute the ratio
    if recip:
        specratio = specslab/specfull_interp
        ylab = "SLAB/FULL"
    else:
        specratio = specfull_interp/specslab
        ylab = "FULL/SLAB"
    
    if plotlog:
        specratio = np.log(specratio)
        ax.set_ylabel("$log$(%s)" % ylab)
    
    ln = ax.semilogx(freqslab*dtplot,specratio,label=regions[r],lw=4,color=bbcol[r])
ax.legend()
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xticks(xtks)
ax.set_xticklabels(xper)
ax.set_xlabel("Period (Years)")
ax.axhline(0,ls='dashed',color="k")
ax.grid(True,ls='dotted')
ax.set_ylim([-2.25,2.25])
#ax.set_title("Ratio of Regional SST Spectra")
    #ratio = specs[]
    
    
#%%
    
spec_num  = 2 # Number in specsall[r]
ref       = 'FULL' # [SLAB or FULL or ref_num]
ref_num   = 0
plotlog   = True
recip     = False

# Interpolate the data
freqslab   = freqsall[0][-1]
freqfull   = freqsall[0][-2]
freqsm     = freqsall[0][0]

if debug:
    interpfull = np.interp(freqslab,freqfull,specsall[0][-2])
    fig,ax = plt.subplots(1,1)
    ax.loglog(freqslab*dtplot,interpfull/dtplot,label="Interp")
    ax.loglog(freqfull*dtplot,specsall[0][-2]/dtplot,ls='dashed',label="Original")
    ax.set_xlim([1e-2,1e0])


fig,ax = plt.subplots(1,1,figsize=(6,4),constrained_layout=True)
for r in rid_sel:
    
    # Interpolate to Reference Frequency
    if ref == 'FULL':
        reffreq = freqfull
        refspec = specsall[r][-2]
    elif ref == 'SLAB':
        reffreq = freqslab
        refspec = specsall[r][-1]
    else:
        reffreq = freqsm
        refspec = specsall[r][ref_num]
        ref     = specnames[ref_num]
        
    
    # Interpolate selected spectra to reference frequencies
    spec_interp = np.interp(reffreq,freqsall[r][spec_num],specsall[r][spec_num])
    
    # Compute the ratio
    if recip:
        specratio = refspec/spec_interp
        ylab = "%s/%s" % (ref,specnames[spec_num])
        flab = "%s-%s" % (ref,specnames[spec_num])
    else:
        specratio = spec_interp/refspec
        ylab = "%s/%s" % (specnames[spec_num],ref)
        flab = "%s-%s" % (ref,specnames[spec_num])
        
    if plotlog:
        specratio = np.log(specratio)
        ax.set_ylabel("$log$(%s)" % ylab)
    
    ln = ax.semilogx(reffreq*dtplot,specratio,label=regions[r],lw=4,color=bbcol[r])
ax.legend()
ax.set_xlim([xtks[0],xtks[-1]])
ax.set_xticks(xtks)
ax.set_xticklabels(xper)
ax.set_xlabel("Period (Years)")
ax.axhline(0,ls='dashed',color="k")
ax.grid(True,ls='dotted')
ax.set_ylim([-2.25,2.25])


plt.savefig("%sRegional_Spectra_Ratio_%s.png"%(figpath,flab),dpi=150,bbox_inche='tight')

#%% Make Multiple Panels

spec_num  = 2
ref       = 'FULL' # [SLAB or FULL or ref_num]
ref_num   = 0
plotlog   = True
recip     = False

# Interpolate the data
freqslab   = freqsall[0][-1]
freqfull   = freqsall[0][-2]
freqsm     = freqsall[0][0]

if debug:
    interpfull = np.interp(freqslab,freqfull,specsall[0][-2])
    fig,ax = plt.subplots(1,1)
    ax.loglog(freqslab*dtplot,interpfull/dtplot,label="Interp")
    ax.loglog(freqfull*dtplot,specsall[0][-2]/dtplot,ls='dashed',label="Original")
    ax.set_xlim([1e-2,1e0])

spec_nums = [-1,2]
refs      = ['FULL','FULL']
ref_nums  = [0,1]
refnames  = ("CESM-FULL","CESM-FULL")
#specnames = ("CESM-SLAB","Entraining")
fig,axs = plt.subplots(2,1,figsize=(6,6),constrained_layout=True,sharex=True)
spid = 0
for a in range(2):
    
    ax = axs[a]
    
    spec_num = spec_nums[a]
    ref      = refs[a]
    ref_num  = ref_nums[a]
    
    
    print(a)
    
    for r in rid_sel:
        print(r)
        
        # Interpolate to Reference Frequency
        if ref == 'FULL':
            reffreq = freqfull
            refspec = specsall[r][-2]
        elif ref == 'SLAB':
            reffreq = freqslab
            refspec = specsall[r][-1]
        else:
            reffreq = freqsm
            refspec = specsall[r][ref_num]
            ref     = refnames[a] #specnames[ref_num]
            
        
        # Interpolate selected spectra to reference frequencies
        spec_interp = np.interp(reffreq,freqsall[r][spec_num],specsall[r][spec_num])
        
        
        # Compute the ratio
        if recip:
            specratio = refspec/spec_interp
            ylab = "%s / %s" % (ref,specnames[spec_num])
            flab = "%s-%s" % (ref,specnames[spec_num])
        else:
            specratio = spec_interp/refspec
            ylab = "%s / %s" % (specnames[spec_num],refnames[a])
            flab = "%s-%s" % (refnames[a],specnames[spec_num])
            
        if plotlog:
            specratio = np.log(specratio)
            ax.set_ylabel("%s" % ylab)
        
        ln = ax.semilogx(reffreq*dtplot,specratio,label=regions[r],lw=4,color=bbcol[r])
    if a == 0:
        ax.legend(ncol=3,loc="lower center")
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Period (Years)")
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    
    ax.axhline(0,ls='dashed',color="k")
    ax.grid(True,ls='dotted',which='major',color='gray')
    ax.grid(True,ls='dotted',which='minor',alpha=0.5,color='gray',lw=0.5)
    ax.set_ylim([-2.25,2.25])
    
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
    spid += 1

fig.supylabel("Log Ratio")
#txt = viz.add_ylabel("Log Ratio",ax=axs.flatten())

if pubready:
    plt.savefig("%sFig10_Regional_Spectra_Ratio.eps"%(figpath),dpi=1200,bbox_inche='tight',format='eps')
else:
    plt.savefig("%sRegional_Spectra_Ratio_2-panel_%s.png"%(figpath,flab),dpi=150,bbox_inche='tight')

#%%
# #%% Remake using subplot grids (Note Working)


# rowtitles = ("Autocorrelation","Power Spectra")

# fig = plt.figure(constrained_layout=True,figsize=(12,8))
# fig.suptitle("Subtropical Gyre West (Left) and East (Right)",fontsize=14)

# subfigs = fig.subfigures(nrows=2,ncols=1)
# for row,subfig in enumerate(subfigs):
#     # Create 1x4 subplots per subfig
#     axs = subfig.subplots(nrows=1, ncols=2)
#     subfig.suptitle(rowtitles[row],fontsize=12)
    
#     # Plot Autocorrelation
#     if row == 0:
#         for i, ax in enumerate(axs):
            
#             rid    = order[i] 
#             kmonth = kmonths[rid]
#             title  = "%s (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])
#             ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
            
#             # Plot Each Stochastic Model
#             for mid in range(3):
#                 ax.plot(lags,sstac[rid][mid],color=mcolors[mid],label=modelnames[mid],lw=alw)
#                 ax.fill_between(lags,cfstoch[rid,mid,lags,0],cfstoch[rid,mid,lags,1],
#                                 color=mcolors[mid],alpha=0.10)
            
#             # Plot CESM
#             for cid in range(2):
#                 ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],ls=cesmline[cid],lw=alw)
#                 ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
#                                 color=cesmcolor[cid],alpha=0.10)
                
#             if i == 1:
#                 ax.set_ylabel("")
#                 ax.legend(ncol=2,fontsize=12)
    
#     # Plot Power Spectra
#     elif row == 1: 
        
#         for i,ax in enumerate(axs):
#             speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
            
#             ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
#                                  ax=ax,plottitle="",
#                                  xlm=xlm,xtick=xtks,return_ax2=True)
            
#             # Turn off title and second axis labels
#             ax.set_title("")
#             ax2.set_xlabel("")
#             sxtk2 = ax2.get_xticklabels()
#             sxtk2new = np.repeat("",len(sxtk2))
#             ax2.set_xticklabels(sxtk2new)
            
#             # Move period labels to ax1
#             ax.set_xticklabels(1/xtks)
#             ax.set_xlabel("Period (Years)")
#             plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
#             if i == 1:
#                 ax.set_ylabel("")
#             ax.set_ylim(specylim)
            
    

# plt.show()

#%% Appendix Plot: North Atlantic SST

fig = plt.figure()

