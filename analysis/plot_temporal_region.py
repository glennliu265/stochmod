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
#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20220128/"
   
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

# Analysis Options
lags = np.arange(0,37,1)

# Options to determine the experiment ID
mconfig   = "SLAB_PIC"
nyrs      = 1000        # Number of years to integrate over
runid     = "011"
darkmode   = False
if darkmode:
    plt.style.use("dark_background")
    dfcol = "w"
else:
    plt.style.use("default")
    dfcol = "k"


# Indicate the experiment file name, display name (for plotting), and output save name
fname = 'forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run011_ampq3_method4_dmp0'
frcnamelong = "EOF Forcing (90% Variance)"
exname ="90perc_variance"

# Calculate in place (or load output from sm_postprocess_output)
calc_inplace=False

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
regions     = ("SPG","STG","TRO","NAT","NNAT","STGe","STGw")        # Region Names
bboxes      = (bbox_SP,bbox_ST,bbox_TR,bbox_NA,bbox_NA_new,bbox_ST_e,bbox_ST_w) # Bounding Boxes
regionlong  = ("Subpolar","Subtropical","Tropical","North Atlantic","North Atlantic","Subtropical (East)","Subtropical (West)",)
bbcol       = ["Blue","Red","Yellow","Black","Black"]
bbcol       = ["Blue","Red","Yellow","Black","Black","magenta","red"]
bbsty       = ["solid","dashed","solid","dotted","dotted","dashed","dotted"]

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


#%% load some additional data

# Load lat/lon regional
# Get one of the outputs
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
    rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,fname)
    ld = np.load(rsst_fn,allow_pickle=True)#.item()
    sstac   = ld['autocorr_region'].item() # I think its [region][model][lag]
    kmonths  = ld['kmonths'].item()
    
    # Load data for CESM1-PIC
    cesmacs= []
    expid      = "CESM1-PIC"
    rsst_fn    = "%s/proc/SST_Region_Autocorrelation_%s_ensorem0.npz" % (datpath,expid)
    ldc        = np.load(rsst_fn,allow_pickle=True)
    cesmacs    = ldc['autocorr_region'].item() # [Region] x [Model]
    
    # Calculate Confidence internals ----
    
    
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

# Load in SSTs for each region
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

for rid in range(len(regions)-1):
    for model in range(len(cesmname)):
        sstin  = sstcesm[rid][model]
        sstvar = np.var(sstin)
        print("Variance for region %s, model %s is %f" % (regions[rid],cesmname[model],sstvar))
        sstvarscesm[rid,model]   = sstvar
#%% Do some spectral analysis

ssmooth    = 65
cnsmooths  = [75,65]
pct        = 0.10
nsmooth    = np.concatenate([np.ones(3)*ssmooth,np.ones(2)*cnsmooths])
smoothname = "smth-obs%03i-full%02i-slab%02i" % (ssmooth,cnsmooths[0],cnsmooths[1])


pct     = 0.10
dofs    = [1000,1000,898,1798] # In number of years

# # Unpack and stack data (DELETE This section, I think it's a holdover)
# insst = []
# for model in range(len(modelnames)):
#     insst.append(sstall[rid,model,:]) # Append each stochastic model result
#     #print(np.var(sstall[fid,rid,model,:]))
# insst.append(sstcesm[rid][0]) # Append CESM-FULL
# insst.append(sstcesm[rid][1]) # Append CESM-SLAB 

# insstvars = []
# for s in insst:
#     insstvars.append(np.var(s))
#     #print(np.var(s))

# # Calculate Spectra and confidence Intervals
# specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct)
# alpha = 0.05
# bnds = []
# for nu in dofs:
#     lower,upper = tbx.confid(alpha,nu*2)
#     bnds.append([lower,upper])
    

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
    
specsall  = ss # array[forcing][region][model]
freqsall  = ff
Cfsall    = cc
bndsall   = bb
sstvarall = vv

#%% Function to add subplot labels


#%% Now let's plot two regions side by side, with 
# Autocorr on top row
# Power Spectra on the bottom

alw = 3
exclude_consth = True # Set to true to NOT plot constant h model
notitle        = True


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
    
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
    
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
        
    
    speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
    
    
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


#%% SM Draft 2, Plot all Regionals in 1 Plot


# Plotting Params
# ---------------
alw            = 3
exclude_consth = True # Set to true to NOT plot constant h model
notitle        = True
plotslab       = False # Set to True to plot the slab model simulation

plotlog        = True # Set to True to Plot in Log Scale

if exclude_consth:
    plotid = [1,2]
    plotidspec = [1,2,3,4] # Exclude constant h
else:
    plotid = [0,1,2]
    plotidspec = [0,1,2,3,4]

if plotslab is False: # Remove slab simulation
    plotidspec.remove(4)
    specylim_spg = [0,0.42]
    specylim_stg = [0,0.15]
else:
    specylim_spg = [0,0.8]
    specylim_stg = [0,0.3]

rids = [0,6,5,]
order = rids




# Do Plotting
# ---------------
fig,axs =plt.subplots(2,3,figsize=(16,8))
sp_id = 0
# Plot the autocorrelation (top row)
for i in range(len(rids)):
    
    ax = axs[0,i]
    
    rid    = order[i] 
    kmonth = kmonths[rid]
    #title  = "%s Autocorrelation (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])

    title  = "%s" % (regionlong[rid]) # No Month
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title="")
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
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
        ax.set_ylabel("Autocorrelation")
        
    if i == 1: # Add legend and x-label to middle plot
        ax.legend(ncol=2,fontsize=12)
    else:
        ax.set_xlabel("")
        
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1


# Plot the power spectra (bottom row)
for i in range(len(rids)):
    
    ax  = axs[1,i]
    rid = order[i]
    
    #speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[rid][i]) for i in range(len(insst)) ]
    speclabels=["" for i in range(len(insst))]
    
    
    if plotlog:
        ax,ax2 = viz.plot_freqlog(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                             ax=ax,plottitle=regionlong[rid],
                             xlm=xlm,xtick=xtks,return_ax2=True,
                             plotids=plotidspec,legend=False)
    else:
        ax,ax2 = viz.plot_freqlin(specsall[rid],freqsall[rid],speclabels,speccolors,lw=alw,
                             ax=ax,plottitle=regionlong[rid],
                             xlm=xlm,xtick=xtks,return_ax2=True,
                             plotids=plotidspec,legend=False)
    
    # Turn off title and second axis labels
    ax.set_title("")
    ax2.set_xlabel("")
    sxtk2 = ax2.get_xticklabels()
    sxtk2new = np.repeat("",len(sxtk2))
    ax2.set_xticklabels(sxtk2new)
    
    # Move period labels to ax1
    ax.set_xticklabels(xper)
    ax.set_xlabel("Period (Years)")
    
    if plotlog is False:
        rotation  =55
        xfontsize =8
    else:
        rotation  =0
        xfontsize =8
    
    plt.setp(ax.get_xticklabels(), rotation=rotation,fontsize=xfontsize)
    
    if i == 0:# Turn off y label except for leftmost plot
        ax.set_ylabel("Power Spectrum ($K^2 /cpy$)")
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
        
    if i == 2: # Just turn off for last STG plot
        ax.yaxis.set_ticklabels([])
    
    
    #title = "%s Power Spectra" % (regions[rid])
    title = ""
    ax.set_title(title,color=bbcol[rid],fontsize=12)
    
    ax = viz.label_sp(sp_id,fontsize=20,fig=fig,labelstyle="(%s)",case='lower')
    sp_id += 1

plt.tight_layout()
plt.savefig("%sRegional_Autocorrelation_Spectra%s.png"%(figpath,smoothname),
            dpi=200,transparent=False)

#%% Make the Corresponding Bounding Box


"""
Old Param Combinations that worked...

Having the bounding box and legend box right below it
bboxtemp = [-90,5,15,68]
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(4.5,3))
ax.text(-69,69,"Bounding Boxes",ha='center',bbox=props,fontsize=12) # (works for SPG Only)
ax.legend(ncol=1,fontsize=8,loc=6,bbox_to_anchor=(0, .75))

"""

cid = 0
rids = [0,6,5,]
bboxtemp = [-85,-5,15,65]

fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(3,2))
ax = viz.add_coast_grid(ax,bboxtemp,fill_color='gray',ignore_error=True)
fig.patch.set_alpha(1)  # solution
# # Plot the amv pattern
props = dict(boxstyle='square', facecolor='white', alpha=0.8)


#ax.text()

#ax.text(-69,69,"Bounding Boxes",ha='center',bbox=props,fontsize=12) # (works for SPG Only)


# Add text
ax.text(-38,50,"SPG",ha='center',fontsize=15,weight='bold') 
ax.text(-60,27,"STGw",ha='center',fontsize=15,weight='bold') 
ax.text(-25,27,"STGe",ha='center',fontsize=15,weight='bold') 
#ax.text()


# First PLot Solid lines below
for bb in rids:
    ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                          color=bbcol[bb],linestyle="solid",linewidth=3,return_line=True)

# Then plot dashed lines above
ls = []
for bb in rids:
    
    ax,ll = viz.plot_box(bboxes[bb],ax=ax,leglab=regions[bb],
                          color=bbcol[bb],linestyle="dashed",linewidth=3,return_line=True)
    ls.append(ll)

#BBox right below ttiel
#ax.legend(ncol=1,fontsize=8,loc=6,bbox_to_anchor=(0, .75))

#ax.text(-41,50,"SPG",ha='center',bbox=props,fontsize=25)

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
    
spec_num = -1
ref       = 'FULL' # [SLAB or FULL or ref_num]
ref_num   = 0
plotlog  = True
recip    = False

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
    spec_interp = np.interp(reffreq,freqs[spec_num],specsall[r][spec_num])
    
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

spec_num = -1
ref       = 'FULL' # [SLAB or FULL or ref_num]
ref_num   = 0
plotlog  = True
recip    = False

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
            ref     = refnames[a] #specnames[ref_num]
            
        
        # Interpolate selected spectra to reference frequencies
        spec_interp = np.interp(reffreq,freqs[spec_num],specsall[r][spec_num])
        
        
        # Compute the ratio
        if recip:
            specratio = refspec/spec_interp
            ylab = "%s/%s" % (ref,specnames[spec_num])
            flab = "%s-%s" % (ref,specnames[spec_num])
        else:
            specratio = spec_interp/refspec
            ylab = "%s/%s" % (specnames[spec_num],refnames[a])
            flab = "%s-%s" % (refnames[a],specnames[spec_num])
            
        if plotlog:
            specratio = np.log(specratio)
            ax.set_ylabel("log(%s)" % ylab)
        
        ln = ax.semilogx(reffreq*dtplot,specratio,label=regions[r],lw=4,color=bbcol[r])
    if a == 0:
        ax.legend(ncol=2)
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Period (Years)")
    ax.set_xlim([xtks[0],xtks[-1]])
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    
    ax.axhline(0,ls='dashed',color="k")
    ax.grid(True,ls='dotted')
    ax.set_ylim([-2.25,2.25])
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
    spid += 1
    


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
