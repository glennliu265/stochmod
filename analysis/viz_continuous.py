#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Figures for Continuous Stochastic Model Runs

Adapted from viz_AMV_comparison.py

Created on Thu Feb 24 16:22:35 2022

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
    figpath     = projpath + "02_Figures/20220222/"
   
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
fnames =["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method5_dmp0"%i for i in range(10)]
frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
exname ="Fprime_amq0_method5_cont"


# Visualize Continuous run 200, Qnet 
fnames =["forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run2%02d_ampq3_method5_dmp0"%i for i in range(10)]
frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
exname ="Qnet_amq3_method5_cont"


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


cint   = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int = np.arange(-0.45,0.50,0.05)
#bboxplot = [-100,20,0,80]

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

proj = ccrs.PlateCarree()
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

# -------------------------------------------
# %% A U T O C O R R E L A T I O N  P L O T S 
# -------------------------------------------
#%% Load Autocorrelation

# Load for stochastic model experiments
sstacs  = []
kmonths = []
for f in range(len(fnames)):
    
    # Load the dictionary [h-const, h-vary, entrain]
    expid = fnames[f]
    rsst_fn = "%sproc/SST_Region_Autocorrelation_%s.npz" % (datpath,expid)
    ld = np.load(rsst_fn,allow_pickle=True)#.item()
    sstac   = ld['autocorr_region'].item() # I think its [region][model][lag]
    kmonth  = ld['kmonths'].item()
    sstacs.append(sstac)
    kmonths.append(kmonth)

# Load data for CESM1-PIC
cesmacs= []
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_Region_Autocorrelation_%s_ensorem0.npz" % (datpath,expid)
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
cfcesm = np.zeros((len(regions),2,len(lags),2)) # [Region x Model x Lag x Upper/Lower]
ns     = [1798,898]
for rid in range(len(regions)):
    for mid in range(2):
        inac                = cesmacs[rid][mid]
        cfs                 = calc_conflag(inac,conf,tails,ns[mid])
        cfcesm[rid,mid,:,:] = cfs.copy()
#%% Updated Panel Plot

rid_sel = [0,5,6,2,4]
mid_sel = [0,2]

plt.style.use("default")

for f in range(len(frcnamelong)):
    
    fig,axs = viz.init_2rowodd(3, proj=None,figsize=(14,6))
    for r in range(len(rid_sel)):
        
        rid = rid_sel[r]
        ax = axs[r]
        kmonth = kmonths[f][rid]
        
        # Plot some differences
        title      = "%s (Lag 0 = %s)" % (regionlong[rid],mons3[kmonth])
        ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
        #ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
        #ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')
        
        # Plot Each Model
        for mid in mid_sel:
            ax.plot(lags,sstacs[f][rid][mid],color=mcolors[mid],label=modelnames[mid])
            ax.fill_between(lags,cfstoch[f,rid,mid,lags,0],cfstoch[f,rid,mid,lags,1],
                            color=mcolors[mid],alpha=0.10)
            
        # Plot CESM
        for cid in range(2):
            ax.plot(lags,cesmacs[rid][cid],color=cesmcolor[cid],label=cesmname[cid],ls=cesmline[cid])
            ax.fill_between(lags,cfcesm[rid,cid,lags,0],cfcesm[rid,cid,lags,1],
                            color=cesmcolor[cid],alpha=0.10)
            
        if r == 0:
            ax.legend()
        if r%2 == 1:
            ax.set_ylabel("")
        if r<2:
            ax.set_xlabel("")
    
    title = "SST Autocorrelation (Forcing = 90 perc. Variance [%s])" % (frcnamelong[f])
    plt.suptitle(title,y=1.02)
    savename = "%sSST_Autocorrelation_Comparison_%s.png" % (figpath,fnames[f])
    plt.savefig(savename,dpi=150,bbox_inches='tight')

# -------------------------------------------
# %% A M V  P A T T E R N
# -------------------------------------------
#%% Next load/plot the AMV Patterns

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

# Load data for CESM1-PIC
expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/AMV_Region_%s_ensorem0.npz" % (datpath,expid)
ldc        = np.load(rsst_fn,allow_pickle=True)
cesmpat = ldc['amvpat_region'].item()
cesmidx = ldc['amvidx_region'].item()

# Load Seasonal Patterns
if exname == 'seasonal':
    ldname = datpath + "proc/CESM1-PIC_ensorem0_SeasonalAMV.npz"
    ld     = np.load(ldname,allow_pickle=True)
    
    snames = ("Ann","DJF","MAM","JJA","SON")
    samvpats = ld['samvpats']
    long = ld['lon']
    latg = ld['lat'] # Global lat/lon    


# Set AMV Plotting specs, function
sel_rid = 4

#bboxplot = [-85, 5 ,5,60]
bboxplot = [-80,0,5,60]
plotbbox = False
def plot_amvpat(lon,lat,amvpat,ax,add_bbox=False,bbox_NNA=[-80, 0, 10, 65],blabel=[1,0,0,1]):
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,ignore_error=True)
    pcm = ax.contourf(lon,lat,amvpat,levels=cint,cmap=cmocean.cm.balance)
    ax.pcolormesh(lon,lat,amvpat,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
    cl = ax.contour(lon,lat,amvpat,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fontsize=8)
    
    if add_bbox:
        ax,ll = viz.plot_box(bbox_NNA,ax=ax,leglab="AMV",
                             color="k",linestyle="dashed",linewidth=2,return_line=True)
    return pcm,ax

# Load masks for damping test
dmsks = scm.load_dmasks(bbox=[lon[0],lon[-1],lat[0],lat[-1]])
dmsks.append(dmsks[-1])

#%% Plot AMV Patterns (Model vs CESM) for each forcing
# SM Paper Draft 2 

notitle = True
cmax  = 0.5
cstep = 0.025
lstep = 0.05
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]

for f in tqdm(range(len(fnames))):
    for rid in range(5):
        if rid != sel_rid:
            continue
        
        if darkmode:
            plt.style.use('dark_background')
            savename = "%sSST_AMVPattern_Comparison_region%s_%s_dark.png" % (figpath,regions[rid],fnames[f])
            dfcol = 'w'
        else:
            plt.style.use('default')
            savename = "%sSST_AMVPattern_Comparison_region%s_%s.png" % (figpath,regions[rid],fnames[f])
            dfcol = 'k'
        
        spid = 0
        proj = ccrs.PlateCarree()
        fig,axs = plt.subplots(2,3,subplot_kw={'projection':proj},constrained_layout=True,
                               figsize=(14,8))
        # 
        # figsize=(12,6)
        # ncol = 3
        # fig,axs = viz.init_2rowodd(ncol,proj,figsize=figsize,oddtop=False,debug=True)
        
        # Plot Stochastic Model Output
        nmods = len(amvpats[f][rid])
        for mid in range(nmods):
            ax = axs.flatten()[mid]
            
            # Set Labels, Axis, Coastline
            if mid == 0:
                blabel = [1,0,0,0]
            elif mid == 1:
                blabel = [0,0,0,1]
            else:
                blabel = [0,0,0,0]
            
            # Make the Plot
            ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color=dfcol,
                                    fill_color='gray')
            pcm = ax.contourf(lon,lat,amvpats[f][rid][mid].T,levels=cint,cmap=cmocean.cm.balance)
            ax.pcolormesh(lon,lat,amvpats[f][rid][mid].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
            cl = ax.contour(lon,lat,amvpats[f][rid][mid].T,levels=cl_int,colors="k",linewidths=0.5)
            ax.clabel(cl,levels=cl_int,fontsize=8)
            
            ax.set_title("%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(modelnames[mid],np.var(amvids[f][rid][mid])))
            if plotbbox:
                ax,ll = viz.plot_box(bbox_NNA,ax=ax,leglab="AMV",
                                     color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
                
            viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.1)
            
            ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
            spid += 1
            
        # Plot CESM1
        axs[1,1].axis('off')
        
        for cid in range(2):
            if cid == 0:
                ax = axs[1,2]
                blabel = [0,0,0,1]
                spid = 4 # Flipped order
            else:
                ax = axs[1,0]
                blabel = [1,0,0,1]
                spid = 3
                
            # Make the Plot
            ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color=dfcol,
                                    fill_color='gray')
            pcm = ax.contourf(lon180g,latg,cesmpat[rid][cid].T,levels=cint,cmap=cmocean.cm.balance)
            ax.pcolormesh(lon180g,latg,cesmpat[rid][cid].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
            cl = ax.contour(lon180g,latg,cesmpat[rid][cid].T,levels=cl_int,colors="k",linewidths=0.5)
            ax.clabel(cl,levels=cl_int,fontsize=8)
            ax.set_title("%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(cesmname[cid],np.var(cesmidx[rid][cid])))
            if plotbbox:
                ax,ll = viz.plot_box(bbox_NNA,ax=ax,leglab="AMV",
                                     color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
            ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7)
        
        cb = fig.colorbar(pcm,ax=axs[1,1],orientation='horizontal')
        cb.set_ticks(cint[::4])
        cb.ax.set_xticklabels(clb,rotation=45)
        #cb.ax.set_xticklabels(cint[::2],rotation=90)
        #tick_start = np.argmin(abs(cint-cint[0]))
        #cb.ax.set_xticklabels(cint[tick_start::2],rotation=90)
        if notitle is False:
            plt.suptitle("%s AMV Pattern and Index Variance [Forcing = %s]" % (regionlong[rid],frcnamelong[f]),fontsize=14)
        
        plt.savefig(savename,dpi=150,bbox_inches='tight')
#%% Compare AMV Patterns for a selected model
# (For Continuous Runs)

mid     = 2
rid     = 4
nexps   = len(fnames)
plotcb  = False

cmax  = 0.5
cstep = 0.025
lstep = 0.05
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]


# Get AMV Patterns
# ----------------
amvpats_cont = np.zeros((len(lon),len(lat),nexps)) * np.nan # [lon x lat x exp]
amvids_cont  = np.zeros((amvids[0][0][0].shape[0],nexps)) * np.nan # [time x exp]
for f in range(nexps):
    amvpats_cont[...,f] = amvpats[f][rid][mid]
    amvids_cont[...,f]  = amvids[f][rid][mid]
    
# Plot for each segment
# ---------------------
rowcol  = (2,5)

ylabels = ["%03d to %03d" % (i*1000,i*1000+999) for i in range(nexps)]

fig,axs = plt.subplots(rowcol[0],rowcol[1],subplot_kw={'projection':proj},figsize=(16,6),
                       constrained_layout=True)
spid = 0
for f in range(10):
    ax = axs.flatten()[f]
    
    # Set Labels, Axis, Coastline
    blabel = [0,0,0,0]
    if f%rowcol[1] == 0:
        blabel[0]  = 1
    if f > rowcol[1] -1:
        blabel[-1] = 1
    
    # Make the Plot
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color=dfcol,
                            fill_color='gray',ignore_error=True)
    ax.set_title("Years %s" % ylabels[f])
    
    # Plot contours
    pcm = ax.contourf(lon,lat,amvpats_cont[...,f].T,levels=cint,cmap=cmocean.cm.balance,extend='both')
    cl = ax.contour(lon,lat,amvpats_cont[...,f].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fontsize=8)
    
    # Plot Mask
    viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.1)
    
    # Label Subplots
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=12,alpha=0.7,fontcolor=dfcol)
    spid += 1
    
# Colorbar
if plotcb:
    cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.010,pad=0.01)
    cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")

# Saving Figure 
plt.savefig("%sAMV_Comparison_%s_%s_model%i.png" % (figpath,exname,regions[rid],mid),dpi=150)


# Plot the average pattern, compare with CESM
# -------------------------------------------
amv_avg     = amvpats_cont.mean(-1)
amv_std_avg = np.var(amvids_cont,0).mean(-1) 
cid     = 1 if mid<1 else 0 # Select CESM comparison
fig,axs = plt.subplots(1,2,subplot_kw={'projection':proj},figsize=(8,3),
                       constrained_layout=True)
spid = 0
for i in range(2):
    ax = axs.flatten()[i]
    if i == 0:
        blabel  = [1,0,0,1]
        amvplot = amv_avg.T 
        ptitle   = "Stochastic Model (%s, $\sigma_{AMV}^{2}=%.4f$)" % (modelnames[mid],amv_std_avg)
        plon,plat = lon,lat
    else:
        blabel = [0,0,0,1]
        amvplot = cesmpat[rid][cid].T
        ptitle   = "CESM1 (%s, $\sigma_{AMV}^{2}=%.4f$)" % (cesmname[cid],np.var(cesmidx[rid][cid]))
        plon,plat = lon180g,latg
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color=dfcol,
                            fill_color='gray',ignore_error=True)
    ax.set_title(ptitle)
    
    # Plot contours
    pcm = ax.contourf(plon,plat,amvplot,levels=cint,cmap=cmocean.cm.balance,extend='both')
    cl = ax.contour(plon,plat,amvplot,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fontsize=8)
    
    # Plot Mask
    if i == 0:
        viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.1)
    
    # Label Subplots
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7,fontcolor=dfcol)
    spid += 1
    
# Saving Figure 
plt.savefig("%sAMV_Comparison_%s_%s_model%i_AVG.png" % (figpath,exname,regions[rid],mid),dpi=150)

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
        nmod = len(sstdicts[fid][rid])
        for model in range(nmod):
            sstin  = sstdicts[fid][rid][model]
            sstvar = np.var(sstin)
            print("Variance for forcing %s, region %s, model %s is %f" % (fnames[fid],regions[rid],modelnames[model],sstvar))
            
            sstall[fid,rid,model,:] = sstin.copy()
            sstvars[fid,rid,model]   = sstvar

#%% Load corresponding CESM Data

expid      = "CESM1-PIC"
rsst_fn    = "%s/proc/SST_RegionAvg_%s_ensorem0.npy" % (datpath,expid)
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

ssmooth = 65
cnsmooths = [75,65]
pct        = 0.10
nsmooth = np.concatenate([np.ones(3)*ssmooth,np.ones(2)*cnsmooths])
smoothname = "smth-obs%03i-full%02i-slab%02i" % (ssmooth,cnsmooths[0],cnsmooths[1])


pct     = 0.10
rid     = 4
fid     = 0
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
    for rid in range(len(regions)):
        
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
    
#%% Plot Params

rid_sel  = [0,5,6,2,4,]
speccolors = ["r","magenta","Orange",dfcol,"gray"]
specnames  = np.hstack([modelnames,cesmname])
    
#%% Make the plot (Frequency x Power)
timemax = None
xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
dt   = 3600*24*30


#speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[i]) for i in range(len(insst)) ]

for f in tqdm(range(len(frcnamelong))): 
        fig,axs = plt.subplots(2,2,figsize=(16,8))
        for rid in range(4):
            ax    = axs.flatten()[rid]
            speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[f][rid][i]) for i in range(len(insst)) ]
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

#%% Make some updated plots
# -------------------------

# Linear Frequency Plots

# Spectra Plotting Params
xlm = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]


for f in tqdm(range(len(frcnamelong))):
    #fig,axs = plt.subplots(2,3,figsize=(24,10),sharey=False)
    fig,axs = viz.init_2rowodd(3, proj=None,figsize=(18,8))
    for r,rid in enumerate(rid_sel):
        
        speclabels = ["%s (%.3f $^{\circ}C^2$)" % (specnames[i],sstvarall[f][rid][i]) for i in range(len(insst)) ]
        ax    = axs[r]
        ax,ax2 = viz.plot_freqlin(specsall[f][rid],freqsall[f][rid],speclabels,speccolors,
                             ax=ax,plottitle=regionlong[rid],xlm=xlm,xtick=xtks,return_ax2=True)
        
        plt.setp(ax2.get_xticklabels(), rotation=50,fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=50,fontsize=8)
        
        if r <3:
            ax2.set_xlabel("Period (Years)")
            ax.set_xlabel("")
            
        if r not in [0,3]:
            ax.set_ylabel("")
            
        if rid == 0: # SPG
            ax.set_ylim([0,1.25])
        else:
            ax.set_ylim([0,0.35])
    plt.tight_layout()
    plt.suptitle("Regional AMV Index Spectra (Forcing=%s)"%(frcnamelong[f]),y=1.01)
    savename = "%sSST_Spectra_Comparison_%s_Linear-Decadal.png" % (figpath,fnames[f])
    plt.savefig(savename,dpi=150,bbox_inches='tight')



#%%
#% Plot the spectra
fig,axs = plt.subplots(1,2,figsize=(16,4))
for i in range(2):
    ax = axs[i]
    plotid = plotids[i]
    
    if plottype == "freqxpower":
        ax,ax2 = viz.plot_freqxpower(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlin":
        ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    elif plottype == "freqlog":
        ax,ax2 = viz.plot_freqlog(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                             ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True)
    
    if i == 1:
        ax.set_ylabel("")
    ax.set_xlabel("")
        
    
    
    ax.set_ylim(ylm)
    
fig.text(0.5, -0.05, 'Frequency (cycles/year)', ha='center',fontsize=12)
#plt.suptitle("SST Power Spectra at 50$\degree$N, 30$\degree$W",y=1.15,fontsize=14)
plt.suptitle(sharetitle,y=1.22,fontsize=14)
savename = "%sNASST_Spectra_Stochmod_%s_%s_pct%03i.png" % (outpath,plottype,smoothname,pct*100)
plt.savefig(savename,dpi=200,bbox_inches='tight')

#%% PLot SST Variance to show regiosns that belw up


sstplot = sstdicts[0][4][2]


