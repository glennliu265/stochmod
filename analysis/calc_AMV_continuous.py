#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV for a continuous simulation

Upper section taken from viz_continuous.py

Created on Wed Mar  2 15:11:39 2022

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
    figpath     = projpath + "02_Figures/20220315/"
   
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
fnames   = ["forcingflxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0_1000yr_run2%02d_ampq0_method4_dmp0"%i for i in range(10)]
frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
exname   = "Fprime_amq0_method5_cont"

# # Visualize Continuous run 200, Qnet 
# fnames =["forcingflxeof_090pct_SLAB-PIC_eofcorr2_1000yr_run2%02d_ampq3_method5_dmp0"%i for i in range(10)]
# frcnamelong = ["$F'$ run 2%02d" % (i) for i in range(10)]
# exname ="Qnet_amq3_method5_cont"

# Plotting Params
darkmode = False
debug    = False

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
bbox_ST_w   = [-80,-40,20,40]
bbox_ST_e   = [-40,-10,20,40]
bbox_NAextr = [-80,0,20,60]

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

# Load dmsks
dmsks = scm.load_dmasks(bbox=[lon[0],lon[-1],lat[0],lat[-1]])
dmsks.append(dmsks[-1])
#%% For each model read in the data

if debug:
    for f in range(10):
        print("Range is %i to %i" % (f*1000,(f+1)*1000-1))
    f = 0
    fname = fnames[f]


# Load in data and take annual average
sst_all = []
for f,fname in tqdm(enumerate(fnames)):
    ld = np.load(datpath+"stoch_output_%s.npz"%fname,allow_pickle=True)
    ssts = ld['sst']
    if f == 0:
        lonr   = ld['lon']
        latr   = ld['lat']
    ssts_ann = proc.ann_avg(ssts,3)
    sst_all.append(ssts_ann)
sst_all = np.concatenate(sst_all,axis=3) # [model x lon x lat x year]

#%% Load data for CESM

# Copied from Explore_Regional_Properties.ipynb, 03/11/2022
mconfigs = ("SLAB","FULL")
cdatpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
bbox     = [np.floor(lonr[0]),np.ceil(lonr[-1]),np.floor(latr[0]),np.ceil(latr[-1])]
print("Simulation bounding Box is %s "% (str(bbox)))

sst_cesm = []
for mconfig in mconfigs:
    fname   = "%sCESM1_%s_postprocessed_NAtl.nc" % (cdatpath,mconfig)
    ds      = xr.open_dataset(fname)
    dsreg   = ds.sel(lon=slice(bbox[0],bbox[1]),lat=slice(bbox[2],bbox[3]))
    sst_cesm.append(dsreg.SST.values)



#%% Now Compute the AMV Pattern

applymask          = True
amvbboxes          = ([-80,0,10,65],[-80,0,20,60],[-80,0,40,60],[-80,0,0,65])
nboxes             = len(amvbboxes)
nmod,nlon,nlat,nyr = sst_all.shape

# Compute the AMV Pattern (for the stochastic model)
amvpats = np.zeros((nmod,nlon,nlat,nboxes))*np.nan # [model x lon x lat x region]
amvids  = np.zeros((nmod,nyr,nboxes))      *np.nan
camvpats  = [] # [bbox][cesm-config]
camvids   = []
for b,bbin in tqdm(enumerate(amvbboxes)):
    
    
    # Do for Stochastic Models
    for mid in range(nmod):
        
        if applymask:
            inmask = dmsks[mid]
        else:
            inmask = None
        sst_in = sst_all[mid,...]
        amvid,amvpat = proc.calc_AMVquick(sst_in,lonr,latr,bbin,anndata=True,
                                          runmean=False,dropedge=5,mask=inmask)
        amvpats[mid,...,b]  = amvpat.copy()
        amvids[mid,...,b]   = amvid.copy()
        
    # Do for CESM
    cpats = []
    cids  = []
    for i in range(2):
        sst_in = sst_cesm[i]
        amvid,amvpat = proc.calc_AMVquick(sst_in,lonr,latr,bbin,anndata=False,
                                          runmean=False,dropedge=5,mask=None)
        cpats.append(amvpat)
        cids.append(amvid)
    camvpats.append(cpats)
    camvids.append(cids)

#%% Plot Traditional AMV Pattern (3 Panel) for each AMV bbox

b         = 0
bbox_plot = [-85,5,0,60]
fig,axs   = plt.subplots(1,3,figsize=(12,6),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
#cint = np.arange(-.5,0.525,0.025)
for mid in range(3):
    blabel = [0,0,0,1]
    if b == 0:
        blabel[0] = 1
    
    ax  = axs.flatten()[b]
    ax  = viz.add_coast_grid(ax,bbox_plot,fill_color='gray',blabels=blabel)
    
    cf= ax.contourf(lonr,latr,amvpats[mid,:,:,b].T,levels=cint,cmap='cmo.balance')
    cl= ax.contour(lonr,latr,amvpats[mid,:,:,b].T,levels=cl_int,colors='k',linewidths=0.55)
    ax.clabel(cl)
    ax.set_title(modelnames[mid])
    
    ax = viz.plot_box(bbin,ax=ax,linewidth=1.5,linestyle='dashed')
    #plt.setp(axs[mopt, :], ylabel=maskopt[mopt])
cb = fig.colorbar(cf,ax=axs.flatten(),fraction=0.0156)
cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Comparison_bbox_allmodels.png"%(figpath),dpi=150)

#%% Compare BBOX for a selected model

mid       = 0
bbox_plot = [-85,5,0,60]
fig,axs   = plt.subplots(1,3,figsize=(12,6),
                       subplot_kw={'projection':ccrs.PlateCarree()},constrained_layout=True)
#cint = np.arange(-.5,0.525,0.025)
for b,bbin in enumerate(amvbboxes):
    blabel = [0,0,0,1]
    if b == 0:
        blabel[0] = 1
    
    ax  = axs.flatten()[b]
    ax  = viz.add_coast_grid(ax,bbox_plot,fill_color='gray',blabels=blabel)
    
    cf= ax.contourf(lonr,latr,amvpats[mid,:,:,b].T,levels=cint,cmap='cmo.balance')
    cl= ax.contour(lonr,latr,amvpats[mid,:,:,b].T,levels=cl_int,colors='k',linewidths=0.55)
    ax.clabel(cl)
    ax.set_title(amvbboxes[b])
    
    ax = viz.plot_box(bbin,ax=ax,linewidth=1.5,linestyle='dashed')
    #plt.setp(axs[mopt, :], ylabel=maskopt[mopt])
cb = fig.colorbar(cf,ax=axs.flatten(),fraction=0.0156)
cb.set_label("AMV Pattern ($K \sigma_{AMV}^{-1}$)")
plt.savefig("%sAMV_Comparison_bboxes.png"%(figpath),dpi=150)

#%% Redo Stochastic Model Paper Plot
# Copied from viz_AMV_comparison.py (03/11/2022)

# Plot settings
notitle    = True
darkmode   = False
cmax       = 0.5
cstep      = 0.025
lstep      = 0.05
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb        = ["%.2f"%i for i in cint[::4]]


sel_rid   = 3

plotbbox  = False


# Begin Plotting
# ----------------
rid   = sel_rid
bbin  = amvbboxes[rid]
bbstr = "lon%ito%i_lat%ito%i" % (bbin[0],bbin[1],bbin[2],bbin[3])

spid = 0
proj = ccrs.PlateCarree()
fig,axs = plt.subplots(2,2,subplot_kw={'projection':proj},figsize=(9,9),
                       constrained_layout=True)

if darkmode:
    plt.style.use('dark_background')
    
    
    savename = "%sSST_AMVPattern_Comparison_%s_region%s_mask%i_dark.png" % (figpath,fnames[f],bbstr,applymask)
    fig.patch.set_facecolor('black')
    dfcol = 'k'
else:
    plt.style.use('default')
    savename = "%sSST_AMVPattern_Comparison_%s_region%s_mask%i.png" % (figpath,fnames[f],bbstr,applymask)
    fig.patch.set_facecolor('white')
    dfcol = 'k'




# figsize=(12,6)
# ncol = 3
# fig,axs = viz.init_2rowodd(ncol,proj,figsize=figsize,oddtop=False,debug=True)

# Plot Stochastic Model Output
for aid,mid in enumerate([0,2]):
    ax = axs.flatten()[aid]
    
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
    pcm = ax.contourf(lon,lat,amvpats[mid,:,:,rid].T,levels=cint,cmap='cmo.balance')
    ax.pcolormesh(lon,lat,amvpats[mid,:,:,rid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(lon,lat,amvpats[mid,:,:,rid].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fontsize=8)
    
    
    ax.set_title("%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(modelnames[mid],np.var(amvids[mid,:,rid])))
    if plotbbox:
        ax,ll = viz.plot_box(amvbboxes[rid],ax=ax,leglab="AMV",
                             color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
        
    viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.3)
    ax.set_facecolor=dfcol
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7,fontcolor=dfcol)
    spid += 1
    
# Plot CESM1
#axs[1,1].axis('off')

for cid in range(2):
    
    ax = axs[1,cid]
    if cid == 0:
        #ax = axs[1,0]
        blabel = [1,0,0,1]
        spid = 2
        
        #spid = 3 # Flipped order
    else:
        blabel = [0,0,0,1]
        #ax = axs[1,0]
        
        spid = 3
        
    # Make the Plot
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color=dfcol,
                            fill_color='gray')
    pcm = ax.contourf(lon,lat,camvpats[rid][cid].T,levels=cint,cmap='cmo.balance')
    ax.pcolormesh(lon,lat,camvpats[rid][cid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(lon,lat,camvpats[rid][cid].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fontsize=8)
    ax.set_title("CESM-%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(mconfigs[cid],np.var(camvids[rid][cid])))
    if plotbbox:
        ax,ll = viz.plot_box(amvbboxes[rid],ax=ax,leglab="",
                             color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=0.7,fontcolor=dfcol)

cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.030,pad=0.05)
cb.set_ticks(cint[::4])
cb.ax.set_xticklabels(clb,rotation=45)
cb.set_label("SST ($K \, \sigma_{AMV}^{-1}$)")
#cb.ax.set_xticklabels(cint[::2],rotation=90)
#tick_start = np.argmin(abs(cint-cint[0]))
#cb.ax.set_xticklabels(cint[tick_start::2],rotation=90)
if notitle is False:
    plt.suptitle("%s AMV Pattern and Index Variance [Forcing = %s]" % (regionlong[rid],frcnamelong[f]),fontsize=14)

plt.savefig(savename,dpi=150,bbox_inches='tight')



#%% Lets analyze conditions at a particular point
input_path = datpath+"model_input/"
frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0"
inputs     = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs


#%% SM Paper Draft 3 (CESM AMV Inset for Seasonal Cycle Figure)

# Calculate the AMV over bounding box bbin
cid      = 1 # Set the CESM model
bbin     = [-80,0,20,60]
bboxplot = [-80,0,10,60] 
sst_in = sst_cesm[cid]
amvid,amvpat = proc.calc_AMVquick(sst_in,lonr,latr,bbin,anndata=True,
                                  runmean=False,dropedge=5,mask=inmask)

# Prepare Tick Labels
cl_int    = np.arange(-0.45,0.50,0.05)
cb_lab    = np.arange(-.5,.6,.1)

# Make the Plot
fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(6,4))
ax = viz.add_coast_grid(ax,bbox=bboxplot)
ax = viz.add_coast_grid(ax,bboxplot,blabels=[1,0,0,1],line_color=dfcol,
                        fill_color='gray')
pcm = ax.contourf(lon,lat,amvpat.T,levels=cint,cmap='cmo.balance',extend='both')
#ax.pcolormesh(lon,lat,amvpat.T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
cl = ax.contour(lon,lat,amvpat.T,levels=cl_int,colors="k",linewidths=0.5)
ax.scatter(-30,50,200,marker="*",facecolor='yellow',zorder=9,edgecolor='k',linewidth=.5)
ax.clabel(cl,levels=cl_int,fontsize=8)

# Add Colorbar, Reduce tick labels
cb = fig.colorbar(pcm,ax=ax,orientation="horizontal",fraction=0.050,pad=0.1)
cb.set_label("SST ($K \sigma_{AMV}^{-1}$)")
cb.set_ticks(cb_lab)

plt.savefig("%sAMV_Patterns_Indv_%s.png"% (figpath,mconfigs[cid]),dpi=200,bbox_inches='tight')

#%%