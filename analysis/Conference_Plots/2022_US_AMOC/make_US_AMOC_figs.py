#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

US AMOC Team Meeting Poster Plots

Created on Wed Apr 20 16:59:56 2022

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean
from tqdm import tqdm
from time import time

import cartopy.feature as cfeature

import matplotlib as mpl
#%% Check fonts

#mpl.rcParams


mpl.rcParams['font.sans-serif'] = "Red Hat Display"
mpl.rcParams['font.family']     = "sans-serif"

#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family']      = 'STIXGeneral'

# plt.rcParams['font.family'] = 'TeX Gyre Termes'
# plt.rcParams["mathtext.fontset"] = "stix"

#from matplotlib import font_manager

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20230220/" # 0422
   
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


#%% Set other params

outdpi = 300


# -----------------------------------------------
#%% (01) Make Stochastic Model AMV Pattern Plots
# -----------------------------------------------

"""
Load output from calc_AMV_continuous.py
"""

# Load the output
dpath1 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/analysis/Conference_Plots/2022_US_AMOC/"
ldname = "%sUSAMOC_AMV_Patterns_bboxlon-80to0_lat20to60_mask0.npz" % (dpath1)
ld     = np.load(ldname,allow_pickle=True)
print("Found the following keys: %s" % ld.files)

bbin      = ld['bbox_in']
bbstr     = ld['bbox_str']
applymask = ld['mask_insig_hf']
amvpats   = ld['amv_pats_sm']
camvpats  = ld['amv_pats_cesm']
lon       = ld['lon']
lat       = ld['lat']
amvids    = ld['amv_ids_sm']
camvids   = ld['amv_ids_cesm']

# Load dmsks
dmsks     = scm.load_dmasks(bbox=[lon[0],lon[-1],lat[0],lat[-1]])
dmsks.append(dmsks[-1])

#%% US AMOC Meeting 4-Panel Plot of AMV Pattern
#   Copied from Stochastic Model Plot (04/20/2022)
#mpl.rc('font',**{'fontname':'Proxima Nova'})

# Labels (Likely from sm_stylesheet)
modelnames = ("Constant h","Vary h","Entraining")
mconfigs   = ("SLAB","FULL")

# Plot settings
bboxplot   = [-80,0,5,60]
proj       = ccrs.PlateCarree()

# Contouring
cmax       = 0.5
cstep      = 0.025
lstep      = 0.05
cint,cl_int= viz.return_clevels(cmax,cstep,lstep)
clb        = ["%.2f"%i for i in cint[::4]]
cl_int     = cint

# Toggles
notitle    = True
darkmode   = True
plotbbox   = False
useC       = True

# Font Sizes
clabel_fsz = 12 
title_fsz  = 14

# Labels
bbfancy    = "Lon %i to %i, Lat %i to %i" % tuple(bbin)


# Begin Plotting
# ----------------
spid  = 0
fig,axs = plt.subplots(2,2,subplot_kw={'projection':proj},figsize=(12,8),
                       constrained_layout=True)
if darkmode:
    plt.style.use('dark_background')
    savename = "%sSST_AMVPattern_Comparison_region%s_mask%i_dark.png" % (figpath,bbstr,applymask)
    fig.patch.set_facecolor('k')#('#0B1D3F')
    dfcol = 'w'
    cset = "k"
    spalpha = 0.4 # Subplot rectangle
else:
    #plt.style.use('science')
    savename = "%sSST_AMVPattern_Comparison_region%s_mask%i.png" % (figpath,bbstr,applymask)
    fig.patch.set_facecolor('white')
    #fig.patch.set_facecolor("#C6F1FF")
    dfcol = 'k'
    cset = "w"#"#C6F1FF"
    spalpha = 0.7 # Subplot rectangle

# Plot Stochastic Model Output
# ----------------------------
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
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color='k',
                            fill_color='gray')
    
    pcm = ax.contourf(lon,lat,amvpats[mid,:,:].T,levels=cint,cmap='cmo.balance')
    cl = ax.contour(lon,lat,amvpats[mid,:,:].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int[::2],fontsize=clabel_fsz,fmt="%.02f")
    
    
    if useC:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)"%(modelnames[mid],np.var(amvids[mid,:]))
    else:
        ptitle = "%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(modelnames[mid],np.var(amvids[mid,:]))
    ax.set_title(ptitle,fontsize=title_fsz)
    if plotbbox:
        ax,ll = viz.plot_box(bbin,ax=ax,leglab="AMV",
                             color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
    
    viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.3)
    ax.set_facecolor = cset
    
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=spalpha,fontcolor='k')
    spid += 1
    
# Plot CESM1
# ----------
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
    ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color='k',
                            fill_color='gray')
    pcm = ax.contourf(lon,lat,camvpats[cid].T,levels=cint,cmap='cmo.balance')
    #ax.pcolormesh(lon,lat,camvpats[cid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
    cl = ax.contour(lon,lat,camvpats[cid].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int[::2],fontsize=clabel_fsz,fmt="%.02f")
    if useC:
        ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f$\degree C^2$)"%(mconfigs[cid],np.var(camvids[cid]))
    else:
        ptitle = "CESM-%s ($\sigma^2_{AMV}$ = %.04f $K^2$)"%(mconfigs[cid],np.var(camvids[cid]))
    ax.set_title(ptitle,fontsize=title_fsz)
    if plotbbox:
        ax,ll = viz.plot_box(bbin,ax=ax,leglab="",
                             color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
    ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=spalpha,fontcolor='k')
    
    ax.set_facecolor=cset
    
# Set Colorbar parameters
if useC:
    x = 1.16
    y = .89
    pad = -0.12
    cbar_label = "SST ($\degree C \, \sigma_{AMV}^{-1}$)"
else:
    x   = 1.15
    y   = .89
    pad = -0.09
    cbar_label = "SST ($K \, \sigma_{AMV}^{-1}$)"

# Make Colorbar and Label
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.030,pad=pad)
ax = axs[0,1] # Reference to first Colorbar
ax.text(x,y,cbar_label,horizontalalignment='center',verticalalignment='center',
        transform=ax.transAxes,fontsize=clabel_fsz)

    

if notitle is False:
    plt.suptitle("%s AMV Pattern and Index Variance" % (bbfancy),fontsize=14)
    
fig.set_size_inches(9.95, 6.6)

plt.savefig(savename,dpi=outdpi,bbox_inches='tight')

# -----------------------------------------------
#%% (02) Make HadiSST AMV Plot
# -----------------------------------------------

"""
Load data generated from: predict_amv/Analysis/plot_clim_CESM.py

"""
dtr      = True
savepath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220422/"
savename = "%sHadISST_AMV_detrend%i_lon-80to0_lat0to65.npz" % (savepath,dtr)

ld       = np.load(savename,allow_pickle=True)
print(ld.files)


amvpath  = ld['amvpat_hadisst']
amvidraw = ld['naisst_hadisst']
amvid    = ld['amvid_hadisst']
timesyr  = ld['times_yr']
lonh     = ld['lon']
lath     = ld['lat']
#%% Make the plots

plotdark=False
bbox    =[-80,0,0,65] # North Atlantic [lonW, lonE, latS, latN]


pdark = plotdark
if pdark:
    plt.style.use('dark_background')
    basecol = "w"
else:
    plt.style.use('default')
    basecol = "k"

# Plot the AMV Index
maskneg = amvidraw<0
maskpos = amvidraw>=0
timeplot = np.arange(0,len(amvid),1)
fig,ax = plt.subplots(1,1,figsize=(8,3))
ax.grid(True,ls='dotted')
ax.set_xticks(timeplot[::120])
ax.set_xticklabels(timesyr[::120])
#ax.plot(timeplot,amvid,label="AMV Index",color='gray',lw=.75,ls='dashdot')
ax.bar(timeplot[maskneg],amvidraw[maskneg],label="AMV-",color='cornflowerblue',width=1,alpha=1)
ax.bar(timeplot[maskpos],amvidraw[maskpos],label="AMV+",color='tomato',width=1,alpha=1)
ax.plot(timeplot,np.convolve(amvid,np.ones(20)/20,mode='same'),label="10-yr Low-Pass Filter",color=basecol,lw=1.2)
ax.axhline([0],color=basecol,ls='dashed',lw=0.9)
ax.set_ylabel("AMV Index ($^{\circ}C$)")
ax.set_ylim([-1,1])
ax.set_xlim([0,len(amvid)])
ax.set_xlabel("Years")
ax.set_title("AMV Index, Distribution by Year (HadISST)")
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
if plotdark:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i_dark.png"% (figpath,dtr),dpi=200,transparent=True)
else:
    plt.savefig("%sHadISST_AMV_Index_intime_ECML_detrend%i.png"% (figpath,dtr),dpi=200)


# Plot Spatial Pattern

# Plot Ense

fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.add_coast_grid(ax,bbox=bbox2)
ax = viz.plot_box(bbox,ax=ax,linestyle='dashed',linewidth=2,color=basecol)
pcm = ax.contourf(lonh,lath,amvpath,levels=cints,cmap=cmocean.cm.balance)
cl = ax.contour(lonh,lath,amvpath,levels=cintsl,colors="k",linewidths=0.75)
ax.clabel(cl,fontsize=8,fmt="%.2f")
ax.set_title("HadISST AMV Spatial Pattern ($^{\circ}C / 1\sigma_{AMV}$) \n1870-2018")
fig.colorbar(pcm,ax=ax,fraction=0.05,orientation='horizontal',pad=0.07)
ax.add_feature(cfeature.LAND,facecolor='gray')
#plt.tight_layout()
if plotdark:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i_dark.png"% (figpath,dtr),dpi=outdpi,transparent=True)
else:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i.png"% (figpath,dtr),dpi=outdpi)


#%% Combine onto a single plot

# Set some parameter

bbox    = [-80,0,0,65]
bbox2   = [-85,5,-5,70]

darkmode = False

clabel_fsz = 12
axis_fsz   = 14
title_fsz  = 16
tick_fsz   = 12
legend_fsz = 14


lonfix = [-80,-60,-40,-20,0]
latfix = [0  ,20 ,40 ,60]
cints=np.arange(-0.60,0.65,0.05)
cintsl = np.arange(-0.6,0.7,0.1)
cbtick   = np.arange(-.5,.75,.25)


if darkmode:
    plt.style.use('dark_background')
    dfcol = "w"
else:
    plt.style.use('default')
    dfcol = "k"

## Set 1 (figsize=(15,4)  wspace=0.01)


# Initialize
fig = plt.figure(constrained_layout=False, facecolor='w',figsize=(13,4))
gs  = fig.add_gridspec(nrows=1, ncols=3, left=.02, right=1,
                      hspace=.075, wspace=0.01)

# Plot the Map (right)
ax0 = fig.add_subplot(gs[0,2],projection=ccrs.PlateCarree())
ax  = ax0
ax  = viz.add_coast_grid(ax,bbox=bbox2,line_color=dfcol,fix_lon=lonfix,fix_lat=latfix,
                         ignore_error=True)
ax  = viz.plot_box(bbox,ax=ax,linestyle='dashed',linewidth=2,color=basecol)
pcm = ax.contourf(lonh,lath,amvpath,levels=cints,cmap=cmocean.cm.balance)
cl = ax.contour(lonh,lath,amvpath,levels=cintsl,colors="k",linewidths=0.75)
ax.clabel(cl,fontsize=clabel_fsz,fmt="%.2f")
ax.set_title("AMV Spatial Pattern ($\degree C \sigma_{AMV}^{-1}$)",fontsize=title_fsz)
cb = fig.colorbar(pcm,ax=ax,ticks=cbtick,
                  fraction=0.05,orientation='horizontal',pad=0.1)
cb.ax.tick_params(labelsize=tick_fsz)

ax.add_feature(cfeature.LAND,facecolor='gray')

# Plot the indices (left)
ax1 = fig.add_subplot(gs[0,0:2])
ax = ax1
maskneg  = amvidraw<0
maskpos  = amvidraw>=0
timeplot = np.arange(0,len(amvid),1)
ax.grid(True,ls='dotted')
ax.set_xticks(timeplot[::120])
ax.set_xticklabels(timesyr[::120],fontsize=tick_fsz)
ax.bar(timeplot[maskneg],amvidraw[maskneg],label="AMV-",color='cornflowerblue',width=1,alpha=1)
ax.bar(timeplot[maskpos],amvidraw[maskpos],label="AMV+",color='tomato',width=1,alpha=1)
ax.plot(timeplot,np.convolve(amvid,np.ones(20)/20,mode='same'),label="10-yr Low-Pass Filter",color=basecol,lw=1.2)
ax.axhline([0],color=basecol,ls='dashed',lw=0.9)
ax.set_ylabel("AMV Index ($^{\circ}C$)",fontsize=axis_fsz)
ax.set_ylim([-1,1])
ax.set_xlim([0,len(amvid)])
ax.set_xlabel("Years",fontsize=axis_fsz)
ax.tick_params(labelsize=tick_fsz)
ax.set_title("AMV Index (HadISST, 1870 to 2018)",fontsize=title_fsz)
ax.legend(fontsize=legend_fsz,ncol=3)




fig.set_size_inches(11.44,3.5)

if plotdark:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i_dark.png"% (figpath,dtr),dpi=outdpi,transparent=True,bbox_inches='tight')
else:
    plt.savefig("%sHadISST_AMVPAttern_EnsaVg_detrend%i.png"% (figpath,dtr),dpi=outdpi,bbox_inches='tight')
    
# ----------------------------
#%% (03) Make the locator plot
# ----------------------------



# ----------------------------------------------
# %% (04) Make the Spectra/Autocorrelation Plots
# ----------------------------------------------


# Set up Configuration
config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn009"   # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy"   #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at 
config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = figpath # Note need to fix this
config['smooth_forcing'] = False
config['method']      = 0 # Correction Factor (q-corr, lambda-based)
config['favg' ]       = False
config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)

# Load additional datasets
fullauto = np.load(datpath+"../CESM_clim/TS_FULL_Autocorrelation.npy")

# From SM Stylesheet -------------------------------------------------
# SM Lower Hierarchy (05/25/2021)
ecol_lower       = ["blue",'cyan','gold','red']
els_lower        = ["dotted","dashdot","dashed","solid"]
labels_lower     = ["All Constant",
                     r"Vary $F'$",
                     r"Vary $\lambda_a$",
                     "Vary $F'$ and $\lambda_a$"] 

# SM Upper Hierarchy (05/25/2021)
labels_upper = ["h=50m",
                 "Vary $F'$ and $\lambda_a$",
                 "Vary $F'$, $h$, and $\lambda_a$",
                 "Entraining"]
ecol_upper = ('mediumorchid','red','magenta','orange')
els_upper = ["dashdot","solid","dotted","dashed"]


# Confidence Level Calculations
conf  = 0.95
tails = 2

# End SM Stylesheet --------------------------------------------------

#%% Clean Run

#% Load some data into the local workspace for plotting
query   = config['query']
mconfig = config['mconfig']
lags    = config['lags']
ftype   = config['ftype']
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])

# Run Model
#config['Fpt'] = np.roll(Fpt,1)
ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt       =params
darkmode = True

# Read in CESM autocorrelation for all points'
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = scm.load_data(mconfig,ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()
sstall = sst

#%% Load Constant_v_vary experiments
savenames = "%sconst_v_vary_%s_runid%s_%s.npz" % (datpath,config['mconfig'],config['runid'],config['fname'])
print("Loading %s" % savenames)
ld        = np.load(savenames,allow_pickle=True)
c_acs     = ld['acs']
c_ssts    = ld['ssts']
expids    = ld['expids']
confs     = ld['confs']

#%% Calculate Conf Intervals

nlags   = len(lags)
cfstoch = np.zeros([4,nlags,2])
for m in range(4):
    inac = ac[m]
    
    n = int(len(sst[m])/12)
    print(n)
    cfs = proc.calc_conflag(inac,conf,tails,n)
    cfstoch[m,:,:] = cfs
cfslab = proc.calc_conflag(cesmauto2,conf,tails,898)
cffull = proc.calc_conflag(cesmautofull,conf,tails,1798)

#%% Load and calculate CESM Spectra

cssts = scm.load_cesm_pt(datpath + "../",loadname='both',grabpoint=[-30,50])

#%% Calculate Spectra

debug    = False
notitle  = True

# Some edits for Draft 4
use_ann     = True # Option to take ann avg before calcualtion
useC        = True # Swap to label as degree celsius

# Smoothing Params
nsmooth    = 350
cnsmooths  = [100,100]
pct        = 0.00

smoothname = "smth-obs%03i-full%02i-slab%02i" % (nsmooth,cnsmooths[0],cnsmooths[1])

# Spectra Plotting Params
plottype = "freqlin"
xlm  = [1e-2,5e0]
#xper = np.array([200,100,50,25,10,5,2,1,0.5]) # number of years
xper = np.array([100,50,20,10,5,2])
xtks = 1/xper
xlm  = [xtks[0],xtks[-1]]
ylm  = [0,3.0]
plotids = [[0,1,2,3,8],
           [5,6,7]
           ]


# Combine lower and upper hierarchy
inssts   = [c_ssts[0][1],c_ssts[1][1],c_ssts[2][1],c_ssts[3][1],sst[1],sst[2],sst[3],cssts[0],cssts[1]]
nsmooths = np.concatenate([np.ones(len(inssts)-2)*nsmooth,cnsmooths])
labels   = np.concatenate([labels_lower,labels_upper[1:],['CESM-FULL','CESM-SLAB']])
if useC:
    speclabels = ["%s (%.2f$ \, \degree C^{2}$)" % (labels[i],np.var(inssts[i])) for i in range(len(inssts))]

else:
    speclabels = ["%s (%.2f$ \, K^{2}$)" % (labels[i],np.var(inssts[i])) for i in range(len(inssts))]
allcols  = np.concatenate([ecol_lower,ecol_upper[1:],[dfcol,"gray"]])

# Calculate Autocorrelation (?)
allacs,allconfs = scm.calc_autocorr(inssts,lags,kmonth+1,calc_conf=True)

# Convert Dict --> Array
oac=[]
ocf=[]
for i in range(len(allacs)):
    oac.append(allacs[i])
    ocf.append(allconfs[i])
allacs=oac
allconfs=ocf

# Do spectral Analysis
if use_ann:
    inssts = [proc.ann_avg(sst,0) for sst in inssts]
    inssts[-2] = inssts[-2][1:] # Drop 1st year for even yrs
    inssts[-1] = inssts[-1][1:] # Drop 1st year for even yrs
    dt = 3600*24*365
else:
    dt = None
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(inssts,nsmooths,pct,dt=dt)

# Convert to list for indexing NumPy style
convert = [specs,freqs,speclabels,allcols]
for i in range(len(convert)):
    convert[i] = np.array(convert[i])
specs,freqs,speclabels,allcols = convert

# Compute the variance of each ts
sstvars     = [np.var(insst) for insst in inssts]
sstvars_lp  = [np.var(proc.lp_butter(insst,120,6)) for insst in inssts]
sststds = [np.std(insst) for insst in inssts]

if useC:
    sstvars_str = ["%s (%.2f $\degree C$)" % (labels[sv],sstvars[sv]) for sv in range(len(sstvars))]
else:
    sstvars_str = ["%s (%.2f $K^2$)" % (labels[sv],sstvars[sv]) for sv in range(len(sstvars))]



#%% Make the plot

# Set Parameters
axis_fsz   = 16
legend_fsz = 14
lw         = 4
tick_fsz   = 14


legend_outside = False

# Prepare AC Labels
xtk2 = np.arange(0,39,3)
xtk2_labels = []
kmonth_seen = []
mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
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

# Prepare AC Plot parameters
plotacs = c_acs 
model   = 1


# Set AC Plot limits
xlim_ac = [0,36]
ylim_ac = [-.1,1]

# Initialize Figure
fig,axs = plt.subplots(2,2,figsize=(10,6),constrained_layout=True)

# Plot the autocorrelation
for ia in range(2):
    
    ax = axs[0,ia]
    
    # Labeling
    if ia == 0:
        ax.set_ylabel("Correlation",fontsize=axis_fsz)
        ax.set_xlabel("Lag (Months)",fontsize=axis_fsz)
    
    ax.set_xticks(xtk2)
    ax.set_xticklabels(xtk2_labels,fontsize=tick_fsz)
    
    # Plot for each model in lower hierarchy
    if ia == 0:
        # Plot CESM-SLAB
        ax.plot(lags,cesmauto2[lags],lw=lw,
                label="CESM1 SLAB",color='gray',marker="o",markersize=4)
        ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)
        
        # Plot SM (Lower Hierarchy)
        for i,e in enumerate([0,1,2,3]):
            cfs = confs[e,model,:,:]
            ax.plot(lags,plotacs[e][model],lw=lw,
                    label=labels_lower[i],color=ecol_lower[i],ls=els_lower[i],marker="o",markersize=4)
            ax.fill_between(lags,cfs[:,0],cfs[:,1],color=ecol_lower[i],alpha=0.2)
            
    elif ia == 1:
        # Plot CESM-FULL
        ax.plot(lags,cesmautofull,lw=lw,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
        ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)
        
        # Plot SM (Upper Hierarchy)
        for i in range(2,4):
            ax.plot(lags,ac[i],lw=lw,
                    label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=3)
            ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)
    
    ax.set_xlim(xlim_ac)
    ax.set_ylim(ylim_ac)
    
    if legend_outside:
        if ia == 0:
            leg = ax.legend(ncol=3,fontsize=legend_fsz,
                            bbox_to_anchor=(1.0, 1.3),edgecolor="k")
        elif ia == 1:
            leg = ax.legend(ncol=2,fontsize=legend_fsz,
                            bbox_to_anchor=(0.87, 1.3),edgecolor="k")
    else:
        ax.legend(ncol=2,fontsize=legend_fsz,)
        
    plt.setp(ax.get_yticklabels(),fontsize=tick_fsz)


# -----------------------------------------------------------------------------

# Parameters
titles  = ["",""]
linearx = 0

# Set Labels
xtks       = [1/100,1/20,1/10,1/5,1/2]
xtk_labels = ["%.1f \n%iy" % (xtk,1/xtk) for xtk in xtks]


# Plot the spectra
for i in range(2):
    
    ax = axs[1,i]
    
    if i == 0:
        ax.set_ylabel("Variance ($\degree C^2 \, cpy^{-1}$)",fontsize=axis_fsz)
        
    
    
    plotid  = plotids[i]

    ax,ax2 = viz.plot_freqlin(specs[plotid],freqs[plotid],speclabels[plotid],allcols[plotid],
                         ax=ax,plottitle=titles[i],xtick=xtks,xlm=xlm,return_ax2=True,
                         lw=lw,linearx=linearx,legend=False,usegrid=False)
    
    
    if i == 0:
         ax.set_xlabel("")
         if useC:
             ax.set_ylabel("Power ($\degree C^2/cpy$)",fontsize=axis_fsz)
         else:
             ax.set_ylabel("Power ($K^2/cpy$)",fontsize=axis_fsz)
         ax.set_xlabel("Frequency (cycles/year)",fontsize=axis_fsz)
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")
    ax2.get_xaxis().set_ticks([])
    ax.set_xticklabels(xtk_labels,fontsize=tick_fsz)
    
    ax.set_ylim([0,3])
    
    ax.xaxis.grid(True,ls='dashed')
    plt.setp(ax.get_yticklabels(),fontsize=tick_fsz)

fig.set_size_inches(15.03, 9.08)

savename = "%sSpectra_AC_SPGTest.png" % (figpath)
plt.savefig(savename,dpi=outdpi,bbox_inches='tight')

#%% Figure out the poster size

fig,ax = plt.subplots(1,1)

test = np.ones((42,48))

test1 = np.zeros((40,30))

h = np.arange(0,48,1)
w = np.arange(0,42,1)

h1 = np.arange(0,test1.shape[1],1)
w1 = np.arange(0,test1.shape[0],1)

ax.pcolormesh(w,h,test.T,cmap='cmo.dense')
ax.pcolormesh(w1+1,h1+5,test1.T,cmap='cmo.balance')

ax.grid(True)


ax.set_aspect('equal')


#%% OSM Summary Figure, Updated

notitle    = True
darkmode   = True
cbar_horiz = True

cmax  = 0.5
cstep = 0.025
lstep = 0.05
cint,cl_int=viz.return_clevels(cmax,cstep,lstep)
clb = ["%.2f"%i for i in cint[::4]]
cl_int  = cint

f = 0
rid = 4

if cbar_horiz is True:
    corient  = 'horizontal'
    cfraction = 0.07
    cpad      = 0.05
    figsize   = (10,5)
else:
    corient  = 'horizontal'
    cfraction = 0.018
    cpad      = 0.02
    figsize   = (8,3)
    
    
if darkmode:
    plt.style.use('dark_background')
    savename = "%sSST_AMVPattern_Comparison_dark_summary.png" % (figpath)
    fig.patch.set_facecolor('black')
    dfcol = 'k'
else:
    plt.style.use('default')
    savename = "%sSST_AMVPattern_Comparison_summary.png" % (figpath)
    fig.patch.set_facecolor('white')
    dfcol = 'k'

spid = 0
proj = ccrs.PlateCarree()
fig,axs = plt.subplots(1,2,subplot_kw={'projection':proj},figsize=figsize,
                       constrained_layout=True)


aid = 0
mid = 2
ax = axs.flatten()[aid]

# Set Labels, Axis, Coastline
blabel = [1,0,0,1]

# Make the Plot
ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color='k',
                        fill_color='gray')

pcm = ax.contourf(lon,lat,amvpats[mid,:,:].T,levels=cint,cmap='cmo.balance')
cl = ax.contour(lon,lat,amvpats[mid,:,:].T,levels=cl_int,colors="k",linewidths=0.5)
ax.clabel(cl,levels=cl_int[::2],fontsize=clabel_fsz,fmt="%.02f")

ptitle = "%s" % (modelnames[mid])

ax.set_title(ptitle,fontsize=title_fsz)
if plotbbox:
    ax,ll = viz.plot_box(bbin,ax=ax,leglab="AMV",
                         color=dfcol,linestyle="dashed",linewidth=2,return_line=True)

viz.plot_mask(lon,lat,dmsks[mid],ax=ax,markersize=0.3)
ax.set_facecolor = cset

ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=spalpha,fontcolor='k')
spid += 1
    
# Plot CESM1
# ----------
cid = 1
ax = axs[1]
blabel = [0,0,0,1]
    
# Make the Plot
ax = viz.add_coast_grid(ax,bboxplot,blabels=blabel,line_color='k',
                        fill_color='gray')
pcm = ax.contourf(lon,lat,camvpats[cid].T,levels=cint,cmap='cmo.balance')
#ax.pcolormesh(lon,lat,camvpats[cid].T,vmin=cint[0],vmax=cint[-1],cmap='cmo.balance',zorder=-1)
cl = ax.contour(lon,lat,camvpats[cid].T,levels=cl_int,colors="k",linewidths=0.5)
ax.clabel(cl,levels=cl_int[::2],fontsize=clabel_fsz,fmt="%.02f")

ptitle = "CESM-%s" % (mconfigs[cid])

ax.set_title(ptitle,fontsize=title_fsz)
if plotbbox:
    ax,ll = viz.plot_box(bbin,ax=ax,leglab="",
                         color=dfcol,linestyle="dashed",linewidth=2,return_line=True)
ax = viz.label_sp(spid,case='lower',ax=ax,labelstyle="(%s)",fontsize=16,alpha=spalpha,fontcolor='k')

ax.set_facecolor=cset
    
# Set Colorbar parameters
if useC:
    x = 1.16
    y = .89

    cbar_label = "SST ($\degree C \, \sigma_{AMV}^{-1}$)"
else:
    x   = 1.15
    y   = .89

    cbar_label = "SST ($K \, \sigma_{AMV}^{-1}$)"

# Make Colorbar and Label
cb = fig.colorbar(pcm,ax=axs.flatten(),orientation='vertical',fraction=0.015,pad=0.01)
# ax = axs[0] # Reference to first Colorbar
# ax.text(x,y,cbar_label,horizontalalignment='center',verticalalignment='center',
#         transform=ax.transAxes,fontsize=clabel_fsz)

    

if notitle is False:
    plt.suptitle("%s AMV Pattern and Index Variance" % (bbfancy),fontsize=14)
    
fig.set_size_inches(9.95, 6.6)


#cb = fig.colorbar(pcm,ax=axs.flatten(),orientation=corient,fraction=cfraction,pad=cpad)
cb.set_ticks(cint[::4])
#cb.ax.set_xticklabels(clb,rotation=45)
#cb.set_label("SST ($K \, \sigma_{AMV}^{-1}$)")
#cb.ax.set_xticklabels(cint[::2],rotation=90)
#tick_start = np.argmin(abs(cint-cint[0]))
#cb.ax.set_xticklabels(cint[tick_start::2],rotation=90)
if notitle is False:
    plt.suptitle("%s AMV Pattern and Index Variance [Forcing = %s]" % (regionlong[rid],frcnamelong[f]),fontsize=14)

plt.savefig(savename,dpi=150,bbox_inches='tight')