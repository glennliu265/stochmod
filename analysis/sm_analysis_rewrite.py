#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Re-written Analysis Script for Stochastic Model Ouput

Created on Sat Jul 24 20:22:54 2021

@author: gliu
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cmocean


#%%
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20210810/"
   
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
runid     = "003"

# Analysis (7/26/2021, comparing 80% variance threshold and 5 or 3 EOFs)
#fnames      = ["flxeof_080pct_SLAB-PIC","flxeof_5eofs_SLAB-PIC","flxeof_3eofs_SLAB-PIC"]
#frcnamelong = ("80% Variance Threshold","5 EOFs","3 EOFs")

# Analysis: Trying different number of EOFs
neofs       = [1,2,50]#[1,2,3,5,10,25,50]
#fnames      = ["flxeof_qek_%ieofs_SLAB-PIC" % x for x in neofs]
fnames      = ["flxeof_qek_%ieofs_SLAB-PIC_JJA" % x for x in neofs]
#fnames      = ["flxeof_%ieofs_SLAB-PIC" % x for x in neofs]
frcnamelong = ["%i EOFs" % x for x in neofs]

#%% Post Process Outputs (Calculate AMV, Autocorrelation)
for frcname in fnames:
    expid = "forcing%s_%iyr_run%s" % (frcname,nyrs,runid) 
    scm.postprocess_stochoutput(expid,datpath,rawpath,outpathdat,lags)
    print("Completed Post-processing for Experiment: %s" % expid)
    
#%% Visualize AMV

# Regional Analysis Settings
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")        # Region Names
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA) # Bounding Boxes
cint   = np.arange(-0.45,0.50,0.05) # Used this for 7/26/2021 Meeting
cl_int = np.arange(-0.45,0.50,0.05)
bboxplot = [-100,20,0,80]
modelnames  = ("Constant h","Vary h","Entraining")

#%% Experiment names
# -- SelectExperiment -- 
fid   = 2
frcname = fnames[fid]
runid = "003"
expid = "forcing%s_%iyr_run%s" % (frcname,nyrs,runid) 
regid = 3

# Load lat/lon regional
lon = np.load(datpath+"lon.npy")
lat = np.load(datpath+"lat.npy")

# Load post-propcssed output
# AMV_Region_forcingflxeof_5eofs_SLAB-PIC_1000yr_run001.npz
ldpath = datpath + "proc/AMV_Region_%s.npz" % expid
ld = np.load(ldpath,allow_pickle=True)

# Load. Things are organized by region, then by model
amvidx = ld['amvidx_region'].item()[regid]
amvpat = ld['amvpat_region'].item()[regid]


fig,axs = plt.subplots(1,3,figsize=(8,3),subplot_kw={'projection':ccrs.PlateCarree()})
for p in range(len(amvpat)):
    ax = axs.flatten()[p]
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
    ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance,zorder=-1)
    #cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
    #ax.clabel(cl,levels=cl_int)
    #pcm = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance)
    #ax.set_title(modelnames[p])
    #fig.colorbar(pcm,ax=ax,fraction=0.036)
    ax.set_title(modelnames[p])
fig.colorbar(pcm,ax=axs.ravel().tolist(),orientation='vertical',shrink=0.5,pad=0.01)#,pad=0.015)
plt.suptitle("%s AMV Pattern ($\circ C$ per $\sigma_{AMV}$, Forcing: %s)"%(regions[regid],frcnamelong[fid]),y=0.85,fontsize=14)
plt.savefig("%sAMV_Pattern_%s_region%s.png"%(figpath,expid,regions[regid]),dpi=200,bbox_inches = 'tight')




#%% Individual AMV Plots (rather than panel based)


for p in range(len(amvpat)):
    fig,ax = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    pcm = ax.contourf(lon,lat,amvpat[p].T,levels=cint,cmap=cmocean.cm.balance)
    cl = ax.contour(lon,lat,amvpat[p].T,levels=cl_int,colors="k",linewidths=0.5)
    ax.clabel(cl,levels=cl_int,fmt="%.2f",fontsize=8)
    
    #pcm = ax.pcolormesh(lon,lat,amvpat[p].T,vmin=cint[0],vmax=cint[-1],cmap=cmocean.cm.balance)
    #ax.set_title(modelnames[p])
    #fig.colorbar(pcm,ax=ax,fraction=0.036)
    
    ax.set_title(modelnames[p] + " ($\circ C$ per $\sigma_{AMV}$) \n Forcing: %s" % frcnamelong[fid])
    fig.colorbar(pcm,ax=ax,orientation='horizontal',shrink=0.75)#,pad=0.015)
    plt.savefig("%sAMV_Pattern_%s_region%s_model%s.png"%(figpath,expid,regions[regid],modelnames[p]),dpi=200,bbox_tight='inches')


#%% Load in regional SSTs, and perform spectral analysis

# Stored in nested dict...
# Outer dict: here keys are 0-3, for each region
# Inner dict: keys are 0-2, for each model

# Load in SSTs for each region
sstdicts = []
runids = np.repeat("003",len(neofs))
#runids = ["003"]#["002","002","002","002","002","003","003"] # Accomodate different RunIDs
for f in range(len(neofs)):
    # Load the dictionary
    expid = "forcing%s_%iyr_run%s" % (fnames[f],nyrs,runids[f]) # Get experiment name
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

for rid in range(len(regions)):
    for model in range(len(cesmname)):
        sstin  = sstcesm[rid][model]
        sstvar = np.var(sstin)
        print("Variance for region %s, model %s is %f" % (regions[rid],cesmname[model],sstvar))
        sstvarscesm[rid,model]   = sstvar
        
#%% Make a plot of the variance... compare for each forcing scenario
elabels = ["%i EOFs"% x for x in neofs]
eofaxis = np.arange(0,51,1)
#xtk = [1,2,3,5,10,25,50]
xtk = [1,2,5,10,25,50]
#xtkplot = 
for rid in range(len(regions)):
    fig,ax  = plt.subplots(1,1,figsize=(8,4))
    for model in range(len(modelnames)):
        ax.plot(xtk,sstvars[:,rid,model],marker="o",label=modelnames[model])
    
    ax.axhline(sstvarscesm[rid,0],color="k",label="CESM-FULL (%.3f $degC^{2}$)"%(sstvarscesm[rid,0]),ls='dashed')
    ax.axhline(sstvarscesm[rid,1],color="gray",label="CESM-SLAB (%.3f $degC^{2}$)" %(sstvarscesm[rid,1]),ls='dashed')
    ax.legend(ncol=2)
    ax.set_xticks(xtk)
    #ax.set_xticklabels(elabels)
    ax.set_ylabel("$(\circ C)^{2}$")
    ax.set_xlabel("Number of EOF Patterns")
    ax.set_title("%s AMV Index Variance vs. Number of EOF Patterns \n used to force the stochastic model" % (regions[rid]))
    ax.grid(True,ls='dotted')
    plt.savefig("%svariance_vs_nEOF_by_model_runid%s_nyr%i_region%s.png"%(figpath,runid,nyrs,regions[rid]),dpi=150)

#%% Do some spectral analysis
nsmooth = 5
pct     = 0.10

rid  = 3
fid  = 5
dofs = [1000,1000,898,1798] # In number of years

# Unpack and stack data
insst = []
for model in range(len(modelnames)):
    insst.append(sstall[fid,rid,model,:]) # Append each stochastic model result
insst.append(sstcesm[rid][0]) # Append CESM-FULL
insst.append(sstcesm[rid][1]) # Append CESM-SLAB 

# Calculate Spectra and confidence Intervals
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(insst,nsmooth,pct)
alpha = 0.05
bnds = []
for nu in dofs:
    lower,upper = tbx.confid(alpha,nu*2)
    bnds.append([lower,upper])
    
#%% Make the plot
timemax = None
xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
dt   = 3600*24*30

specnames  = np.hstack([modelnames,cesmname])
speclabels = ["%s (%.3f $degC^2$)" % (specnames[i],np.var(insst[i])) for i in range(len(insst)) ]

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


speccolors = ["b","r","m","k","gray"]

plotdt = 3600*24*365
freq   = freqs[0]

fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(len(specs)):
    ax.plot(freqs[i]*plotdt,specs[i]/plotdt,label=speclabels[i],color=speccolors[i])
    ax.plot(freqs[i]*plotdt,CCs[i][:,1]/plotdt,label="",color=speccolors[i],ls='dashed')
    
ax.legend()



#i = 1
#ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
#ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
#ax.plot(freqcesmslab*plotdt,CLs[1][:,0]/plotdt,color='gray',label="CESM1 SLAB AR1",ls=':')
#ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
#ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
#ax.plot(freqcesmfull*plotdt,CLs[0][:,0]/plotdt,color='black',label="CESM1 FULL AR1",ls=':')
#ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
#ax.set_xlabel("")
#ax.set_title("CESM1 NASSTI (SLAB vs. FULL) \n nsmooth=%i"%(nsmooths[0]))

ax.set_xlim(xlms)
ax.set_xticks(xtks)
xtick=np.array(xtks)
htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
# Set xtick labels
htax.set_xticklabels(xtkl)
#ax.set_ylim([0,2])
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("%sPreliminNATSpec"%figpath,dpi=200)




