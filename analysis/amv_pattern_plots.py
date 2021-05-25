#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 22:27:27 2021

@author: gliu
"""


import numpy as np
import sys
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%% User Edits

# Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath + '/proc/'
outpathfig  = projpath + '02_Figures/20210509/'




# Plot Settings
plt.style.use("default")
cmap = cmocean.cm.balance
cints = np.arange(-.55,.60,.05)
cintslb = np.arange(-.50,.6,.1)


#%% More Params

runids=['303']
lags = np.arange(0,37,1)

# Options to determine the experiment ID
fscale  = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
runid = '303'

# Set region variables
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
rcol = ('b','r',[0,1,0],'k')
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]

#funiforms=[0,1,3,5.5,7,]
funiforms=[2.5]
fnames=["NAO"]
#fnames  = ["Random","Uniform","NAO","EAP","NAO+EAP"]

fcolors = ["teal","springgreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']
mconfig = "SLAB_PIC"
applyfac = 2

# Set Model Names
modelname = ("MLD Fixed","MLD Mean", "MLD Seasonal", "MLD Entrain")
modelfname = ("MLDFixed","MLDMean", "MLDSeasonal", "MLDEntrain")
modelnamenew = ("h=50m","Constant $h$","Vary $h$","Entraining")

#%%

region = 3
model  = 3
# # Load AMV Spatial Patterns
amvsp = {}
amvid = {}
for funiform in funiforms:
    
    # Set experiment ID
    expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)
    
    amvload = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)
    
    amvsp[funiform] = amvload['amvpat_region'].item()[region][model] # Contains [region][model]
    amvid[funiform] = amvload['amvidx_region'].item()[region][model]
    
    print("Loaded in post-processed data for %s" % expid)
    
    
    
# Load lat/lon coordinates
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")



# Additionally load in other pattern
# Set experiment ID
expid = "%s_%iyr_funiform1_run203_fscale001_applyfac%i" %(mconfig,nyrs,applyfac)
amvload = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)

amvsp15 = amvload['amvpat_region'].item()[region][model] # Contains [region][model]
amvid15 = amvload['amvidx_region'].item()[region][model]

#%% Make the plots



i = 6
mult=1
fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
#ax = viz.init_map(bbox_NA,ax=ax)
#pcm = ax.contourf(lonr,latr,amvsp[i].T,levels=cints,cmap=cmap)
#fig.colorbar(pcm,ax=ax,fraction=0.05)
ax,cb = viz.plot_AMV_spatial(amvsp15.T*mult,lonr,latr,bbox_NA,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True)
cb.set_ticks(cintslb)
ax.set_title("AMV Pattern (NHFLXSTD)")
plt.savefig(outpathfig+"AMV_Spatial_Pattern_%s_%s_runid%s.png"%(modelfname[model],"NHFLXSTD","203"))



for i in range(5):
    if i == 1:
        mult= 10
    else:
        mult = 1
    
    fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    #ax = viz.init_map(bbox_NA,ax=ax)
    #pcm = ax.contourf(lonr,latr,amvsp[i].T,levels=cints,cmap=cmap)
    #fig.colorbar(pcm,ax=ax,fraction=0.05)
    ax,cb = viz.plot_AMV_spatial(amvsp[funiforms[i]].T*mult,lonr,latr,bbox_NA,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True)
    cb.set_ticks(cintslb)
    ax.set_title("AMV Pattern (%s)"%fnames[i])
    plt.savefig(outpathfig+"AMV_Spatial_Pattern_%s_%s_runid%s.png"%(modelfname[model],fnames[i],runid))

cints = np.arange(-.5,.52,.02)
cintslb = np.arange(-.5,.6,.1)
fig,axs = plt.subplots(2,3,figsize=(6,6),subplot_kw={"projection":ccrs.PlateCarree()})
for i in range(5):
    ax = axs.flatten()[i]
    ax = viz.init_map(bbox_NA,ax=ax)
    pcm = ax.contourf(lonr,latr,amvsp[funiforms[i]].T,levels=cints,cmap=cmap)
    cl = ax.contour(lonr,latr,amvsp[funiforms[i]].T,levels=cintslb,colors='k',linewidths=0.25)
    ax.set_title("AMV Pattern (%s)"%fnames[i])
fig.colorbar(pcm,ax=axs.ravel().tolist())

#%% Loop by model type (assuming just 1 experiment) 
region = 3
vscale = 1
funiform = 2.5

# # Load AMV Spatial Patterns
amvsp = {}
amvid = {}
for funiform in funiforms:
    
    # Set experiment ID
    expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)
    
    amvload = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)
    
    amvsp[funiform] = amvload['amvpat_region'].item()[region] # Contains [region][model]
    amvid[funiform] = amvload['amvidx_region'].item()[region]
    
    print("Loaded in post-processed data for %s" % expid)
    
    

for m in range(4):
    
    if m == 4:
        vscale=1
    fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    #ax = viz.init_map(bbox_NA,ax=ax)
    #pcm = ax.contourf(lonr,latr,amvsp[i].T,levels=cints,cmap=cmap)
    #fig.colorbar(pcm,ax=ax,fraction=0.05)
    
    ax,cb = viz.plot_AMV_spatial(amvsp[funiform][m].T*vscale,lonr,latr,bbox_NA,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True)
    cb.set_ticks(cintslb)
    ax.set_title("AMV Pattern (%s)"%modelname[m])
    plt.savefig(outpathfig+"AMV_Spatial_Pattern_%s_%s_runid%s_vscale%i.png"%(modelfname[m],fnames[0],runid,vscale))
    
    
#%% Make a 3-panel Figure

region   = 3
vscale   = 1
funiform = 1.5

# # Load AMV Spatial Patterns
amvsp = {}
amvid = {}
for funiform in funiforms:
    
    # Set experiment ID
    expid = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)
    
    amvload = np.load("%sAMV_Region_%s.npz"%(outpathdat,expid),allow_pickle=True)
    
    amvsp[funiform] = amvload['amvpat_region'].item()[region] # Contains [region][model]
    amvid[funiform] = amvload['amvidx_region'].item()[region]
    
    print("Loaded in post-processed data for %s" % expid)


fig,axs = plt.subplots(1,3,figsize=(13,5),subplot_kw={"projection":ccrs.PlateCarree()})
for i,m in enumerate([1,2,3]):
    ax = axs.flatten()[i]
    if m == 4:
        vscale=1
    #fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    
    
    #ax = viz.init_map(bbox_NA,ax=ax)
    #pcm = ax.contourf(lonr,latr,amvsp[i].T,levels=cints,cmap=cmap)
    #fig.colorbar(pcm,ax=ax,fraction=0.05)
    ax,cs = viz.plot_AMV_spatial(amvsp[funiform][m].T*vscale,lonr,latr,bbox_NA,cmap,cint=cints,ax=ax,fmt="%.2f",fontsize=12,omit_cbar=True)
    cb.set_ticks(cintslb)
    ax.set_title("%s"%modelnamenew[m],fontsize=14)
cb = fig.colorbar(cs,ax=axs.ravel().tolist(),orientation='horizontal',shrink=0.5,pad=0.1)
cb.set_ticks(cintslb)
plt.suptitle("AMV SST' Patterns from the Stochastic Model ($^{\circ}C / \sigma_{AMV}$)",fontsize=16)
#plt.tight_layout()
plt.savefig(outpathfig+"AMV_Spatial_Pattern_3panel_%s_runid%s_vscale%i.png"%(fnames[0],runid,vscale),bbox_inches='tight')


#%% Make similar plots from other datasets

# AMV (CESM-SLAB)



