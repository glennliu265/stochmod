#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV from HadISST Data
Also save detrended data for stochastic model comparison

Created on Sat Aug 22 21:05:08 2020

@author: gliu
"""


from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs
import xarray as xr


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

#%% User Edits

startyr = 1900 # Start year for AMV Analysis
bbox    =[-80,0 ,0,65] # AMV bbox
# Path to SST data from obsv
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20210524/'
proc.makedir(outpath)
datpath = projpath + '01_Data/'
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"


## DETRENDING OPTIONS
method = 2


#%% Load in HadISST Data

# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")
hlat   = np.squeeze(obvhad['LAT'])
hlon   = np.squeeze(obvhad['LON'])
hyr    = obvhad['YR']
hsst   = obvhad['SST']

# Change hsst to lon x lat x time
hsst = np.transpose(hsst,(2,1,0))

# Take the set time period
monstart = (startyr+1-hyr[0,0])*12
hsst = hsst[:,:,monstart::]
nyrs = int(hsst.shape[2]/12)

#%% Fix Latitude Dimensions for HSST
# currently it is arranged 90:-90, need to flip to south first

# Find north and south latitude points
hsouth = np.where(hlat <= 0)
hnorth = np.where(hlat > 0)

# Find corresponding points in data
hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]

# Stitch things together, reversing the order 
hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)

#%% Remove Seasonal Cycle first and plot

dsfirst = np.reshape(hsstnew,(360,180,nyrs,12))
dsfirst = dsfirst - np.mean(dsfirst,axis=2)[:,:,None,:]
dsfirst = np.reshape(dsfirst,(360,180,hsstnew.shape[2]))

# Detrend
# start= time.time()
# dtdsfirst,dsymodall,_,_ = proc.detrend_dim(dsfirst,2)
# print("Detrended in %.2fs" % (time.time()-start))


# Detrend
nlon = 360
nlat = 180
nmon = nyrs*12
start= time.time()
indata = dsfirst.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(indata,1)
x = np.arange(0,nmon,1)

if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T

# Replace back into dataset
dtdata = np.zeros((nlon*nlat,nmon))*np.nan
dtdata[okpts,:] = okdt
dtdata = dtdata.reshape(nlon,nlat,nmon)
print("Detrended in %.2fs" % (time.time()-start))

# # Plot Seasonal Cycle Removal and Detrended
lonf = -30
latf = 64
tper = np.arange(0,hsstnew.shape[2])
klon,klat = proc.find_latlon(lonf,latf,hlon,hlatnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdata[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

hlat = hlatnew.copy()
hsst = dtdata.copy()
#hsst = dsfirst.copy()

# Save data (MONTHLY)
hadname  = "%sHadISST_detrend%i_startyr%i.npz" % (datpath,method,startyr)
np.savez(hadname,**{
    'sst':dtdata,
    'lat':hlatnew,
    'lon':hlon,
    'yr':hyr},allow_pickle=True)


h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)

#%% Try Another Calculation Method
#Methods
# 0) Regress anomaly onto global mean
# 1...N) Remove N-degree polynomial



# Get timedim 
nyr = int(hsst.shape[2]/12)
x = np.arange(0,nyr,1)

# First get annual averaged data
hsstann = proc.ann_avg(hsstnew,2)
hsstann = hsstann - hsstann.mean(2)[:,:,None]

# Get nan points
hsstann = hsstann.reshape(360*180,nyr)
okdata,knan,okpts = proc.find_nan(hsstann,1)


if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T
    

# Replace back into data
hsstdt = np.ones((360*180,nyr)) * np.nan
hsstdt[okpts,:] = okdt
hsstdt = hsstdt.reshape(360,180,nyr)


 


# Use this data to calculate amv
h_amv,h_regr = proc.calc_AMVquick(hsstdt,hlon,hlat,bbox,anndata=True)



#%% Plot AMV

addbox = 1
# Set regions for analysis
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = bbox#[-75,20,0,90]
    

#regions = ("SPG","STG","TRO","NAT")
#bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)


#% Make AMV Spatial Plots
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)
#clab = cint
fig,axs = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   
scolors  = ["black","cornflowerblue","crimson","gold"]

# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color=scolors[0],return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")

ax,l2 = viz.plot_box(bbox_ST,ax=ax,color=scolors[2],return_line=True,leglab='STG',linewidth=lwb)
ax,l1 = viz.plot_box(bbox_SP,ax=ax,color=scolors[1],return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=scolors[3],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%% Mini HadiSST AMV Plot NAT

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-1,1.1,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(4,3),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs,fontsize=8)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=12)   


# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
#ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_NATOnly_miniplot.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)


#%%  Make AMV Spatial Plots, but only show the N.Atl Bounding Box

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   


# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
#ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_NATOnly.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%%  Make AMV Spatial Plots, but only show the NSPG Bounding Box

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(4,3),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   


# Add region plots
ax = axs
lwb = 1.5
#ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_TROOnly.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)



#%% Plot AMV INdex

xtks = np.arange(0,140,20)
xtkl = np.arange(startyr,startyr+140,20)
fig,ax = plt.subplots(1,1,figsize=(5,1))
ax = viz.plot_AMV(h_amv,ax=ax)
ytks = np.arange(-0.5,0.75,0.25)
    
ax.set_xticks(xtks)
ax.set_xticklabels(xtkl)
ax.set_ylim(-.5,.5)
ax.set_yticks(ytks)
ax.set_title("HadISST AMV Index (%s to %s)" % (startyr,hyr[0,-1]),fontsize=10) 
outname = '%sHadISST_AMVIDX_%s-%s_dtmethod%i.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)
    

#%% UPDATE (MAY 24th 2021, Plots for Generals Presentation)

def plot_AMV_generals(lat,lon,amvpattern,vscale=1):
    """
    Customized AMV Plot for Generals Presentation (for consistent plotting)
    
    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    lon : TYPE
        DESCRIPTION.
    amvpattern : [lon x alt]
        DESCRIPTION.
    vscale : INT
        Amt of times to scale AMV pattern by
    Returns
    -------
    None.
    """
    bbox = [-80,0 ,0,65]
    
    # Set up plot params
    plt.style.use('default')
    cmap = cmocean.cm.balance
    cints = np.arange(-.55,.60,.05)
    cintslb = np.arange(-.50,.6,.1)
    
    # Make the plot
    fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    ax,cb = viz.plot_AMV_spatial(amvpattern.T*vscale,lon,lat,bbox,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True,
                                 fontsize=8)
    cb.set_ticks(cintslb)
    return fig,ax,cb

fig,ax,cb = plot_AMV_generals(hlat,hlon,h_regr)
ax.set_title("AMV Pattern (HadISST; 1900 to 2018) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"HadISST_AMV_Spatial_Pattern_%i_to_2018_detrend%i.png"%(startyr,method),bbox_inches='tight')

#%% 

    


    