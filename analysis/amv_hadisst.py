#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV from HadISST Data

Created on Sat Aug 22 21:05:08 2020

@author: gliu
"""


from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs


import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz

#%% User Edits

startyr = 1900 # Start year for AMV Analysis
bbox    =[-80,0 ,0,65] # AMV bbox
# Path to SST data from obsv
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20200823/'
datpath = projpath + '01_Data/'
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

#%% Load in HadISST Data


# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")
hlat = np.squeeze(obvhad['LAT'])
hlon = np.squeeze(obvhad['LON'])
hyr  = obvhad['YR']
hsst = obvhad['SST']

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
start= time.time()
dtdsfirst,dsymodall,_,_ = proc.detrend_dim(dsfirst,2)
print("Detrended in %.2fs" % (time.time()-start))

# # Plot Seasonal Cycle Removal and Detrended
lonf = -30
latf = 64
tper = np.arange(0,hsstnew.shape[2])
klon,klat = proc.find_latlon(lonf,latf,hlon,hlatnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdsfirst[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

hlat = hlatnew.copy()
hsst = dtdsfirst.copy()

h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)

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
cint = np.arange(-1,1.1,0.1)
#clab = cint
fig,axs = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
plotbbox = [-100,20,-20,90]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=12)   


# Add region plots
ax = axs
lwb = 1.5
ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb)
ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb)
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb)


ax.legend([l1,l2,l3,l4],labels=regions,ncol=2,loc='upper left')


outname = '%sHadISST_AMVpattern_%s-%s.png' % (outpath,startyr,hyr[0,-1])
plt.savefig(outname, bbox_inches="tight",dpi=200)

