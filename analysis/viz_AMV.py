#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:20:26 2020

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs
import cartopy

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
from amv import proc,viz
from matplotlib import gridspec




datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/SLAB_PIC/"
outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/02_Figures/Weekly_Meetings/"



zz = np.load('%sEOF_AMV_PIC_SLAB.npz'%datpath,allow_pickle=True)
amvpat = zz['patterns']
amvid  = zz['indices']
aavgs  = zz['aavg']
lon    = zz['lon']
lat    = zz['lat']
times  =  zz['times']
anames = zz['amvnames']

# Plot AMV
# Plot settings
bbox = [280-360, 0, 0, 65]
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
#cint = np.arange(-1,1.1,0.1)
cint = np.arange(-0.5,0.55,0.05)


i= 0


#Plot Spatial AMV
fig,ax = plt.subplots(1,1,figsize=(5,4),subplot_kw={'projection':ccrs.PlateCarree()})
ax = viz.plot_AMV_spatial(amvpat[i].T,lon,lat,bbox,cmap,cint=cint,ax=ax)    
#ax.set_title("CESM-SLAB AMV Pattern (PIC, %s Index)" %(anames[i]))
ax.set_title("CESM-SLAB AMV Pattern (Pre-industrial Control 101-1001)")
#plt.tight_layout
plt.savefig("%sAMV_Pattern_PIC_SLAB_%s_Teleconf.png"%(outpath,anames[i]),dpi=200)

# Plot AMV Index
fig,ax = plt.subplots(1,1,figsize=(6,2))
plt.style.use('seaborn')
xtks = np.arange(0,int(len(times)/12),1)
ax = viz.plot_AMV(amvid[i].squeeze(),ax=ax)
ax.set_title("AMV Index; CESM-SLAB (PIC, %s Index)" %(anames[i]))
ax.set_xlabel("Year")
ax.set_ylabel("AMV Index")
plt.savefig("%sAMV_Index_PIC_SLAB_%s.png"%(outpath,anames[i]),dpi=200)


 # e = 12
 #    xtks = np.arange(0,nmon,120)
 #    xlb = np.arange(int(start[0:4]),int(end[0:4]),10)
 #    fig,ax=plt.subplots(1,1)
 #    ax = amv.plot_AMV(amvidx[e,:].squeeze(),ax=ax)
 #    ax.set_xticks(xtks)
 #    ax.set_xticklabels(xlb)
 #    ax.set_xlabel("Year")
 #    ax.set_ylabel("AMV Index")
 #    ax.set_title("AMV Index for Ens. Member %i, CESMLE %i-%s, Detrend %i; Filter %i" % (e,xlb[0],end[0:4],deg,lpf))





