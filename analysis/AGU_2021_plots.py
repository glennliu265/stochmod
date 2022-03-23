#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plots for my poster presentation in AGU 2021

Created on Fri Feb 25 11:59:00 2022

@author: gliu
"""

import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
from scipy.interpolate import interp1d
from tqdm import tqdm
import scm
import time
import cartopy.crs as ccrs
from scipy import signal
import calendar as cal
#%% User Edits (from viz_synth_stochmod_combine.py)


projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20220128/'
input_path  = datpath + 'model_input/'
proc.makedir(outpath)

# Put slab version first, then the load_load func. searches in the same
# directory replace "SLAB_PIC" with "FULL_PIC"
frcname = "flxeof_090pct_FULL-PIC_eofcorr2"

# Which point do you want to visualize conditions for?
lonf = -55#-30
latf = 11 #50
flocstring = "lon%i_lat%i" % (lonf,latf)
locstring = "%i$\degree$N, %i$\degree$W" % (latf,np.abs(lonf))

# Additional Plotting Parameters
bbox = [-80,0,10,65]

# # Load Slab MLD
hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")

# Load Old Forcing
flxstd = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")

# Get mons3 from calendar function
mons3 = [cal.month_abbr[i] for i in np.arange(1,13,1)]

# Load limask
limask = np.load(datpath+"model_input/limask180_FULL-HTR.npy")

# # Regional Analysis Settings (NEW, with STG Split)
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


#%% Load Data


#%% AGU Style Plot (Vertically Stacked)

fig,axs     = plt.subplots(2,1,figsize=(6,8),sharex=True,sharey=True)

# Plot Lower Hierarchy
ax = axs[0]

title = r"Adding Varying Damping ($\lambda_a$) and Forcing ($\alpha$)"
title = ""
plotacs = c_acs
model   = 1

# Option to add tiled variable
addvar  = False
plotvar = Fpt
ylab    = "Forcing ($W/m^{2}$)"
#plotvar = Fpt/np.roll(Fpt,1) # Plot Ratio Current Month/Prev Month
#ylab    = "Forcing Ratio (Current/Previous)"
#plotvar = Fpt - np.roll(Fpt,1)
plotvar = damppt
ylab =  "Atmospheric Damping ($W/m^{2}$)"

xtk2       = np.arange(0,37,2)
if addvar:
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=plotvar)
    ax3.set_ylabel(ylab)
    ax3.yaxis.label.set_color('gray')
else:
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=4)
#ax.scatter(lags,cesmauto2[lags],10,label="",color='k')
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)
#ax.grid(minor=True)

for i,e in enumerate([0,1,2,3]):
    
    title=""
    ax.set_ylabel("")
    
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label=labels_lower[i],color=ecol_lower[i],ls=els_lower[i],marker="o",markersize=4)
    #ax.scatter(lags,plotacs[e][model],10,label="",color=ecol[i])
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color=ecol_lower[i],alpha=0.2)
    
    ax.legend(fontsize=10,ncol=3)
ax.set_ylabel("")
fig.text(0.04,0.5,'Correlation',va='center', rotation='vertical')


# --------------------------------------------

# Plot Upper Hierarchy
ax = axs[1]
#title = "Adding Varying Mixed Layer Depth ($h$) and Entrainment"
title = ""
#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)

# Plot CESM Data
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
#ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='gray',marker="o",markersize=3)
#ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

for i in range(2,4):
    ax.plot(lags,ac[i],label=labels_upper[i],color=ecol_upper[i],ls=els_upper[i],marker="o",markersize=3)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=ecol_upper[i],alpha=0.25)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
ax.set_ylabel("")

plt.suptitle("Monthly SST Autocorrelation at 50$\degree$N, 30$\degree$W",fontsize=12,y=1.01)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

plt.savefig(outpath+"Autocorrelation_2-PANEL_%s_vertical.png"%locstring,dpi=200,bbox_inches='tight')