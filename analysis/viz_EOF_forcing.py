#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Visualize some forcing parameters from the stochastic model (basinwide)

This was created for the Stochastic Model Paper Outline

Created on Tue Oct 12 11:28:34 2021

@author: gliu
"""

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx

from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
from scipy import signal
from tqdm import tqdm

import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import calendar as cal
import matplotlib.gridspec as gridspec

import scm
import time
import cmocean


#%% 

projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath    = projpath + '01_Data/'
outpath    = projpath + '02_Figures/20220518/'
input_path  = datpath + 'model_input/'
proc.makedir(outpath)

# Put slab version first, then the load_load func. searches in the same
# directory replace "SLAB_PIC" with "FULL_PIC"
Fprimename = "_Fprime_rolln0" # Set to None to use Qnet forcing

frcname    = "flxeof_090pct_SLAB-PIC_eofcorr2"
if Fprimename is not None:
    frcname += Fprimename
    #frcame = proc.addstrtoext(frcname,Fprimename)


#%% Some utilities

#%% Load the damping

# Use the function used for sm_rewrite.py
inputs = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs

# Load the slab mld
hblt = np.load(input_path + "SLAB_PIC_hblt.npy")

lon180=lon

mons3 = proc.get_monstr(3)
#%% Load EOF based forcing

# Damping parameters
mcfs = ["SLAB-PIC","FULL-PIC"]
dampings = [dampingslab,dampingfull]

# EOF Parameters
N_mode = 200
bbox    = [260,20,0,65]
bboxtext = "lon%ito%i_lat%ito%i" % (bbox[0],bbox[1],bbox[2],bbox[3])
vthres  = 0.90

# Plotting for the paper outline
bboxplot = [-85,5,5,60]
bbox_NA  = [-80,0 ,10,65]

# Load teh data
ampfactors = []
qcorrs     = []
thresids = []
eofalls = []
eofslps = []
for model in tqdm(range(2)):
    
    mcname = mcfs[model]
    
    # Convert Damping to calculate q-corr
    damping = dampings[model]
    if model == 0:
        h_in = h.copy()
    else:
        h_in = hblt.copy()
    lbd_a = scm.convert_Wm2(damping,h_in,dt=3600*24*30)
    corr  = 1/(scm.method2(lbd_a,original=False))
    qcorrs.append(corr)
    
    # Load Data (EOFs)
    # ----------------
    savename  = "%sNHFLX_%s_%iEOFsPCs_%s.npz" % (input_path,mcname,N_mode,bboxtext)
    if Fprimename is not None:
        savename = proc.addstrtoext(savename,Fprimename)
    ld        = np.load(savename,allow_pickle=True)
    eofall    = ld['eofall']
    eofslp    = ld['eofslp']
    #pcall     = ld['pcall']
    varexpall = ld['varexpall']
    eofalls.append(eofall)
    eofslps.append(eofslp)

    # Load Data (Variance Corrections)
    # --------------------------------
    savename1 = input_path + "../NHFLX_EOF_Ratios_%s.npz" % mcname
    if Fprimename is not None:
        savename1 = proc.addstrtoext(savename1,Fprimename)
    ld = np.load(savename1)
    varflx_ratio = ld['varflx_ratio']
    nlon,nlat,_,_ = varflx_ratio.shape
    # varflx_EOF = ld['varflx_EOF']
    # varflx_ori = ld['varflx_ori']
    # varflx_ratio = ld['varflx_ratio']
    # vratio_alone = ld['varflx_ratio_alone']
    # eof_corr     = eofall * 1/vratio_alone

    # Calculate cumulative variance at each EOF
    # ----------------------------------------
    cvarall = np.zeros(varexpall.shape)
    for NN in range(N_mode):
        cvarall[NN,:] = varexpall[:NN+1,:].sum(0)
        
    # Find Indices of variance threshold
    # ----------------------------------
    thresid = np.argmax(cvarall>vthres,axis=0)
    thresperc = []
    for i in range(12):
        print("Before")
        print(cvarall[thresid[i]-1,i])
        print("After")
        print(cvarall[thresid[i],i])
        
        # Append percentage
        thresperc.append(cvarall[thresid[i],i])
    thresperc = np.array(thresperc)
    thresids.append(thresid)
    
    # Calculate correction factor
    # ---------------------------
    thresperc = np.zeros([nlon,nlat,12])*np.nan
    for im in range(12):
        thresidx = thresid[im]
        thresperc[:,:,im] = varflx_ratio[:,:,im,thresidx] # Index for each month
    ampfactor = 1/thresperc
    ampfactors.append(ampfactor)

#%% Make a few plots
debug = False

# Amplification factor
def plot_ampfactor(lon,lat,ampfactor,bboxplot,bbox_NA,clvls=np.arange(1,1.55,.05),
                   ax=None,title=True,cbar=True,plotbbox=True,cpad=0.05,plotid=None,blabel=[1,0,0,1]):
    if ax is None:
        ax=plt.gca()
    # ampfactpr = [lon x lat x time]
    ax       = viz.add_coast_grid(ax=ax,bbox=bboxplot,fill_color='gray',blabels=blabel)
    
    if plotid is None:
        plotvar = ampfactor.mean(-1).squeeze().T # Plot the mean
    else:
        plotvar = ampfactor[:,:,plotid].mean(-1).squeeze().T
        
    
    pcm      = ax.contourf(lon,lat,plotvar,levels=clvls,cmap="Oranges",extend='both')
    cl       = ax.contour(lon,lat,plotvar,levels=clvls,colors='k',linewidths=.2)
    if plotbbox:
        ax,ll = viz.plot_box(bbox_NA,ax=ax,leglab="AMV",
                             color="k",linestyle="dashed",linewidth=2,return_line=True)
    ax.clabel(cl,levels=np.arange(1,1.5,.1),fontsize=8)
    if cbar:
        cb       = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.055,pad=cpad)
    if title:
        ax.set_title(r"Annual Mean Variance Correction ($Q_{net}/Q_{EOF}$)"+ "\nContour Interval = 0.05")
    return ax

# EOF bar graph
def plot_eofbar(mons3,thresid,vthres,ax=None,title=True):
    # thresid = [mon x 12]
    if ax is None:
        ax=plt.gca()
    ax.bar(mons3,thresid,color=[0.56,0.90,0.70],alpha=0.80)
    if title:
        ax.set_title("Number of EOFs required \n to explain %i"%(vthres*100)+"% of the $Q_{net}$ variance")
    ax.set_ylabel("# EOFs")
    ax.grid(True,ls='dotted')
    
    # Label Everything
    rects = ax.patches
    labels = thresid
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + -5, label, ha="center", va="bottom"
        )
    return ax
#% Plot the correction
def plot_qcorr(lon,lat,qcorr,bboxplot,clvls=np.arange(1,1.55,.05),ax=None,
               title=True,cbar=True,plotbbox=True,cpad=0.05):
    if ax is None:
        ax = plt.gca()
    ax       = viz.add_coast_grid(ax=ax,bbox=bboxplot,fill_color='gray')
    pcm      = ax.contourf(lon,lat,qcorr.T,levels=clvls,cmap="Blues",extend='both')
    cl       = ax.contour(lon,lat,qcorr.T,levels=clvls,colors="k",linewidths=0.2)
    if plotbbox:
        ax,ll = viz.plot_box(bbox_NA,ax=ax,leglab="AMV",
                             color="k",linestyle="dashed",linewidth=2,return_line=True)
    ax.clabel(cl,levels=np.arange(1,1.5,.1),fontsize=8)
    if cbar:
        cb       = fig.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.055,pad=cpad)
    if title:
        ax.set_title(r"Ratio of Stochastic Forcing to $Q_{net}$" + "\n Contour Interval = 0.05")
    return ax

#%% Test out separately
if debug:
    fig,ax   = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4))
    plot_ampfactor(lon,lat,ampfactors[model],bboxplot,bbox_NA)
    
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    plot_eofbar(mons3,thresid,vthres)
    
    fig,ax   = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,4))
    plot_qcorr(lon,lat,qcorrs[model].mean(-1),bboxplot,ax=ax)


#%% Plot altogether in a subplot ()

proj = ccrs.PlateCarree()
fig  = plt.figure(figsize=(16,10))
gs   = gridspec.GridSpec(2,3)

for model in range(2):
    
    tbool = True if model == 0 else False
    cbool = True if model == 1 else False
    
    
    # Bar Plot
    ax1 = plt.subplot(gs[model,0:1])
    ax1 = plot_eofbar(mons3,thresids[model],vthres,ax=ax1,title=tbool)
    

    # Q-corr
    ax3 = plt.subplot(gs[model,2],projection=proj)
    plot_qcorr(lon,lat,qcorrs[model].mean(-1),bboxplot,ax=ax3,title=tbool,cbar=cbool)
    
# Add CESM Slab and FULL Labels
fig.text(0.07, 0.70, 'SLAB', va='center', rotation='vertical',fontsize=16)
fig.text(0.07, 0.30, 'FULL', va='center', rotation='vertical',fontsize=16)
figname = "%sFig04_Forcing_Corrections.png"% (outpath)
if Fprimename is not None:
    figname = proc.addstrtoext(figname,Fprimename)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% For Fprime (no lambda-correction, just plot 2 things)

proj = ccrs.PlateCarree()
fig  = plt.figure(figsize=(14,10),constrained_layout=True)
gs   = gridspec.GridSpec(2,2)

for model in range(2):
    
    tbool = True if model == 0 else False
    cbool = True if model == 1 else False
    
    # Bar Plot
    title1 = "# of EOFs required \n to explain %i"%(vthres*100)+"% of $F'$ variance"
    ax1 = plt.subplot(gs[model,0:1])
    ax1 = plot_eofbar(mons3,thresids[model],vthres,ax=ax1,title=False)
    
    # Amp Factor
    title2 = r"Local Variance Correction ($F' : F'_{EOF}$)" + "\n Contour Interval = 0.05"
    ax2 = plt.subplot(gs[model,1],projection=proj)
    ax2 = plot_ampfactor(lon,lat,ampfactors[model],bboxplot,
                         bbox_NA,ax=ax2,title=False,cbar=cbool,plotbbox=False,cpad=0.08)
    
    if model == 0:
        ax1.set_title(title1)
        ax2.set_title(title2)
    
    # # Q-corr
    # ax2 = plt.subplot(gs[model,2],projection=proj)
    # plot_qcorr(lon,lat,qcorrs[model].mean(-1),bboxplot,ax=ax3,title=tbool,cbar=cbool)
    
# Add CESM Slab and FULL Labels
fig.text(0.07, 0.70, 'CESM-SLAB', va='center', rotation='vertical',fontsize=16)
fig.text(0.07, 0.30, 'CESM-FULL', va='center', rotation='vertical',fontsize=16)
figname = "%sFig04_Forcing_Corrections_noampq.png"% (outpath)
if Fprimename is not None:
    figname = proc.addstrtoext(figname,Fprimename)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% Quickly plot out the amplification factor for all months

fig,axs = plt.subplots(4,3,figsize=(12,10),
                       subplot_kw={'projection':proj},constrained_layout=True)

mplot = np.roll(np.arange(0,12),1)

sname = ["Winter","Spring","Summer","Fall"]
for i in range(12):
    
    ax = axs.flatten()[i]
    im = mplot[i]
    
    blabel = [0,0,0,0]
    if i%3 == 0:
        blabel[0] = 1
        s = int(i/3)
        ax.text(-0.24, 0.5, '%s'% (sname[s]), va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes)
        
    if i>8:
        blabel[-1] = 1
    #title2 = r"Local Variance Correction ($F' : F'_{EOF}$)" + "\n Contour Interval = 0.05"
    
    ax = plot_ampfactor(lon,lat,ampfactors[model],bboxplot,
                         bbox_NA,ax=ax,title=False,cbar=False,plotbbox=False,
                         cpad=False,plotid=[im,],blabel=blabel)
    #ax.set_title(mons3[im])
    viz.label_sp(mons3[im],ax=ax,labelstyle="%s",alpha=0.75,usenumber=True)
    
    
figname = "%sFig04_Variance_Correction_amq.png"% (outpath)
if Fprimename is not None:
    figname = proc.addstrtoext(figname,Fprimename)
plt.savefig(figname,dpi=150,bbox_inches='tight')

#%% -- -- -- -- -- -- -- -- How does NAO/EOF look like after each correction?


# indicate selections
im    = 0 # Month Index
model = 1 # Model Index 
N     = 0 # MODE Index

# plotting params
clvl = np.arange(-60,65,5)

# Set some strings
vizstring    = "%s_Month%02i_Mode%03i" % (mcfs[model],im+1,N+1)


titlestrings  = ("EOF %02i (no correction)" % (N+1),
                "With Local Variance Correction (to 100%)",
                "With Local Variance Correction and Q-correction")
                #Note, add variance explained later


# Select model
eofin   = eofalls[model][:,:,im,N]
ampin   = ampfactors[model][:,:,im]
qcorrin = qcorrs[0][:,:,im]



#init plot
def plot_NAOpat(lon,lat,eofpat,clvl,title,bbox,ax=None,cmap='PRGn'):
    if ax is None:
        ax = plt.gca()
    ax  = viz.add_coast_grid(ax,bbox=bbox)
    pcm = ax.contourf(lon,lat,eofpat,levels=clvl,cmap=cmap,extend='both')
    cl  = ax.contour(lon,lat,eofpat,levels=clvl,colors="k",linewidths=0.5)
    ax.clabel(cl,fontsize=10)
    ax.set_title(title)
    return pcm,ax



fig,axs = plt.subplots(1,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(18,4))

for ip in range(3):
    
    ax = axs.flatten()[ip]
    
    
    if ip == 0:
        plotvar = eofin.T
    elif ip == 1:
        plotvar = (eofin*ampin).T
    else:
        plotvar = (eofin*ampin*qcorrin).T
        
    pcm,ax = plot_NAOpat(lon,lat,plotvar,clvl,titlestrings[ip],bbox,ax=ax)

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.015)
cb.set_label("$Q_{net}$ ,Interval = 5 $W/m^2$ per $\sigma_{PC}$")


plt.savefig("%sEOF_Variance_Correction_effects_%s.png" % (outpath,vizstring),bbox_inches='tight',dpi=200)
#%% Test PLot Qcorr

    
fig,ax   = plt.subplots(1,1,constrained_layout=True,
                        subplot_kw={'projection':ccrs.PlateCarree()},figsize=(8,6))
plot_qcorr(lon,lat,qcorrs[model].mean(-1),bboxplot,ax=ax)
ax.plot(-36,58,markersize=20,marker="x",color="yellow")
plt.savefig("%sQcorr_Plotpt.png"%(outpath),dpi=150)

