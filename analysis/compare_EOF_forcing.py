#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Compare simple qcorrection vs. theoretical correction

Created on Mon Feb 14 14:37:24 2022


1) Maps of Variance for Fprime and Amplified Qnet (annual average)
- Ratio Maps
- Overall Variance
- Low frequency variance?

2) Comparison of Spectra and seasonal cycle at 50N,30W

3) Impact on SST (after model run with new forcing)

4) How does this impact our results, if at all?


@author: gliu

"""

import xarray as xr
import numpy as np
import glob
import time

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from tqdm import tqdm

import cmocean
#%%
stormtrack = 0

if stormtrack == 1:
    # Module Paths
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
elif stormtrack == 0:
    # Module Paths
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    
    
    datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    #datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/"
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220113/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    
    
from amv import proc,viz
import scm

#%% User Edits

projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
figpath     = projpath + "02_Figures/20220214/"
proc.makedir(figpath)

mconfig = "FULL"


lonf = -30
latf = 50

mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
vlabels  = ["$Q_{net} \, (Corrected)$","$F'$"]
summodes = False # Set to True to load the summed modes

plt.style.use('default') 


#%% Use the function used for sm_rewrite.py, load in lambda
# [lon x lat x month]

frcname    = "flxeof_090pct_FULL-PIC_eofcorr2"
input_path = datpath
inputs     = scm.load_inputs('SLAB_PIC',frcname,input_path,load_both=True)
lon,lat,h,kprevall,dampingslab,dampingfull,alpha,alpha_full = inputs
hblt = np.load(datpath + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]
if mconfig == 'SLAB':
    lbd = dampingslab
    h_in   = hblt
elif mconfig == 'FULL':
    lbd    = dampingfull
    h_in   = h
    

def correct_forcing(F,lbd,h,dt=3600*24*30):
    
    # Convert from Wm2 to deg/sec
    lbd_a   = scm.convert_Wm2(lbd,h,dt)
    F       = scm.convert_Wm2(F,h,dt) # [lon x lat x time]
    
    # Apply correction
    underest = scm.method2(lbd_a.mean(2)[...,None],original=False)
    t_end = F.shape[2]
    ntile = int(t_end/F.shape[2])
    ampmult = np.tile(1/np.sqrt(underest),ntile)
    Fnew = F * ampmult
    
    # Convert back to W/m2
    Fnew = scm.convert_Wm2(Fnew,h,dt,reverse=True)
    return Fnew

# Tile Lambda
#lbd = np.tile(lbd,nyrs)


#%% Load in forcing






#%% Load in EOFs

if summodes:
    Qnetname   = "flxeof_090pct_%s-PIC_eofcorr2.npy" % mconfig
    Fprimename = proc.addstrtoext(Qnetname,"_Fprime")
    sumaxis  = 0
    modeaxis = -2
else:
    # Note --> These look funky... gotta check them
    #Qnetname   = "NHFLX_%s-PIC_200EOFsPCs_lon260to20_lat0to65_eofcorr2.npz" % mconfig
    
    Qnetname = "NHFLX_%s-PIC_200EOFsPCs_lon260to20_lat0to65.npz" % mconfig
    Fprimename = proc.addstrtoext(Qnetname,"_Fprime_rolln0")
    sumaxis = 1
    modeaxis = -1

eofs    = []
pcs     = []
varexps = []
for i,fname in tqdm(enumerate([Qnetname,Fprimename])):
    
    if summodes:
        eof = np.load(datpath + fname)
        eofs.append(eof)
        
        lon,lat = scm.load_latlon()
    else:
        
        npz = np.load(datpath+fname,allow_pickle=True)
        if i == 0:
            lon = npz['lon']
            lat = npz['lat']
        eofall = npz['eofall']
        pcall  = npz['pcall']
        
        #%Flip sign to match NAO+ (negative heat flux out of ocean/ -SLP over SPG) ----
        spgbox     = [-60,20,40,80]
        eapbox     = [-60,20,40,60] # Shift Box west for EAP
        
        N_modeplot = 5
        for N in tqdm(range(N_modeplot)):
            if N == 1:
                chkbox = eapbox # Shift coordinates west
            else:
                chkbox = spgbox
            for m in range(12):

                sumflx = proc.sel_region(eofall[:,:,[m],N],lon,lat,chkbox,reg_avg=True)

                if sumflx > 0:
                    print("Flipping sign for NHFLX, mode %i month %i" % (N+1,m+1))
                    eofall[:,:,m,N]*=-1
                    pcall[N,m,:] *= -1

                    
        eofs.append(eofall) # [lon x lat x month x mode] (or mode x month for summodes)
        pcs.append(pcall)
        varexps.append(npz['varexpall'])

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring = "Lon %.f, Lat %.f" % (lonf,latf)
#%% Apply amplitude correction based on lambda

nmode   = eofs[0].shape[modeaxis]
eofcorr = np.zeros(eofs[0].shape)

for N in tqdm(range(nmode)):
    if summodes:
        Fin = eofs[0][:,:,N,:] # [lon x lat x month]
        eofcorr[:,:,N,:] = correct_forcing(Fin,lbd,h_in)
    else:
        Fin = eofs[0][:,:,:,N] # [lon x lat x month]
        eofcorr[:,:,:,N] = correct_forcing(Fin,lbd,h_in)
    
eofuncorr = eofs[0].copy()
eofs[0] = eofcorr

#np.nanmax(np.abs(eofcorr-eofuncorr))
    
#%% Example Seasonal Cycle in forcing at selected point


colors = ("mediumblue","firebrick","magenta")
lw = 2
fig,ax = plt.subplots(1,1)
eofrss_all = []
for i in range(2):
    if i == 1:
        ls = 'dashed'
    else:
        ls = 'solid'
    
    eofrss = np.linalg.norm(eofs[i][klon,klat,...],axis=sumaxis) #[month]
    ax.plot(mons3,eofrss,label=vlabels[i],ls=ls,lw=lw,color=colors[i])
    eofrss_all.append(eofrss)

eofrss = np.linalg.norm(eofuncorr[klon,klat,...],axis=sumaxis,) #[month]
#ax.plot(mons3,eofrss,label="$Q_{net}$ (Uncorrected)",ls='dotted',lw=lw)


ax.legend()
ax.grid(True,ls='dotted')
ax.set_title("Stochastic Forcing Amplitude @ %s"% locstring)
plt.savefig("%sEOF_Forcing_Amplitudes_Fprime_v_Qnet.png"% (figpath),dpi=150)


#  Plot the Ratio
fig,ax = plt.subplots(1,1)
ax.plot(mons3,eofrss_all[1]/eofrss_all[0])
ax.grid(True,ls='dotted')
ax.set_title("Ratio \n Qnet (Corrected) / F'")
ax.set_ylim([0.75,1.25])


#%% Compare the EOF Patterns
# eofs are [lon x lat x mon x mode]

N_mode = 0
im     = 0

for im in tqdm(range(12)):
    cint = np.arange(-80,85,5)
    
    bboxplot   = [-100,20,0,65]
    
    fig,axs = plt.subplots(1,2,figsize=(12,6),constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()})
    for i in range(2):
        ax = axs[i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        
        if i < 2:
            cf = ax.contourf(lon,lat,eofs[i][:,:,im,N_mode].T,levels=cint,cmap='cmo.balance',
                             extend='both')
            ax.set_title(vlabels[i])
        else:
            cf = ax.contourf(lon,lat,(eofs[1][:,:,im,N_mode]-eofs[0][:,:,im,N_mode]).T,
                             levels=cint,
                             cmap='cmo.balance')
            ax.set_title("%s - %s" % (vlabels[1],vlabels[0]))
            
        #ax.clabel(cf,colors='k')
    fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
    
    plt.suptitle("Qnet and F' for EOF %i, Month %i" % (N_mode+1,im+1),y=0.85)
    
    plt.savefig("%sQnet_v_Fprime_Monthly_EOF_Pattern_Mode%i_month%02d.png"% (figpath,N_mode+1,im+1),
                dpi=150)
#%%

N_mode = 0
im     = 0
loop   = np.arange(0,12,1)

cint = np.arange(-15,16,1)
#cint = np.arange(-85,86,1)

for im in tqdm(loop):
    fig,ax = plt.subplots(1,1,figsize=(12,6),constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()})
    
    ax = viz.add_coast_grid(ax,bbox=bboxplot)
    
    cf = ax.contourf(lon,lat,(eofs[1][:,:,im,N_mode]-eofs[0][:,:,im,N_mode]).T,
                     levels=cint,
                     cmap='cmo.balance')
    ax.set_title("%s - %s" % (vlabels[1],vlabels[0]))
            
        #ax.clabel(cf,colors='k')
    fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.045)
    
    plt.suptitle("EOF %i, Month %i" % (N_mode+1,im+1),y=1.05)
    plt.savefig("%sQnet_v_Fprime_Monthly_EOF_Pattern_Mode%i_month%02d_diff.png"% (figpath,N_mode+1,im+1),
                dpi=150)
# -----------------------------------------------------------------------------
#%% Examine the Actual Amplitude of Forcing

eofrss_all = []
for i in range(2):
    eofrss_all.append(np.linalg.norm(eofs[i],axis=3))
eofrss_all = np.array(eofrss_all) # [forcing, lon, lat, month]

bboxplot   = [-100,20,0,65]
for im in range(12):
    fig,axs = plt.subplots(1,2,figsize=(12,6),constrained_layout=True,
                           subplot_kw={'projection':ccrs.PlateCarree()})
    for i in range(2):
        ax = axs[i]
        ax = viz.add_coast_grid(ax,bbox=bboxplot)
        
        plotcf = eofrss_all[1,...,im] - eofrss_all[0,...,im] # 
        ptitle = "%s - %s" % (vlabels[1], vlabels[0])
        cint   = np.arange(-20,21,1)
        
        if i == 1 :
            cint   = np.arange(-1.5, 1.6,0.1)
            plotcf = np.log(np.abs(plotcf))
            ptitle = "Log ratio (%s)" % (ptitle)
            
        #print(cint)
        cf = ax.contourf(lon,lat,plotcf.T,levels=cint,cmap='cmo.balance',extend='both')
        
        ax.set_title(ptitle)
    
        fig.colorbar(cf,ax=ax,orientation='horizontal',fraction=0.045)
    
    plt.suptitle("Qnet and F' (Month %i)" % (im+1),y=0.85)
    
    plt.savefig("%sQnet_v_Fprime_Monthly_EOF_Pattern_ALL_month%02d.png"% (figpath,im+1),
                dpi=150)
    
#%% Plot seasonal averages

# Plotting params
bboxplot   = [-100,20,0,65]
cint   = np.arange(-10,10.5,.5)

# Calcualte differences and seasonal average
eofrss_diff  = eofrss_all[1,...] - eofrss_all[0,...]
savgs,snames = proc.calc_savg(eofrss_diff,debug=True,return_str=True)

fig,axs = plt.subplots(2,2,figsize=(12,8),constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})
for i in range(4):
    
    blabel = [0,0,0,0]
    if i > 1:
        blabel[-1] = 1
    if i in [0,2]:
        blabel[0]  = 1
    
    ax = axs.flatten()[i]
    ax.set_title(snames[i])
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blabel,fill_color='gray')
    cf = ax.contourf(lon,lat,savgs[i].T,levels=cint,cmap='cmo.balance',extend='both')
    
plt.suptitle("Seasonally-Averaged Differences in Forcing Amplitude \n %s - %s" % (vlabels[1], vlabels[0]))

cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)


plt.savefig("%sQnet-v-Fprime_Savg_EOF_Pattern_Differences.png"% (figpath),
            dpi=150)