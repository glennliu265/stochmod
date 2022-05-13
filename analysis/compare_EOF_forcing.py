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
    outpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20220518/"

    lipath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/landicemask_enssum.npy"
    #llpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    
    
from amv import proc,viz
import scm

#%% User Edits

projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
figpath     = projpath + "02_Figures/20220518/"
proc.makedir(figpath)

mconfig      = "FULL"
correct_qnet = True

lonf = -30
latf = 50

mons3       = [viz.return_mon_label(m,nletters=3) for m in np.arange(1,13)]
vlabels    = ["$Q_{net} \, (Corrected)$","$F'$"]
vlabels_fn = ("Qnet","Fprime")
summodes = False # Set to True to load the summed modes

plt.style.use('default') 

# Apply variance threshold (if vthres < 1)
vthres  = 0.90
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
eofslps = []
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
        eofslp = npz['eofslp']
        
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
                    eofslp[:,:,m,N] *= -1

                    
        eofs.append(eofall) # [lon x lat x month x mode] (or mode x month for summodes)
        pcs.append(pcall)
        eofslps.append(eofslp)
        varexps.append(npz['varexpall'])

klon,klat = proc.find_latlon(lonf,latf,lon,lat)
locstring = "Lon %.f, Lat %.f" % (lonf,latf)
#%% Apply amplitude correction based on lambda

nmode   = eofs[0].shape[modeaxis]
if correct_qnet:
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

#%% If option is set, apply the variance threshold criterion


# Calculate cumulative variance at each EOF

if vthres < 1:
    
    for f in range(2): # Loop for each forcing type
        
        # Get needed information
        varexpall = varexps[f]
        N_mode    = varexpall.shape[0]
        eofall    = eofs[f]
    
        # Calculate cumulative variance explained
        # ---------------------------------------
        cvarall = np.zeros(varexpall.shape)
        for i in range(N_mode):
            cvarall[i,:] = varexpall[:i+1,:].sum(0)
            
        # Find indices of a variance threshold
        # ------------------------------------
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
        
        # Drop EOFs beyond the threshold
        # ------------------------------
        eofforce = eofall.copy() # [lon x lat x month x pc]
        cvartest = cvarall.copy()
        for i in range(12):
            # Set all points after crossing the variance threshold to zero
            stop_id = thresid[i]
            print("Variance of %f  at EOF %i for Month %i "% (cvarall[stop_id,i],stop_id+1,i+1))
            eofforce[:,:,i,stop_id+1:] = 0
            cvartest[stop_id+1:,i] = 0
        
        eofs[f] = eofforce # !! Replace the variable 

    
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
    
#%% Plot seasonal averages of each type of forcing, and the difference

# Plotting params
bboxplot   = [-100,20,0,65]

for f in range(3):
    
    if f < 2:
        cint   = np.arange(0,110,10)
        # Get the forcing type
        eofrss_diff = eofrss_all[f,...]
        ptitle  = "Forcing Amplitude (%s)" % (vlabels[f])
        outname = "%sForcingAmplitude_%s.png" % (figpath,vlabels_fn[f])
        cmap = 'cmo.thermal_r'
    else:
        cint    = np.arange(-10,11,1)
        # Calcualte differences and seasonal average
        eofrss_diff  = eofrss_all[1,...] - eofrss_all[0,...]
        ptitle  = "Seasonally-Averaged Differences in Forcing Amplitude \n %s - %s" % (vlabels[1], vlabels[0])
        outname = "%sQnet-v-Fprime_Savg_EOF_Pattern_Differences.png"% (figpath)
        cmap = 'cmo.balance'
        
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
        cf = ax.contourf(lon,lat,savgs[i].T,levels=cint,cmap=cmap,extend='both')
        
    plt.suptitle(ptitle)
    
    cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
    
    plt.savefig(outname,
                dpi=150)



#%% Plot seasonal averages for a given mode

N_modes = np.arange(0,10,1)

# Plotting params
bboxplot   = [-100,20,0,65]
cint   = np.arange(-60,65,5)

for N_mode in N_modes:
    for f in range(2):
        
        
        # Get forcing and compute seasonal average
        eof_sel  =  eofs[f][:,:,:,N_mode]
        savgs,snames = proc.calc_savg(eof_sel,debug=True,return_str=True)
        
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
            
        plt.suptitle("Seasonally-Averaged Forcing: %s, EOF %i" % (vlabels[f], N_mode+1))
        
        cb = fig.colorbar(cf,ax=axs.flatten(),orientation='horizontal',fraction=0.045)
        cb.set_label("Forcing ($Wm^{-2}$)")
        plt.savefig("%s%s_Savg_EOF_Pattern_Differences_mode%i.png"% (figpath,vlabels_fn[f],N_mode+1),
                    dpi=150)
        
#%% Set up Plot of EOF

nplot = 5
iplot = 1 # 0=Qnet, 1=Fprime


plotmodes = np.arange(10,15,1)#np.arange(5) # np.arange(5,10)  #

fnt_title = 14

if plotmodes[0] == 0:
    cint_slp = np.arange(-500,550,50)
    cint     = np.arange(-60,65,5)
    slp_lbl  = cint_slp[::2]
elif plotmodes[0] == 5:
    cint_slp = np.arange(-200,220,20)
    cint     = np.arange(-30,33,3)
    slp_lbl  = cint_slp[::2]
elif plotmodes[0] == 10:
    cint_slp = np.arange(-100,110,10)
    cint     = np.arange(-20,22,2)
    slp_lbl  = cint_slp[::2]

fig,axs = plt.subplots(5,4,figsize=(16,12),constrained_layout=True,
                       subplot_kw={'projection':ccrs.PlateCarree()})

for N in tqdm(range(len(plotmodes))):
    iN = plotmodes[N]
    
    
    flxin = eofs[iplot][...,iN]     # Lon x Lat x Month
    slpin = eofslps[iplot][...,iN]
    
    flx_savg,snames = proc.calc_savg(flxin,return_str=True)
    slp_savg        = proc.calc_savg(slpin)
    var_savg        = proc.calc_savg(varexps[-1][iN,...]*100)
    
    for s,sname in enumerate(snames):
        
        ax = axs[N,s]
        
        # Labeling
        if N == 0:
            ax.set_title(snames[s],fontsize=fnt_title)
        blabel=[0,0,0,0]
        if s == 0:
            blabel[0] = 1
            
            ax.text(-0.24, 0.5, 'EOF %i'% (iN+1), va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes,fontsize=fnt_title)
        if N == (nplot-1):
            blabel[-1] = 1
        ax = viz.add_coast_grid(ax,bbox=bboxplot,ignore_error=True,
                                blabels=blabel,fill_color='gray')
        
        pcm = ax.contourf(lon,lat,flx_savg[s].T,levels=cint,cmap='cmo.balance',extend='both')
        cl  = ax.contour(lon,lat,slp_savg[s].T,levels=cint_slp,colors='k',linewidths=0.75)
        ax.clabel(cl,slp_lbl)
        
        ax = viz.label_sp("%.2f" % var_savg[s]+"%",labelstyle="%s",alpha=0.7,usenumber=True,ax=ax)

cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
cb.set_label("$Q_{net}$ ($W m^{-2}$), (+) downwards")
plt.savefig("%sEOF_SLP_FLX_Fprime_mode%ito%i.png" %  (figpath,plotmodes[0]+1,plotmodes[-1]+1),dpi=200,bbox_inches='tight')

#%% Compute the power spectra of each principle component


nsmooth   = 500
pct       = 0.10
dt        = 3600*24*365

xper      = np.array([100,20,10,5,2])
xtks      = 1/xper
xlm       = [xtks[0],xtks[-1]]

plotmodes = np.arange(0,5,1)
nplot     = len(plotmodes)

fig,axs = plt.subplots(nplot,1,figsize=(8,8),sharey=True)

for N in range(nplot):
    ax   = axs[N]
    iN   = plotmodes[N]
    
    pcin   = pcs[iplot][iN,:,:-1].T # Transpose to [Time x Month]
    pclist = [pcin[:,m] for m in range(12)] # Separate into a list
    
    specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(pclist,nsmooth,pct,dt=dt)
    
    for m in range(12):
        if m > 0:
            break
        ax.plot(freqs[m]*dt,specs[m]/dt,label="m=%i"% (m+1),lw=3)
        
        nfreq  = len(freqs[m])
        df     = (freqs[m][1]-freqs[m][0])*dt
        sigmaf = np.var(pclist[m])/nfreq/df
        ax.plot(freqs[m]*dt,np.ones(nfreq)*sigmaf,ls='dashed',color="k")
        
        
    if N == 0:
        ax.legend(ncol=4,fontsize=10)
    
    ax.set_xlim(xlm)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    
    
# DO montecarlo
nmc   = 1000
ntime = len(pclist[0])
for n in tqdm(range(nmc)):
    randts = np.random.normal(0,1,ntime)
    specs,freqs,_,_,_ = scm.quick_spectrum(pclist,nsmooth,pct,dt=dt)
    if n == 0:
        specsmc = np.zeros((nmc,len(specs[0])))
    specsmc[n,:] = specs[0]

specsort = np.sort(specsmc,axis=0) # Sort for each frequency
conf     = int(0.05 * nmc)
confs    = [specsort[:conf,:]/dt,specsort[-conf:,:]/dt]

#%%

nsmooth   = 200
pct       = 0.10
dt        = 3600*24*365

xper      = np.array([100,20,10,5,2])
xtks      = 1/xper
xlm       = [xtks[0],xtks[-1]]

plotmodes = np.arange(0,5,1)
nplot     = len(plotmodes)


vlms      = [1.5,2.2]

fig,axs = plt.subplots(nplot,1,figsize=(8,14),sharey=True)

for N in range(nplot):
    ax   = axs[N]
    iN   = plotmodes[N]
    
    pcin   = pcs[iplot][iN,:,:-1].T # Transpose to [Time x Month]
    pclist = [pcin[:,m] for m in range(12)] # Separate into a list
    
    specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(pclist,nsmooth,pct,dt=dt)
    
    if vlms is None:
        pcm=ax.pcolormesh(freqs[m]*dt,np.arange(1,13,1),np.array(specs)/dt,cmap="inferno",shading='nearest')
        fig.colorbar(pcm,ax=ax,fraction=0.025,pad=0.01)
    else:
        pcm=ax.pcolormesh(freqs[m]*dt,np.arange(1,13,1),np.array(specs)/dt,
                          vmin=vlms[0],vmax=vlms[-1],cmap="inferno",shading='nearest')
        if N == nplot-1:
            cb = fig.colorbar(pcm,ax=axs.flatten(),fraction=0.025,pad=0.01)
            cb.set_label("Power $(W/m^{2})^2 cpy^{-1}$")
        
    ax.text(-0.15, 0.5, 'EOF %i'% (iN+1), va='bottom', ha='center',
        rotation='horizontal', rotation_mode='anchor',
        transform=ax.transAxes,fontsize=fnt_title)
    
    ax.set_xlim(xlm)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    
    ax.grid(True,ls='dotted')
    ax.set_yticks(np.arange(1,13))
    ax.set_yticklabels(proc.get_monstr(3))

#%%



#fig,ax = plt.subplots(1,1)

#freqs[0]
    
    

#%%


