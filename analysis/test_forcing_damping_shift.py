#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test The Effect of Rolling Forcing and Damping

2024.01.17
Moved from stochmod/analysis/Debug_Forcing
Copied upper section from reemergence/stochmod_point

Created on Wed Jan 17 15:03:58 2024

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
from tqdm import tqdm
import copy

#%% Import Custom Modules
amvpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/" # amv module
scmpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/"

sys.path.append(amvpath)
sys.path.append(scmpath)

from amv import proc,viz
import scm
import amv.loaders as dl
import yo_box as ybx


#%% General Variables/User Edits

# Path to Input Data
input_path = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/'

figpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20240119/"
proc.makedir(figpath)

#%% Load the Different Forcings

# Forcing File Names
fnames = [
    
    "SLAB_PIC_NAO_EAP_NHFLX_Forcing_DJFM-MON.npy",
    "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy",
    "flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0.npy",
    
    ]
fnames_long = ["NAO-EAP DJFM (pt)","FLXSTD EOF (pt,SLAB)","FLXSTD EOF (pt,FULL)"]

# Load and print dimensions
fload   = [np.load(input_path+fn) for fn in fnames]
[print(fl.shape)for fl in fload]

# Square and sum the EOFs
floadsq = [np.sqrt(np.nansum((fl**2),2)) for fl in fload] 

# Load Lat/Lon for plotting
lon,lat = scm.load_latlon()
lonf,latf=-30,50
klon,klat = proc.find_latlon(lonf,latf,lon,lat)

# Compare EOF 1 Month 1 of both forcings
imode = 1
sq    = True
imon  = 0
vmax  = 80
fig,axs = viz.geosubplots(1,2,figsize=(12,8))
for a in range(2):
    ax = axs[a]
    ax = viz.add_coast_grid(ax,bbox=[-80,0,0,65],fill_color="k")
    if sq:
        plotvar = floadsq[a][:,:,imon].T
        eofstr  = "SUM"
    else:
        fload[a][:,:,imode,imon].T
        eofstr  = "%s" % (imode+1)
    
    pcm=ax.pcolormesh(lon,lat,plotvar,vmin=-vmax,vmax=vmax,cmap="RdBu_r")
    ax.set_title(fnames_long[a],fontsize=18)
plt.suptitle("EOF %s, Month %02i" % (eofstr,imon+1),fontsize=18)
fig.colorbar(pcm,ax=axs.flatten(),orientation='horizontal',fraction=0.025,pad=0.01)

#%% Add in Fprime Std (takes awhile to load)

# Load Fprime 
fname    = "Fprime_PIC_SLAB_rolln0.nc"
dsf      = xr.open_dataset(input_path+"../"+fname).Fprime.load()
dsf      = proc.format_ds(dsf)

# Flip Longitude
dsf = proc.format_ds(dsf)

# Compute Monthly variance
dsmonvar = dsf.groupby('time.month').var('time')
fprimestd = dsmonvar.values.transpose(2,1,0)

#%% Examine what is going on at the point

# Restrict to Point
fpt   = [fl[klon,klat,:,:] for fl in fload]
fptsq = [np.sqrt(np.nansum(f**2,0)) for f in fpt] 

fprimept = np.sqrt(fprimestd[klon,klat,:])

# Load plotting variables
mons3 = proc.get_monstr(nletters=3)
fig,ax = viz.init_monplot(1,1)

for ff in range(len(fptsq)):
    plotvar = fptsq[ff]
    ax.plot(mons3,plotvar,label=fnames_long[ff],marker="d")

ax.plot(mons3,fprimept,color='gray',ls='dashed',marker="x",label="std(Fprime) (SLAB)")
ax.legend()


#plt.pcolormesh(fload[1][:,:,0,0].T),plt.colorbar() # Debug Plot

#%% Ok, now try loading the damping
"""

"New" descriptes the settings in sm_rewrite_loop where:
    mode    = 5
    ensostr = ""
    lag     = lag1
    
"""
# Assuming new is default lagstr1, ensolag is removed
# Method 5 (not sure what this is again?)

dampfn = [
    
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy",
    "SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode5_lag1.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode4.npy",
    "FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode5_lag1.npy",
    
    ]

dampnames = [
    
    "SLAB PIC (old)",
    "SLAB PIC (new)",
    "FULL PIC (old)",
    "FULL PIC (new)",
    
    ]

# Load and print shape
dampload = [np.load(input_path+fn) for fn in dampfn]
[print(fl.shape) for fl in dampload]

# Restrict to Point
dpt   = [fl[klon,klat,:] for fl in dampload]


#%% Plot Damping to Compare


fig,ax = viz.init_monplot(1,1)

for ff in range(len(dpt)):
    plotvar = dpt[ff]
    ax.plot(mons3,plotvar,label=dampnames[ff],marker="d")

#ax.plot(mons3,np.sqrt(fprimestd[klon,klat,:]),color='gray',ls='dashed',marker="x",label="std(Fprime) (SLAB)")
ax.legend()

#%% Run a silly simulation with this set

# Set up forcing and other parameters
dt       = 30*3600*24
nyrs     = 10000
eta      = np.random.normal(0,1,nyrs*12)

hblt     = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

lags     = np.arange(37)

#%% Set Forcings and dampings
forcings = [fptsq[1],fptsq[2],fprimept]
dampings = [dpt[0],dpt[1],dpt[1]]
expnames = ["Old","New","Fstd"]
nexps    = len(forcings)

#%% Load output from synth_stochmod before for comparison

outdir   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/debug_stochmod/"
savename = "%ssynth_stochmod_combine_output.npz" % outdir

ld  = np.load(savename,allow_pickle=True)
mvs = ld['monvars']
lbs = ld['labels']

acs_old = ld['acs'] 



#%% Experiment with different rolls
forcingroll = 0
dampingroll = 0
rollstr     = "damproll%i_forceroll%i" % (dampingroll,forcingroll)

#%% Integrate the stochastic Model

debug = False

outputs = []
for ex in range(nexps):
    
    
    # Get Forcing/Damping and Roll
    f_in = np.roll(forcings[ex].copy(),forcingroll)
    d_in = np.roll(dampings[ex].copy(),dampingroll)
    
    
    # Set up Stochastic Model Input...
    smconfig = {}
    smconfig['eta']     = eta.copy()               # White Noise Forcing
    smconfig['forcing'] = f_in.copy()[None,None,:] # [x x 12] # Forcing Amplitude (W/m2)
    smconfig['damping'] = d_in.copy()[None,None,:] # Damping Amplitude (degC/W/m2)
    smconfig['h']       = np.ones((1,1,12)) * hblt # MLD (meters)
    
    if debug: # Debug Plot of the Inputs
        fig,ax = viz.init_monplot(1,1,)
        ax.plot(mons3,smconfig['forcing'].squeeze(),label='forcing')
        ax2 = ax.twinx()
        ax2.plot(mons3,smconfig['damping'].squeeze(),label='damping',color='red')
        ax.legend()
        ax.set_title("Experiment %i" % (ex+1))
    
    # Run Stochastic Model (No entrain) ---------------------------------------
    # Convert units (W/m2 to degC/S)
    smconfig['damping']=scm.convert_Wm2(smconfig['damping'],smconfig['h'],dt)[None,None,:]
    smconfig['forcing']=scm.convert_Wm2(smconfig['forcing'],smconfig['h'],dt)
    
    # Make Forcing
    smconfig['Fprime']= np.tile(smconfig['forcing'],nyrs) * smconfig['eta'][None,None,:]
    
    # Do Integration
    output = scm.integrate_noentrain(smconfig['damping'],smconfig['Fprime'],debug=True)
    # -------------------------------------------------------------------------
    
    outputs.append(output)


# % Calculate some diagnostics
ssts      = [o[0].squeeze() for o in outputs]
tsmetrics = scm.compute_sm_metrics(ssts)


monvars = tsmetrics['monvars']
acs_all = tsmetrics['acfs']

#%% Plot forcing and damping

ytks  = np.arange(10,80,10)
ytks2 = np.arange(0,36,5)

fig,axs = viz.init_monplot(3,1,figsize=(6,5.5))

for ff in range(nexps):
    
    ax = axs[ff]
    
    f_in = np.roll(forcings[ff].copy(),forcingroll)
    d_in = np.roll(dampings[ff].copy(),dampingroll)
    
    ax.plot(mons3,f_in,color="cornflowerblue",lw=2.5,label="forcing",marker="o")
    ax.tick_params(axis='y', colors='cornflowerblue')
    
    ax2 = ax.twinx()
    ax2.plot(mons3,d_in,color="red",lw=2.5,label="damping",marker='d',ls='dashed')
    ax2.tick_params(axis='y', colors='red')
    
    
    ax.set_ylim([10,70])
    ax.set_yticks(ytks)
    
    ax2.set_ylim([0,35])
    ax2.set_yticks(ytks2)
    
    viz.label_sp(expnames[ff],x=0.45,ax=ax,labelstyle="%s",usenumber=True)
plt.suptitle("Damping Shift (%i) | Forcing Shift (%i)" % (dampingroll, forcingroll))
savename = "%sDebug_Forcing_v_Damping_%s.png" % (figpath,rollstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot monvar

fig,ax = viz.init_monplot(1,1)

for ff in range(nexps):
    plotvar = monvars[ff]
    ax.plot(mons3,plotvar,label=expnames[ff],marker="o")

ax.plot(mons3,mvs[-1],label="SLAB",color="gray",ls="dashed")
ax.legend()
ax.set_ylim([0.5,1.5])

plt.suptitle("Damping Shift (%i) | Forcing Shift (%i)" % (dampingroll, forcingroll))
savename = "%sDebug_Monthly_Variance_%s.png" % (figpath,rollstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

#%% Plot autocorrelation

kmonth = 2
xtksl  = np.arange(0,37,3)
title  = "ACF (Lag 0 = %s) | Damping Shift (%i) | Forcing Shift (%i)" % (mons3[kmonth],dampingroll, forcingroll)
fig,ax = viz.init_acplot(kmonth,xtksl,lags,title=title)

for ff in range(nexps):
    plotvar = acs_all[kmonth][ff]
    ax.plot(lags,plotvar,label=expnames[ff],marker="o")
    
    
ax.plot(lags,acs_old[8],label=lbs[8],ls='dashed')
ax.legend()

savename = "%sDebug_ACF_basemonth%i_%s.png" % (figpath,kmonth+1,rollstr)
plt.savefig(savename,dpi=150,bbox_inches='tight')

# -----------------------------------------------------------------------------
#%% For a given experiment, test different rolling of damping/forcing
# -----------------------------------------------------------------------------

# Selec input forcing/damping
#ex           = 0
forcing_in   = fprimept#forcings[ex]
damping_in   = dpt[1] # dampings[ex]

# Select the forcing/damping roll amounts (amt to roll forward)
frolls = [0,0,1,1]#np.arange(13)##np.zeros(13)#[0,0,1,1] # forcing
drolls = [0,1,0,1]#np.zeros(13)#np.arange(13)#[0,1,0,1] # damping

# Loop for combinations
ncombo = len(frolls)
routputs  = [] # [Combo][term (SST, Forcing, Damping)]
rforcings = []
rdampings = []
rollnames = []
for n in range(ncombo):
    
    forcingroll = frolls[n]
    dampingroll = drolls[n]
    rollstr     = "damproll%02i_forceroll%02i" % (dampingroll,forcingroll)
    
    # Get Forcing/Damping and Roll
    f_in = np.roll(forcing_in.copy(),forcingroll)
    d_in = np.roll(damping_in.copy(),dampingroll)
    
    # Copied from above =======================================================
    # Set up Stochastic Model Input...
    smconfig = {}
    smconfig['eta']     = eta.copy()               # White Noise Forcing
    smconfig['forcing'] = f_in.copy()[None,None,:] # [x x 12] # Forcing Amplitude (W/m2)
    smconfig['damping'] = d_in.copy()[None,None,:] # Damping Amplitude (degC/W/m2)
    smconfig['h']       = np.ones((1,1,12)) * hblt # MLD (meters)
    
    if debug: # Debug Plot of the Inputs
        fig,ax = viz.init_monplot(1,1,)
        ax.plot(mons3,smconfig['forcing'].squeeze(),label='forcing')
        ax2 = ax.twinx()
        ax2.plot(mons3,smconfig['damping'].squeeze(),label='damping',color='red')
        ax.legend()
        ax.set_title("Experiment %i" % (ex+1))
    
    # Run Stochastic Model (No entrain) ---------------------------------------
    # Convert units (W/m2 to degC/S)
    smconfig['damping']=scm.convert_Wm2(smconfig['damping'],smconfig['h'],dt)[None,None,:]
    smconfig['forcing']=scm.convert_Wm2(smconfig['forcing'],smconfig['h'],dt)
    
    # Make Forcing
    smconfig['Fprime']= np.tile(smconfig['forcing'],nyrs) * smconfig['eta'][None,None,:]
    
    # Do Integration
    output = scm.integrate_noentrain(smconfig['damping'],smconfig['Fprime'],debug=True)
    
    #  ========================================================================
    
    # Append Output
    routputs.append(output)
    rollnames.append(rollstr)
    
    rforcings.append(f_in)
    rdampings.append(d_in)

# Calculate the Output
ssts      = [o[0].squeeze() for o in routputs]
tsmetrics = scm.compute_sm_metrics(ssts)


#%%

nc     = 2
kmonth = 1

for nc in range(ncombo):
    
    ytks  = np.arange(10,80,10)
    ytks2 = np.arange(0,36,5)
    
    fig,axs = viz.init_monplot(3,1,figsize=(6,8),skipaxis=[2,])
    
    # Plot Monthly Variance
    ax = axs[0]
    plotvar = tsmetrics['monvars'][nc]
    ax.plot(mons3,plotvar,label=rollnames[nc],marker="o",color='purple')
    
    # Plot Base
    plotvar = tsmetrics['monvars'][0]
    ax.plot(mons3,plotvar,label="base",marker="o",color='purple',alpha=0.2)
    
    ax.plot(mons3,mvs[-1],label="SLAB",color="gray",ls="dashed")
    ax.legend()
    ax.set_ylim([0,2])
    ax.set_ylabel("SST Variance (degC2)")
    
    
    # --- <0> . <0> --- <0> . <0> ---
    # Plot Forcing and Damping Shift
    ax = axs[1]
    
    # Plot Forcing/Damping (Base)
    ax.plot(mons3,rforcings[0],color="cornflowerblue",lw=2.5,label="forcing",marker="o",alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(mons3,rdampings[0],color="red",lw=2.5,label="damping",marker='d',ls='dashed',alpha=0.15)
    
    # Plot Forcing (Rolled)
    ax.plot(mons3,rforcings[nc],color="cornflowerblue",lw=2.5,label="forcing",marker="o")
    ax.tick_params(axis='y', colors='cornflowerblue')
    
    # Plot Damping (Rolled)
    ax2.plot(mons3,rdampings[nc],color="red",lw=2.5,label="damping",marker='d',ls='dashed')
    ax2.tick_params(axis='y', colors='red')
    
    # Set Limits and Ticks
    ax.set_ylim([10,70])
    ax.set_yticks(ytks)
    ax.set_ylabel("Forcing (W/m2)")
    
    ax2.set_ylim([0,35])
    ax2.set_yticks(ytks2)
    ax2.set_ylabel("Damping (degC/W/m2)")
    # Subplot Title
    # viz.label_sp(expnames[ff],x=0.45,ax=ax,labelstyle="%s",usenumber=True)
    
    
    # --- <0> . <0> --- <0> . <0> ---
    ax     = axs[2]
    title3 = ""

    ax,_ = viz.init_acplot(kmonth,xtksl,lags,ax=ax,title=title3)
    
    ax.set_xlim([0,lags[-1]])
    ax.set_xticks(xtksl)
    
    # Plot Base
    ax.plot(lags,tsmetrics['acfs'][0][kmonth],label="base",marker="o",color='purple',alpha=0.2)
    
    # Plot Rolled Version
    ax.plot(lags,tsmetrics['acfs'][nc][kmonth],label=rollnames[nc],marker="o",color='purple')
    
    # Plot SLAB
    ax.plot(lags,acs_old[8],label=lbs[8],ls='dashed',color='gray')
    ax.set_xticks(xtksl)
    
    plt.suptitle("Damping Shift (%i) | Forcing Shift (%i)" % (drolls[nc], frolls[nc]),y=1.01)
    
    
    savename = "%s/Debug_Monvar_%s.png" % (figpath,rollnames[nc])
    plt.savefig(savename,dpi=150,bbox_inches='tight')
