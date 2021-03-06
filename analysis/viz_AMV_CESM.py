#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and Visualize AMV
from CESM PiC Runs

Created on Mon May 24 22:55:19 2021

@author: gliu
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy import linalg,stats
from scipy.signal import butter,filtfilt
from scipy.io import loadmat
import sys
import cartopy.crs as ccrs



import cmocean

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm
import yo_box as ybx

#%% User Edits

projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20210628/'
proc.makedir(outpath)

datpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"

bbox = [-80,0 ,0,60]

runmean=True

#% ----------------------
#%% Load PiC Data
#% ----------------------
st = time.time()
# Load full sst data from model
ld  = np.load(datpath+"FULL_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
sstfull = ld['TS']
ld2 = np.load(datpath+"SLAB_PIC_ENSOREM_TS_lag1_pcs2_monwin3.npz" ,allow_pickle=True)
sstslab = ld2['TS']

# Load lat/lon
lat    = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LAT'].squeeze()
lon360 = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()

print("Loaded PiC Data in %.2fs"%(time.time()-st))

# --------------
#%% Preprocessing
# --------------

def preproc_CESMPIC(sst):
    
    # Apply Land/Ice Mask
    mask = np.load(datpath+"landicemask_enssum.npy")
    sst = sst * mask[None,:,:]
    
    # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
    sst = sst.transpose(2,1,0)
    
    # Flip longitude
    st = time.time()
    lon180,sst = proc.lon360to180(lon360,sst)
    print("Flipped Longitude in %.2fs"%(time.time()-st))
    
    # Remove monthly anomalies
    st = time.time()
    nlon,nlat,ntime = sst.shape
    sst = sst.reshape(nlon,nlat,int(ntime/12),12)
    ssta = sst - sst.mean(2)[:,:,None,:]
    print("Deseasoned in %.2fs"%(time.time()-st))
    print("Mean was %e" % (np.nanmax(ssta.mean(2))))
    
    ssta = ssta.reshape(nlon,nlat,int(ntime/12)*12)
    
    return ssta,lon180


# Preprocess (Apply Land/ice Mask, Adjust Dimensions, Remove Anomalies)
sstas = []
for sst in [sstfull,sstslab]:
    ssta,lon180 = preproc_CESMPIC(sst)
    sstas.append(ssta)

sstfulla,sstslaba = sstas
nlon,nlat,ntimef = sstfulla.shape
_,_,ntimes = sstslaba.shape



# # # Apply Land/Ice Mask
# # mask = np.load(datpath+"landicemask_enssum.npy")
# # sstfull = sstfull * mask[None,:,:]
# # sstslab = sstslab * mask[None,:,:]

# # # Adjust dimensions [time x lat x lon] --> [lon x lat x time]
# # sstfull = sstfull.transpose(2,1,0)
# # sstslab = sstslab.transpose(2,1,0)


# # Flip longitude
# st = time.time()
# lon180,sstfull = proc.lon360to180(lon360,sstfull)
# _,sstslab = proc.lon360to180(lon360,sstslab)
# print("Flipped Longitude in %.2fs"%(time.time()-st))

# # Remove monthly anomalies
# st = time.time()
# nlon,nlat,ntimef = sstfull.shape
# _,_,ntimes = sstslab.shape
# sstslab = sstslab.reshape(nlon,nlat,int(ntimes/12),12)
# sstfull = sstfull.reshape(nlon,nlat,int(ntimef/12),12)
# sstslaba = sstslab - sstslab.mean(2)[:,:,None,:]
# sstfulla = sstfull - sstfull.mean(2)[:,:,None,:]
# print("Deseasoned in %.2fs"%(time.time()-st))

# #
# #proc.sel_region(sstfull,lon360,


# -----------------
#%% Calculate AMV
# -----------------

# Calculate AMV Index
idxs = []
pats = []
for sst in sstas:
    
    amvidx,amvpattern=proc.calc_AMVquick(sst,lon180,lat,bbox,order=5,cutofftime=10,anndata=False,runmean=runmean)
    
    idxs.append(amvidx)
    pats.append(amvpattern)

# -----------------
#%% Plot AMV Patterns
# -----------------

# Plot AMV Patterns
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

fig,ax,cb = plot_AMV_generals(lat,lon180,pats[0])
ax.set_title("AMV Pattern (CESM1-FULL; 400 to 2200) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"CESM1-FULL_AMV_Spatial_Pattern.png",bbox_inches='tight')


fig,ax,cb = plot_AMV_generals(lat,lon180,pats[-1])
ax.set_title("AMV Pattern (CESM1-SLAB; 101 to 1001 ) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"CESM1-SLAB_AMV_Spatial_Pattern.png",bbox_inches='tight')


#
# %% Calculate NASSTI
#

awgt = 1
nassti = []
for sst in sstas:
    
    aa_sst = proc.area_avg(sst,bbox,lon180,lat,awgt)
    nassti.append(aa_sst)


#
# Save NASSTI
#
fn = datpath + "CESM-PIC_NASSTI.npz"
np.savez(fn,**{
         'nassti_slab': nassti[-1],
         'nassti_full': nassti[0]}
        )

# ---------------------------
#%% Do some spectral analysis
# ---------------------------

# amvid = []
# for k in amvidx.keys():
#     amvid.append(amvidx[k])
# amvid = np.array(amvid)
    

enumvar = idxs #nassti


# # -------------------------------------------
# # First calculate for CESM1 (full and slab)
# # -------------------------------------------
# Key Params
plotcesm = True
cnames  = ["CESM1 FULL","CESM1 SLAB"]
nsmooths = [10,10] # Set Smothing
#nsmooths = [250,125]

timemax = None#250*12

# Other Params
pct     = 0.10
opt     = 1
dt      = 3600*24*30
tunit   = "Years"
clvl    = [0.95]
axopt   = 3
clopt   = 1

dtplot = 3600*24*365

# Calculate spectra
freq1s,P1s,CLs = [],[],[]
for i,sstin in enumerate(enumvar):
    
    # Limit to maximum time
    if timemax is None:
        sstin=sstin
    else:
        sstin = sstin[:timemax]
    
    # Calculate and Plot
    sps = ybx.yo_spec(sstin,opt,nsmooths[i],pct,debug=False)
    P,freq,dof,r1=sps
    
    # Plot if option is set
    if plotcesm:
        
        pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dtplot,clvl=clvl,axopt=axopt,clopt=clopt)
        print(r1)
        fig,ax,h,hcl,htax,hleg = pps
        #ax,htax = viz.make_axtime(ax,htax)
        #ax = viz.add_yrlines(ax)
        #ax.set_title("%s Spectral Estimate \n nsmooth=%i, taper = %.2f" % (cnames[i],nsmooths[i],pct*100) +r"%")
        #ax.grid(True,which='both',ls='dotted')
        #ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
        #plt.tight_layout()
        #plt.savefig("%sNASSTI_SpectralEstimate_%s_nsmooth%i_taper%i.png"%(outpath,cnames[i],nsmooths[i],pct*100),dpi=200)
    CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
    
    P    = P*dt
    freq = freq/dt
    CC   = CC*dt
    P1s.append(P)
    freq1s.append(freq)
    CLs.append(CC)

# Read outvariables
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s
clfull,clslab = CLs

#%% Remake the Plot(Linear)

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


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
ax.plot(freqcesmslab*plotdt,CLs[1][:,0]/plotdt,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
ax.plot(freqcesmfull*plotdt,CLs[0][:,0]/plotdt,color='black',label="CESM1 FULL AR1",ls=':')
ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL)")
plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i.png"%(outpath,nsmooths[i],pct*100),dpi=200)

#%% Plot Linear, but over a particular range

xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
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


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
ax.plot(freqcesmslab*plotdt,CLs[1][:,0]/plotdt,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
ax.plot(freqcesmfull*plotdt,CLs[0][:,0]/plotdt,color='black',label="CESM1 FULL AR1",ls=':')
#ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL) \n nsmooth=%i"%(nsmooths[0]))

ax.set_xlim(xlms)
ax.set_xticks(xtks)
xtick=np.array(xtks)
htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
# Set xtick labels
htax.set_xticklabels(xtkl)

ax.set_ylim([0,1])

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i_decadal_timemax%i.png"%(outpath,nsmooths[i],pct*100,timemax),dpi=200)



#%% Remake the plot (variance preserving)




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


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))

i = 1
ax.semilogx(freqcesmslab*plotdt,Pcesmslab*freqcesmslab,color='gray',label="CESM1 SLAB" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[-1])))
ax.semilogx(freqcesmslab*plotdt,CLs[1][:,1]*freqcesmslab,color='gray',label="CESM1 SLAB AR1 95% Significance",ls='dashed')
ax.semilogx(freqcesmslab*plotdt,CLs[1][:,0]*freqcesmslab,color='gray',label="CESM1 SLAB AR1",ls=':')
ax.semilogx(freqcesmfull*plotdt,Pcesmfull*freqcesmfull,color='black',label="CESM1 FULL" + "$\; (\sigma=%.2f ^{\circ}C$)"%(np.std(nassti[0])))
ax.semilogx(freqcesmfull*plotdt,CLs[0][:,1]*freqcesmfull,color='black',label="CESM1 FULL AR1 95% Significance",ls='dashed')
ax.semilogx(freqcesmfull*plotdt,CLs[0][:,0]*freqcesmfull,color='black',label="CESM1 FULL AR1",ls=':')

# Set x limits
#xlm = [1/(plotdt*),1/(plotdt*1)]
xlm = [5e-4,10]
ax.set_xlim(xlm)
ylm = [-.01,.4]

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqcesmfull,"Years",dtplot,mode='log-lin')



# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtk]
htax.set_xticklabels(xtkl)

#ax,htax = lin_quickformat(ax,plotdt,freqcesmfull)
ax.set_xlabel("")
ax.set_title("CESM1 NASSTI (SLAB vs. FULL),nsmooth=%i"%nsmooths[0])
plt.tight_layout()
#plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i.png"%(outpath,nsmooths[i],pct*100),dpi=200)





#%% Load in some stochastic model results, and compare

datpath2  = projpath + '01_Data/model_output/'
fscale    = 1 # Number to scale NAO and other forcings by
nyrs      = 1000        # Number of years to integrate over
applyfac  = 2
mconfig   = "SLAB_PIC"
runid     = "303"
funiform  = 1.5
expid     = "%s_%iyr_funiform%i_run%s_fscale%03d_applyfac%i" %(mconfig,nyrs,funiform,runid,fscale,applyfac)


sst = np.load(datpath2+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lonr = np.load(datpath+"lon.npy")
latr = np.load(datpath+"lat.npy")

# Select NAtl Region for each model


#% Calculate AMV Index
amvtime = time.time()
amvidx = {}
amvpat = {}

for model in range(4):
    amvidx[model],amvpat[model] = proc.calc_AMVquick(sst[model],lonr,latr,bbox,order=5,cutofftime=10,anndata=False,runmean=runmean)
    
print("Calculated AMV variables for region in %.2f" % (time.time()-amvtime))


#%% Plot AMV Pattern
modeln = ["MLD Fixed","MLD Mean", "MLD Seasonal", "Entraining"]
ecolors = ['blue','r','magenta','orange']
enames = modeln
for model in [1,2,3]:
    
    fig,ax,cb = plot_AMV_generals(latr,lonr,amvpat[model])
    ax.set_title("AMV Pattern (Stochastic Model %s) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$" % (modeln[model]))
    plt.savefig(outpath+"Stochmod_AMV_Spatial_Pattern_model%i.png"%model,bbox_inches='tight')
    
    
#%% Calculate Spectra


nsmooths2 = np.ones(4)* 1
specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(amvidx,nsmooths2,pct)

# Plot Spectra
fig,ax = plt.subplots(1,1,figsize=(6,4))

for i in range(4):
    ax.plot(freqs[i]*plotdt,specs[i]/plotdt,color=ecolors[i],label=enames[i])
    
    ax.plot(freqs[i]*plotdt,CCs[i][:,1]/plotdt,color=ecolors[i],alpha=0.5,ls='dashed')
    ax.plot(freqs[i]*plotdt,CCs[i][:,0]/plotdt,color=ecolors[i],alpha=0.5,ls='dotted')

    

# Set x limits
xtick = np.arange(0,1.7,.2)
ax.set_xticks(xtick)

# Set Labels
ax.set_ylabel("Frequency x Power ($\degree C^{2}$)",fontsize=12)
ax.set_xlabel("Frequency (cycles/year)",fontsize=12)
htax = viz.twin_freqaxis(ax,freqs[1],"Years",plotdt,mode='lin-lin',xtick=xtick)

ax = viz.add_yrlines(ax,dt=plotdt)

#ylm = [-.01,.4]
# Set xtick labels
xtkl = ["%.1f" % (1/x) for x in xtick]
htax.set_xticklabels(xtkl)
ax.legend()
ax.set_title("AMV Idx Spectral Estimates (Stochastic Model)")
plt.tight_layout()
plt.savefig(outpath+"AMVIDX_Stochastic_Model.png",dpi=200)



#%% comparitive lin-lin plot


xlms = [0,0.2]
xtks = [0,0.02,0.04,0.1,0.2]
xtkl = 1/np.array(xtks)
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


plotdt = 3600*24*365
fig,ax = plt.subplots(1,1,figsize=(6,4))


# Plot CESM
i = 1

ax.plot(freqcesmslab*plotdt,Pcesmslab/plotdt,color='gray',label="CESM1 SLAB")
ax.plot(freqcesmslab*plotdt,CLs[1][:,1]/plotdt,color='gray',label="",ls='dashed')

ax.plot(freqcesmfull*plotdt,Pcesmfull/plotdt,color='black',label="CESM1 FULL")
ax.plot(freqcesmfull*plotdt,CLs[0][:,1]/plotdt,color='black',label="",ls='dashed')






ax.set_xlabel("")
#ax.set_title("CESM1 NASSTI (SLAB vs. FULL) \n nsmooth=%i"%(nsmooths[0]))

ax.set_xlim(xlms)
ax.set_xticks(xtks)
xtick=np.array(xtks)
htax = viz.twin_freqaxis(ax,freq,"Years",dt,mode='lin-lin',xtick=xtick)
# Set xtick labels
htax.set_xticklabels(xtkl)

ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("%sNASSTI_SpectralEstimate_nsmooth%i_taper%i_decadal_timemax%i.png"%(outpath,nsmooths[i],pct*100,timemax),dpi=200)







