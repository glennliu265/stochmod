#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Create spectral analysis plots for default runs

Created on Sun Mar 21 21:40:25 2021
Copied blocks from the stochmod_synth script

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

#%% Settings

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath = projpath + '02_Figures/20210424/'

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
#fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")
fullauto = np.load(datpath+"CESM_clim/TS_FULL_Autocorrelation.npy")


mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
labelsnew = ["h=50m","Constant $h$","Vary $h$","Entraining"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','red','magenta','orange')
els = ["dashdot","solid","dotted","dashed"]
#hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

# Set up Configuration
config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn001"   # White Noise ID
config['fname']       = "FLXSTD"   #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at 
config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = projpath + '02_Figures/20210223/'
config['smooth_forcing'] = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)
#%% Functions

def interp_quad(ts):
    
    # Interpolate qdp as well
    tsr = np.roll(ts,1)
    tsquad = (ts+tsr)/2
    
    fig,ax = plt.subplots(1,1)
    ax.set_title("Interpolation")
    ax.plot(np.arange(1.5,13.5,1),ts,label="ori",color='k',marker='d')
    ax.plot(np.arange(1,13,1),tsquad,label="shift",marker="o",color='red')
    ax.set_xticks(np.arange(1,13,1))
    ax.grid(True,ls="dotted")
    
    return tsquad
    #ax.set_xticklabels()
    
    
    
def adjust_axis(ax,htax,dt,multiple):
    
    # Divisions of time
    # dt  = 3600*24*30
    # fs  = dt*12
    # xtk      = np.array([1/fs/100,1/fs/50, 1/fs/25, 1/fs/10 , 1/fs/5, 1/fs])
    # xtkm    = ["%i" % np.round(i) for i in 1/xtk/dt]
    # xtklabel = ['%.1e \n (century)'%xtk[0],'%.1e \n (50yr)'%xtk[1],'%.1e \n (25yr)'%xtk[2],'%.1e \n (decade)'%xtk[3],'%.1e \n (5year)'%xtk[4],'%.2e \n (year)'%xtk[5]]
    
    fs = dt*multiple
    xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
    xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
    xtkl     = ["%.1e" % s for s in xtk]
    for i,a in enumerate([ax,htax]):
        
        a.set_xticks(xtk)
        if i == 0:
            
            a.set_xticklabels(xtkl)
        else:
            a.set_xticklabels(xtkm)
    return ax,htax

def make_axtime(ax,htax,denom='year'):
    
    # Units in Seconds
    dtday = 3600*24
    dtyr  = dtday*365
    
    fnamefull = ("Millennium","Century","Decade","Year","Month")
    if denom == 'month':
        
        # Set frequency (by 10^n months, in seconds)
        fs = [1/(dtyr*1000),1/(dtyr*100),1/(dtyr*10),1/(dtyr),1/(dtday*30)]
        xtk      = np.array(fs)#/dtin
        
        # Set frequency tick labels
        fsl = ["%.1e" % s for s in xtk]
        
        # Set period tick labels
        per = [ "%.2e \n (%s) " % (int(1/fs[i]/(dtday*30)),fnamefull[i]) for i in range(len(fnamefull))]
        
        # Set axis names
        axl_bot = "Frequency (cycles/sec)" # Axis Label
        axl_top = "Period (Months)"
        
        
    elif denom == 'year':
        
        # Set frequency (by 10^n years, in seconds)
        denoms = [1000,100,10,1,.1]
        fs = [1/(dtyr*1000),1/(dtyr*100),1/(dtyr*10),1/(dtyr),1/(dtyr*.1)]
        xtk      = np.array(fs)#/dtin
        
        # Set tick labels for frequency axis
        fsl = ["%.3f" % (fs[i]*dtyr) for i in range(len(fs))]
        
        # Set period tick labels
        per = [ "%.2e \n (%s) " % (denoms[i],fnamefull[i]) for i in range(len(fnamefull))]
        
        # Set axis labels
        axl_bot = "Frequency (cycles/year)" # Axis Label
        axl_top = "Period (Years)"

    
    
    for i,a in enumerate([ax,htax]):
        a.set_xticks(xtk)
        if i == 0:
            a.set_xticklabels(fsl)
            a.set_xlabel("")
            a.set_xlabel(axl_bot)
        else:
            a.set_xticklabels(per)
            a.set_xlabel("")
            a.set_xlabel(axl_top)
    return ax,htax
    
def set_monthlyspec(ax,htax):

    # Orders of 10
    dt = 3600*24*30
    fs = dt*3
    xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
    xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
    xtkl     = ["%.1e" % s for s in xtk]
    for i,a in enumerate([ax,htax]):
        a.set_xticks(xtk)
        if i == 0:
            a.set_xticklabels(xtkl)
        else:
            a.set_xticklabels(xtkm)
    return ax,htax
#% ------------
#%% Clean Run
#% ------------

#% Load some data into the local workspace for plotting
query   = config['query']
mconfig = config['mconfig']
lags    = config['lags']
ftype   = config['ftype']
locstring      = "lon%i_lat%i" % (query[0],query[1])
locstringtitle = "Lon: %.1f Lat: %.1f" % (query[0],query[1])

# Run Model
#config['Fpt'] = np.roll(Fpt,1)
ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
[o,a],damppt,mldpt,kprev,Fpt       =params

# Read in CESM autocorrelation for all points'
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = scm.load_data(mconfig,ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

# Plot some differences
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=params[2],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.plot(lags,cesmautofull,color='k',label='CESM Full',ls='dashdot')

for i in range(1,4):
    ax.plot(lags,ac[i],label=labels[i],color=expcolors[i])

ax.legend()
ax3.set_ylabel("Mixed Layer Depth (m)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"Default_Autocorrelation.png",dpi=200)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

sstall = sst
# --------------------
#%% Autocorrelation Plot with Confidence Intervals
# --------------------
conf  =0.95
tails = 2

def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags

nlags   = len(lags)
cfstoch = np.zeros([4,nlags,2])
for m in range(4):
    inac = ac[m]
    n = int(len(sst[m])/12)
    cfs = calc_conflag(inac,conf,tails,n)
    cfstoch[m,:,:] = cfs
cfslab = calc_conflag(cesmauto2,conf,tails,898)
cffull = calc_conflag(cesmautofull,conf,tails,1798)

fig,ax     = plt.subplots(1,1,figsize=(6,4))
title = "SST Autocorrelation: Adding Varying $h$ and Entrainment"
#title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM1 SLAB",color='k',marker="o",markersize=3)
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM1 Full',ls='dashdot',marker="o",markersize=3)
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

for i in range(1,4):
    ax.plot(lags,ac[i],label=labelsnew[i],color=expcolors[i],ls=els[i],marker="o",markersize=3)
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=expcolors[i],alpha=0.25)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=10,ncol=3)
plt.tight_layout()
plt.savefig(outpath+"Autocorrelation_MLDComplexity_%s.png"%locstring,dpi=200)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

#
# %% Plot Two Variables Together (Seasonal Cycle)
#

#fig,ax = plt.subplots(1,1,figsize=(6,2))
fig,ax = plt.subplots(1,1,figsize=(4,3))

ax.plot(mons3,mldpt,color='mediumblue',lw=0.75,marker="o",markersize=4)
ax.set_ylabel("Mixed-Layer Depth ($m$)")
ax.yaxis.label.set_color('mediumblue')
ax.set_xlim([0,11])

ax.tick_params(axis='x', labelrotation=45)
ax2 = ax.twinx()    
ax2.plot(mons3,Fpt,color='orangered',ls='solid',lw=0.75,marker="d",markersize=4,label="$1\sigma \; Forcing \; (Wm^{-2}$)")
ax2.yaxis.label.set_color('k')
ax2.plot(mons3,damppt,color='limegreen',ls='solid',label="$\lambda_a \; (Wm^{-2} \, ^{\circ} C^{-1})$",
         marker="x",markersize=5,lw=0.75)
ax2.legend(fontsize=8)
ax2.set_ylabel("$1\sigma \; Forcing, \; \lambda_{a}$")
ax2.set_xlim([0,11])
ax.grid(True,ls='dotted')

ax.set_title("Seasonal Cycle at %s"%locstringtitle)
plt.tight_layout()
plt.savefig(outpath+"Scycle_MLD_Forcing_%s_Narrow.png"%locstring,dpi=150)


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
#% --------------------------------
#%% Prepare to Do spectral analysis
#% --------------------------------

# Parameters
pct     = 0.10
nsmooth = 1
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1
specnames = "nsmooth%i_taper%i" % (nsmooth,pct*100)


# # -------------------------------------------
# # First calculate for CESM1 (full and slab)
# # -------------------------------------------
# Key Params
plotcesm = True
cnames  = ["CESM1 FULL","CESM1 SLAB"]
nsmooths = [500,250] # Set Smothing

# Other Params
pct     = 0.10
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1

# Retrieve point
lonf,latf = config['query']
if lonf < 0:
    lonf += 360
klon360,klat = proc.find_latlon(lonf,latf,lon360,lat)
fullpt = sstfull[:,klat,klon360]
slabpt = sstslab[:,klat,klon360]

# Calculate spectra
freq1s,P1s,CLs = [],[],[]
for i,sstin in enumerate([fullpt,slabpt]):
    
    # Calculate and Plot
    sps = ybx.yo_spec(sstin,opt,nsmooths[i],pct,debug=False)
    P,freq,dof,r1=sps
    
    # Plot if option is set
    if plotcesm:
        pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
        fig,ax,h,hcl,htax,hleg = pps
        ax,htax = viz.make_axtime(ax,htax)
        ax = viz.add_yrlines(ax)
        ax.set_title("%s Spectral Estimate \n nsmooth=%i, taper = %.2f" % (cnames[i],nsmooths[i],pct*100) +r"%")
        plt.tight_layout()
        plt.savefig("%sSpectralEstimate_%s_nsmooth%i_taper%i.png"%(outpath,cnames[i],nsmooths[i],pct*100),dpi=200)
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

## Note, section below is commented out as these are precomputed spectra
# # -------------------------------------------
# #%% Load results from cesm slab
# # -------------------------------------------
# ld = np.load("%s/model_output/CESM_PIC_Spectra_%s.npz"%(datpath,specnames),allow_pickle=True)
# Pcesmfulla    = ld['specfull']
# Pcesmslaba    = ld['specslab']
# freqcesmfulla = ld['freqfull']
# freqcesmslaba = ld['freqslab']

# # Retrieve Data For Point
# lonf = query[0]
# latf = query[1]
# if lonf < 0:
#     lonf += 360
# klon360,klat   = proc.find_latlon(lonf,latf,lon360,lat) # Global, 360 lon

# Pcesmfull    = Pcesmfulla[:,klat,klon360]
# Pcesmslab    = Pcesmslaba[:,klat,klon360]
# freqcesmfull = freqcesmfulla[:,klat,klon360]
# freqcesmslab = freqcesmslaba[:,klat,klon360]


# # -------------
# # Location Plot
# # -------------
# fig, ax= plt.subplots(1,1)
# ax.pcolormesh(lon360,lat,Pcesmfulla[0,:,:])
# ax.scatter(lonf,latf,100,marker="x",color='r')

# -----------------------------------------------------------------
# %%Calculate and make individual plots for stochastic model output
# -----------------------------------------------------------------
sstall = sst

#nsmooth=nsmooth*10/2
specparams  = []
splotparams = []
specs = []
freqs = []
for i in range(4):
    sstin = sstall[i]
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    specs.append(P*dt)
    freqs.append(freq/dt)
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    fig,ax,h,hcl,htax,hleg = pps
    
    if i < 2:
        l1 =ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
        #l2 =ax.semilogx(freqcesmslab,clslab[:,0]*freqcesmslab,label="CESM-SLAB (AR1)",color='red',lw=0.75,alpha=0.4)
        #l3 =ax.semilogx(freqcesmslab,clslab[:,1]*freqcesmslab,label="CESM-SLAB (95%)",color='blue',lw=0.75,alpha=0.4)
    else:
        l1 =ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='gray',lw=0.75)
        #l2 =ax.semilogx(freqcesmfull,clfull[:,0]*freqcesmfull,label="CESM-FULL (AR1)",color='red',lw=0.75,alpha=0.4)
        #l3 =ax.semilogx(freqcesmfull,clfull[:,1]*freqcesmfull,label="CESM-FULL (95%)",color='blue',lw=0.75,alpha=0.4)

    if axopt != 1:
        #dt = 12*365*3600
        dtin = 3600*24*365
        #ax,htax=make_axtime(ax,htax,dt)
        ax,htax=make_axtime(ax,htax)
    
    #ax.semilogx(freqcesmfull,freqcesmfull*Pcesmfull,'gray',label="CESM-FULL")
    #vlv = [1/(100*dt*12),1/(10*12*dt),1/(12*dt)]
    vlv = [1/(100*365*24*3600),1/(10*365*24*3600),1/(365*24*3600)]
    vll = ["Century","Decade","Year"]
    for vv in vlv:
        ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    
    
    ax.grid(True,which='both',ls='dotted')
    ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
    ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
    ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
    plt.tight_layout()
    plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i_%s.png"%(outpath,labels[i],nsmooth,pct*100,axopt,locstring),dpi=200)

#%% Plot spectra for SST from PIC

# fullpic  = "FULL_PIC_SST_lon330_lat50.npy"
# slabpic  = "SLAB_PIC_SST_lon330_lat50.npy"
# cesmfull = np.load(datpath+fullpic)
# cesmslab = np.load(datpath+slabpic)



# sstin = cesmfull
# sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
# P,freq,dof,r1=sps
# pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
# splotparams.append(pps)
# fig,ax,h,hcl,htax,hleg = pps

# def set_monthlyspec(ax,htax):
    
#     # Divisions of time
#     # dt  = 3600*24*30
#     # fs  = dt*12
#     # xtk      = np.array([1/fs/100,1/fs/50, 1/fs/25, 1/fs/10 , 1/fs/5, 1/fs])
#     # xtkm    = ["%i" % np.round(i) for i in 1/xtk/dt]
#     # xtklabel = ['%.1e \n (century)'%xtk[0],'%.1e \n (50yr)'%xtk[1],'%.1e \n (25yr)'%xtk[2],'%.1e \n (decade)'%xtk[3],'%.1e \n (5year)'%xtk[4],'%.2e \n (year)'%xtk[5]]
    
#     # Orders of 10
#     dt = 3600*24*30
#     fs = dt*3
#     xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
#     xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
#     xtkl     = ["%.1e" % s for s in xtk]
#     for i,a in enumerate([ax,htax]):
        
#         a.set_xticks(xtk)
#         if i == 0:
            
#             a.set_xticklabels(xtkl)
#         else:
#             a.set_xticklabels(xtkm)
#     return ax,htax
# if axopt != 1:
#     ax,htax = set_monthlyspec(ax,htax)

# #xt
# vlv = [1/(100*12*dt),1/(12*10*dt),1/(12*dt)]
# vll = ["Century","Decade","Year"]
# for vv in vlv:
#     ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)


# ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
# ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
# ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
# plt.tight_layout()
# plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,'CESM_FULL',nsmooth,pct*100,axopt),dpi=200)

#%% Plot all experiments together

expcolors = ('blue','orange','magenta','red')

# Set up variance preserving plot
freq = freqs[0]
fig,ax = plt.subplots(1,1)
ax.set_ylabel("Frequency x Power",fontsize=13)

for i in np.arange(1,4):
    print(specs[i].sum())
    ax.semilogx(freqs[i],freqs[i]*specs[i],label=labels[i],color=expcolors[i],lw=0.75)
ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='k',lw=0.75)
ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)

xmin = 10**(np.floor(np.log10(np.min(freq))))

ax.set_xlim([xmin,0.5/dt])

ax.grid(True,ls='dotted')
freqtick = ax.get_xticks()
yy = ax.get_ylim()
# ax.set_xscale("log")
# ax.set_yscale("linear")
period =1/freq
htax   =ax.twiny()
htax.set_xscale("log")
htax.set_yscale("linear")
xtkl = ["%.1f"% s for s in np.fix(1/freqtick/dt)]
htax.set_xlim([xmin,0.5/dt])
htax.set_xlabel("Period (%s)"%tunit,fontsize=13)
#ax,htax = set_monthlyspec(ax,htax)

ax.legend()


ax,htax=make_axtime(ax,htax)
#ax,htax=adjust_axis(ax,htax,dt,1.2)

#ax.semilogx(freqcesmfull,freqcesmfull*Pcesmfull,'gray',label="CESM-FULL")
#vlv = [1/(100*dt*12),1/(10*12*dt),1/(12*dt)]
vlv = [1/(100*365*24*3600),1/(10*365*24*3600),1/(365*24*3600)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    
    
    ax.grid(True,which='both',ls='dotted')

#ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum at %s \n" % (locstringtitle) + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i_%s.png"%(outpath,'COMPARISON',nsmooth,pct*100,axopt,locstring),dpi=200)


# --------------------------------------------------
#%% Spectral Analysis, but using annual averaged data
# --------------------------------------------------
slabann = proc.ann_avg(cesmslab,0)
fullann = proc.ann_avg(cesmfull,0)
sstann = []
for i in range(4):
    sstann.append(proc.ann_avg(sst[i],0))
nyr   = sstann[0].shape[0]
simtime = np.arange(0,sst[0].shape[0])
years = np.arange(0,nyr) 

# Check Annual Averaging
i = 3
fig,ax = plt.subplots(1,1)
ax.plot(simtime,sst[i],color='g',lw=0.5)
ax.plot(simtime[::12],sstann[i],color='k')
ax.set_xlim([0,120])

# Spectral Analysis
# Parameters
pct     = 0.0
nsmooth = 200
opt     = 1
dt      = 3600*24*365
tunit   = "Years"
clvl    = [0.95]
axopt   = 3
clopt   = 1
# -------------------------------------------
# First calculate for CESM1 (full and slab)
# -------------------------------------------
freq1s,P1s, = [],[]
for sstin in [fullann,slabann]:
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    P,freq,dof,r1=sps
    
    P    = P*dt
    freq = freq/dt
    
    P1s.append(P)
    freq1s.append(freq)
Pannfull,Pannslab = P1s
freqannfull,freqannslab = freq1s
# -------------------------------------------
# Bextm calculate for the Individual experiments
# -------------------------------------------
specparams  = []
splotparams = []
specs = []
freqs = []
for i in range(4):
    sstin = sstann[i]
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    specs.append(P*dt)
    freqs.append(freq/dt)
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    fig,ax,h,hcl,htax,hleg = pps
    
    if i < 2:
        
        l1 =ax.semilogx(freqannslab,Pannslab*freqannslab,label="CESM-SLAB",color='gray',lw=0.75)
    else:
        l1 =ax.semilogx(freqannfull,Pannfull*freqannfull,label="CESM-FULL",color='gray',lw=0.75)
    

    if axopt != 1:
        ax,htax = adjust_axis(ax,htax,dt,1)
    
    #ax.semilogx(freqcesmfull,freqcesmfull*Pcesmfull,'gray',label="CESM-FULL")
    
    vlv = [1/(100*dt),1/(10*dt),1/(dt)]
    vll = ["Century","Decade","Year"]
    for vv in vlv:
        ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    
    
    ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
    ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
    ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
    plt.tight_layout()
    plt.savefig("%sPowerSpectra_Ann_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,labels[i],nsmooth,pct*100,axopt),dpi=200)

# -------------------------------------
# %% Spectral Analysis for Damping Var or ForcingVa
# -------------------------------------


sstexp   = dampvarsst
expname2 = "DampVary"
labels2  = ["Variable Damping","Constant Damping","Constant Damping and Forcing"]
m        = 1 # Model Number (see labels)

# Parameters
pct     = 0.0
nsmooth = 200
opt     = 1
dt      = 3600*24*30
tunit   = "Months"
clvl    = [0.95]
axopt   = 3
clopt   = 1

# -------------------------------------------
# First calculate for CESM1 (full and slab)
# -------------------------------------------
fullpic  = "FULL_PIC_SST_lon330_lat50.npy"
slabpic  = "SLAB_PIC_SST_lon330_lat50.npy"
cesmfull = np.load(datpath+fullpic)
cesmslab = np.load(datpath+slabpic)
freq1s,P1s, = [],[]
for sstin in [cesmfull,cesmslab]:
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    P,freq,dof,r1=sps
    
    P    = P*dt
    freq = freq/dt
    
    P1s.append(P)
    freq1s.append(freq)
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s


# -----------------------------------------------------------------
# Calculate and make individual plots for stochastic model output
# -----------------------------------------------------------------
specparams  = []
splotparams = []
specs = []
freqs = []
for i in range(len(labels2)):
    sstin = sstexp[i][m]
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    specparams.append(sps)
    
    P,freq,dof,r1=sps
    specs.append(P*dt)
    freqs.append(freq/dt)
    pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    splotparams.append(pps)
    fig,ax,h,hcl,htax,hleg = pps
    
    if m < 2:
        
        l1 =ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
    else:
        l1 =ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='gray',lw=0.75)
    
    def set_monthlyspec(ax,htax):

        # Orders of 10
        dt = 3600*24*30
        fs = dt*3
        xtk      = np.array([1/(fs*10**-p) for p in np.arange(-11+7,-6+7,1)])
        xtkm     = ["%.1f"% s for s in np.round(1/xtk/dt)]
        xtkl     = ["%.1e" % s for s in xtk]
        for i,a in enumerate([ax,htax]):
            
            a.set_xticks(xtk)
            if i == 0:
                
                a.set_xticklabels(xtkl)
            else:
                a.set_xticklabels(xtkm)
        return ax,htax
    if axopt != 1:
        ax,htax = set_monthlyspec(ax,htax)
    
    #ax.semilogx(freqcesmfull,freqcesmfull*Pcesmfull,'gray',label="CESM-FULL")
    
    vlv = [1/(100*12*dt),1/(12*10*dt),1/(12*dt)]
    vll = ["Century","Decade","Year"]
    for vv in vlv:
        ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
    
    
    ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
    ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
    ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
    plt.tight_layout()
    plt.savefig("%sPowerSpectra_%s_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,expname2,labels[i],nsmooth,pct*100,axopt),dpi=200)


# Make the plot ---
# Set up variance preserving plot
freq = freqs[0]
fig,ax = plt.subplots(1,1)
ax.set_ylabel("Frequency x Power",fontsize=13)
for i in range(len(labels2)):
    ax.semilogx(freqs[i],freqs[i]*specs[i],label=labels2[i],color=expcolors[i],lw=0.75)
ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)


    

xmin = 10**(np.floor(np.log10(np.min(freq))))
ax.set_xlim([xmin,0.5/dt])
ax.grid(True,ls='dotted')
freqtick = ax.get_xticks()
yy = ax.get_ylim()
# ax.set_xscale("log")
# ax.set_yscale("linear")
period =1/freq
htax   =ax.twiny()
htax.set_xscale("log")
htax.set_yscale("linear")
xtkl = ["%.1f"% s for s in np.fix(1/freqtick/dt)]
htax.set_xlim([xmin,0.5/dt])
htax.set_xlabel("Period (%s)"%tunit,fontsize=13)
ax,htax = adjust_axis(ax,htax,dt,1.2)
ax.legend()
vlv = [1/(3600*24*365*100),1/(3600*24*365*10),1/(3600*24*365)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)
ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum \n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_%s_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,expname2,'COMPARISON',nsmooth,pct*100,axopt),dpi=200)


#
# %% Quick Plot of CESM Variance
# 

fullpic  = "FULL_PIC_SST_lon330_lat50.npy"
slabpic  = "SLAB_PIC_SST_lon330_lat50.npy"
cesmfull = np.load(datpath+fullpic)
cesmslab = np.load(datpath+slabpic)

fig,ax=plt.subplots(1,1,figsize=(8,3))


csst = [cesmfull,cesmslab]
ccol = ['k','gray']
clab = ["CESM-Full","CESM-Slab"]
for i in range(2):
    
    sstann = proc.ann_avg(csst[i],0)
    
    #win = np.ones(10)/10
    #sstann = np.convolve(sstann,win,mode='valid')
    
    yrs = np.arange(0,sstann.shape[0])
    
    plabel = clab[i] + r", 1$\sigma=%.2f$" % np.std(sstann)
    ax.plot(sstann,label=plabel,lw=0.5,color=ccol[i])
    
    print("Std for %s is %.2f"%(labels[i],np.std(sst[i])))
ax.legend(fontsize=8,ncol=3)
ax.set_xlabel("Years")
ax.set_ylabel("degC")
ax.set_title("CESM SST (Annual)")
ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig("%sCESMSST_comparison.png"%(outpath),dpi=150)



#%% Quick plot of the Stochmod output

fig,ax=plt.subplots(1,1,figsize=(8,3))

for i in [1,2,3]:
    
    sstann = proc.ann_avg(sst[i],0)
    plabel = labels[i] + r", 1$\sigma=%.2f$" % np.std(sstann)
    win = np.ones(10)/10
    sstann = np.convolve(sstann,win,mode='valid')
    
    yrs = np.arange(0,sstann.shape[0])
    
    
    ax.plot(sstann,label=plabel,lw=0.5,color=expcolors[i])
    #ax.plot(sst[i],label=plabel,lw=0.5,color=expcolors[i])
    print("Std for %s is %.2f"%(labels[i],np.std(sst[i])))
    print("Std for Ann mean %s is %.2f"%(labels[i],np.std(sstann)))
ax.legend(fontsize=8,ncol=3)
ax.set_xlabel("Years")
ax.set_ylabel("degC")
ax.set_title("Stochastic Model SST (10-year Running Mean)")
ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig("%sStochasticModelSST_comparison.png"%(outpath),dpi=150)


#%% Check the area under the curve


for i in [1,2,3]:
    freq = freqs[i]*dt
    spec = specs[i]/dt
    nf = len(spec)
    df = np.abs((freq[:-1]-freq[1:])).mean()
    svar = (freq*df).sum()
    
    
    sstvar = sst[i].var()

