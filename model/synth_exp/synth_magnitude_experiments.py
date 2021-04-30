#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Stochastic Model, See the impact of varying magnitude
of each seasonal cycle

Created on Thu Apr 29 17:37:09 2021

@author: gliu
"""

from scipy.interpolate import interp1d
from scipy.io import loadmat,savemat
from scipy import signal
from tqdm import tqdm

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import time
import cmocean
import sys


# Custom Modules
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import yo_box as ybx
import scm




#%% Settings

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath = projpath + '02_Figures/20210430/'
proc.makedir(outpath)

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
#fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")
fullauto = np.load(datpath+"CESM_clim/TS_FULL_Autocorrelation.npy")


mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','orange','magenta','red')
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
config.pop('Fpt',None)
config['Fpt'] = np.array([55.278503, 53.68089 , 42.456623, 33.448967, 22.954145, 22.506973,
       22.151728, 24.135042, 33.337887, 40.91648 , 44.905064, 51.132706])
config['Fpt'] = np.array([60.278503, 53.68089 , 42.456623, 33.448967, 22.954145, 22.506973,
       22.151728, 19.135042, 33.337887, 40.91648 , 50.905064, 58.132706])
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

# Calculate Confidence Intervals
conf  =0.95
tails = 2

nlags   = len(lags)
cfstoch = np.zeros([4,nlags,2])
for m in range(4):
    inac = ac[m]
    n = int(len(sst[m])/12)
    cfs = proc.calc_conflag(inac,conf,tails,n)
    cfstoch[m,:,:] = cfs
cfslab = proc.calc_conflag(cesmauto2,conf,tails,898)
cffull = proc.calc_conflag(cesmautofull,conf,tails,1798)

# Plot the Autocorrelation
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation (%s) \n Lag 0 = %s" % (locstringtitle,mons3[mldpt.argmax()])
#ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title,loopvar=damppt)
ax,ax2= viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='k',alpha=0.10)

ax.plot(lags,cesmautofull,color='k',label='CESM Full',ls='dashdot')
ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='k',alpha=0.10)

for i in range(1,4):
    ax.plot(lags,ac[i],label=labels[i],color=expcolors[i])
    ax.fill_between(lags,cfstoch[i,:,0],cfstoch[i,:,1],color=expcolors[i],alpha=0.25)

ax.legend()
#ax3.set_ylabel("Heat Flux Feedback ($W/m^{2}$)")
#ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
#plt.savefig(outpath+"Default_Autocorrelation_CF_%s.png"%locstring,dpi=200)

# Save Default Values
dampdef = damppt.copy()
mlddef = mldpt.copy()
Fptdef = Fpt.copy()

# **************************************************
#%% Grid Sweep Experiments : Damping, Constant Value
# **************************************************
expname="DampingCVary"
testvalues = np.arange(1,26,1)
testparam  = 'damppt'

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in tqdm(enumerate(testvalues)):
    #config['Fpt'] = np.ones(12)*37.24208402633666
    st = time.time()
    config[testparam] = np.ones(12)*val
    ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    config.pop(testparam,None)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

config.pop(testparam,None)
#%% Visualize the results

model = 1 # 
nlag = len(config['lags'])
nexp = len(testvalues)

acalls = np.zeros((nexp,nlag))
for e,ac in enumerate(acall):
    acalls[e,:] = acall[e][model]

# Pcolor Plot
fig,ax = plt.subplots(1,1)
pcm = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
fig.colorbar(pcm,ax=ax)
ax.set_ylabel("Damping (W/m2)")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
plt.savefig("%sLag_v_Damping_pcolor_%s.png"%(outpath,expname),dpi=200)


# Pcolor Plot, differences
fig,ax = plt.subplots(1,1)
pcm = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
fig.colorbar(pcm,ax=ax)
ax.set_ylabel("Damping (W/m2)")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
plt.tight_layout()
plt.savefig("%sLag_v_Damping_pcolor_diff_%s.png"%(outpath,expname),dpi=200)

# Plot minimum RMSE
rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
print("Minumum RMSE was for %i with value %f" % (np.argmin(rmses),rmses.min()))

# Line Plots
fig,ax = plt.subplots(1,1)
ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
for lam in range(25):
    ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/25)*.5,color='b')
    
ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)"%testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")   
ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
ax.grid(True,ls='dotted')
plt.savefig("%sLag_v_Damping_lineplot_%s.png"%(outpath,expname),dpi=200)

XX,YY = np.meshgrid(config['lags'],testvalues[1:])
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma')
fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
ax.set_ylim(25,0)
ax.set_xlim(0,38)
ax.set_zlim(0,1)
ax.set_xticks(np.arange(0,37,6))
ax.set_ylabel("Damping $(W/m^{2})$")
ax.set_xlabel("Lag (months)")
ax.set_zlabel("Correlation")
ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
plt.tight_layout()
plt.savefig("%sLag_v_Damping_3Dplot_%s.png"%(outpath,expname),dpi=200)

# *********************************************************
#%% Grid Sweep Experiments II : Damping, Seasonal Magnitude
# *********************************************************
expname="DampingVVary"

testvalues = np.arange(0.1,2.1,.1)
testparam  = 'damppt'

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in tqdm(enumerate(testvalues)):
    st = time.time()
    config[testparam] = damppt*val
    ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

config.pop(testparam,None)
#%% Visualize the results

model = 1

nlag = len(config['lags'])
nexp = len(testvalues)

acalls = np.zeros((nexp,nlag))
for e,ac in enumerate(acall):
    acalls[e,:] = acall[e][model]

#ytk=np.arange(0,len(testvalues),1)
ytk = np.arange(.1,2.2,.2)

# Pcolor Plot
fig,ax = plt.subplots(1,1,figsize=(8,8))
im = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
#im = ax.imshow(acalls,cmap='magma',vmin=0,vmax=1)
fig.colorbar(im,ax=ax,fraction=0.015)
#ax.set_yticklabels(ytk)
#ax.set_yticklabels(testvalues,fontsize=10)
#plt.gca().invert_yaxis()
ax.set_ylabel("Damping Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
plt.savefig("%sLag_v_Damping_pcolor_%s.png"%(outpath,expname),dpi=200)

# Pcolor Plot, differences
fig,ax = plt.subplots(1,1,figsize=(8,4))
im = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
#im = ax.imshow(acalls-cesmauto,cmap=cmocean.cm.balance,vmin=-.5,vmax=.5)
fig.colorbar(im,ax=ax,fraction=0.015)
#ax.set_yticks(ytk)
#ax.set_yticklabels(testvalues,fontsize=10)
#plt.gca().invert_yaxis()
ax.set_ylabel("Damping Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
plt.savefig("%sLag_v_Damping_pcolordiff_%s.png"%(outpath,expname),dpi=200)

# Plot minimum RMSE
rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
print("Minumum RMSE was for %i with value %f" % (testvalues[np.argmin(rmses)],rmses.min()))


# Line Plots
fig,ax = plt.subplots(1,1)
ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
for lam in range(len(testvalues)):
    ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/len(testvalues))*.5,color='b')
    
ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")   
ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
ax.grid(True,ls='dotted')
plt.savefig("%sLag_v_Damping_lineplot_%s.png"%(outpath,expname),dpi=200)

XX,YY = np.meshgrid(config['lags'],testvalues[1:])
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma',vmin=0,vmax=1)
fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
ax.set_ylim(testvalues[-1],testvalues[0])
ax.set_xlim(0,38)
ax.set_zlim(0,1)
ax.set_xticks(np.arange(0,37,6))
ax.set_ylabel("Damping Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_zlabel("Correlation")
ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
plt.tight_layout()
plt.savefig("%sLag_v_Damping_3Dplot_%s.png"%(outpath,expname),dpi=200)


# Plot damping values
fig,ax = plt.subplots(1,1)
for i in range(len(testvalues)):
    ax.plot(mons3,paramsall[i][1],label="",alpha=(i/len(testvalues))*.5,color='b')
ax.plot(mons3,damppt,color='k',label='Original Seasonal Cycle')
ax.plot(mons3,paramsall[np.argmin(rmses)][1],color='r',label="Best, (%.2fx)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Damping (W/m2)")    
ax.grid(True,ls='dotted')
plt.savefig("%sDamping_Values_%s.png"%(outpath,expname),dpi=200)

# *************************************************
#%% Grid Sweep Experiments III : Forcing (Constant)
# *************************************************
expname="ForcingCVary"

testvalues = np.arange(10,101,1)
testparam  = 'Fpt'

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in tqdm(enumerate(testvalues)):
    
    #config['Fpt'] = np.ones(12)*37.24208402633666
    st = time.time()
    config[testparam] = np.ones(12)*val
    ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

config.pop(testparam,None)
#%% Visualize the results

model = 1

nlag = len(config['lags'])
nexp = len(testvalues)

acalls = np.zeros((nexp,nlag))
for e,ac in enumerate(acall):
    acalls[e,:] = acall[e][model]


# Pcolor Plot
fig,ax = plt.subplots(1,1)
pcm = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
fig.colorbar(pcm,ax=ax)
ax.set_ylabel("Forcing (W/m2)")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
plt.savefig("%sLag_v_Forcing_pcolor_%s.png"%(outpath,expname),dpi=200)


# Pcolor Plot, differences
fig,ax = plt.subplots(1,1)
pcm = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
fig.colorbar(pcm,ax=ax)
ax.set_ylabel("Forcing (W/m2)")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
plt.tight_layout()
plt.savefig("%sLag_v_Forcing_pcolor_diff_%s.png"%(outpath,expname),dpi=200)

# Plot minimum RMSE
rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
print("Minumum RMSE was for %i with value %f" % (np.argmin(rmses),rmses.min()))


# Line Plots
fig,ax = plt.subplots(1,1)
ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
for lam in range(25):
    ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/25)*.5,color='b')
    
ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)"%testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")   
ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
ax.grid(True,ls='dotted')
plt.savefig("%sLag_v_Forcing_lineplot_%s.png"%(outpath,expname),dpi=200)

XX,YY = np.meshgrid(config['lags'],testvalues[1:])
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma')
fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
ax.set_ylim(100,0)
ax.set_xlim(0,38)
ax.set_zlim(0,1)
ax.set_xticks(np.arange(0,37,6))
ax.set_ylabel("Forcing $(W/m^{2})$")
ax.set_xlabel("Lag (months)")
ax.set_zlabel("Correlation")
ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
plt.tight_layout()
plt.savefig("%sLag_v_Forcing_3Dplot_%s.png"%(outpath,expname),dpi=200)

# *********************************************************
#%% Grid Sweep Experiments IV : Forcing, Seasonal Magnitude
# *********************************************************
testvalues = np.arange(0.1,2.1,.1)*10#[0.25,0.5,1,2,4,8,16,32,64,128]

testparam  = 'Fpt'
expname="ForcingVVary1m"

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in tqdm(enumerate(testvalues)):
    st = time.time()
    #vals[7]  = val
    config[testparam] = Fpt*testvalues[i]
    ac,sst,dmp,frc,ent,Td,kmonth,params=scm.synth_stochmod(config,projpath=projpath)

    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))
    
config.pop(testparam,None)
#%% Visualize the results

model = 1

nlag = len(config['lags'])
nexp = len(testvalues)

acalls = np.zeros((nexp,nlag))
for e,ac in enumerate(acall):
    acalls[e,:] = acall[e][model]

#ytk=np.arange(0,len(testvalues),1)
ytk = np.arange(.1,2.2,.2)

# Pcolor Plot
fig,ax = plt.subplots(1,1,figsize=(8,8))
im = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
#im = ax.imshow(acalls,cmap='magma',vmin=0,vmax=1)
fig.colorbar(im,ax=ax,fraction=0.015)
#ax.set_yticklabels(ytk)
#ax.set_yticklabels(testvalues,fontsize=10)
#plt.gca().invert_yaxis()
ax.set_ylabel("Forcing Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
plt.savefig("%sLag_v_Forcing_pcolor_%s.png"%(outpath,expname),dpi=200)

# Pcolor Plot, differences
fig,ax = plt.subplots(1,1,figsize=(8,4))
im = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
#im = ax.imshow(acalls-cesmauto,cmap=cmocean.cm.balance,vmin=-.5,vmax=.5)
fig.colorbar(im,ax=ax,fraction=0.015)
#ax.set_yticks(ytk)
#ax.set_yticklabels(testvalues,fontsize=10)
#plt.gca().invert_yaxis()
ax.set_ylabel("Forcing Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
plt.savefig("%sLag_v_Forcing_pcolordiff_%s.png"%(outpath,expname),dpi=200)

# Plot minimum RMSE
rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
print("Minumum RMSE was for %i with value %f" % (testvalues[np.argmin(rmses)],rmses.min()))


# Line Plots
fig,ax = plt.subplots(1,1)
ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
for lam in range(len(testvalues)):
    ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/len(testvalues))*.5,color='b')
    
ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")   
ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
ax.grid(True,ls='dotted')
plt.savefig("%sLag_v_Forcing_lineplot_%s.png"%(outpath,expname),dpi=200)

XX,YY = np.meshgrid(config['lags'],testvalues[1:])
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma',vmin=0,vmax=1)
fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
ax.set_ylim(testvalues[-1],testvalues[0])
ax.set_xlim(0,38)
ax.set_zlim(0,1)
ax.set_xticks(np.arange(0,37,6))
ax.set_ylabel("Forcing Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_zlabel("Correlation")
ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
plt.tight_layout()
plt.savefig("%sLag_v_Forcing_3Dplot_%s.png"%(outpath,expname),dpi=200)


# Plot Forcing values
fig,ax = plt.subplots(1,1)
for i in range(len(testvalues)):
    ax.plot(mons3,paramsall[i][4],label="",alpha=(i/len(testvalues))*.5,color='b')
ax.plot(mons3,Fpt,color='k',label='Original Seasonal Cycle')
ax.plot(mons3,paramsall[np.argmin(rmses)][4],color='r',label="Best, (%.2fx)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Forcing (W/m2)")    
ax.grid(True,ls='dotted')
plt.savefig("%sForcing_Values_%s.png"%(outpath,expname),dpi=200)

