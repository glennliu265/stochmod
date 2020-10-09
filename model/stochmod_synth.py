#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:25:29 2020

@author: gliu
"""

# Fully Synthetic Stochastic Model

from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#import seaborn as sns
import xarray as xr
import time
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

from amv import viz
import matplotlib.pyplot as plt
import time


# --------------
# % Path and Functions--------------------------------------------------------
# --------------

# Path Setting and import modules
startall = time.time()
projpath    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
outpath     = projpath + '02_Figures/Scrap/'
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'   
import scm
from amv import proc,viz


def scm_synt(hclim,dampingr,NAO1,randts,lags,applyfac,T0=0,returncomponents=False):
    
    t_end = randts.shape[0] # Get simulation length
    
    
    # ----------------------------
    # % Set-up damping parameters and kprev
    # ----------------------------
    # Converts damping parameters from raw form (Watts/m2) to (deg/sec)
    # Also calculates beta and FAC
    # Note: Consider combining with NAO Forcing conversion?
   
        
    # Find Kprev
    kpreva,_ = scm.find_kprev(hclim)
    viz.viz_kprev(hclim,kpreva)
    
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(hclim,dampingr,dt,ND=False,rho=rho,cp0=cp0,hfix=hfix)
    
    
    """
    Out Format:
        lbd -> Dict (keys 0-3) representing each mode, damping parameter
        lbd_entr -> array of entrainment damping
        FAC -> Dict (keys 0-3) representing each model, integration factor
        beta ->array [Mon]
        kprev -> array [Mon]
    """
    
    # ----------------------------
    # % Set Up Forcing           ------------------------------------------------
    # ----------------------------
    
    # Convert NAO from W/m2 to degC/sec. Returns dict with keys 0-2 (same as scm.convert_NAO)
    NAOF={}
    conversionfac = dt/cp0/rho
    if applyfac == 0: # Dont apply MLD to NAOF
        for hi in range(3):
            NAOF[hi] = NAO1 * conversionfac
    else: # Apply each MLD case to NAOF
        NAOF[0] = NAO1 * conversionfac /hfix
        NAOF[1] = NAO1 * conversionfac / hclim.max()
        NAOF[2] = NAO1 * conversionfac / hclim
    
    # Use random time series to scale the forcing pattern
    F = {}
    tilecount = int(12/NAOF[0].shape[0]*nyr)
    for hi in range(3):
        F[hi] = np.tile(NAOF[hi].squeeze(),tilecount) *randts * fscale 
        
        # Save Forcing if option is set
        if saveforcing == 1:
            np.save(output_path+"stoch_output_%s_Forcing.npy"%(runid),F)
    
    """
    Output:
        F - dict (keys = 0-2, representing each MLD treatment) [time (simulation length)]
        Fseas - dict (keys = 0-2, representing each MLD treatment) [ month]
        
    """
    
    # ----------
    # %RUN MODELS -----------------------------------------------------------------
    # ----------
    # Set mulFAC condition based on applyfac
    if applyfac == 2:
        multFAC = 1 # Don't apply integrationreduction factor if applyfac is set to 0 or 1
    else:
        multFAC = 0

    
    #% Run Models <SET>
    
    """
    Inputs:
        FAC    - Dict of Integration Factors (0-3)
        lbd    - Dict of Damping Parameters (0-3)
        F      - Dict of Forcings (0-2)
        kprev  - Array/Vector of Entraining Months
        beta   - Array/Vector of Entrainment velocities
        hclima - Array/Vector of MLD cycle 
    
        Fixed Params: T0 (Initial SST), t_end (end timestep), multFAC (0 or 1)
    
    Output
        sst - Dict of model output (0-3)
    
    """
    
    # Preallocate dictionary to store results (Use tuple instead?)
    sst = {}
    # Run Model Without Entrainment
    # Loop for each Mixed Layer Depth Treatment
    for hi in range(3):
        start = time.time()
        
        # Select damping and FAC based on MLD
        FACh = FAC[hi]
        lbdh = lbd[hi]
        
        # Select Forcing
        Fh  = F[hi]
                
        # Run Point Model
        start = time.time()
        sst[hi],_,_=scm.noentrain(t_end,lbdh,T0,Fh,FACh,multFAC=multFAC)
        elapsed = time.time() - start
        tprint = "\nNo Entrain Model, hvarmode %i, ran in %.2fs" % (hi,elapsed)
        print(tprint)    
            
    
    # Run Model With Entrainment
    start = time.time()
    Fh      = F[2]
    FACh    = FAC[3]
    sst[3]  = scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kpreva,FACh,multFAC=multFAC)
    elapsed = time.time() - start
    tprint  = "\nEntrain Model ran in %.2fs" % (elapsed)
    print(tprint)    
    
    
    # Calculate Autocorrelation
    kmonth = hclim.argmax() # kmonth is the INDEX of the mongth

    autocorr = {}
    for model in range(4):
        
        # Get the data
        tsmodel = sst[model]
        tsmodel = proc.year2mon(tsmodel) # mon x year
        
        # Deseason (No Seasonal Cycle to Remove)
        tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
        
        # Plot
        autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
    
    if returncomponents:
        return sst,autocorr,lbd,FAC,beta,kpreva,F
    return sst,autocorr


# --------------
# % Set Params--------------------------------------------------------
# --------------

# Settings
genrand  = 1
nyr      = 10000
runid    ="000"
fscale   = 1 # Mutiplier to scale forcing
fstd     = 1 # Standard Deviation of Forcing
t_end    = 12000 
funiform = 99# Forcing Type 

# Other intengration Constants
t_end    = 12*nyr      # Calculates Integration Period
dt       = 60*60*24*30 # Timestep size (Will be used to multiply lambda)
T0       = 0           # Initial temperature [degC]
hfix     = 50          # Fixed MLD value (meters)
cp0      = 3850 # Specific Heat [J/(kg*C)]
rho      = 1025 # Density of Seawater [kg/m3]

# Other Settings
# Apply fac options
# 0) Forcing is just the White Noise For ings
# 1) Forcing is white noise (numerator) and includes MLD
# 2) Forcing includes both MLD seasonal cycle AND integration factor
applyfac    = 1 # Apply integration factor and MLD to scaling
nobeta      = 0 # Set to 1 to not include beta in lbd_entrain
saveforcing = 0 # Save Forcing for each point (after scaling, etc)

# Indicate number of lags
lags = np.arange(0,37,1)

## ------------ Script Start -------------------------------------------------

print("Now Running stochmod_region with the following settings: \n")
print("funiform  = " + str(funiform))
print("genrand   = " + str(genrand))
print("fstd      = " + str(fstd))
print("runid     = " + runid)
print("fscale    = " + str(fscale))
print("nyr       = " + str(nyr))
print("Data will be saved to %s" % datpath)
allstart = time.time()

# Set experiment ID
expid = "SYNT_%iyr_funiform%i_run%s_fscale%03d" %(nyr,funiform,runid,fscale)

# Set synthetic timeseries
#dampingr = 
#hclim    =

# SPG Averaged Values
hclim = np.array([302.73951441, 406.22087459, 411.76222341, 249.39158241,
        54.50208403,  21.8281317 ,  21.31412946,  24.702055  ,
        34.68349695,  54.14000163, 104.86907876, 183.21180518])
dampingr = np.array([15.59430453, 12.88939794, 10.80892098,  9.67647269,  8.82878655,
        8.16798998,  9.86816666, 12.92381506, 16.09537405, 18.36355278,
        18.65877477, 18.20871571])
NAO1 = np.ones((12))

# Generate Random Time Series
if genrand == 1: # Generate new time series
    print("Generating New Data")
    randts = np.random.normal(0,fstd,size=t_end) # Removed Divide by 4 to scale between -1 and 1
    np.save(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)
else: # Load old data
    print("Loading Old Data")
    randts = np.load(output_path+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))





# Loop for different expriment settings ,varying lambda and applyfac
applyfacs   = [0,1,2]
experiments = []
names = []
i = 0
for a in range(3): # Vary for 3 cases of forcing treatment
    
    applyfac = applyfacs[a]
    
    for l in range(3): # Vary for 3 cases of damping
        
        if l == 0:
            dampinga = np.ones(12)*dampingr.mean()
            nm = "FIX"
        elif l == 1:
            dampinga = np.ones(12)*dampingr.max()
            nm = "MAX"
        elif l == 2:
            dampinga = dampingr.copy()
            nm = "VAR"
        experiments.append(scm_synt(hclim,dampinga,NAO1,randts,lags,applyfac))
        names.append("applyfac%i_lambda%s" % (a,nm))
        
        i += 1

#%% Make some  plots
region = 1
loctitle = "SPG"
kmonth = hclim.argmax() + 1

# Set Strings
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$","$EAP_{DJFM}$","(EAP+NAO)_{DJFM}")
regions = ("SPG","STG","TRO","NAT")
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
monname=('January','February','March','April','May','June','July','August','September','October','November','December')


mons3tile = np.tile(np.array(mons3),int(np.floor(len(lags)/12))) 
mons3tile = np.roll(mons3tile,-kmonth+1)

for i in range(len(experiments)):
    _,autocorr = experiments[i]
    
    
    expname = expid + "_" + names[i]
    
    # Load CESM Data
    #cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()
    
    xlim = [0,36]
    xtk =  np.arange(xlim[0],xlim[1],2)
    plt.style.use("seaborn-bright")
    
    
    # plot results
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    plt.style.use("seaborn-bright")
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(xtk)
    ax2.set_xticklabels(mons3tile[xtk], rotation = 45)
    ax2.set_axisbelow(True)
    ax2.grid(zorder=0,alpha=0)
    
    # Plot CESM
    #accesm = cesmauto[region]
    #ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)
    
    for model in range(4):
        ax.plot(lags,autocorr[model],label=modelname[model])
    plt.title("%s SST Autocorrelation at %s \n Experiment %s" % (monname[kmonth-1],loctitle,names[i]))
    ax.set_xticks(xtk)
    ax.set_xlim(xlim)
    ax.legend(fontsize=8,ncol=4)
    plt.grid(True)
    plt.style.use("seaborn-bright")
    plt.tight_layout()
    plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s.png"%(kmonth,expname),dpi=200)
    print(i)
        




# ---


#%% Plot Point

model = 3 # Select Model( 0:hfix || 1:hmax || 2:hvar || 3: entrain)

mon = 6
xl  = [11940,12000]
#yl = [-1e200,1e200] 



sstpt = sst[model]
if model == 3:
    Fpt = F[2]
else:
    Fpt = F[model]

# Calculate Annual Averages
sstptann = proc.ann_avg(sstpt,0)
Fptann   = proc.ann_avg(Fpt,0)

# Set plotting parameters and text
tper = np.arange(0,t_end)
yper = np.arange(0,t_end,12)
fstats = viz.quickstatslabel(Fpt)
tstats = viz.quickstatslabel(sstpt)

# Start plot
fig,ax = plt.subplots(2,1,figsize=(8,6))
plt.style.use('ggplot')

plt.subplot(2,1,1)
plt.plot(tper,Fpt)
plt.plot(yper,Fptann,color='k',label='Ann. Avg')
plt.ylabel("Forcing ($^{\circ}C/s$)",fontsize=10)
plt.legend()
plt.title("%s Forcing at %s with Fscale %.2e \n %s " % (forcingname[funiform],loctitle,fscale,fstats))


plt.subplot(2,1,2)
plt.plot(tper,sstpt)
plt.plot(yper,sstptann,color='k',label='Ann. Avg')
plt.ylabel("SST ($^{\circ}C$)",fontsize=10)
plt.xlabel("Time(Months)",fontsize=10)
plt.legend()
#plt.yscale('log')
#plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.title("SST (%s) \n %s" % (modelname[model],tstats))

plt.xlim(xl)
#plt.ylim(yl)
plt.xticks(np.arange(11940,12006,6))
plt.vlines(np.arange(xl[0]+mon-1,xl[1],12),1,1)


plt.tight_layout()

plt.savefig(outpath+"Stochmodpt_dsdt_SST_model%i_%s.png"%(model,expid),dpi=200)





#%%

fig,ax = plt.subplots(1,1)
ax.plot(tper[5000:100],sstpt[1:100])
#ax.set_yscale('log')
#ax.set_ylim([-2e287,0])

#%% Plot the sst autocorrelation

# Load CESM Data
cesmauto = np.load(projpath + "01_Data/Autocorrelation_Region.npy",allow_pickle=True).item()


lags = np.arange(0,37,1)
xlim = [0,36]
xtk =  np.arange(xlim[0],xlim[1]+2,2)
plt.style.use("seaborn-bright")


kmonth = hclim.argmax() # kmonth is the INDEX of the mongth

autocorr = {}
for model in range(4):
    
    # Get the data
    tsmodel = sst[model]
    tsmodel = proc.year2mon(tsmodel) # mon x year
    
    # Deseason (No Seasonal Cycle to Remove)
    tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
    
    # Plot
    autocorr[model] = proc.calc_lagcovar(tsmodel2,tsmodel2,lags,kmonth+1,0)
    

# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")

# Plot CESM
accesm = cesmauto[region]
ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)

for model in range(4):
    ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale: %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s.png"%(kmonth+1,expid),dpi=200)



# Plot Some relevant parameters
#%% Plot the inputs (RAW)
fig,axs=plt.subplots(3,1,sharex=True,figsize=(6,6))
ax=axs[0]
ax.set_title("Mixed Layer Depth")
ax.set_ylabel("meters")
ax.plot(mons3,hclima,color='b')

ax=axs[1]
ax.set_title("$\lambda_{a}$")
ax.set_ylabel("W$m^{-2}$")
ax.plot(mons3,dampinga,color='r')

ax=axs[2]
ax.set_title("$Forcing$")
ax.set_ylabel("W$m^{-2}$")
ax.plot(mons3,dampinga,color='r')

#%% Plot secondary inputs, post processing


#% Plot the 4 lambdas
fig,axs = plt.subplots(3,1,sharex=True,figsize=(6,6))

ax=axs[0]
for model in range(4):
    ax.plot(mons3,lbd[model],label=modelname[model])
ax.set_ylabel("degC/sec")
ax.set_xlabel("Month")
ax.set_title(r"$\lambda (\rho cp_{0}H)^{-1}$")
ax.legend()
    
ax=axs[1]
for model in range(4):
    ax.plot(mons3,FAC[model],label=modelname[model])
ax.set_title(r"Integration Factor")    

#%%

model = 2


xlims = (0,36)
# Plot Autocorrelation
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn")
#for model in range(4):
ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")

# Plot seasonal autocorrelation
choosevar = "MLD"

if choosevar == "Damping":
    invar = dampinga
    varname = "Damping (ATMOSPHERIC)"
    
elif choosevar == "Beta":
    invar = beta
    varname = "$ln(h(t+1)/h(t))$"
    
elif choosevar == "MLD":
    invar = hclima
    varname = "Mixed Layer Depth"
    
elif choosevar == "Lambda Entrain":
    #invar = np.exp(-lbd_entr*dt/(rho*cp0*hpt))
    invar = lbd_entr
    varname = "Lambda (Entrain)"
    
elif choosevar == "Lambda":
    invar = lbd[model]
    varname = "Lambda Mode %i" % model
    
elif choosevar == "Forcing":
    invar = naopt
    varname = "NAO Forcing (W$m^{-2}$)"
elif choosevar == "Fmag":
    invar = Fseas[model] #* fscale
    varname = "Forcing Magnitude with %s (degC)" % modelname[model]
elif choosevar == "NAO":
    invar = naopt
    varname = "NAO Forcing"
elif choosevar == "Damping Term":

    if hm == 0:
        h = hfix
    elif hm == 1:
        h = np.max(hpt)
    elif hm == 2:
        h = hpt
    
    if hm < 3:
        #invar = np.exp(-damppt*dt/(rho*cp0*h))
        invar = np.exp(-lbd[hm])
    else:
        #invar = np.exp(-lbd_entr*dt/(rho*cp0*hpt))
        invar = np.exp(-lbd_entr)
    varname = "Damping Term (degC/mon) Mode %i" % hm


    
# Find index of maximum and roll so that it is now the first entry (currently set to only repeat for 5 years)
kmonth = hclima.argmax()
maxfirsttile = np.tile(np.roll(invar,-1*kmonth-1),3)
maxfirsttile = np.concatenate((maxfirsttile,[maxfirsttile[0]]))

# Twin axis and plot
ax2 = ax.twinx()
ax2.plot(lags,maxfirsttile,color=[0.6,0.6,0.6])
ax2.grid(False)
ax2.set_ylabel(varname)
plt.xlim(xlims)
plt.tight_layout()
plt.savefig(outpath+"scycle2Fixed_SST_Autocorrelationv%s_Mon%02d_%s.png"%(choosevar,kmonth+1,expid),dpi=200)


fig,ax = plt.subplots(2,1)
ax[0].plot(mons3,hclima)
ax[1].plot(mons3,lbd[2])

#%% Plot to Compare Inclusion of Beta in Damping

# Note: I first ran the script with nobeta=1, and saved the following:
#autocorr_nobeta = np.copy(autocorr[3])
# Make sure to run again with nobeta=0 and run the autocorrelation calculation

modcol = ['b','g','r','darkviolet']

# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")

# Plot CESM
accesm = cesmauto[region]
ax = viz.ensemble_plot(accesm,0,ax=ax,color='k',ysymmetric=0,ialpha=0.05)

# Plot Hseas Model
model =2
ax.plot(lags,autocorr[model],label=modelname[model],color=modcol[model])

# Plot Entrain with beta
model =3
ax.plot(lags,autocorr[model],label=modelname[model]+" with Beta",color=modcol[model],ls="dotted")

# Plot Entrain without beta
ax.plot(lags,autocorr_nobeta,label=modelname[model]+" without Beta",color=modcol[model])

plt.title("%s SST Autocorrelation at %s \n Forcing %s Fscale: %.2e" % (monname[kmonth],loctitle,forcingname[funiform],fscale))
plt.xticks(xtk)
plt.legend(ncol=3,fontsize=8)
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_%s_betacomparison.png"%(kmonth+1,expid),dpi=200)


#%% Tiny plot of beta

fig,ax = plt.subplots(1,1,figsize=(4,2))
ax.plot(mons3,beta)
ax.set_title("Beta")
ax.set_xlabel("Months")
plt.savefig(outpath+"Beta_%s.png"%(locstring),dpi=200)

#%% Comparison plot of the lambda, before and after beta inclusion

modcol = ['b','g','r','darkviolet']
fig,ax = plt.subplots(1,1,figsize=(4,3))
lbdname = [0,0,r"$\lambda (\rho cp_{0}H)^{-1}$",r"$\lambda (\rho cp_{0}H)^{-1}$"+r"+ $w_{e}H^{-1}$"]
for model in [2,3]:
    ax.plot(mons3,lbd[model],label=lbdname[model],color=modcol[model])
ax.set_ylabel("degC/sec")
plt.legend()
plt.savefig(outpath+"Lbd_Seas_comparison_%s.png"%(locstring),dpi=200)



