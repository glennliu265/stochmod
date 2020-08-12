#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:41:47 2020

@author: gliu


Script to run stochmod at a single point, with a selected set of experiments...

"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import xarray as xr
import time

from scipy import signal
from scipy import stats


# Add Module to search path
import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
import scm
from amv import viz,proc

#%% User Edits -----------------------------------------------------------------     

# Point Selection
lonf      = -30
latf      = 65

# Experiment Settings
funiform = 4 # 0 = nonuniform; 1 = uniform; 2 = NAO-like (DJFM); 3 = NAO DJFM- Monthly Flx; 4 = NAO+Flx Monthly

# Model Parameters...
nyr = 10000
t_end = 12*nyr
dt = 3600*24*30
T0 = 0
seasonal_damping = 0
seasonal_forcing = 0

# Forcing Parameters
runid = "001"
genrand = 0
fscale   = 10

# Ocean Parameters
hfix     = 50

# Other settings
cp0      = 4218
rho      = 1000

# Autocorrelation Parameters#
kmon       = 2                 # Lag 0 base month
lags       = np.arange(0,61,1) # Number of lags to include
detrendopt = 0                 # Detrend before autocorrelation

# Bounding Box for visualization
lonW = -80
lonE = 20
latS = -20
latN = 90
bbox = [lonW,lonE,latS,latN]

# Paths
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
scriptpath = projpath + '03_Scripts/stochmod/'
datpath = projpath + '01_Data/'
outpath = projpath + '02_Figures/20200811/'

# plotting strings
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")
forcingname = ("All Random","Uniform","$(NAO & NHFLX)_{DJFM}$","$NAO_{DJFM}  &  NHFLX_{Mon}$","$(NAO  &  NHFLX)_{Mon}$")
mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
#%% Load Damping and MLD Data

# Load Damping Data
damppath = '/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/01_hfdamping/01_Data/'
dampmat = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp = loadmat(damppath+dampmat)
lon = np.squeeze(loaddamp['LON1'])
lat = np.squeeze(loaddamp['LAT'])
damping = loaddamp['ensavg']

# Load MLD data
hclim      = np.load(datpath+"HMXL_hclim.npy") # Climatological MLD
kprevall   = np.load(datpath+"HMXL_kprev.npy") # Entraining Month



#%% Load Random Time Series...
if genrand == 1:
    randts = np.random.normal(0,1,size=t_end)/4
    np.save(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid),randts)
else:
    randts = np.load(datpath+"stoch_output_%iyr_run%s_randts.npy"%(nyr,runid))
    
#%% Load NAO-like forcing

# DJFM NHFLX Regressed to DJFM NAOIndex
if funiform == 2:
    
    naoforce = np.load(datpath+"NAO_Forcing_EnsAvg.npy") #lon x lat

# Monthly NHFLX regressed to DJFM NAOIndex
elif funiform == 3:

    naoforcing = np.load(datpath+"Monthly_NAO_Regression.npy") #[Ens x Mon x Lat x Lon]
    NAO1 = np.nanmean(naoforcing,axis=0) # Take PC1, Ens Mean and Transpose
    NAO1 = np.transpose(NAO1,(2,1,0))
    
    # Convert Longitude from degrees East
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    lon180,NAO1 = proc.lon360to180(lon360,NAO1)

    naoforce = NAO1 * -1     #lon x lat x mon

# Monthly NHFLX regressed to Monthly NAOIndex
elif funiform == 4:
    
    lon360 =  np.load(datpath+"CESM_lon360.npy")
    naomon = np.load(datpath+"NAO_Monthly_Regression_PC.npz")
    naomon = naomon['eofall']
    naomon = np.nanmean(naomon,0) # Take ens mean
    naomon = np.transpose(naomon,(2,1,0))
    _,naoforce = proc.lon360to180(lon360,naomon) #lon x lat x mon
else:
    naopt = 1

#%% Run model 4 times, once for each treatment of the ocean....


klon,klat = proc.find_latlon(lonf,latf,lon,lat)


# Get terms for the point
damppt = damping[klon,klat,:]
hpt    = hclim[klon,klat,:]
kprev  = kprevall[klon,klat,:]
kmon   = hpt.argmax()
if funiform > 2:
    naopt  = naoforce[klon,klat,:]
elif funiform == 2:
    naopt = naoforce[klon,klat]

# Introduce Artificial seasonal cycle in damping
if seasonal_damping == 1:
    scycle = np.nanmean(damppt) + np.sin(np.pi*np.arange(0,12)/11) * np.nanmax(np.abs(damppt))
    damppt = np.roll(scycle,-1*int(4-np.abs(damppt).argmax()))
    plt.plot(damppt)



# Set Damping Parameters
lbd,lbd_entr,FAC,beta = scm.set_stochparams(hpt,damppt,dt,ND=False)

# Introduce artificial seasonal cycle in forcing
if seasonal_forcing == 1:
    scycle = np.sin(np.pi*np.arange(0,12)/11) * np.nanmax(np.abs(naopt)) #+ np.nanmean(naopt) 
    naopt = np.roll(scycle,-1*int(4-np.abs(naopt).argmax()))
    plt.plot(naopt)

F = {}
Fmagall = {}
for l in range(3):
    
    hvarmode = l
    
    # Set up forcing term
    if hvarmode == 0:
        hchoose = 50
    elif hvarmode == 1:
        hchoose = np.max(np.abs(hpt))
    elif hvarmode == 2:
        hchoose = hpt
    
    
    if funiform >=2:
        Fmag = naopt*dt/cp0/rho/hchoose * FAC
    else:
        Fmag = np.array([1]) # No other forcing magnitude applied
    
    
    if funiform == 2 & l < 2:
        Fmag = np.array(Fmag)
    
    Fmagall[l] = np.copy(Fmag)
    
    
    # Tile forcing magnitude to repeat seasonal cycle    
    if funiform <= 1:
        F[l] = randts * Fmag * fscale
    elif (hvarmode == 2) | (funiform > 2):
        F[l] = randts * np.tile(Fmag,nyr) * fscale #* np.tile(FAC,nyr)
        
    else:
        F[l] = randts * Fmag * fscale

# Run models
sst = {}
noise = {}
dampts = {}
 # Loop nonentraining model
for l in range(3):
    start = time.time()
    sst[l],noise[l],dampts[l] = scm.noentrain(t_end,lbd[l],T0,F[l])
    elapsed = time.time() - start
    tprint = "No Entrain Model, mode %i, ran in %.2fs" % (l,elapsed)
    print(tprint)

# Loop for entrain model
if hvarmode == 2:
    start = time.time()
    sst[3],noise[3],dampts[3],eentrain,td=scm.entrain(t_end,lbd_entr,T0,F[2],beta,hpt,kprev,FAC)
    elapsed = time.time() - start
    tprint = "Entrain Model ran in %.2fs" % (elapsed)
    print(tprint)



#%% Plot Point

model = 3 # Select Model( 0:hfix || 1:hmax || 2:hvar || 3: entrain)

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
plt.title("%s Forcing at LON: %02d LAT: %02d \n %s" % (forcingname[funiform],lonf,latf,fstats))


plt.subplot(2,1,2)
plt.plot(tper,sstpt)
plt.plot(yper,sstptann,color='k',label='Ann. Avg')
plt.ylabel("SST ($^{\circ}C$)",fontsize=10)
plt.xlabel("Time(Months)",fontsize=10)
plt.legend()
#plt.title("Detrended, Deseasonalized SST at LON: %02d LAT: %02d \n Mean: %.2f || Std: %.2f || Max: %.2f" % (lonf,latf,np.nanmean(sstpt),np.nanstd(sstpt),np.nanmax(np.abs(sstpt))))
plt.title("SST (%s) \n %s" % (modelname[model],tstats))


plt.tight_layout()

plt.savefig(outpath+"Stochmodpt_dsdt_SST_run%s_lon%02d_lat%02d_model%i_funiform%i_fscale%i.png"%(runid,lonf,latf,model,funiform,fscale),dpi=200)


#%% # Plot Each Term in Model To compare < NO ENTRAIN > 
#model =  0# 0-2
yrstart = 0
yrstop  = nyr


for model in range(3):

    locstringfig = "LON: %02d LAT: %02d" % (lonf,latf)
    monrange = np.arange(12*yrstart,12*yrstop)
    
    
    sstpt   = sst[model][monrange]
    noisept = np.pad(noise[model],(1,0),'constant')[monrange]
    damptspt = np.pad(dampts[model],(1,0),'constant')[monrange]
    Fpt = F[model][monrange]
     
    # Append zero to end
    # noisept = np.pad(noisept,(1,0),'constant')
    # damptspt = np.pad(damptspt,(1,0),'constant')
    
    
    # Find maximum
    maxvals = [np.nanmax(np.abs(sstpt)),
               np.nanmax(np.abs(noisept)),
               np.nanmax(np.abs(damptspt))
               ]
    maxval = np.max(maxvals)
    
    
    units = "$^{\circ}C$"
    
    
    # Plot for no-entrain model
    fig,ax = plt.subplots(3,1,figsize=(8,6)) 
    plt.style.use('ggplot')
    
    plt.subplot(3,1,1)
    figtitle="Forcing %s" % forcingname[funiform]
    viz.plot_annavg(noisept,units,figtitle,ymax=maxval)
    
    plt.subplot(3,1,2)
    figtitle = "Damping"
    viz.plot_annavg(damptspt,units,figtitle,ymax=maxval)
    
    plt.subplot(3,1,3)
    figtitle = "SST %s"%  modelname[model]
    viz.plot_annavg(sstpt,units,figtitle,ymax=maxval)
    
    title = "Model [%s] at %s (Yrs %i - %i)" % (modelname[model],locstringfig,yrstart,yrstop,)
    plt.suptitle(title,fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outpath+"Stochmodpt_dsdt_ModelParamComp_run%s_lon%02d_lat%02d_model%i_funiform%i_fscale%i.png"%(runid,lonf,latf,model,funiform,fscale),dpi=200)


#%% Repeat plot for the entraining model




model = 3
yrstart = 0
yrstop  = nyr


monrange = np.arange(12*yrstart,12*yrstop)



# Get points and append zeros where necessary
sstpt   = sst[model][monrange]
noisept = np.pad(noise[model],(1,0),'constant')[monrange]
damptspt = np.pad(dampts[model],(1,0),'constant')[monrange]
entrainpt = np.pad(eentrain,(1,0),'constant')[monrange]


# Find maximum
maxvals = [np.nanmax(np.abs(sstpt)),
           np.nanmax(np.abs(noisept)),
           np.nanmax(np.abs(damptspt)),
           np.nanmax(np.abs(entrainpt)),
           ]
maxval = np.ceil(np.max(maxvals))
ymax = None

# Plot for no-entrain model
fig,ax = plt.subplots(4,1,figsize=(8,8)) 
plt.style.use("ggplot")

plt.subplot(4,1,1)
figtitle="Forcing %s" % forcingname[funiform]
viz.plot_annavg(noisept,units,figtitle,ymax=ymax)
    
plt.subplot(4,1,2)
figtitle = "Damping"
viz.plot_annavg(damptspt,units,figtitle,ymax=ymax)

plt.subplot(4,1,3)
figtitle = "Entrain"
viz.plot_annavg(entrainpt,units,figtitle,ymax=ymax)


plt.subplot(4,1,4)
figtitle = "SST %s"%  modelname[model]
viz.plot_annavg(sstpt,units,figtitle,ymax=ymax)
    
title = "Model [%s] at %s (Yrs %i - %i)" % (modelname[model],locstringfig,yrstart,yrstop,)
plt.suptitle(title,fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(outpath+"Stochmodpt_dsdt_ModelParamComp_run%s_lon%02d_lat%02d_model%i_funiform%i_fscale%i.png"%(runid,lonf,latf,model,funiform,fscale),dpi=200)



#%% Plot forcing magnitude, before scaling

fig,ax = plt.subplots(1,1,figsize=(5,3))
plt.style.use("seaborn-bright")
for l in range(3):
    ax.plot(Fmagall[l]*fscale,label=modelname[l])
plt.xticks(np.arange(0,12,1),mons3)
plt.legend()
plt.title("%s Forcing Magnitude Comparison %ix" %(forcingname[funiform],fscale))
plt.savefig(outpath+"Fmag_Comparison_funiform%i_fscale_%i_Fac0.png"%(funiform,fscale),dpi=200)


# Plot with application of reduction factor (without hfixed/max version, much less...)
fig,ax = plt.subplots(1,1,figsize=(5,3))
plt.style.use("seaborn-bright")
for l in range(3):
    ax.plot(Fmagall[l]*fscale*FAC,label=modelname[l])
plt.xticks(np.arange(0,12,1),mons3)
plt.legend()
plt.title("%s Forcing Magnitude Comparison %ix" %(forcingname[funiform],fscale))
plt.savefig(outpath+"Fmag_Comparison_funiform%i_fscale_%i_Fac1.png"%(funiform,fscale),dpi=200)

#%% Plot MLD and Damping parameter to compare


fig,ax1=plt.subplots(1,1,figsize=(5,3))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('$\lambda_{net} (W m^{-2} K^{-1}$)')
ln1 = ax1.plot(mons3,damppt,color='r',label=r'$\lambda_{net}$')
ax1.tick_params(axis='y',labelcolor=color)
ax1.grid(None)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mixed Layer Depth (m)',color=color)

ln2 = ax2.plot(mons3,hpt,color='b',label=r'HMXL')
ax2.tick_params(axis='y',labelcolor=color)
ax2.grid(None)

# Set Legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=0)

# Set Title
titlestr = 'MLD and $\lambda$ (Ensemble Average) \n Lon: ' + str(lonf) + ' Lat: '+ str(latf)
plt.title(titlestr)
plt.tight_layout()
#plt.grid(True)
plt.savefig(outpath+"Damping_MLD_Compare_lon%02d_lat%02d.png"%(lonf,latf),dpi=200)

#%% Plot the sst autocorrelation



xlim = [0,61]
xtk =  np.arange(xlim[0],xlim[1]+2,2)

kmonth = hpt.argmax()
autocorr = {}
for model in range(4):
    
    # Get the data
    tsmodel = sst[model]
    tsmodel = proc.year2mon(tsmodel) # mon x year
    
    # Deseason (No Seasonal Cycle to Remove)
    tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
    
    # Plot
    autocorr[model] = proc.calc_lagcovar(tsmodel,tsmodel,lags,kmonth+1,0)
    

# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")
for model in range(4):
    ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("Month %i SST Autocorrelation at LON:%i Lat:%i \n Forcing %s" % (kmonth+1,lonf,latf,forcingname[funiform]))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"SST_Autocorrelation_Mon%02d_run%s_lon%02d_lat%02d_funiform%i_fscale%i.png"%(kmonth+1,runid,lonf,latf,funiform,fscale),dpi=200)


#%% Repeat, calculating autocorrelation for forcing

xlim = [0,36]

kmonth = hpt.argmax()
fautocorr = {}
for model in range(3):
    
    # Get the data
    tsmodel = F[model]
    tsmodel = proc.year2mon(tsmodel) # mon x year
    
    # Deseason (No Seasonal Cycle to Remove)
    tsmodel2 = tsmodel - np.mean(tsmodel,1)[:,None]
    
    # Plot
    fautocorr[model] = proc.calc_lagcovar(tsmodel,tsmodel,lags,kmonth,0)


# plot results
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")
for model in range(3):
    ax.plot(lags,fautocorr[model],label=modelname[model])
plt.title("Month %i Forcing Autocorrelation at LON:%i Lat:%i \n Forcing %s" % (kmonth+1,lonf,latf,forcingname[funiform]))
plt.legend()
plt.xlim(xlim)
plt.style.use("seaborn-bright")
plt.savefig(outpath+"Forcing_Autocorrelation_Mon%02d_run%s_lon%02d_lat%02d_funiform%i_fscale%i.png"%(kmonth+1,runid,lonf,latf,funiform,fscale),dpi=200)

#%% Diagnostic Autocorrelation Plot 


xlims = (0,36)
# Plot Autocorrelation
fig,ax = plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn")
for model in range(4):
    ax.plot(lags,autocorr[model],label=modelname[model])
plt.title("Month %i SST Autocorrelation at LON:%i Lat:%i \n Forcing %s" % (kmonth+1,lonf,latf,forcingname[funiform]))
plt.xticks(xtk)
plt.legend()
plt.grid(True)
plt.xlim(xlim)
plt.style.use("seaborn-bright")

# Plot seasonal autocorrelation

choosevar = "Forcing"
hm = 0

if choosevar == "Damping":
    invar = damppt
    varname = "Damping (ATMOSPHERIC)"
elif choosevar == "Beta":
    invar = beta
    varname = "$ln(h(t+1)/h(t))$"
elif choosevar == "MLD":
    invar = hpt
    varname = "Mixed Layer Depth"
elif choosevar == "Lambda Entrain":
    invar = lbd_entr
    varname = "Lambda (Entrain)"
    
elif choosevar == "Lambda":
    invar = lbd[hm]
    varname = "Lambda Mode %i" % hm
    
elif choosevar == "Forcing":
    invar = naopt
    varname = "NAO Forcing (W$m^{-2}$)"
elif choosevar == "Fmag":
    invar = Fmagall[hm]
    varname = "Fmag Mode %i" % hm
elif choosevar == "NAO":
    invar = naopt
    varname = "NAO Forcing"


    
# Find index of maximum and roll so that it is now the first entry (currently set to only repeat for 5 years)
kmonth = hpt.argmax()
maxfirsttile = np.tile(np.roll(invar,-1*kmonth),5)
maxfirsttile = np.concatenate((maxfirsttile,[maxfirsttile[0]]))

# Twin axis and plot
ax2 = ax.twinx()
ax2.plot(lags,maxfirsttile,color=[0.6,0.6,0.6])
ax2.grid(False)
ax2.set_ylabel(varname)
plt.xlim(xlims)
plt.savefig(outpath+"scycle2Fixed_SST_Autocorrelationv%s_Mon%02d_run%s_lon%02d_lat%02d_funiform%i_fscale%i.png"%(choosevar,kmonth+1,runid,lonf,latf,funiform,fscale),dpi=200)




#%% Plot NAO Forcing at this point


fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(mons3,naopt)
ax.set_title("NAO Forcing (Monthly), Ensemble Mean")
ax.set_ylabel("$Wm^{-1}\sigma^{-1}$")
plt.savefig(outpath+"NAOForcing_funiform%i.png"%funiform,dpi=200)

#%% Plot NAO Forcing and hclim
fig,ax1=plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Depth(m)')
ln1 = ax1.plot(mons3,hpt,color='r',label=r'HMXL')
ax1.tick_params(axis='y',labelcolor=color)
ax1.grid(None)
#ax1.set_xticks(np.arange(1,13,1),mons3)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("$Wm^{-1}\sigma^{-1}$",color='blue')

ln2 = ax2.plot(mons3,naopt,color='b',label=r'NAO Forcing')
ax2.tick_params(axis='y',labelcolor=color)
ax2.grid(None)

# Set Legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=0)

# Set Title
titlestr = ' MLD and NAO Forcing (Ensemble Average) \n Lon: ' + str(lonf) + ' Lat: '+ str(latf)
plt.title(titlestr)
plt.tight_layout()
plt.savefig(outpath+"NAOForcing_MLD_funiform%i.png"%funiform,dpi=200)


#%% Plot Beta and hclim
fig,ax1=plt.subplots(1,1,figsize=(6,4))
plt.style.use("seaborn-bright")
color = 'tab:red'
ax1.set_xlabel('Month')
ax1.set_ylabel('Depth(m)')
ln1 = ax1.plot(mons3,hpt,color='r',label=r'HMXL')
ax1.tick_params(axis='y',labelcolor=color)
ax1.grid(None)
#ax1.set_xticks(np.arange(1,13,1),mons3)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel("$log(h/h^{t+1})$",color='blue')

ln2 = ax2.plot(mons3,beta,color='b',label=r'Beta')
ax2.tick_params(axis='y',labelcolor=color)
ax2.grid(None)

# Set Legend
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs,loc=0)

# Set Title
titlestr = ' MLD and NAO Forcing (Ensemble Average) \n Lon: ' + str(lonf) + ' Lat: '+ str(latf)
plt.title(titlestr)
plt.tight_layout()
plt.savefig(outpath+"NAOForcing_MLD_funiform%i.png"%funiform,dpi=200)

