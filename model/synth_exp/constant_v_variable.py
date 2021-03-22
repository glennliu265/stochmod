#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with the Synthetic Stochastic Model

Investigate Constant vs. Varying terms in the Non-Entraining Model

Created on Sun Mar 21 18:40:41 2021
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
outpath = projpath + '02_Figures/20210322_AMVTeleconf/'

# Load in control data for 50N 30W
#fullauto =np.load(datpath+"Autocorrelation_30W50N_FULL_PIC_12805.npy",allow_pickle=True)
fullauto = np.load(datpath+"FULL_PIC_autocorr_lon330_lat50_lags0to36_month2.npy")

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
labels=["MLD Fixed","MLD Mean","MLD Seasonal","MLD Entrain"]
#labels=["MLD (MAX)","MLD Seasonal","MLD Entrain"]
#colors=["red","orange","magenta","blue"]
expcolors = ('blue','orange','magenta','red')
hblt = 54.61088498433431 # Meters, the mixed layer depth used in CESM Slab

config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0
config['fstd']        = 1
config['t_end']       = 120000    # Number of months in simulation
config['runid']       = "syn001"  # White Noise ID
config['fname']       = "FLXSTD" #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1
config['query']       = [-30,50]
config['applyfac']    = 2 # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = projpath + '02_Figures/20210223/'
config['smooth_forcing'] = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)
#%% Functions

def load_data(mconfig,ftype,projpath=None):
    
    """
    Inputs
    ------
    mconfig : STR
        Model Configuration (SLAB_PIC or FULL_HTR)
    ftype : STR
        Forcing Type ('DJFM-MON' or ... )
    projpath : STR (optional)
        Path to project folder (default uses path on laptop)
    
    Outputs
    -------
    mld : ARRAY 
        Monhtly Mean Mixed Layer Depths
    kprevall : ARRAY
        Detrainment Months
    lon : ARRAY
        Longitudes (-180 to 180)
    lat : ARRAY
        Latitudes
    lon360 : ARRAY
        Longitudes (0 to 360)
    cesmslabac : ARRAY
        Autocorrelation at each point in the CESM Slab
    damping : ARRAY
        Monthly ensemble mean Heat flux feedback
    forcing : ARRAY
        Monthly forcing at each point (NAO, EAP, EOF3)
    """
    
    # Set Paths
    if projpath is None:
        projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/'
    input_path  = datpath + 'model_input/'
    
    # Load Data (MLD and kprev) [lon x lat x month]
    if mconfig == "FULL_HTR": # Load ensemble mean historical MLDs
        mld            = np.load(input_path+"%s_HMXL_hclim.npy"% mconfig) # Climatological MLD
        kprevall       = np.load(input_path+"%s_HMXL_kprev.npy"% mconfig) # Entraining Month
    else: # Load PIC MLDs 
        mld            = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
        kprevall       = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month
    
    mld1kmean      = np.load(input_path+"FULL_PIC_HMXL_hclim_400to1400mean.npy") # Entraining Month
    
    # Load Lat/Lon, autocorrelation
    dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
    loaddamp       = loadmat(input_path+dampmat)
    lon            = np.squeeze(loaddamp['LON1'])
    lat            = np.squeeze(loaddamp['LAT'])
    cesmslabac     = np.load(datpath+"CESM_clim/TS_SLAB_Autocorrelation.npy") #[mon x lag x lat x lon]
    lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()
    
    # Load damping [lon x lat x mon]
    if mconfig == "SLAB_PIC":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")
    elif mconfig=="FULL_HTR":
        damping = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig020_dof082_mode4.npy")
    
    # Load Forcing  [lon x lat x pc x month]
    forcing = np.load(input_path+mconfig+ "_NAO_EAP_NHFLX_Forcing_%s.npy" % ftype)#[:,:,0,:]
    
    return mld,kprevall,lon,lat,lon360,cesmslabac,damping,forcing,mld1kmean

def synth_stochmod(config,verbose=False,viz=False,
                   dt=3600*24*30,rho=1026,cp0=3996,hfix=50,T0=0):
    """
    Parameters
    ----------
    config : DICT
        'mconfig'
        'ftype'
        'genrand'
        'fstd'
        't_end'
        'runid'
        'fname'
    dt : INT (optional)
        Timestep in seconds (monthly timestep by default)
        
'output_path'        

    Returns
    -------
    None.
    """
    # Load data
    # ---------
    mld,kprevall,lon,lat,lon360,cesmslabac,damping,forcing,mld1kmean = load_data(config['mconfig'],config['ftype'])
    hblt  = np.load(datpath+"SLAB_PIC_hblt.npy")
    
    if verbose:
        print("Loaded Data")
    
    # Generate Random Forcing
    # -----------------------
    if config['genrand']:
        randts = np.random.normal(0,config['fstd'],config['t_end'])
        np.save(config['output_path'] + "Forcing_fstd%.2f_%s.npy"% (config['fstd'],config['runid']),randts)
        if verbose:
            print("Generating New Forcing")
    else:
        randts = np.load(output_path+"Forcing_fstd%.2f_%s.npy"% (config['fstd'],config['runid']))
        if verbose:
            print("Loading Old Forcing")
    
    # Select Forcing [lon x lat x mon]
    if config['fname'] == 'NAO':
        forcing = forcing[:,:,0,:]
    elif config['fname'] == 'EAP':
        forcing = forcing[:,:,1,:]
    elif config['fname'] == 'EOF3':
        forcing = forcing[:,:,2,:]
    elif config['fname'] == 'FLXSTD':
        forcing = np.load("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/SLAB_PIC_NHFLXSTD_Forcing_MON.npy")
    
    # Restrict input parameters to point (or regional average)
    params = scm.get_data(config['pointmode'],config['query'],lat,lon,
                          damping,mld,kprevall,forcing)
    [o,a],damppt,mldpt,kprev,Fpt = params
    kmonth = mldpt.argmax()
    print("Restricted Parameters to Point. Kmonth is %i"%kmonth)
    
    # Apply 3 month smoothing if option is set
    if config['smooth_forcing'] == True:
        Fpt = np.convolve(np.hstack([Fpt[-1],Fpt,Fpt[0]]),np.ones(3)/3,mode='valid')
        #params[4] = Fpt
    
    # Check for synthetic points, and assign to variable if it exists
    synthflag = []
    if 'mldpt' in config:
        mldpt = config['mldpt']
        synthflag.append('mld')
    if 'Fpt' in config:
        Fpt = config['Fpt']
        synthflag.append('forcing')
    if 'damppt' in config:
        damppt = config['damppt']
        synthflag.append('damping')
    if verbose:
        print("Detected synthetic forcings for %s"%str(synthflag))
    
    if viz:
        synth = [damppt,mldpt,Fpt]
        fig,ax = viz.summarize_params(lat,lon,params,synth=synth)
    
    # Prepare forcing
    mldmean = hblt[o,a,:].mean()
    Fh = {}
    nyrs = int(config['t_end']/12)
    
    if config['applyfac'] in [0,3]: # White Noise Forcing, unscaled by MLD
        for h in range(3):
            Fh[h] = randts * np.tile(Fpt,nyrs)
    else: # White Noise Forcing + MLD
        for h in range(3):
            if h == 0: # Fixed 50 meter MLD
                Fh[h] = randts * np.tile(Fpt,nyrs) * (dt/(cp0*rho*hfix))
            elif h == 1: # Seasonal Mean MLD
                Fh[h] = randts * np.tile(Fpt,nyrs) * (dt/(cp0*rho*mldmean))
            elif h == 2: # Seasonall Varying mean MLD
                Fh[h] = randts * np.tile(Fpt/mldpt,nyrs) * (dt/(cp0*rho))
    
    # Convert Parameters
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(mldpt,damppt,dt,ND=False,hfix=hfix)
    if verbose:
        print("Completed parameter setup!")
    
    # Run the stochastic model
    multFAC = 0
    if config['applyfac'] > 1: # Apply Integration factor
        multFAC = 1
    
    
    sst         = {}
    dampingterm = {}
    forcingterm = {}
    for i in range(3): # No Entrainment Cases
        sst[i],forcingterm[i],dampingterm[i] = scm.noentrain(config['t_end'],lbd[i],T0,Fh[i],FAC[i],multFAC=multFAC,debug=True)
    
    sst[3],dampingterm[3],forcingterm[3],entrainterm,Td=scm.entrain(config['t_end'],
                       lbd[3],T0,Fh[2],
                       beta,mldpt,kprev,
                       FAC[3],multFAC=multFAC,
                       debug=True,debugprint=False)
    if verbose:
        print("Model Runs Complete!")
    
    # Reassign Params
    params = ([o,a],damppt,mldpt,kprev,Fpt)
    
    
    ## Basically no effect, so commented out..
    # #Detrend again to be sure
    # for i in range(4):
    #     sst[i] = signal.detrend(sst[i],type='linear')
    
    # Calculate Autocorrelation
    autocorr = scm.calc_autocorr(sst,config['lags'],kmonth+1)
    if verbose:
        print("Autocorrelation Calculations Complete!")
    return autocorr,sst,dampingterm,forcingterm,entrainterm,Td,kmonth,params


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
ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
[o,a],damppt,mldpt,kprev,Fpt = params

# Read in CESM autocorrelation for all points'
kmonth = np.argmax(mldpt)
print("Kmonth is %i"%kmonth)
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = load_data(mconfig,ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]

# Plot some differences
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=params[2],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')

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

#
#%% Run the Experiment
#


expids   = [] # String Indicating the Variable Type
acs      = []
ssts     = []
damps    = []
mlds     = []
forces   = []
explongs = []

for vmld in [False,True]:
    
    if vmld:
        config['mldpt'] = mlddef
    else:
        config['mldpt'] = np.ones(12)*mlddef.mean()

    for vdamp in tqdm([False,True]):
        
        if vdamp:
            config['damppt'] = dampdef
        else:
            config['damppt'] = np.ones(12)*dampdef.mean()
            
        
        for vforce in [False,True]:
            
                if vforce:
                    config['Fpt'] = Fptdef
                else:
                    config['Fpt'] = np.ones(12)*Fptdef.mean()
                
                # Set experiment name
                expid = "vdamp%i_vforce%i_vmld%i" % (vdamp,vforce,vmld)
                explong = "Vary Damping (%s) Forcing (%s) MLD (%s)" % (vdamp,vforce,vmld)
                
                
                # Run Model
                ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
                [o,a],damppt,mldpt,kprev,Fpt = params
                
                # Save variables
                acs.append(ac)
                ssts.append(ssts)
                expids.append(expid)
                damps.append(damppt)
                mlds.append(mldpt)
                forces.append(Fpt)
                explongs.append(explong)
                print("Completed %s"%expid)
                print(mldpt)
                
                # Clean Config, End Forcing Loop
                config.pop('Fpt',None)
                print(Fpt)
                
        # Clean Config, End Damping Loop
        config.pop('damppt',None)
        print(damppt)
    # Clean Config, End MLD Loop
    config.pop('mldpt',None)
    
#%%
# Calculate Confidence Levels
n     = 10000
conf  = 0.95
tails = 2
nexp = len(expids)
nlags = len(lags)
confs = np.zeros((nexp,4,nlags,2))
for i in tqdm(range(8)): # Loop for each experiment
    
    for m in range(4): # Loop for each model

        for l,lag in enumerate(lags): # Loop for each lag
            acin = acs[i][m][l]
            cfout = proc.calc_pearsonconf(acin,conf,tails,n)
            
            confs[i,m,l,:] = cfout


def calc_conflag(ac,conf,tails,n):
    cflags = np.zeros((len(ac),2))
    for l in range(len(ac)):
        rhoin = ac[l]
        cfout = proc.calc_pearsonconf(rhoin,conf,tails,n)
        cflags[l,:] = cfout
    return cflags


cfslab = calc_conflag(cesmauto2,conf,tails,898)
cffull = calc_conflag(fullauto,conf,tails,1798)
#for l,lag in enumerate(lags)
    
        
        


#%% Plot Model Results, All Together
model  = 2
plotac = acs
# Plot some differences
xtk2       = np.arange(0,37,2)
fig,ax     = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')

for i in range(len(expids)):
    ax.plot(lags,plotac[i][model],label=expids[i])

# ax.legend()
# ax3.set_ylabel("Mixed Layer Depth (m)")
# ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
#plt.savefig(outpath+"Default_Autocorrelation.png",dpi=200)

#%% Plot 4x4 with constant mixed layer depth
plotacs = acs
model   = 1

figs,axs = plt.subplots(2,2,figsize=(8,8))
for i,e in enumerate([0,1,2,3]):
    ax     = axs.ravel()[i]
    #title  = explongs[e]
    title=""
    xtk2       = np.arange(0,37,3)
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
    
    ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
    ax.scatter(lags,cesmauto2[lags],10,label="",color='k')
    ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)
    ax.set_ylabel("")
    
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label="Stochastic Model",color='b')
    ax.scatter(lags,plotacs[e][model],10,label="",color='b')
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color='b',alpha=0.4)
    ax.legend()
    if i == 0:
        ax.set_ylabel("Constant Damping",fontsize=14)
        ax.set_title("Constant Forcing",fontsize=14)
    if i == 1:
        ax.set_title("Variable Forcing",fontsize=14)
    if i == 2:
        ax.set_ylabel("Variable Damping",fontsize=14)
plt.suptitle("Stochastic Model (No Entrainment), Constant MLD",fontsize=14)
plt.tight_layout()
plt.savefig("%sAutocorrelation_ConstvVary_MLDConst.png"%outpath,dpi=150)


#%% Plot 4x4 with variable MLD
plotacs = acs
model   = 2

figs,axs = plt.subplots(2,2,figsize=(8,8))
for i,e in enumerate([4,5,6,7]):
    ax     = axs.ravel()[i]
    #title  = explongs[e]
    title=""
    xtk2       = np.arange(0,37,3)
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
    
    ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
    ax.scatter(lags,cesmauto2[lags],10,label="",color='k')
    ax.fill_between(lags,cfslab[lags,0],cfslab[lags,1],color='gray',alpha=0.4)
    ax.set_ylabel("")
    
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label="Stochastic Model",color='m')
    ax.scatter(lags,plotacs[e][model],10,label="",color='m')
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color='m',alpha=0.4)
    ax.legend()
    if i == 0:
        ax.set_ylabel("Constant Damping",fontsize=14)
        ax.set_title("Constant Forcing",fontsize=14)
    if i == 1:
        ax.set_title("Variable Forcing",fontsize=14)
    if i == 2:
        ax.set_ylabel("Variable Damping",fontsize=14)
plt.suptitle("Stochastic Model (No Entrainment), Varying MLD",fontsize=14)
plt.tight_layout()
plt.savefig("%sAutocorrelation_ConstvVary_MLDVary.png"%outpath,dpi=150)



#%% Plot 4x4 with ENTRAINMENT
plotacs = acs
model   = 3

figs,axs = plt.subplots(2,2,figsize=(8,8))
for i,e in enumerate([4,5,6,7]):
    ax     = axs.ravel()[i]
    #title  = explongs[e]
    title=""
    xtk2       = np.arange(0,37,3)
    ax,ax2 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,title=title)
    
    ax.plot(lags,fullauto[lags],label="CESM FULL",color='k',ls='dashdot')
    ax.scatter(lags,fullauto[lags],10,label="",color='k',ls='dashdot')
    ax.fill_between(lags,cffull[lags,0],cffull[lags,1],color='gray',alpha=0.4)
    ax.set_ylabel("")
    
    cfs = confs[e,model,:,:]
    ax.plot(lags,plotacs[e][model],label="Stochastic Model",color='r')
    ax.scatter(lags,plotacs[e][model],10,label="",color='r')
    ax.fill_between(lags,cfs[:,0],cfs[:,1],color='r',alpha=0.4)
    ax.legend()
    if i == 0:
        ax.set_ylabel("Constant Damping",fontsize=14)
        ax.set_title("Constant Forcing",fontsize=14)
    if i == 1:
        ax.set_title("Variable Forcing",fontsize=14)
    if i == 2:
        ax.set_ylabel("Variable Damping",fontsize=14)
plt.suptitle("Stochastic Model (with Entrainment)",fontsize=14)
plt.tight_layout()
plt.savefig("%sAutocorrelation_ConstvVary_Entrain.png"%outpath,dpi=150)
    