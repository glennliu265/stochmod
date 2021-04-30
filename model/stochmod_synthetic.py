#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,Test Synthetic Stochastic Model

Created on Tue Jan 12 03:46:45 2021

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

#%% Set Options
#bboxsim  = [-100,20,-20,90] # Simulation Box


#pcnames = ["NAO","EAP","NAO+EAP"]
#exps = 

#config['mldpt']
#config['Fpt']
#config['damppt']

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
output_path = datpath + 'model_output/'
outpath = projpath + '02_Figures/20210316/'

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


#%%


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
#%% Clean Run

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
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=params[4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')

ax.plot(ac[1],label="Qnet Forcing",color='orange')
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"Default_Autocorrelation.png",dpi=200)




# --------------------------------------------------
#%% Quick comparison plot of the autocorrelation
# -------------------------------------------------
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
plt.savefig(outpath+"Compare_Autocorrelation_CESM.png",dpi=200)


#%% Quick plot of the output

fig,ax=plt.subplots(1,1,figsize=(8,3))

for i in [1,2,3]:
    
    sstann = proc.ann_avg(sst[i],0)
    
    win = np.ones(12)/12
    sstann = np.convolve(sstann,win,mode='valid')
    
    yrs = np.arange(0,sstann.shape[0])
    
    plabel = labels[i] + r", 1$\sigma=%.2f$" % np.std(sst[i])
    ax.plot(sstann,label=plabel,lw=0.5,color=expcolors[i])
    
    print("Std for %s is %.2f"%(labels[i],np.std(sst[i])))
ax.legend(fontsize=8,ncol=3)
ax.set_xlabel("Years")
ax.set_ylabel("degC")
ax.set_title("Stochastic Model SST (12-year Running Mean)")
plt.tight_layout()
plt.savefig("%sStochasticModelSST_comparison.png"%(outpath),dpi=150)
# -------------------------
#%% # Run some experiments
# -------------------------

testparam  = "smooth_forcing"
testvalues = [False,True]
testcolors = ['b','orange']

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    st = time.time()
    config[testparam] = val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[1][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    ax.plot(acall[i][1],label="%s = %s" % (testparam,val),color=testcolors[i])
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_test%s.png"%(locstring,testparam),dpi=200)

# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
ax.plot(mons3,paramsall[0][4],label="Forcing (Unsmoothed)",color='b')
ax.plot(mons3,paramsall[1][4],label="Forcing (Smoothed)",color='orange')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Forcing (W/m2)")
plt.savefig(outpath+"Forcing_Differences_%s_test%s.png"%(locstring,testparam),dpi=200)
config['smooth_forcing'] = False

# -------------------------
# %% Try different damping
# -------------------------

# Load data from prep_HF in hfdamping module
from scipy.io import loadmat
mt1 = loadmat(datpath+'model_input/'+"SLAB_PIC_nhflxdamping_monwin3_sig005_dof020_mode4_lag1.mat")
lon1 = mt1['LON1'][0]
lat  = mt1['LAT'][0]
dp1  = mt1['damping']
mt1 = loadmat(datpath+'model_input/'+"SLAB_PIC_nhflxdamping_monwin3_sig005_dof020_mode4_lag123.mat")
dp123 = mt1['damping']

lonf,latf = config['query']
ko1,ka = proc.find_latlon(lonf,latf,lon1,lat)

testparam  = "damppt"
testvalues = [damppt,dp1[ko1,ka],dp123[ko1,ka]]
testnames  = ['Lags 1-2','Lags 1',"Lags 1-3"]
testcolors = ['b','orange','magenta']


acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    st = time.time()
    config[testparam] = val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[1][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    ax.plot(acall[i][1],label="%s" % (testnames[i]),color=testcolors[i])
ax3.set_ylabel("Dampiing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_test%s_dampingtest.png"%(locstring,testparam),dpi=200)

# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
for i in range(3):
    ax.plot(mons3,paramsall[i][1],label=testnames[i],color=testcolors[i])
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Damping (W/m2)")
plt.savefig(outpath+"Damping_Differences_%s_test%s.png"%(locstring,testparam),dpi=200)
config['damppt'] = False
#plt.savefig(outpath+"Dampingvalues_WithSEAS_MLD_%s_test%s_dampingtest.png"%(locstring,testparam),dpi=200)

#%% Plot Autocorrelation (All Models)

title = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])

xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i in [0,1]:
    ax.plot(ac[i],label=labels[i],color=colors[i])
#ax.legend(ncol=3,fontsize=10)
#ax3.set_ylabel("MLD (m)")
#ax3.set_ylabel("Damping (W/m2)")
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s.png"%(locstring),dpi=200)

#%% Plot autocorrelation of different terms for a particular model

i = 1 # Select the model

# First, calculate the autocorrelation
vnames = ["SST","Damping Term","Forcing Term","Entrain Term"]
vcalc = [sst[i],dmp[i],frc[i]]
vac = []
for v in vcalc:
    vac.append(scm.calc_autocorr([v],config['lags'],kmonth+1))
    


fig,ax = plt.subplots(1,1)
title="Autocorrelation by Term for %s model \n %s" % (labels[i],locstringtitle,)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')

for i,v in enumerate(vac):
    ax.plot(lags,v[0],label=vnames[i])
ax.legend()
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_byTerm_MLD_%s.png"%(locstring),dpi=200)

#%% Try Sliding Autocorrelation Around to find the best match

sstmod = sst[1]
cesmac  = cesmslabac[:,lags,ka,ko]
stochac = np.zeros(cesmac.shape)
for km in range(12):
    stochac[km,:] = scm.calc_autocorr([sstmod],lags,km+1)[0]

fig,axs = plt.subplots(3,4,sharey=True,figsize=(8,6))
xtksm = np.arange(0,37,12)
for km in range(12):
    ax = axs.flatten()[km]
    
    ax.plot(lags,cesmac[km,:],color='k',ls='dashed',lw=0.75)
    ax.plot(lags,stochac[km,:],color='orange',lw=0.75)
    ax.set_title("lag 0 = " + mons3[km])
    ax.set_xticks(xtksm)
    ax.set_yticks(np.arange(0,1.5,.5))
    ax.grid(True,ls='dotted')
plt.tight_layout()
plt.savefig(outpath+"Testing_Lag0_Slabmodel_Finstantaneous_%s.png"%locstring,dpi=200)
    

# ---------------------------
#%% Try different damping (4x)
# ---------------------------
# custdamp = np.array([22.55689888, 14.7803278 , 11.66007557,  9.64517097*1.5,  9.59150019,
#         7.84583902,  7.60521469, 10.24848289, 16.25020237, 22.80579529*1.5,
#        27.11528091*2, 26.625214  ])
config['damppt']  = custdamp
config['genrand'] = 0
st = time.time()
ac1,sst1,dmp1,frc1,ent1,Td1,kmonth1,params1=synth_stochmod(config)
print("Ran script in %.2fs"%(time.time()-st))

fig,ax = plt.subplots(1,1)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=custdamp,title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i in [1]:
    ax.plot(ac[i],label=labels[i]  + "Original",color=colors[i])
    ax.plot(ac1[i],label=labels[i] + " Custom Damping",color='blue',ls='dashed')

ax3.set_ylabel("Damping (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_DampingTest_%s.png"%(locstring),dpi=200)

config.pop('damppt',None)
# ---------------------------
#%% FORCING MODIFICATION Experiments
# ---------------------------

Fori  = np.array([53.36403275, 50.47200521, 43.19549306, 32.95324516, 26.30336189,
           22.53761546, 22.93124771, 26.54155223, 32.79647001, 39.71981049,
           45.65141678, 50.43875758])
Ftest = np.ones(12)*53.36403275
#Ftest = np.array([53.36403275, 53.36403275*2, 43.19549306, 32.95324516, 26.30336189,
           # 22.53761546, 22.93124771, 26.54155223, 32.79647001, 39.71981049,
           # 45.65141678, 50.43875758])

testparam  = "Fpt"
testvalues = [Fori,Ftest]
testnames = ['Original',"Sustained"]
testcolors = ['b','orange']

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    st = time.time()
    config[testparam] = val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))
    
#%

# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
ax.plot(mons3,paramsall[0][4],label="Forcing (Unsmoothed)",color='b')
ax.plot(mons3,paramsall[1][4],label="Forcing (Smoothed)",color='orange')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Forcing (W/m2)")
plt.savefig(outpath+"Forcing_Differences_%s_test%s.png"%(locstring,testparam),dpi=200)



nyr = int(sstall[0][1].shape[0]/12)
fig,ax = plt.subplots(1,1)
ax.plot(mons3,sstall[1][1].reshape(nyr,12).T,label="Original",color=[.5,.5,1],alpha=0.1)
ax.plot(mons3,sstall[1][1].reshape(nyr,12).std(0),label='stdev',color='b',ls='dashed')
#ax.plot(sstall[1][1],label="Test",color='orange')
ax.set_xlim([0,100])

# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[1][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    lab = "%s = %s" % (testparam,testnames[i])
    #lab = ""
    ax.plot(acall[i][1],label=lab,color=testcolors[i])
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_test%s.png"%(locstring,testparam),dpi=200)

# ---------------------------
#%% Try forcing interpolation
# ---------------------------

Fquad = interp_quad(Fpt)
dampquad = interp_quad(damppt)
testparam  = "Fpt"
testvalues = [Fpt,Fquad]
testcolors = ['b','orange']
valnames   = ["Original Forcing","Shifted Forcing"]

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    st = time.time()
    config[testparam] = val
    if i == 0:
        config.pop('damppt',None)
    else:
        config['damppt'] = dampquad
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))
    
#%
# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[0][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    ax.plot(acall[i][1],label="%s = %s" % (testparam,valnames[i]),color=testcolors[i])
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_test%s.png"%(locstring,testparam),dpi=200)



# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
ax.plot(mons3,paramsall[0][4],label="%s %s"% (testparam,valnames[0]),color='b')
ax.plot(mons3,paramsall[1][4],label="%s %s"% (testparam,valnames[1]),color='orange')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Forcing (W/m2)")
plt.savefig(outpath+"%s_Differences_%s_test%s.png"%(testparam,locstring,testparam),dpi=200)

config.pop('damppt',None)
config.pop('Fpt',None)

# ---------------------------
#%% Update Plot for stohastic model (compare sinusoidal vs updated version)
# ---------------------------
xtks = np.arange(1,13,1)
Fpt1 = np.sin(-1*np.pi*xtks/6+10)*0.3*20*-1
plt.plot(mons3,Fpt1)


Fquad = interp_quad(Fpt)

testparam  = "Fpt"
expname    = "SinusoidvsActual"
testvalues = [Fpt1,Fpt]
testcolors = ['b','orange']
valnames   = ["Sinusoidal Forcing","Qnet Forcing"]

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    config['damppt']   = np.ones(12)*13
    config['genrand']  = 0
    config['applyfac'] = 2
    st = time.time()
    config[testparam] = val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))
    
#%



# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[0][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    ax.plot(acall[i][1],label="%s = %s" % (testparam,valnames[i]),color=testcolors[i])
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"%s_AC_WithSEAS_MLD_%s_test%s.png"%(expname,locstring,testparam),dpi=200)



# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
ax.plot(mons3,paramsall[0][4],label="%s %s"% (testparam,valnames[0]),color='b')
ax.plot(mons3,paramsall[1][4],label="%s %s"% (testparam,valnames[1]),color='orange')
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Forcing (W/m2)")
plt.savefig(outpath+"%s_%s_Differences_%s_test%s.png"%(expname,testparam,locstring,testparam),dpi=200)

# -------------------------
# %% Try Fixed and NonFixed Damping
# -------------------------

testparam  = "damppt"
testvalues = [damppt,np.ones(12)*damppt.mean()]
testnames  = ['DampingVar',"DampingFix"]
testcolors = ['b','orange']



acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in enumerate(testvalues):
    st = time.time()
    config[testparam] = val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))


# Run Additional experiment with fixed damping AND forcing
config['Fpt'] = np.ones(12)*Fpt.mean()
ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
sstall.append(sst)


# Plot some differences
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
title      = "SST Autocorrelation at %s (Lag 0 = %s)" % (locstringtitle,mons3[mldpt.argmax()])
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=paramsall[1][4],title=title)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
for i,val in enumerate(testvalues):
    ax.plot(acall[i][1],label="%s" % (testnames[i]),color=testcolors[i])
ax3.set_ylabel("Forcing (W/m2)")
ax3.yaxis.label.set_color('gray')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_WithSEAS_MLD_%s_test%s_dampingtest.png"%(locstring,testparam),dpi=200)

# Plot Differences in parameter
fig,ax = plt.subplots(1,1)
for i in range(2):
    ax.plot(mons3,paramsall[i][1],label=testnames[i],color=testcolors[i])
ax.legend()
ax.grid(True,ls='dotted')
ax.set_ylabel("Damping (W/m2)")
config['damppt'] = False

dampvarsst = sstall.copy()

#%% Plot Autocorrelatiom (Just Slab)

labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,figsize=(6,4))
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
#ax.plot(lags,autocorr[1],label=labels[i])
#ax.plot(lags,exps[0],label="Sinusoidal Forcing",color='b',ls='solid')
ax.plot(lags,autocorr[4],label="No Seasonal Cycle",color='k',ls='dotted')
#ax.plot(lags,exps[1],label="NAO Forcing",color='magenta',ls='solid')
ax3.set_ylabel("Forcing (W/m2)")
ax.plot(lags,autocorr[1],label="NAO Forcing, Mean MLD",color='magenta')

ax.plot(lags,exps[0],label="Sinusoidal Forcing, Max MLD",color='gold')
#ax.plot(lags,autocorr[1],label="Sin. Forcing + Seasonal Damping",color='r')
ax3.yaxis.label.set_color('gray')
#ax3.set_ylim([-30,30])
#ax3.axhline(y=0,color='gray',linestyle="dotted",lw=0.75)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(outpath+"AC_Slab_Stoch_Comparison_Sinusoid_MaxMLD.png",dpi=200)

#%% Init exp parameter
exps = {}
expnames = ["Sinusoid","NAO"]
expcolors = ['b']

expf = {}
#%% Save Experiment
exps[0] = autocorr[1].copy()
expf[0] = Fpt.copy()

np.save(outpath+"Exps_saved.npy",exps)



#%% Just Plot a few autocorrelation curves

queries = [[-30,50],[-70,30],[]]
acplots = []
for q in queries:
    kmonth = 
    ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
    acplots.append()


ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]


# **********************************
#%% Experiment Set 1 Varying Forcing (Month of Maximum)
# **********************************

# Seasonally varying forcing, shifting around
Fs   = []
mmax = []
ploti    = 1
for m in tqdm(range(12)):
    # Make Forcing
    Fpt = np.sin(-1*np.pi*(xtks)/6+m)*0.3*20*-1
    #Fpt = np.sin(-1*np.pi*(xtks)/6+m)+1
    Fs.append(Fpt)
    mmax.append(Fpt.argmax()+1)
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    
    # Run the stochastic model
    sst = {}
    for i in range(3):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    else:
        sst[3]=scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False)
    
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)
    
    # Make Plot
        # Plot Autocorrelation
    #labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
    xtk2 = np.arange(0,37,2)
    fig,ax = plt.subplots(1,1)
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
    ax.plot(lags,cesmauto,label="CESM SLAB",color='k')
    
    ax.plot(autocorr[ploti],label=labels[ploti])
    
    ax.legend(ncol=3,fontsize=10)
    ax3.set_ylabel("Forcing")
    ax3.yaxis.label.set_color('gray')
    plt.suptitle("MLD Max SST Autocorrelation, Forcing Max = %s"% (mons3[mmax[m]-1]))
    plt.tight_layout()
    plt.savefig("%sSeasonal_Forcing_mmax%i_model%i.png"%(outpath,mmax[m],ploti),dpi=200)
    
    
# **********************************
#%% Experiment Set 2 Varying Forcing (Magnitude)
# **********************************


F = np.ones(mld.shape) * np.sin(-1*np.pi*xtks/6+10)[None,None,:]
Fpt = []
Ffac = [10**x for x in np.linspace(-6,6,12)]
acall = []
for m in tqdm(range(12)):
    # Make Forcing
    Fpt = Ffac[m]*np.sin(-1*np.pi*xtks/6+10)
    Fs.append(Fpt)
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    
    # Run the stochastic model
    sst = {}
    for i in range(3):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    else:
        sst[3]=scm.entrain(t_end,lbd[3],T0,Fh,beta,hclim,kprev,FAC[3],multFAC=multFAC,debug=False)
    
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)
    acall.append(autocorr)



    
# Make Plot
# Plot Autocorrelation
#labels=["MLD Fixed","MLD Max","MLD Seasonal","MLD Entrain"]
xtk2 = np.arange(0,37,2)
fig,ax = plt.subplots(1,1)
ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
ax.plot(lags,cesmauto,label="CESM SLAB",color='k')
for i in range(12):
    ax.plot(acall[i][1],label=Ffac[i])
ax.legend(ncol=1,fontsize=10)

ax3.set_ylabel("Forcing")
ax3.yaxis.label.set_color('gray')
plt.suptitle("MLD Max SST Autocorrelation, Forcing Max = %s"% (mons3[mmax[m]-1]))
plt.tight_layout()
plt.savefig("%sSeasonal_Forcing_mmax%i_model%i.png"%(outpath,mmax[m],ploti),dpi=200)


#
# Quantifying sensitivity to forcing...
#


#%%

for mm in tqdm(range(12)):
    Fpt = np.sin(-1*np.pi*xtks/6+mm)*0.3*20*-1
    Fh     = randts * np.tile(Fpt,int(t_end/12))
    sst = {}
    for i in range(2):
        sst[i] = scm.noentrain(t_end,lbd[i],T0,Fh,FAC[i],multFAC=multFAC,debug=False)
    # Calculate Autocorrelation
    kmonth = hclim.argmax()
    autocorr = scm.calc_autocorr(sst,lags,kmonth+1)

    xtk2 = np.arange(0,37,2)
    fig,ax = plt.subplots(1,figsize=(6,4))
    ax,ax2,ax3 = viz.init_acplot(kmonth,xtk2,lags,ax=ax,loopvar=Fpt)
    ax.plot(lags,cesmauto2[lags],label="CESM SLAB",color='k')
    #ax.plot(lags,autocorr[1],label=labels[i])
    ax.plot(lags,autocorr[1],label="Sinusoidal Forcing",color='b',ls='solid')
    #ax.plot(lags,autocorr[4],label="No Seasonal Cycle",color='k',ls='dotted')
    ax3.set_ylabel("Forcing (W/m2)")
    ax3.yaxis.label.set_color('gray')
    ax3.set_ylim([-12,12])
    ax3.axhline(y=0,color='gray',linestyle="dotted",lw=0.75)
    plt.suptitle("mm=%i"%mm)
    ax.legend()
    plt.savefig(outpath+"Sinusoida_Forcing_Slab_mm%i.png"%(mm),dpi=200)
    
# # **********************************
# #%% Grid Sweep Experiments : Damping
# # **********************************
# testvalues = np.arange(1,26,1)
# testparam  = 'damppt'

# acall     = []
# sstall    = []
# kmonthall = []
# paramsall = []

# for i,val in tqdm(enumerate(testvalues)):
#     #config['Fpt'] = np.ones(12)*37.24208402633666
#     st = time.time()
#     config[testparam] = np.ones(12)*val
#     ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
#     acall.append(ac)
#     sstall.append(sst)
#     kmonthall.append(kmonth)
#     paramsall.append(params)
    
#     print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# #%% Visualize the results
# import cmocean
# model = 1
# expname="DampingCVary"
# nlag = len(config['lags'])
# nexp = len(testvalues)

# acalls = np.zeros((nexp,nlag))
# for e,ac in enumerate(acall):
#     acalls[e,:] = acall[e][model]


# # Pcolor Plot
# fig,ax = plt.subplots(1,1)
# pcm = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
# fig.colorbar(pcm,ax=ax)
# ax.set_ylabel("Damping (W/m2)")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Damping_pcolor_%s.png"%(outpath,expname),dpi=200)


# # Pcolor Plot, differences
# fig,ax = plt.subplots(1,1)
# pcm = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
# fig.colorbar(pcm,ax=ax)
# ax.set_ylabel("Damping (W/m2)")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
# plt.tight_layout()
# plt.savefig("%sLag_v_Damping_pcolor_diff_%s.png"%(outpath,expname),dpi=200)

# # Plot minimum RMSE
# rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
# print("Minumum RMSE was for %i with value %f" % (np.argmin(rmses),rmses.min()))


# # Line Plots
# fig,ax = plt.subplots(1,1)
# ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
# for lam in range(25):
#     ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/25)*.5,color='b')
    
# ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)"%testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Correlation")
# ax.set_xlabel("Lag (months)")   
# ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
# ax.grid(True,ls='dotted')
# plt.savefig("%sLag_v_Damping_lineplot_%s.png"%(outpath,expname),dpi=200)

# XX,YY = np.meshgrid(config['lags'],testvalues[1:])
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
# surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma')
# fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
# ax.set_ylim(25,0)
# ax.set_xlim(0,38)
# ax.set_zlim(0,1)
# ax.set_xticks(np.arange(0,37,6))
# ax.set_ylabel("Damping $(W/m^{2})$")
# ax.set_xlabel("Lag (months)")
# ax.set_zlabel("Correlation")
# ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
# plt.tight_layout()
# plt.savefig("%sLag_v_Damping_3Dplot_%s.png"%(outpath,expname),dpi=200)

# # *********************************************************
# #%% Grid Sweep Experiments II : Damping, Seasonal Magnitude
# # *********************************************************
# testvalues = np.arange(0.1,2.1,.1)#[0.25,0.5,1,2,4,8,16,32,64,128]

# testparam  = 'damppt'

# acall     = []
# sstall    = []
# kmonthall = []
# paramsall = []

# for i,val in tqdm(enumerate(testvalues)):
#     st = time.time()
#     config[testparam] = damppt*val
#     ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
#     acall.append(ac)
#     sstall.append(sst)
#     kmonthall.append(kmonth)
#     paramsall.append(params)
    
#     print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# #%% Visualize the results

# model = 1
# expname="DampingVVary"
# nlag = len(config['lags'])
# nexp = len(testvalues)

# acalls = np.zeros((nexp,nlag))
# for e,ac in enumerate(acall):
#     acalls[e,:] = acall[e][model]

# #ytk=np.arange(0,len(testvalues),1)
# ytk = np.arange(.1,2.2,.2)

# # Pcolor Plot
# fig,ax = plt.subplots(1,1,figsize=(8,8))
# im = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
# #im = ax.imshow(acalls,cmap='magma',vmin=0,vmax=1)
# fig.colorbar(im,ax=ax,fraction=0.015)
# #ax.set_yticklabels(ytk)
# #ax.set_yticklabels(testvalues,fontsize=10)
# #plt.gca().invert_yaxis()
# ax.set_ylabel("Damping Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Damping_pcolor_%s.png"%(outpath,expname),dpi=200)

# # Pcolor Plot, differences
# fig,ax = plt.subplots(1,1,figsize=(8,4))
# im = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
# #im = ax.imshow(acalls-cesmauto,cmap=cmocean.cm.balance,vmin=-.5,vmax=.5)
# fig.colorbar(im,ax=ax,fraction=0.015)
# #ax.set_yticks(ytk)
# #ax.set_yticklabels(testvalues,fontsize=10)
# #plt.gca().invert_yaxis()
# ax.set_ylabel("Damping Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Damping_pcolordiff_%s.png"%(outpath,expname),dpi=200)

# # Plot minimum RMSE
# rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
# print("Minumum RMSE was for %i with value %f" % (testvalues[np.argmin(rmses)],rmses.min()))


# # Line Plots
# fig,ax = plt.subplots(1,1)
# ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
# for lam in range(len(testvalues)):
#     ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/len(testvalues))*.5,color='b')
    
# ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)" % testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Correlation")
# ax.set_xlabel("Lag (months)")   
# ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
# ax.grid(True,ls='dotted')
# plt.savefig("%sLag_v_Damping_lineplot_%s.png"%(outpath,expname),dpi=200)

# XX,YY = np.meshgrid(config['lags'],testvalues[1:])
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
# surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma',vmin=0,vmax=1)
# fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
# ax.set_ylim(testvalues[-1],testvalues[0])
# ax.set_xlim(0,38)
# ax.set_zlim(0,1)
# ax.set_xticks(np.arange(0,37,6))
# ax.set_ylabel("Damping Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_zlabel("Correlation")
# ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
# plt.tight_layout()
# plt.savefig("%sLag_v_Damping_3Dplot_%s.png"%(outpath,expname),dpi=200)


# # Plot damping values
# fig,ax = plt.subplots(1,1)
# for i in range(len(testvalues)):
#     ax.plot(mons3,paramsall[i][1],label="",alpha=(i/len(testvalues))*.5,color='b')
# ax.plot(mons3,damppt,color='k',label='Original Seasonal Cycle')
# ax.plot(mons3,paramsall[np.argmin(rmses)][1],color='r',label="Best, (%.2fx)" % testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Damping (W/m2)")    
# ax.grid(True,ls='dotted')
# plt.savefig("%sDamping_Values_%s.png"%(outpath,expname),dpi=200)

# # **********************************
# #%% Grid Sweep Experiments III : Forcing
# # **********************************
# testvalues = np.arange(10,101,1)
# testparam  = 'Fpt'

# acall     = []
# sstall    = []
# kmonthall = []
# paramsall = []

# for i,val in tqdm(enumerate(testvalues)):
#     #config['Fpt'] = np.ones(12)*37.24208402633666
#     st = time.time()
#     config[testparam] = np.ones(12)*val
#     ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
#     acall.append(ac)
#     sstall.append(sst)
#     kmonthall.append(kmonth)
#     paramsall.append(params)
    
#     print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# #%% Visualize the results
# import cmocean
# model = 1
# expname="ForcingCVary"
# nlag = len(config['lags'])
# nexp = len(testvalues)

# acalls = np.zeros((nexp,nlag))
# for e,ac in enumerate(acall):
#     acalls[e,:] = acall[e][model]


# # Pcolor Plot
# fig,ax = plt.subplots(1,1)
# pcm = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
# fig.colorbar(pcm,ax=ax)
# ax.set_ylabel("Forcing (W/m2)")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Forcing_pcolor_%s.png"%(outpath,expname),dpi=200)


# # Pcolor Plot, differences
# fig,ax = plt.subplots(1,1)
# pcm = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
# fig.colorbar(pcm,ax=ax)
# ax.set_ylabel("Forcing (W/m2)")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
# plt.tight_layout()
# plt.savefig("%sLag_v_Forcing_pcolor_diff_%s.png"%(outpath,expname),dpi=200)

# # Plot minimum RMSE
# rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
# print("Minumum RMSE was for %i with value %f" % (np.argmin(rmses),rmses.min()))


# # Line Plots
# fig,ax = plt.subplots(1,1)
# ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
# for lam in range(25):
#     ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/25)*.5,color='b')
    
# ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)"%testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Correlation")
# ax.set_xlabel("Lag (months)")   
# ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
# ax.grid(True,ls='dotted')
# plt.savefig("%sLag_v_Forcing_lineplot_%s.png"%(outpath,expname),dpi=200)

# XX,YY = np.meshgrid(config['lags'],testvalues[1:])
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
# surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma')
# fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
# ax.set_ylim(100,0)
# ax.set_xlim(0,38)
# ax.set_zlim(0,1)
# ax.set_xticks(np.arange(0,37,6))
# ax.set_ylabel("Forcing $(W/m^{2})$")
# ax.set_xlabel("Lag (months)")
# ax.set_zlabel("Correlation")
# ax.set_title("SST Autocorrelation by Damping (W/m2) \n Lag 0 = Feb")
# plt.tight_layout()
# plt.savefig("%sLag_v_Forcing_3Dplot_%s.png"%(outpath,expname),dpi=200)


# # *********************************************************
# #%% Grid Sweep Experiments IV : Forcing, Seasonal Magnitude
# # *********************************************************
# testvalues = np.arange(0.1,2.1,.1)*10#[0.25,0.5,1,2,4,8,16,32,64,128]
# #vals = np.ones(12)
# testparam  = 'Fpt'

# acall     = []
# sstall    = []
# kmonthall = []
# paramsall = []

# for i,val in tqdm(enumerate(testvalues)):
#     st = time.time()
#     #vals[7]  = val
#     config[testparam] = Fpt*testvalues[i]
#     ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
#     acall.append(ac)
#     sstall.append(sst)
#     kmonthall.append(kmonth)
#     paramsall.append(params)
    
#     print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

# #%% Visualize the results

# model = 1
# expname="ForcingVVary1m"
# nlag = len(config['lags'])
# nexp = len(testvalues)

# acalls = np.zeros((nexp,nlag))
# for e,ac in enumerate(acall):
#     acalls[e,:] = acall[e][model]

# #ytk=np.arange(0,len(testvalues),1)
# ytk = np.arange(.1,2.2,.2)

# # Pcolor Plot
# fig,ax = plt.subplots(1,1,figsize=(8,8))
# im = ax.pcolormesh(config['lags'],testvalues,acalls,vmin=0,vmax=1,cmap='magma')
# #im = ax.imshow(acalls,cmap='magma',vmin=0,vmax=1)
# fig.colorbar(im,ax=ax,fraction=0.015)
# #ax.set_yticklabels(ytk)
# #ax.set_yticklabels(testvalues,fontsize=10)
# #plt.gca().invert_yaxis()
# ax.set_ylabel("Forcing Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Forcing_pcolor_%s.png"%(outpath,expname),dpi=200)

# # Pcolor Plot, differences
# fig,ax = plt.subplots(1,1,figsize=(8,4))
# im = ax.pcolormesh(config['lags'],testvalues,acalls-cesmauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
# #im = ax.imshow(acalls-cesmauto,cmap=cmocean.cm.balance,vmin=-.5,vmax=.5)
# fig.colorbar(im,ax=ax,fraction=0.015)
# #ax.set_yticks(ytk)
# #ax.set_yticklabels(testvalues,fontsize=10)
# #plt.gca().invert_yaxis()
# ax.set_ylabel("Forcing Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
# plt.savefig("%sLag_v_Forcing_pcolordiff_%s.png"%(outpath,expname),dpi=200)

# # Plot minimum RMSE
# rmses = np.mean(np.sqrt((acalls-cesmauto)**2),1)
# print("Minumum RMSE was for %i with value %f" % (testvalues[np.argmin(rmses)],rmses.min()))


# # Line Plots
# fig,ax = plt.subplots(1,1)
# ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
# for lam in range(len(testvalues)):
#     ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/len(testvalues))*.5,color='b')
    
# ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)" % testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Correlation")
# ax.set_xlabel("Lag (months)")   
# ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
# ax.grid(True,ls='dotted')
# plt.savefig("%sLag_v_Forcing_lineplot_%s.png"%(outpath,expname),dpi=200)

# XX,YY = np.meshgrid(config['lags'],testvalues[1:])
# fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
# surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma',vmin=0,vmax=1)
# fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
# ax.set_ylim(testvalues[-1],testvalues[0])
# ax.set_xlim(0,38)
# ax.set_zlim(0,1)
# ax.set_xticks(np.arange(0,37,6))
# ax.set_ylabel("Forcing Multiplier")
# ax.set_xlabel("Lag (months)")
# ax.set_zlabel("Correlation")
# ax.set_title("SST Autocorrelation by Forcing (W/m2) \n Lag 0 = Feb")
# plt.tight_layout()
# plt.savefig("%sLag_v_Forcing_3Dplot_%s.png"%(outpath,expname),dpi=200)


# # Plot Forcing values
# fig,ax = plt.subplots(1,1)
# for i in range(len(testvalues)):
#     ax.plot(mons3,paramsall[i][4],label="",alpha=(i/len(testvalues))*.5,color='b')
# ax.plot(mons3,Fpt,color='k',label='Original Seasonal Cycle')
# ax.plot(mons3,paramsall[np.argmin(rmses)][4],color='r',label="Best, (%.2fx)" % testvalues[np.argmin(rmses)])
# ax.legend()
# ax.set_ylabel("Forcing (W/m2)")    
# ax.grid(True,ls='dotted')
# plt.savefig("%sForcing_Values_%s.png"%(outpath,expname),dpi=200)


# *********************************************************
#%% Grid Sweep Experiments II : Mixed layer Depth, Seasonal Magnitude
# *********************************************************
testvalues = np.arange(0.1,2.1,.1)#[0.25,0.5,1,2,4,8,16,32,64,128]

testparam  = 'mldpt'

acall     = []
sstall    = []
kmonthall = []
paramsall = []

for i,val in tqdm(enumerate(testvalues)):
    st = time.time()
    config[testparam] = mldpt*val
    ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
    acall.append(ac)
    sstall.append(sst)
    kmonthall.append(kmonth)
    paramsall.append(params)
    
    print("Ran script for %s = %s in %.2fs"%(testparam,str(val),time.time()-st))

#%% Visualize the results



model = 3
expname="MLDVVary"
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
ax.set_ylabel("MLD Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Lag 0 = Feb)")
plt.savefig("%sLag_v_MLD_pcolor_%s.png"%(outpath,expname),dpi=200)

# Pcolor Plot, differences
fig,ax = plt.subplots(1,1,figsize=(8,4))
im = ax.pcolormesh(config['lags'],testvalues,acalls-fullauto,vmin=-.5,vmax=.5,cmap=cmocean.cm.balance)
#im = ax.imshow(acalls-cesmauto,cmap=cmocean.cm.balance,vmin=-.5,vmax=.5)
fig.colorbar(im,ax=ax,fraction=0.015)
#ax.set_yticks(ytk)
#ax.set_yticklabels(testvalues,fontsize=10)
#plt.gca().invert_yaxis()
ax.set_ylabel("MLD Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_title("SST Autocorrelation (Stochastic Model - CESM) \n (Lag 0 = Feb)")
plt.savefig("%sLag_v_MLD_pcolordiff_%s.png"%(outpath,expname),dpi=200)

# Plot minimum RMSE
rmses = np.mean(np.sqrt((acalls-fullauto)**2),1)
print("Minumum RMSE was for %i with value %f" % (testvalues[np.argmin(rmses)],rmses.min()))


# Line Plots
fig,ax = plt.subplots(1,1)
ax.plot(lags,cesmauto,color='k',label='CESM-SLAB')
ax.plot(lags,fullauto,color='k',label='CESM Full',ls='dashdot')
for lam in range(len(testvalues)):
    ax.plot(config['lags'],acalls[lam,:],label="",alpha=(lam/len(testvalues))*.5,color='b')
    
ax.plot(lags,acalls[np.argmin(rmses),:],color='r',label="Best, (%f)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag (months)")   
ax.set_title("SST Autocorrelation by MLD (m) \n Lag 0 = Feb")
ax.grid(True,ls='dotted')
plt.savefig("%sLag_v_MLD_lineplot_%s.png"%(outpath,expname),dpi=200)

XX,YY = np.meshgrid(config['lags'],testvalues[1:])
fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'},figsize=(12,4))
surf = ax.plot_surface(XX,YY,acalls[1:,...],cmap='magma',vmin=0,vmax=1)
fig.colorbar(surf,ax=ax,orientation='horizontal',fraction=0.02)
ax.set_ylim(testvalues[-1],testvalues[0])
ax.set_xlim(0,38)
ax.set_zlim(0,1)
ax.set_xticks(np.arange(0,37,6))
ax.set_ylabel("MLD Multiplier")
ax.set_xlabel("Lag (months)")
ax.set_zlabel("Correlation")
ax.set_title("SST Autocorrelation by MLD (W/m2) \n Lag 0 = Feb")
plt.tight_layout()
plt.savefig("%sLag_v_MLD_3Dplot_%s.png"%(outpath,expname),dpi=200)


# Plot damping values
fig,ax = plt.subplots(1,1)
for i in range(len(testvalues)):
    ax.plot(mons3,paramsall[i][2],label="",alpha=(i/len(testvalues))*.5,color='b')
ax.plot(mons3,mldpt,color='k',label='Original Seasonal Cycle')
ax.plot(mons3,paramsall[np.argmin(rmses)][2],color='r',label="Best, (%.2fx)" % testvalues[np.argmin(rmses)])
ax.legend()
ax.set_ylabel("MLD (m)")    
ax.grid(True,ls='dotted')
plt.savefig("%sMLD_Values_%s.png"%(outpath,expname),dpi=200)





# --------------------------------------------
#%% Save test output for ingestion into matlab
# --------------------------------------------
fullpic  = "FULL_PIC_SST_lon330_lat50.npy"
slabpic  = "SLAB_PIC_SST_lon330_lat50.npy"
cesmfull = np.load(datpath+fullpic)
cesmslab = np.load(datpath+slabpic)

sst1 =sst[0]
sst2 =sst[1]
sst3 =sst[2]
sst4 =sst[3]
ssts = []
for i in range(4):
    ssts.append(sst[i])
ssts = np.array(ssts)

fnmat = datpath + "Stochastic_Model_Test_%s.mat" % locstring
print(fnmat)
savemat(fnmat,{'ssts':ssts,'cesmfull':cesmfull,'cesmslab':cesmslab})
fnmat = datpath + "CESM_%s.mat" % locstring
savemat({'cesmfull':cesmfull,'cesmslab':cesmslab})