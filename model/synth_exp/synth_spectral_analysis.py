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
    lbd,lbd_entr,FAC,beta = scm.set_stochparams(mldpt,damppt,dt,ND=False,hfix=hfix,hmean=mldmean)
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
ac,sst,dmp,frc,ent,Td,kmonth,params=synth_stochmod(config)
[o,a],damppt,mldpt,kprev,Fpt       =params

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

sstall = sst
# --------------------
#%% Calculate Spectra
# --------------------

#% ----------------------
#%% Do spectral analysis
#% ------------------------

# Parameters
pct     = 0.10
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
freq1s,P1s,CLs = [],[],[]
for sstin in [cesmfull,cesmslab]:
    sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
    P,freq,dof,r1=sps
    
    CC = ybx.yo_speccl(freq,P,dof,r1,clvl)
    #pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
    
    P    = P*dt
    freq = freq/dt
    CC   = CC*dt
    
    P1s.append(P)
    freq1s.append(freq)
    CLs.append(CC)
Pcesmfull,Pcesmslab = P1s
freqcesmfull,freqcesmslab = freq1s
clfull,clslab = CLs


# -----------------------------------------------------------------
# Calculate and make individual plots for stochastic model output
# -----------------------------------------------------------------
nsmooth=nsmooth*10/2
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
        
    
    if i < 2:
        l1 =ax.semilogx(freqcesmslab,Pcesmslab*freqcesmslab,label="CESM-SLAB",color='gray',lw=0.75)
        l2 =ax.semilogx(freqcesmslab,clslab[:,0]*freqcesmslab,label="CESM-SLAB (AR1)",color='red',lw=0.75,alpha=0.4)
        l3 =ax.semilogx(freqcesmslab,clslab[:,1]*freqcesmslab,label="CESM-SLAB (95%)",color='blue',lw=0.75,alpha=0.4)
    else:
        l1 =ax.semilogx(freqcesmfull,Pcesmfull*freqcesmfull,label="CESM-FULL",color='gray',lw=0.75)
        l2 =ax.semilogx(freqcesmfull,clfull[:,0]*freqcesmfull,label="CESM-FULL (AR1)",color='red',lw=0.75,alpha=0.4)
        l3 =ax.semilogx(freqcesmfull,clfull[:,1]*freqcesmfull,label="CESM-FULL (95%)",color='blue',lw=0.75,alpha=0.4)

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
    plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,labels[i],nsmooth,pct*100,axopt),dpi=200)

#%% Plot spectra for SST from PIC

fullpic  = "FULL_PIC_SST_lon330_lat50.npy"
slabpic  = "SLAB_PIC_SST_lon330_lat50.npy"
cesmfull = np.load(datpath+fullpic)
cesmslab = np.load(datpath+slabpic)



sstin = cesmfull
sps = ybx.yo_spec(sstin,opt,nsmooth,pct,debug=False)
P,freq,dof,r1=sps
pps = ybx.yo_specplot(freq,P,dof,r1,tunit,dt=dt,clvl=clvl,axopt=axopt,clopt=clopt)
splotparams.append(pps)
fig,ax,h,hcl,htax,hleg = pps

def set_monthlyspec(ax,htax):
    
    # Divisions of time
    # dt  = 3600*24*30
    # fs  = dt*12
    # xtk      = np.array([1/fs/100,1/fs/50, 1/fs/25, 1/fs/10 , 1/fs/5, 1/fs])
    # xtkm    = ["%i" % np.round(i) for i in 1/xtk/dt]
    # xtklabel = ['%.1e \n (century)'%xtk[0],'%.1e \n (50yr)'%xtk[1],'%.1e \n (25yr)'%xtk[2],'%.1e \n (decade)'%xtk[3],'%.1e \n (5year)'%xtk[4],'%.2e \n (year)'%xtk[5]]
    
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

#xt
vlv = [1/(100*12*dt),1/(12*10*dt),1/(12*dt)]
vll = ["Century","Decade","Year"]
for vv in vlv:
    ax.axvline(vv,color='k',ls='dashed',label=vll,lw=0.75)


ax.set_xlabel("Frequency (cycles/sec)",fontsize=13)
ax.set_ylabel(r"Frequency x Power $(^{\circ}C)^{2}$",fontsize=13)
ax.set_title("Power Spectrum for %s" % labels[i] + "\n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,'CESM_FULL',nsmooth,pct*100,axopt),dpi=200)

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
ax.set_title("Power Spectrum \n" + "nsmooth=%i, taper=%.2f" % (nsmooth,pct))
plt.tight_layout()
plt.savefig("%sPowerSpectra_%s_nsmooth%i_pct%03d_axopt%i.png"%(outpath,'COMPARISON',nsmooth,pct*100,axopt),dpi=200)


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

