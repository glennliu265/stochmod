#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Debug Difference between synth_stochmod and stochmod_rewrite scripts

Copied upper section form viz_synth_stochmod_combine.py

Created on Sun Dec 17 19:28:13 2023

@author: gliu
"""


import sys
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cartopy.crs as ccrs

#%% Set Paths, Import Custom Modules
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
    datpath     = projpath + '01_Data/model_output/'
    rawpath     = projpath + '01_Data/model_input/'
    outpathdat  = datpath + '/proc/'
    figpath     = projpath + "02_Figures/20231218/"
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")

elif stormtrack == 1:
    datpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/"
    rawpath     = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    outpathdat  = datpath + '/proc/'
    
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

from amv import proc,viz
import scm
import tbx

proc.makedir(figpath)
#%% Import Parameters for Stochmod

cwd = os.getcwd()
sys.path.append(cwd+"/../")
import stochmod_params as sparams




#%% Load Constant_v_vary experiments

# Set up Configuration
config = {}
config['mconfig']     = "SLAB_PIC" # Model Configuration
config['ftype']       = "DJFM-MON" # Forcing Type
config['genrand']     = 0          # Toggle to generate new random timeseries
config['fstd']        = 1          # Set the standard deviation N(0,fstd)
config['t_end']       = 120000     # Number of months in simulation
config['runid']       = "syn009"   # White Noise ID
config['fname']       = "flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0.npy"   #['NAO','EAP,'EOF3','FLXSTD']
config['pointmode']   = 1          # Set to 1 to generate a single point
config['query']       = [-30,50]   # Point to run model at 
config['applyfac']    = 2          # Apply Integration Factor and MLD to forcing
config['lags']        = np.arange(0,37,1)
config['output_path'] = datpath # Note need to fix this
config['smooth_forcing'] = False
config['method']      = 0 # Correction Factor (q-corr, lambda-based)
config['favg' ]       = False

config.pop('Fpt',None)
config.pop('damppt',None)
config.pop('mldpt',None)


# Load SSTs
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

# Grab and Combine
labels_lower     = ["All Constant (Level 1)",
                     r"Vary $F'$ (Level 2b)",
                     r"Vary $\lambda_a$ (Level 2a)",
                     "Vary $F'$ and $\lambda_a$ (Level 3)"] 
colors =["red","violet","orange","k",'gray']
# SM Upper Hierarchy (05/25/2021)
labels_upper = ["h=50m",
                 "Vary $F'$ and $\lambda_a$ (Level 3)",
                 "Vary $F'$, $h$, and $\lambda_a$ (Level 4)",
                 "Entraining (Level 5)"]

inssts   = [sst[1],sst[2],sst[3],]
labels   = np.concatenate([labels_upper[1:],])

sst_in_pt = np.array(inssts).T
#%% Load experiments from stochmod_region
# Get the list of files
expname      = "default" # Indicate Experiment name (see stochmod_params)
flist        = sparams.rundicts[expname]
print(flist)
continuous   = True


#% Load the files, processing by runid

bbox_sel = [-45,-10,50,60]
#bbox_sel = [-31,-30,49,50]
debug    = False


f        = 0
ssts     = []
#ssts_reg = []
for f in range(len(flist)):
    # Load Information for the run
    expid   = flist[f]
    ld      = np.load(datpath+"stoch_output_%s.npz"%(expid),allow_pickle=True)
    sst     = ld['sst'] # [hconfig x lon x lat x time]
    lon     = ld['lon']
    lat     = ld['lat']
    
    # Select region
    sst      = sst.transpose(1,2,3,0) # [lon x lat x time x hconfig]
    nlon,nlat,ntime,nmodels=sst.shape
    sst      = sst.reshape(nlon,nlat,ntime*nmodels) 
    sst_ravg = proc.sel_region(sst,lon,lat,bbox_sel,reg_avg=True,awgt=1)
    sst_ravg = sst_ravg.reshape(ntime,nmodels)
    ssts.append(sst_ravg)
    
    # Also Select the region
    sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
    if f == 0:
        nlons,nlats,_=sstr.shape
        ssts_reg = np.ones((nlons,nlats,ntime,3,len(flist)))*np.nan
    ssts_reg[:,:,:,:,f] = sstr.reshape(nlons,nlats,ntime,nmodels)
        #ssts_reg.append(sstr.reshape())
    
    
    
    #sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
    #ssts_reg.append(sstr)
    
    # Check 
    if debug:
        #sstr,lonr,latr=proc.sel_region(sst,lon,lat,bbox_sel)
        fig,ax = plt.subplots(1,1,subplot_kw={'projection':ccrs.PlateCarree()})
        ax = viz.add_coast_grid(ax,fill_color="k")
        ax.set_extent([-80,0,0,65])
        pcm = ax.pcolormesh(lonr,latr,sstr[:,:,0].T)

ssts     = np.concatenate(ssts,axis=0) # [time x hconfig]


# Do some reshaping/dimension reorganization
ssts_reg_new = ssts_reg.transpose(0,1,4,2,3) # [lon x lat x file x time x hconfig]
ssts_reg_new = ssts_reg_new.reshape(nlons,nlats,len(flist)*ntime,3) # [lon x lat x all_time x hconfig]
ssts_reg_new = ssts_reg_new.reshape(nlons,nlats,int(len(flist)*ntime/12),12,3) # [lon x lat x yr x mon x hconfig]
#


# Calclate monthly variance for SM
monvar_sm    = [np.nanvar(ssts_reg_new[:,:,:,:,ii],2) for ii in range(3)]

lonf        = -30
latf        = 50
klons,klats = proc.find_latlon(lonf, latf,lonr,latr)


sst_in_reg  = ssts_reg_new[klons,klats,:,:,:].reshape(len(flist)*ntime,3) # [time x hconfig]

#%% Load CESM and Compare
# Read in CESM autocorrelation for all points'
fullauto = np.load(datpath+"../CESM_clim/TS_FULL_Autocorrelation.npy")
kmonth   = 1
print("Kmonth is %i"%kmonth)
_,_,lon,lat,lon360,cesmslabac,damping,_,_ = scm.load_data(mconfig,ftype)
ko,ka     = proc.find_latlon(query[0]+360,query[1],lon360,lat)
cesmauto2 = cesmslabac[kmonth,:,ka,ko]
cesmauto  = cesmauto2[lags]
cesmautofull = fullauto[kmonth,lags,ko,ka]

fnames     = ["CESM1_FULL_postprocessed_NAtl.nc","CESM1_SLAB_postprocessed_NAtl.nc"]
mnames     = ["FCM","SOM"] 
mcolors    = ["k","gray"]

#datpathc    = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/"
sstscesm      = []
ssts_reg_cesm = []
for f in range(2):
    varname = "SST"
    sst_fn = fnames[f]
    ds   = xr.open_dataset(datpath+"../"+sst_fn)
    ds   = ds.sel(lon=slice(bbox_sel[0],bbox_sel[1]),lat=slice(bbox_sel[2],bbox_sel[3]))
    lonc = ds.lon.values
    latc = ds.lat.values
    sstc = ds[varname].values # [lon x lat x time]
    sstscesm.append(sstc)
    ssts_reg_cesm.append(sstc.copy())


# Compute the mean
sstscesm      = [proc.area_avg(s,bbox_sel,lonc,latc,1) for s in sstscesm]
#ssts_reg_cesm = 

# Reshape
nlonc,nlatc,_ = ssts_reg_cesm[0].shape
ntimes_cesm = [s.shape[0] for s in sstscesm]
nyrs_cesm   = [int(t/12) for t in ntimes_cesm]
sstscesm     = [sstscesm[ii].reshape(nyrs_cesm[ii],12) for ii in range(2)]


ssts_reg_cesm = [ssts_reg_cesm[ii].reshape(nlonc,nlatc,nyrs_cesm[ii],12) for ii in range(2)] #
klonc,klatc   = proc.find_latlon(lonf,latf,lonc,latc)
sst_cesm = [sst[klonc,klatc,:,:].flatten() for sst in ssts_reg_cesm]

#%% Now check both monthly variance an

basemonth   = 2
# Stack
ssts_in_all = np.concatenate([sst_in_pt,sst_in_reg],axis=1) # [time x hconfig*sim]
ssts_in     = [ssts_in_all[:,ii] for ii in range(6)] + sst_cesm
sstacs      = scm.calc_autocorr(ssts_in,lags,basemonth,calc_conf=False,)

#%% Compare

names = []
fig,ax = plt.subplots(1,1,constrained_layout=True)
for ii in range(6):
    if ii < 3:
        sim="Synth"
        ls='dashed'
    else:
        sim="Region"
        ls='solid'
    label= labels[ii%3] + " (%s)" % (sim)
    
    ax.plot(lags,sstacs[ii],label=label,c=colors[ii%3],ls=ls)
    names.append(label)
    
ax.plot(lags,sstacs[7],label="SOM",color="gray")
ax.plot(lags,sstacs[6],label="FCM",color="k")
ax.plot(lags,cesmauto,label="SOM (old)",color="gray",ls='dashed')
ax.plot(lags,cesmautofull,label="FCM (old)",color="k",ls='dashed')

names = names + ["FCM","SOM","FCM (old)","SOM (old)"]

ax.legend()
ax.set_title("Differences Between Scripts")

#%% Check Monthly Variance...

nyrs    = [int(sst.shape[0]/12) for sst in ssts_in]
monvars = [ssts_in[ii].reshape(nyrs[ii],12).var(0) for ii in range(8)] 

# Plot it
mons3 = proc.get_monstr(nletters=3)
fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(10,6))
for ii in range(6):
    if ii < 3:
        sim="Synth"
        ls='dashed'
    else:
        sim="Region"
        ls='solid'
    label= labels[ii%3] + " (%s)" % (sim)
    
    ax.plot(mons3,monvars[ii],label=label,c=colors[ii%3],ls=ls)
    
ax.plot(mons3,monvars[7],label="SOM",color="gray")
ax.plot(mons3,monvars[6],label="FCM",color="k")
ax.legend()
ax.set_title("Differences Between Scripts")


#%% Check Spectra

nsmooth   = 350
cnsmooths = [100,100]
nsmooths  = np.concatenate([np.ones(6)*nsmooth,cnsmooths])
dt        = 3600*24*30
pct        = 0.00

specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(ssts_in,nsmooths,pct,dt=dt)

# Convert to list for indexing NumPy style
convert = [specs,freqs,]
for i in range(len(convert)):
    convert[i] = np.array(convert[i])
specs,freqs = convert

# Compute the variance of each ts
sstvars     = [np.var(insst) for insst in ssts_in]
#sstvars_lp  = [np.var(proc.lp_butter(insst,120,6)) for insst in inssts]
#sststds = [np.std(insst) for insst in inssts]
#%%

colors_stack = ['red', 'violet', 'orange'] + ['red', 'violet', 'orange']   +  [ 'k', 'gray']
fig,ax       = plt.subplots(1,1,figsize=(8,4))
xtks         = [0.01, 0.05, 0.1 , 0.2 , 0.5 ]
for ii in range(8):
    if ii < 3:
        sim="Synth"
        ls='dotted'
        alpha=0.5
    else:
        sim="Region"
        ls='solid'
        alpha=1
    
    ax.plot(freqs[ii]*dt,specs[ii]/dt,label=names[ii],ls=ls,color=colors_stack[ii],alpha=alpha)



ax2 = ax.twiny()
for a in [ax,ax2]:
    a.set_xticks(xtks)
    a.set_xlim([xtks[0],xtks[-2]])
    a.set_ylim([0,22])
ax2.set_xticklabels(1/np.array(xtks))
ax.legend()

ax.set_xlabel("Frequency (1/mon)")
ax.set_ylabel("Power")

#%% Save Output So I can compare in other scripts

acs_out = sstacs + [cesmautofull,cesmauto]


outdict = {
    'names'   : names,
    'lags'    : lags,
    'lonf'    : lonf,
    'latf'    : latf,
    'ssts'    : ssts_in,
    'acs'     : acs_out,
    'monvars' : monvars,
    }


sstacs_out = [sstacs[i] for i in range(len(sstacs))] + [cesmautofull,cesmauto2]

outdict = {
    "names"     : names,
    "lags"      : lags,
    "ssts"      : ssts_in,
    "acs"       : sstacs,
    "monvars"   : monvars,
    "specs"     : specs,
    "freqs"     : freqs,
    "specparams": [nsmooths,pct,dt],
    "colors"    : colors_stack
    }

savename = "%sproc/SPG_Point_SST_Characteristics.npz" % datpath
np.savez(savename,**outdict,allow_pickle=True)


#%% CHECK DIFFERENCES IN INPUT PARAMETERS

#fig,ax = plt
