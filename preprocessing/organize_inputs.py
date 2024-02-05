#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Organize Stochastic Model Inputs

NOTE Need to check groupby operation in organize inputs

Load in and check inputs


Dimension Conventions:
----------------------
MLD:     [ensemble x month x lat x lon]
Forcing: [ensemble x mode x month x lat x lon]
Damping: [ensemble x lag x month x lag x lon]

Scenarios:
----------
PIC_FULL : CESM1 PiControl  (400-2200) , Fully Coupled Model
PIC_SLAB : CESM1 PiControl  (200-1100) , SLAB Ocean model, 
HTR_FULL : CESM1 Historical (1920-2005), Fully Coupled Model, 42-member large ensemble

Other Definitions:
------------------

Damping:
    - ensorem: ENSO removal step in heat flux feedback (hff) calculation
        - 1 = ENSO was removed for calculations
        - 0 = ENSO was retained
    - lag<N> : N is amount of heat flux feedback lags included in average/or was exponentially fit to. (N=1,2, or 3)
    - mode<N>: Significance testing applied for heat flux feedback.
        - 1 = No Testing; 2 = SST autocorr; 3 = SST-FLX crosscorr, 4 = Both, 5 - replace insignificant values in FULL with those from SLAB
    
    

Forcing
PIC_FULL_Fprime_EOF_090pct_eofcorr2_rolln0.nc
<Scenario>_<Modeltype>_Fprime_EOF_090pct_eofcorr2_rolln0.nc

Created on Mon Jan 22 10:36:59 2024

@author: gliu

"""

#%% Import Stuff

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


#%%  Simple funcs

def save_smoutput(varname,varout,indict,savename):
    da      = xr.DataArray(varout,coords=indict,dims=indict,name=varname)
    edict   = {varname: {'zlib': True}}
    da.to_netcdf(savename,encoding=edict)
    print("Saving to %s" % savename)
    return None



# =============================================================================
#%% Taken from scm.load_inputs, output from the stochastic model paper 
# =============================================================================
figpath  = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/02_Figures/20240202/"
input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
proc.makedir(figpath)


hpath = input_path + "mld/"
fpath = input_path + "forcing/"
dpath = input_path + "damping/"

# Load Latlon
lon180,lat=scm.load_latlon()

# Load Months
mons = np.arange(1,13,1)
ens  = np.arange(1,43,1)


# Plotting Things
mons3 = proc.get_monstr(nletters=3)

mld_ds = {}

# -----------------------------------------------------------------------------
#%% Mixed Layer Depth (seems to be from finalize_inputs)
# -----------------------------------------------------------------------------

# Load in original input for PIC_FULL ---------
h         = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Global Climatological MLD in [meters] # [Lon 180 (288) x Lat (192) x Month (12)]
kprevall  = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month (Kprev)

h_rf      = h.transpose(2,1,0) # month x lat x lon

# Save output
indict = {'mon':mons,
          'lat':lat,
          'lon':lon180}
varname  = "h"
savename = "%sPIC_FULL_HMXL_hclim.nc" % hpath
save_smoutput(varname,h_rf,indict,savename)

# Check
dsh = xr.open_dataset(savename)
dsh.h.isel(mon=0).plot()
dsh.close()

mld_ds['PIC_FULL_HMXL'] = dsh.copy()

#%% Repeat for HBLT (PIC_FULL)

h         = np.load(input_path + "FULL_PIC_HBLT_hclim.npy") # Global Climatological MLD in [meters] # [Lon 180 (288) x Lat (192) x Month (12)]
h_rf      = h.transpose(2,1,0) # month x lat x lon

# Save output
indict = {'mon':mons,
          'lat':lat,
          'lon':lon180}
varname  = "h"
savename = "%sPIC_FULL_HBLT_hclim.nc" % hpath
save_smoutput(varname,h_rf,indict,savename)

# Check
dsh = xr.open_dataset(savename)
dsh.h.isel(mon=0).plot()
dsh.close()

mld_ds['PIC_FULL_HBLT'] = dsh.copy()

#%% Repeat for HMXL in Historical

# Tried loading this, but it seems to be the ensemble average
#h         = np.load(input_path+"FULL_HTR_HMXL_hclim.npy") # Global Climatological MLD in [meters] # [Lon 180 (288) x Lat (192) x Month (12)]

# Open File (only 40 members! :(02_DeepLearning/03_Scripts/preprocessing/prepare_dampingterm_ML.py)
ncname = "CESM1LE_HMXL_Seasonal_1920_2005_EnsAll.nc"
dsld = xr.open_dataset(input_path + "../" + ncname) # [Lat x Lon360 x Month x Ensemble]
dsld = proc.lon360to180_ds(dsld,lonname='lon')

# Plot for Debugging
dsld.HMXL.isel(month=0,ensemble=0).plot() 

# Rename and reorganize to [ens x mon x lat x lon]
dsld = dsld.HMXL.transpose('ensemble','month','lat','lon')
dsout = dsld.rename({'ensemble':'ens',"month":'mon',})
dsout = dsout.rename('h')

# Convert cm --> m
dsout = dsout/100

# Save output
edict    ={'h':{'zlib':True}}
savename = "%sHTR_FULL_HMXL_hclim.nc" % hpath
dsout.to_netcdf(savename,encoding=edict)

# Check
dsh = xr.open_dataset(savename)
dsh.h.isel(mon=0,ens=0).plot()
dsh.close()

mld_ds['HTR_FULL_HMXL'] = dsh.copy()

#%% Get SLAB HBLT

# Open File
h = np.load(input_path+"SLAB_PIC_hblt.npy")
plt.pcolormesh(h[...,0].T),plt.colorbar()
h_rf      = h.transpose(2,1,0) # month x lat x lon

# Save output
indict  = {'mon':mons,
          'lat':lat,
          'lon':lon180}
varname  = "h"
savename = "%sPIC_SLAB_HBLT_hclim.nc" % hpath
save_smoutput(varname,h_rf,indict,savename)

# Check
dsh = xr.open_dataset(savename)
dsh.h.isel(mon=0).plot()
dsh.close()

mld_ds['PIC_SLAB_HBLT'] = dsh.copy()


# -----------------------------------------------------------------------------



#%% Damping
# -----------------------------------------------------------------------------

dampdict = {}

#%% Do for PIC FULL
method  = 5 # 5- replace insignificant values in FULL with those from SLAB
lagstr  = "lag1"
ensostr = ""
#dampingslab   = np.load(input_path+mconfig+"_NHFLX_Damping_monwin3_sig005_dof893_mode%i_%s%s.npy" % (method,lagstr,ensostr))
#dampingfull   = np.load(input_path+"FULL_PIC"+"_NHFLX_Damping_monwin3_sig005_dof1893_mode%i_%s%s.npy" % (method,lagstr,ensostr))

# Load and postprocess FULL (Last modified 02.24.2022)
dampingfull = np.load(input_path+"FULL_PIC_NHFLX_Damping_monwin3_sig005_dof1893_mode5_lag1.npy") # [Lon x Lat x month]

dfull_rf    = dampingfull.transpose(2,1,0) # month x lat x lon

# Save output
indict   = {'mon':mons,
          'lat':lat,
          'lon':lon180}
varname  = "lambda_qnet"
savename = "%sPIC_FULL_NHFLX_Damping_mode5_lag1_ensorem1.nc" % dpath
save_smoutput(varname,dfull_rf,indict,savename)
        


# Check and Copy
dsh = xr.open_dataset(savename)
dsh.lambda_qnet.isel(mon=0).plot()
dsh.close()

dampdict['PIC_FULL_NHFLX'] = dsh.copy()

#%% Do for PIC SLAB

# Load and postprocess SLAB
d= np.load(input_path+"SLAB_PIC_NHFLX_Damping_monwin3_sig005_dof893_mode5_lag1.npy") # [Lon x Lat x month]
d_rf = d.transpose(2,1,0) # month x lat x lon

# Save output
indict   = {'mon':mons,
          'lat':lat,
          'lon':lon180}
varname  = "lambda_qnet"
savename = "%sPIC_SLAB_NHFLX_Damping_mode5_lag1_ensorem1.nc" % dpath
save_smoutput(varname,d_rf,indict,savename)

# Check
dsh = xr.open_dataset(savename)
dsh.lambda_qnet.isel(mon=0).plot()
dsh.close()

dampdict['PIC_SLAB_NHFLX'] = d_rf.copy()


# -----------------------------------------------------------------------------
#%% Load the forcing
# -----------------------------------------------------------------------------

# From sm_rewrite_loop, do SLAB
fnameslab = input_path + 'flxeof_090pct_SLAB-PIC_eofcorr2_Fprime_rolln0' + ".npy"

frc = np.load(fnameslab)     # [lon x lat x modes x month]
frc_rf = frc.transpose(2,3,1,0) # [mode x month x lat x lon]

nmodes   = frc_rf.shape[0]
eof     = np.arange(1,nmodes+1,1)

# Save output
indict   = {
            'eof': eof,
            'mon':mons,
            'lat':lat,
            'lon':lon180}

varname  = "alpha"
savename = "%sPIC_SLAB_Fprime_EOF_090pct_eofcorr2_rolln0.nc" % fpath
save_smoutput(varname,frc_rf,indict,savename)


# Check
dsh = xr.open_dataset(savename)
dsh.alpha.isel(mon=0,eof=0).plot()
dsh.close()

#%% Repeat for Full

# From sm_rewrite_loop, do FULL
fnameslab = input_path + 'flxeof_090pct_FULL-PIC_eofcorr2_Fprime_rolln0' + ".npy"

frc = np.load(fnameslab)     # [lon x lat x modes x month]
frc_rf = frc.transpose(2,3,1,0) # [mode x month x lat x lon]

nmodes   = frc_rf.shape[0]
eof     = np.arange(1,nmodes+1,1)

# Save output
indict   = {
            'eof': eof,
            'mon':mons,
            'lat':lat,
            'lon':lon180}

varname  = "alpha"
savename = "%sPIC_FULL_Fprime_EOF_090pct_eofcorr2_rolln0.nc" % fpath
save_smoutput(varname,frc_rf,indict,savename)

# Check
dsh = xr.open_dataset(savename)
dsh.alpha.isel(mon=0,eof=0).plot()
dsh.close()

#%% Process Fprime fstd forcing, from qcorr_simple output

# PIC SLAB
fn  = "Fprime_PIC_SLAB_rolln0.nc"
ds  = xr.open_dataset(input_path + "../" + fn).load()
dsf = proc.format_ds(ds)

dsfstd = dsf.groupby('time.month').std('time')

dsfstd = dsfstd.rename({'month':'mon'})
dsfstd = dsfstd.rename({'Fprime':'alpha'})

#
edict     = proc.make_encoding_dict(dsfstd)
savename = "%sPIC_SLAB_Fprime_rolln0.nc" % fpath
dsfstd.to_netcdf(savename,encoding=edict)

# Check
dsh = xr.open_dataset(savename)
dsh.alpha.isel(mon=0).plot()

#%% Repeat for PIC FULL

# PIC SLAB
fn  = "Fprime_PIC_FULL_rolln0.nc"
ds  = xr.open_dataset(input_path + "../" + fn).load()
dsf = proc.format_ds(ds)
dsfstd = dsf.groupby('time.month').std('time')
dsfstd = dsfstd.rename({'month':'mon'})
dsfstd = dsfstd.rename({'Fprime':'alpha'})

# 
edict     = proc.make_encoding_dict(dsfstd)
savename = "%sPIC_FULL_Fprime_rolln0.nc" % fpath
dsfstd.to_netcdf(savename,encoding=edict)

# Check
dsh = xr.open_dataset(savename)
dsh.alpha.isel(mon=0).plot()


# <0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0><0>
#%% Compare forcings

ncnames = ["PIC_SLAB_Fprime_rolln0.nc",
          "PIC_FULL_Fprime_rolln0.nc",
          "PIC_SLAB_Fprime_EOF_090pct_eofcorr2_rolln0.nc",
          "PIC_FULL_Fprime_EOF_090pct_eofcorr2_rolln0.nc"]

frcnames = ["std(F'), SLAB",
            "std(F'), FULL",
            "EOF, SLAB",
            "EOF, FULL"
            ]

fcolors  = ["mediumblue","salmon","magenta","goldenrod"]
fmarkers = ["o","x","o","x"]
fls      = ["solid",'solid','dashed','dashed']

# Load Forcings
ds_all       = [xr.open_dataset(fpath+nc).load() for nc in ncnames]

# Get magnitude of forcings for EOF-based
for ii in [2,3]:
    
    ds = ds_all[ii]
    ds = np.sqrt((ds**2).sum('eof'))
    ds_all[ii] = ds

#%% Plot forcing value at a test point
lonf = -30
latf = 50
locfn,loctitle=proc.make_locstring(lonf,latf)
ds_pt = [ds.sel(lon=lonf,lat=latf,method='nearest') for ds in ds_all]
fig,ax = viz.init_monplot(1,1)
for ii,da in enumerate(ds_pt):
    ax.plot(mons3,da.alpha,label=frcnames[ii],lw=1.5,color=fcolors[ii],marker=fmarkers[ii],ls=fls[ii])
ax.legend()
ax.set_title("Forcing Amplitude @ %s" % loctitle)
ax.set_xlabel("Month")
ax.set_ylabel("W/$m^2$")

savename = "%sForcing_comparison_Fstd_EOF.png" % (figpath)
plt.savefig(savename,dpi=150,bbox_inches='tight')
