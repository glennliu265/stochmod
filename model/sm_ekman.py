#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run Stochastic Model with Ekman Forcing Included.
Only run the entraining model.

Created/copied from run_sm_rewrite.py on Tue Jan  4 17:50:26 2022

@author: gliu
"""

# %%Dependencies
import numpy as np
import xarray as xr
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy import stats,signal
from tqdm import tqdm
import glob

import sys
stormtrack =0

if stormtrack == 0:
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    
    input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
    output_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_output/"

elif stormtrack == 1:
    
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")
    
    input_path  = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_input/"
    output_path = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/model_output/" 

from amv import proc,viz
import scm
import time
import yo_box as ybx
import tbx
#%% User Edits (copied from sm_rewrite_loop.py on 2021.01.04)
# Landice Mask Name
limaskname = "limask180_FULL-HTR.npy" 


# Try a custom simulation
"""
Fixed MLD Experiment
"""
# cexpname = "hfix50_slab"
# h_cust = np.ones((288,192,12)) * 50 # 50 m slab
# custom_params = {}
# custom_params['h'] = h_cust
# custom_params['q_ek'] = qek_name
# #'forcing' and 'lambda' are two other options
# hconfigs      = [0] # Just run the slab simulation


# Model Params
ampq       = True # Set to true to multiply stochastic forcing by a set value
mconfig    = "SLAB_PIC" # (Automatically loads both)

# Running Parameters
runid      = "011"
pointmode  = 0 
points     = [-30,50]
bboxsim    = [-80,0,0,65] # Simulation Box

useslab    = False # Set to True to use SLAB_CESM parameters for all...
savesep    = False # Set to True to save the outputs differently

# Additional Constants
t_end      = 12000 # Sim Length
dt         = 3600*24*30 # Timestep
T0         = 0 # Init Temp

# Forcing Correction Method (q-corr)
ampq   = 3 #0 = none 1 = old method, 2 = method 1, 3 = method 2

# Damping Significance Test Method
method = 4 # 1 = No Testing; 2 = SST autocorr; 3 = SST-FLX crosscorr, 4 = Both 

# Point information
lonf = -30
latf = 50
debug = False

#expname = 

frcname = "flxeof_090pct_SLAB-PIC_eofcorr2"

qek_name = "Qek_eof_090pct_FULL_PIC_eofcorr0.npy"



check       = True
intgrQ      = False
chk_damping = False

"""

run_sm_rewrite(expname,mconfig,input_path,limaskname,
                   runid,t_end,frcname,ampq,
                   bboxsim,pointmode,points=[-30,50],
                   dt=3600*24*30,
                   debug=False,check=True,
                   useslab=False,savesep=False,
                   intgrQ=False,
                   method=4,chk_damping=False):
    

Inputs
------

1. expname [STR]: Name of experiment output
2. mconfig [STR]: Model Configuration



chk_damping [BOOL] : set True to set negative damping values to 0

"""
    
start = time.time()

if debug:
    t_end = 120 # Just run 10 yr

#%%


# Load data in
# ------------
lon,lat,h,kprevall,damping,dampingfull,alpha,alpha_full = scm.load_inputs(mconfig,frcname,input_path,
                                                                      load_both=True,method=method)
hblt = np.load(input_path + "SLAB_PIC_hblt.npy") # Slab fixed MLD
hblt = np.ones(hblt.shape) * hblt.mean(2)[:,:,None]

# **Additionally Load Qek
# -----------------------
qek_raw = np.load(input_path+qek_name)


# Apply landice mask to all inputs
# --------------------------------
limask    = np.load(input_path+limaskname)
h           *= limask[:,:,None]
kprevall    *= limask[:,:,None]
damping     *= limask[:,:,None]
dampingfull *= limask[:,:,None]
alpha       *= limask[:,:,None,None]
alpha_full  *= limask[:,:,None,None]
hblt        *= limask[:,:,None]
qek_raw     *= limask[:,:,None,None]


if 'h' in custom_params.keys(): # Replace with custom parameters
    h = custom_params['h']
    

# Restrict to region or point (Need to fix this section)
# ---------------------------
inputs = [h,kprevall,damping,dampingfull,alpha,alpha_full,hblt,qek_raw]
if pointmode == 0:
    outputs,lonr,latr = scm.cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
else:
    outputs = scm.cut_regions(inputs,lon,lat,bboxsim,pointmode,points=points)
h,kprev,damping,dampingfull,alpha,alpha_full,hblt,qekr = outputs


# Remove negative damping values, if option is set
# ------------------------------------------------
if chk_damping:
    dmask = (damping<0)
    dmaskfull = (dampingfull<0)

# Generate White Noise
# --------------------
st = time.time()
forcing_full,forcing_qekr = scm.make_forcing(alpha_full,runid,frcname,t_end,input_path,check=check,alpha_full=qekr)
print("Finished forcing in %.2fs"% (time.time()-st))

# 88888 Check above tiling works, then write eperiment for qek



h_in = h.copy() # Variable MLD
f_in = forcing_full
d_in = dampingfull.copy()


# Convert to w/m2
# ---------------
lbd_a   = scm.convert_Wm2(d_in,h_in,dt)
F       = scm.convert_Wm2(f_in,h_in,dt) # [lon x lat x time]

#
# If Option is set, amplitfy F to account for underestimation
# -----------------------------------------------------------
if ampq:
    a        = 1-lbd_a
    a        = 1-lbd_a.mean(2)[...,None]
    if ampq == 1:
        print("Doing Old Correction")
        underest = 2*a**2 / (1+a) # Var(Q) = underest*Var(q)
    elif ampq == 2:
        print("Correcting with method 1")
        underest = scm.method1(lbd_a.mean(2)[...,None])
    elif ampq == 3:
        print("Correcting with method 2")
        underest = scm.method2(lbd_a.mean(2)[...,None],original=False) # Original = uncorrected version with error
        
    ntile = int(t_end/a.shape[2])
    ampmult = np.tile(1/np.sqrt(underest),ntile)
    F *= ampmult
    
# Convert and add ekman forcing
# -----------------------------
F_noek = F.copy()
Fek     = scm.convert_Wm2(forcing_qekr,h_in,dt)
F       = F + Fek

# Integrate Stochastic Model
# --------------------------
T,damping_term,forcing_term,entrain_term,Td   = scm.integrate_entrain(h_in,kprev,lbd_a,F,T0=0,multFAC=True,debug=True)

beta = scm.calc_beta(h_in)
lbd  = lbd_a + beta
FAC  = scm.calc_FAC(lbd_a)

FACtile = np.tile(FAC,int(Fek.shape[-1]/12))
fig,ax = plt.subplots(1,1)
ax.plot((FACtile*F)[33,22],color='orange')
ax.plot((forcing_term)[33,22],color='k')
ax.set_xlim([11800,12000])

# Save Output
# -------------
expname = "%sstoch_output_forcing%s_Qek.npz" % (output_path,frcname)
np.savez(expname,**{
    'sst' : T,
    'lon' : lonr,
    'lat' : latr,
    'damping_term' : damping_term,
    'entrain_term' : entrain_term,
    'Td'           : Td,
    'forcing_term' : F_noek,
    'ekman_term'   : Fek,
    'FAC'          : FAC
    },allow_pickle=True)
print("Saved output to %s in %.2fs" % (expname,time.time()-start))
print("Function completed in %.2fs" % (time.time()-start))

