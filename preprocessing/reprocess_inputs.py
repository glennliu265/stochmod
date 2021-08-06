#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Prepare some inputs, after sm edit

Scrap script to reprocess some inputs
Created on Fri Jul 23 19:33:02 2021

@author: gliu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import tqdm

sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%%

input_path = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/model_input/"
limaskname = "landicemask_enssum.npy" 


#%%


lat    = np.load(input_path+"CESM1_lat.npy")
lon180 = np.load(input_path+"CESM1_lon180.npy")
lon360 = np.load(input_path+"CESM1_lon360.npy")


#%% Load limask (this is in lon360, lat x lon)
limask = np.load(input_path+limaskname)
l1,limask180 = proc.lon360to180(lon360,limask.T[:,:])

# Save again
linamenew = input_path+"limask180_FULL-HTR.npy"
np.save(linamenew,limask180)


#%% Reload and save the FLXstd forcing
klon,klat = proc.find_latlon(-30,50,lon180,lat)

# Load (lon x lat x month)
flxforce = np.load(input_path + "SLAB_PIC_NHFLXSTD_Forcing_MON.npy")

# (lon x lat x pc x month)
flxforcenew = flxforce[:,:,None,:]

# Save new version
flxnamenew = input_path + "flxstd_SLAB-PIC.npy"
np.save(flxnamenew,flxforcenew)


#%%




