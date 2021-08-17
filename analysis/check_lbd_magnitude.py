#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Check dlambda/dt magnitude to make sure it is not too big or small

Created on Fri May 21 00:43:29 2021

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

from scipy import signal,stats

from scipy.ndimage.filters import uniform_filter1d

#%% User Edits

# Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/'
input_path  = datpath + 'model_input/'
#output_path = datpath + 'model_output/'
outpath = "/Users/gliu/Downloads/06_School/05_Spring2021/EPS231/Project/Figures/20210513/"
proc.makedir(outpath)

mons3=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
rho = 1026
cp0 = 3996

lonf  = -30
latf  = 50

#%%


# Load SSTs
ts = np.load(datpath+"TS_PIC_Full.npy") # [yr x mon x lat x lon360]

# Load Estimated Damping [288, 192, 12] [W/m2/C]
damping = np.load(input_path+"SLAB_PIC"+"_NHFLX_Damping_monwin3_sig005_dof894_mode4.npy")

# Load Lat/Lon, autocorrelation
dampmat        = 'ensavg_nhflxdamping_monwin3_sig020_dof082_mode4.mat'
loaddamp       = loadmat(input_path+dampmat)
lon            = np.squeeze(loaddamp['LON1'])
lat            = np.squeeze(loaddamp['LAT'])
lon360         = loadmat("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/CESM1_LATLON.mat")['LON'].squeeze()

# Load mixed layer depths
mld            = np.load(input_path+"FULL_PIC_HMXL_hclim.npy") # Climatological MLD
kprevall       = np.load(input_path+"FULL_PIC_HMXL_kprev.npy") # Entraining Month

#%%

# Get Point Indices
klon,klat = proc.find_latlon(lonf,latf,lon,lat)
if lonf < 0:
    lonfn = lonf+360
else:
    lonfn = lonf
klon360,_ = proc.find_latlon(lonfn,latf,lon360,lat)

# Get values at a point
tspt   = ts[:,:,klat,klon360]
tspt   = tspt - tspt.mean(0)[None,:] # Remove monthly means
damppt = damping[klon,klat,:]
mldpt  = mld[klon,klat,:]


#%% First find maximum difference in lbd globally

dampdiff = damping - np.roll(damping,shift=1,axis=2)
dampdiffmax = np.nanmax(np.abs(dampdiff),axis=2)
plt.pcolormesh(dampdiffmax.T,vmin=0,vmax=10),plt.colorbar()
print(np.nanmax(np.abs(dampdiff)))

#%% Next find the damping values for the SPG point only

dt = 3600*24*30

dampconvert = damppt / (rho*cp0*mldpt)




indamp = dampconvert

dampdiff2 = indamp - np.roll(indamp,1)
dampmax2 = np.max(np.abs(dampdiff2))
print(dampmax2)



dampdiff2

