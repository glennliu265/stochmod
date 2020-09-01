#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Loop to run the stochastic model at 1 point with specified experiment settings
Created on Mon Aug 31 19:05:07 2020

@author: gliu
"""


import sys
import time
import numpy as np

#%% Determine System
startall = time.time()
stormtrack = 0
if stormtrack == 0:
    projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
#    scriptpath  = projpath + '03_Scripts/stochmod/'
    datpath     = projpath + '01_Data/'
   
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
    sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")


elif stormtrack == 1:
    datpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/Model_Data/"
    sys.path.append("/home/glliu/00_Scripts/01_Projects/00_Commons/")
    sys.path.append("/home/glliu/00_Scripts/01_Projects/01_AMV/02_stochmod/stochmod/model/")

#%%

import stochmod_region as sr

# # of runs
pointmode = 2 # Set to 1 to output data for the point speficied below
points=[-30,50] # Lon, Lat for pointmode

naoscale = 10 # Number to scale NAO and other forcings by

# Integration Options
nyr      = 10000        # Number of years to integrate over
fstd     = 0.3         # Standard deviation of the forcing

bboxsim  = [-100,20,-20,90] # Simulation Box

# Do a stormtrackloop
runids = ["001"]
funiforms = [0,1,2,5,6]

# Set region variables
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = [-80,0 ,0,65]
regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
rcol = ('b','r',[0,1,0],'k')
rcolmem = [np.array([189,202,255])/255,
           np.array([255,134,134])/255,
           np.array([153,255,153])/255,
           [.75,.75,.75]]


# Set Forcing Names and colors
funiforms=[0,1,2,5,6]
fnames  = ["Random","Uniform","NAO (DJFM)","EAP (DJFM)","NAO+EAP (DJFM)"]
fcolors = ["teal","springgreen","b","tomato","m"]
fstyles = ["dotted","dashed",'solid','solid','solid']

# Set Model Names
modelname = ("MLD Fixed","MLD Max", "MLD Clim", "Entrain")


for region in range(4):
    
    bboxsim = bboxes[region]
    
    for runid in runids:
        
        
        for funiform in funiforms:
            
            
            if funiform < 2:
                fscale = 1
                genrand=1
            else:
                fscale = naoscale
                genrand=0
        
            
        
            sr.stochmod_region(pointmode,funiform,fscale,runid,genrand,nyr,fstd,bboxsim,stormtrack,points=points)
            print("Completed run %s funiform %s (Runtime Total: %.2fs)" % (runid,funiform,time.time()-startall))