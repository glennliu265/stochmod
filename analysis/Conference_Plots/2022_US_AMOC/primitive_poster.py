#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roughly figure out how the poster will look like size-wize

Created on Thu Apr 21 15:18:39 2022

@author: gliu
"""

import matplotlib.pyplot as plt
import numpy as np

# Declare posterboard sizes
test = np.ones((42,48))   # Maximum size/size of posterboard
test1 = np.zeros((40,30)) # Size of the poster

# Create height/width arrays
h  = np.arange(0,48,1)
w  = np.arange(0,42,1)
h1 = np.arange(0,test1.shape[1],1) # Create arrays to indicate location of poster
w1 = np.arange(0,test1.shape[0],1)

# Plot it!
fig,ax = plt.subplots(1,1)
ax.pcolormesh(w,h,test.T,cmap='cmo.dense')
ax.pcolormesh(w1+1,h1+5,test1.T,cmap='cmo.balance')
ax.grid(True)
ax.set_aspect('equal')