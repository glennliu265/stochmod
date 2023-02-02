#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test if magnitude of average differences is the same as
average difference of the magnitude

Created on Wed Aug 17 18:58:41 2022

@author: gliu
"""

import numpy as np


x = np.random.normal(0,1,(22,44,100,2))
y = np.random.normal(0,3,(22,44,100,2))



# Take differences first, then compute magnitude

# U vertical diff
xdiff_0 = (x[:,:,:30,1] - x[:,:,:30,0]).mean(2)   # First Period
xdiff_1 = (x[:,:,-30:,1] - x[:,:,-30:,0]).mean(2) # Last Period
xdiffd = xdiff_1 - xdiff_0

# V vertical diff
ydiff_0 = (y[:,:,:30,1] - y[:,:,:30,0]).mean(2)
ydiff_1 = (y[:,:,-30:,1] - y[:,:,-30:,0]).mean(2)
ydiffd = ydiff_1 - ydiff_0

# Calculate modulo
modulodiff = np.sqrt(xdiffd**2 + ydiffd**2)


# Compute magnitude, then differences
m       = np.sqrt(x**2 + y**2)
mdiff_0 = (m[:,:,:30,1] - m[:,:,:30,0]).mean(2) # Mean vertical diff of each period
mdiff_1 = (m[:,:,-30:,1] - m[:,:,-30:,0]).mean(2)
mdiffd  = mdiff_1 - mdiff_0 # Difference two periods

print(np.nanmax(np.abs((modulodiff-mdiffd).flatten())))