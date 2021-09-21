#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NumPy Reminders

Small snippets to remind myself of how numpy works

Created on Mon Aug 16 11:49:43 2021

@author: gliu
"""


import numpy as np
import matplotlib.pyplot as plt

#%% np.tile

yr  = 10
test = np.tile(np.arange(12),yr)

test_yrfirst = test.reshape(yr,12)
test_monfirst = test.reshape(12,yr)

plt.plot(test_yrfirst.T),plt.title("Array: (Year, Month)") # The correct way to reshape
plt.plot(test_monfirst.T),plt.title("Array: (Month, Year)") # Incorrect way