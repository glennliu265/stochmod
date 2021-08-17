#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Verify Theoretical Heat Flux, based on Claude's Note
Created on Fri Jul 30 15:53:40 2021

@author: gliu
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%


timescale = 10 # E-folding timescale in months

# Calculate lambda and integration factor
lbd = 1/timescale
FAC = (1-np.exp(-lbd))/lbd
a = 1-lbd

# Print values to confirm
print(a)
print(lbd)
print(FAC)

# Print out a few more test values
print(2*a**2/(1+a))
print((2*a**2 + 2*a-2)/(1+a))
print((-2-2*a**3)/(1-a**2))
print((-2-2*a**3+4*a)/(1-a**2))



#cos_term = 

# Plot difference between 1-lambda and np.exp(-lbd)
lbd_rng = 1/np.arange(0.1,24,0.1)
fig,ax = plt.subplots(1,1)
ax.plot(1-lbd_rng)
ax.plot(np.exp(-lbd_rng),color="r")
ax.legend()
ax.set_xlim([1,24])

# Difference seems to mostly occur when 1/lbd < 1, two lines converge
# as 1/lbd --> inf

# Function to calculate the new F_zz ratio
def calc_FQQ_new(a,omega,FZZ=None,ZeqZp=False,autocalc=False):
    lbd = 1-a
    FAC = (1-np.exp(-lbd))/lbd
    cos_term = np.cos(2*np.pi*omega)
    denom    = 1 + a**2 - 2*a*cos_term
    
    if ZeqZp: # Assume F_ZZ = 0.90 * F_Z'Z'
        numer = 2*a**2 * (1-cos_term)
    else: # Assume F_ZZ = 0.8 F_Z'Z'
        numer = 0.2 - 0.2 * a * (1+cos_term) + a**2 * (2 - 1.8*cos_term)
    
    if autocalc:  # Calculate F_ZZ = FAC**2 F_Z'Z'
        denoms = (1+a**2-2*a*cos_term)
        term2 = lbd**2/denoms
        term3 = 2*lbd*FAC**2*(1-a*cos_term)/denoms
        
        ratios = 1 + term2 - term3
        return ratios
    
    if FZZ is None: # Just return the constant
        return numer/denom
    else: # Otherwise, multiply constant across all frequencies
        return FZZ * numer/denom

# Print different values of the output ratio which multiplies F_z'z'
omega1 = np.array([0,0.25,0.5]) # case where z != z'
r_out1 = calc_FQQ_new(a,omega1)
print(r_out1)
r_out2 = calc_FQQ_new(a,omega1,ZeqZp=True) # Case where we assume z = z'
print(r_out2)
r_out3 = calc_FQQ_new(a,omega1,autocalc=True)
print(r_out3)

#%% For the next section, recreate the theoretical spectral analysis plots





