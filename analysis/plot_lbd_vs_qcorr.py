#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot Lambda vs q-corr
Created on Mon Aug 23 19:51:02 2021

@author: gliu
"""

import numpy as np



outpath = ""

lbd_test = np.linspace(-1,1,20)
a        = 1-lbd_test

qcorr    = 2*a**2 /(1+a)


fig,ax   = plt.subplots(1,1)
ax.plot(lbd_test,qcorr)
ax.set_ylabel("q-corr")
ax.set_xlabel("$\lambda$ (1/mon)")
ax.grid(True,ls='dotted')
ax.set_title(r"$\lambda$ vs. q-corr ($\frac{2a^2}{1+a}$)")
plt.savefig("")
