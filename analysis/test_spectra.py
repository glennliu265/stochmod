#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test Spectra Ann_Avg

What happens when you take annual average 
of white noise time series and compute the spectra?

How does smoothing impact the power at high/low frequencies?


Created on Mon Apr  4 11:34:02 2022

@author: gliu
"""



tsmon  = np.random.normal(0,1,120000) #[120000,]
annavg = proc.ann_avg(tsmon,0) # [10000,]
tsyr   = np.random.normal(0,1,10000)

dtplot = 3600*24*365

inssts  = [tsmon,annavg]#,tsyr]
dts     = [3600*24*30,3600*24*365,]#3600*24*365,]
nsmooth = [1500,1500,500]
labels = ["Monthly","Annually Averaged",]#"Annual-Random"]


fig,ax = plt.subplots(1,1)
specs = []
freqs = []
for i,insst in enumerate(inssts):
    
    
    
    spec,freq,CCs,dofs,r1s = scm.quick_spectrum([insst,],nsmooth[i],0.10,dt=dts[i])
    
    specs.append(spec[0])
    freqs.append(freq[0])
    
    
    ax.plot(freqs[i]*dtplot,specs[i]/dtplot,label=labels[i])

ax.legend()
ax.set_xlim([0,0.5])





