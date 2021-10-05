#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate AMV from HadISST Data
Also save detrended data for stochastic model comparison

Incudes
 - Save detrended HadISST Data for regional analysis, etc
 - Examine the Effect of Detrending, compare to Frankignoul et al. 2017 
 - AMV plots for general exam

Created on Sat Aug 22 21:05:08 2020

@author: gliu
"""


from scipy.io import loadmat
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import time
import cmocean
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd

import sys
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/03_Scripts/")
sys.path.append("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/03_Scripts/stochmod/model/")
from amv import proc,viz
import scm

#%% User Edits
tstart = '1900-01-01'
tend   = '2018-12-31'

startyr = int(tstart[:4]) # Start year for AMV Analysis
bbox    =[-80,0 ,0,65]    # AMV bbox
# Path to SST data from obsv
projpath = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
outpath = projpath + '02_Figures/20211001/'
proc.makedir(outpath)
datpath = projpath + '01_Data/'
datpath2 = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/00_Commons/01_Data/"

## DETRENDING OPTIONS
method = 2
dropedge = 5 # Number of years to drop off when doing regression


#%% Load in HadISST Data

# Load in observation SST data to compare
obvhad = loadmat(datpath2+"hadisst.1870_2018.mat")
hlat   = np.squeeze(obvhad['LAT'])
hlon   = np.squeeze(obvhad['LON'])
hyr    = obvhad['YR']
hsst   = obvhad['SST']

# Change hsst to lon x lat x time
hsst = np.transpose(hsst,(2,1,0))

#%%

# Make time array using pandas, then convert to npdatetime64
t = pd.date_range('1870-01-01',periods=hsst.shape[-1],freq="MS")
ids = (t>=tstart) * (t<=tend) # Get the starting and ending times
#times = t[ids].to_datetime()



# Take the set time period
#monstart = (startyr+1-hyr[0,0])*12

hsst = hsst[:,:,ids]
nyrs = int(hsst.shape[2]/12)


#%% Fix Latitude Dimensions for HSST
# currently it is arranged 90:-90, need to flip to south first

# Find north and south latitude points
hsouth = np.where(hlat <= 0)
hnorth = np.where(hlat > 0)

# Find corresponding points in data
hsstsouth = np.squeeze(hsst[:,hsouth,:])[:,::-1,:]
hsstnorth = np.squeeze(hsst[:,hnorth,:])[:,::-1,:]

# Stitch things together, reversing the order 
hlatnew = np.squeeze(np.concatenate((hlat[hsouth][::-1],hlat[hnorth][::-1])))
hsstnew = np.concatenate((hsstsouth,hsstnorth),axis=1)

#%% Remove Seasonal Cycle first and plot

debug = True

nlon,nlat,nmon = hsst.shape

dsfirst = np.reshape(hsstnew,(360,180,nyrs,12))
dsfirst = dsfirst - np.mean(dsfirst,axis=2)[:,:,None,:]
dsfirst = np.reshape(dsfirst,(360,180,hsstnew.shape[2]))

# Try to also deseason using sinusoid fit
hcopy = hsstnew.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(hcopy,1) # [Space x time]
x,E = proc.remove_ss_sinusoid(okdata.T) # {Time x Space}
ss  = E@x
okdata_ds = (okdata.T - ss).T
dssinu = np.zeros((nlon*nlat,nmon))*np.nan
dssinu[okpts,:] = okdata_ds
dssinu          = dssinu.reshape(nlon,nlat,nmon)

if debug:
    klon,klat = proc.find_latlon(330,55,hlon,hlatnew)
    
    sstpt = [dsfirst[klon,klat,:],
             dssinu[klon,klat,:]]
    
    nsmooths = [1,1]
    pct      = 0.10
    specs,freqs,CCs,dofs,r1s = scm.quick_spectrum(sstpt,nsmooths,pct)
    
    
    xlm = [1e-2,5e0]
    xper = np.array([200,100,75,50,25,10,5,1,0.5]) # number of years
    xtks = 1/xper
    xlm  = [xtks[0],xtks[-1]]
    dt   = 3600*24*365
    
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    
    
    mnames = ["Mon-Anom","Sinusoid"]
    mcols  = ['b','r']
    msty   = ["solid",'dashed']
    for i in range(2):
        ax.plot(freqs[i]*dt,specs[i]/dt,label=mnames[i],color=mcols[i],ls=msty[i])
    ax.set_xlim(xlm)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xper)
    ax.set_ylim([0,0.5])
# Detrend
# start= time.time()
# dtdsfirst,dsymodall,_,_ = proc.detrend_dim(dsfirst,2)
# print("Detrended in %.2fs" % (time.time()-start))

#%%
# Detrend
start= time.time()
indata = dsfirst.reshape(nlon*nlat,nmon)
okdata,knan,okpts = proc.find_nan(indata,1)
x = np.arange(0,nmon,1)

if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T

# Replace back into dataset
dtdata = np.zeros((nlon*nlat,nmon))*np.nan
dtdata[okpts,:] = okdt
dtdata = dtdata.reshape(nlon,nlat,nmon)
print("Detrended in %.2fs" % (time.time()-start))

# # Plot Seasonal Cycle Removal and Detrended
lonf = -30
latf = 64
tper = np.arange(0,hsstnew.shape[2])
klon,klat = proc.find_latlon(lonf,latf,hlon,hlatnew)
fig,ax = plt.subplots(1,1,figsize=(8,4))
ax.plot(tper,hsstnew[klon,klat,:],color='k',label="raw")
ax.plot(tper,dsfirst[klon,klat,:],color='b',label="deseasonalized")
ax.plot(tper,dtdata[klon,klat,:],color='r',label="deseasonalized,detrended")
ax.set_title("Deseasonalize First")
plt.legend()

hlat = hlatnew.copy()
hsst = dtdata.copy()
#hsst = dsfirst.copy()

# Save data (MONTHLY)
hadname  = "%sHadISST_detrend%i_startyr%s_endyr%s.npz" % (datpath,method,tstart[:4],tend[:4])
np.savez(hadname,**{
    'sst':dtdata,
    'lat':hlatnew,
    'lon':hlon,
    'yr':hyr},allow_pickle=True)

h_amv,h_regr = proc.calc_AMVquick(hsst,hlon,hlat,bbox)

#%% Try Another Calculation Method
#Methods
# 0) Regress anomaly onto global mean
# 1...N) Remove N-degree polynomial



# Get timedim 
nyr = int(hsst.shape[2]/12)
x = np.arange(0,nyr,1)

# First get annual averaged data
hsstann = proc.ann_avg(hsstnew,2)
hsstann = hsstann - hsstann.mean(2)[:,:,None]

# Get nan points
hsstann = hsstann.reshape(360*180,nyr)
okdata,knan,okpts = proc.find_nan(hsstann,1)


if method == 0:
    # Calculate global mean SST
    glomean = okdata.mean(0)
    # Regress back to the original data to get the global component
    beta,b=proc.regress_2d(glomean,okdata)
    # Subtract this from the original data
    okdt = okdata - beta[:,None]

    # Calculate quadratic trend
else: 
    okdt,model = proc.detrend_poly(x,okdata,method)
    
    fig,ax=plt.subplots(1,1)
    ax.scatter(x,okdata[44,:],label='raw')
    ax.plot(x,model[44,:],label='fit')
    ax.scatter(x,okdt[:,44],label='dt')
    ax.set_title("Visualize Detrending Method %i"%method)
    okdt = okdt.T
    

# Replace back into data
hsstdt = np.ones((360*180,nyr)) * np.nan
hsstdt[okpts,:] = okdt
hsstdt = hsstdt.reshape(360,180,nyr)


 


# Use this data to calculate amv
h_amv,h_regr = proc.calc_AMVquick(hsstdt,hlon,hlat,bbox,anndata=True)



#%% Plot AMV

addbox = 1
# Set regions for analysis
bbox_SP = [-60,-15,40,65]
bbox_ST = [-80,-10,20,40]
bbox_TR = [-75,-15,0,20]
bbox_NA = bbox#[-75,20,0,90]
    

#regions = ("SPG","STG","TRO","NAT")
#bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)
regions = ("SPG","STG","TRO","NAT")
bboxes = (bbox_SP,bbox_ST,bbox_TR,bbox_NA)


#% Make AMV Spatial Plots
cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)
#clab = cint
fig,axs = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   
scolors  = ["black","cornflowerblue","crimson","gold"]

# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color=scolors[0],return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")

ax,l2 = viz.plot_box(bbox_ST,ax=ax,color=scolors[2],return_line=True,leglab='STG',linewidth=lwb)
ax,l1 = viz.plot_box(bbox_SP,ax=ax,color=scolors[1],return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=scolors[3],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%% Mini HadiSST AMV Plot NAT

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-1,1.1,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(4,3),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs,fontsize=8)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=12)   


# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
#ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_NATOnly_miniplot.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)


#%%  Make AMV Spatial Plots, but only show the N.Atl Bounding Box

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(6,4),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   


# Add region plots
ax = axs
lwb = 1.5
ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
#ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_NATOnly.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)

#%%  Make AMV Spatial Plots, but only show the NSPG Bounding Box

cmap = cmocean.cm.balance
cmap.set_bad(color='yellow')
cint = np.arange(-.5,.6,0.1)

#clab = cint
fig,axs = plt.subplots(1,1,figsize=(4,3),subplot_kw={'projection':ccrs.PlateCarree()})
plt.style.use('ggplot')
plotbbox = [-100,10,-5,80]

varin = h_regr.T
viz.plot_AMV_spatial(varin,hlon,hlat,plotbbox,cmap,cint=cint,pcolor=0,ax=axs)
axs.set_title("HadISST AMV SST Pattern (%s to %s)" % (startyr,hyr[0,-1]),fontsize=14)   


# Add region plots
ax = axs
lwb = 1.5
#ax,l4 = viz.plot_box(bbox_NA,ax=ax,color='k',return_line=True,leglab='NAT',linewidth=lwb,linestyle="solid")
#ax,l2 = viz.plot_box(bbox_ST,ax=ax,color='r',return_line=True,leglab='STG',linewidth=lwb)
#ax,l1 = viz.plot_box(bbox_SP,ax=ax,color='b',return_line=True,leglab='SPG',linewidth=lwb,linestyle='dashed')
ax,l3 = viz.plot_box(bbox_TR,ax=ax,color=[0,1,0],return_line=True,leglab='TRO',linewidth=lwb,linestyle='dashed')


plt.legend(fancybox=True,shadow=True,loc='upper center')
#leg = ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(0, -0.1),loc='upper left')
#leg(fancybox=True,shadow=True)
#ax.legend([l1,l2,l3,l4],labels=regions,ncol=4,bbox_to_anchor=(-0.1, 1.1),loc='upper left')

outname = '%sHadISST_AMVpattern_%s-%s_dtmethod%i_TROOnly.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)



#%% Plot AMV INdex

xtks = np.arange(0,140,20)
xtkl = np.arange(startyr,startyr+140,20)
fig,ax = plt.subplots(1,1,figsize=(5,1))
ax = viz.plot_AMV(h_amv,ax=ax)
ytks = np.arange(-0.5,0.75,0.25)
    
ax.set_xticks(xtks)
ax.set_xticklabels(xtkl)
ax.set_ylim(-.5,.5)
ax.set_yticks(ytks)
ax.set_title("HadISST AMV Index (%s to %s)" % (startyr,hyr[0,-1]),fontsize=10) 
outname = '%sHadISST_AMVIDX_%s-%s_dtmethod%i.png' % (outpath,startyr,hyr[0,-1],method)
plt.savefig(outname, bbox_inches="tight",dpi=200)
    

#%% UPDATE (MAY 24th 2021, Plots for Generals Presentation)

def plot_AMV_generals(lat,lon,amvpattern,vscale=1):
    """
    Customized AMV Plot for Generals Presentation (for consistent plotting)
    
    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    lon : TYPE
        DESCRIPTION.
    amvpattern : [lon x alt]
        DESCRIPTION.
    vscale : INT
        Amt of times to scale AMV pattern by
    Returns
    -------
    None.
    """
    bbox = [-80,0 ,0,65]
    
    # Set up plot params
    plt.style.use('default')
    cmap = cmocean.cm.balance
    cints = np.arange(-.55,.60,.05)
    cintslb = np.arange(-.50,.6,.1)
    
    # Make the plot
    fig,ax = plt.subplots(1,1,figsize=(5,5),subplot_kw={"projection":ccrs.PlateCarree()})
    ax,cb = viz.plot_AMV_spatial(amvpattern.T*vscale,lon,lat,bbox,cmap,cint=cints,ax=ax,fmt="%.2f",returncbar=True,
                                 fontsize=8)
    cb.set_ticks(cintslb)
    return fig,ax,cb

fig,ax,cb = plot_AMV_generals(hlat,hlon,h_regr)
ax.set_title("AMV Pattern (HadISST; 1900 to 2018) \n Contour Interval: 0.05 $\degree C / \sigma_{AMV}$")
plt.savefig(outpath+"HadISST_AMV_Spatial_Pattern_%i_to_2018_detrend%i.png"%(startyr,method),bbox_inches='tight')

#%% Try Different Detrending Methods
# ** Make sure you have "okdata" from above

#methods = [None,0,1,2,3,4]
#mnames  = ["No Detrend","Global","Linear","Quadratic","Cubic","4th-Order"]
methods = [0,1,2,3,4,5]
mnames  = ["Global","Linear","Quadratic","Cubic","4th-Order","5th-Order"]

amvpats = []
amvids  = []
for m in tqdm(range(len(methods))):
    method = methods[m]
    
    if method is None:
        okdt = okdata.copy() #- okdata.mean(1)[:,None] # Just anomalize
        
    elif method == 0:
        
        # Calculate global mean SST
        glomean = okdata.mean(0)
        
        # Regress back to the original data to get the global component
        beta,b=proc.regress_2d(glomean,okdata)
        
        # Subtract this from the original data
        okdt = okdata - beta[:,None]
    
    else: # Remove N-order polynomial
    
        okdt,model = proc.detrend_poly(x,okdata,method)
        fig,ax=plt.subplots(1,1)
        ax.scatter(x,okdata[44,:],label='raw')
        ax.plot(x,model[44,:],label='fit')
        ax.scatter(x,okdt[:,44],label='dt')
        ax.set_title("Visualize Detrending Method %i"%method)
        okdt = okdt.T
    

    # Replace back into data
    hsstdt = np.ones((360*180,nyr)) * np.nan
    hsstdt[okpts,:] = okdt
    hsstdt = hsstdt.reshape(360,180,nyr)
    
    # Use this data to calculate amv
    h_amv,h_regr = proc.calc_AMVquick(hsstdt,hlon,hlat,bbox,anndata=True,dropedge=5)
    amvpats.append(h_regr)
    amvids.append(h_amv)

#%% Load limopt
ds= xr.open_dataset("/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/01_Data/lim-opt/HadISST-AMO+PDO-LIMopt.nc")
amo = ds.AMO.values


zlat   = ds.lat.values
ztimes = ds.time.values
zpat = ds.SSTAMO.values
lon360 = ds.lon.values
lon180,zpat180 = proc.lon360to180(lon360,zpat.T)
zpat180[np.where(np.abs(zpat180) < 1e-10)] = np.nan


#%% Plot the AMV Patterns

clim = .5 #0.025
cstp = 0.025
cmult = 2
cint = np.arange(-clim,clim+cstp,cstp)
cl_int = np.arange(-clim,clim+cstp*cmult,cstp*cmult)



bboxplot = [-80,0 ,0,55]
fig,axs = plt.subplots(2,3,subplot_kw={'projection':ccrs.PlateCarree()},figsize=(12,6))

for m in tqdm(range(len(methods))):
    
    
    
    blb = [0,0,0,0]
    if m == 0:
        blb = [1,0,0,0]
    if m == 3:
        blb = [1,0,0,1]
    if m >3:
        blb = [0,0,0,1]
    
    ax = axs.flatten()[m]
    ax = viz.add_coast_grid(ax,bbox=bboxplot,blabels=blb)
    
    
    if m == 0: # Plot LimOpt
        cf = ax.contourf(lon180,zlat,zpat180.T*np.std(amo),levels=cint,cmap=cmocean.cm.balance)
        cl = ax.contour(lon180,zlat,zpat180.T*np.std(amo),levels=cl_int,colors='k',linewidths=0.5)
        ax.set_title("LIMopt" + "(var = %.4f)"%(np.var(amo)))
    else:
        cf = ax.contourf(hlon,hlat,amvpats[m].T,levels=cint,cmap=cmocean.cm.balance)
        cl = ax.contour(hlon,hlat,amvpats[m].T,levels=cl_int,colors='k',linewidths=0.5)
        ax.set_title(mnames[m] + "(var = %.4f)"%(np.var(amvids[m])))
    ax.clabel(cl,fontsize=8)
cb = fig.colorbar(cf,ax=axs.flatten(),fraction=0.05,orientation='horizontal')
cb.set_label("AMV Pattern for SST; Contour Interval=%.3f ($\degree C \sigma_{AMV}^{-1}$)"%cstp)
plt.suptitle("Testing Detrending Methods for AMV (HadISST %i to %i)"%(startyr,2018))
plt.savefig("%sAMV_Detrending_Test_HadISST_%ito2018.png"%(outpath,startyr),dpi=150,bbox_inches='tight')




#%% Plot the AMV indices

plotnums2 = np.arange(5,118-7)

xtks = np.arange(0,130,10)
xtkl = np.arange(startyr,startyr+130,10)
ytks = np.arange(-0.5,0.75,0.25)

mcols = ["k","blue","red","magenta","cyan","limegreen"]
msty  = ["solid","dashed","dotted","dashed","solid","dashdot"]

fig,ax = plt.subplots(1,1,figsize=(8,3))

for m in tqdm(range(len(methods))):
    if m == 0:
        ax.plot(plotnums2,amo,label="LIMopt",color=mcols[m],ls=msty[m])
    ln = ax.plot(amvids[m],label=mnames[m],color=mcols[m],ls=msty[m])
    #ax = viz.plot_AMV(amvids[m],ax=ax)

ax.legend(ncol=4)
ax.axhline(0,color="k",ls='dotted')

ax.set_xticks(xtks)
ax.set_xticklabels(xtkl)
ax.set_ylim(-.5,.5)
ax.set_yticks(ytks)
ax.grid(True,ls='dotted')
plt.savefig("%sAMV_Detrending_Test_HadISST_%ito2018.png"%(outpath,startyr),dpi=150,bbox_inches='tight')



    