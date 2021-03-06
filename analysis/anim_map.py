#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:32:54 2020

@author: gliu
"""

from matplotlib.animation import FuncAnimation
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np

import cmocean
import time


#%% Set Paths
projpath   = "/Users/gliu/Downloads/02_Research/01_Projects/01_AMV/02_stochmod/"
datpath     = projpath + '01_Data/model_output/'
rawpath     = projpath + '01_Data/model_input/'
outpathdat  = datpath + '/proc/'
outpathfig  = projpath + '02_Figures/20200823/'



# Load variables
nyrs    = 1000
funiform= 6
runid   = "006"
fscale  = 10
expid = "%iyr_funiform%i_run%s_fscale%03d" % (nyrs,funiform,runid,fscale)
sst = np.load(datpath+"stoch_output_%s.npy"%(expid),allow_pickle=True).item()
lon = np.load(datpath+"lon.npy")
lat = np.load(datpath+"lat.npy")

randts = np.load(datpath+"stoch_output_1000yr_run006_randts.npy")

#%% Prepare for input intp the animation


# Prepare variables
invar = sst[3].transpose(1,0,2)

# Make a land/ice mask
mask = invar.sum(2)
mask[~np.isnan(mask)] = 1

# Animation parameters
frames = 120 #Indicate number of frames
figsize = (4,3)
vm = [-5,5]
interval = 0.1
bbox = [-80,0,0,80]
fps= 10
savetype="gif"
dpi=100


#%% Try to animate a forcing map (mp4.gif) [INEFFIIENT VERSION]


# Define Figure to create base map for plotting

    

# Draw onto the figure
def draw(lon,lat,invar,frame,vm,add_colorbar):
    ax = plt.gca()
    plotvar = invar[...,frame] # Assume dims [lonxlatxtime]
    pcm     = plt.pcolormesh(lon,lat,plotvar,vmin=vm[0],vmax=vm[1],cmap=cmocean.cm.balance)
    title   = "t = %i" % frame
    ax.set_title(title)
    if add_colorbar==True:
        plt.colorbar(pcm,ax=ax,orientation='horizontal',fraction=0.040,pad=0.05)
    print("\rCompleted frame %i"%frame,end="\r",flush=True)
    return pcm
    

# # Indicate initial conditions
def drawinit():
    return draw(lon,lat,invar,0,vm,add_colorbar=True)

# Indicate other conditions
def animate(frame):
    return draw(lon,lat,invar,frame,vm,add_colorbar=False)

#% Run Animatinon

start = time.time()
fig,ax = make_figure(bbox) # Make the basemap

# Pass figure animator and draw on it
# blit = True, redraw only parts that have changed

ani = animation.FuncAnimation(fig, animate, frames, interval=interval, blit=False, init_func=drawinit, repeat=False)

# Save the animation
if savetype == "mp4":
    ani.save("%stest_anim_sstentrain.mp4"%(outpathfig),writer=animation.FFMpegWriter(fps=fps),dpi=dpi)
    plt.close(fig)
elif savetype == "gif":
    ani.save("%stest_anim_sstentrain.gif"%(outpathfig),writer='imagemagick',fps=fps,dpi=dpi)
    plt.close(fig)

print("Animation completed in %.2fs"%(time.time()-start))


#%%   % Run Animatinon Based on
## https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

def make_figure(bbox):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    
    # Set extent
    ax.set_extent(bbox)
    
    # Add filled coastline
    ax.add_feature(cfeature.LAND,facecolor='k',zorder=10)
    
    # Add Gridlines
    gl = ax.gridlines(draw_labels=True,linewidth=0.75,color='gray',linestyle=':')
    
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LongitudeFormatter(degree_symbol='')
    gl.yformatter = LatitudeFormatter(degree_symbol='')
    gl.xlabel_style={'size':8}
    gl.ylabel_style={'size':8}
    
    return fig,ax

start = time.time()
fig,ax = make_figure(bbox) # Make the basemap
pcm = ax.pcolormesh(lon,lat,invar[...,0],vmin=vm[0],vmax=vm[1],cmap=cmocean.cm.balance)
fig.colorbar(pcm,orientation='horizontal',fraction=0.040,pad=0.05)

def animate(i):
     pcm.set_array(invar[...,i].flatten())
     ax.set_title("t = %i" % i)
     print("\rCompleted frame %i"%i,end="\r",flush=True)
     
anim = FuncAnimation(
    fig, animate, interval=interval, frames=frames, blit=False,)

anim.save('%ssst_test.gif'%outpathfig, writer='imagemagick',fps=fps,dpi=dpi)

# Pass figure animator and draw on it
# blit = True, redraw only parts that have changed

print("Animation completed in %.2fs"%(time.time()-start))
#%%   Make timeseries animation


invar = randts
## https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

t = np.arange(0,len(randts),1)

start = time.time()
fig,ax = plt.subplots(1,1,figsize=(4,1)) # Make the basemap
line = ax.plot(t[0],randts[0],color='r',lw=1.5)
                      

def animate(i):
     line.set_xdata(t[0:i])
     line.set_ydata(randts[0:i])
     print("\rCompleted frame %i"%i,end="\r",flush=True)
     
anim = FuncAnimation(
    fig, animate, interval=interval, frames=frames, blit=False,)

anim.save('%srandts_test.gif'%outpathfig, writer='imagemagick',fps=fps,dpi=dpi)

# Pass figure animator and draw on it
# blit = True, redraw only parts that have changed
print("Animation completed in %.2fs"%(time.time()-start))



fig,ax = plt.subplots(1,1,figsize=(4,2)) # Make the basemap
plt.plot(t,randts,color='r',lw=0.5)
plt.ylabel("Forcing (degC/sec)")
plt.xlabel("Months")
plt.title("Forcing Example N(0,0.3)")
plt.tight_layout()
plt.savefig(outpathfig+"randtsex.png",dpi=200)

#%% Animate forcing map

F = np.load(datpath+"stoch_output_1000yr_run0_funiform0_fscale001_Forcing.npy")


invar = F.transpose(1,0,2)* mask[:,:,None]

start = time.time()
fig,ax = make_figure(bbox) # Make the basemap
pcm = ax.pcolormesh(lon,lat,invar[...,0],vmin=vm[0],vmax=vm[1],cmap=cmocean.cm.balance)
fig.colorbar(pcm,orientation='horizontal',fraction=0.040,pad=0.05)



def animate(i):
     pcm.set_array(invar[...,i].flatten())
     ax.set_title("t = %i" % i)
     print("\rCompleted frame %i"%i,end="\r",flush=True)
     
anim = FuncAnimation(
    fig, animate, interval=interval, frames=frames, blit=False,)

anim.save('%sForcingAnim.gif'%outpathfig, writer='imagemagick',fps=fps,dpi=dpi)

# Pass figure animator and draw on it
# blit = True, redraw only parts that have changed
print("Animation completed in %.2fs"%(time.time()-start))