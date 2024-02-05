#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Copied from test_forcing_damping_shift.py

Created on Tue Jan 30 14:17:05 2024

@author: gliu
"""



#%%



#%%



mlds     = [hmxl,]
forcings = [fprimept,]
dampings = [dpt[1],]
nexps    = len(mlds)


hcolors  = ["red","violet","orange"]
hmarkers = ["d","x","o"]
hnames   = ["Vary $F'$ and $\lambda_a$ (Level 3)",
"Vary $F'$, $h$, and $\lambda_a$ (Level 4)",
"Entraining (Level 5)"]

old_index   = False # Set to True for case where T(1) = lbd(1)*T(0) + F(1) instead of F(0) 
half_mode   = False

forcingroll     = 0
dampingroll     = 0
mldroll         = 0

hdenom_lbd_roll = 0 # How to roll h in the denominator of lambda. If 0, it will be the same as mldroll

# Temp fix for when forcing indexing is corrected for nonentraining code only..
# When forcingroll is set to 1, do not roll case for Level4 (vary MLD)
# made for purposes of generating AMS figure for Martha, will remove this when 
# code is corrected
entrainpatch = False 

rollstr      = "froll%i_droll%i_hroll%i_oldindex%i_halfmode%i" % (forcingroll,dampingroll,mldroll,old_index,half_mode)

outputs      = []
fcopy        = []
dcopy        = []
hcopy        = []
for ex in range(nexps):
    
    # Get Forcing/Damping and Roll
    f_in = np.roll(forcings[ex].copy(),forcingroll)
    h_in = np.roll(mlds[ex].copy(),mldroll)
    d_in = np.roll(dampings[ex].copy(),dampingroll)
    
    if half_mode:
        # Then average : T_in = [T(1) + T(roll)]/2
        f_in = (f_in + forcings[ex].copy())/2
        h_in = (h_in + mlds[ex].copy())/2
        d_in = (d_in + dampings[ex].copy())/2
    
    outputs_h = []
    for hconfig in range(3):
        
        # Set up Stochastic Model Input...
        smconfig = {}
        smconfig['eta']     = eta.copy()               # White Noise Forcing
        smconfig['forcing'] = f_in.copy()[None,None,:] # [x x 12] # Forcing Amplitude (W/m2)
        smconfig['damping'] = d_in.copy()[None,None,:] # Damping Amplitude (degC/W/m2)
        
        if hconfig == 0:
            smconfig['h']       = np.ones((1,1,12))*hblt # MLD (meters)
        else:
            smconfig['h']       = h_in.copy()[None,None,:] # MLD (meters)
        
        # --------------------------------------------------------
        # Manual case while functions are being repaired
        if (entrainpatch) and (forcingroll == 1) and (hconfig ==1):
            print("Temporary patch for Level 4")
            
            f_fix = np.roll(forcings[ex].copy(),0)
            h_fix = np.roll(mlds[ex].copy(),-1)
            d_fix = np.roll(dampings[ex].copy(),-1)
            
            smconfig['forcing'] = f_fix.copy()[None,None,:]
            smconfig['h']       = h_fix.copy()[None,None,:]
            smconfig['damping'] = d_fix.copy()[None,None,:]
        # --------------------------------------------------------
        
        
        # Calculate Kprev
        kout,_              = scm.find_kprev(h_in,)
        smconfig['kprev']   = kout[None,None]
        
        
        # Convert units (W/m2 to degC/S)
        if (hdenom_lbd_roll==0) or hconfig == 0:  # Dont do this for SLAB run because h is the same value 
            smconfig['damping']=scm.convert_Wm2(smconfig['damping'],smconfig['h'],dt)[None,None,:]
        elif hconfig !=0: 
            smconfig['damping']=scm.convert_Wm2(smconfig['damping'],np.roll(mlds[ex].copy(),hdenom_lbd_roll),dt)[None,None,:]
            
        smconfig['forcing']=scm.convert_Wm2(smconfig['forcing'],smconfig['h'],dt)
        
        
        # Run Stochastic Model (No entrain) ---------------------------------------
        # Make Forcing
        smconfig['Fprime']= np.tile(smconfig['forcing'],nyrs) * smconfig['eta'][None,None,:]
        
        # Do Integration
        if hconfig < 2:
            output = scm.integrate_noentrain(smconfig['damping'],smconfig['Fprime'],
                                             debug=True,old_index=old_index)
        else:
            output = scm.integrate_entrain(smconfig['h'],smconfig['kprev'],smconfig['damping'],
                                           smconfig['Fprime'],debug=True,return_dict=True,old_index=old_index)
        
        # -------------------------------------------------------------------------
        outputs_h.append(output)
        
    outputs.append(outputs_h)
    fcopy.append(f_in.copy())
    dcopy.append(d_in.copy())
    hcopy.append(h_in.copy())
#%
outputs_in = outputs[0] # Drop ex since I'm only using 1 value for now
# % Calculate some diagnostics
ssts      = [outputs_in[0][0],outputs_in[1][0],outputs_in[2]['T']]#[o[0].squeeze() for o in outputs]
ssts      = [s.squeeze() for s in ssts]
tsmetrics = scm.compute_sm_metrics(ssts)


monvars = tsmetrics['monvars']
acs_all = tsmetrics['acfs']