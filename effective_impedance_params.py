"""
Parameters used in the calculation of effective impedance and the LLD threshold
"""
import numpy as np

# LHC
# f_rf = 400.79e6 # Hz
# h = 35640 
# V_0 = 6e6 # V
# gamma_tr = 55.76
# E_0 = 0.45e12 # ev

#---SPS---
f_rf = 200394434.91980886
h = 4620
V_0 = 7.202e6
gamma_tr = 17.951
E_0 = 451150975677.2243

qV_0 = V_0*1 # eV
phi_s0 = np.pi # Synchronous phase
f_c = 1*f_rf # Cutoff frequency
f_0 = f_rf/h
mu = 2
phi_max_array = np.linspace(0.5,1.5,20)
phi_max = 1.3

# Broadband resonator impedance
Q = 1
f_r = 10*f_rf
ImZ_over_k = 0.07 # Ohm