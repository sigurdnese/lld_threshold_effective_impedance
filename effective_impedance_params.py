"""
Parameters used in the calculation of effective impedance and the LLD threshold
"""
import numpy as np

# Parameters
f_rf = 400.79e6 # Hz
f_c = 1*f_rf # Cutoff frequency
h = 35640 
f_0 = f_rf/h
mu = 2
V_0 = 6e6 # V
qV_0 = V_0*1 # eV
phi_s0 = np.pi # Synchronous phase
phi_max = 1.3
gamma_tr = 55.76
E_0 = 0.45e12 # ev

# Broadband resonator impedance
Q = 1
f_r = 10*f_rf
ImZ_over_k = 0.07 # Ohm
