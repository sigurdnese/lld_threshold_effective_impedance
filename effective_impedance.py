"""
Calculates the effective impedance for a broadband resonator + hom
"""
import numpy as np
import time
from effective_impedance_funcs import *
from effective_impedance_params import *

T = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M_%S', T)
start_time = time.time()
print("Timer starts now")

k_eff = int(f_r/f_0)

# ---- Calculating effective impedance ----

k_full = np.arange(-k_eff,k_eff+1,1)
k_full = np.delete(k_full,np.argwhere(k_full==0))

k_sampled = np.arange(-k_eff,k_eff+1,10)
k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

k_half = np.arange(1,k_eff+1,1)

k_half_sampled = np.arange(1,k_eff+1,10)

# Choose k list
k = k_half_sampled
y = k*phi_max/h

# Set up Z_k
broadband_resonator_impedance = resonator_impedance(k,1,f_r)
Q_2 = 1000
f_r2 = 0.4*f_r
hom = 1

effective_impedance = np.real_if_close(get_effective_impedance(np.imag(broadband_resonator_impedance+hom*resonator_impedance(k,Q_2,f_r2)),mu,phi_max,k))
print(f'Effective impedance for Q_2={Q_2}, f_r2/f_r={f_r2/f_r}: {effective_impedance}')


print("Process finished --- %s seconds ---" % (time.time() - start_time))










