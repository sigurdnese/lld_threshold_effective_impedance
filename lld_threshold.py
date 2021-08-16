"""
Calculates the LLD threshold for a broadband resonator + hom
"""
import numpy as np
import time
from effective_impedance_funcs import *
from effective_impedance_params import *

def choose_adaptive_mode(array, namespace=globals()):
    """ 
    Obtain name of adaptive k_eff mode in order to create appropriate filename
    """
    arrayname = [name for name in namespace if namespace[name] is array][0]
    return array, arrayname

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

G_diag = make_G_diag(mu,k*phi_max/h,phi_max)

# Set up Z_k
broadband_resonator_impedance = resonator_impedance(k,1,f_r)
Q_2 = 200
f_r2 = 0.4*f_r
hom = 1
impedance_model = np.imag(broadband_resonator_impedance+hom*resonator_impedance(k,Q_2,f_r2))
zero_crossing_indices = np.where(np.diff(np.sign(impedance_model/k)))[0]
n_cumsum = np.imag(numerator_cumsum(impedance_model,mu,phi_max,k,G_diag=G_diag))
cumsum_peak_index = zero_crossing_indices[np.argmax(n_cumsum[zero_crossing_indices])]
last_zero_crossing_index = zero_crossing_indices[-1]
first_zero_crossing_index = zero_crossing_indices[0]
# Choose index for adaptive k_eff mode
adaptive_index, adaptive_mode_name = choose_adaptive_mode(cumsum_peak_index)
k_bool = np.array(np.abs(k)<k[adaptive_index])
k_max = k[adaptive_index]

effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,K_bool=k_bool,G_diag=G_diag))

y_max = k_max*phi_max/h

xi_threshold = get_xi_threshold(mu,y_max,effective_impedance,phi_max)
print(f'xi_th = {xi_threshold} at Q_2 = {Q_2}, f_r,2/f_r={f_r2/f_r} using {adaptive_mode_name}')


print("Process finished --- %s seconds ---" % (time.time() - start_time))










