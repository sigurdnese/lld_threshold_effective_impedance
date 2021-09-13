"""
Calculates the LLD threshold for SPS impedance model as function of phi_max
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

# k_full = np.arange(-k_eff,k_eff+1,1)
# k_full = np.delete(k_full,np.argwhere(k_full==0))

# k_sampled = np.arange(-k_eff,k_eff+1,10)
# k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

# k_half = np.arange(1,k_eff+1,1)

# k_half_sampled = np.arange(1,k_eff+1,10)

# # Choose k list
# k = k_half_sampled
# y = k*phi_max/h
# print(k)

# Import MELODY results
files = np.array(glob.glob(f'/afs/cern.ch/work/s/snese/sps_cluster/results/TODO_threshold.txt'))

melody_xi_down = np.array([])
melody_xi_up = np.array([])

for f in files:
    phi_max_melody = float(files[0].split('phimax',1)[1].split('_exact',1)[0])
    table = np.loadtxt(f)
    melody_xi_down = np.append(melody_xi_down,table[1,0])
    melody_xi_up = np.append(melody_xi_up,table[0,0])

melody_xi_down = melody_xi_down[melody_f_r2.argsort()]
melody_xi_up = melody_xi_up[melody_f_r2.argsort()]
melody_f_r2.sort()
melody_xi = np.mean([melody_xi_down,melody_xi_up],axis=0)
ax.scatter(phi_max_melody,melody_xi)

# Import SPS impedance model
impedance_file = np.load('sps_impedance.npz')
k = impedance_file['freq']/f_0
k = k[1:]
impedance_model= impedance_file['ImZ'][1:]

G_diag = make_G_diag(mu,k*phi_max/h,phi_max)

zero_crossing_indices = np.where(np.diff(np.sign(impedance_model/k)))[0]
xi_threshold_array = []
for phi_max in phi_max_array:
    n_cumsum = np.imag(numerator_cumsum(impedance_model,mu,phi_max,k,G_diag=G_diag))
    cumsum_peak_index = np.argmax(n_cumsum)
    # Choose index for adaptive k_eff mode
    adaptive_index, adaptive_mode_name = choose_adaptive_mode(cumsum_peak_index)
    k_bool = np.array(np.abs(k)<k[adaptive_index])
    k_max = k[adaptive_index]

    effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,K_bool=k_bool,G_diag=G_diag))

    y_max = k_max*phi_max/h

    xi_threshold = get_xi_threshold(mu,y_max,effective_impedance,phi_max)
    xi_threshold_array.append(xi_threshold)
    print(f'xi_th = {xi_threshold} at phi_max={phi_max} using {adaptive_mode_name}')

print("Process finished --- %s seconds ---" % (time.time() - start_time))

plt.plot(phi_max_array,xi_threshold_array,'--o')
plt.show()









