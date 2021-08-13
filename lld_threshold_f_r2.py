"""
Calculates the loss of Landau damping threshold using effective impedance
As a function of f_r2
Compares with MELODY results in directory '../LHC/results_threshold_bbr_eff_imp/n_m97'
"""
import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg') # For remote connection without graphics
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15})
import mpmath
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

# HOM impedance
Q_2_array = np.array([10])
f_r2_array = np.linspace(0,1*f_r,50)
f_hom = 1
print(f'hom = {f_hom}')

# Obtain MELODY results
files = np.array(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_ImZn0.07_kmax20.0_nrh10.0_phimax1.30_exact_mu2.0_VRF6.0MV_Q_hom{Q_2_array[0]}_*f_hom1.0_threshold.txt'))

melody_f_r2 = np.array([])
melody_xi_down = np.array([])
melody_xi_up = np.array([])

for f in files:
    melody_f_r2 = np.append(melody_f_r2,int(f.split('nr_hom',1)[1].split('_f_hom',1)[0])*f_0)
    table = np.loadtxt(f)
    melody_xi_down = np.append(melody_xi_down,table[1,0])
    melody_xi_up = np.append(melody_xi_up,table[0,0])

melody_xi_down = melody_xi_down[melody_f_r2.argsort()]
melody_xi_up = melody_xi_up[melody_f_r2.argsort()]
melody_f_r2.sort()

melody_xi = np.mean([melody_xi_down,melody_xi_up],axis=0)

fig, ax = plt.subplots(1,1)

ax.scatter(melody_f_r2/f_r,melody_xi,color='C0')

effective_impedance_fixed = []
effective_impedance_adaptive = []
xi_threshold_fixed = []
xi_threshold_adaptive = []
for f_r2 in f_r2_array:
    for adaptive_k_eff in [True,False]:
        print(f'Adaptive k_eff {adaptive_k_eff}')   
        k_eff_bbr = int(f_r/f_0)
        if adaptive_k_eff:
            k_top = int(1.1*f_r/f_0) # Slightly wider range allowing finding of zero crossing at f_r
        else:
            k_top = int(1*k_eff_bbr)

        k_full = np.arange(-k_top,k_top+1,1)
        k_full = np.delete(k_full,np.argwhere(k_full==0))

        k_sampled = np.arange(-k_top,k_top+1,10)
        k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

        k_half = np.arange(1,k_top+1,1)

        k_half_sampled = np.arange(1,k_top+1,10)

        # Choose k list
        k = k_half_sampled

        G_diag = make_G_diag(mu,k*phi_max/h,phi_max)

        # Set up Z_k
        broadband_resonator_impedance = resonator_impedance(k,1,f_r)

        for Q_2 in Q_2_array:
            impedance_model = np.imag(broadband_resonator_impedance+f_hom*resonator_impedance(k,Q_2,f_r2))

            if adaptive_k_eff:
                zero_crossing_indices = np.where(np.diff(np.sign(impedance_model/k)))[0]
                n_cumsum = np.imag(numerator_cumsum(impedance_model,mu,phi_max,k,G_diag=G_diag))
                cumsum_peak_index = zero_crossing_indices[np.argmax(n_cumsum[zero_crossing_indices])]
                last_zero_crossing_index = zero_crossing_indices[-1]
                first_zero_crossing_index = zero_crossing_indices[0]
                # Choose index for adaptive k_eff mode
                adaptive_index, adaptive_mode_name = choose_adaptive_mode(cumsum_peak_index)
                k_bool = np.array(np.abs(k)<k[adaptive_index])
                k_max = k[adaptive_index]
            else:
                k_bool = np.array([True])
                k_max = k_full[-1]

            # Obtain (ImZ/k)_eff
            effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,k_bool,G_diag=np.imag(G_diag)))
            print(f'(ImZ/k)_eff = {effective_impedance}')

            y_max = k_max*phi_max/h
            xi_threshold = get_xi_threshold(mu,y_max,effective_impedance,phi_max)
            print(f'xi_th = {xi_threshold} at Q_2 = {Q_2}, f_r,2/f_r={f_r2/f_r}')
            if adaptive_k_eff:
                effective_impedance_adaptive.append(effective_impedance)
                xi_threshold_adaptive.append(xi_threshold)
            else:
                effective_impedance_fixed.append(effective_impedance)
                xi_threshold_fixed.append(xi_threshold)

ax.plot(f_r2_array/f_r,xi_threshold_fixed,color='red',label=f'Fixed $k_{{eff}}$')
ax.plot(f_r2_array/f_r,xi_threshold_adaptive,color='black',label='Adaptive $k_{eff}$')
ax.set_xlabel('$f_{r,2}/f_r$')
ax.set_ylabel('$\zeta_\\mathrm{th}$')
ax.set_ylim(0,1.6e-2)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.legend()
ax.set_title(f'$Q_2={Q_2_array[0]}$, f_hom={f_hom}')

plt.tight_layout()
# plt.savefig(f'Figures/MELODY_comparison/lxplus/comparison_{adaptive_mode_name}_{timestamp}.pdf')
plt.show()

print("Process finished --- %s seconds ---" % (time.time() - start_time))