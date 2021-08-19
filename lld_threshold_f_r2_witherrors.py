"""
Calculates the loss of Landau damping threshold using effective impedance
As a function of f_r2
Shows relative error wrt MELODY results
Compares with MELODY results in directory '../LHC/results_threshold_bbr_eff_imp/n_m97'
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
Q_2_array = np.array([200])
f_hom = 1
print(f'hom = {f_hom}')

# Obtain MELODY results
files1 = np.array(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_ImZn0.07_kmax20.0_nrh10.0_phimax1.30_exact_mu2.0_VRF6.0MV_Q_hom{Q_2_array[0]}_*f_hom1.0_threshold.txt')) 
files2 = np.array(glob.glob(f'/afs/cern.ch/work/s/snese/cluster/results/results_threshold_bbr_eff_imp/n_m142/lhc_bbr_ImZn0.07_kmax20_nrh10_phimax2.00_exact_mu2.0_VRF6.0MV_Q_hom{Q_2_array[0]:.1f}_*f_hom1.0_threshold.txt')) # Obtained by cluster

for files in [files2]:
    phi_max = float(files[0].split('phimax',1)[1].split('_exact',1)[0])
    print(phi_max)
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
    f_r2_array = melody_f_r2 # Same x-axis as MELODY in order to calculate errors

    fig, ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(melody_f_r2/f_r,melody_xi,color='C0')

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

            # ---- Calculating effective impedance ----

            k_full = np.arange(-k_top,k_top+1,1)
            k_full = np.delete(k_full,np.argwhere(k_full==0))

            k_sampled = np.arange(-k_top,k_top+1,10)
            k_sampled = np.delete(k_sampled,np.argwhere(k_sampled==0))

            k_half = np.arange(1,k_top+1,1)

            k_half_sampled = np.arange(1,k_top+1,10)

            # Choose k list
            k = k_half_sampled

            # Set up Z_k
            broadband_resonator_impedance = resonator_impedance(k,1,f_r)

            for Q_2 in Q_2_array:
                impedance_model = np.imag(broadband_resonator_impedance+f_hom*resonator_impedance(k,Q_2,f_r2))

                if adaptive_k_eff:
                    zero_crossing_indices = np.where(np.diff(np.sign(impedance_model/k)))[0]
                    n_cumsum = np.imag(numerator_cumsum(impedance_model,mu,phi_max,k))
                    cumsum_peak_index = np.argmax(n_cumsum)
                    last_zero_crossing_index = zero_crossing_indices[-1]
                    first_zero_crossing_index = zero_crossing_indices[0]
                    # Choose index for adaptive k_eff mode
                    adaptive_index, adaptive_mode_name = choose_adaptive_mode(cumsum_peak_index)
                    k_bool = np.array(np.abs(k)<k[adaptive_index])
                    k_max = k[adaptive_index]
                else:
                    k_bool = np.array([True])
                    k_max = k_full[-1]

                # (ImZ/k)_eff
                effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,k_bool))
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

    xi_threshold_fixed = np.array(xi_threshold_fixed)
    errors_fixed = (xi_threshold_fixed-melody_xi)/melody_xi
    xi_threshold_adaptive = np.array(xi_threshold_adaptive)
    errors_adaptive = (xi_threshold_adaptive-melody_xi)/melody_xi

    np.savetxt(f'errors/errors_fixed_phimax{phi_max}_Q_2_{Q_2_array[0]}_f_hom_{f_hom}_lenf_r2_{len(f_r2_array)}.txt',np.column_stack([f_r2_array/f_r,errors_fixed]))
    np.savetxt(f'errors/errors_adaptive_phimax{phi_max}_{adaptive_mode_name}_Q_2_{Q_2_array[0]}_f_hom_{f_hom}_lenf_r2_{len(f_r2_array)}.txt',np.column_stack([f_r2_array/f_r,errors_adaptive]))

    print(errors_fixed)
    print(errors_adaptive)
    max_error_index_fixed = np.argmax(np.abs(errors_fixed))
    max_error_index_adaptive = np.argmax(np.abs(errors_adaptive))
    print(f'Fixed k_eff: Max error {errors_fixed[max_error_index_fixed]} at f_r2/f_r = {melody_f_r2[max_error_index_fixed]/f_r}')
    print(f'Adaptive k_eff ({adaptive_mode_name}): Max error {errors_adaptive[max_error_index_adaptive]} at f_r2/f_r = {melody_f_r2[max_error_index_adaptive]/f_r}')

    ax[0].plot(f_r2_array/f_r,xi_threshold_adaptive,'o--',color='black',label='Adaptive $k_{eff}$')
    ax[0].plot(f_r2_array/f_r,xi_threshold_fixed,'o--',color='red',label=f'Fixed $k_{{eff}}$')
    ax[0].set_xlabel('$f_{r,2}/f_r$')
    ax[0].set_ylabel('$\zeta_\\mathrm{th}$')
    ax[0].set_ylim(bottom=0)
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax[0].set_title(f'$Q_2={Q_2_array[0]}$, $\phi_\\mathrm{{max}}={phi_max}$')

    ax[1].plot(f_r2_array/f_r,errors_adaptive,color='black',label='Adaptive $k_{eff}$')
    ax[1].plot(f_r2_array/f_r,errors_fixed,color='red',label=f'Fixed $k_{{eff}}$')
    ax[1].set_ylim(-2,2)
    ax[1].set_title('Relative error')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'Figures/MELODY_comparison/lxplus/comparison_errors_{timestamp}.pdf')
    # plt.show()

print("Process finished --- %s seconds ---" % (time.time() - start_time))