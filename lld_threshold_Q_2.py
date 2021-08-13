"""
Calculates the loss of Landau damping threshold using effective impedance
As a function of Q_2
Compares with MELODY results in directory '../LHC/results_threshold_bbr_eff_imp/n_m97'
"""
import glob
import numpy as np
import matplotlib
# matplotlib.use('Agg') # For remote connection without graphics
import matplotlib.pyplot as plt
import mpmath
import time
from effective_impedance_funcs import *
from effective_impedance_params import *

T = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M_%S', T)
start_time = time.time()
print("Timer starts now")

# HOM impedance
Q_2_array = np.linspace(1,500,50)
# Q_2_array = np.array([1000])
f_r2 = 0.3*f_r
hom = 1
print(f'hom = {hom}')

for adaptive_k_eff in [True,False]:
    print(f'Adaptive k_eff {adaptive_k_eff}')   
    k_eff_bbr = int(f_r/f_0)
    if adaptive_k_eff:
        k_top = int(1.1*f_r/f_0) # Slightly wider range allowing finding of zero crossing at f_r
        pcolor = 'black'
        plabel = 'Adaptive $k_{eff}$'
    else:
        k_top = k_eff_bbr
        pcolor = 'red'
        plabel = 'Fixed $k_{eff}$'

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
    xi_threshold_array = []
    effective_impedance_array = []

    # Obtain MELODY results
    files = np.array(glob.glob(f'../LHC/results_threshold_bbr_eff_imp/n_m97/lhc_bbr_ImZn0.07_kmax20.0_nrh10.0_phimax1.30_exact_mu2.0_VRF6.0MV_Q_hom*_nr_hom{int(f_r2/f_0)}_f_hom1.0_threshold.txt'))

    melody_Q_2 = np.array([])
    melody_xi_down = np.array([])
    melody_xi_up = np.array([])

    for f in files:
        melody_Q_2 = np.append(melody_Q_2,int(f.split('Q_hom',1)[1].split('_nr_hom',1)[0]))
        table = np.loadtxt(f)
        melody_xi_down = np.append(melody_xi_down,table[1,0])
        melody_xi_up = np.append(melody_xi_up,table[0,0])
    melody_xi = np.mean([melody_xi_down,melody_xi_up],axis=0)

    for Q_2 in Q_2_array:
        impedance_model = np.imag(broadband_resonator_impedance+hom*resonator_impedance(k,Q_2,f_r2))

        if adaptive_k_eff:
            # -- k_eff at last zero crossing of impedance model --
            zero_crossing_index = np.where(np.diff(np.sign(impedance_model/k)))[0][-1]
            k_bool = np.array(np.abs(k)<k[zero_crossing_index])
            k_max = k[zero_crossing_index]
        else:
            k_bool = np.array([True])
            k_max = k_full[-1]

        # Obtain (ImZ/k)_eff
        effective_impedance = np.real_if_close(get_effective_impedance(impedance_model,mu,phi_max,k,k_bool))
        print(f'(ImZ/k)_eff = {effective_impedance}')

        y_max = k_max*phi_max/h
        xi_threshold = get_xi_threshold(mu,y_max,effective_impedance,phi_max)
        print(f'xi_th = {xi_threshold} at Q_2 = {Q_2}, f_r,2/f_r={f_r2/f_r}')
        xi_threshold_array.append(xi_threshold)
        effective_impedance_array.append(effective_impedance)
    plt.plot(Q_2_array,xi_threshold_array,color=pcolor,label=plabel)

savetext = False
plot = True

if plot:
    plt.scatter(melody_Q_2,melody_xi)
    plt.xlabel('$Q_2$')
    plt.ylabel('$\zeta_\\mathrm{th}$')
    plt.legend()
    plt.title(f'$f_{{r,2}}/f_r={f_r2/f_r}$')
    plt.ylim(0,0.015)
    # plt.savefig(f'Figures/MELODY_comparison/lxplus/comparison_{timestamp}.png')
    plt.show()

if savetext:
    with open('lld_threshold.txt','a') as file:
        file.write(f'\nadaptive k_eff {adaptive_k_eff}, hom {hom}, f_r,2/f_r {f_r2/f_r}, Q_2 {Q_2}: xi_th = {xi_threshold}')

print("Process finished --- %s seconds ---" % (time.time() - start_time))










