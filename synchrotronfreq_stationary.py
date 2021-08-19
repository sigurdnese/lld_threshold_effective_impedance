"""
Calculates and plots the syncrotron frequency as a function of the phase deviation,
and the energy of synchrotron oscillations. Also calculates/plots the derivative of the latter.
Saves data for comparison to MELODY in 'melody_syncfreq.py'.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15})
from scipy import integrate
from matplotlib import rc
from effective_impedance_params import *

omega_rf = 2*np.pi*f_rf # rad/s
omega_0 = omega_rf/h

# Phi axis
deltaphi1 =  np.linspace(0,0.8*np.pi,1000)
K = np.zeros_like(deltaphi1)
argument = np.sin(deltaphi1/2)

# Integrand
def k(theta,x):
    return 1/(np.sqrt(1-(x**2)*(np.sin(theta))**2))

for i in range(len(deltaphi1)):
    K[i] = integrate.quad(k,0,np.pi/2,args=(argument[i],))[0]

angularsynchfreq_omega_s0 = (np.pi)/(2*K)
synchfreq_f_s0 = angularsynchfreq_omega_s0

angularsynchfreq_approx = (1-deltaphi1**2/16)
synchfreq_approx = angularsynchfreq_approx

H_c = qV_0/(2*np.pi)*(np.cos(deltaphi1+phi_s0)-np.cos(phi_s0)+deltaphi1*np.sin(phi_s0))
E = 2*np.pi*H_c/qV_0

synchfreq_f_s0_der_E = np.gradient(synchfreq_f_s0,E)

np.savez('fs_E_no_int_eff.npz', fs_noint_arr = synchfreq_f_s0, fs_noint_der_arr = synchfreq_f_s0_der_E, e_noint_arr = E)

plt.plot(deltaphi1,synchfreq_f_s0)
plt.plot(deltaphi1,synchfreq_approx,'--')
plt.xlabel('$\Delta\phi_1$')
plt.ylabel('$f_s/f_{s0}$')
plt.tight_layout()
plt.show()

plt.plot(E,synchfreq_f_s0)
plt.plot(E,1-(1/8)*E)
plt.xlabel('$\mathcal{E}$')
plt.ylabel('$f_s/f_{s0}$')
plt.tight_layout()
plt.show()

plt.plot(E,synchfreq_f_s0_der_E)
plt.xlabel('$\mathcal{E}$')
plt.ylabel('$d(f_s/f_{s0})/d\mathcal{E}$')
plt.axhline(-1/8,linestyle='--')
plt.tight_layout()
plt.show()