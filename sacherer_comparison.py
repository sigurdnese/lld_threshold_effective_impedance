import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from effective_impedance_params import *
from effective_impedance_funcs import *
plt.rcParams.update({'font.size':15})

def choose_adaptive_mode(array, namespace=globals()):
    """ 
    Obtain name of adaptive k_eff mode in order to create appropriate filename
    """
    arrayname = [name for name in namespace if namespace[name] is array][0]
    return array, arrayname

# Approximate line density distribution from paper
phi = np.linspace(-phi_max,phi_max)
l = (1-phi**2/phi_max**2)**(mu+0.5) #lambda/lambda_0

# Calculate rms
def integrand(Phi):
    return (1-Phi**2/phi_max**2)**(2*mu+1)
rms = (1/(np.sqrt(2*phi_max)))*np.sqrt(integrate.quad(integrand,-phi_max,phi_max))[0]
sigma = rms/(2*np.pi*f_rf)

k_eff = int(f_r/f_0)
k = np.arange(-k_eff,k_eff+1,1)
k = np.delete(k,np.argwhere(k==0))

k_half_sampled = np.arange(1,1.1*k_eff+1,10)

omega_0 = 2*np.pi*f_0
# Sacherer, Gaussian line distribution
get_h_gaussian = lambda k: ((k*omega_0*sigma)**2)*np.exp(-(k*omega_0*sigma)**2)
h_gaussian = get_h_gaussian(k)
H_gaussian = h_gaussian/np.sum(h_gaussian)

# Sacherer, parabolic line distribution
get_h_parabolic = lambda k: 4*(1-np.cos(2*k*phi_max/h))/((4*(k*phi_max/h)**2-4*np.pi**2)**2)
h_parabolic = get_h_parabolic(k)
H_parabolic = h_parabolic/np.sum(h_parabolic)

# plt.plot(k,H_gaussian,label='Gaussian')
# plt.plot(k,H_parabolic,label='Parabolic')
# plt.xlabel('k')
# plt.ylabel('Spectral power density $H_2(k)$')
# plt.legend()
# plt.show()

# Impedance model
hom = 1
Q_2 = 200
f_r2_array = np.linspace(0,f_r,100)
bbr = resonator_impedance(k,1,f_r)
bbr_half_sampled = resonator_impedance(k_half_sampled,1,f_r)
G_diag = make_G_diag(mu,k_half_sampled*phi_max/h,phi_max)

effective_impedance_array = []
chi_array = []
parabolic_zeff_array = []
gaussian_zeff_array = []
for f_r2 in f_r2_array:
    impedance_model = np.imag(bbr+hom*resonator_impedance(k,Q_2,f_r2))
    impedance_model_half_sampled = np.imag(bbr_half_sampled+hom*resonator_impedance(k_half_sampled,Q_2,f_r2))
    zero_crossing_indices = np.where(np.diff(np.sign(impedance_model_half_sampled/k_half_sampled)))[0]
    n_cumsum = np.imag(numerator_cumsum(impedance_model_half_sampled,mu,phi_max,k_half_sampled,G_diag=G_diag))
    # cumsum_peak_index = zero_crossing_indices[np.argmax(n_cumsum[zero_crossing_indices])]
    cumsum_peak_index = np.argmax(n_cumsum)
    # Choose index for adaptive k_eff mode
    adaptive_index, adaptive_mode_name = choose_adaptive_mode(cumsum_peak_index)
    k_max = k_half_sampled[adaptive_index]
    y_max = k_max*phi_max/h
    chi_array.append(chi(mu,y_max))
    # print(f_r2/f_r,zero_crossing_indices,adaptive_index)
    k_bool = np.array(np.abs(k_half_sampled)<k_half_sampled[adaptive_index])
    effective_impedance = np.real_if_close(get_effective_impedance(impedance_model_half_sampled,mu,phi_max,k_half_sampled,K_bool=k_bool,G_diag=G_diag))
    effective_impedance_array.append(effective_impedance)

    parabolic_zeff = np.sum((impedance_model/k)*H_parabolic)
    parabolic_zeff_array.append(parabolic_zeff)
    gaussian_zeff = np.sum((impedance_model/k)*H_gaussian)
    gaussian_zeff_array.append(gaussian_zeff)

parabolic_zeff_array = np.array(parabolic_zeff_array)
gaussian_zeff_array = np.array(gaussian_zeff_array)
effective_impedance_array = np.array(effective_impedance_array)
parabolic_error = (parabolic_zeff_array-effective_impedance_array)/(effective_impedance_array)
gaussian_error = (gaussian_zeff_array-effective_impedance_array)/(effective_impedance_array)

print(f'Parabolic max error: {np.max(parabolic_error)} at {f_r2_array[np.argmax(parabolic_error)]/f_r}')
print(f'Gaussian max error:  {np.max(gaussian_error)} at {f_r2_array[np.argmax(parabolic_error)]/f_r}')


plt.plot(f_r2_array/f_r,effective_impedance_array)
plt.plot(f_r2_array/f_r,parabolic_zeff_array,label='Parabolic Sacherer')
plt.plot(f_r2_array/f_r,gaussian_zeff_array,label='Gaussian Sacherer')
plt.title(f'$Q_2={Q_2}$')
plt.xlabel('$f_{r,2}/f_r$')
plt.ylabel('$\\mathrm{Im}(Z_k/k)_\\mathrm{eff}$')
plt.legend(handletextpad=0.2,frameon=False,loc=4)
plt.tight_layout()
plt.show()

# plt.plot(f_r2_array/f_r,chi_array)
# plt.show()

# plt.plot(f_r2_array/f_r,1/(np.array(effective_impedance_array)*np.array(chi_array)))
# plt.show()

