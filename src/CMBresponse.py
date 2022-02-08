import classy as Class
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck15 as cosmo

######################################################
#  Setting fiducial parameters
######################################################
h = cosmo.h
om_dm = cosmo.Odm0*h*h
om_b = cosmo.Ob0*h*h
sigma8 = .8159
tau = .078
#tau = 0.05430842
ns = .9667


######################################################
#  Setting up plots
######################################################

font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.mathtext.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]

######################################################
#  Finding where pivot points will be, as well as 
#  the indices of the pivots closest to what we want
######################################################

N = 1 #Number of basis functions
zmin_pert = 500 #min and max redshift of perturbations
zmax_pert = 1700

ll_max = 2500

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

z_of_pert = np.array([900, 1100, 1300]) # redshifts we will introduce perturbations at
#z_of_pert = np.array([900, 1100]) # redshifts we will introduce perturbations at


common_settings = {'output' : 'tCl, lCl',
                   # LambdaCDM parameters
                   'h':h,
                   'omega_b':om_b,
                   'omega_cdm':om_dm,
                   #'A_s':2.100549e-09,
                   'sigma8': sigma8,
                   'n_s':ns,
                   #'z_reio': 8.5,
                   'tau_reio': tau,
                   'thermodynamics_verbose': 0,
                   'lensing': 'yes',
                   #'reio_parametrization': 'reio_none',
                   'recombination': 'RECFAST',
                   'perturb_xe': 'yes',
                   'xe_pert_num': N,
                   'zmin_pert': zmin_pert,
                   'zmax_pert': zmax_pert,
                   #'use_spline_xe_pert': 'yes'
                   }  

print(common_settings)

M = Class.Class()
M.set(common_settings)

#amp = (3./2)*.01
#amplitudes = np.linspace(amp-.1, amp+.1, 21)
amplitudes = np.linspace(-.1, .1, 5)
middle = int((amplitudes.shape[0]-1)/2)

dz = 10.58

actual_zi = []
responses = []
perts = []

ll = np.zeros(ll_max)

run_count = 0
run_total = z_of_pert.shape[0]*amplitudes.shape[0]

print("{0} of {1} runs completed\r".format(run_count, run_total))

for zi in z_of_pert:
    
    actual_zi.append(zi)
    array_of_cls = []
    
    width=dz/2.355
    
    #M.set({'xe_single_width': dz})
    M.set({'xe_single_width': width})

    M.set({'xe_single_zi': zi})

    for qi in amplitudes:
    #Setting up string of perturbation amplitudes
        amp_str = "{}".format(qi)
        M.set({'xe_pert_amps': amp_str}) #sets perturbation
        M.compute() ##Run CLASS

        cls = M.lensed_cl(ll_max)
        ll = cls['ell'][2:]
        array_of_cls.append(cls['tt'][2:])
        if(qi==amplitudes[-1]):
            perts.append((M.get_thermodynamics()['z'], M.get_thermodynamics()['xe_pert']))
        run_count+=1
        print("{0} of {1} runs completed\r".format(run_count, run_total))

    
    array_of_cls = np.stack(array_of_cls)
    fid = array_of_cls[middle]
    derivs = np.gradient(array_of_cls, amplitudes[1]-amplitudes[0], axis=0, edge_order=2)
    response = derivs[middle]/fid
    responses.append(response)

fig, ax = plt.subplots()

ax.plot(ll, responses[0], label="$z_i = {}$".format(actual_zi[0]))
ax.plot(ll, responses[1], label="$z_i = {}$".format(actual_zi[1]))
#ax.plot(ll, responses[2], label="$z_i = {}$".format(actual_zi[2]))
ax.set_xlabel("$\ell$", fontsize=14)
ax.set_ylabel("$\partial_i \ln C_{\ell}^{TT}$", fontsize=14)

plt.legend()
plt.show()
