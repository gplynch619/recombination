import classy as Class
import os
import numpy as np
import time

As = 2.100549e-9
ln10e10As = np.log(1e10*As)

standard_params = {
    "omega_b": 0.0223828,
    "omega_cdm": 0.1201075,
    "n_s": 0.9660499,
    "tau_reio": 0.05430842,
    "ln10^{10}A_s": ln10e10As,
    #"100*theta_s": 1.04,
    "H0": 67.32117
}

common_settings = {'output' : 'tCl,pCl,lCl',
                   'thermodynamics_verbose': 0,
                   'input_verbose': 0,
                   'lensing': 'yes',
                   'xe_pert_type': 'none',
                   'N_ur': 2.0308,
                   'N_ncdm': 1,
                   'm_ncdm': 0.06,
                   'T_ncdm': 0.71611, #1 species of massive neutrinos
                  }

precision_settings= {'accurate_lensing': 1,
                        'k_max_tau0_over_l_max': 25,
                        'perturbations_sampling_stepsize': 0.05,
                        'l_max_scalars': 10000,
                        'non_linear': 'hmcode',
                        'eta_0':0.603,
                        'c_min': 3.13}

class_settings = {}
class_settings.update(common_settings)
class_settings.update(standard_params)
class_settings.update(precision_settings)
M = Class.Class()
tic = time.perf_counter()
M.set(class_settings)
M.compute()
toc = time.perf_counter()

num_omp_threads = int(os.environ["OMP_NUM_THREADS"])

#with open("omp_thread_timings.txt", "a") as f:
#    f.write("{}\t{}\n".format(toc-tic, num_omp_threads))
