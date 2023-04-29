import sys,os
import argparse
import numpy as np
from cobaya.run import run 
import pandas as pd
import scipy
import scipy.stats as stats
from mpi4py import MPI


##################################################
# Start and stop refer to the redshifts of the 
# anchor points. The free pivots will be spaced
# between tese points
##################################################
def rbf_kernel(var, length):
    def k(t1, t2):
        exponent = -scipy.spatial.distance.cdist(t1, t2, 'sqeuclidean')/(2*length**2)
        return var**2*np.exp(exponent)
    return k

def create_cov(k, points):
    X = np.expand_dims(points, 1)
    return k(X, X)

def condition(k, X, Y):
    xvalues = Y[:,0]
    yvalues = Y[:,1]
    mat = create_cov(k, np.hstack([X, xvalues]))
    sig_XX = mat[:len(X), :len(X)]
    sig_YY = mat[len(X):, len(X):]
    sig_XY = mat[:len(X), len(X):]
    sig_YX = sig_XY.T
    mean = np.einsum("ij,jk,k->i", sig_XY, np.linalg.inv(sig_YY), yvalues)
    cov = sig_XX - np.einsum("ij,jk,kl->il", sig_XY, np.linalg.inv(sig_YY), sig_YX)

    return mean,cov

def create_pivot_redshifts(N, start, stop):
    pivots = np.linspace(start=start, stop=stop, num=N+2)
    free_pivots = pivots[1:-1]

    str_pivots = ["{:.4f}".format(p) for p in pivots]
    str_pivots = ",".join(str_pivots)

    return [pivots, free_pivots, str_pivots]

def create_cobaya_input(pivot_info, scale=1, length=50):
    N_pivots = len(pivot_info[0])
    zmin = pivot_info[0][0]
    zmax = pivot_info[0][-1]

    planck2018_baseline = {"omega_b": 0.02237,
                           "omega_cdm": 0.1200,
                           "n_s": 0.9649,
                           "logA": 3.044,
                           "tau_reio": 0.0544}

    planck2018_sigma = {"omega_b": 0.00015,
                         "omega_cdm": 0.0012,
                         "n_s": 0.0042,
                         "logA": 0.014,
                         "tau_reio": 0.0073}

    rs = 2
    ps = 10

    kernel = rbf_kernel(scale, length)
    fixed_points = np.array([[zmin, 0], [zmax, 0]])
    mu, sig = condition(kernel, pivot_info[0], fixed_points)

    def prior_wrapper(points):
        return stats.multivariate_normal.logpdf(points, mean=np.zeros(sig.shape[0]), cov=sig, allow_singular=True)

    theory_info = {"classy": 
                    {
                     "ignore_obsolete": True,
                     "extra_args":
                        {"xe_pert_type": "control",
                         "lensing": "yes",
                         "start_sources_at_tau_c_over_tau_h": 0.004,
                         "zmin_pert": zmin,
                         "zmax_pert": zmax,
                         "xe_pert_num": N_pivots,
                         "xe_control_pivots": pivot_info[2]}}}

    likelihood_info = {"planck_2018_lowl.TT": '',
                       "planck_2018_lowl.EE": '',
                       "planck_2018_highl_plik.TTTEEE_lite": ''}
    
    param_info = [("logA", {"prior": {"dist": "norm", "loc": planck2018_baseline["logA"], "scale": ps*planck2018_sigma["logA"]},
                            "ref": {"dist": "norm", "loc": planck2018_baseline["logA"], "scale": rs*planck2018_sigma["logA"]}, 
                            "latex": r"\ln{10^{10}A_s}",
                            "drop": True}),
                  ("ln10^{10}A_s", {"value": "lambda logA: logA",
                                  "derived": False}),
                  ("n_s", {"prior": {"dist": "norm", "loc": planck2018_baseline["n_s"], "scale": ps*planck2018_sigma["n_s"]},
                           "ref": {"dist": "norm", "loc": planck2018_baseline["n_s"], "scale": rs*planck2018_sigma["n_s"]},
                           "latex": r"n_s"}),
                  ("omega_b", {"prior": {"dist": "norm", "loc": planck2018_baseline["omega_b"], "scale": ps*planck2018_sigma["omega_b"]},
                           "ref": {"dist": "norm", "loc": planck2018_baseline["omega_b"], "scale": rs*planck2018_sigma["omega_b"]},
                           "latex": r"\omega_b"}),
                  ("omega_cdm", {"prior": {"dist": "norm", "loc": planck2018_baseline["omega_cdm"], "scale": ps*planck2018_sigma["omega_cdm"]},
                           "ref": {"dist": "norm", "loc": planck2018_baseline["omega_cdm"], "scale": rs*planck2018_sigma["omega_cdm"]},
                           "latex": r"\omega_{cdm}"}),
                  ("tau_reio", {"prior": {"dist": "norm", "loc": planck2018_baseline["tau_reio"], "scale": ps*planck2018_sigma["tau_reio"]},
                           "ref": {"dist": "norm", "loc": planck2018_baseline["tau_reio"], "scale": rs*planck2018_sigma["tau_reio"]},
                           "latex": r"\tau_{reio}"}),
                  ("h", 0.5), 
                  ("clamp", {"derived": 'lambda A_s, tau_reio: 1e9*A_s*np.exp(-2*tau_reio)',
                             "latex": "10^9 A_\\mathrm{s} e^{-2\tau}"})]

    param_info.append(("q_0", {"value": 0., "drop": True}))

    for i in np.arange(1, len(pivot_info[1])+1):
        #cp_input = ("q_{}".format(i), {"prior": {"dist": "uniform", "min": -10, "max": 10},
        #                               "ref": {"dist": "norm", "loc": 0, "scale": 2},
        #                               "drop": True})
        cp_input = ("q_{}".format(i), {#"ref": {"dist": "norm", "loc": 0, "scale": scale},
                                        "prior": {"min": scale-2, "max": scale+2},
                                        "drop": True})
        param_info.append(cp_input)
    param_info.append(("q_{}".format(len(pivot_info[1])+1), {"value": 0., "drop": True}))
    control_point_names = []
    for i in np.arange(len(pivot_info[0])):
        control_point_names.append("q_{}".format(i))
    control_point_names_str = ",".join(control_point_names)

    function_string = 'lambda {}: ",".join(map(str, [{}]))'.format(control_point_names_str, control_point_names_str)
    param_info.append(("xe_control_points", {"value": function_string, "derived": False}))
    param_info = dict(param_info)

    #prior_info = {"control_point_joint_prior": 'lambda {0}: multivariate_normal.logpdf([{1}], loc=None, scale={2})'.format(control_point_names_str, control_point_names_str, sig)}
    #prior_info = {"control_point_joint_prior": 'prior_wrapper([{}])'.format(control_point_names_str)}
    prior_info = {"control_point_joint_prior": 'lambda {}: stats.multivariate_normal.logpdf([{}], mean={}, cov={}, allow_singular=True)'.format(control_point_names_str, control_point_names_str, np.zeros(len(pivot_info[0])).tolist(), sig.tolist())}

    sampler_info = {"minimize": {"ignore_prior": False}}

    cobaya_info = {"theory": theory_info, "likelihood": likelihood_info, "params": param_info, "prior": prior_info, "sampler": sampler_info, "debug": True}

    return cobaya_info

def main():
    print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--zmax', type=float, required=True)
    parser.add_argument('--zmin', type=float, required=True)
    parser.add_argument('--o', type=str, required=True)
    parser.add_argument('--H0', type=float, required=True)
    parser.add_argument('--var', type=float, required=True)
    parser.add_argument('--length', type=float, required=True)
    args = parser.parse_args()

    N_free_control_point = args.N
    zmin = args.zmin
    zmax = args.zmax
    outfile = args.o
    h = args.H0

    pivot_info = create_pivot_redshifts(N_free_control_point, zmin, zmax)
    input = create_cobaya_input(pivot_info, scale=args.var, length=args.length)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    input["params"]["h"] = h
    _, sampler = run(input)
    comm.Barrier()
    bestfit_point = sampler.products()["minimum"].data
    bestfit_point["h"] = h
    
    if(rank==0):
        bestfit_point.to_csv(outfile, mode='a', index=False, header=False)

    MPI.Finalize()
    sys.exit(0)

if __name__=="__main__":
    main()