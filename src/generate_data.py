import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import os
import time
from mpi4py import MPI
from scipy.stats import qmc
import itertools
import classy as Class

######################################################
#  Read parameters
######################################################
param_file = sys.argv[1]
param = SourceFileLoader(param_file, param_file).load_module()
param_ranges = param.param_ranges

Nparams = len(param_ranges.keys())
N = param.N
######################################################
#  Create Latin Hypercube of samples
######################################################
def create_models(N, ranges):
    sampler = qmc.LatinHypercube(d=Nparams, seed=0) # seed for reproducibility 
    #sampler = qmc.LatinHypercube(d=Nparams)
    samples = sampler.random(n=N)
    i=0
    for range in ranges.values():
        samples.T[i] *= range[1] - range[0]
        samples.T[i] += range[0]
        i+=1

    return samples

def create_name_mapping(param_names):
    mapping = {}
    i=0
    for name in param_names:
        mapping[i] = name
        i+=1
    return mapping
######################################################
#  Set up MPI
######################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nworkers = size-1
next_worker = itertools.cycle(np.arange(1, size))
is_working = {}
for r in np.arange(size):
    is_working[str(r)] = False

pause = 0.1
short_pause = 1e-4
######################################################
#  Compute
######################################################


if rank==0:
    name_mapping = create_name_mapping(param_ranges.keys())
else:
    name_mapping = None

name_mapping = comm.bcast(name_mapping, root=0)

if rank==0:
    np.random.seed(1)
    data = create_models(N, param_ranges)
    print(data)
    idx = 0
    while data.shape[0] > idx:
        target = next(next_worker)
        if comm.iprobe(target):
            _ = comm.recv(source=target)
            comm.send(data[idx], dest=target)
            idx+=1
            is_working[str(r)] = 1
        else:
            is_working[str(r)] = 0

        if all(value == 0 for value in is_working.values()):
            time.sleep(pause)
        else:
            time.sleep(short_pause)

    for i in np.arange(1, size):
        comm.send("done", dest=i)

if rank!=0:
    common_settings = {'output' : 'tCl,pCl,lCl',
                   'thermodynamics_verbose': 0,
                   'input_verbose': 1,
                   'lensing': 'yes',
                   'xe_pert_type': 'none'
                  }
    pr_cover_tau = 0.004
    ll_max = 2500
    precision_settings = {"start_sources_at_tau_c_over_tau_h": pr_cover_tau}
    M = Class.Class()
    M.set(common_settings)
    M.set(precision_settings)
    model_file = open(os.path.join(param.outdir_root,"models_rank{}.txt".format(rank)), "ab")
    tt_file = open(os.path.join(param.outdir_root,"tt_rank{}.txt".format(rank)), "ab")
    while True:
        comm.send("waiting for a model", dest=0)
        model = comm.recv(source=0)
        if type(model).__name__ == 'str': #breaks when receiving "done" signal
            break
        settings = {}
        for i,param in enumerate(model):
            settings[name_mapping[i]] = model[i]
        M.set(settings)
        try:
            M.compute()
            np.savetxt(model_file, model[np.newaxis])
            np.savetxt(tt_file, M.lensed_cl(ll_max)['tt'][2:][np.newaxis])
        except:
            print("Model {} failed ".format(model))
    model_file.close()
    tt_file.close()
    #done

MPI.Finalize()
sys.exit(0)
