import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import os
import resource
import subprocess
import pickle
import time
import classy as Class
from mpi4py import MPI
from scipy.stats import qmc
import itertools

def main():

    ######################################################
    #  Read parameters
    ######################################################
    model_list_file = sys.argv[1]

    ll_max = 10000

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

    nfailed = np.zeros(1)
    total_failed = np.zeros(1)
    if rank==0:
        with open(model_list_file, 'rb') as f:
            model_list = pickle.load(f)
        #print(data)
        idx = 0
        while len(model_list) > idx:
            target = next(next_worker)
            if comm.iprobe(target):
                _ = comm.recv(source=target)
                comm.send(model_list[idx], dest=target)
                idx+=1
                is_working[str(r)] = 1
            else:
                is_working[str(r)] = 0

            if all(value == 0 for value in is_working.values()):
                time.sleep(pause)
            else:
                time.sleep(short_pause)
            if(idx%1000==0):
                print("{} have been sent".format(idx))

        for i in np.arange(1, size):
            comm.send("done", dest=i)

    if rank!=0:
        settings = {}
        
        common_settings = {'output' : 'tCl,pCl,lCl',
                   'thermodynamics_verbose': 0,
                   'input_verbose': 0,
                   'lensing': 'yes',
                   'xe_pert_type': 'control',
                   'xe_pert_num': 7,
                   'zmin_pert': 800,
                   'zmax_pert': 1400,
                   'xe_control_pivots': "800.0000,900.0000,1000.0000,1100.0000,1200.0000,1300.0000,1400.0000",
                   'N_ur': 2.0308,
                   'N_ncdm': 1,
                   'm_ncdm': 0.06,
                   'T_ncdm': 0.71611, #1 species of massive neutrinos
                   'l_max_scalars': ll_max,

                  }
        
        ### Increased precision for high multipoles, settings from 2109.04451

        precision_settings = {'accurate_lensing': 1,
                            'k_max_tau0_over_l_max': 15,
                            'perturb_sampling_stepsize': 0.05,
                            'start_sources_at_tau_c_over_tau_h': 0.004}

        settings.update(common_settings)
        settings.update(precision_settings)
        output_spectra = ['tt', 'te', 'ee', 'pp']
        outfiles = {}
        for xx in output_spectra:
            outfiles[xx] = os.path.join('../data/precision_spectra',"{}_spectrum.dat.{}".format(xx, rank))

        while True:
            comm.send("waiting for a model", dest=0)
            model = comm.recv(source=0)
            if type(model).__name__ == 'str': #breaks when receiving "done" signal
                break

            ind_info = model.popitem()

            print("RANK {} has model {}".format(rank, model))
            settings.update(model)

            M = Class.Class()
            M.set(settings)

            try:
                M.compute()
                success=True
            
            except Class.CosmoComputationError as failure_message:
                print("Mode failed")
                print(str(failure_message)+'\n')
                success=False
            
            except Class.CosmoSevereError as critical_message:
                print("Something went wrong when calling CLASS" + str(critical_message))
                success=False

            if success:
                for xx in output_spectra:
                    spectrum = M.lensed_cl(4000)[xx][2:]
                    out_array = np.hstack((ind_info["global_index"], spectrum))
                    with open(outfiles[xx], 'ab') as f:
                        np.savetxt(f, [out_array])
            else:
                nfailed[0]+=1
            
            M.struct_cleanup()
        
    comm.Reduce(nfailed, total_failed, MPI.SUM, 0)
    if(rank==0):
        print("{0}/{1} models succeeded".format(len(model_list)-total_failed[0], len(model_list)))
    comm.Barrier()

    MPI.Finalize()
    sys.exit(0)

if __name__=="__main__":
    main()