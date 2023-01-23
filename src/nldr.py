import numpy as np
import os, sys
import subprocess
#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

def call_minimizer(h, outfile, N_free_control_point, zmin, zmax, var, length):
    command_string = "mpirun -n 4 python minimizer.py --N {0} --zmin {1} --zmax {2} --o {3} --H0 {4} --var {5} --length {6}".format(N_free_control_point, zmin, zmax, outfile, h, var, length)
    command = command_string.split(" ")
    subproc = subprocess.run(command)

def main():
    use_pool = False
    basedir = "/Users/gabe/projects/recombination"
    filename = "nldr_result_multi.csv"
    outfile = os.path.join(basedir, "data/{}".format(filename))

    starting_at = 0
    if os.path.exists(outfile):
        starting_at = sum(1 for line in open(outfile))

    h_list = np.linspace(0.5, 0.9, 40)
    N = 17
    zmin = 650
    zmax = 1400
    
    #parameters for CP priors
    var = 1.0
    length = 100

    if(use_pool):
        pool_size = 2
        pool = Pool(pool_size)

    for h in h_list[starting_at:]:
        if(use_pool):
            pool.apply_async(call_minimizer, (h, outfile, N, zmin, zmax, var, length))
        else:
            call_minimizer(h, outfile, N, zmin, zmax, var, length)
    
    if(use_pool):
        pool.close()
        pool.join()

if __name__=="__main__":
    main()
    