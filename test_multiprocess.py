import os
import time
import numpy as np
import emcee
from multiprocessing import Pool
import pm_func_edr3

if __name__ == '__main__':

    os.environ["OMP_NUM_THREADS"] = "1"




    np.random.seed(42)
    initial = np.random.randn(32,5)
    nwalkers, ndim = initial.shape
    nsteps = 100

    sampler = emcee.EnsembleSampler(nwalkers, ndim, pm_func_edr3.log_prob)
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    serial_time = end - start
    print("Serial took {0:.1f} seconds:".format(serial_time))


    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pm_func_edr3.log_prob, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_time = end = start
        print("Multiprocessing took {0:.1f} seconds.".format(multi_time))
        print("{0:.1f} times faster than serial".format(serial_time / multi_time))
