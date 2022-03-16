'''
Name: smc_mcmc
Creator: Paul Zivick
Last edited: 3/11/2022

Purpose: Finding the best fit parameters, using the van der Marel et al. 2002 formalism,
for the kinematics of an extended stellar object. In particular this is designed around
the SMC. At a minimum, it takes in a set of data with positions, a list of parameters
to fit for and the corresponding priors, and outputs the best fit values. The initial
reason for this switch from a Jupyter notebook is the ability to use multiprocessing
as running on a single core would probably take ~60 hours.
'''

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
import matplotlib
from pm_func_edr3 import *
from mcmc_funcs import *
from multiprocessing import Pool
import os

if __name__ == '__main__':

    dir = "D:/pzivi/Research/Gurtina_Project/"

    datasource = "allstar_Model2_oldcore"
    data = Table.read(dir + "Data/"+datasource+".fits")

    nwalk = 50
    nsamp = 1500
    ver = "nw" + str(nwalk) + "_ns" + str(nsamp)

    print("adding something new".)

    print("Number of stars in " + datasource + " is " + str(len(data)))

    sub_ind = np.random.choice(len(data), size=100000)
    data=data[sub_ind]

####
   # Create columns to add that contain an average error and correlation for the stars\
   # Probably want to eventually get rid of this bit to standardize it
####

    avg_pmra_error = 0.0796
    avg_pmdec_error = 0.0741
    avg_pmra_pmdec_corr = -0.0361

    pmra_error = np.full((len(data),), avg_pmra_error)
    pmdec_error = np.full((len(data),), avg_pmdec_error)
    pmra_pmdec_corr = np.full((len(data),), avg_pmra_pmdec_corr)


    data.add_column(avg_pmra_error, name='pmra_error')
    data.add_column(avg_pmdec_error, name='pmdec_error')
    data.add_column(avg_pmra_pmdec_corr, name='pmra_pmdec_corr')

####

    # eventually will be a read-in argument
    perm = 'standard'

    priors, nparams, params = get_priors(perm)

####
   # Define some SMC options
   # Will probably get folded into a dictionary at some point
####

    smc_options = np.asarray([[60.6, 13.04, -73.10, 148.0], \
                              [62.8, 16.26, -72.42, 145.6]])

    ra0_ind, dec0_ind, dist0_ind, vsys_ind0 = 0, 1, 2, 3
    zivick21, gaia18 = 0, 1

    option = zivick21

#### Set values for the run, including maximum radius, RA/Dec center of the system,
#### and distance to system (in kpc), options will be listed Ursa Minor/Draco

    cenra = smc_options[option][ra0_ind]
    cendec = smc_options[option][dec0_ind]
    dist = smc_options[option][dist0_ind]
    vsys = smc_options[option][vsys_ind0]

    center = np.asarray([np.deg2rad(cenra), np.deg2rad(cendec)])


####
   # Now do the setup steps for the MCMC run
####

    cov = calc_cov_wcs(data) #calculate covariance

    p0 = np.random.rand(nwalk, nparams)
    p0 = create_initial(p0, priors)

    # setting this to avoid issues with Anaconda packages that already use multiple cores
    os.environ["OMP_NUM_THREADS"] = "1"

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalk, nparams, ln_prob, args=[data, cov, perm, dist, cenra, cendec, vsys], pool=pool)

        print("Running burn-in...")
        state = sampler.run_mcmc(p0, 100, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, nsamp, progress=True)

    chain_test = sampler.get_chain()[:, :, 0].T
    samples = sampler.get_chain(flat=True, discard=500)

    print(samples.shape)

    outfilecore = datasource + "_" + perm

    headerstr = " ".join(params)
    np.savetxt(dir + 'Output/' + outfilecore + "_" + ver + "_mcmc_results.txt", samples, header=headerstr, fmt="%1.5f")

    np.savetxt(dir + 'Output/' + outfilecore + "_" + ver + "_marginalized_density.txt", chain_test, fmt="%1.5f")

    print(
        "Mean acceptance frac  tion: {0:.3f}".format(
            np.mean(sampler.acceptance_fraction)
        )
    )

    print(
        "Mean autocorrelation time: {0:.3f} steps".format(
            np.mean(sampler.get_autocorr_time())
        )
    )
