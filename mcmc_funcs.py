'''
Name: mcmc_funcs.py
Creator: Paul Zivick
Last Edited: 3/11/2022

Purpose: A collection of functions from the SMC_MCMC_v1 notebook that are part of the MCMC
process and not part of the kinematic formalism (those functions live in pm_func_edr3).
Detailed explanations of each function along with any necessary math can be found in the
original notebook.
'''

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
import matplotlib
from pm_func_edr3 import *


################################################################
#### Function to define parameters and priors. This will eventually
#### be outdated once I build in directionary capabilities like
#### Knut currently uses.

def get_priors(perm):

    if perm == 'standard':
        nparams = 11
        params = ['pmra0', 'pmdec0', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.],
                            [-2., 0.], [-2., 0.], [0., 360.], [-90., 90.], [-1., 2.]])

    elif perm == 'full':
        nparams = 15
        params = ['pmra', 'pmdec', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid', 'ra0', 'dec0', 'dist0', 'vsys']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.], \
                            [-3., 0.], [-3., 0.], [0., 360.], [-90., 90.], [-1., 2.], \
                            [0., 30.], [-75., -65.],[55., 65.], 120., 170.])

    else:
        nparams = 11
        params = ['pmra', 'pmdec', 'thet', 'incl', 'rad0', 'rotvel', 'sigma_pmra', 'sigma_pmdec', \
                  'phi_tid', 'theta_tid', 'vel_tid']
        priors = np.asarray([[-5., 5.], [-5., 5.], [0., 360.], [-90., 90.], [-2., 1.], [-50., 50.],
                            [-3., 0.], [-3., 0.], [0., 360.], [-90., 90.], [-1., 2.]])

    return priors, nparams, params

################################################################

################################################################
#### Function to evenly distribute initial walkers over the full
#### possible prior range

def create_initial(p0, priors):

    for a in range(len(p0[0])):
        p0[:,a] = (p0[:,a] * (priors[a][1] - priors[a][0])) + priors[a][[0]]

    return p0

################################################################

################################################################
#### Function to calculate the covariance matrices for the
#### proper motion data (should maybe be in pm_func_edr3)

def calc_cov_wcs(data):

    cov = np.zeros((len(data), 2, 2))
    cov[:,0,0] = data['pmra_error']**2
    cov[:,0,1] = data['pmra_error'] * data['pmdec_error'] * data['pmra_pmdec_corr']
    cov[:,1,0] = data['pmra_error'] * data['pmdec_error'] * data['pmra_pmdec_corr']
    cov[:,1,1] = data['pmdec_error']**2

    return cov

################################################################

################################################################
#### Function defining the likelihood for Gaussian proper motions

def pm_gauss_model_wcs(pmra, pmdec, cov, pmra0, pmdec0):

    diff = np.zeros((len(pmra), 2, 1))

    diff[:,0,0] = pmra - pmra0
    diff[:,1,0] = pmdec - pmdec0

    model = np.zeros((len(pmra)))

    covdet = cov[:,0,0]*cov[:,1,1] - cov[:,0,1]*cov[:,1,0]

    invcov = np.zeros((len(cov), 2, 2))

    invcov[:,0,0] = cov[:,0,0] / covdet
    invcov[:,0,1] = cov[:,0,1] / covdet
    invcov[:,1,0] = cov[:,1,0] / covdet
    invcov[:,1,1] = cov[:,1,1] / covdet

    model = covdet**(-0.5) * np.exp(-0.5 * (diff[:,0,0] * (invcov[:,1,1] * diff[:,0,0] -invcov[:,0,1] * diff[:,1,0]) + diff[:,1,0] * (-1.0 * invcov[:,1,0] * diff[:,0,0] + invcov[:,0,0] * diff[:,1,0])))

    return model

################################################################

################################################################
#### Function that defines the full likelihood depending on the
#### particular set of parameters being analyzed

def total_ln_like(theta, data, cov, perm, dist0, ra0, dec0, vsys):

    #data, cov, cov2, mu_x1, mu_y1, mu_x2, mu_y2, sigma_x2, sigma_y2 - PM
    # data, Rmax, ah, pa, ellip- spatial

    cov_i = np.copy(cov) #this line is so that the original covariance matrix doesn't get overwritten


## permutation for only the proper motion for likelihood
    if (perm == 'standard'):
        pmra0, pmdec0, thet, incl, rad0, rotvel, sigma_pmra, sigma_pmdec, \
                  phi_tid, theta_tid, vel_tid = theta

        cov_i[:,0,0] += (10**sigma_pmra)**2
        cov_i[:,1,1] += (10**sigma_pmdec)**2

        pmra, pmdec, rv = pmmodel(pmra0, pmdec0, dist0, vsys, ra0, dec0, np.deg2rad(data['RA']), np.deg2rad(data['DEC']), incl, thet, (10**rad0), rotvel, phi_tid, theta_tid, (10**vel_tid))


        # Calculate the PM likelihoods for each population
        model_like = pm_gauss_model_wcs(data['PMRA'], data['PMDEC'], cov, pmra, pmdec)

        like = np.sum(np.log(model_like))


## permutation for only the spherical spatial case
    else:
        pmra0, pmdec0, thet, incl, rad0, rotvel, sigma_pmra, sigma_pmdec, \
                  phi_tid, theta_tid, vel_tid = theta

        cov_i[:,0,0] += (10**sigma_pmra)**2
        cov_i[:,1,1] += (10**sigma_pmdec)**2

        pmra, pmdec, rv = pmmodel(pmra0, pmdec0, dist0, vsys, ra0, dec0, np.deg2rad(data['RA']), np.deg2rad(data['DEC']), incl, thet, (10**rad0), rotvel, phi_tid, theta_tid, (10**vel_tid))


        # Calculate the PM likelihoods for each population
        model_like = pm_gauss_model_wcs(data['PMRA'], data['PMDEC'], cov, pmra, pmdec)

        like = np.sum(np.log(model_like))


    return like

################################################################

################################################################
#### function to check that all priors are within flat prior range.
#### For now it's set up assuming flat priors. Will examine how to
#### modify the setup to include the ability to assign Gaussian priors.

def ln_prior(theta, perm):
    if (perm == 'standard'):
        pmra0, pmdec0, thet, incl, rad0, rotvel, sigma_pmra, sigma_pmdec, phi_tid, theta_tid, vel_tid = theta

        lp = 0. if (-5. < pmra0 < 5.) and (-5. <= pmdec0 <= 5.) and (0. <= thet <= 360.) \
        and (-90. < incl < 90.) and (-2. < rad0 < 1.) and (-50. < rotvel < 50.) and (-2. < sigma_pmra < 0.) \
        and (-2. < sigma_pmdec < 0.) and (0. < phi_tid < 360.) and (-90. < theta_tid < 90.) \
        and (-1. < vel_tid < 2.) else -np.inf

    else:
        lp = -np.inf

    return lp

################################################################

################################################################
#### Function to keep samples within priors and make sure real
#### values are being returned

def ln_prob(theta, data, cov, perm, dist0, ra0, dec0, vsys):

    lp = ln_prior(theta, perm)
    if not np.isfinite(lp):
        return -np.inf
    try:
        lnprobval = lp + total_ln_like(theta, data, cov, perm, dist0, ra0, dec0, vsys)
    except ValueError: # NaN value case
        lnprobval = -np.inf

    return lnprobval

################################################################

################################################################
#### Function to actually run the MCMC sampler

def run_MCMC(data, nparams, perm='standard', ra0=np.deg2rad(13.0), dec0=np.deg2rad(-71.0), dist0=60.0, vsys=145., priors=None, nw=40, ns=1000):

    # Calculate the covariance matrices
    cov = calc_cov_wcs(data)

    # Set the number of dimensions and number of walkers desired
    ndim, nwalkers = nparams, nw

    # Create the initial state used to sample the space
    p0 = np.random.rand(nwalkers, ndim)

    p0 = create_initial(p0, priors)

    # Create the sampler itself
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args=[data, cov, perm, dist0, ra0, dec0, vsys])

    print("Running burn-in...")
    state = sampler.run_mcmc(p0, 100, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, ns, progress=True);

    return sampler, cov

################################################################
