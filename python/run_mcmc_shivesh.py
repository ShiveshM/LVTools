import numpy as np
import subprocess
import multiprocessing
import emcee
from emcee.utils import MPIPool
import time
import sys
import lvsearchpy as lv
from copy import deepcopy
import tqdm

#paths
central_data_path = '/data/user/smandalia/GolemTools/sources/LVTools'
effective_area_path= central_data_path + '/data/effective_area.h5'
events_path= central_data_path + '/data/simple_corrected_data_release.dat'
chris_flux_path= central_data_path + '/data/conventional_flux.h5'
kaon_flux_path= central_data_path + '/data/kaon_flux.h5'
pion_flux_path= central_data_path + '/data/pion_flux.h5'
prompt_flux_path= central_data_path + '/data/prompt_flux.h5'

this = r'g'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
if this == r'a':
    lvsearch.SetEnergyExponent(0.)
if this == r'c':
    lvsearch.SetEnergyExponent(1.)
if this == r't':
    lvsearch.SetEnergyExponent(2.)
if this == r'g':
    lvsearch.SetEnergyExponent(3.)
if this == r's':
    lvsearch.SetEnergyExponent(4.)
if this == r'j':
    lvsearch.SetEnergyExponent(5.)
lvsearch.SetVerbose(False)

terms = {
    r'a': (-27, -18),
    # r'c': (-28, -24),
    r'c': (-32, -25),
    # r't': (-33, -26),
    r't': (-36, -26),
    r'g': (-38, -27),
    # r'g': (-40, -25),
    r's': (-42, -30),
    r'j': (-46, -33),
}

#calculate likelihood from c++
def llhCPP(theta):
    t = deepcopy(theta)
    t[-3] = np.power(10.,t[-3])

    re = t[-3]*np.sin(np.arccos(t[-2]))*np.cos(t[-1])
    im = t[-3]*np.sin(np.arccos(t[-2]))*np.sin(t[-1])
    tr = t[-3]*t[-2]

    t[-3:] = np.array([re, im, tr])
    output=lvsearch.llhFull(t)
    # print 'params ', theta
    # print 'llh    ', -output
    return -output
 
def lnprior(theta):
    normalization, cosmic_ray_slope, pik, prompt_norm, astro_norm, astro_gamma, rad, costhe, phi = theta
    ranges = terms[this]

    if ranges[0] < rad < ranges[1] and -1 < costhe < 1 and -np.pi < phi < np.pi \
       and  0 < normalization < 5 and -1 < cosmic_ray_slope < 1 and 0 < pik < 2.0 \
       and 0 < prompt_norm < 100. and 0 < astro_norm < 100. and -1 <astro_gamma < 1:
        return 0.0
    return -np.inf

def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
                return -np.inf
        return lp + llhCPP(theta)

## MCMC business

tt = time.time()
print("Initializing walkers")
ndim = 9
nwalkers = 200
ntemps = 1
# burnin = 100
burnin = 1000
ranges = terms[this]
p0_base=[1., 0., 1., 1. , 1. , 0., ranges[0], 0., 0.]
p0_std = [0.3, 0.05, 0.1, 0.1, 0.1, 0.1, 5., 3., 3.]
p0 = np.random.normal(p0_base, p0_std, size=[ntemps, nwalkers, ndim])
p0[:,:,-3:] = np.random.uniform(low=[ranges[0], -1, -np.pi], high=[ranges[1], 1, np.pi], size=[ntemps, nwalkers, 3])

threads = multiprocessing.cpu_count()
# threads = 1
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, llhCPP, lnprior, threads=threads)
print("Running burn-in")
for result in tqdm.tqdm(sampler.sample(p0, iterations=burnin), total=burnin):
    pos, prob, state = result
sampler.reset()
# nsteps = 1000
nsteps = 10000
width = 30
for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
    pass
sys.stdout.write("\n")
print("Time elapsed", time.time()-tt)

# samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
# samples = sampler.chain[:, :, :].reshape((-1, ndim))
samples = sampler.chain[0, :, :, :].reshape((-1, ndim))
# samples[:,6] = np.log10(samples[:,6])

print sampler.acceptance_fraction
print np.unique(samples[:,0]).shape

np.save("chains/chain_full_6_180202",samples)

print 'DONE!'

