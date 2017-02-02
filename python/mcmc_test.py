import numpy as np
import subprocess
import emcee
import time
import sys
import lvsearchpy as lv
from copy import deepcopy
import tqdm

#paths
effective_area_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/effective_area.h5'
events_path='/data/icecube/software/LVTools/data/simple_corrected_data_release.dat'
chris_flux_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/conventional_flux.h5'
kaon_flux_path='/data/icecube/software/LVTools/data/kaon_flux.h5'
pion_flux_path='/data/icecube/software/LVTools/data/pion_flux.h5'
prompt_flux_path='/data/icecube/software/LVTools/data/prompt_flux.h5'
output_file_path='/data/icecube/software/LVTools_package/LVTools/python/scan/output_'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetEnergyExponent(1.)
lvsearch.SetVerbose(False)

#calculate likelihood from c++
def llhCPP(theta):
    theta[-3] = np.power(10.,theta[-3])
    t = deepcopy(theta)
    # t[-3] = np.power(10.,t[-3])
    re = t[-3]*np.sin(t[-2])*np.cos(t[-1])
    im = t[-3]*np.sin(t[-2])*np.sin(t[-1])
    tr = t[-3]*np.cos(t[-2])
    t[-3:] = np.array([re, im, tr])
    output=lvsearch.llhFull(t)
    # print 'params ', theta
    # print 'llh    ', -output
    return -output
 
def lnprior(theta):
    normalization, cosmic_ray_slope, pik, prompt_norm, astro_norm, astro_gamma, rad, the, phi = theta
    if -30 < rad < -23 and 0 < the < np.pi and -np.pi < phi < np.pi:
    # if -30 < logRCmutau < -23 and -30 < logICmutau < -23 and -30 < logCmumu < -23 :
    #if -30 < logRCmutau < -23 and -30 < logICmutau < -23 and -30 < logCmumu < -23 \
    #        and 0.1 < normalization < 10 and -0.1 < cosmic_ray_slope < 0.1 and 0.1 < pik < 2.0 and 0 < prompt_norm < 10. and 0 < astro_norm < 10. and -0.5 <astro_gamma < 0.5:
        return 0.0
    return -np.inf

def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	# print lp + llhCPP(theta)
	return lp + llhCPP(theta)

## MCMC business

import matplotlib.pylab as plt
theta = [1.,0.,1.,1.,1.,0.,-28,1.6,0] 
a = np.linspace(-30, -23, 30)
ll_arr = []
for x in a:
    theta[-3] = x
    ll_arr.append(llhCPP(theta))

plt.plot(a, ll_arr)
plt.savefig('test_llhCPP.png')

# tt = time.time()
# print("Initializing walkers")
# ndim = 9
# nwalkers = 200
# # ntemps = 10
# ntemps = 5
# burnin = 1000
# # p0_base = [1.,0.,1.,1.,1.,0.,-28,0,0]
# # p0 = [p0_base + 0.1*np.random.rand(ndim) for i in range(nwalkers)]
# p0_base=[1., 0., 1., 1. , 1. , 0., -26.5, 0., 0.]
# p0_std = [0.3, 0.05, 0.1, 0.1, 0.1, 0.1, 1., 3., 3.]
# p0 = np.random.normal(p0_base, p0_std, size=[ntemps, nwalkers, ndim])
# p0[:,:,-1] = (p0[:,:,-1] % 2*np.pi) - np.pi
# p0[:,:,-2] = p0[:,:,-2] % np.pi

# # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
# sampler = emcee.PTSampler(ntemps, nwalkers, ndim, llhCPP, lnprior)
# print("Running burn-in")
# # pos, prob, state = sampler.run_mcmc(p0, 10000)
# for result in tqdm.tqdm(sampler.sample(p0, iterations=burnin), total=burnin):
#     pos, prob, state = result
# sampler.reset()
# # nsteps = 100000
# # nsteps = 50000
# nsteps = 100000
# width = 30
# # sampler.run_mcmc(pos,500) #regular run
# # for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
# #     n = int((width+1) * float(i) / nsteps)
# #     sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
# for _ in tqdm.tqdm(sampler.sample(pos, iterations=nsteps), total=nsteps):
#     pass
# sys.stdout.write("\n")
# print("Time elapsed", time.time()-tt)

# # samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
# # samples = sampler.chain[:, :, :].reshape((-1, ndim))
# samples = sampler.chain[0, :, :, :].reshape((-1, ndim))
# samples[:,6] = np.log10(samples[:,6])

