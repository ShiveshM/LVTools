import numpy as np
import subprocess
import emcee
import lvsearchpy as lv

# paths
effective_area_path='/data/user/smandalia/GolemTools/sources/LVTools/data/effective_area.h5'
events_path='/data/user/smandalia/GolemTools/sources/LVTools/data/simple_corrected_data_release.dat'
chris_flux_path='/data/user/smandalia/GolemTools/sources/LVTools/data/conventional_flux.h5'
kaon_flux_path='/data/user/smandalia/GolemTools/sources/LVTools/data/kaon_flux.h5'
pion_flux_path='/data/user/smandalia/GolemTools/sources/LVTools/data/pion_flux.h5'
prompt_flux_path='/data/user/smandalia/GolemTools/sources/LVTools/data/prompt_flux.h5'

# constructing object
lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
lvsearch.SetEnergyExponent(3.)
lvsearch.SetVerbose(False)
# print lvsearch.GetExpectationDistribution([0.00000000e+00,3.42678594e-03,3.72106605e+00,0.00000000e00,0.00000000e+00,-4.08785930e-01,1.62975083e-26,9.77009957e-25,-1.26185688e-25])
# print lvsearch.llhFull([0.00000000e+00,3.42678594e-03,3.72106605e+00,0.00000000e00,0.00000000e+00,-4.08785930e-01,1.62975083e-26,9.77009957e-25,-1.26185688e-25])
print lvsearch.llh([3.28348012564e-36,  -1.18764070099e-35, 3.37597301502e-35])

