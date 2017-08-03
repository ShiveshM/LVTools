#!/usr/bin/env python
import os
import numpy as np
import lvsearchpy as lv

if __name__ == '__main__':
    # paths
    effective_area_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/effective_area.h5'
    events_path='/data/icecube/software/LVTools/data/simple_corrected_data_release.dat'
    chris_flux_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/conventional_flux.h5'
    kaon_flux_path='/data/icecube/software/LVTools/data/kaon_flux.h5'
    pion_flux_path='/data/icecube/software/LVTools/data/pion_flux.h5'
    prompt_flux_path='/data/icecube/software/LVTools/data/prompt_flux.h5'
    output_file_path='/data/icecube/software/LVTools_package/LVTools/python/scan/output_'

    # constructing object
    lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
    lvsearch.SetEnergyExponent(5.)
    lvsearch.SetVerbose(False)

    # #calculate likelihood from c++
    # def llhCPP(theta):
    #     output=lvsearch.llh(np.power(10.,theta))
    #     return output[-1]

    # def lnprior(theta):
    #     logRCmutau, logICmutau, logCmumu = theta
    #     if -30 < logRCmutau < -25 and -30 < logICmutau < -25 and -30 < logCmumu < -25:
    #         return 0.0
    #     return -np.inf

    # def lnprob(theta):
    #         lp = lnprior(theta)
    #         #print(lp)
    #         if not np.isfinite(lp):
    #                 return -np.inf
    #         return lp + llhCPP(theta)

    # real is three parameters to scan
    # - real
    # - imaginary
    # - trace
    # should be ~1200
    # real = [-28,-28,-28]
    # [-23, -30]

    # # a
    # re, im, tr = (2.89942285e-23, 9.54771611e-24, -1.53436841e-22)
    # values = lvsearch.llh([re, im, tr])
    # print 'bf', values

    # # c
    # re, im, tr = (-1.91791026e-30, 1.00000000e-33, 9.54548457e-31)
    # values = lvsearch.llh([re, im, tr])
    # print 'bf', values

    # # t
    # re, im, tr = (-1.59228279e-34, 1.00000000e-34, 2.53536449e-34)
    # values = lvsearch.llh([re, im, tr])
    # print 'bf', values

    # # g
    # re, im, tr = (-2.78255940e-36, 1.29154967e-35, -3.59381366e-35)
    # values = lvsearch.llh([re, im, tr])
    # print values

    # # s
    # re, im, tr = (-1.07226722e-39, 6.13590727e-40, -3.27454916e-39)
    # values = lvsearch.llh([re, im, tr])
    # print 'bf', values

    # j
    re, im, tr = (-1.20450354e-43, 2.00923300e-44, -2.71858824e-43)
    values = lvsearch.llh([re, im, tr])
    print 'bf', values

    re, im, tr = np.power(10., -100), np.power(10., -100), np.power(10., -100)
    values = lvsearch.llh([re, im, tr])
    print 'null', values
