#!/usr/bin/env python
import os
import mpmath as mp
import numpy as np
import argparse
import lvsearchpy as lv

mp.mp.dps = 55  # Computation precision is 55 digits

class FullPaths(argparse.Action):
    """
    Append user- and relative-paths
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest,
                os.path.abspath(os.path.expanduser(values)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--this', type=str, required=True,
                        help='''Which parameter to scan a,t,c etc.''')

    # parser.add_argument('--real', type=mp.mpf, required=True,
    #                     help='''Value of real parameter''')

    parser.add_argument('--run', type=int, required=True, help='''Run ID''')
    args = parser.parse_args()

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
    if args.this == 'a':
        lvsearch.SetEnergyExponent(0.)
    elif args.this == 'c':
        lvsearch.SetEnergyExponent(1.)
    elif args.this == 't':
        lvsearch.SetEnergyExponent(2.)
    elif args.this == 'g':
        lvsearch.SetEnergyExponent(3.)
    elif args.this == 's':
        lvsearch.SetEnergyExponent(4.)
    elif args.this == 'j':
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
    this = args.this
    points = []
    with open('/data/icecube/software/LVTools_package/LVTools/python/scan/numpy/'+this+'_'+str(args.run)+'.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            points.append([mp.mpf(x.strip()) for x in line.split(' ')])
    points = np.array(points).T
    re = points[0]
    im = points[1]
    tr = points[2]

    with open(output_file_path+str(args.run)+'.txt', 'w') as f:
        for c in xrange(len(re)):
            # p0_base_hp = [args.real, a, b]
            p0_base_hp = [re[c], im[c], tr[c]]
            p0_base = map(float, p0_base_hp)
            # any_zero = False
            # for elem in p0_base:
            #     if elem == 0:
                    # any_zero = True
            # if any_zero:
                # continue
            values = lvsearch.llh(p0_base)
            np.savetxt(f, values[:-1], fmt='%s', newline=' ', delimiter=' ')
            np.savetxt(f, p0_base_hp, fmt='%s', newline=' ', delimiter=' ')
            f.write(str(values[-1]) + '\n')
