#!/usr/bin/env python
import os
import numpy as np
import argparse
import lvsearchpy as lv

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

    parser.add_argument('-p', '--real', type=float, required=True,
                        help='''Value of real parameter''')

    parser.add_argument('--min', type=float, required=True, help='''Min value''')

    parser.add_argument('--max', type=float, required=True, help='''Max value''')

    parser.add_argument('--n-points', type=int, required=True,
                        help='''Number of points to sample''')

    parser.add_argument('--run', type=int, required=True, help='''Run ID''')
    args = parser.parse_args()

    # paths
    effective_area_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/effective_area.h5'
    events_path='/data/icecube/software/LVTools/data/simple_corrected_data_release.dat'
    chris_flux_path='/data/icecube/data/Astro_numu_north_2yr/effective_area_release/conventional_flux.h5'
    kaon_flux_path='/data/icecube/software/LVTools/data/kaon_flux.h5'
    pion_flux_path='/data/icecube/software/LVTools/data/pion_flux.h5'
    prompt_flux_path='/data/icecube/software/LVTools/data/prompt_flux.h5'
    output_file_path='/data/icecube/software/LVTools_package/LVTools/python/scan/lowE_cut/c/output_'

    # constructing object
    lvsearch = lv.LVSearch(effective_area_path,events_path,chris_flux_path,kaon_flux_path,pion_flux_path,prompt_flux_path)
    lvsearch.SetEnergyExponent(1.)
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
    imaginary = np.logspace(args.min, args.max, args.n_points)
    imaginary = np.hstack((imaginary, -np.logspace(args.min, args.max, args.n_points)))

    trace = imaginary

    with open(output_file_path+str(args.run)+'.txt', 'w') as f:
        for a in imaginary:
            for b in trace:
                p0_base = [np.power(10., args.real), a, b]
                values = lvsearch.llh(p0_base)
                np.savetxt(f, values[:-1], fmt='%s', newline=' ', delimiter=' ')
                np.savetxt(f, p0_base, fmt='%s', newline=' ', delimiter=' ')
                f.write(str(values[-1]) + '\n')

                p0_base = [-np.power(10., args.real), a, b]
                values = lvsearch.llh(p0_base)
                np.savetxt(f, values[:-1], fmt='%s', newline=' ', delimiter=' ')
                np.savetxt(f, p0_base, fmt='%s', newline=' ', delimiter=' ')
                f.write(str(values[-1]) + '\n')
