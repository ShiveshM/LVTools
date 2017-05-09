#!/usr/bin/env python
from operator import sub
import mpmath as mp
import numpy as np
import argparse
import cPickle as pickle

mp.mp.dps = 50  # Computation precision is 50 digits


def main(runs, this):
    terms = {
	r'a': (-27, -18),
	r'c': (-30, -23),
	r't': (-34, -24),
	r'g': (-38, -27),
	r's': (-42, -30),
	r'j': (-46, -33)
    }

    ls = mp.linspace(terms[this][0], terms[this][1], runs)
    r = [mp.power(10., x) for x in ls]
    # theta = mp.linspace(0.00001, mp.pi-0.00001, runs)
    costheta = mp.linspace(-1.0+0.00001, 1.0-0.00001, runs)
    phi = mp.linspace(-mp.pi+0.000001, mp.pi-0.000001, runs)

    points = []
    for x in r:
        # for y in theta:
        for y in costheta:
            for z in phi:
                # re = x*mp.sin(y)*mp.cos(z)
                # im = x*mp.sin(y)*mp.sin(z)
                # tr = x*mp.cos(y)

                re = x*mp.sin(mp.acos(y))*mp.cos(z)
                im = x*mp.sin(mp.acos(y))*mp.sin(z)
                tr = x*y

                points.append(np.array([re, im, tr]))
    points = np.vstack(points)

    # re = map(lambda x, y, z: x*mp.sin(y)*mp.cos(z), r, theta, phi)
    # im = map(lambda x, y, z: x*mp.sin(y)*mp.sin(z), r, theta, phi)
    # tr = map(lambda x, y: x*mp.cos(y), r, theta)

    re, im, tr = points[:,0], points[:,1], points[:,2]

    cou = 1
    filo = './numpy/'+this+'_'+str(cou)+'.txt'
    nfil = 0
    f = open(filo, 'w')
    for x in xrange(len(re)):
        if nfil == runs**2:
            f.close()
            nfil = 0
            cou += 1
            filo = './numpy/'+this+'_'+str(cou)+'.txt'
            f = open(filo, 'w')
        f.write(str(re[x])+' '+str(im[x])+' '+str(tr[x])+'\n')
        nfil += 1

    # with open('./numpy/'+this+'.txt', 'w') as f:
    #     for i, x in enumerate(re):
    #         if i == len(re) - 1: f.write(str(x))
    #         else: f.write(str(x) + ' ')
    #     f.write('\n')
    #     for i, x in enumerate(im):
    #         if i == len(re) - 1: f.write(str(x))
    #         else: f.write(str(x) + ' ')
    #     f.write('\n')
    #     for i, x in enumerate(tr):
    #         if i == len(re) - 1: f.write(str(x))
    #         else: f.write(str(x) + ' ')
    #     f.write('\n')

    # pickle.dump(points, open('./numpy/'+this+'.pckl', 'wb'))

    # np.save('./numpy/'+this, points)

    # r_2 = map(
    #     lambda x, y, z: mp.sqrt(mp.power(x, 2) + mp.power(y, 2) + mp.power(z, 2)),
    #     re, im, tr
    # )
    # theta_2 = map(lambda z, r: mp.acos(z / r), tr, r)
    # phi_2 = map(lambda x, y: mp.atan2(y, x), re, im)

    # print 'diffs'
    # print map(lambda x, y: sub(x, y), r_2, r)
    # print
    # print map(lambda x, y: sub(x, y), theta_2, theta)
    # print
    # print map(lambda x, y: sub(x, y), phi_2, phi)
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--this', type=str, required=True,
                        help='''Which parameter to scan a,t,c etc.''')

    parser.add_argument('--runs', type=float, required=True,
                        help='''Number of points''')

    args = parser.parse_args()

    main(runs=args.runs, this=args.this)
