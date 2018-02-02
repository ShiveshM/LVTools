#! /usr/bin/env python
"""
From an MCMC chains file, make a triangle plot.
"""

from __future__ import absolute_import, division

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

import getdist
from getdist import plots
from getdist import mcsamples

rc('text', usetex=True)
rc('font', **{'family':'serif', 'serif':['Computer Modern'], 'size':18})
cols = ['#29A2C6','#FF6D31','#FFCB18','#73B66B','#EF597B', '#333333']

font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 18}

name = '180202_1ksteps'

TchainNAN=np.load("chains/chain_full_6_"+name+".npy")

labels=[r"N_{atm}",r"\Delta \gamma",r"R_{\pi K}",
       r"N_{prompt}",r"N_{astro}",r"\Delta \gamma_{astro}",
       r"\log_{10}\left(\rho_6/{\rm GeV^2}\right)",r"\cos\theta_6",r"\phi/rad"]

print np.min(TchainNAN[:,6]),np.max(TchainNAN[:,6])
TsampleNAN=mcsamples.MCSamples(samples=TchainNAN,
                              names=["norm","deltag","rpik","prompt","astro","deltaa","rho","cth","phi"],
                           ranges=[[0.8,2.1],[-0.06,0.06],[0.7,1.6],
                                   [0,20],[0,20],[-0.5,0.5],
                                   [-38,-27],[-1,1],[-np.pi,np.pi]
                                 ],
                           labels=labels,)
TsampleNAN.updateSettings({'contours': [0.95, 0.99]})

TsampleNAN.num_bins_2D=500
TsampleNAN.fine_bins_2D=500
TsampleNAN.smooth_scale_2D=0.01

g = plots.getSubplotPlotter(width_inch=5)
g.settings.legend_fontsize=20
g.settings.axes_fontsize=15
g.plot_2d([TsampleNAN],param1="rho",param2="cth",
         filled=True, colors = ("limegreen","k"));
plt.xlim(-38, -27)

g.export("images/LV_dim6_"+name+".png")

