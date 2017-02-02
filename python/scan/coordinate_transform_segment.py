import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

mp.mp.dps = 35  # Computation precision is 35 digits

linear = False

if linear:
    terms = {
        r'c': (np.float128('5e-25'), np.float128('5e-28'))
    }
else:
    terms = {
        r'a': (-27, -18),
        r'c': (-30, -23),
        r't': (-34, -24)
    }

this = r'c'

if linear:
    mini = terms[this][0]
    maxi = terms[this][1]
else:
    mini = np.float128('1e'+str(terms[this][0]))
    maxi = np.float128('1e'+str(terms[this][1]))
print 'mini, maxi', mini, maxi

print 'loading '+this+' data'
data = []
data = from_file('./'+this+'/data.hdf5')['data']
print 'done loading data'
print 'data', data.shape

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
min_llh = np.float64(min_entry[9])

print 'min', min_entry



print 'transforming coordinates'
r = np.sqrt(np.square(data[:,6]) + np.square(data[:,7]) + np.square(data[:,8]))
print 'r', np.min(r), np.max(r)
theta = np.arccos(data[:,8] / r)
phi = np.arctan2(data[:,7], data[:,6])


# r = map(
#     lambda x, y, z: mp.sqrt(mp.power(x, 2) + mp.power(y, 2) + mp.power(z, 2)),
#     data[:,6], data[:,7], data[:,8]
# )
# theta = map(lambda z, r: mp.acos(z / r), data[:,8], r)
# phi = map(lambda x, y: mp.atan2(y, x), data[:,6], data[:,7])

data[:,6] = r
data[:,7] = theta
data[:,8] = phi


sort_column = 8
n_bins = 49
phi_bins = np.linspace(-np.pi, np.pi, n_bins+1)
# phi_bins = np.linspace(0, np.pi, n_bins+1)
data_sorted = data[data[:,sort_column].argsort()]
phi_digitize = np.digitize(data_sorted[:,sort_column], phi_bins)
print 'phi_digitize', phi_digitize
uni, c = np.unique(phi_digitize, return_counts=True)
print 'uni', uni
print 'c', c

s = 0
sep_arrays = []
for idx in xrange(n_bins):
    sep_arrays.append(data_sorted[s:s+c[idx]])
    s += c[idx]

for x in xrange(n_bins):
    sep_arrays[x] = np.vstack(sep_arrays[x]).T

print 'sep_arrays', len(sep_arrays)

import itertools
# fig = plt.figure(figsize=(34, 17))
fig = plt.figure(figsize=(12, 10))
assert np.sqrt(n_bins).is_integer()
gs = gridspec.GridSpec(2, 1)
gs.update(hspace=0.01, wspace=0.01)
fig.text(0.5, 0.05, r'${\rm r}$', ha='center')
fig.text(0.06, 0.5, r'$\theta$', va='center', rotation='vertical')
n = 0
indexes = np.array([0, 34])
for idx, array in enumerate(sep_arrays):
    if idx not in indexes:
        continue
    ax = fig.add_subplot(gs[n])
    plt.gca().set_autoscale_on(False)

    reduced_llh = array[9] - min_llh
    print n
    print np.min(array[9]), np.max(array[9])
    print np.min(reduced_llh), np.max(reduced_llh)
    print

    llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
    data_90_percent = array.T[llh_90_percent].T

    llh_99_percent = (reduced_llh > 11.34)
    data_99_percent = array.T[llh_99_percent].T

    if not linear:
        ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # do ratios
    lim=(mini, maxi)
    ax.set_xlim(lim)
    ax.set_ylim(0, np.pi)
    # ax.set_ylim(-np.pi, np.pi)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    # if idx % np.sqrt(n_bins) != 0:
    #     [y.set_visible(False) for y in yticks]
    # else:
    #     yticks[len(yticks) / 2].set_visible(False)
    # if idx - (n_bins - np.sqrt(n_bins)) < 0:
    #     [x.set_visible(False) for x in xticks]
    # else:
    #     xticks[len(xticks) / 2].set_visible(False)

    # if n == 3:
    #     if (len(yticks) / 2) + 3 < len(yticks):
    #         yticks[(len(yticks) / 2) + 3].set_visible(False)
    #     if (len(yticks) / 2) - 3 >= 0:
    #         yticks[(len(yticks) / 2) - 3].set_visible(False)
    # elif n == 7:
    #     if (len(xticks) / 2) + 3 < len(xticks):
    #         xticks[(len(xticks) / 2) + 3].set_visible(False)
    #     if (len(xticks) / 2) - 3 >= 0:
    #         xticks[(len(xticks) / 2) - 3].set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    if any(reduced_llh == 0):
        bf = array.T[reduced_llh == 0].T
        ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    caption = r'${0:.2f} < \Phi < {1:.2f}$'.format(phi_bins[idx], phi_bins[idx+1])
    at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    n = n + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('spherical_'+this+'_4.png', bbox_inches='tight', dpi=150)
print 'done'
