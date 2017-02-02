import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

print 'loading data'
data = []
prefix = './t/output_'
for x in xrange(100):
    filename = prefix + str(x+1) + '.txt'
    data.append(np.genfromtxt(filename))
data = np.vstack(data)

to_file({'data': data}, './t/data.hdf5')
# data = from_file('./t/data.hdf5')['data']
print 'done loading data'

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
min_llh = np.float64(min_entry[9])

print 'min', min_entry

sort_column = 8
data_sorted = data[data[:,sort_column].argsort()]
uni, c = np.unique(data[:,sort_column], return_counts=True)

n = len(uni)
assert len(np.unique(c)) == 1
c = c[0]
col_array = []
for col in data_sorted.T:
    col_array.append(col.reshape(n, c))
col_array = np.stack(col_array)
sep_arrays = []
for x in xrange(n):
    sep_arrays.append(col_array[:,x])

import itertools
fig = plt.figure(figsize=(34, 17)) # fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 20)
# gs = gridspec.GridSpec(20, 20)
# gs.update(hspace=0.07, wspace=0.07)
# fig.text(0.5, 0.09, r'${\rm Re}(a_{\mu\tau})\:(GeV)$', ha='center')
# fig.text(0.1, 0.5, r'${\rm Im}(a_{\mu\tau})\:(GeV)$', va='center', rotation='vertical')
gs.update(hspace=0.01, wspace=0.01)
fig.text(0.5, 0.07, r'${\rm Re}(t_{\mu\tau})\:({\rm GeV})$', ha='center')
fig.text(0.11, 0.5, r'${\rm Im}(t_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical')
indexes = [15, 30, 50, 75, 95, 115, 150, 170, 185]
for idx, array in enumerate(sep_arrays):
    ax = fig.add_subplot(gs[idx])
    plt.gca().set_autoscale_on(False)

    reduced_llh = array[9] - min_llh
    print idx
    print np.min(array[9]), np.max(array[9])
    print np.min(reduced_llh), np.max(reduced_llh)
    print map(np.min, (array[6], array[7], array[8]))
    print

    llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
    data_90_percent = array.T[llh_90_percent].T

    llh_99_percent = (reduced_llh > 11.34)
    data_99_percent = array.T[llh_99_percent].T

    ax.set_xscale('symlog', linthreshx=1e-35, linthreshy=1e-35)
    ax.set_yscale('symlog', linthreshx=1e-35, linthreshy=1e-35)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    lim=(-np.power(10, np.float128(-23)), np.power(10, np.float128(-23)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    if idx % 20 != 0:
        [y.set_visible(False) for y in yticks]
    else:
        yticks[len(yticks) / 2].set_visible(False)
    if idx - 180 < 0:
        [x.set_visible(False) for x in xticks]
    else:
        xticks[len(xticks) / 2].set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='.', alpha=1, linewidths=0, edgecolors='face', s=2)
    ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='.', alpha=1, linewidths=0, edgecolors='face', s=2)
    if any(reduced_llh == 0):
        bf = array.T[reduced_llh == 0].T
        ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    at = AnchoredText(r'$t_{\mu\mu}={\rm '+r'{0:.2E}'.format(array[8][0])+r'}\:{\rm GeV}$', prop=dict(size=5), frameon=True, loc=10)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

fig.savefig('2d_lv_neg_t.png', bbox_inches='tight', dpi=150)
print 'done'
