import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

terms = {
    r'a': (-27, -18),
    r'c': (-30, -23),
    r't': (-34, -24)
}

this = r'a'
mini = np.float128('1e'+str(terms[this][0]))
maxi = np.float128('1e'+str(terms[this][1]))

print 'loading '+this+' data'
data = []
data = from_file('./'+this+'/data.hdf5')['data']
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
# fig = plt.figure(figsize=(34, 17))
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(3, 3)
gs.update(hspace=0.01, wspace=0.01)
fig.text(0.5, 0.05, r'${\rm Re}('+this+r'_{\mu\tau})\:({\rm GeV})$', ha='center')
fig.text(0.06, 0.5, r'${\rm Im}('+this+r'_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical')
if this == r'a':
    indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
    indexes *= 2
if this == r'c':
    indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
if this == r't':
    indexes = np.array([30, 50, 75, 85, 95, 105, 115, 150, 170])
n = 0
for idx, array in enumerate(sep_arrays):
    if idx not in indexes:
        continue
    # ax = fig.add_subplot(gs[idx])
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

    ax.set_xscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    ax.set_yscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    lim=(-maxi, maxi)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    # if idx % 20 != 0:
    if n % 3 != 0:
        [y.set_visible(False) for y in yticks]
    else:
        yticks[len(yticks) / 2].set_visible(False)
    # if idx - 180 < 0:
    if n - 6 < 0:
        [x.set_visible(False) for x in xticks]
    else:
        xticks[len(xticks) / 2].set_visible(False)

    if n == 3:
	if (len(yticks) / 2) + 3 < len(yticks):
	    yticks[(len(yticks) / 2) + 3].set_visible(False)
        if (len(yticks) / 2) - 3 >= 0:
	    yticks[(len(yticks) / 2) - 3].set_visible(False)
    elif n == 7:
	if (len(xticks) / 2) + 3 < len(xticks):
	    xticks[(len(xticks) / 2) + 3].set_visible(False)
        if (len(xticks) / 2) - 3 >= 0:
	    xticks[(len(xticks) / 2) - 3].set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    if this == r'a':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.3)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.3)
    elif this == r'c':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    elif this == r't':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.7)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.7)
    if any(reduced_llh == 0):
        bf = array.T[reduced_llh == 0].T
        ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    caption = r'$'+this+r'_{\mu\mu}= '+r'{0:.2E}'.format(array[8][0]).replace(r'E', r'\times 10^{')+r'}\:{\rm GeV}$'
    at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    n = n + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'.pdf', bbox_inches='tight', dpi=150)
print 'done'
