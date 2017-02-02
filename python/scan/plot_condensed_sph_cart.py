import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

terms = {
    r'a_s': (-27, -18),
    r'c_s': (-30, -23),
    r't_s': (-34, -24)
}

this = r'c_s'
mini = np.float128('1e'+str(terms[this][0]))
maxi = np.float128('1e'+str(terms[this][1]))

print 'loading '+this+' data'
data = []
prefix = './'+this+'/output_'
for x in xrange(100):
    filename = prefix + str(x+1) + '.txt'
    data.append(np.genfromtxt(filename))
data = np.vstack(data)
print 'done loading data'

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
# min_llh = np.float64(min_entry[9])
min_llh = 1.21337335e+03

print 'min', min_entry

sort_column = 8
n_bins = 12
tr_bins = np.hstack([list(reversed(-np.linspace(mini, maxi, n_bins+1,
                                                dtype=np.float64))),
                     np.linspace(mini, maxi, n_bins+1, dtype=np.float64)]).astype(np.float64)
print tr_bins.dtype
# tr_bins = np.linspace(0, np.pi, n_bins+1)
data_sorted = data[data[:,sort_column].argsort()].astype(np.float64)
print data_sorted.dtype
tr_digitize = np.digitize(data_sorted[:,sort_column], tr_bins)
print 'tr_digitize', tr_digitize
uni, c = np.unique(tr_digitize, return_counts=True)
print 'uni', uni
print 'c', c
print 'len(c)', len(c)

s = 0
sep_arrays = []
for idx in xrange((n_bins+1)*2-1):
    sep_arrays.append(data_sorted[s:s+c[idx]])
    s += c[idx]

for x in xrange(n_bins*2):
    sep_arrays[x] = np.vstack(sep_arrays[x]).T

print 'sep_arrays', len(sep_arrays)

import itertools
# fig = plt.figure(figsize=(34, 17))
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(5, 5)
gs.update(hspace=0.01, wspace=0.01)
fig.text(0.5, 0.05, r'${\rm Re}('+this[0]+r'_{\mu\tau})\:({\rm GeV})$', ha='center')
fig.text(0.06, 0.5, r'${\rm Im}('+this[0]+r'_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical')
indexes = np.array([5, 20])
n = 0
for idx, array in enumerate(sep_arrays):
    # if idx not in indexes:
    #     continue
    print array
    ax = fig.add_subplot(gs[n])
    plt.gca().set_autoscale_on(False) 

    reduced_llh = array[9] - min_llh
    print n
    print np.min(array[9]), np.max(array[9])
    print np.min(reduced_llh), np.max(reduced_llh)
    print

    llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
    data_90_percent = array.T[llh_90_percent].T
    # data_90_percent = array

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
    if n % 5 != 0:
        [y.set_visible(False) for y in yticks]
    else:
        yticks[len(yticks) / 2].set_visible(False)
    if n - 20 < 0:
        [x.set_visible(False) for x in xticks]
    else:
        xticks[len(xticks) / 2].set_visible(False)

    # if n == 3:
	# if (len(yticks) / 2) + 3 < len(yticks):
	    # yticks[(len(yticks) / 2) + 3].set_visible(False)
    #     if (len(yticks) / 2) - 3 >= 0:
	    # yticks[(len(yticks) / 2) - 3].set_visible(False)
    # elif n == 7:
	# if (len(xticks) / 2) + 3 < len(xticks):
	    # xticks[(len(xticks) / 2) + 3].set_visible(False)
    #     if (len(xticks) / 2) - 3 >= 0:
	    # xticks[(len(xticks) / 2) - 3].set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    if this == r'a_s':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.3)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.3)
    elif this == r'c_s':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    elif this == r't_s':
	ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.7)
	ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.7)
    if any(reduced_llh == 0):
        bf = array.T[reduced_llh == 0].T
        ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    caption = r'${0:.2E} < {1}'.format(tr_bins[idx], this[0])+r'_{\mu\mu}'+' < {0:.2E}'.format(tr_bins[idx+1]).replace(r'E', r'\times 10^{')+r'}\:{\rm GeV}$'
    # caption = r'$'+this[0]+r'_{\mu\mu}= '+r'{0:.2E}'.format(array[8][0]).replace(r'E', r'\times 10^{')+r'}\:{\rm GeV}$'
    at = AnchoredText(caption, prop=dict(size=4), frameon=True, loc=1)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    n = n + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'.png', bbox_inches='tight', dpi=150)
print 'done'
