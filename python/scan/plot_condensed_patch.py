import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

terms = {
    # r'a': (-27, -18),
    r'a': (-27, -22),
    # r'c': (-30, -23),
    r'c': (-30, -25),
    # r't': (-34, -24)
    r't': (-34, -28)
}

this = r't'
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
# fig = plt.figure(figsize=(12, 10))
fig = plt.figure(figsize=(4, 4))
n_bins = 1
gs = gridspec.GridSpec(int(np.sqrt(n_bins)), int(np.sqrt(n_bins)))
gs.update(hspace=0.01, wspace=0.01)
if this == r'a':
    fig.text(0.5, 0.01, r'${\rm Re}('+this+r'_{\mu\tau})\:({\rm GeV})$', ha='center')
    fig.text(0.0001, 0.5, r'${\rm Im}('+this+r'_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical')
elif this == r'c':
    fig.text(0.5, 0.01, r'${\rm Re}('+this+r'_{\mu\tau})$', ha='center')
    fig.text(0.0001, 0.5, r'${\rm Im}('+this+r'_{\mu\tau})$', va='center', rotation='vertical')
elif this == r't':
    fig.text(0.5, 0.01, r'${\rm Re}('+this+r'_{\mu\tau})\:({\rm GeV}^{-1})$', ha='center')
    fig.text(0.0001, 0.5, r'${\rm Im}('+this+r'_{\mu\tau})\:({\rm GeV}^{-1})$', va='center', rotation='vertical')
if this == r'a':
    indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
    indexes *= 2
if this == r'c':
    indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
if this == r't':
    indexes = np.array([30, 50, 75, 85, 95, 105, 115, 150, 170])
indexes = np.array([95])
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

    base = 10
    smoothing = 1e-3

    llh_90_percent = (reduced_llh > 6.25)# & (reduced_llh < 11.34)
    data_90_percent = array.T[llh_90_percent].T
    x = data_90_percent[6][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    y = data_90_percent[7][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    uniques = np.unique(np.log(x)/np.log(base))
    bw = np.min(np.diff(uniques))
    uni_x_split = np.split(uniques, np.where(np.diff(uniques) > bw*1.5)[0] + 1)
    for uni_x in uni_x_split:
        p_x_l, p_y_l = [], []
        p_x_u, p_y_u = [], []
        for uni in uni_x:
            idxes = np.where(np.log(x)/np.log(base) == uni)[0]
            ymin, ymax = 1, 0
            for idx in idxes:
                if y[idx] < ymin: ymin = y[idx]
                if y[idx] > ymax: ymax = y[idx]
            p_x_l.append(uni)
            p_y_l.append(ymin)
            p_x_u.append(uni)
            p_y_u.append(ymax)
        p_x_l, p_y_l = np.array(p_x_l, dtype=np.float64), np.array(p_y_l, dtype=np.float64)
        p_x_u, p_y_u = np.array(list(reversed(p_x_u)), dtype=np.float64), np.array(list(reversed(p_y_u)), dtype=np.float64)
        p_x = np.hstack([p_x_l, p_x_u])
        p_y = np.hstack([p_y_l, p_y_u])
        p_x = np.r_[p_x, p_x[0]]
        p_y = np.r_[p_y, p_y[0]]
        try:
            tck, u = splprep([p_x, p_y], s=smoothing, per=True)
            xi, yi = splev(np.linspace(0, 1, 1000), tck)
            ax.fill(np.power(base, xi), yi, 'b')
        except:
            ax.fill(np.power(base, p_x), p_y, 'b')

    llh_99_percent = (reduced_llh > 11.34)
    data_99_percent = array.T[llh_99_percent].T
    x = data_99_percent[6][(data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    y = data_99_percent[7][(data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    uniques = np.unique(np.log(x)/np.log(base))
    bw = np.min(np.diff(uniques))
    uni_x_split = np.split(uniques, np.where(np.diff(uniques) > bw*1.5)[0] + 1)
    for uni_x in uni_x_split:
        p_x_l, p_y_l = [], []
        p_x_u, p_y_u = [], []
        for uni in uni_x:
            idxes = np.where(np.log(x)/np.log(base) == uni)[0]
            ymin, ymax = 1, 0
            for idx in idxes:
                if y[idx] < ymin: ymin = y[idx]
                if y[idx] > ymax: ymax = y[idx]
            p_x_l.append(uni)
            p_y_l.append(ymin)
            p_x_u.append(uni)
            p_y_u.append(ymax)
        p_x_l, p_y_l = np.array(p_x_l, dtype=np.float64), np.array(p_y_l, dtype=np.float64)
        p_x_u, p_y_u = np.array(list(reversed(p_x_u)), dtype=np.float64), np.array(list(reversed(p_y_u)), dtype=np.float64)
        p_x = np.hstack([p_x_l, p_x_u])
        p_y = np.hstack([p_y_l, p_y_u])
        p_x = np.r_[p_x, p_x[0]]
        p_y = np.r_[p_y, p_y[0]]
        try:
            tck, u = splprep([p_x, p_y], s=smoothing, per=True)
            xi, yi = splev(np.linspace(0, 1, 1000), tck)
            ax.fill(np.power(base, xi), yi, 'r')
        except:
            ax.fill(np.power(base, p_x), p_y, 'r')

    # ax.set_xscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    # ax.set_yscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # lim=(-maxi, maxi)
    lim=(mini, maxi)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    # if idx % 20 != 0:
    # if n % 3 != 0:
    #     [y.set_visible(False) for y in yticks]
    # else:
    #     yticks[len(yticks) / 2].set_visible(False)
    # # if idx - 180 < 0:
    # if n - 6 < 0:
    #     [x.set_visible(False) for x in xticks]
    # else:
    #     xticks[len(xticks) / 2].set_visible(False)
    # yticks[len(yticks) / 2].set_visible(False)
    # xticks[len(xticks) / 2].set_visible(False)

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

    # if any(reduced_llh == 0):
    #     bf = array.T[reduced_llh == 0].T
    #     ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    # caption = r'$'+this+r'_{\mu\mu}= '+r'{0:.2E}'.format(array[8][0]).replace(r'E', r'\times 10^{')+r'}\:{\rm GeV}$'
    # at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax.add_artist(at)
    n = n + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'.eps', bbox_inches='tight', dpi=150)
print 'done'
