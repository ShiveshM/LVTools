import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import splev, splprep

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\usepackage{accents}']
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['mathtext.rm'] = 'Computer Modern'
mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

# terms = {
#     # r'a': (-27, -18),
#     r'a': (-27, -22),
#     # r'c': (-30, -23),
#     # r'c': (-30, -25),
#     r'c': (-33, -28),
#     # r't': (-34, -24),
#     # r't': (-34, -29),
#     r't': (-38, -33),
#     # r'g': (-38, -27),
#     r'g': (-38, -33),
#     # r's': (-42, -30),
#     r's': (-42, -37),
#     # r'j': (-46, -33),
#     r'j': (-49, -42),
# }

# pretty superk
terms = {
    r'a': (-27, -22),
    r'c': (-31, -26),
    r't': (-35, -30),
    r'g': (-39, -34),
    r's': (-43, -38),
    r'j': (-47, -42),
}

this = r'a'
mini = np.float128('1e'+str(terms[this][0]))
maxi = np.float128('1e'+str(terms[this][1]))

print 'loading '+this+' data'
data = []
data = from_file('pretty/'+this+'/data.hdf5')['data']
print 'done loading data'

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
min_llh = np.float64(min_entry[9])

if this == r'a':
    min_llh = 1211.24276
elif this == r'c':
    min_llh = 1213.29182
elif this == r't':
    min_llh = 1213.46361
elif this == r'g':
    min_llh = 1213.61553763624
elif this == r's':
    min_llh = 1213.61553763624
elif this == r'j':
    min_llh = 1213.59369

print 'bestfit', min_entry
# assert 0

sort_column = 8
data_sorted = data[data[:,sort_column].argsort()]
uni, c = np.unique(data[:,sort_column], return_counts=True)

n = len(uni)
print 'n', n
print 'c', c
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
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{a}^{(3)}_{\mu\tau})\:({\rm GeV})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{a}^{(3)}_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical', fontsize=18)
elif this == r'c':
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{c}^{(4)}_{\mu\tau})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{c}^{(4)}_{\mu\tau})$', va='center', rotation='vertical', fontsize=18)
elif this == r't':
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{a}^{(5)}_{\mu\tau})\:({\rm GeV}^{-1})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{a}^{(5)}_{\mu\tau})\:({\rm GeV}^{-1})$', va='center', rotation='vertical', fontsize=18)
elif this == r'g':
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{c}^{(6)}_{\mu\tau})\:({\rm GeV}^{-2})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{c}^{(6)}_{\mu\tau})\:({\rm GeV}^{-2})$', va='center', rotation='vertical', fontsize=18)
elif this == r's':
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{a}^{(7)}_{\mu\tau})\:({\rm GeV}^{-3})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{a}^{(7)}_{\mu\tau})\:({\rm GeV}^{-3})$', va='center', rotation='vertical', fontsize=18)
elif this == r'j':
    fig.text(0.5, -0.04, r'${\rm Re}(\accentset{\circ}{c}^{(8)}_{\mu\tau})\:({\rm GeV}^{-4})$', ha='center', fontsize=18)
    fig.text(-0.08, 0.5, r'${\rm Im}(\accentset{\circ}{c}^{(8)}_{\mu\tau})\:({\rm GeV}^{-4})$', va='center', rotation='vertical', fontsize=18)
# if this == r'a':
#     indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
#     indexes *= 2
# if this == r't':
#     indexes = np.array([30, 50, 75, 85, 95, 105, 115, 150, 170])
# if this == r'c':
#     indexes = np.array([15, 30, 50, 75, 95, 115, 150, 170, 185])
# indexes = np.array([95])
indexes = np.array([len(sep_arrays) / 2])
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
    # smoothing = 0

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
            ax.fill(np.power(base, xi), yi, 'r', edgecolor='k', linewidth=1)
        except:
            ax.fill(np.power(base, p_x), p_y, 'r', edgecolor='k', linewidth=1)

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
            ax.fill(np.power(base, xi), yi, 'b', edgecolor='k', linewidth=1)
        except:
            ax.fill(np.power(base, p_x), p_y, 'b', edgecolor='k', linewidth=1)

    miny_90 = np.min(data_90_percent[7][(data_90_percent[7] > 0)])
    minmask_90 = (data_90_percent[7] == miny_90)
    print 'Re min_90', np.min(data_90_percent[6][minmask_90][(data_90_percent[6][minmask_90] > 0)])

    miny_99 = np.min(data_99_percent[7][(data_99_percent[7] > 0)])
    minmask_99 = (data_99_percent[7] == miny_99)
    print 'Re min_99', np.min(data_99_percent[6][minmask_99][(data_99_percent[6][minmask_99] > 0)])

    minx_90 = np.min(data_90_percent[6][(data_90_percent[6] > 0)])
    minmask_90 = (data_90_percent[6] == minx_90)
    print 'Im min_90', np.min(data_90_percent[7][minmask_90][(data_90_percent[7][minmask_90] > 0)])

    minx_99 = np.min(data_99_percent[6][(data_99_percent[6] > 0)])
    minmask_99 = (data_99_percent[6] == minx_99)
    print 'Im min_99', np.min(data_99_percent[7][minmask_99][(data_99_percent[7][minmask_99] > 0)])

    abs_im_90 = data_90_percent[6][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    abs_re_90 = data_90_percent[7][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    print 'test im 90', abs_im_90[np.argmin(abs_im_90 + abs_re_90)]
    print 'test re 90', abs_re_90[np.argmin(abs_im_90 + abs_re_90)]

    im_eq_90 = data_90_percent[6][(data_90_percent[6] == data_90_percent[7]) & (data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    re_eq_90 = data_90_percent[7][(data_90_percent[6] == data_90_percent[7]) & (data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    print 'diag im min 90', np.min(im_eq_90)
    print 'diag re min 90', np.min(re_eq_90)

    im_eq_99 = data_99_percent[6][(data_99_percent[6] == data_99_percent[7]) & (data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    re_eq_99 = data_99_percent[7][(data_99_percent[6] == data_99_percent[7]) & (data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    print 'diag im min 99', np.min(im_eq_99)
    print 'diag re min 99', np.min(re_eq_99)
    # assert 0

    # ax.scatter(data_99_percent[6], data_99_percent[7], c='green', marker='s', alpha=1, linewidths=1, edgecolors='face', s=1.2, zorder=9)
    # ax.scatter(data_90_percent[6], data_90_percent[7], c='black', marker='s', alpha=1, linewidths=1, edgecolors='face', s=1.2, zorder=10)

    # ax.set_xscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    # ax.set_yscale('symlog', linthreshx=mini/10., linthreshy=mini/10.)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

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
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.5, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.5, linewidth=0.2)

    # if any(reduced_llh == 0):
    #     bf = array.T[reduced_llh == 0].T
    #     ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    # caption = r'$'+this+r'_{\mu\mu}= '+r'{0:.2E}'.format(array[8][0]).replace(r'E', r'\times 10^{')+r'}\:{\rm GeV}$'
    # at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax.add_artist(at)
    n = n + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'_pretty.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_'+this+'_pretty.eps', bbox_inches='tight', dpi=150)
print 'done'
