import numpy as np
import numpy.ma as ma
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText

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

# pretty superk
terms = {
    r'a': (-27, -22),
    r'c': (-31, -26),
    r't': (-35, -30),
    r'g': (-39, -34),
    r's': (-43, -38),
    r'j': (-47, -42),
}

this = r'j'
mini = np.float128('1e'+str(terms[this][0] - 1))
maxi = np.float128('1e'+str(terms[this][1] + 1))

# print 'loading data'
# data = []
# prefix = './epilogue/'+this+'/output_'
# for x in xrange(100):
#     filename = prefix + str(x+1) + '.txt'
#     data.append(np.genfromtxt(filename))
# data = np.vstack(data)
# to_file({'data': data}, './epilogue/'+this+'/data.hdf5')
# print 'done loading data'

data = []
data = from_file('epilogue/'+this+'/data.hdf5')['data']
print 'done loading data'

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
local_min_llh = np.float64(min_entry[9])

# if this == r'a':
#     min_llh = 1211.24276
# elif this == r'c':
#     min_llh = 1213.29182
# elif this == r't':
#     min_llh = 1213.46361
# elif this == r'g':
#     min_llh = 1213.61553763624
# elif this == r's':
#     min_llh = 1213.61553763624
# elif this == r'j':
#     min_llh = 1213.59369

min_llh = local_min_llh

print 'min_llh', min_llh
print 'min_entry', min_entry

pos_re_pos_di_idx = np.argmin(ma.masked_array(data[:,9], mask=~((data[:,6] > 0) & (data[:,8] > 0))))
neg_re_pos_di_idx = np.argmin(ma.masked_array(data[:,9], mask=~((data[:,6] < 0) & (data[:,8] > 0))))
pos_re_neg_di_idx = np.argmin(ma.masked_array(data[:,9], mask=~((data[:,6] > 0) & (data[:,8] < 0))))
neg_re_neg_di_idx = np.argmin(ma.masked_array(data[:,9], mask=~((data[:,6] < 0) & (data[:,8] < 0))))

print
print 'pos_re_pos_di', data[pos_re_pos_di_idx,:]
print 'neg_re_pos_di', data[neg_re_pos_di_idx,:]
print 'pos_re_neg_di', data[pos_re_neg_di_idx,:]
print 'neg_re_neg_di', data[neg_re_neg_di_idx,:]
print

assert 0

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
print len(sep_arrays)
if len(sep_arrays) == 50:
    fig = plt.figure(figsize=(34, 17)) # fig = plt.figure(figsize=(10, 10))
    boxes = (5, 10)
    gs = gridspec.GridSpec(*boxes)
if len(sep_arrays) == 100:
    fig = plt.figure(figsize=(34, 34)) # fig = plt.figure(figsize=(10, 10))
    boxes = (10, 10)
    gs = gridspec.GridSpec(*boxes)
if len(sep_arrays) == 200:
    fig = plt.figure(figsize=(34, 17)) # fig = plt.figure(figsize=(10, 10))
    boxes = (10, 20)
    gs = gridspec.GridSpec(*boxes)
# gs = gridspec.GridSpec(10, 20)
# gs = gridspec.GridSpec(20, 20)
# gs.update(hspace=0.07, wspace=0.07)
# fig.text(0.5, 0.09, r'${\rm Re}(a_{\mu\tau})\:(GeV)$', ha='center')
# fig.text(0.1, 0.5, r'${\rm Im}(a_{\mu\tau})\:(GeV)$', va='center', rotation='vertical')
gs.update(hspace=0.01, wspace=0.01)
fig.text(0.5, 0.07, r'${\rm Re}('+this+r'_{\mu\tau})\:({\rm GeV})$', ha='center')
fig.text(0.11, 0.5, r'${\rm Im}('+this+r'_{\mu\tau})\:({\rm GeV})$', va='center', rotation='vertical')
# indexes = range(20)
for idx, array in enumerate(sep_arrays):
    # if idx not in indexes:
    #     continue
    ax = fig.add_subplot(gs[idx])
    plt.gca().set_autoscale_on(False)

    reduced_llh = array[9] - min_llh
    print idx
    print np.min(array[9]), np.max(array[9])
    print np.min(reduced_llh), np.max(reduced_llh)
    print map(np.min, (array[6], array[7], array[8]))
    print

    # base = 10
    # smoothing = 1e-3
    # # smoothing = 0

    llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
    data_90_percent = array.T[llh_90_percent].T

    # llh_90_percent = (reduced_llh > 6.25)# & (reduced_llh < 11.34)
    # data_90_percent = array.T[llh_90_percent].T
    # x = data_90_percent[6]
    # y = data_90_percent[7]
    # # x = data_90_percent[6][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    # # y = data_90_percent[7][(data_90_percent[6] > 0) & (data_90_percent[7] > 0)]
    # uniques = np.unique(np.log(x)/np.log(base))
    # try:
    #     bw = np.min(np.diff(uniques))
    # except:
    #     continue
    # uni_x_split = np.split(uniques, np.where(np.diff(uniques) > bw*1.5)[0] + 1)
    # for uni_x in uni_x_split:
    #     p_x_l, p_y_l = [], []
    #     p_x_u, p_y_u = [], []
    #     for uni in uni_x:
    #         idxes = np.where(np.log(x)/np.log(base) == uni)[0]
    #         ymin, ymax = 1, 0
    #         for i_idx in idxes:
    #             if y[i_idx] < ymin: ymin = y[i_idx]
    #             if y[i_idx] > ymax: ymax = y[i_idx]
    #         p_x_l.append(uni)
    #         p_y_l.append(ymin)
    #         p_x_u.append(uni)
    #         p_y_u.append(ymax)
    #     p_x_l, p_y_l = np.array(p_x_l, dtype=np.float64), np.array(p_y_l, dtype=np.float64)
    #     p_x_u, p_y_u = np.array(list(reversed(p_x_u)), dtype=np.float64), np.array(list(reversed(p_y_u)), dtype=np.float64)
    #     p_x = np.hstack([p_x_l, p_x_u])
    #     p_y = np.hstack([p_y_l, p_y_u])
    #     p_x = np.r_[p_x, p_x[0]]
    #     p_y = np.r_[p_y, p_y[0]]
    #     try:
    #         tck, u = splprep([p_x, p_y], s=smoothing, per=True)
    #         xi, yi = splev(np.linspace(0, 1, 1000), tck)
    #         ax.fill(np.power(base, xi), yi, 'r', edgecolor='k', linewidth=1)
    #     except:
    #         ax.fill(np.power(base, p_x), p_y, 'r', edgecolor='k', linewidth=1)

    llh_99_percent = (reduced_llh > 11.34)
    data_99_percent = array.T[llh_99_percent].T
    # x = data_99_percent[6]
    # y = data_99_percent[7]
    # # x = data_99_percent[6][(data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    # # y = data_99_percent[7][(data_99_percent[6] > 0) & (data_99_percent[7] > 0)]
    # uniques = np.unique(np.log(x)/np.log(base))
    # try:
    #     bw = np.min(np.diff(uniques))
    # except:
    #     continue
    # uni_x_split = np.split(uniques, np.where(np.diff(uniques) > bw*1.5)[0] + 1)
    # for uni_x in uni_x_split:
    #     p_x_l, p_y_l = [], []
    #     p_x_u, p_y_u = [], []
    #     for uni in uni_x:
    #         idxes = np.where(np.log(x)/np.log(base) == uni)[0]
    #         ymin, ymax = 1, 0
    #         for i_idx in idxes:
    #             if y[i_idx] < ymin: ymin = y[i_idx]
    #             if y[i_idx] > ymax: ymax = y[i_idx]
    #         p_x_l.append(uni)
    #         p_y_l.append(ymin)
    #         p_x_u.append(uni)
    #         p_y_u.append(ymax)
    #     p_x_l, p_y_l = np.array(p_x_l, dtype=np.float64), np.array(p_y_l, dtype=np.float64)
    #     p_x_u, p_y_u = np.array(list(reversed(p_x_u)), dtype=np.float64), np.array(list(reversed(p_y_u)), dtype=np.float64)
    #     p_x = np.hstack([p_x_l, p_x_u])
    #     p_y = np.hstack([p_y_l, p_y_u])
    #     p_x = np.r_[p_x, p_x[0]]
    #     p_y = np.r_[p_y, p_y[0]]
    #     try:
    #         tck, u = splprep([p_x, p_y], s=smoothing, per=True)
    #         xi, yi = splev(np.linspace(0, 1, 1000), tck)
    #         ax.fill(np.power(base, xi), yi, 'b', edgecolor='k', linewidth=1)
    #     except:
    #         ax.fill(np.power(base, p_x), p_y, 'b', edgecolor='k', linewidth=1)

    ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='.', alpha=1, linewidths=0, edgecolors='face', s=2)
    ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='.', alpha=1, linewidths=0, edgecolors='face', s=2)

    if any(reduced_llh == local_min_llh - min_llh):
        bf = array.T[reduced_llh == local_min_llh - min_llh].T
        ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xscale('symlog', linthreshx=mini, linthreshy=mini)
    ax.set_yscale('symlog', linthreshx=mini, linthreshy=mini)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)

    lim=(-maxi, maxi)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    if idx % boxes[1] != 0:
        [y.set_visible(False) for y in yticks]
    if idx - (boxes[0]*boxes[1] - boxes[1]) < 0:
        [x.set_visible(False) for x in xticks]

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    at = AnchoredText(r'$'+this+r'_{\mu\mu}={\rm '+r'{0:.2E}'.format(array[8][0])+r'}\:{\rm GeV}$', prop=dict(size=5), frameon=True, loc=10)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

fig.savefig('lowE_cut_'+this+'_full_slice.png', bbox_inches='tight', dpi=150)
print 'done'
