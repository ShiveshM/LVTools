import mpmath as mp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from scipy.spatial import ConvexHull
from scipy.interpolate import splev, splprep

from pisa.utils.fileio import to_file
from pisa.utils.fileio import from_file

mp.mp.dps = 35  # Computation precision is 35 digits

linear = False

if linear:
    terms = {
        r'c_s': (mp.mpf('5e-25'), mp.mpf('5e-28'))
    }
else:
    terms = {
        r'a_s': (-25, -18),
        r'c_s': (-28, -24),
        r't_s': (-33, -26),
        r'c_s_nm': (-30, -23),
	r'g_s': (-37, -29),
	r's_s': (-42, -30),
        r'test': (-30, -23)
    }

this = r'g_s'
string = this[0]

if linear:
    mini = terms[this][0]
    maxi = terms[this][1]
else:
    mini = mp.mpf('1e'+str(terms[this][0]))
    maxi = mp.mpf('1e'+str(terms[this][1]))
print 'mini, maxi', mini, maxi

print 'loading '+this+' data'

data = []
prefix = './'+this+'/output_'

# for x in xrange(100):
#     filename = prefix + str(x+1) + '.txt'
#     with open(filename, 'r') as f:
#         for line in f.readlines():
#             b = []
#             ls = [x.strip() for x in line.split(' ')]
#             b += map(float, ls[:6])
#             b += map(mp.mpf, ls[6:-1])
#             b += [float(ls[-1])]
#             b = np.array(b)
#             data.append(b)
# data = np.vstack(data)
# # to_file({'data': data}, './'+this+'/data_o.pckl')

# for x in xrange(100):
#     filename = prefix + str(x+1) + '.txt'
#     data.append(np.genfromtxt(filename))
# data = np.vstack(data)

# data = from_file('./'+this+'/data_o.pckl')['data']

# print 'done loading data'
# print 'data', data.shape

# print 'transforming coordinates'
# r = map(
#     lambda x, y, z: mp.sqrt(mp.power(x, 2) + mp.power(y, 2) + mp.power(z, 2)),
#     data[:,6], data[:,7], data[:,8]
# )
# theta = map(lambda z, r: mp.acos(z / r), data[:,8], r)
# phi = map(lambda x, y: mp.atan2(y, x), data[:,6], data[:,7])

# r = np.power(np.square(data[:,6]) + np.square(data[:,7]) + np.square(data[:,8]), 0.5)
# theta = np.arccos(data[:,8] / r)
# phi = np.arctan2(data[:,7], data[:,6])

# data[:,6] = r
# data[:,7] = theta
# data[:,8] = phi

# to_file({'data': data}, './'+this+'/data.pckl')

data = from_file('./'+this+'/data.pckl')['data']

data[:,6] = data[:,6].astype(np.float128)
data[:,7] = data[:,7].astype(np.float128)
data[:,8] = data[:,8].astype(np.float128)
print 'r', np.min(data[:,6]), np.max(data[:,6])
print 'theta', np.min(data[:,7]), np.max(data[:,7])
print 'phi', np.min(data[:,8]), np.max(data[:,8])

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
min_llh = min_entry[9]
# min_llh = 1.21337335e+03

print 'min', min_entry

sort_column = 8
data_sorted = data[data[:,sort_column].argsort()]
uni, c = np.unique(data[:,sort_column], return_counts=True)
print uni, c
print len(uni)
print np.unique(c)

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

print 'sep_arrays', len(sep_arrays)

import itertools
# fig = plt.figure(figsize=(34, 17))
# fig = plt.figure(figsize=(12, 10))
fig = plt.figure(figsize=(6, 4))
resolution = 50 # the number of vertices

# n_bins = 100
# n_bins = 9
n_bins = 1
gs = gridspec.GridSpec(int(np.sqrt(n_bins)), int(np.sqrt(n_bins)))
gs.update(hspace=0.01, wspace=0.01)
if this == r'a_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'$', ha='center')
elif this == r'c_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV})$', ha='center')
elif this == r't_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^2)$', ha='center')
elif this == r'g_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^3)$', ha='center')
elif this == r's_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^4)$', ha='center')
fig.text(0.05, 0.5, r'$\theta_{0}/\pi$'.format(string), va='center', rotation='vertical')
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", lw=0.5, alpha=0.9)
fig.text(0.217, 0.82, r'Allowed', color='green', fontsize=16, ha='center', va='center')
if this == r'a_s':
    fig.text(0.45, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r'c_s':
    fig.text(0.58, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r't_s':
    fig.text(0.48, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r'g_s':
    fig.text(0.48, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r's_s':
    fig.text(0.48, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
s = 0
# indexes = np.array([7, 15, 25, 37, 47, 57, 75, 85, 92])
indexes = np.array([47])
for idx, array in enumerate(sep_arrays):
    if idx not in indexes:
        continue
    print array
    ax = fig.add_subplot(gs[s])
    plt.gca().set_autoscale_on(False)

    reduced_llh = array[9] - min_llh
    print s
    print np.min(array[9]), np.max(array[9])
    print np.min(reduced_llh), np.max(reduced_llh)
    print

    llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
    # llh_90_percent = (reduced_llh > -1)
    data_90_percent = array.T[llh_90_percent].T.astype(np.float128)

    llh_99_percent = (reduced_llh > 11.34)
    data_99_percent = array.T[llh_99_percent].T.astype(np.float128)

    if not linear:
        ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    # do ratios
    lim=map(np.float128, (mini, maxi))
    ax.set_xlim(lim)
    ax.set_ylim(0, 1)
    # ax.set_ylim(0, np.pi)
    # ax.set_ylim(-np.pi, np.pi)

    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    # # if idx % 20 != 0:
    # if s % 3 != 0:
    #     [y.set_visible(False) for y in yticks]
    # else:
    #     yticks[len(yticks) / 2].set_visible(False)
    # # if idx - 180 < 0:
    # if s - 6 < 0:
    #     [x.set_visible(False) for x in xticks]
    # else:
    #     xticks[len(xticks) / 2].set_visible(False)

#     if s == 3:
#         if (len(yticks) / 2) + 3 < len(yticks):
#             yticks[(len(yticks) / 2) + 3].set_visible(False)
#         if (len(yticks) / 2) - 3 >= 0:
#             yticks[(len(yticks) / 2) - 3].set_visible(False)
#     elif s == 7:
#         if (len(xticks) / 2) + 3 < len(xticks):
#             xticks[(len(xticks) / 2) + 3].set_visible(False)
#         if (len(xticks) / 2) - 3 >= 0:
#             xticks[(len(xticks) / 2) - 3].set_visible(False)

    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)
    for xmaj in ax.xaxis.get_majorticklocs():
        ax.axvline(x=xmaj, ls=':', color='gray', alpha=0.7, linewidth=0.2)

    base = 10
    smoothing = 1e-3

    x = data_90_percent[6]
    y = data_90_percent[7]/np.pi
    uniques = np.unique(np.log(x)/np.log(base))
    bw = np.min(np.diff(uniques))
    print 'bw', bw
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

    x = data_99_percent[6]
    y = data_99_percent[7]/np.pi
    uniques = np.unique(np.log(x)/np.log(base))
    bw = np.min(np.diff(uniques))
    print 'bw', bw
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

    # points = np.vstack([np.log10(data_90_percent[6])/np.log10(base), data_90_percent[7]/np.pi]).T
    # print 'points.shape', points.shape
    # hull = ConvexHull(points, qhull_options='QbB')

    # x = np.array(points[hull.vertices,0], dtype=np.float64)
    # y = np.array(points[hull.vertices,1], dtype=np.float64)
    # x = np.r_[x, x[0]]
    # y = np.r_[y, y[0]]
    # tck, u = splprep([x, y], s=0, per=True)
    # xi, yi = splev(np.linspace(0, 1, 1000), tck)
    # ax.fill(np.power(base, xi), yi, 'r')

    # points = np.vstack([np.log10(data_99_percent[6])/np.log10(base), data_99_percent[7]/np.pi]).T
    # print 'points.shape', points.shape
    # hull = ConvexHull(points)

    # x = np.array(points[hull.vertices,0], dtype=np.float64)
    # y = np.array(points[hull.vertices,1], dtype=np.float64)
    # x = np.r_[x, x[0]]
    # y = np.r_[y, y[0]]
    # tck, u = splprep([x, y], s=0, per=True)
    # xi, yi = splev(np.linspace(0, 1, 1000), tck)
    # ax.fill(np.power(base, xi), yi, 'b')

    # ax.scatter(data_99_percent[6], data_99_percent[7], c='blue', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    # ax.scatter(data_90_percent[6], data_90_percent[7], c='red', marker='s', alpha=1, linewidths=0, edgecolors='face', s=0.6)
    # if any(reduced_llh == 0):
    #     bf = array.T[reduced_llh == 0].T
    #     ax.scatter(bf[6], bf[7], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=100)

    # caption = r'$\Phi = {0:.4f}$'.format(float(array[8][0]))
    # at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax.add_artist(at)
    s = s + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_sph_patch_'+this+'.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_sph_patch_'+this+'.eps', bbox_inches='tight', dpi=150)
print 'done'
