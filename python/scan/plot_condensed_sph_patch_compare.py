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
        r'test': (-30, -23),
        r'c_plus20': (-28, -24),
        r'c_neg20': (-28, -24),
        r'a_plus50': (-25, -18),
        r'a_neg50': (-25, -18),
        r't_plus50': (-33, -26),
        r't_neg50': (-33, -26),
    }

this = r'a_plus50'
comp = r'a_s'
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

data_comp = from_file('./'+comp+'/data.pckl')['data']

data_comp[:,6] = data_comp[:,6].astype(np.float128)
data_comp[:,7] = data_comp[:,7].astype(np.float128)
data_comp[:,8] = data_comp[:,8].astype(np.float128)

argmin_comp = np.argmin(data_comp[:,9])
min_entry_comp = data_comp[argmin,:]
min_llh_comp = min_entry[9]
# min_llh_comp = 1.21337335e+03

sort_column_comp = 8
data_sorted_comp = data_comp_comp[data_comp[:,sort_column].argsort()]
uni_comp, c_comp = np.unique(data_comp[:,sort_column_comp], return_counts=True)

n_comp = len(uni_comp)
assert len(np.unique(c_comp)) == 1
c_comp = c_comp[0]
col_array_comp = []
for col_comp in data_sorted_comp.T:
    col_array_comp.append(col_comp.reshape(n_comp, c_comp))
col_array_comp = np.stack(col_array_comp)
sep_arrays_comp = []
for x in xrange(n_comp):
    sep_arrays_comp.append(col_array_comp[:,x])

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
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV})$', ha='center')
if this == r'a_plus50':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV})$', ha='center')
if this == r'a_neg50':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV})$', ha='center')
elif this == r'c_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'$', ha='center')
elif this == r'c_plus20':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'$', ha='center')
elif this == r'c_neg20':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'$', ha='center')
elif this == r't_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^{-1})$', ha='center')
elif this == r't_plus50':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^{-1})$', ha='center')
elif this == r't_neg50':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^{-1})$', ha='center')
elif this == r'g_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^{-2})$', ha='center')
elif this == r's_s':
    fig.text(0.5, 0.01, r'$\rho_{0}'.format(string)+r'\:({\rm GeV}^{-3})$', ha='center')
fig.text(0.05, 0.5, r'$\theta_{0}/\pi$'.format(string), va='center', rotation='vertical')
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", lw=0.5, alpha=0.9)
fig.text(0.217, 0.82, r'Allowed', color='green', fontsize=16, ha='center', va='center')
if this == r'a_s':
    fig.text(0.45, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
if this == r'a_plus50':
    fig.text(0.45, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
if this == r'a_neg50':
    fig.text(0.45, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r'c_s':
    fig.text(0.58, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r'c_plus20':
    fig.text(0.58, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r'c_neg20':
    fig.text(0.58, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r't_s':
    fig.text(0.48, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r't_plus50':
    fig.text(0.48, 0.5, r'Excluded', color='red', fontsize=16, bbox=bbox_props,
             ha='center', va='center')
elif this == r't_neg50':
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

    # reduced_llh = array[9] - min_llh
    # print s
    # print np.min(array[9]), np.max(array[9])
    # print np.min(reduced_llh), np.max(reduced_llh)
    # print
    diff_llh = array[9] - sep_arrays_comp[idx][9]
    # caption = r'$\Phi = {0:.4f}$'.format(float(array[8][0]))
    # at = AnchoredText(caption, prop=dict(size=7), frameon=True, loc=10)
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax.add_artist(at)
    s = s + 1

# fig.savefig('test.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_sph_diff_'+this+'.png', bbox_inches='tight', dpi=150)
fig.savefig('condensed_sph_diff_'+this+'.eps', bbox_inches='tight', dpi=150)
print 'done'
