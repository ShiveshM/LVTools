import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText

prefix = 'output_'

data = []
print 'loading data'
for x in xrange(100):
    filename = prefix + str(x+1) + '.txt'
    data.append(np.genfromtxt(filename))
data = np.vstack(data)
print 'done loading data'

argmin = np.argmin(data[:,9])
min_entry = data[argmin,:]
min_llh = min_entry[9]

reduced_llh = data[:,9] - min_llh

llh_90_percent = (reduced_llh > 6.25) & (reduced_llh < 11.34)
data_90_percent = data[llh_90_percent]

llh_99_percent = (reduced_llh > 11.34)
data_99_percent = data[llh_99_percent]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

lim=(-30, -23)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_zlim(lim)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
ax.set_xlabel(r'${\rm Re}(c_{\mu\tau})$')
ax.set_ylabel(r'${\rm Im}(c_{\mu\tau})$')
ax.set_zlabel(r'$c_{\mu\mu}$')

ax.scatter(data_99_percent[:,6], data_99_percent[:,7], data_99_percent[:,8], c='blue', marker='.', alpha=0.3, linewidths=0, edgecolors='face', s=2, depthshade=False)
ax.scatter(data_90_percent[:,6], data_90_percent[:,7], data_90_percent[:,8], c='red', marker='.', alpha=0.3, linewidths=0, edgecolors='face', s=2, depthshade=False)
ax.scatter(data[:,6][argmin], data[:,7][argmin], data[:,8][argmin], c='yellow', marker='*', alpha=1, linewidths=0.2, edgecolors='black', s=40, depthshade=False)

at = AnchoredText(r'${\rm Re}(c_{\mu\tau})$'+'={0:.2f}'.format(data[:,6][argmin]) + '\n' +
                  r'${\rm Im}(c_{\mu\tau})$'+'={0:.2f}'.format(data[:,7][argmin]) + '\n' +
                  r'$c_{\mu\mu}$'+'={0:.2f}'.format(data[:,8][argmin]) + '\n' +
                  '-LLH={0:.2f}'.format(min_llh),
                  prop=dict(size=10), frameon=True, loc=2)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5")
ax.add_artist(at)

fig.savefig('3d_lv.png', bbox_inches='tight', dpi=150)
# fig.savefig('3d_lv.pdf', bbox_inches='tight', dpi=150)
print 'done'
